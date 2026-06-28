"""Phase 1 — Hb-anchored internal HU calibration for Dr Stone.

A non-contrast scan carries two physically-known reference materials: air
(-1000 HU by definition) and aortic blood (whose true HU is set by the patient's
hemoglobin). The patient's measured Hb (from CBC at the time of CT) lets us
compute the *expected* blood HU, so the discrepancy between expected and
observed aortic HU isolates the scanner/calibration error. Two anchors
{air -> -1000, blood -> a + b*Hb} define a per-scan linear correction
HU_corr = alpha + beta*HU_raw applied to stone voxels.

This module:
  1. measures per-scan anchors (eroded aortic-lumen median HU via TotalSegmentator
     aorta mask; extracorporeal air median HU),
  2. fits the population blood-HU vs Hb model (HU_aorta ~ a + b*Hb) and validates
     it (the justification for using Hb to set the blood target),
  3. exposes the per-scan calibration (alpha, beta) and an apply() helper.

Stone-HU variance-reduction validation follows in Phase 2 (needs stone masks).

Run:
    python -m drstone.calibration --limit 40      # measure sample + fit
    python -m drstone.calibration                 # all curated patients
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

from drstone import config as C

warnings.filterwarnings("ignore")

# Measurement constants
# True air sits near -1000 HU; the out-of-FOV reconstruction padding sits much
# lower (~-2048). Measure the observed air HU inside a band around -1000 so the
# padding is excluded.
AIR_HU_LO = -1100.0
AIR_HU_HI = -950.0
AORTA_ERODE_MM = 2.0       # erode lumen to avoid wall partial-volume
AORTA_CALC_HU = 150.0      # exclude vascular calcification from the lumen median
MIN_AORTA_VOX = 200        # require a credible luminal sample
TS_DEVICE = "gpu"          # single GPU


def load_ct(series_dir: str):
    """Read a DICOM series -> (sitk image, HU array ZYX, spacing ZYX)."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    # A curated folder is one selected series, but PACS exports sometimes drop a
    # tiny secondary object (dose screenshot) alongside it. Pick the series ID
    # with the most files so we read the full volume, not a 2-slice stray.
    ids = reader.GetGDCMSeriesIDs(series_dir)
    if ids:
        best = max(ids, key=lambda sid: len(reader.GetGDCMSeriesFileNames(series_dir, sid)))
        files = reader.GetGDCMSeriesFileNames(series_dir, best)
    else:
        files = reader.GetGDCMSeriesFileNames(series_dir)
    if not files:
        raise RuntimeError(f"no DICOM series in {series_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()                      # HU already rescaled by SimpleITK
    arr = sitk.GetArrayFromImage(img).astype(np.float32)   # (Z, Y, X)
    sp = img.GetSpacing()                       # (x, y, z)
    spacing_zyx = (sp[2], sp[1], sp[0])
    return img, arr, spacing_zyx


def segment_aorta(sitk_img, workdir: str):
    """TotalSegmentator aorta mask on the CT, returned as a (Z,Y,X) bool array
    on the SAME grid as the CT (mask written from the same NIfTI we feed in)."""
    import SimpleITK as sitk
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator

    ct_path = os.path.join(workdir, "ct.nii.gz")
    sitk.WriteImage(sitk_img, ct_path)
    seg_dir = os.path.join(workdir, "seg")
    os.makedirs(seg_dir, exist_ok=True)
    totalsegmentator(ct_path, output=seg_dir, task="total", fast=True,
                     roi_subset=["aorta"], device=TS_DEVICE, quiet=True)
    apath = os.path.join(seg_dir, "aorta.nii.gz")
    if not os.path.exists(apath):
        return None
    m = np.asarray(nib.load(apath).dataobj)     # NIfTI is (X,Y,Z)
    mask_zyx = np.transpose(m, (2, 1, 0)) > 0
    return mask_zyx


def measure_anchors(series_dir: str) -> dict:
    """Return {aorta_hu, aorta_nvox, air_hu, air_nvox} for one curated series."""
    from scipy import ndimage as ndi
    _img, ct, spacing = load_ct(series_dir)
    out = {"air_hu": np.nan, "air_nvox": 0, "aorta_hu": np.nan, "aorta_nvox": 0,
           "ct_shape": "x".join(map(str, ct.shape))}

    # Air anchor (true air ~ -1000; exclude out-of-FOV padding near -2048)
    air = ct[(ct >= AIR_HU_LO) & (ct <= AIR_HU_HI)]
    if air.size:
        out["air_hu"] = float(np.median(air))
        out["air_nvox"] = int(air.size)

    with tempfile.TemporaryDirectory() as wd:
        mask = segment_aorta(_img, wd)
    if mask is None or mask.sum() < MIN_AORTA_VOX:
        return out

    # Erode the lumen (avoid wall PV) and exclude calcification.
    it = max(1, int(round(AORTA_ERODE_MM / max(spacing[1], 0.5))))
    eroded = ndi.binary_erosion(mask, iterations=it)
    if eroded.sum() < MIN_AORTA_VOX:
        eroded = mask
    lumen = ct[eroded]
    lumen = lumen[(lumen > -50) & (lumen < AORTA_CALC_HU)]   # blood, no calcium
    if lumen.size >= MIN_AORTA_VOX:
        out["aorta_hu"] = float(np.median(lumen))
        out["aorta_nvox"] = int(lumen.size)
    return out


# --------------------------------------------------------------------------
# Population fit + calibration
# --------------------------------------------------------------------------
def fit_population(df: pd.DataFrame):
    """Fit HU_aorta ~ a + b*Hb (OLS + robust Theil-Sen). df needs aorta_hu, hemoglobin."""
    from scipy import stats
    d = df.dropna(subset=["aorta_hu", "hemoglobin"])
    d = d[(d["aorta_hu"] > -50) & (d["aorta_hu"] < 150)]
    x, y = d["hemoglobin"].values, d["aorta_hu"].values
    ols = stats.linregress(x, y)
    ts = stats.theilslopes(y, x)                 # robust (slope, intercept, lo, hi)
    return {
        "n": int(len(d)),
        "ols_slope": float(ols.slope), "ols_intercept": float(ols.intercept),
        "ols_r": float(ols.rvalue), "ols_p": float(ols.pvalue),
        "ts_slope": float(ts[0]), "ts_intercept": float(ts[1]),
        "a": float(ols.intercept), "b": float(ols.slope),
    }


def calibration_params(air_obs, blood_obs, hb, a, b):
    """Two-anchor linear map HU_corr = alpha + beta*HU_raw through
    {air_obs -> -1000, blood_obs -> a + b*Hb}."""
    target_blood = a + b * hb
    denom = (blood_obs - air_obs)
    if abs(denom) < 1e-6:
        return 0.0, 1.0
    beta = (target_blood - (-1000.0)) / denom
    alpha = -1000.0 - beta * air_obs
    return alpha, beta


def apply_calibration(hu, alpha, beta):
    return alpha + beta * hu


# --------------------------------------------------------------------------
# Batch + report
# --------------------------------------------------------------------------
def patient_table() -> pd.DataFrame:
    """One row per curated cohort patient: curated_dir + Hb + scanner tags."""
    link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"),
                       dtype={"canonical_mrn": str})
    link = link[link["curated_dir"].notna()]
    cols = ["canonical_mrn", "curated_dir", "hemoglobin", "kvp", "kernel",
            "quality_tier", "slice_thickness"]
    return link[cols].drop_duplicates("canonical_mrn").reset_index(drop=True)


def main():
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
    pts = patient_table()
    if limit:
        pts = pts.head(limit)
    print(f"Measuring anchors on {len(pts)} curated patients...")
    rows = []
    for i, r in pts.iterrows():
        try:
            a = measure_anchors(r["curated_dir"])
        except Exception as e:
            print(f"  [{i}] {r['canonical_mrn']}: FAILED ({e})")
            a = {"air_hu": np.nan, "aorta_hu": np.nan, "air_nvox": 0, "aorta_nvox": 0}
        rec = {**r.to_dict(), **a}
        rows.append(rec)
        if (i + 1) % 10 == 0:
            ok = sum(1 for x in rows if not np.isnan(x.get("aorta_hu", np.nan)))
            print(f"  {i + 1}/{len(pts)} done ({ok} with aorta HU)")
    df = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "drstone_anchors.csv")
    df.to_csv(out, index=False)

    fit = fit_population(df)
    a, b = fit["a"], fit["b"]
    # Per-patient two-anchor calibration params (ready to apply to stone voxels).
    ab = df.apply(lambda r: calibration_params(r["air_hu"], r["aorta_hu"],
                                               r["hemoglobin"], a, b)
                  if pd.notna(r["aorta_hu"]) and pd.notna(r["air_hu"])
                  and pd.notna(r["hemoglobin"]) else (np.nan, np.nan), axis=1)
    df["cal_alpha"] = [x[0] for x in ab]
    df["cal_beta"] = [x[1] for x in ab]
    import json
    json.dump(fit, open(os.path.join(C.OUTPUT_DIR, "drstone_calibration_fit.json"), "w"), indent=2)
    df.to_csv(out, index=False)

    print("\n==== Hb-anchored calibration — population fit ====")
    print(f"patients with valid aorta HU: {fit['n']}")
    print(f"  HU_aorta = {fit['a']:.2f} + {fit['b']:.2f} * Hb   "
          f"(OLS r={fit['ols_r']:.2f}, p={fit['ols_p']:.1e})")
    print(f"  robust (Theil-Sen): slope={fit['ts_slope']:.2f}, intercept={fit['ts_intercept']:.2f}")
    print(f"  [literature expectation: ~1.5-2 HU per g/dL]")

    # Validation scatter (also a grant/preliminary-data figure)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        d = df.dropna(subset=["aorta_hu", "hemoglobin"])
        d = d[(d["aorta_hu"] > -50) & (d["aorta_hu"] < 150)]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.scatter(d["hemoglobin"], d["aorta_hu"], s=18, alpha=0.7, color="#b03030")
        xs = np.linspace(d["hemoglobin"].min(), d["hemoglobin"].max(), 50)
        ax.plot(xs, a + b * xs, "k-", lw=2,
                label=f"HU = {a:.1f} + {b:.2f}·Hb  (r={fit['ols_r']:.2f}, n={fit['n']})")
        ax.set_xlabel("Hemoglobin (g/dL)"); ax.set_ylabel("Aortic blood HU (non-contrast)")
        ax.set_title("Hb-anchored HU calibration: aortic blood HU vs hemoglobin")
        ax.legend(loc="lower right", fontsize=9); fig.tight_layout()
        figp = os.path.join(C.OUTPUT_DIR, "drstone_hb_aorta_fit.png")
        fig.savefig(figp, dpi=200); print(f"  scatter figure -> {figp}")
    except Exception as e:
        print(f"  (figure skipped: {e})")
    print(f"\nWrote anchors + per-patient calibration params -> {out}")


if __name__ == "__main__":
    main()
