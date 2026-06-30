"""Vascular / chronic-infection imaging markers from the non-contrast CT.

  aortic calcification : high-HU (>130) voxels in the aortic wall shell (aorta
                         dilated, bone excluded). A readout of disordered
                         calcium-phosphate / mineral metabolism (CKD-MBD, high
                         Ca x PO4, secondary hyperparathyroidism) — the same axis
                         that precipitates CALCIUM-PHOSPHATE stones, and NOT
                         otherwise measured (no serum phosphate/PTH in the data).
  bladder              : volume + a crude soft-tissue-wall fraction (chronic
                         cystitis/obstruction -> struvite). Exploratory:
                         distension-confounded, struvite n=27.
  prostate calc        : high-HU foci in prostate (males only). Weak/nonspecific.

All from the `total` TS task in one pass; patient-level. Sharded/resumable.

Run:  python -m drstone.vascular_extract [--nshards N --shard K [--suffix s --reverse]]
"""

from __future__ import annotations

import glob
import os
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd

from drstone import config as C

OUT = os.path.join(C.OUTPUT_DIR, "drstone_vascular.csv")
CALC_HU = 130.0
AORTA_WALL_MM = 2.0       # wall shell around the (non-contrast) aortic lumen
BONE_PAD_MM = 2.0         # keep adjacent vertebral bone out of the aortic ROI
BONE_ROIS = ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
             "vertebrae_L5", "vertebrae_T12"]
ROIS = ["aorta", "urinary_bladder", "prostate"] + BONE_ROIS


def extract_vascular(series_dir: str) -> dict:
    from scipy import ndimage as ndi
    from drstone.calibration import load_ct
    from drstone.stone_segmentation import _run_ts
    out = {"aortic_calc_vol_mm3": np.nan, "aortic_calc_frac": np.nan,
           "aortic_calc_nfoci": np.nan, "aorta_vol_mm3": np.nan,
           "bladder_vol_mm3": np.nan, "bladder_wall_frac": np.nan,
           "prostate_present": 0, "prostate_calc_vol_mm3": np.nan}
    img, ct, spacing = load_ct(series_dir)
    voxvol = float(np.prod(spacing)); inplane = max(spacing[1], 0.4)
    with tempfile.TemporaryDirectory() as wd:
        masks = _run_ts(img, ROIS, wd)
    if not masks:
        return out

    def _it(mm):
        return max(1, int(mm / inplane))

    # ---- aortic wall calcification --------------------------------------
    aorta = masks.get("aorta")
    if aorta is not None and aorta.sum() > 200:
        bone = np.zeros(ct.shape, bool)
        for r in BONE_ROIS:
            if r in masks:
                bone |= masks[r]
        shell = ndi.binary_dilation(aorta, iterations=_it(AORTA_WALL_MM)) & ~aorta
        if bone.any():
            shell &= ~ndi.binary_dilation(bone, iterations=_it(BONE_PAD_MM))
        calc = shell & (ct > CALC_HU)
        _, nf = ndi.label(calc)
        out["aorta_vol_mm3"] = float(aorta.sum() * voxvol)
        out["aortic_calc_vol_mm3"] = float(calc.sum() * voxvol)
        out["aortic_calc_frac"] = float(calc.sum() / max(1, aorta.sum()))
        out["aortic_calc_nfoci"] = int(nf)

    # ---- bladder volume + crude wall fraction (exploratory) -------------
    bl = masks.get("urinary_bladder")
    if bl is not None and bl.sum() > 200:
        out["bladder_vol_mm3"] = float(bl.sum() * voxvol)
        # soft-tissue wall voxels (25-70 HU) vs near-water urine (<25 HU)
        bhu = ct[bl]
        out["bladder_wall_frac"] = float(((bhu >= 25) & (bhu <= 70)).mean())

    # ---- prostate calcification (males) ---------------------------------
    pr = masks.get("prostate")
    if pr is not None and pr.sum() > 100:
        out["prostate_present"] = 1
        out["prostate_calc_vol_mm3"] = float((pr & (ct > CALC_HU)).sum() * voxvol)
    return out


# --------------------------------------------------------------------------
# Sharded, resumable harness
# --------------------------------------------------------------------------
def cohort_patients() -> pd.DataFrame:
    link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"),
                       dtype={"canonical_mrn": str})
    ms = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv"),
                     dtype={"canonical_mrn": str})
    coh = set(ms[ms.match_quality.isin(["rank_mass", "single_comp"])].canonical_mrn)
    return (link[link.curated_dir.notna() & link.canonical_mrn.isin(coh)]
            .drop_duplicates("canonical_mrn").reset_index(drop=True))


def _arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main() -> None:
    warnings.filterwarnings("ignore")
    limit = int(_arg("--limit")) if "--limit" in sys.argv else None
    nshards = int(_arg("--nshards", 1)); shard = int(_arg("--shard", 0))
    suffix = _arg("--suffix", ""); reverse = "--reverse" in sys.argv
    out_path = OUT if nshards == 1 else OUT.replace(".csv", f".part{shard}{suffix}.csv")

    pts = cohort_patients()
    if nshards > 1:
        pts = pts[pts.index % nshards == shard].reset_index(drop=True)
    if reverse:
        pts = pts.iloc[::-1].reset_index(drop=True)
    if limit:
        pts = pts.head(limit)

    def load_done():
        d = set()
        for src in glob.glob(OUT.replace(".csv", f".part{shard}*.csv")) + [out_path]:
            if os.path.exists(src):
                try:
                    d |= set(pd.read_csv(src, dtype={"canonical_mrn": str}).canonical_mrn)
                except Exception:
                    pass
        return d

    done = load_done()
    if done:
        print(f"resuming: {len(done)} patients done", flush=True)
    print(f"cohort: {len(pts)} patients ({len(pts) - len(done & set(pts.canonical_mrn))} to do)",
          flush=True)
    t0 = time.time()
    for i, r in pts.iterrows():
        mrn = r["canonical_mrn"]
        if mrn in load_done():
            continue
        t = time.time()
        try:
            feats = extract_vascular(r["curated_dir"])
        except Exception as e:
            print(f"[{i+1}/{len(pts)}] {mrn} FAILED: {e}", flush=True)
            continue
        pd.DataFrame([{"canonical_mrn": mrn, **feats}]).to_csv(
            out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(f"[{i+1}/{len(pts)}] {mrn}: aortic_calc={feats['aortic_calc_vol_mm3']}, "
              f"prostate={feats['prostate_present']} in {time.time()-t:.0f}s "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"done -> {out_path}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
