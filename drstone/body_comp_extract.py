"""Body-composition imaging features from the non-contrast CT (microbiome/
metabolic hypothesis): hepatic steatosis and muscle quality.

  fatty liver : liver HU - spleen HU (negative => steatosis), median HU of each
                eroded organ (robust to vessels). Candidate readout of the
                gut-dysbiosis/NAFLD state that raises enteric oxalate absorption
                (Oxalobacter loss) -> selectively favors CaOx over CaPO4.
  muscle      : mean HU of paraspinal (autochthon) + psoas (iliopsoas) muscle —
                a height-free, FOV-robust myosteatosis/sarcopenia marker (low HU =
                fatty infiltration), + muscle volume.

All from the `total` TotalSegmentator task (liver/spleen/autochthon/iliopsoas) in
one pass. Patient-level -> one row per canonical_mrn. Sharded/resumable; run via
`python -m`.

Run:  python -m drstone.body_comp_extract [--nshards N --shard K [--suffix s --reverse]]
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

OUT = os.path.join(C.OUTPUT_DIR, "drstone_body_comp.csv")
ORGAN_ERODE_MM = 4.0
MUSCLE_ROIS = ("autochthon_left", "autochthon_right",
               "iliopsoas_left", "iliopsoas_right")
ROIS = ["liver", "spleen"] + list(MUSCLE_ROIS)


def extract_body_comp(series_dir: str) -> dict:
    from scipy import ndimage as ndi
    from drstone.calibration import load_ct
    from drstone.stone_segmentation import _run_ts
    out = {k: np.nan for k in ("liver_hu", "spleen_hu", "liver_spleen_diff",
                               "muscle_hu", "muscle_vol_mm3")}
    img, ct, spacing = load_ct(series_dir)
    voxvol = float(np.prod(spacing)); inplane = max(spacing[1], 0.4)
    with tempfile.TemporaryDirectory() as wd:
        masks = _run_ts(img, ROIS, wd)
    if not masks:
        return out

    def _it(mm):
        return max(1, int(mm / inplane))

    def organ_median(name):
        m = masks.get(name)
        if m is None or m.sum() < 200:
            return None
        core = ndi.binary_erosion(m, iterations=_it(ORGAN_ERODE_MM))
        if core.sum() < 50:
            core = m
        return float(np.median(ct[core]))

    lv, sp = organ_median("liver"), organ_median("spleen")
    if lv is not None:
        out["liver_hu"] = lv
    if sp is not None:
        out["spleen_hu"] = sp
    if lv is not None and sp is not None:
        out["liver_spleen_diff"] = lv - sp

    musc = np.zeros(ct.shape, bool)
    for r in MUSCLE_ROIS:
        if r in masks:
            musc |= masks[r]
    if musc.sum() > 200:
        core = ndi.binary_erosion(musc, iterations=_it(2.0))
        if core.sum() < 50:
            core = musc
        out["muscle_hu"] = float(np.median(ct[core]))
        out["muscle_vol_mm3"] = float(musc.sum() * voxvol)
    return out


# --------------------------------------------------------------------------
# Sharded, resumable harness (mirrors imaging_extra_extract)
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
            feats = extract_body_comp(r["curated_dir"])
        except Exception as e:
            print(f"[{i+1}/{len(pts)}] {mrn} FAILED: {e}", flush=True)
            continue
        pd.DataFrame([{"canonical_mrn": mrn, **feats}]).to_csv(
            out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(f"[{i+1}/{len(pts)}] {mrn}: L-S={feats['liver_spleen_diff']}, "
              f"musc_hu={feats['muscle_hu']} in {time.time()-t:.0f}s "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"done -> {out_path}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
