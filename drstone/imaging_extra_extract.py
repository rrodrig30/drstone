"""Step 2 extraction: patient-level imaging features beyond the stone itself.

  nephrocalcinosis : parenchymal calcification load — high-HU voxels inside the
                     eroded renal parenchyma, with the collecting-system stone(s)
                     and bone/vessel calcification removed. Direct imaging of the
                     distal-RTA / hyperparathyroidism / CaP phenotype, and NOT a
                     function of any lab we hold.
  vertebral BMD    : opportunistic bone density — median HU of the trabecular core
                     of L1 (+ L2/L3) from the vertebra masks. Age/sex-confounded;
                     tested as a controlled "does it beat age?" check.

One TotalSegmentator pass per patient (kidneys + vertebrae + the stone-exclusion
ROIs). Patient-level → one row per canonical_mrn. Runs via `python -m` (TS
multiprocessing). Resumable + shardable like gradient_extract.

Run:  python -m drstone.imaging_extra_extract [--nshards N --shard K [--suffix s --reverse]]
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

OUT = os.path.join(C.OUTPUT_DIR, "drstone_imaging_extra.csv")
CALC_HU = 130.0           # calcification threshold on non-contrast CT
CALC_FOCUS_MIN_MM3 = 2.0  # ignore single-voxel noise
CALC_FOCUS_MAX_MM3 = 50.0 # exclude stone bleed-through (large contiguous calcification)
KID_ERODE_MM = 4.0        # parenchymal core (away from pelvis / partial volume)
STONE_PAD_MM = 3.0        # remove the collecting-system stone(s)
BONE_PAD_MM = 2.0         # remove rib/vertebra/vessel calcification bleed-in
VERT_ERODE_MM = 4.0       # trabecular core of the vertebral body
BMD_LEVELS = ("vertebrae_L1", "vertebrae_L2", "vertebrae_L3")


def compute_nephro(ct, spacing, masks) -> dict:
    """Nephrocalcinosis (parenchymal calcification load) from precomputed masks."""
    from scipy import ndimage as ndi
    from drstone.stone_segmentation import (
        REGION_ROIS, EXCLUDE_ROIS, STONE_HU, REGION_DILATE_MM,
        EXCLUDE_DILATE_MM, MIN_STONE_MM3, MAX_STONE_MM3)
    out = {"nephrocalc_vol_mm3": np.nan, "nephrocalc_frac": np.nan,
           "nephrocalc_peak_hu": np.nan, "nephrocalc_nfoci": np.nan,
           "kidney_vol_mm3": np.nan}
    voxvol = float(np.prod(spacing)); inplane = max(spacing[1], 0.4)

    def _it(mm):
        return max(1, int(mm / inplane))

    kid = np.zeros(ct.shape, bool)
    for r in ("kidney_left", "kidney_right"):
        if r in masks:
            kid |= masks[r]
    region = np.zeros(ct.shape, bool)
    for r in REGION_ROIS:
        if r in masks:
            region |= masks[r]
    excl = np.zeros(ct.shape, bool)
    for r in EXCLUDE_ROIS:
        if r in masks:
            excl |= masks[r]

    stone = np.zeros(ct.shape, bool)
    if region.any():
        region_d = ndi.binary_dilation(region, iterations=_it(REGION_DILATE_MM))
        excl_d = ndi.binary_dilation(excl, iterations=_it(EXCLUDE_DILATE_MM)) if excl.any() else excl
        lab, n = ndi.label((ct > STONE_HU) & region_d & ~excl_d)
        for i in range(1, n + 1):
            comp = lab == i
            if MIN_STONE_MM3 <= comp.sum() * voxvol <= MAX_STONE_MM3:
                stone |= comp

    if kid.any():
        roi = ndi.binary_erosion(kid, iterations=_it(KID_ERODE_MM))
        if stone.any():
            roi &= ~ndi.binary_dilation(stone, iterations=_it(STONE_PAD_MM))
        if excl.any():
            roi &= ~ndi.binary_dilation(excl, iterations=_it(BONE_PAD_MM))
        nvox = int(roi.sum())
        if nvox > 0:
            # keep only small scattered foci; large contiguous = stone bleed-through
            lab_c, ncomp = ndi.label(roi & (ct > CALC_HU))
            small = np.zeros(ct.shape, bool); nfoci = 0
            for i in range(1, ncomp + 1):
                comp = lab_c == i
                if CALC_FOCUS_MIN_MM3 <= comp.sum() * voxvol <= CALC_FOCUS_MAX_MM3:
                    small |= comp; nfoci += 1
            out["kidney_vol_mm3"] = float(kid.sum() * voxvol)
            out["nephrocalc_vol_mm3"] = float(small.sum() * voxvol)
            out["nephrocalc_frac"] = float(small.sum() / nvox)
            out["nephrocalc_peak_hu"] = float(ct[small].max()) if small.any() else 0.0
            out["nephrocalc_nfoci"] = int(nfoci)
    return out


def compute_bmd(ct, spacing, masks) -> dict:
    """Opportunistic trabecular-core BMD (median HU) of L1 (+ L2/L3)."""
    from scipy import ndimage as ndi
    out = {"bmd_l1_hu": np.nan, "bmd_lumbar_hu": np.nan, "bmd_nlevels": 0}
    inplane = max(spacing[1], 0.4)
    levels = []
    for lv in BMD_LEVELS:
        m = masks.get(lv)
        if m is not None and m.sum() > 50:
            core = ndi.binary_erosion(m, iterations=max(1, int(VERT_ERODE_MM / inplane)))
            if core.sum() >= 30:
                levels.append((lv, float(np.median(ct[core]))))
    by = dict(levels)
    if "vertebrae_L1" in by:
        out["bmd_l1_hu"] = by["vertebrae_L1"]
    if levels:
        out["bmd_lumbar_hu"] = float(np.mean([v for _, v in levels]))
        out["bmd_nlevels"] = len(levels)
    return out


def _ts_masks(series_dir):
    from drstone.calibration import load_ct
    from drstone.stone_segmentation import _run_ts, REGION_ROIS, EXCLUDE_ROIS
    img, ct, spacing = load_ct(series_dir)
    rois = sorted(set(REGION_ROIS + EXCLUDE_ROIS + list(BMD_LEVELS)))
    with tempfile.TemporaryDirectory() as wd:
        masks = _run_ts(img, rois, wd)
    return ct, spacing, masks


def extract_imaging_extra(series_dir: str) -> dict:
    out = {"nephrocalc_vol_mm3": np.nan, "nephrocalc_frac": np.nan,
           "nephrocalc_peak_hu": np.nan, "nephrocalc_nfoci": np.nan,
           "kidney_vol_mm3": np.nan, "bmd_l1_hu": np.nan,
           "bmd_lumbar_hu": np.nan, "bmd_nlevels": 0}
    ct, spacing, masks = _ts_masks(series_dir)
    if not masks:
        return out
    out.update(compute_nephro(ct, spacing, masks))
    out.update(compute_bmd(ct, spacing, masks))
    return out


def measure_full(series_dir: str, max_stones: int = 6) -> dict:
    """One TS pass -> stones (for the picker) + nephrocalcinosis (for the flag)."""
    from drstone.stone_segmentation import stones_from_masks
    ct, spacing, masks = _ts_masks(series_dir)
    if not masks:
        return {"stones": [], "nephrocalc": {}}
    return {"stones": stones_from_masks(ct, spacing, masks, max_stones),
            "nephrocalc": compute_nephro(ct, spacing, masks)}


# --------------------------------------------------------------------------
# Sharded, resumable harness (mirrors gradient_extract)
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
            feats = extract_imaging_extra(r["curated_dir"])
        except Exception as e:
            print(f"[{i+1}/{len(pts)}] {mrn} FAILED: {e}", flush=True)
            continue
        row = {"canonical_mrn": mrn, **feats}
        pd.DataFrame([row]).to_csv(out_path, mode="a",
                                   header=not os.path.exists(out_path), index=False)
        print(f"[{i+1}/{len(pts)}] {mrn}: neph_frac={feats['nephrocalc_frac']}, "
              f"bmd_l1={feats['bmd_l1_hu']} in {time.time()-t:.0f}s "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"done -> {out_path}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
