"""Deconvolve each stone's HU histogram into two overlapping Gaussian peaks and
quantify the two-ness — for predicting multicomponent stones.

Rationale: a laminated dual stone superposes two component HU distributions that
overlap into one broad, often skewed peak. Two symmetric peaks of unequal height
sum to a SKEWED composite, so skewness + a fitted 2-vs-1-Gaussian separation may
flag two components better than raw moments. Per stone we fit GaussianMixture(1)
and (2) to the native HU voxels and record:

  dc_skew, dc_kurt   distribution shape (excess kurtosis)
  dc_bimod           Sarle's bimodality coefficient (skew^2+1)/kurt; >0.555 -> bimodal
  dc_dbic            BIC(1) - BIC(2)  (positive => two components preferred)
  dc_musep           |mu1 - mu2| / pooled SD  (standardized peak separation)
  dc_wmin            smaller mixing weight     (how balanced the two peaks are)
  dc_sdratio         SD ratio of the two fitted peaks

Needs the per-stone HU voxels -> re-segmentation. Run via `python -m`; sharded.

Run: python -m drstone.deconvolve_extract [--nshards N --shard K [--suffix s --reverse]]
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

OUT = os.path.join(C.OUTPUT_DIR, "drstone_deconv.csv")
_KEYS = ("dc_skew", "dc_kurt", "dc_bimod", "dc_dbic", "dc_musep", "dc_wmin",
         "dc_sdratio", "dc_nvox")


def deconv_features(hu: np.ndarray) -> dict:
    from scipy import stats
    from sklearn.mixture import GaussianMixture
    out = {k: np.nan for k in _KEYS}
    n = int(hu.size); out["dc_nvox"] = n
    if n < 30:
        return out
    out["dc_skew"] = float(stats.skew(hu))
    out["dc_kurt"] = float(stats.kurtosis(hu))            # excess
    kfull = out["dc_kurt"] + 3.0
    out["dc_bimod"] = float((out["dc_skew"] ** 2 + 1.0) / kfull) if kfull > 0 else np.nan
    x = hu.reshape(-1, 1).astype(np.float64)
    try:
        g1 = GaussianMixture(1, random_state=0).fit(x)
        g2 = GaussianMixture(2, covariance_type="full", random_state=0).fit(x)
        out["dc_dbic"] = float(g1.bic(x) - g2.bic(x))
        mu = g2.means_.ravel(); w = g2.weights_.ravel(); sd = np.sqrt(g2.covariances_.ravel())
        pooled = float(np.sqrt((w * sd ** 2).sum()))
        out["dc_musep"] = float(abs(mu[0] - mu[1]) / pooled) if pooled > 0 else 0.0
        out["dc_wmin"] = float(w.min())
        out["dc_sdratio"] = float(sd.max() / max(sd.min(), 1e-6))
    except Exception:
        pass
    return out


def stone_hu_voxels(series_dir: str, max_stones: int = 6) -> list:
    """Native HU voxel arrays for each qualifying stone (largest first)."""
    from scipy import ndimage as ndi
    from drstone.calibration import load_ct
    from drstone.stone_segmentation import (
        _run_ts, REGION_ROIS, EXCLUDE_ROIS, STONE_HU, REGION_DILATE_MM,
        EXCLUDE_DILATE_MM, MIN_STONE_MM3, MAX_STONE_MM3)
    img, ct, spacing = load_ct(series_dir)
    voxvol = float(np.prod(spacing)); inplane = max(spacing[1], 0.4)
    with tempfile.TemporaryDirectory() as wd:
        masks = _run_ts(img, REGION_ROIS + EXCLUDE_ROIS, wd)
    if not masks:
        return []
    region = np.zeros(ct.shape, bool)
    for r in REGION_ROIS:
        if r in masks:
            region |= masks[r]
    if region.sum() == 0:
        return []
    excl = np.zeros(ct.shape, bool)
    for r in EXCLUDE_ROIS:
        if r in masks:
            excl |= masks[r]
    region = ndi.binary_dilation(region, iterations=max(1, int(REGION_DILATE_MM / inplane)))
    if excl.any():
        excl = ndi.binary_dilation(excl, iterations=max(1, int(EXCLUDE_DILATE_MM / inplane)))
    lab, n = ndi.label((ct > STONE_HU) & region & ~excl)
    stones = []
    for i in range(1, n + 1):
        comp = lab == i; vol = comp.sum() * voxvol
        if MIN_STONE_MM3 <= vol <= MAX_STONE_MM3:
            stones.append((vol, ct[comp].astype(np.float64)))
    stones.sort(key=lambda s: -s[0])
    return [hu for _v, hu in stones[:max_stones]]


def cohort_patients() -> pd.DataFrame:
    link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"), dtype={"canonical_mrn": str})
    ms = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv"), dtype={"canonical_mrn": str})
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
        print(f"resuming: {len(done)} done", flush=True)
    print(f"cohort: {len(pts)} patients ({len(pts) - len(done & set(pts.canonical_mrn))} to do)", flush=True)
    t0 = time.time()
    for i, r in pts.iterrows():
        mrn = r["canonical_mrn"]
        if mrn in load_done():
            continue
        t = time.time()
        try:
            hus = stone_hu_voxels(r["curated_dir"])
        except Exception as e:
            print(f"[{i+1}/{len(pts)}] {mrn} FAILED: {e}", flush=True); continue
        rows = [{"canonical_mrn": mrn, "stone_idx": j, **deconv_features(hu)}
                for j, hu in enumerate(hus)]
        if rows:
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        print(f"[{i+1}/{len(pts)}] {mrn}: {len(hus)} stones in {time.time()-t:.0f}s "
              f"(elapsed {(time.time()-t0)/60:.1f}m)", flush=True)
    print(f"done -> {out_path}  ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
