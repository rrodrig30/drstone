"""Rich per-stone feature extraction (Dr Stone, expanded feature set).

Segments ALL qualifying stones in a non-contrast CT (not just the largest, so
multi-stone patients can be matched + their N recovered) and computes, on NATIVE
voxels (no resample blurring), a mechanistically-motivated feature set:

  HU distribution shape : mean/std/percentiles/IQR/range, skewness, kurtosis,
                          Shapiro normality stat, histogram entropy, GMM #modes
                          (multimodal => multi-component stone), core-vs-rim
                          zonation (layered mixtures).
  shape                 : volume, surface, sphericity, elongation.

The handcrafted HU summary (peak/mean/p95) stays native here too. Texture (full
IBSI/GLCM) can be layered on later; the histogram-shape features above are the
high-yield, small-N-robust subset.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from drstone.calibration import load_ct
from drstone.stone_segmentation import (
    _run_ts, REGION_ROIS, EXCLUDE_ROIS, STONE_HU, REGION_DILATE_MM,
    EXCLUDE_DILATE_MM, MIN_STONE_MM3, MAX_STONE_MM3)

# Cap components per patient: no patient has >4 analyzed stones, so keeping the
# few largest (after bone/vessel exclusion) drops the calcific-fleck/phlebolith
# flood while retaining every real stone for rank-matching.
MAX_STONES_PER_PATIENT = 6


def hu_distribution_features(hu: np.ndarray) -> dict:
    from scipy import stats
    f = {}
    f["hu_mean"] = float(hu.mean()); f["hu_std"] = float(hu.std())
    f["hu_min"] = float(hu.min()); f["hu_max"] = float(hu.max()); f["hu_peak"] = f["hu_max"]
    for p in (5, 25, 50, 75, 95):
        f[f"hu_p{p}"] = float(np.percentile(hu, p))
    f["hu_iqr"] = f["hu_p75"] - f["hu_p25"]
    f["hu_range"] = f["hu_max"] - f["hu_min"]
    f["hu_skew"] = float(stats.skew(hu)) if hu.size > 2 else np.nan
    f["hu_kurtosis"] = float(stats.kurtosis(hu)) if hu.size > 3 else np.nan
    # normality / "purity": pure stone -> near-unimodal/Gaussian
    s = hu if hu.size <= 5000 else np.random.RandomState(0).choice(hu, 5000, replace=False)
    try:
        f["hu_shapiro"] = float(stats.shapiro(s)[0])
    except Exception:
        f["hu_shapiro"] = np.nan
    hist, _ = np.histogram(hu, bins=32, density=True)
    hist = hist[hist > 0]
    f["hu_entropy"] = float(-(hist * np.log(hist)).sum()) if hist.size else np.nan
    f["hu_nmodes"] = gmm_modes(hu)
    return f


def gmm_modes(hu: np.ndarray, max_k: int = 3) -> int:
    """Number of HU components by GMM/BIC — a multimodal stone is multi-component."""
    from sklearn.mixture import GaussianMixture
    x = hu.reshape(-1, 1).astype(np.float64)
    if x.shape[0] < 30:
        return 1
    bics = []
    for k in range(1, max_k + 1):
        try:
            g = GaussianMixture(k, covariance_type="full", random_state=0).fit(x)
            bics.append(g.bic(x))
        except Exception:
            bics.append(np.inf)
    return int(np.argmin(bics) + 1)


def shape_features(mask: np.ndarray, spacing) -> dict:
    from scipy import ndimage as ndi
    voxvol = float(np.prod(spacing))
    n = int(mask.sum())
    vol = n * voxvol
    out = {"volume_mm3": vol, "n_vox": n, "surface_mm2": np.nan,
           "sphericity": np.nan, "elongation": np.nan}
    try:
        from skimage import measure
        verts, faces, _n, _v = measure.marching_cubes(mask.astype(np.float32), level=0.5, spacing=spacing)
        area = float(measure.mesh_surface_area(verts, faces))
        out["surface_mm2"] = area
        if area > 0:
            out["sphericity"] = float((np.pi ** (1 / 3) * (6 * vol) ** (2 / 3)) / area)
    except Exception:
        pass
    zz, yy, xx = np.where(mask)
    if n > 3:
        coords = np.c_[zz * spacing[0], yy * spacing[1], xx * spacing[2]].astype(np.float64)
        ev = np.linalg.eigvalsh(np.cov(coords.T))
        ev = np.sort(ev)[::-1]
        if ev[0] > 0:
            out["elongation"] = float(np.sqrt(max(ev[1], 0) / ev[0]))
    return out


# Stone-density-gradient gating: a core-vs-periphery split is only meaningful
# when the stone has enough voxels to form concentric shells; below this the
# "taper" is slice-thickness partial-volume, not mineralogy.
GRAD_MIN_VOX = 150          # total stone voxels to attempt a depth split
GRAD_MIN_SHELL = 15         # each of core/periphery shell must reach this


def zonation_feature(ct: np.ndarray, mask: np.ndarray, spacing=None) -> dict:
    """Core-vs-periphery HU zonation (layered/zoned mixtures, density gradient).

    Two views:
      hu_core_minus_rim : legacy mean difference of a 1-voxel erosion core vs
                          its 1-voxel rim (kept for continuity with prior runs).
      hu_core_over_rim  : the density-gradient RATIO HU_core / HU_periphery, with
                          shells defined by physical depth (mm) from the surface
                          via a spacing-aware Euclidean distance transform, split
                          at the 33rd/67th depth percentiles, medians (robust to
                          the partial-volume rim tail). Emitted only when the
                          stone is large enough (grad_measurable=1); NaN otherwise
                          so the model never sees a slice-thickness artifact.
    """
    from scipy import ndimage as ndi
    out = {"hu_core_minus_rim": np.nan, "hu_core_p50": np.nan,
           "hu_rim_p50": np.nan, "hu_core_over_rim": np.nan, "grad_measurable": 0}

    core1 = ndi.binary_erosion(mask, iterations=1)
    rim1 = mask & ~core1
    if core1.sum() >= 10 and rim1.sum() >= 10:
        out["hu_core_minus_rim"] = float(ct[core1].mean() - ct[rim1].mean())

    sampling = spacing if spacing is not None else 1.0
    edt = ndi.distance_transform_edt(mask, sampling=sampling)
    depth = edt[mask]
    if depth.size >= GRAD_MIN_VOX:
        lo, hi = np.percentile(depth, [33, 67])
        if hi > lo:
            core = mask & (edt >= hi)
            rim = mask & (edt > 0) & (edt <= lo)
            if core.sum() >= GRAD_MIN_SHELL and rim.sum() >= GRAD_MIN_SHELL:
                cmed = float(np.median(ct[core]))
                rmed = float(np.median(ct[rim]))
                out["hu_core_p50"] = cmed
                out["hu_rim_p50"] = rmed
                out["grad_measurable"] = 1
                if rmed > 0:
                    out["hu_core_over_rim"] = cmed / rmed
    return out


def extract_all_stones(series_dir: str) -> list:
    """Return a list of per-stone feature dicts for ALL qualifying stones."""
    from scipy import ndimage as ndi
    img, ct, spacing = load_ct(series_dir)
    voxvol = float(np.prod(spacing))
    inplane = max(spacing[1], 0.4)
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
    cand = (ct > STONE_HU) & region & ~excl
    if cand.sum() == 0:
        return []
    lab, n = ndi.label(cand)
    stones = []
    for i in range(1, n + 1):
        comp = lab == i
        vol = comp.sum() * voxvol
        if vol < MIN_STONE_MM3 or vol > MAX_STONE_MM3:
            continue
        hu = ct[comp]
        zz, yy, xx = np.where(comp)
        feats = {"centroid_z": float(zz.mean()), "centroid_y": float(yy.mean()),
                 "centroid_x": float(xx.mean())}
        feats.update(hu_distribution_features(hu))
        feats.update(shape_features(comp, spacing))
        feats.update(zonation_feature(ct, comp, spacing))
        stones.append(feats)
    stones.sort(key=lambda s: -s["volume_mm3"])     # largest first (for rank matching)
    return stones[:MAX_STONES_PER_PATIENT]


def main():
    import sys
    import warnings
    import pandas as pd
    from drstone import config as C
    warnings.filterwarnings("ignore")
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None

    link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"),
                       dtype={"canonical_mrn": str})
    pts = link[link["curated_dir"].notna()].drop_duplicates("canonical_mrn").reset_index(drop=True)
    if limit:
        pts = pts.head(limit)
    print(f"Extracting all-stone rich features for {len(pts)} curated patients...")
    rows = []
    for i, r in pts.iterrows():
        try:
            stones = extract_all_stones(r["curated_dir"])
        except Exception as e:
            print(f"  [{i}] {r['canonical_mrn']}: FAILED ({e})"); stones = []
        for j, s in enumerate(stones):
            rows.append({"canonical_mrn": r["canonical_mrn"], "stone_idx": j,
                         "n_stones_seg": len(stones), **s})
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(pts)} patients ({len(rows)} stones)")
    df = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "drstone_all_stones.csv")
    df.to_csv(out, index=False)
    print(f"\n{len(df)} stones from {df['canonical_mrn'].nunique()} patients -> {out}")


if __name__ == "__main__":
    main()
