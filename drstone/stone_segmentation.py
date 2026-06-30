"""Auto-segment the kidney stone in a non-contrast CT (Dr Stone Phase 2).

Strategy (non-contrast, stones are high-attenuation):
  1. TotalSegmentator (fast) for the urinary search region (kidneys + bladder)
     and the structures to exclude (skeleton + great vessels — bone and vascular
     calcification are the usual false positives).
  2. Threshold high-HU voxels inside the dilated urinary region, minus a dilated
     skeleton/vessel exclusion.
  3. Connected components, size-filter to plausible stone volumes, and select the
     stone (largest qualifying component — single-stone primary cohort).

Returns raw HU statistics (peak/mean/p95) + volume; calibration schemes are
applied downstream in the A/B test, not here.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np

from drstone.calibration import load_ct, TS_DEVICE

# Stone detection constants
STONE_HU = 150.0            # catch low-attenuation (uric-acid) stones too
REGION_DILATE_MM = 12.0     # kidney/bladder -> collecting system + proximal ureter
EXCLUDE_DILATE_MM = 3.0     # pad skeleton/vessels before subtracting
MIN_STONE_MM3 = 4.0         # ~2 mm stone
MAX_STONE_MM3 = 20000.0     # large staghorn ceiling (exclude bone)

# TotalSegmentator 'total' ROIs we need.
REGION_ROIS = ["kidney_left", "kidney_right", "urinary_bladder"]
EXCLUDE_ROIS = ["vertebrae_L1", "vertebrae_L2", "vertebrae_L3", "vertebrae_L4",
                "vertebrae_L5", "vertebrae_T12", "hip_left", "hip_right",
                "sacrum", "aorta", "inferior_vena_cava"]


def _run_ts(sitk_img, roi_subset, workdir):
    """Run TotalSegmentator (fast) for a subset of ROIs; return {name: mask ZYX}."""
    import SimpleITK as sitk
    import nibabel as nib
    from totalsegmentator.python_api import totalsegmentator
    ct_path = os.path.join(workdir, "ct.nii.gz")
    sitk.WriteImage(sitk_img, ct_path)
    seg_dir = os.path.join(workdir, "seg")
    os.makedirs(seg_dir, exist_ok=True)
    totalsegmentator(ct_path, output=seg_dir, task="total", fast=True,
                     roi_subset=roi_subset, device=TS_DEVICE, quiet=True)
    out = {}
    for roi in roi_subset:
        p = os.path.join(seg_dir, roi + ".nii.gz")
        if os.path.exists(p):
            m = np.asarray(nib.load(p).dataobj)
            out[roi] = np.transpose(m, (2, 1, 0)) > 0
    return out


def _locate_stone(series_dir: str):
    """Core: return (ct ZYX, spacing ZYX, stone_mask|None, stats)."""
    from scipy import ndimage as ndi
    img, ct, spacing = load_ct(series_dir)
    voxvol = float(spacing[0] * spacing[1] * spacing[2])
    inplane = max(spacing[1], 0.4)
    stats = {"found": False, "peak_hu": np.nan, "mean_hu": np.nan, "p95_hu": np.nan,
             "volume_mm3": np.nan, "n_vox": 0, "centroid_zyx": None, "bbox": None}

    with tempfile.TemporaryDirectory() as wd:
        masks = _run_ts(img, REGION_ROIS + EXCLUDE_ROIS, wd)
    if not masks:
        return ct, spacing, None, stats
    shape = ct.shape
    region = np.zeros(shape, bool)
    for r in REGION_ROIS:
        if r in masks:
            region |= masks[r]
    if region.sum() == 0:
        return ct, spacing, None, stats
    excl = np.zeros(shape, bool)
    for r in EXCLUDE_ROIS:
        if r in masks:
            excl |= masks[r]

    region = ndi.binary_dilation(region, iterations=max(1, int(REGION_DILATE_MM / inplane)))
    if excl.any():
        excl = ndi.binary_dilation(excl, iterations=max(1, int(EXCLUDE_DILATE_MM / inplane)))

    cand = (ct > STONE_HU) & region & ~excl
    if cand.sum() == 0:
        return ct, spacing, None, stats
    lab, n = ndi.label(cand)
    best, best_size = None, 0
    for i in range(1, n + 1):
        comp = lab == i
        vol = comp.sum() * voxvol
        if vol < MIN_STONE_MM3 or vol > MAX_STONE_MM3:
            continue
        if comp.sum() > best_size:           # single-stone: largest qualifying object
            best, best_size = comp, comp.sum()
    if best is None:
        return ct, spacing, None, stats

    hu = ct[best]
    zz, yy, xx = np.where(best)
    stats.update(found=True, peak_hu=float(hu.max()), mean_hu=float(hu.mean()),
                 p95_hu=float(np.percentile(hu, 95)), volume_mm3=float(best_size * voxvol),
                 n_vox=int(best_size),
                 centroid_zyx=[float(zz.mean()), float(yy.mean()), float(xx.mean())],
                 bbox=[int(zz.min()), int(zz.max()), int(yy.min()), int(yy.max()),
                       int(xx.min()), int(xx.max())])
    return ct, spacing, best, stats


def segment_stone(series_dir: str) -> dict:
    """Segment the dominant stone; return raw-HU stats only (ab_test compat)."""
    _ct, _sp, _mask, stats = _locate_stone(series_dir)
    return {k: stats[k] for k in
            ("found", "peak_hu", "mean_hu", "p95_hu", "volume_mm3", "n_vox")}


def segment_stone_full(series_dir: str):
    """Return (ct ZYX, spacing ZYX, stone_mask, stats incl. centroid/bbox)."""
    return _locate_stone(series_dir)


def list_stones(series_dir: str, max_stones: int = 6) -> list:
    """List ALL qualifying stones (largest first) with HU + volume + anatomic
    location, so a user can pick the intended stone in a multi-stone scan."""
    img, ct, spacing = load_ct(series_dir)
    with tempfile.TemporaryDirectory() as wd:
        masks = _run_ts(img, REGION_ROIS + EXCLUDE_ROIS, wd)
    return stones_from_masks(ct, spacing, masks, max_stones)


def stones_from_masks(ct, spacing, masks, max_stones: int = 6) -> list:
    """The stone-listing core, on precomputed TS masks (shareable TS pass)."""
    from scipy import ndimage as ndi
    if not masks:
        return []
    voxvol = float(np.prod(spacing))
    inplane = max(spacing[1], 0.4)
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
    region_d = ndi.binary_dilation(region, iterations=max(1, int(REGION_DILATE_MM / inplane)))
    if excl.any():
        excl = ndi.binary_dilation(excl, iterations=max(1, int(EXCLUDE_DILATE_MM / inplane)))
    cand = (ct > STONE_HU) & region_d & ~excl
    if cand.sum() == 0:
        return []
    lab, n = ndi.label(cand)
    # Lateralize against modestly-dilated organ masks (renal-pelvis stones sit
    # just outside the parenchymal kidney mask).
    loc_it = max(1, int(8.0 / inplane))
    loc_masks = []
    for rn, nm in [("kidney_left", "Left kidney"), ("kidney_right", "Right kidney"),
                   ("urinary_bladder", "Bladder")]:
        if rn in masks:
            loc_masks.append((nm, ndi.binary_dilation(masks[rn], iterations=loc_it)))
    out = []
    for i in range(1, n + 1):
        comp = lab == i
        vol = comp.sum() * voxvol
        if vol < MIN_STONE_MM3 or vol > MAX_STONE_MM3:
            continue
        hu = ct[comp]
        zz, yy, xx = np.where(comp)
        cz, cy, cx = int(zz.mean()), int(yy.mean()), int(xx.mean())
        loc = "Renal / ureteral"
        for nm, m in loc_masks:
            if m[cz, cy, cx]:
                loc = nm
                break
        # Max axis-aligned caliper (mm) for the size-based acute decision.
        dmax = max((zz.max() - zz.min() + 1) * spacing[0],
                   (yy.max() - yy.min() + 1) * spacing[1],
                   (xx.max() - xx.min() + 1) * spacing[2])
        out.append({"peak_hu": float(hu.max()), "mean_hu": float(hu.mean()),
                    "p95_hu": float(np.percentile(hu, 95)), "volume_mm3": float(vol),
                    "max_diameter_mm": float(dmax), "location": loc})
    out.sort(key=lambda s: -s["volume_mm3"])
    return out[:max_stones]
