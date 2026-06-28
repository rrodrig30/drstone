"""Phase 4a — stone ROI extraction + isotropic resampling for the 3D CNN.

Per the design decision: resample to a MODEST isotropic spacing (1.0 mm default,
not 0.5 mm — the native through-plane is 2-3 mm, so 0.5 mm is mostly fabricated
detail and blurs peak HU). We crop a fixed physical cube around the stone and
resample only that ROI. The native handcrafted HU features are kept separate
(already in drstone_stone_hu.csv) so the CNN's smoothing can't blur them.

Spacing is configurable (env DRSTONE_PATCH_ISO_MM) so we can A/B 1.0 vs 0.5 vs
native, exactly like the calibration test.

Run (as a module — nnUNet spawns workers that re-import the entry):
    python -m drstone.interpolate            # all single-stone patients
    python -m drstone.interpolate --limit 3  # quick test
Outputs:
    drstone_data/patches/<mrn>.npy   int16 HU cube (PATCH_VOX^3) + _mask.npy
    drstone_data/drstone_patches.csv manifest (mrn, path, label, centroid, found)
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.stone_segmentation import segment_stone_full

warnings.filterwarnings("ignore")

PATCH_ISO_MM = float(os.environ.get("DRSTONE_PATCH_ISO_MM", "1.0"))
PATCH_PHYS_MM = 40.0                                  # physical cube edge around stone
PATCH_VOX = int(round(PATCH_PHYS_MM / PATCH_ISO_MM))  # output cube side
PATCH_DIR = os.path.join(C.OUTPUT_DIR, "patches")
os.makedirs(PATCH_DIR, exist_ok=True)
HU_CLIP = (-1024, 3071)


def extract_patch(ct, spacing, centroid_zyx):
    """Crop a physical cube around the stone and resample to isotropic PATCH_ISO_MM."""
    from scipy import ndimage as ndi
    cz, cy, cx = [int(round(v)) for v in centroid_zyx]
    half = [int(np.ceil((PATCH_PHYS_MM / 2.0) / s)) for s in spacing]    # native half-voxels
    z0, z1 = cz - half[0], cz + half[0]
    y0, y1 = cy - half[1], cy + half[1]
    x0, x1 = cx - half[2], cx + half[2]
    # pad-safe crop (fill out-of-bounds with air)
    sub = np.full((z1 - z0, y1 - y0, x1 - x0), HU_CLIP[0], np.float32)
    zs, ys, xs = max(z0, 0), max(y0, 0), max(x0, 0)
    ze, ye, xe = min(z1, ct.shape[0]), min(y1, ct.shape[1]), min(x1, ct.shape[2])
    sub[zs - z0:ze - z0, ys - y0:ye - y0, xs - x0:xe - x0] = ct[zs:ze, ys:ye, xs:xe]
    # resample to isotropic (cubic for CT; clip to suppress overshoot)
    zoom = [spacing[i] / PATCH_ISO_MM for i in range(3)]
    iso = ndi.zoom(sub, zoom, order=3)
    iso = np.clip(iso, *HU_CLIP)
    # center crop / pad to exact PATCH_VOX^3
    out = np.full((PATCH_VOX,) * 3, HU_CLIP[0], np.float32)
    s = iso.shape
    src = [slice(max(0, (s[a] - PATCH_VOX) // 2), max(0, (s[a] - PATCH_VOX) // 2) + min(s[a], PATCH_VOX)) for a in range(3)]
    dst = [slice(max(0, (PATCH_VOX - s[a]) // 2), max(0, (PATCH_VOX - s[a]) // 2) + min(s[a], PATCH_VOX)) for a in range(3)]
    out[dst[0], dst[1], dst[2]] = iso[src[0], src[1], src[2]]
    return out.astype(np.int16)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"),
                       dtype={"canonical_mrn": str})
    ss = link[(link["single_stone_patient"] == True) & (link["curated_dir"].notna())]
    ss = ss.drop_duplicates("canonical_mrn").reset_index(drop=True)
    if limit:
        ss = ss.head(limit)
    print(f"Extracting {PATCH_VOX}^3 patches @ {PATCH_ISO_MM}mm iso for {len(ss)} patients...")
    rows = []
    for i, r in ss.iterrows():
        rec = {"canonical_mrn": r["canonical_mrn"], "dominant_parent": r["dominant_parent"],
               "found": False, "patch_path": ""}
        try:
            ct, spacing, mask, stats = segment_stone_full(r["curated_dir"])
            if stats["found"]:
                patch = extract_patch(ct, spacing, stats["centroid_zyx"])
                pp = os.path.join(PATCH_DIR, f"{r['canonical_mrn']}.npy")
                np.save(pp, patch)
                if mask is not None:
                    mp = extract_patch(mask.astype(np.float32) * 1000, spacing, stats["centroid_zyx"])
                    np.save(os.path.join(PATCH_DIR, f"{r['canonical_mrn']}_mask.npy"),
                            (mp > 500).astype(np.uint8))
                rec.update(found=True, patch_path=pp,
                           centroid=str(stats["centroid_zyx"]), peak_hu=stats["peak_hu"])
        except Exception as e:
            print(f"  [{i}] {r['canonical_mrn']}: FAILED ({e})")
        rows.append(rec)
        if (i + 1) % 10 == 0:
            ok = sum(1 for x in rows if x["found"])
            print(f"  {i + 1}/{len(ss)} ({ok} patches)")
    man = pd.DataFrame(rows)
    out = os.path.join(C.OUTPUT_DIR, "drstone_patches.csv")
    man.to_csv(out, index=False)
    print(f"\n{int(man['found'].sum())}/{len(man)} patches written -> {PATCH_DIR}\nmanifest -> {out}")


if __name__ == "__main__":
    main()
