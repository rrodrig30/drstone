"""Deconvolution features vs. the multicomponent endpoint — the 15th family.

Tests whether fitting two overlapping Gaussians per stone (peak separation,
bimodality coefficient, skewness, 2-vs-1 BIC) detects multicomponent stones that
the raw HU-shape moments could not. Guards against two confounds:
  - size: dc_dbic (BIC) scales with voxel count, and larger stones may be more
    often multicomponent -> a 'volume-only' baseline and a size-robust set (no
    dc_dbic) are reported.
  - the size-robust set is also added to HU-shape to test any *incremental* value.

Run: python -m drstone.deconv_ab
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.decision_nodes import SHAPE, shap_top
from drstone.multicomponent_ab import PCOLS, _merge_all, boot, oof

DECONV = ["dc_skew", "dc_kurt", "dc_bimod", "dc_dbic", "dc_musep", "dc_wmin", "dc_sdratio"]
ROBUST = ["dc_skew", "dc_bimod", "dc_musep", "dc_wmin", "dc_sdratio"]  # no size-confounded dbic


def main():
    df = _merge_all()
    dec = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_deconv.csv"),
                      dtype={"canonical_mrn": str})
    df = df.merge(dec, on=["canonical_mrn", "stone_idx"], how="left").reset_index(drop=True)
    g = df["patient_bag"]
    second = df[PCOLS].apply(lambda r: sorted(r.values)[-2], axis=1)
    print(f"cohort {len(df)} stones; deconv measurable (dc_musep non-null): "
          f"{int(df.dc_musep.notna().sum())}")

    sets = {"volume-only": ["volume_mm3"], "deconv-all (7f)": DECONV,
            "deconv-robust (5f)": ROBUST, "HU-shape (10f)": SHAPE,
            "HU-shape+deconv": SHAPE + ROBUST}
    res = {}
    for tau in (0.05, 0.10, 0.25):
        ym = (second >= tau).astype(int)
        if not (12 < ym.sum() < len(ym) - 12):
            continue
        print(f"\n== multicomponent >= {tau:.2f} (n_mixed {int(ym.sum())}/{len(ym)}) ==")
        rows = {}
        for name, feats in sets.items():
            feats = [c for c in feats if c in df.columns]
            a, lo, hi = boot(ym.values, oof(df, feats, ym, g), g.values)
            print(f"  {name:20s} AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]"
                  f"{'  *' if lo > 0.5 else ('  (<chance)' if hi < 0.5 else '')}")
            rows[name] = {"auc": a, "ci": [lo, hi]}
        res[f"thr_{int(tau*100)}"] = rows

    ym = (second >= 0.10).astype(int)
    print("\nSHAP deconv-all (thr 0.10):",
          [(f, round(v, 3)) for f, v in shap_top(df, DECONV, ym, 7)])
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_deconv_ab.json"), "w"), indent=2)
    print(f"\nresults -> {os.path.join(C.MODEL_DIR, 'drstone_deconv_ab.json')}")


if __name__ == "__main__":
    main()
