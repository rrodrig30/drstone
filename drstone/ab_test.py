"""Phase 2 A/B test — does the Hb-anchored HU calibration reduce the
within-composition variance of stone HU?

Logic: stones of the same composition should read the same HU. If a calibration
removes scanner offset/gain, the spread of stone HU *within a composition group*
should shrink. We segment each single-stone patient's stone, measure its raw HU,
apply four schemes, and compare within-composition spread:

  raw          : measured HU
  air_only     : offset so air -> -1000 (per-scan offset removal)
  blood_const  : two-anchor {air -> -1000, blood -> a + b*mean(Hb)}  (offset+gain,
                 NO patient Hb — isolates scanner correction)
  hb_anchored  : two-anchor {air -> -1000, blood -> a + b*Hb_patient} (adds physiology)

Comparisons isolate each effect: air_only vs raw (offset), blood_const vs air_only
(gain), hb_anchored vs blood_const (the Hb term — the question from Phase 1).

Run (as a module — nnUNet spawns workers that re-import the entry):
    python -m drstone.ab_test --limit 30
    python -m drstone.ab_test
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.calibration import calibration_params
from drstone.stone_segmentation import segment_stone

warnings.filterwarnings("ignore")

PURE_FRACTION = 0.80        # "pure-ish" dominant component for clean groups
MIN_GROUP = 5               # minimum stones per composition group to test
HU_STAT = "mean_hu"         # primary stone-HU statistic (stable); peak also saved


def build_stone_table(limit=None) -> pd.DataFrame:
    """Segment stones for single-stone curated patients; merge anchors + labels."""
    link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"),
                       dtype={"canonical_mrn": str})
    anch = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_anchors.csv"),
                       dtype={"canonical_mrn": str})
    ss = link[(link["single_stone_patient"] == True) & (link["curated_dir"].notna())]
    ss = ss.drop_duplicates("canonical_mrn").reset_index(drop=True)
    if limit:
        # spread the sample across compositions
        ss = (ss.groupby("dominant_parent", group_keys=False)
                .apply(lambda g: g.head(max(2, limit // 5))).head(limit))
    keep = ["canonical_mrn", "curated_dir", "dominant_parent", "mass_mg",
            "kvp", "kernel", "slice_thickness", "quality_tier"] + \
           [c for c in ss.columns if c.startswith("p_")]
    ss = ss[keep]
    amerge = anch[["canonical_mrn", "air_hu", "aorta_hu", "hemoglobin"]]
    df = ss.merge(amerge, on="canonical_mrn", how="left")

    rows = []
    for i, r in df.iterrows():
        try:
            res = segment_stone(r["curated_dir"])
        except Exception as e:
            print(f"  [{i}] {r['canonical_mrn']}: seg FAILED ({e})")
            res = {"found": False}
        rec = r.to_dict()
        rec.update({k: res.get(k, np.nan) for k in
                    ["found", "peak_hu", "mean_hu", "p95_hu", "volume_mm3", "n_vox"]})
        rows.append(rec)
        if (i + 1) % 10 == 0:
            ok = sum(1 for x in rows if x.get("found"))
            print(f"  segmented {i + 1}/{len(df)} ({ok} stones found)")
    return pd.DataFrame(rows)


def apply_schemes(df: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
    """Add HU under each calibration scheme to the chosen stone-HU statistic."""
    df = df.copy()
    hb_mean = df["hemoglobin"].mean()
    raw = df[HU_STAT].values
    out_raw, out_air, out_bc, out_hb = [], [], [], []
    for _, r in df.iterrows():
        hu, air, aorta, hb = r[HU_STAT], r["air_hu"], r["aorta_hu"], r["hemoglobin"]
        out_raw.append(hu)
        out_air.append(hu + (-1000.0 - air) if pd.notna(air) else np.nan)
        if pd.notna(air) and pd.notna(aorta) and pd.notna(hb):
            al_c, be_c = calibration_params(air, aorta, hb_mean, a, b)   # constant blood target
            al_h, be_h = calibration_params(air, aorta, hb, a, b)        # Hb-specific target
            out_bc.append(al_c + be_c * hu)
            out_hb.append(al_h + be_h * hu)
        else:
            out_bc.append(np.nan); out_hb.append(np.nan)
    df["hu_raw"], df["hu_air_only"] = out_raw, out_air
    df["hu_blood_const"], df["hu_hb_anchored"] = out_bc, out_hb
    return df


def within_group_sd(df, valcol, groupcol="dominant_parent"):
    d = df.dropna(subset=[valcol])
    g = d.groupby(groupcol)[valcol]
    resid = d[valcol] - g.transform("mean")
    k = d[groupcol].nunique()
    return float(np.sqrt((resid ** 2).sum() / max(1, len(d) - k)))


def anova_f(df, valcol, groupcol="dominant_parent"):
    from scipy import stats
    groups = [x[valcol].dropna().values for _, x in df.groupby(groupcol)
              if x[valcol].notna().sum() >= 2]
    if len(groups) < 2:
        return np.nan, np.nan
    f, p = stats.f_oneway(*groups)
    return float(f), float(p)


def main():
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    fit = json.load(open(os.path.join(C.OUTPUT_DIR, "drstone_calibration_fit.json")))
    a, b = fit["a"], fit["b"]

    print("Segmenting stones for single-stone patients...")
    df = build_stone_table(limit)
    found = df[df["found"] == True].copy()
    print(f"\nstones segmented: {len(found)}/{len(df)}")

    df2 = apply_schemes(found, a, b)
    df2.to_csv(os.path.join(C.OUTPUT_DIR, "drstone_stone_hu.csv"), index=False)

    # Pure-ish stones, adequately-sized composition groups
    df2["dom_frac"] = df2.apply(lambda r: r.get(f"p_{r['dominant_parent']}", np.nan), axis=1)
    pure = df2[df2["dom_frac"] >= PURE_FRACTION].copy()
    counts = pure["dominant_parent"].value_counts()
    keep_groups = counts[counts >= MIN_GROUP].index
    pure = pure[pure["dominant_parent"].isin(keep_groups)]

    schemes = ["hu_raw", "hu_air_only", "hu_blood_const", "hu_hb_anchored"]
    print("\n================ A/B variance-reduction test ================")
    print(f"pure stones (dom>={PURE_FRACTION}) in groups n>={MIN_GROUP}: {len(pure)}  "
          f"groups: {dict(pure['dominant_parent'].value_counts())}")
    print(f"\n{'scheme':16s} {'within-comp SD':>15s} {'ANOVA F':>10s} {'p':>10s}")
    base_sd = None
    for s in schemes:
        sd = within_group_sd(pure, s)
        f, p = anova_f(pure, s)
        if base_sd is None:
            base_sd = sd
        delta = (sd - base_sd) / base_sd * 100
        print(f"{s:16s} {sd:15.1f} {f:10.2f} {p:10.1e}   ({delta:+.1f}% vs raw)")

    print("\nInterpretation:")
    print("  air_only vs raw     -> value of per-scan offset removal")
    print("  blood_const vs air  -> value of gain (beta) correction")
    print("  hb_anchored vs bc   -> value of the patient-Hb term (the Phase-1 question)")

    # figure: within-composition SD by scheme
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sds = [within_group_sd(pure, s) for s in schemes]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar([s.replace("hu_", "") for s in schemes], sds, color="#3a6ea5")
        ax.set_ylabel("Within-composition stone-HU SD (lower = better)")
        ax.set_title("Calibration A/B: same-composition stone-HU spread")
        for i, v in enumerate(sds):
            ax.text(i, v, f"{v:.0f}", ha="center", va="bottom")
        fig.tight_layout()
        fp = os.path.join(C.OUTPUT_DIR, "drstone_ab_variance.png")
        fig.savefig(fp, dpi=200); print(f"\nfigure -> {fp}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"per-stone table -> {os.path.join(C.OUTPUT_DIR, 'drstone_stone_hu.csv')}")


if __name__ == "__main__":
    main()
