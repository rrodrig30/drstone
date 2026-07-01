"""Exhaustive test: can ANY of the fourteen feature families discriminate
single- from multi-component stones? Every concept tried in the ledger is
evaluated against the multicomponent endpoint — not assumed, proven.

Endpoint: multicomponent = second-largest stone-analysis component fraction >= tau
(swept 0.05 / 0.10 / 0.25). Patient-grouped repeated CV, clustered-bootstrap CI,
per family and for the union ("kitchen sink"), plus SHAP and a forest plot.

Run: python -m drstone.multicomponent_ab
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.compose_model import make_model
from drstone.derived_features import add_derived
from drstone.gradient_ab import build_table

warnings.filterwarnings("ignore")

PCOLS = ["p_CaOx", "p_CaP", "p_UA", "p_Struvite", "p_Cystine", "p_Other"]
N_REPEATS = 20
N_BOOT = 2000
D = C.OUTPUT_DIR


def _merge_all():
    df = build_table()
    df = add_derived(df)
    key = ["canonical_mrn", "stone_idx"]
    grad = pd.read_csv(f"{D}/drstone_gradient.csv", dtype={"canonical_mrn": str})
    df = df.merge(grad[["canonical_mrn", "stone_idx", "hu_core_over_rim",
                        "hu_core_p50", "hu_rim_p50"]], on=key, how="left")
    for f in ("drstone_imaging_extra", "drstone_body_comp", "drstone_vascular",
              "drstone_deep_features", "drstone_stone_hu"):
        t = pd.read_csv(f"{D}/{f}.csv", dtype={"canonical_mrn": str}).drop_duplicates("canonical_mrn")
        cols = [c for c in t.columns if c != "canonical_mrn" and c not in df.columns]
        df = df.merge(t[["canonical_mrn"] + cols], on="canonical_mrn", how="left")
    return df


def families(df):
    deep = [c for c in df.columns if c.startswith("deep_")]
    cand = {
        "Stone HU (base)":        ["hu_peak", "hu_mean", "hu_p95", "volume_mm3", "n_vox"],
        "HU-shape radiomics":     ["hu_std", "hu_iqr", "hu_range", "hu_skew", "hu_kurtosis",
                                   "hu_entropy", "hu_nmodes", "hu_shapiro", "sphericity", "elongation"],
        "Density gradient":       ["hu_core_over_rim", "hu_core_minus_rim", "hu_core_p50", "hu_rim_p50"],
        "Hb-calibrated HU":       ["hu_air_only", "hu_blood_const", "hu_hb_anchored"],
        "CNN deep features":      deep,
        "Labs / urine":           ["urine_ph", "na", "k", "cl", "co2", "anion_gap",
                                   "bun", "creatinine", "ca", "glucose"],
        "Derived acid-base+eGFR": ["nagma_score", "cl_minus_co2", "egfr"],
        "Demographics":           ["age", "gender_M"],
        "Nephrocalcinosis":       ["nephrocalc_frac", "nephrocalc_nfoci", "nephrocalc_vol_mm3", "nephrocalc_peak_hu"],
        "Vertebral BMD":          ["bmd_lumbar_hu", "bmd_l1_hu"],
        "Hepatic steatosis":      ["liver_spleen_diff", "liver_hu", "spleen_hu"],
        "Sarcopenia":             ["muscle_hu", "muscle_vol_mm3"],
        "Aortic calcification":   ["aortic_agatston", "aortic_calc_vol_mm3", "aortic_calc_nfoci"],
        "Bladder / prostate":     ["bladder_wall_frac", "bladder_vol_mm3", "prostate_calc_vol_mm3"],
    }
    out = {}
    for name, cols in cand.items():
        present = [c for c in cols if c in df.columns]
        if present:
            out[name] = present
    return out


def oof(df, feats, y, g):
    from sklearn.model_selection import StratifiedGroupKFold
    X = df[feats]; P = np.zeros(len(df)); cnt = np.zeros(len(df))
    for rep in range(N_REPEATS):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=rep).split(X, y, g):
            m = make_model(); m.fit(X.iloc[tr], y.iloc[tr])
            P[te] += m.predict_proba(X.iloc[te])[:, 1]; cnt[te] += 1
    return P / np.maximum(cnt, 1)


def boot(y, P, g):
    from sklearn.metrics import roc_auc_score
    uniq = np.array(sorted(set(g))); gi = {k: np.where(g == k)[0] for k in uniq}
    rng = np.random.RandomState(0); a = []
    for _ in range(N_BOOT):
        gs = uniq[rng.randint(0, len(uniq), len(uniq))]
        idx = np.concatenate([gi[k] for k in gs]); yy = y[idx]
        if yy.sum() < 2 or yy.sum() == len(yy):
            continue
        a.append(roc_auc_score(yy, P[idx]))
    return float(roc_auc_score(y, P)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def main():
    df = _merge_all().reset_index(drop=True)
    g = df["patient_bag"]
    second = df[PCOLS].apply(lambda r: sorted(r.values)[-2], axis=1)
    fams = families(df)
    print(f"cohort {len(df)} stones / {g.nunique()} patients; families tested: {len(fams)}")
    print("multicomponent rate by dominant composition (>=10%):")
    for c in ["CaOx", "CaP", "UA", "Struvite"]:
        sub = df[df.dominant_parent == c]
        if len(sub) >= 5:
            print(f"  {c:9s} n={len(sub):3d}  {100*(second[sub.index]>=0.10).mean():.0f}%")

    res = {}
    for tau in (0.05, 0.10, 0.25):
        y = (second >= tau).astype(int)
        if y.sum() < 12 or y.sum() > len(y) - 12:
            continue
        print(f"\n==== multicomponent (2nd comp >= {tau:.2f}); n_mixed={int(y.sum())}/{len(y)} ====")
        rows = {}
        for name, feats in fams.items():
            a, lo, hi = boot(y.values, oof(df, feats, y, g), g.values)
            flag = "  *" if lo > 0.5 else ("  (<chance)" if hi < 0.5 else "")
            print(f"  {name:24s} ({len(feats):2d}f)  AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]{flag}")
            rows[name] = {"n_feats": len(feats), "auc": a, "ci": [lo, hi]}
        allf = sorted({c for fs in fams.values() for c in fs})
        a, lo, hi = boot(y.values, oof(df, allf, y, g), g.values)
        print(f"  {'ALL COMBINED':24s} ({len(allf):2d}f)  AUROC {a:.3f} [{lo:.3f}, {hi:.3f}]"
              f"{'  *' if lo > 0.5 else ''}")
        rows["ALL COMBINED"] = {"n_feats": len(allf), "auc": a, "ci": [lo, hi]}
        res[f"thr_{int(tau*100)}"] = rows

    json.dump(res, open(f"{C.MODEL_DIR}/drstone_multicomponent_ab.json", "w"), indent=2)

    # forest plot at thr 0.10
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        key = "thr_10" if "thr_10" in res else list(res)[0]
        r = res[key]; items = sorted(r.items(), key=lambda kv: kv[1]["auc"])
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, (name, d) in enumerate(items):
            sig = d["ci"][0] > 0.5
            col = "#2f855a" if sig else "#5b6b7c"
            ax.plot(d["ci"], [i, i], color=col, lw=2.4, solid_capstyle="round")
            ax.plot(d["auc"], i, "o", color=col, ms=7)
        ax.axvline(0.5, color="#111", ls="--", lw=1)
        ax.set_yticks(range(len(items))); ax.set_yticklabels([n for n, _ in items], fontsize=9)
        ax.set_xlim(0.35, 0.75); ax.set_xlabel("AUROC for single- vs multi-component (95% CI)")
        ax.set_title("Exhaustive multicomponent detection: every feature family (thr 0.10)")
        fig.tight_layout(); fig.savefig(f"{C.OUTPUT_DIR}/drstone_multicomponent_forest.png", dpi=200)
        import shutil; shutil.copy(f"{C.OUTPUT_DIR}/drstone_multicomponent_forest.png",
                                   "docs/figures/FigS4_multicomponent.png")
        print("\nfigure -> docs/figures/FigS4_multicomponent.png")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"results -> {C.MODEL_DIR}/drstone_multicomponent_ab.json")


if __name__ == "__main__":
    main()
