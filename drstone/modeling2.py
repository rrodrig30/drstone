"""Expanded composition modeling + feature-group ablation (Dr Stone).

Lifted-N, multi-stone cohort (patient-level CV) with a pre-specified, mechanistic
feature set and a GROUP ablation that quantifies what each family adds:

  hu_summary    peak/mean/p95 HU                      (attenuation - the known signal)
  hu_shape      skew/kurtosis/#modes/normality/IQR/   (DISTRIBUTION shape -> mixtures,
                range/entropy/core-vs-rim              calcium subtypes)
  shape         volume/surface/sphericity/elongation  (morphology -> struvite, etc.)
  labs          BMP + urine pH + anion gap            (acid-base: dRTA->brushite;
                                                        low urine pH->UA)
  demographics  age, sex, sex x pH, age x HU          (priors + interactions)

Pre-registered hypotheses: anion-gap/dRTA pattern (normal-AG acidosis + alkaline
urine) -> calcium phosphate/brushite; low urine pH (+ high glucose) -> uric acid;
sex x pH interaction -> UA in men with acid urine.

Patient-level StratifiedGroupKFold (multiple stones per patient). Clean-label
stones (rank_mass / single_comp) for the primary; SHAP on the full model.

Run:  python -m drstone.modeling2
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C

warnings.filterwarnings("ignore")

GROUPS = {
    "hu_summary": ["hu_peak", "hu_mean", "hu_p95"],
    "hu_shape": ["hu_std", "hu_iqr", "hu_range", "hu_skew", "hu_kurtosis",
                 "hu_shapiro", "hu_entropy", "hu_nmodes", "hu_core_minus_rim"],
    "shape": ["volume_mm3", "surface_mm2", "sphericity", "elongation"],
    "labs": ["urine_ph", "na", "k", "cl", "co2", "bun", "creatinine", "ca",
             "glucose", "anion_gap"],
    "demographics": ["age", "gender_M", "sex_x_ph", "age_x_hu"],
}
MONO = {"hu_peak": -1, "hu_mean": -1, "hu_p95": -1, "urine_ph": -1}   # P(UA): HU,pH down


def build_table(clean_only=True):
    ms = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv"),
                     dtype={"canonical_mrn": str, "patient_bag": str})
    stones = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"),
                         dtype={"canonical_mrn": str})
    labs = stones[["canonical_mrn", "urine_ph", "na", "k", "cl", "co2", "bun",
                   "creatinine", "ca", "glucose", "age", "gender"]].drop_duplicates("canonical_mrn")
    df = ms.merge(labs, on="canonical_mrn", how="left")
    # Metabolic derivations + interactions
    df["anion_gap"] = df["na"] - df["cl"] - df["co2"]
    df["gender_M"] = (df["gender"] == "M").astype(float)
    df["sex_x_ph"] = df["gender_M"] * df["urine_ph"]
    df["age_x_hu"] = df["age"] * df["hu_peak"] / 1000.0
    if clean_only:
        df = df[df["match_quality"].isin(["rank_mass", "single_comp"])].copy()
    return df


def make_model(features):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mono = [MONO.get(f, 0) for f in features]
    return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05,
                                          monotonic_cst=mono, l2_regularization=1.0,
                                          random_state=0)


def group_cv_auc(df, feats, y, groups, n_splits=5):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in sgkf.split(df[feats], y, groups):
        m = make_model(feats); m.fit(df[feats].iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(df[feats].iloc[te])[:, 1]
    return roc_auc_score(y, oof)


def main():
    df = build_table(clean_only=True)
    y = df["y_ua"]; groups = df["patient_bag"]
    print(f"clean-label stones: {len(df)} from {groups.nunique()} patients, UA={int(y.sum())}")

    print("\n==== single-group AUC (what each family alone delivers) ====")
    for g, feats in GROUPS.items():
        feats = [f for f in feats if f in df.columns]
        print(f"  {g:14s} {group_cv_auc(df, feats, y, groups):.3f}")

    print("\n==== cumulative ablation (add groups in order) ====")
    order = ["hu_summary", "hu_shape", "shape", "labs", "demographics"]
    cum = []
    prev = None
    for g in order:
        cum += [f for f in GROUPS[g] if f in df.columns]
        auc = group_cv_auc(df, cum, y, groups)
        delta = "" if prev is None else f"  ({auc - prev:+.3f})"
        print(f"  + {g:14s} ({len(cum):2d} feats)  AUC={auc:.3f}{delta}")
        prev = auc
    full = cum

    # SHAP on full model
    try:
        import shap, matplotlib
        matplotlib.use("Agg"); import matplotlib.pyplot as plt
        m = make_model(full); m.fit(df[full], y)
        try:
            sv = shap.TreeExplainer(m).shap_values(df[full])
        except Exception:
            sv = shap.Explainer(lambda d: m.predict_proba(d)[:, 1],
                                shap.maskers.Independent(df[full], 100))(df[full]).values
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        imp = pd.DataFrame({"feature": full, "mean_abs_shap": np.abs(sv).mean(0)}) \
            .sort_values("mean_abs_shap", ascending=False)
        imp.to_csv(os.path.join(C.OUTPUT_DIR, "drstone_shap_importance_v2.csv"), index=False)
        print("\nTop SHAP features (UA, expanded model):")
        print(imp.head(12).to_string(index=False))
        fig, ax = plt.subplots(figsize=(6, 6))
        top = imp.head(15)
        ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#3a6ea5")
        ax.set_xlabel("mean |SHAP|"); ax.set_title("UA (expanded): SHAP feature importance")
        fig.tight_layout(); fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_shap_importance_v2.png"), dpi=200)
    except Exception as e:
        print(f"(SHAP skipped: {e})")


if __name__ == "__main__":
    main()
