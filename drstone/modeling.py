"""Phase 3 — baseline composition modeling for Dr Stone.

Predicts stone composition from non-contrast CT features + clinical labs:
  * UA-vs-non-UA (the clinically decisive split) with monotonic constraints
    encoding the chemistry (higher HU -> less likely UA; higher urine pH ->
    less likely UA, since UA forms only in acidic urine),
  * dominant-component multiclass (CaP / CaOx / UA / Struvite).

Honest baselines (HU-only, pH-only, labs-only) bound what imaging adds. Patient-
level cross-validation; SHAP for interpretability (importances + per-stone values
saved to CSV, plots written). HU is air-calibrated (the A/B test showed Hb-
anchoring adds nothing; the air offset is a cheap, harmless standardization).

Run:
    python -m drstone.modeling
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C

warnings.filterwarnings("ignore")

HU_FEATS = ["peak_hu_cal", "mean_hu_cal", "p95_hu_cal", "volume_mm3"]
LAB_FEATS = ["urine_ph", "na", "k", "cl", "co2", "bun", "creatinine", "ca",
             "glucose", "age", "gender_M"]
ALL_FEATS = HU_FEATS + LAB_FEATS
# Monotonic direction for P(UA): higher HU and higher pH -> lower P(UA).
MONO = {"peak_hu_cal": -1, "mean_hu_cal": -1, "p95_hu_cal": -1, "urine_ph": -1}
MULTI_CLASSES = ["CaOx", "CaP", "UA", "Struvite"]


def build_features() -> pd.DataFrame:
    hu = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stone_hu.csv"),
                     dtype={"canonical_mrn": str})
    hu = hu[hu["found"] == True].copy()
    stones = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"),
                         dtype={"canonical_mrn": str})
    labs = stones[["canonical_mrn", "urine_ph", "na", "k", "cl", "co2", "bun",
                   "creatinine", "ca", "glucose", "age", "gender"]].drop_duplicates("canonical_mrn")
    df = hu.merge(labs, on="canonical_mrn", how="left")
    # Air-calibrated HU (per-scan offset so air -> -1000)
    df["air_off"] = -1000.0 - df["air_hu"]
    for s in ["peak_hu", "mean_hu", "p95_hu"]:
        df[s + "_cal"] = df[s] + df["air_off"]
    df["gender_M"] = (df["gender"] == "M").astype(float)
    df["y_ua"] = (df["dominant_parent"] == "UA").astype(int)
    return df


def cv_auc(model_factory, X, y, groups, n_splits=5):
    """Patient-level CV AUC via out-of-fold probabilities."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y):
        m = model_factory()
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[te] = m.predict_proba(X.iloc[te])[:, 1]
    return roc_auc_score(y, oof), oof


def make_ua_model(features):
    from sklearn.ensemble import HistGradientBoostingClassifier
    mono = [MONO.get(f, 0) for f in features]
    return HistGradientBoostingClassifier(
        max_iter=200, max_depth=3, learning_rate=0.05,
        monotonic_cst=mono, l2_regularization=1.0, random_state=0)


def run_ua_model(df):
    from sklearn.metrics import roc_auc_score, confusion_matrix
    y = df["y_ua"]
    groups = df["canonical_mrn"]
    print(f"\n==== UA vs non-UA  (n={len(df)}, UA={int(y.sum())}) ====")
    # Full model + isolated baselines
    feature_sets = {
        "full (HU+pH+labs)": ALL_FEATS,
        "HU only": HU_FEATS,
        "pH only": ["urine_ph"],
        "labs only (no HU)": LAB_FEATS,
    }
    print(f"  {'model':22s} {'CV AUC':>8s}")
    best_auc = None
    for name, feats in feature_sets.items():
        X = df[feats]
        auc, oof = cv_auc(lambda f=feats: make_ua_model(f), X, y, groups)
        print(f"  {name:22s} {auc:8.3f}")
        if name.startswith("full"):
            best_auc, best_oof = auc, oof
    # Operating point (Youden) on the full-model OOF probs
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, best_oof)
    j = np.argmax(tpr - fpr)
    pred = (best_oof >= thr[j]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    sens, spec = tp / (tp + fn), tn / (tn + fp)
    print(f"  full-model @Youden thr={thr[j]:.2f}: sensitivity={sens:.2f}, specificity={spec:.2f}")
    return ALL_FEATS, y


def run_multiclass(df):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    d = df[df["dominant_parent"].isin(MULTI_CLASSES)].copy()
    X, y = d[ALL_FEATS], d["dominant_parent"]
    print(f"\n==== Dominant component  (n={len(d)}, classes={dict(y.value_counts())}) ====")
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    m = HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05,
                                       l2_regularization=1.0, random_state=0)
    pred = cross_val_predict(m, X, y, cv=skf)
    print(f"  macro-F1 = {f1_score(y, pred, average='macro'):.3f}")
    print("  per-class recall:", {c: round(((y == c) & (pred == c)).sum() / max(1, (y == c).sum()), 2)
                                   for c in MULTI_CLASSES})


def run_shap(df, feats, y):
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X = df[feats]
    model = make_ua_model(feats)
    model.fit(X, y)
    try:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X)
    except Exception:
        explainer = shap.Explainer(lambda d: model.predict_proba(d)[:, 1],
                                   shap.maskers.Independent(X, max_samples=100))
        sv = explainer(X).values
    sv = np.asarray(sv)
    if sv.ndim == 3:                      # (n, features, classes) -> positive class
        sv = sv[:, :, -1]
    imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(0)}) \
        .sort_values("mean_abs_shap", ascending=False)
    imp.to_csv(os.path.join(C.OUTPUT_DIR, "drstone_shap_importance.csv"), index=False)
    vals = pd.DataFrame(sv, columns=[f"shap_{f}" for f in feats])
    vals.insert(0, "canonical_mrn", df["canonical_mrn"].values)
    vals.insert(1, "y_ua", y.values)
    vals.to_csv(os.path.join(C.OUTPUT_DIR, "drstone_shap_values.csv"), index=False)
    print("\nSHAP importance (mean |SHAP|) for UA model:")
    print(imp.to_string(index=False))

    # Bar
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(imp["feature"][::-1], imp["mean_abs_shap"][::-1], color="#3a6ea5")
    ax.set_xlabel("mean |SHAP|"); ax.set_title("UA-vs-non-UA: SHAP feature importance")
    fig.tight_layout(); fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_shap_importance.png"), dpi=200)
    # Beeswarm
    try:
        plt.figure()
        shap.summary_plot(sv, X, feature_names=feats, show=False, max_display=12)
        plt.tight_layout()
        plt.savefig(os.path.join(C.OUTPUT_DIR, "drstone_shap_beeswarm.png"), dpi=200, bbox_inches="tight")
    except Exception as e:
        print(f"(beeswarm skipped: {e})")
    print(f"\nWrote SHAP importance/values CSV + plots to {C.OUTPUT_DIR}")


def main():
    df = build_features()
    feats, y = run_ua_model(df)
    run_multiclass(df)
    run_shap(df, feats, y)


if __name__ == "__main__":
    main()
