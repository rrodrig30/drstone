"""Calibrated multiclass stone-composition model.

Predicts a probability distribution over {CaOx, CaP, UA, Struvite, Other} from a
non-contrast CT (stone HU) + routine metabolic labs + demographics — at
presentation, before any stone analysis. Honest about its limits: single-energy
CT cannot reliably separate calcium-oxalate from calcium-phosphate, so the
distribution is genuinely probabilistic (and the downstream management buckets
are chosen to align with the distinctions the model CAN make).

Run:  python -m drstone.compose_model
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C

warnings.filterwarnings("ignore")

CLASSES = ["CaOx", "CaP", "UA", "Struvite", "Other"]
FEATURES = ["hu_peak", "hu_mean", "hu_p95", "volume_mm3", "urine_ph", "na", "k",
            "cl", "co2", "anion_gap", "bun", "creatinine", "ca", "glucose",
            "age", "gender_M"]


def _map_class(dom):
    return dom if dom in ("CaOx", "CaP", "UA", "Struvite") else "Other"


def build_table():
    ms = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv"),
                     dtype={"canonical_mrn": str, "patient_bag": str})
    ms = ms[ms["match_quality"].isin(["rank_mass", "single_comp"])].copy()
    stones = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"),
                         dtype={"canonical_mrn": str})
    labs = stones[["canonical_mrn", "urine_ph", "na", "k", "cl", "co2", "bun",
                   "creatinine", "ca", "glucose", "age", "gender"]].drop_duplicates("canonical_mrn")
    df = ms.merge(labs, on="canonical_mrn", how="left")
    df["anion_gap"] = df["na"] - df["cl"] - df["co2"]
    df["gender_M"] = (df["gender"] == "M").astype(float)
    df["y"] = df["dominant_parent"].map(_map_class)
    return df


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=300, max_depth=3, learning_rate=0.05,
                                          l2_regularization=1.0, random_state=0)


def main():
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                                 confusion_matrix)
    df = build_table()
    X, y, groups = df[FEATURES], df["y"], df["patient_bag"]
    counts = {c: int((y == c).sum()) for c in CLASSES}
    print(f"cohort: {len(df)} stones / {groups.nunique()} patients")
    print(f"class counts: {counts}")

    # Repeated grouped CV -> averaged OOF probabilities
    classes = [c for c in CLASSES if counts[c] >= 2]
    P = np.zeros((len(df), len(classes))); cnt = np.zeros(len(df))
    for rep in range(20):
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(X, y, groups):
            m = make_model(); m.fit(X.iloc[tr], y.iloc[tr])
            proba = m.predict_proba(X.iloc[te])
            col = {c: i for i, c in enumerate(m.classes_)}
            for j, c in enumerate(classes):
                if c in col:
                    P[te, j] += proba[:, col[c]]
            cnt[te] += 1
    P = P / np.maximum(cnt[:, None], 1)
    P = P / P.sum(1, keepdims=True)
    pred = np.array(classes)[P.argmax(1)]

    print("\n==== multiclass performance (patient-level CV) ====")
    print(f"macro-F1 = {f1_score(y, pred, average='macro', labels=classes):.3f} | "
          f"balanced acc = {balanced_accuracy_score(y, pred):.3f}")
    print("per-class one-vs-rest AUC and recall:")
    aucs = {}
    for j, c in enumerate(classes):
        yc = (y == c).astype(int)
        auc = roc_auc_score(yc, P[:, j]) if yc.sum() >= 2 and yc.sum() < len(yc) else float("nan")
        rec = ((pred == c) & (y == c)).sum() / max(1, (y == c).sum())
        aucs[c] = float(auc)
        print(f"  {c:9s} n={counts[c]:3d}  AUC={auc:.2f}  recall={rec:.2f}")
    cm = confusion_matrix(y, pred, labels=classes)
    print("\nconfusion (rows=true, cols=pred):", classes)
    for i, c in enumerate(classes):
        print(f"  {c:9s} {cm[i].tolist()}")

    # Fit + persist final model on all data
    import joblib
    final = make_model(); final.fit(X, y)
    joblib.dump({"model": final, "features": FEATURES, "classes": list(final.classes_)},
                os.path.join(C.MODEL_DIR, "drstone_compose_model.pkl"))
    json.dump({"classes": classes, "counts": counts, "ovr_auc": aucs,
               "macro_f1": float(f1_score(y, pred, average="macro", labels=classes)),
               "balanced_acc": float(balanced_accuracy_score(y, pred)), "n": len(df)},
              open(os.path.join(C.MODEL_DIR, "drstone_compose_results.json"), "w"), indent=2)

    # Confusion figure (row-normalized)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        cmn = cm / cm.sum(1, keepdims=True).clip(1)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title("Composition confusion (row-normalized)")
        for i in range(len(classes)):
            for k in range(len(classes)):
                ax.text(k, i, f"{cmn[i,k]:.2f}", ha="center", va="center",
                        color="white" if cmn[i, k] > 0.5 else "black", fontsize=9)
        fig.tight_layout(); fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_compose_confusion.png"), dpi=200)
        print(f"\nconfusion figure -> {os.path.join(C.OUTPUT_DIR, 'drstone_compose_confusion.png')}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"compose model saved -> {C.MODEL_DIR}")


if __name__ == "__main__":
    main()
