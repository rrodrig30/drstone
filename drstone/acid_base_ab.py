"""Step 1 A/B: do derived acid-base + renal lab features add signal?

Tests an explicit normal-anion-gap (hyperchloremic) metabolic-acidosis composite,
a hyperchloremia index, and CKD-EPI eGFR — pure arithmetic on existing labs, no
re-segmentation — on (a) the CaOx-vs-CaP head (the weak spot: dRTA->CaP,
GI-loss->CaOx) and (b) the 5-class composition model. Patient-grouped repeated
CV, clustered-bootstrap CIs on the paired AUC deltas, and a SHAP usage check.

Run:  python -m drstone.acid_base_ab
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from drstone import config as C
from drstone.caox_cap_head import HU, LABS, oof_proba
from drstone.compose_model import CLASSES, FEATURES, make_model
from drstone.derived_features import DERIVED, add_derived
from drstone.gradient_ab import build_table

warnings.filterwarnings("ignore")

N_BOOT = 3000


def paired_boot(y, Pa, Pb, groups):
    from sklearn.metrics import roc_auc_score
    uniq = np.array(sorted(set(groups)))
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(0)
    d = []
    for _ in range(N_BOOT):
        gs = uniq[rng.randint(0, len(uniq), len(uniq))]
        idx = np.concatenate([gidx[g] for g in gs])
        yy = y[idx]
        if yy.sum() < 2 or yy.sum() == len(yy):
            continue
        d.append(roc_auc_score(yy, Pb[idx]) - roc_auc_score(yy, Pa[idx]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def head_ab(df):
    from sklearn.metrics import roc_auc_score
    sub = add_derived(df[df["y"].isin(["CaOx", "CaP"])].reset_index(drop=True))
    y = (sub["y"] == "CaP").astype(int)
    g = sub["patient_bag"]
    sets = {"HU+labs": HU + LABS,
            "+acid-base": HU + LABS + ["cl_minus_co2", "nagma_score"],
            "+eGFR": HU + LABS + ["egfr"],
            "+all derived": HU + LABS + DERIVED}
    print(f"== CaOx-vs-CaP head: {len(sub)} stones (CaP={int(y.sum())}, CaOx={int((1-y).sum())}) ==")
    oofs = {n: oof_proba(sub, f, y, g) for n, f in sets.items()}
    base = oofs["HU+labs"]
    print(f"{'feature set':14s} {'AUC':>6s}   delta vs HU+labs  (95% CI, P[Δ>0])")
    res = {}
    for n in sets:
        auc = roc_auc_score(y.values, oofs[n])
        if n == "HU+labs":
            print(f"{n:14s} {auc:6.3f}   (baseline)")
            res[n] = {"auc": float(auc)}
        else:
            m, lo, hi, p = paired_boot(y.values, base, oofs[n], g.values)
            print(f"{n:14s} {auc:6.3f}   {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P={p:.2f}")
            res[n] = {"auc": float(auc), "delta": m, "ci": [lo, hi], "p_gt0": p}

    import shap
    feats = HU + LABS + DERIVED
    mdl = make_model(); mdl.fit(sub[feats], y)
    sv = np.asarray(shap.TreeExplainer(mdl).shap_values(sub[feats]))
    imp = (np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))) if sv.ndim > 2 else np.abs(sv).mean(0))
    imp = np.asarray(imp).ravel()
    order = np.argsort(imp)[::-1]
    print(f"\nSHAP usage of derived features (of {len(feats)}):")
    for dfeat in DERIVED:
        rank = list(np.array(feats)[order]).index(dfeat) + 1
        print(f"  {dfeat:14s} rank {rank:2d}/{len(feats)}  mean|SHAP| {imp[feats.index(dfeat)]:.4f}")
    print("  top-6 overall:", [feats[k] for k in order[:6]])
    return res


def multiclass_ab(df):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    d = add_derived(df.reset_index(drop=True))
    y = d["y"]; g = d["patient_bag"]
    classes = [c for c in CLASSES if (y == c).sum() >= 2]

    def run(feats):
        P = np.zeros((len(d), len(classes))); cnt = np.zeros(len(d))
        for rep in range(20):
            sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
            for tr, te in sgkf.split(d[feats], y, g):
                m = make_model(); m.fit(d[feats].iloc[tr], y.iloc[tr])
                pr = m.predict_proba(d[feats].iloc[te])
                col = {c: i for i, c in enumerate(m.classes_)}
                for j, c in enumerate(classes):
                    if c in col:
                        P[te, j] += pr[:, col[c]]
                cnt[te] += 1
        P = P / np.maximum(cnt[:, None], 1); P = P / P.sum(1, keepdims=True)
        pred = np.array(classes)[P.argmax(1)]
        aucs = {c: (roc_auc_score((y == c).astype(int), P[:, j])
                    if 2 <= (y == c).sum() < len(y) else float("nan"))
                for j, c in enumerate(classes)}
        return (f1_score(y, pred, average="macro", labels=classes),
                balanced_accuracy_score(y, pred), aucs)

    print("\n== 5-class composition: baseline vs +derived ==")
    print(f"  {'set':9s} {'macroF1':>8s} {'balAcc':>7s}  {'CaOx':>5s} {'CaP':>5s} {'UA':>5s} {'Struv':>5s}")
    for tag, feats in [("baseline", FEATURES), ("+derived", FEATURES + DERIVED)]:
        f1, ba, a = run(feats)
        print(f"  {tag:9s} {f1:8.3f} {ba:7.3f}  {a['CaOx']:.3f} {a['CaP']:.3f} "
              f"{a['UA']:.3f} {a.get('Struvite', float('nan')):.3f}")


def main():
    df = build_table()
    res = head_ab(df)
    multiclass_ab(df)
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_acidbase_ab.json"), "w"), indent=2)
    print(f"\nresults -> {os.path.join(C.MODEL_DIR, 'drstone_acidbase_ab.json')}")


if __name__ == "__main__":
    main()
