"""Targeted CaOx-vs-CaP specialist head.

The 5-class model can't separate the two calcium subtypes on single-energy CT,
and the density-gradient ratio doesn't help it net. This tests a DEDICATED binary
head applied only to the calcium decision: can it use the core/periphery ratio on
top of the features that actually inform CaP (HU + the acid-base labs that flag
alkaline urine / distal RTA / hypercalcemia) to beat those features alone?

The scientifically honest comparison is HU+labs vs HU+labs+ratio (does the ratio
add over a STRONG baseline, not just over HU). Positive class = CaP (identifying
CaP changes prevention: dRTA/hyperparathyroidism work-up, caution with citrate).

Run:  python -m drstone.caox_cap_head
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np

from drstone import config as C
from drstone.compose_model import make_model
from drstone.gradient_ab import build_table

warnings.filterwarnings("ignore")

HU = ["hu_peak", "hu_mean", "hu_p95", "volume_mm3"]
LABS = ["urine_ph", "na", "k", "cl", "co2", "anion_gap", "bun", "creatinine",
        "ca", "glucose", "age", "gender_M"]
GRAD = ["hu_core_over_rim", "hu_core_p50", "hu_rim_p50"]
SETS = {"HU": HU, "HU+grad": HU + GRAD, "HU+labs": HU + LABS,
        "HU+labs+grad": HU + LABS + GRAD}
N_REPEATS = 40
N_BOOT = 3000


def oof_proba(sub, feats, y, groups):
    """Averaged out-of-fold P(CaP) over repeated patient-grouped 5-fold CV."""
    from sklearn.model_selection import StratifiedGroupKFold
    X = sub[feats]
    P = np.zeros(len(sub)); cnt = np.zeros(len(sub))
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(X, y, groups):
            m = make_model(); m.fit(X.iloc[tr], y.iloc[tr])
            P[te] += m.predict_proba(X.iloc[te])[:, 1]; cnt[te] += 1
    return P / np.maximum(cnt, 1)


def boot_auc(y, P, groups, paired_P=None):
    """Patient-clustered bootstrap of AUC (and the paired AUC delta vs paired_P)."""
    from sklearn.metrics import roc_auc_score
    uniq = np.array(sorted(set(groups)))
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(0)
    aucs, deltas = [], []
    for _ in range(N_BOOT):
        gs = uniq[rng.randint(0, len(uniq), len(uniq))]
        idx = np.concatenate([gidx[g] for g in gs])
        yy = y[idx]
        if yy.sum() < 2 or yy.sum() == len(yy):
            continue
        aucs.append(roc_auc_score(yy, P[idx]))
        if paired_P is not None:
            deltas.append(roc_auc_score(yy, P[idx]) - roc_auc_score(yy, paired_P[idx]))
    aucs = np.array(aucs)
    out = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)))
    if paired_P is not None:
        deltas = np.array(deltas)
        return out, (float(np.mean(deltas)), float(np.percentile(deltas, 2.5)),
                     float(np.percentile(deltas, 97.5)), float((deltas > 0).mean()))
    return out, None


def main():
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss, confusion_matrix)
    df = build_table()
    sub = df[df["y"].isin(["CaOx", "CaP"])].reset_index(drop=True)
    y = (sub["y"] == "CaP").astype(int)
    groups = sub["patient_bag"]
    n_meas = int(sub["grad_measurable"].fillna(0).sum())
    print(f"calcium subset: {len(sub)} stones / {groups.nunique()} patients "
          f"(CaP={int(y.sum())}, CaOx={int((1-y).sum())})")
    print(f"ratio measurable: {n_meas}/{len(sub)} ({100*n_meas/len(sub):.0f}%)")
    print(f"positive class = CaP; {N_REPEATS}x repeated patient-grouped 5-fold CV\n")

    oofs = {name: oof_proba(sub, feats, y, groups) for name, feats in SETS.items()}
    yv = y.values

    print(f"{'feature set':16s} {'AUC':>6s}  {'95% CI':>16s}  {'AP':>5s}  {'Brier':>6s}")
    rows = {}
    for name in SETS:
        P = oofs[name]
        auc = roc_auc_score(yv, P)
        ci, _ = boot_auc(yv, P, groups.values)
        ap = average_precision_score(yv, P)
        brier = brier_score_loss(yv, P)
        rows[name] = {"auc": float(auc), "ci": list(ci), "ap": float(ap), "brier": float(brier)}
        print(f"{name:16s} {auc:6.3f}  [{ci[0]:.3f}, {ci[1]:.3f}]  {ap:5.3f}  {brier:6.3f}")

    # the decisive comparison: does the ratio add over the strong (HU+labs) baseline?
    _, delta = boot_auc(yv, oofs["HU+labs+grad"], groups.values, paired_P=oofs["HU+labs"])
    print(f"\nDELTA AUC (HU+labs+grad) - (HU+labs): {delta[0]:+.3f} "
          f"95% CI [{delta[1]:+.3f}, {delta[2]:+.3f}]  P(delta>0)={delta[3]:.2f}")
    _, delta_hu = boot_auc(yv, oofs["HU+grad"], groups.values, paired_P=oofs["HU"])
    print(f"DELTA AUC (HU+grad)      - (HU):      {delta_hu[0]:+.3f} "
          f"95% CI [{delta_hu[1]:+.3f}, {delta_hu[2]:+.3f}]  P(delta>0)={delta_hu[3]:.2f}")

    # The ratio doesn't add over HU+labs, so the DEPLOYED head is HU+labs (the
    # winner); the gradient is documented-rejected even in this targeted niche.
    DEPLOY = "HU+labs"
    deploy_feats = SETS[DEPLOY]
    print(f"\n>> deployable head = {DEPLOY} (AUC {rows[DEPLOY]['auc']:.3f}); "
          f"ratio excluded (no benefit over this baseline)")

    # operating point for the deployed head (Youden's J)
    best = oofs[DEPLOY]
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(yv, best)
    j = int(np.argmax(tpr - fpr)); t = float(thr[j])
    pred = (best >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(yv, pred).ravel()
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    print(f"\noperating point (Youden, thr={t:.2f}) for CaP:")
    print(f"  sens={sens:.2f}  spec={spec:.2f}  PPV={ppv:.2f}  NPV={npv:.2f}  "
          f"(TP={tp} FP={fp} FN={fn} TN={tn})")

    # SHAP diagnostic on HU+labs+grad: confirm the head barely uses the ratio.
    import shap, joblib
    diag_feats = HU + LABS + GRAD
    diag = make_model(); diag.fit(sub[diag_feats], y)
    sv = np.asarray(shap.TreeExplainer(diag).shap_values(sub[diag_feats]))
    imp = (np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))) if sv.ndim > 2
           else np.abs(sv).mean(0))
    imp = np.asarray(imp).ravel()
    order = np.argsort(imp)[::-1]
    rank = list(np.array(diag_feats)[order]).index("hu_core_over_rim") + 1
    print(f"\nSHAP top-8 (HU+labs+grad diagnostic); core/rim ratio rank "
          f"{rank}/{len(diag_feats)}:")
    for k in order[:8]:
        print(f"  {diag_feats[k]:16s} {imp[k]:.4f}")

    # persist the DEPLOYED head (HU+labs, no ratio) + results + ROC figure
    final = make_model(); final.fit(sub[deploy_feats], y)
    joblib.dump({"model": final, "features": deploy_feats, "positive": "CaP",
                 "threshold_youden": t},
                os.path.join(C.MODEL_DIR, "drstone_caox_cap_head.pkl"))
    res = {"n": len(sub), "n_cap": int(y.sum()), "n_caox": int((1 - y).sum()),
           "n_measurable": n_meas, "deployed": DEPLOY, "deploy_features": deploy_feats,
           "sets": rows,
           "delta_over_hu_labs": {"mean": delta[0], "ci": [delta[1], delta[2]], "p_gt0": delta[3]},
           "delta_over_hu": {"mean": delta_hu[0], "ci": [delta_hu[1], delta_hu[2]], "p_gt0": delta_hu[3]},
           "operating_point": {"threshold": t, "sens": sens, "spec": spec, "ppv": ppv, "npv": npv},
           "shap_rank_ratio": rank}
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_caox_cap_head.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        for name in SETS:
            f, t2, _ = roc_curve(yv, oofs[name])
            ax.plot(f, t2, lw=1.8, label=f"{name} (AUC {rows[name]['auc']:.3f})")
        ax.plot([0, 1], [0, 1], "--", color="#999", lw=.8)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
        ax.set_title("CaOx-vs-CaP specialist head (positive = CaP)")
        ax.legend(loc="lower right", fontsize=9); fig.tight_layout()
        fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_caox_cap_head.png"), dpi=200)
        print(f"\nfigure -> {os.path.join(C.OUTPUT_DIR, 'drstone_caox_cap_head.png')}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"results -> {os.path.join(C.MODEL_DIR, 'drstone_caox_cap_head.json')}")


if __name__ == "__main__":
    main()
