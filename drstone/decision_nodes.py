"""Decision-node analysis: reorganize stone-composition prediction around the
management decisions that actually diverge, not a composition taxonomy.

  Node A  Dissolvable?        uric acid vs non-UA (medical dissolution vs
                              intervention) -- the one acute decision that changes.
  Node B  Infection (struvite) -- a clinical diagnosis (alkaline turbid malodorous
                              urine, recurrent UTI, sepsis); NOT a modeling target.
  Node C1 Multicomponent?     single- vs multi-component stone from HU architecture.
                              A dual stone laminates (nidus + a distinct shell), so
                              even without resolving composition the two phases leave
                              a footprint (radial core-rim gradient, broadened/
                              multimodal HU). Threshold sweep on the 2nd-component
                              fraction: a cleaner two-phase stone should be more
                              detectable if the lamination hypothesis holds.
  Node C2 Predominant CaP?    CaP rule-in among calcium stones -> metabolic flag
                              (alkaline urine / distal RTA / hyperparathyroidism).

Patient-grouped repeated CV, clustered-bootstrap CIs, SHAP. All on existing
feature tables (no re-segmentation).

Run: python -m drstone.decision_nodes
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.caox_cap_head import HU, LABS, oof_proba
from drstone.compose_model import make_model
from drstone.gradient_ab import build_table

warnings.filterwarnings("ignore")

# HU architecture / shape features (imaging-only; the lamination footprint)
SHAPE = ["hu_std", "hu_iqr", "hu_range", "hu_skew", "hu_kurtosis", "hu_entropy",
         "hu_nmodes", "hu_shapiro", "hu_core_minus_rim", "sphericity", "elongation",
         "hu_peak", "hu_mean", "hu_p95", "volume_mm3", "n_vox"]
PCOLS = ["p_CaOx", "p_CaP", "p_UA", "p_Struvite", "p_Cystine", "p_Other"]
N_BOOT = 3000


def boot_auc(y, P, groups):
    from sklearn.metrics import roc_auc_score
    uniq = np.array(sorted(set(groups)))
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(0); a = []
    for _ in range(N_BOOT):
        gs = uniq[rng.randint(0, len(uniq), len(uniq))]
        idx = np.concatenate([gidx[g] for g in gs]); yy = y[idx]
        if yy.sum() < 2 or yy.sum() == len(yy):
            continue
        a.append(roc_auc_score(yy, P[idx]))
    a = np.array(a)
    from sklearn.metrics import roc_auc_score as _r
    return float(_r(y, P)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def shap_top(df, feats, y, k=8):
    import shap
    m = make_model(); m.fit(df[feats], y)
    sv = np.asarray(shap.TreeExplainer(m).shap_values(df[feats]))
    imp = (np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))) if sv.ndim > 2 else np.abs(sv).mean(0))
    imp = np.asarray(imp).ravel(); order = np.argsort(imp)[::-1]
    return [(feats[i], float(imp[i])) for i in order[:k]]


def node_A(df, res):
    print("\n== NODE A — Dissolvable? (uric acid vs non-UA) ==")
    y = (df["dominant_parent"] == "UA").astype(int); g = df["patient_bag"]
    P = oof_proba(df, HU + LABS, y, g)
    auc, lo, hi = boot_auc(y.values, P, g.values)
    print(f"  HU+labs: AUROC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f}); n_UA={int(y.sum())}/{len(y)}")
    res["node_A"] = {"auc": auc, "ci": [lo, hi], "n_pos": int(y.sum()), "n": len(y)}


def node_C1(df, res):
    print("\n== NODE C1 — Multicomponent? (single vs multi, from HU architecture) ==")
    second = df[PCOLS].apply(lambda r: sorted(r.values)[-2], axis=1)
    g = df["patient_bag"]
    print(f"  {'2nd-comp thr':>12s} {'n_mixed':>8s} {'imaging AUROC (95% CI)':>26s} {'+labs':>16s}")
    sweep = {}
    for tau in (0.05, 0.10, 0.25, 0.40):
        ym = (second >= tau).astype(int)
        if ym.sum() < 10 or ym.sum() > len(ym) - 10:
            continue
        Pi = oof_proba(df, SHAPE, ym, g)
        ai, li, hi_ = boot_auc(ym.values, Pi, g.values)
        Pl = oof_proba(df, SHAPE + LABS, ym, g)
        al, ll, hl = boot_auc(ym.values, Pl, g.values)
        print(f"  {tau:12.2f} {int(ym.sum()):8d}   {ai:.3f} ({li:.3f}-{hi_:.3f})   {al:.3f} ({ll:.3f}-{hl:.3f})")
        sweep[f"thr_{int(tau*100)}"] = {"n_mixed": int(ym.sum()),
                                        "imaging": {"auc": ai, "ci": [li, hi_]},
                                        "plus_labs": {"auc": al, "ci": [ll, hl]}}
    # SHAP at the mid threshold to see if architecture features drive it
    ym = (second >= 0.25).astype(int)
    print("  SHAP (imaging, thr 0.25) — architecture features expected on top:")
    for f, v in shap_top(df, SHAPE, ym):
        print(f"    {f:18s} {v:.4f}")
    res["node_C1"] = sweep


def node_C2(df, res):
    print("\n== NODE C2 — Predominant CaP? (CaP rule-in among calcium stones) ==")
    sub = df[df["dominant_parent"].isin(["CaOx", "CaP"])].reset_index(drop=True)
    y = (sub["dominant_parent"] == "CaP").astype(int); g = sub["patient_bag"]
    P = oof_proba(sub, HU + LABS, y, g)
    auc, lo, hi = boot_auc(y.values, P, g.values)
    from sklearn.metrics import roc_curve, confusion_matrix
    fpr, tpr, thr = roc_curve(y.values, P); j = int(np.argmax(tpr - fpr))
    pred = (P >= thr[j]).astype(int)
    tn, fp, fn, tp = confusion_matrix(y.values, pred).ravel()
    print(f"  HU+labs: AUROC {auc:.3f} (95% CI {lo:.3f}-{hi:.3f}); n_CaP={int(y.sum())}/{len(y)}")
    print(f"  Youden point: sens {tp/(tp+fn):.2f} spec {tn/(tn+fp):.2f} "
          f"PPV {tp/(tp+fp):.2f} NPV {tn/(tn+fn):.2f}")
    res["node_C2"] = {"auc": auc, "ci": [lo, hi], "n_pos": int(y.sum()), "n": len(y)}


def main():
    df = build_table().reset_index(drop=True)
    print(f"cohort: {len(df)} stones / {df.patient_bag.nunique()} patients; "
          f"multicomponent (2nd comp >=5%) = {int((df[PCOLS].apply(lambda r: sorted(r.values)[-2],axis=1)>=0.05).sum())}")
    res = {}
    node_A(df, res)
    node_C1(df, res)
    node_C2(df, res)
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_decision_nodes.json"), "w"), indent=2)
    print(f"\nresults -> {os.path.join(C.MODEL_DIR, 'drstone_decision_nodes.json')}")


if __name__ == "__main__":
    main()
