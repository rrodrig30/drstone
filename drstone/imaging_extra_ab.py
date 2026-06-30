"""Step 2 A/B: do nephrocalcinosis and opportunistic BMD add signal?

Patient-level imaging features merged onto the stone cohort. Tests them on the
CaOx-vs-CaP head (the weak spot) and the 5-class model, with patient-grouped
repeated CV, clustered-bootstrap CIs on the paired AUC deltas, SHAP usage, and a
descriptive read (prevalence/coverage, and each feature by composition — the
biology check). BMD's paired delta is, by construction, the controlled
"does it beat what we already have (incl. age)?" test.

Run:  python -m drstone.imaging_extra_ab
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.caox_cap_head import HU, LABS, oof_proba
from drstone.compose_model import CLASSES, FEATURES, make_model
from drstone.gradient_ab import build_table

warnings.filterwarnings("ignore")

NEPHRO = ["nephrocalc_frac", "nephrocalc_nfoci"]
BMD = ["bmd_lumbar_hu"]
N_BOOT = 3000


def merged():
    df = build_table()
    ix = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_imaging_extra.csv"),
                     dtype={"canonical_mrn": str}).drop_duplicates("canonical_mrn")
    return df.merge(ix, on="canonical_mrn", how="left")


def paired_boot(y, Pa, Pb, groups):
    from sklearn.metrics import roc_auc_score
    uniq = np.array(sorted(set(groups)))
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.RandomState(0); d = []
    for _ in range(N_BOOT):
        gs = uniq[rng.randint(0, len(uniq), len(uniq))]
        idx = np.concatenate([gidx[g] for g in gs]); yy = y[idx]
        if yy.sum() < 2 or yy.sum() == len(yy):
            continue
        d.append(roc_auc_score(yy, Pb[idx]) - roc_auc_score(yy, Pa[idx]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float((d > 0).mean())


def descriptives(df):
    pat = df.drop_duplicates("canonical_mrn")
    has_neph = (pat["nephrocalc_nfoci"].fillna(0) >= 1)
    print(f"\n-- coverage / prevalence (patient-level, n={len(pat)}) --")
    print(f"  nephrocalcinosis present (>=1 focus): {int(has_neph.sum())} "
          f"({100*has_neph.mean():.0f}%)")
    print(f"  BMD measurable (lumbar): {int(pat['bmd_lumbar_hu'].notna().sum())} "
          f"({100*pat['bmd_lumbar_hu'].notna().mean():.0f}%)")
    print("-- by dominant composition (stone-level) --")
    print(f"  {'class':9s} {'n':>4s} {'neph%':>6s} {'neph_frac(mean)':>16s} {'BMD lumbar(mean)':>17s}")
    for c in CLASSES:
        g = df[df["y"] == c]
        if len(g) >= 3:
            nephpct = 100 * (g["nephrocalc_nfoci"].fillna(0) >= 1).mean()
            print(f"  {c:9s} {len(g):4d} {nephpct:6.0f} {g['nephrocalc_frac'].mean():16.4f} "
                  f"{g['bmd_lumbar_hu'].mean():17.1f}")


def head_ab(df):
    from sklearn.metrics import roc_auc_score
    sub = df[df["y"].isin(["CaOx", "CaP"])].reset_index(drop=True)
    y = (sub["y"] == "CaP").astype(int); g = sub["patient_bag"]
    sets = {"HU+labs": HU + LABS, "+nephrocalc": HU + LABS + NEPHRO,
            "+BMD": HU + LABS + BMD, "+both": HU + LABS + NEPHRO + BMD}
    print(f"\n== CaOx-vs-CaP head: {len(sub)} stones (CaP={int(y.sum())}, CaOx={int((1-y).sum())}) ==")
    oofs = {n: oof_proba(sub, f, y, g) for n, f in sets.items()}
    base = oofs["HU+labs"]
    print(f"{'feature set':13s} {'AUC':>6s}   delta vs HU+labs  (95% CI, P[Δ>0])")
    res = {}
    for n in sets:
        auc = roc_auc_score(y.values, oofs[n])
        if n == "HU+labs":
            print(f"{n:13s} {auc:6.3f}   (baseline)"); res[n] = {"auc": float(auc)}
        else:
            m, lo, hi, p = paired_boot(y.values, base, oofs[n], g.values)
            print(f"{n:13s} {auc:6.3f}   {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P={p:.2f}")
            res[n] = {"auc": float(auc), "delta": m, "ci": [lo, hi], "p_gt0": p}

    import shap
    feats = HU + LABS + NEPHRO + BMD
    mdl = make_model(); mdl.fit(sub[feats], y)
    sv = np.asarray(shap.TreeExplainer(mdl).shap_values(sub[feats]))
    imp = (np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))) if sv.ndim > 2 else np.abs(sv).mean(0))
    imp = np.asarray(imp).ravel(); order = np.argsort(imp)[::-1]
    print(f"\nSHAP usage (of {len(feats)}):")
    for ff in NEPHRO + BMD:
        rank = list(np.array(feats)[order]).index(ff) + 1
        print(f"  {ff:18s} rank {rank:2d}/{len(feats)}  mean|SHAP| {imp[feats.index(ff)]:.4f}")
    return res


def multiclass_ab(df):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    y = df["y"]; g = df["patient_bag"]
    classes = [c for c in CLASSES if (y == c).sum() >= 2]

    def run(feats):
        P = np.zeros((len(df), len(classes))); cnt = np.zeros(len(df))
        for rep in range(20):
            sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
            for tr, te in sgkf.split(df[feats], y, g):
                m = make_model(); m.fit(df[feats].iloc[tr], y.iloc[tr])
                pr = m.predict_proba(df[feats].iloc[te]); col = {c: i for i, c in enumerate(m.classes_)}
                for j, c in enumerate(classes):
                    if c in col:
                        P[te, j] += pr[:, col[c]]
                cnt[te] += 1
        P = P / np.maximum(cnt[:, None], 1); P = P / P.sum(1, keepdims=True)
        pred = np.array(classes)[P.argmax(1)]
        aucs = {c: (roc_auc_score((y == c).astype(int), P[:, j])
                    if 2 <= (y == c).sum() < len(y) else float("nan"))
                for j, c in enumerate(classes)}
        return f1_score(y, pred, average="macro", labels=classes), balanced_accuracy_score(y, pred), aucs

    print("\n== 5-class composition: baseline vs +imaging ==")
    print(f"  {'set':12s} {'macroF1':>8s} {'CaOx':>6s} {'CaP':>6s} {'UA':>6s}")
    for tag, feats in [("baseline", FEATURES), ("+nephro+BMD", FEATURES + NEPHRO + BMD)]:
        f1, ba, a = run(feats)
        print(f"  {tag:12s} {f1:8.3f} {a['CaOx']:6.3f} {a['CaP']:6.3f} {a['UA']:6.3f}")


def main():
    df = merged()
    descriptives(df)
    res = head_ab(df)
    multiclass_ab(df)
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_imaging_extra_ab.json"), "w"), indent=2)
    print(f"\nresults -> {os.path.join(C.MODEL_DIR, 'drstone_imaging_extra_ab.json')}")


if __name__ == "__main__":
    main()
