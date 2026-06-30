"""A/B: vascular / chronic-infection markers.

Aortic calcification indexes calcium-phosphate / mineral metabolism (CKD-MBD,
high Ca x PO4) — the axis that precipitates CALCIUM-PHOSPHATE stones and is NOT
otherwise measured (no serum phosphate/PTH). Hypothesis: aortic calcification is
higher in CaP than CaOx, BEYOND age (which drives both). Bladder/prostate markers
are exploratory (struvite n=27).

Steps: (1) biology gate, INCLUDING an age-adjusted check (does aortic calc
separate CaP from CaOx within age tertiles, and via logistic regression
controlling for age?); (2) patient-grouped CV A/B on the CaOx-vs-CaP head and the
5-class model, clustered-bootstrap CIs + SHAP.

Run:  python -m drstone.vascular_ab
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

VASC = ["aortic_agatston", "aortic_calc_vol_mm3", "aortic_calc_nfoci"]
INFECT = ["bladder_wall_frac", "bladder_vol_mm3", "prostate_calc_vol_mm3"]
CALCVAR = "aortic_agatston"   # denoised, density-weighted score for the biology gate
N_BOOT = 3000


def merged():
    df = build_table()
    v = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_vascular.csv"),
                    dtype={"canonical_mrn": str}).drop_duplicates("canonical_mrn")
    return df.merge(v, on="canonical_mrn", how="left")


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


def biology_gate(df):
    from scipy import stats
    print("=== BIOLOGY GATE: aortic calcification by composition (hyp: CaP > CaOx) ===")
    pat = df.drop_duplicates("canonical_mrn")
    print(f"  aorta measurable: {int(pat.aortic_calc_frac.notna().sum())}/{len(pat)} | "
          f"any aortic calc: {int((pat.aortic_calc_vol_mm3.fillna(0) > 0).sum())} "
          f"| prostate present: {int(pat.prostate_present.fillna(0).sum())}")
    print(f"  {'class':9s} {'n':>4s} {'aortic_calc_mm3(med)':>20s} {'nfoci(med)':>11s} {'bladder_wall':>12s}")
    for c in CLASSES:
        g = df[df["y"] == c]
        if len(g) >= 3:
            print(f"  {c:9s} {len(g):4d} {g['aortic_calc_vol_mm3'].median():20.1f} "
                  f"{g['aortic_calc_nfoci'].median():11.0f} {g['bladder_wall_frac'].mean():12.3f}")

    sub = df[df["y"].isin(["CaOx", "CaP"])].dropna(subset=[CALCVAR]).copy()
    ox = sub[sub.y == "CaOx"][CALCVAR]; cp = sub[sub.y == "CaP"][CALCVAR]
    u, p = stats.mannwhitneyu(ox, cp, alternative="two-sided")
    direction = "CaP MORE (supports)" if cp.median() > ox.median() else "CaOx more (against)"
    print(f"\n  unadjusted: CaOx med {ox.median():.1f} vs CaP med {cp.median():.1f} -> {direction}; MWU p={p:.3f}")

    # age-adjusted: logistic regression P(CaP) ~ age + log(aortic_calc) — is the
    # aortic-calc coefficient significant beyond age?
    sub["lcalc"] = np.log1p(sub[CALCVAR])
    sub["age_f"] = pd.to_numeric(sub["age"], errors="coerce")
    s2 = sub.dropna(subset=["age_f"])
    yb = (s2.y == "CaP").astype(int).values
    try:
        from sklearn.linear_model import LogisticRegression
        from scipy.stats import norm
        X = s2[["age_f", "lcalc"]].values
        Xs = (X - X.mean(0)) / X.std(0)
        lr = LogisticRegression().fit(Xs, yb)
        # bootstrap CI on the standardized aortic-calc coefficient (patient-level)
        coefs = []
        rng = np.random.RandomState(0)
        for _ in range(2000):
            idx = rng.randint(0, len(yb), len(yb))
            if yb[idx].sum() < 5 or yb[idx].sum() > len(idx) - 5:
                continue
            try:
                coefs.append(LogisticRegression().fit(Xs[idx], yb[idx]).coef_[0][1])
            except Exception:
                pass
        coefs = np.array(coefs)
        print(f"  age-adjusted logistic: aortic-calc std coef = {lr.coef_[0][1]:+.3f} "
              f"(age coef {lr.coef_[0][0]:+.3f}); 95% CI [{np.percentile(coefs,2.5):+.3f}, "
              f"{np.percentile(coefs,97.5):+.3f}], P(coef>0)={ (coefs>0).mean():.2f}")
    except Exception as e:
        print(f"  (age-adjusted check skipped: {e})")


def head_ab(df):
    from sklearn.metrics import roc_auc_score
    sub = df[df["y"].isin(["CaOx", "CaP"])].reset_index(drop=True)
    y = (sub["y"] == "CaP").astype(int); g = sub["patient_bag"]
    sets = {"HU+labs": HU + LABS, "+aortic-calc": HU + LABS + VASC,
            "+infection": HU + LABS + INFECT, "+both": HU + LABS + VASC + INFECT}
    print(f"\n== CaOx-vs-CaP head: {len(sub)} stones (CaP={int(y.sum())}, CaOx={int((1-y).sum())}) ==")
    oofs = {n: oof_proba(sub, f, y, g) for n, f in sets.items()}
    base = oofs["HU+labs"]
    print(f"{'feature set':14s} {'AUC':>6s}   delta vs HU+labs  (95% CI, P[Δ>0])")
    res = {}
    for n in sets:
        auc = roc_auc_score(y.values, oofs[n])
        if n == "HU+labs":
            print(f"{n:14s} {auc:6.3f}   (baseline)"); res[n] = {"auc": float(auc)}
        else:
            m, lo, hi, p = paired_boot(y.values, base, oofs[n], g.values)
            print(f"{n:14s} {auc:6.3f}   {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  P={p:.2f}")
            res[n] = {"auc": float(auc), "delta": m, "ci": [lo, hi], "p_gt0": p}

    import shap
    feats = HU + LABS + VASC + INFECT
    mdl = make_model(); mdl.fit(sub[feats], y)
    sv = np.asarray(shap.TreeExplainer(mdl).shap_values(sub[feats]))
    imp = (np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))) if sv.ndim > 2 else np.abs(sv).mean(0))
    imp = np.asarray(imp).ravel(); order = np.argsort(imp)[::-1]
    print(f"\nSHAP usage (of {len(feats)}):")
    for ff in VASC + INFECT:
        rank = list(np.array(feats)[order]).index(ff) + 1
        print(f"  {ff:20s} rank {rank:2d}/{len(feats)}  mean|SHAP| {imp[feats.index(ff)]:.4f}")
    return res


def multiclass_ab(df):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import f1_score, roc_auc_score
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
        return f1_score(y, pred, average="macro", labels=classes), aucs

    print("\n== 5-class composition: baseline vs +vascular/infection ==")
    print(f"  {'set':12s} {'macroF1':>8s} {'CaOx':>6s} {'CaP':>6s} {'UA':>6s} {'Struv':>6s}")
    for tag, feats in [("baseline", FEATURES), ("+vasc+infect", FEATURES + VASC + INFECT)]:
        f1, a = run(feats)
        print(f"  {tag:12s} {f1:8.3f} {a['CaOx']:6.3f} {a['CaP']:6.3f} {a['UA']:6.3f} "
              f"{a.get('Struvite', float('nan')):6.3f}")


def main():
    df = merged()
    biology_gate(df)
    res = head_ab(df)
    multiclass_ab(df)
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_vascular_ab.json"), "w"), indent=2)
    print(f"\nresults -> {os.path.join(C.MODEL_DIR, 'drstone_vascular_ab.json')}")


if __name__ == "__main__":
    main()
