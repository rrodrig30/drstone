"""A/B: do hepatic steatosis (fatty liver) and muscle quality (sarcopenia) help?

Microbiome hypothesis: gut dysbiosis raises enteric oxalate absorption (favoring
CaOx) AND drives hepatic fat (NAFLD) — so fatty liver should be MORE prevalent in
CaOx than in CaPO4. This script (1) tests that biology direction first as a cheap
gate, then (2) runs the patient-grouped CV A/B (HU+labs vs +fatty / +muscle /
+both) with clustered-bootstrap CIs + SHAP, and (3) the 5-class impact.

Run:  python -m drstone.body_comp_ab
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

FATTY = ["liver_spleen_diff", "liver_hu"]
MUSCLE = ["muscle_hu", "muscle_vol_mm3"]
N_BOOT = 3000
STEATOSIS_LS = -10.0   # liver-spleen HU difference threshold for steatosis


def merged():
    df = build_table()
    bc = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_body_comp.csv"),
                     dtype={"canonical_mrn": str}).drop_duplicates("canonical_mrn")
    return df.merge(bc, on="canonical_mrn", how="left")


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
    print("=== BIOLOGY GATE: fatty liver by composition (hypothesis: CaOx > CaP) ===")
    pat = df.drop_duplicates("canonical_mrn")
    print(f"  liver/spleen measurable: {int(pat.liver_spleen_diff.notna().sum())}/{len(pat)} "
          f"({100*pat.liver_spleen_diff.notna().mean():.0f}%)  | "
          f"muscle: {int(pat.muscle_hu.notna().sum())}/{len(pat)}")
    print(f"  {'class':9s} {'n':>4s} {'L-S diff(mean)':>15s} {'steatosis%':>11s} {'muscle_hu(mean)':>16s}")
    rates = {}
    for c in CLASSES:
        g = df[df["y"] == c]
        if len(g) >= 3:
            steat = 100 * (g["liver_spleen_diff"] < STEATOSIS_LS).mean()
            rates[c] = (g["liver_spleen_diff"].mean(), steat, g["muscle_hu"].mean())
            print(f"  {c:9s} {len(g):4d} {g['liver_spleen_diff'].mean():15.1f} "
                  f"{steat:11.0f} {g['muscle_hu'].mean():16.1f}")
    # explicit CaOx-vs-CaP direction
    ox, cp = df[df.y == "CaOx"], df[df.y == "CaP"]
    from scipy import stats
    a = ox["liver_spleen_diff"].dropna(); b = cp["liver_spleen_diff"].dropna()
    t, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    direction = ("CaOx FATTIER (supports hypothesis)" if a.mean() < b.mean()
                 else "CaP fattier (CONTRADICTS hypothesis)")
    print(f"\n  CaOx L-S mean {a.mean():.1f} vs CaP {b.mean():.1f} -> {direction}; "
          f"Mann-Whitney p={p:.3f}")
    return rates


def head_ab(df):
    from sklearn.metrics import roc_auc_score
    sub = df[df["y"].isin(["CaOx", "CaP"])].reset_index(drop=True)
    y = (sub["y"] == "CaP").astype(int); g = sub["patient_bag"]
    sets = {"HU+labs": HU + LABS, "+fatty-liver": HU + LABS + FATTY,
            "+muscle": HU + LABS + MUSCLE, "+both": HU + LABS + FATTY + MUSCLE}
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
    feats = HU + LABS + FATTY + MUSCLE
    mdl = make_model(); mdl.fit(sub[feats], y)
    sv = np.asarray(shap.TreeExplainer(mdl).shap_values(sub[feats]))
    imp = (np.abs(sv).mean(axis=tuple(range(sv.ndim - 1))) if sv.ndim > 2 else np.abs(sv).mean(0))
    imp = np.asarray(imp).ravel(); order = np.argsort(imp)[::-1]
    print(f"\nSHAP usage (of {len(feats)}):")
    for ff in FATTY + MUSCLE:
        rank = list(np.array(feats)[order]).index(ff) + 1
        print(f"  {ff:18s} rank {rank:2d}/{len(feats)}  mean|SHAP| {imp[feats.index(ff)]:.4f}")
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

    print("\n== 5-class composition: baseline vs +body-comp ==")
    print(f"  {'set':13s} {'macroF1':>8s} {'CaOx':>6s} {'CaP':>6s} {'UA':>6s}")
    for tag, feats in [("baseline", FEATURES), ("+fatty+muscle", FEATURES + FATTY + MUSCLE)]:
        f1, a = run(feats)
        print(f"  {tag:13s} {f1:8.3f} {a['CaOx']:6.3f} {a['CaP']:6.3f} {a['UA']:6.3f}")


def main():
    df = merged()
    biology_gate(df)
    res = head_ab(df)
    multiclass_ab(df)
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_body_comp_ab.json"), "w"), indent=2)
    print(f"\nresults -> {os.path.join(C.MODEL_DIR, 'drstone_body_comp_ab.json')}")


if __name__ == "__main__":
    main()
