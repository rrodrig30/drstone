"""A/B test: does the stone density-gradient (core/periphery HU ratio) improve
the composition model?

Protocol mirrors compose_model (the deployed model): 20x repeated
StratifiedGroupKFold, patient-grouped, averaged out-of-fold probabilities.
Compares BASELINE features vs BASELINE + hu_core_over_rim on:
  - macro-F1, balanced accuracy
  - per-class one-vs-rest AUC (the CaOx/CaP contrast is where the gradient
    should help, since single-energy CT can't otherwise separate them)
  - a focused CaOx-vs-CaP binary AUC (HU-only vs HU + ratio)
Plus patient-clustered bootstrap CIs on the deltas, a SHAP read on whether the
model actually uses the ratio, per-kernel stability, and a measurability audit.

Run:  python -m drstone.gradient_ab
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.compose_model import CLASSES, FEATURES, make_model, _map_class

warnings.filterwarnings("ignore")

GRAD = "hu_core_over_rim"
N_REPEATS = 20
N_BOOT = 2000


def build_table() -> pd.DataFrame:
    ms = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv"),
                     dtype={"canonical_mrn": str, "patient_bag": str})
    ms = ms[ms["match_quality"].isin(["rank_mass", "single_comp"])].copy()
    grad = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_gradient.csv"),
                       dtype={"canonical_mrn": str})
    ms = ms.merge(grad[["canonical_mrn", "stone_idx", "hu_core_p50", "hu_rim_p50",
                        GRAD, "grad_measurable", "volume_mm3_recheck"]],
                  on=["canonical_mrn", "stone_idx"], how="left", suffixes=("", "_g"))
    stones = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"),
                         dtype={"canonical_mrn": str})
    labs = stones[["canonical_mrn", "urine_ph", "na", "k", "cl", "co2", "bun",
                   "creatinine", "ca", "glucose", "age", "gender"]].drop_duplicates("canonical_mrn")
    df = ms.merge(labs, on="canonical_mrn", how="left")
    df["anion_gap"] = df["na"] - df["cl"] - df["co2"]
    df["gender_M"] = (df["gender"] == "M").astype(float)
    df["y"] = df["dominant_parent"].map(_map_class)
    # kernel / slice thickness for the per-kernel stability check
    hu = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stone_hu.csv"),
                     dtype={"canonical_mrn": str})
    df = df.merge(hu[["canonical_mrn", "kernel", "slice_thickness", "kvp"]]
                  .drop_duplicates("canonical_mrn"), on="canonical_mrn", how="left")
    return df


def repeated_oof(X, y, groups, feats):
    """Averaged OOF probabilities over N_REPEATS grouped 5-fold splits."""
    from sklearn.model_selection import StratifiedGroupKFold
    counts = {c: int((y == c).sum()) for c in CLASSES}
    classes = [c for c in CLASSES if counts[c] >= 2]
    P = np.zeros((len(X), len(classes))); cnt = np.zeros(len(X))
    Xf = X[feats]
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(Xf, y, groups):
            m = make_model(); m.fit(Xf.iloc[tr], y.iloc[tr])
            proba = m.predict_proba(Xf.iloc[te])
            col = {c: i for i, c in enumerate(m.classes_)}
            for j, c in enumerate(classes):
                if c in col:
                    P[te, j] += proba[:, col[c]]
            cnt[te] += 1
    P = P / np.maximum(cnt[:, None], 1)
    P = P / P.sum(1, keepdims=True)
    return P, classes


def score(y, P, classes):
    from sklearn.metrics import f1_score, balanced_accuracy_score, roc_auc_score
    pred = np.array(classes)[P.argmax(1)]
    out = {"macro_f1": float(f1_score(y, pred, average="macro", labels=classes)),
           "balanced_acc": float(balanced_accuracy_score(y, pred)), "ovr_auc": {}}
    yv = y.values
    for j, c in enumerate(classes):
        yc = (yv == c).astype(int)
        out["ovr_auc"][c] = (float(roc_auc_score(yc, P[:, j]))
                             if 2 <= yc.sum() < len(yc) else float("nan"))
    return out, pred


def caox_vs_cap_auc(df, feats):
    """Focused binary AUC on CaOx-vs-CaP stones only, patient-grouped CV."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import roc_auc_score
    sub = df[df["y"].isin(["CaOx", "CaP"])].copy()
    yb = (sub["y"] == "CaOx").astype(int)
    g = sub["patient_bag"]; Xf = sub[feats]
    oof = np.zeros(len(sub)); cnt = np.zeros(len(sub))
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(Xf, yb, g):
            m = make_model(); m.fit(Xf.iloc[tr], yb.iloc[tr])
            oof[te] += m.predict_proba(Xf.iloc[te])[:, 1]; cnt[te] += 1
    oof /= np.maximum(cnt, 1)
    return float(roc_auc_score(yb, oof)), len(sub), int(yb.sum()), oof, yb.values, g.values


def boot_delta(y, Pa, Pb, classes, groups, metric):
    """Patient-clustered bootstrap of metric(B) - metric(A)."""
    from sklearn.metrics import f1_score, roc_auc_score
    uniq = np.array(sorted(set(groups)))
    gidx = {g: np.where(groups == g)[0] for g in uniq}
    yv = y.values

    def _m(P, idx):
        if metric == "macro_f1":
            pred = np.array(classes)[P[idx].argmax(1)]
            return f1_score(yv[idx], pred, average="macro", labels=classes)
        c = metric  # OvR AUC for class c
        j = classes.index(c); yc = (yv[idx] == c).astype(int)
        return roc_auc_score(yc, P[idx, j]) if 2 <= yc.sum() < len(yc) else np.nan

    rng = np.random.RandomState(0); deltas = []
    for _ in range(N_BOOT):
        gs = uniq[rng.randint(0, len(uniq), len(uniq))]
        idx = np.concatenate([gidx[g] for g in gs])
        d = _m(Pb, idx) - _m(Pa, idx)
        if d == d:
            deltas.append(d)
    deltas = np.array(deltas)
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    df = build_table().reset_index(drop=True)
    X, y, groups = df[FEATURES + [GRAD]], df["y"], df["patient_bag"]
    n_meas = int(df["grad_measurable"].fillna(0).sum())
    print(f"cohort: {len(df)} stones / {groups.nunique()} patients")
    print(f"gradient measurable: {n_meas}/{len(df)} stones "
          f"({100*n_meas/len(df):.0f}%); {df[GRAD].notna().sum()} non-null ratios")
    print(f"class counts: {{ {', '.join(f'{c}:{int((y==c).sum())}' for c in CLASSES)} }}")

    base_feats = FEATURES
    grad_feats = FEATURES + [GRAD]
    Pa, classes = repeated_oof(df, y, groups, base_feats)
    Pb, _ = repeated_oof(df, y, groups, grad_feats)
    sa, _ = score(y, Pa, classes)
    sb, _ = score(y, Pb, classes)

    print("\n==== multiclass: BASELINE vs +gradient (patient-grouped CV) ====")
    print(f"{'metric':16s} {'baseline':>10s} {'+grad':>10s} {'delta':>9s}  95% CI (clustered boot)")
    for key in ["macro_f1", "balanced_acc"]:
        d = sb[key] - sa[key]
        ci = boot_delta(y, Pa, Pb, classes, groups.values, key) if key == "macro_f1" else None
        cis = f"[{ci[1]:+.3f}, {ci[2]:+.3f}]" if ci else ""
        print(f"{key:16s} {sa[key]:10.3f} {sb[key]:10.3f} {d:+9.3f}  {cis}")
    print("per-class one-vs-rest AUC:")
    for c in classes:
        a, b = sa["ovr_auc"][c], sb["ovr_auc"][c]
        ci = boot_delta(y, Pa, Pb, classes, groups.values, c)
        print(f"  {c:9s} base={a:.3f}  +grad={b:.3f}  delta={b-a:+.3f}  "
              f"95% CI [{ci[1]:+.3f}, {ci[2]:+.3f}]")

    # ---- focused CaOx vs CaP (where the gradient should matter most) -------
    hu_only = ["hu_peak", "hu_mean", "hu_p95", "volume_mm3"]
    a_auc, n_sub, n_caox, *_ = caox_vs_cap_auc(df, hu_only)
    b_auc, *_ = caox_vs_cap_auc(df, hu_only + [GRAD])
    print(f"\n==== CaOx-vs-CaP binary AUC ({n_sub} stones, {n_caox} CaOx) ====")
    print(f"  HU-only        AUC = {a_auc:.3f}")
    print(f"  HU + ratio     AUC = {b_auc:.3f}   (delta {b_auc-a_auc:+.3f})")

    # ---- SHAP: does the model actually use the ratio? ----------------------
    import joblib, shap
    final = make_model(); final.fit(df[grad_feats], y)
    expl = shap.TreeExplainer(final)
    sv = expl.shap_values(df[grad_feats])
    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 3:
        imp = np.abs(sv).mean(axis=(0, 2))
    else:
        imp = np.abs(sv).mean(axis=0)
    order = np.argsort(imp)[::-1]
    rank = list(np.array(grad_feats)[order]).index(GRAD) + 1
    print(f"\n==== SHAP importance (mean|SHAP|) ====")
    print(f"  {GRAD}: {imp[grad_feats.index(GRAD)]:.4f}  -> rank {rank}/{len(grad_feats)}")
    for k in order[:6]:
        print(f"    {grad_feats[k]:16s} {imp[k]:.4f}")

    # ---- per-kernel stability of the ratio ---------------------------------
    print(f"\n==== ratio stability by reconstruction kernel ====")
    mk = df[df[GRAD].notna()]
    for kern, g in mk.groupby(mk["kernel"].fillna("unknown")):
        if len(g) >= 5:
            print(f"  {str(kern):10s} n={len(g):3d}  ratio mean={g[GRAD].mean():.2f} "
                  f"sd={g[GRAD].std():.2f}  median={g[GRAD].median():.2f}")

    # ---- ratio by composition (the biology check) -------------------------
    print(f"\n==== ratio by dominant composition (measurable stones) ====")
    for c in CLASSES:
        g = mk[mk["y"] == c]
        if len(g) >= 3:
            print(f"  {c:9s} n={len(g):3d}  core/rim mean={g[GRAD].mean():.2f} "
                  f"sd={g[GRAD].std():.2f}")

    # ---- persist + figure --------------------------------------------------
    res = {"n": len(df), "n_patients": int(groups.nunique()), "n_measurable": n_meas,
           "baseline": sa, "plus_gradient": sb,
           "caox_vs_cap": {"hu_only_auc": a_auc, "hu_plus_ratio_auc": b_auc,
                           "n": n_sub, "n_caox": n_caox},
           "shap_rank_of_ratio": rank, "shap_importance_ratio": float(imp[grad_feats.index(GRAD)])}
    json.dump(res, open(os.path.join(C.MODEL_DIR, "drstone_gradient_ab.json"), "w"), indent=2)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = np.arange(len(classes)); w = 0.38
        ax.bar(xs - w/2, [sa["ovr_auc"][c] for c in classes], w, label="baseline", color="#90a4b8")
        ax.bar(xs + w/2, [sb["ovr_auc"][c] for c in classes], w, label="+ core/rim ratio", color="#2b6cb0")
        ax.set_xticks(xs); ax.set_xticklabels(classes); ax.set_ylim(0.45, 0.85)
        ax.axhline(0.5, color="#999", ls="--", lw=.8)
        ax.set_ylabel("one-vs-rest AUC"); ax.set_title("Composition AUC: baseline vs + density gradient")
        ax.legend(); fig.tight_layout()
        fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_gradient_ab.png"), dpi=200)
        print(f"\nfigure -> {os.path.join(C.OUTPUT_DIR, 'drstone_gradient_ab.png')}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"results -> {os.path.join(C.MODEL_DIR, 'drstone_gradient_ab.json')}")


if __name__ == "__main__":
    main()
