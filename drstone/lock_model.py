"""Lock the lean Dr Stone model with proper uncertainty + clinical utility.

Real-world framing: the inputs are exactly what an ED stone patient already has —
a non-contrast stone-protocol CT (peak/mean stone HU) plus routine labs (urine
pH, basic metabolic panel / acid-base, demographics). No DECT, no special
acquisition. Target: uric-acid vs non-uric-acid, the decision that changes
management (medical dissolution vs intervention).

Uncertainty done honestly:
  * repeated StratifiedGroupKFold (patient-level) -> averaged out-of-fold probs
  * patient-clustered bootstrap -> 95% CI for AUC and operating-point metrics
  * calibration (Brier) and decision-curve analysis (net clinical benefit)

Run:  python -m drstone.lock_model
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C

warnings.filterwarnings("ignore")

# Pre-specified lean feature set — all routinely available in the ED.
LEAN_FEATURES = ["hu_peak", "hu_mean", "urine_ph", "co2", "cl", "anion_gap",
                 "bun", "creatinine", "ca", "glucose", "age", "gender_M"]
MONO = {"hu_peak": -1, "hu_mean": -1, "urine_ph": -1}    # P(UA) decreases with HU, pH
N_REPEATS = 25
N_BOOT = 1000
TARGET_SENS = 0.90        # don't miss a medically-dissolvable UA stone


def build_table():
    ms = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_matched_stones.csv"),
                     dtype={"canonical_mrn": str, "patient_bag": str})
    ms = ms[ms["match_quality"].isin(["rank_mass", "single_comp"])].copy()   # clean labels
    stones = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"),
                         dtype={"canonical_mrn": str})
    labs = stones[["canonical_mrn", "urine_ph", "na", "k", "cl", "co2", "bun",
                   "creatinine", "ca", "glucose", "age", "gender"]].drop_duplicates("canonical_mrn")
    df = ms.merge(labs, on="canonical_mrn", how="left")
    df["anion_gap"] = df["na"] - df["cl"] - df["co2"]
    df["gender_M"] = (df["gender"] == "M").astype(float)
    return df


def make_model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    mono = [MONO.get(f, 0) for f in LEAN_FEATURES]
    return HistGradientBoostingClassifier(max_iter=200, max_depth=3, learning_rate=0.05,
                                          monotonic_cst=mono, l2_regularization=1.0,
                                          random_state=0)


def repeated_oof(df, y, groups):
    """Average out-of-fold probabilities over repeated grouped CV."""
    from sklearn.model_selection import StratifiedGroupKFold
    X = df[LEAN_FEATURES]
    acc = np.zeros(len(y)); cnt = np.zeros(len(y))
    for rep in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(5, shuffle=True, random_state=rep)
        for tr, te in sgkf.split(X, y, groups):
            m = make_model(); m.fit(X.iloc[tr], y.iloc[tr])
            acc[te] += m.predict_proba(X.iloc[te])[:, 1]; cnt[te] += 1
    return acc / np.maximum(cnt, 1)


def boot_ci(y, p, groups, stat, n=N_BOOT):
    """Patient-clustered bootstrap 95% CI for a statistic(y, p)."""
    rng = np.random.RandomState(0)
    pts = np.array(sorted(set(groups)))
    by = {g: np.where(groups == g)[0] for g in pts}
    vals = []
    for _ in range(n):
        samp = np.concatenate([by[g] for g in rng.choice(pts, len(pts), replace=True)])
        ys, ps = y.values[samp], p[samp]
        if ys.sum() < 3 or ys.sum() == len(ys):
            continue
        vals.append(stat(ys, ps))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def decision_curve(y, p, thresholds):
    """Net benefit across threshold probabilities (vs treat-all / treat-none)."""
    N = len(y); prev = y.mean()
    rows = []
    for pt in thresholds:
        pred = p >= pt
        tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
        nb = tp / N - fp / N * (pt / (1 - pt))
        nb_all = prev - (1 - prev) * (pt / (1 - pt))
        rows.append((pt, nb, max(nb_all, 0), 0.0))
    return np.array(rows)


def main():
    from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
    df = build_table()
    y = df["y_ua"]; groups = df["patient_bag"].values
    print(f"Locked-model cohort: {len(df)} stones / {df['patient_bag'].nunique()} patients, "
          f"UA={int(y.sum())} ({y.mean()*100:.0f}%)")

    p = repeated_oof(df, y, groups)
    auc = roc_auc_score(y, p)
    auc_lo, auc_hi = boot_ci(y, p, groups, roc_auc_score)
    print(f"\nAUC = {auc:.3f}  (95% CI {auc_lo:.3f}-{auc_hi:.3f})  [{N_REPEATS}x grouped CV, "
          f"patient-clustered bootstrap]")

    # High-sensitivity operating point (don't miss UA)
    fpr, tpr, thr = roc_curve(y, p)
    idx = np.where(tpr >= TARGET_SENS)[0]
    t = thr[idx[0]] if len(idx) else 0.5
    pred = (p >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    sens = tp / (tp + fn); spec = tn / (tn + fp)
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    sens_ci = boot_ci(y, p, groups, lambda yy, pp: ((pp >= t) & (yy == 1)).sum() / max(1, (yy == 1).sum()))
    spec_ci = boot_ci(y, p, groups, lambda yy, pp: ((pp < t) & (yy == 0)).sum() / max(1, (yy == 0).sum()))
    brier = brier_score_loss(y, p)
    print(f"\nOperating point @sensitivity>={TARGET_SENS:.0%} (thr={t:.2f}):")
    print(f"  sensitivity {sens:.2f} (95% CI {sens_ci[0]:.2f}-{sens_ci[1]:.2f})")
    print(f"  specificity {spec:.2f} (95% CI {spec_ci[0]:.2f}-{spec_ci[1]:.2f})")
    print(f"  PPV {ppv:.2f} | NPV {npv:.2f} | Brier {brier:.3f}")

    results = {
        "n_stones": int(len(df)), "n_patients": int(df["patient_bag"].nunique()),
        "n_ua": int(y.sum()), "prevalence": float(y.mean()),
        "features": LEAN_FEATURES, "auc": auc, "auc_ci": [auc_lo, auc_hi],
        "operating_threshold": float(t), "sensitivity": sens, "sensitivity_ci": sens_ci,
        "specificity": spec, "specificity_ci": spec_ci, "ppv": ppv, "npv": npv,
        "brier": brier, "n_repeats": N_REPEATS, "n_boot": N_BOOT,
    }
    json.dump(results, open(os.path.join(C.MODEL_DIR, "drstone_locked_results.json"), "w"), indent=2)

    # Fit + persist the final locked model on all data
    import joblib
    final = make_model(); final.fit(df[LEAN_FEATURES], y)
    joblib.dump({"model": final, "features": LEAN_FEATURES, "threshold": float(t)},
                os.path.join(C.MODEL_DIR, "drstone_locked_model.pkl"))

    # Figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        # ROC
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, color="#b03030", lw=2, label=f"AUC {auc:.2f} (95% CI {auc_lo:.2f}-{auc_hi:.2f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.set_xlabel("1 - specificity"); ax.set_ylabel("sensitivity")
        ax.set_title("Uric-acid stone detection (NCCT + ED labs)"); ax.legend(loc="lower right")
        fig.tight_layout(); fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_locked_roc.png"), dpi=200)
        # Calibration
        frac, mean_pred = calibration_curve(y, p, n_bins=5, strategy="quantile")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(mean_pred, frac, "o-", color="#3a6ea5"); ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("predicted P(UA)"); ax.set_ylabel("observed UA fraction")
        ax.set_title(f"Calibration (Brier {brier:.3f})")
        fig.tight_layout(); fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_locked_calibration.png"), dpi=200)
        # Decision curve
        ths = np.linspace(0.02, 0.6, 40)
        dca = decision_curve(y, p, ths)
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(dca[:, 0], dca[:, 1], color="#b03030", lw=2, label="model")
        ax.plot(dca[:, 0], dca[:, 2], "k--", lw=1, label="treat all as UA")
        ax.axhline(0, color="gray", lw=1, label="treat none")
        ax.set_xlabel("threshold probability"); ax.set_ylabel("net benefit")
        ax.set_title("Decision-curve analysis (clinical utility)"); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_locked_dca.png"), dpi=200)
        print("\nFigures: drstone_locked_{roc,calibration,dca}.png")
    except Exception as e:
        print(f"(figures skipped: {e})")
    print(f"Locked model + results saved to {C.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
