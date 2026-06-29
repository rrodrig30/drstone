"""Recalibrate the CaOx-vs-CaP head's operating threshold for higher CaP
sensitivity (rule-in for the CaP work-up: alkaline urine / distal RTA /
hyperparathyroidism, and caution with potassium citrate).

Thresholds are derived from the patient-grouped cross-validated OOF P(CaP) — not
the in-sample fit — so the reported sensitivity is honest. The chosen
high-sensitivity threshold is written into drstone_caox_cap_head.pkl as
`threshold` and used by the cascade in predict._refine_calcium to fire a CaP
work-up screen.

Run:  python -m drstone.caox_cap_recalibrate [--target 0.90]
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

from drstone import config as C
from drstone.caox_cap_head import SETS, oof_proba
from drstone.gradient_ab import build_table

warnings.filterwarnings("ignore")

DEPLOY = "HU+labs"
TARGETS = [0.80, 0.85, 0.90, 0.95]


def _ops_at(y, P, t):
    from sklearn.metrics import confusion_matrix
    pred = (P >= t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {"thr": float(t), "sens": float(sens), "spec": float(spec),
            "ppv": float(ppv), "npv": float(npv),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)}


def main():
    from sklearn.metrics import roc_curve
    target = float(sys.argv[sys.argv.index("--target") + 1]) if "--target" in sys.argv else 0.90
    df = build_table()
    sub = df[df["y"].isin(["CaOx", "CaP"])].reset_index(drop=True)
    y = (sub["y"] == "CaP").astype(int).values
    groups = sub["patient_bag"]
    print(f"calcium subset: {len(sub)} stones (CaP={int(y.sum())}, CaOx={int((1-y).sum())})")
    print("deriving thresholds from patient-grouped OOF P(CaP)...\n")

    P = oof_proba(sub, SETS[DEPLOY], pd.Series(y), groups)
    fpr, tpr, thr = roc_curve(y, P)

    def thr_for_sens(s):
        idx = np.where(tpr >= s)[0]          # tpr increasing as thr decreases
        return float(thr[idx[0]]) if len(idx) else float(thr[-1])

    # Youden reference
    j = int(np.argmax(tpr - fpr))
    table = {"youden": _ops_at(y, P, thr[j])}
    print(f"{'target sens':>11s} {'thr':>6s} {'sens':>6s} {'spec':>6s} {'PPV':>6s} {'NPV':>6s}  (TP/FP/FN/TN)")
    yo = table["youden"]
    print(f"{'Youden':>11s} {yo['thr']:6.3f} {yo['sens']:6.2f} {yo['spec']:6.2f} "
          f"{yo['ppv']:6.2f} {yo['npv']:6.2f}  ({yo['tp']}/{yo['fp']}/{yo['fn']}/{yo['tn']})")
    for s in TARGETS:
        op = _ops_at(y, P, thr_for_sens(s))
        table[f"sens_{int(s*100)}"] = op
        mark = "  <-- chosen" if abs(s - target) < 1e-9 else ""
        print(f"{s:11.2f} {op['thr']:6.3f} {op['sens']:6.2f} {op['spec']:6.2f} "
              f"{op['ppv']:6.2f} {op['npv']:6.2f}  ({op['tp']}/{op['fp']}/{op['fn']}/{op['tn']}){mark}")

    chosen = _ops_at(y, P, thr_for_sens(target))
    print(f"\nchosen high-sensitivity operating point (target CaP sens {target:.2f}): "
          f"thr={chosen['thr']:.3f}  sens={chosen['sens']:.2f}  spec={chosen['spec']:.2f}")

    # update the deployed head pkl
    import joblib
    pkl = os.path.join(C.MODEL_DIR, "drstone_caox_cap_head.pkl")
    b = joblib.load(pkl)
    b["threshold"] = chosen["thr"]
    b["threshold_target_sens"] = target
    b["operating_table"] = table
    joblib.dump(b, pkl)
    json.dump({"deployed": DEPLOY, "target_sens": target, "chosen": chosen,
               "table": table}, open(os.path.join(C.MODEL_DIR,
               "drstone_caox_cap_threshold.json"), "w"), indent=2)

    # tradeoff figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ts = np.linspace(0.05, 0.95, 181)
        sens = [(_ops_at(y, P, t)["sens"]) for t in ts]
        spec = [(_ops_at(y, P, t)["spec"]) for t in ts]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(ts, sens, label="CaP sensitivity", color="#c05621", lw=2)
        ax.plot(ts, spec, label="specificity", color="#2b6cb0", lw=2)
        ax.axvline(chosen["thr"], color="#2f855a", ls="--", lw=1.2,
                   label=f"chosen thr={chosen['thr']:.2f} (sens {chosen['sens']:.2f})")
        ax.axvline(yo["thr"], color="#999", ls=":", lw=1, label=f"Youden thr={yo['thr']:.2f}")
        ax.set_xlabel("threshold on P(CaP)"); ax.set_ylabel("rate")
        ax.set_title("CaOx-vs-CaP head: sensitivity/specificity vs threshold")
        ax.legend(fontsize=9); ax.grid(alpha=.25); fig.tight_layout()
        fig.savefig(os.path.join(C.OUTPUT_DIR, "drstone_caox_cap_threshold.png"), dpi=200)
        print(f"figure -> {os.path.join(C.OUTPUT_DIR, 'drstone_caox_cap_threshold.png')}")
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"updated head pkl threshold -> {b['threshold']:.3f} (target sens {target:.2f})")


if __name__ == "__main__":
    main()
