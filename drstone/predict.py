"""Point-of-care inference for the locked Dr Stone model.

Takes the data an ED clinician already has — stone HU from the non-contrast CT
plus routine labs — and returns a calibrated uric-acid probability with a
per-case SHAP rationale. Decision support, not a substitute for stone analysis.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from drstone import config as C

_BUNDLE = None


def _load():
    global _BUNDLE
    if _BUNDLE is None:
        import joblib
        import shap
        b = joblib.load(os.path.join(C.MODEL_DIR, "drstone_locked_model.pkl"))
        b["explainer"] = shap.TreeExplainer(b["model"])
        try:
            r = json.load(open(os.path.join(C.MODEL_DIR, "drstone_locked_results.json")))
            b["prevalence"] = float(r.get("prevalence", 0.11))
            b["auc"] = float(r.get("auc", float("nan")))
            b["npv"] = float(r.get("npv", float("nan")))
        except Exception:
            b["prevalence"], b["auc"], b["npv"] = 0.11, float("nan"), float("nan")
        _BUNDLE = b
    return _BUNDLE


def _num(v):
    try:
        x = float(v)
        return x if not np.isnan(x) else np.nan
    except (TypeError, ValueError):
        return np.nan


def predict(form: dict) -> dict:
    """form: raw clinical inputs (hu_peak, hu_mean, urine_ph, na, cl, co2, bun,
    creatinine, ca, glucose, age, sex). Missing values are allowed."""
    b = _load()
    feats = b["features"]
    na, cl, co2 = _num(form.get("na")), _num(form.get("cl")), _num(form.get("co2"))
    sex = str(form.get("sex", "")).strip().upper()
    gender_m = 1.0 if sex.startswith("M") else (0.0 if sex.startswith("F") else np.nan)
    vals = {
        "hu_peak": _num(form.get("hu_peak")), "hu_mean": _num(form.get("hu_mean")),
        "urine_ph": _num(form.get("urine_ph")), "co2": co2, "cl": cl,
        "anion_gap": (na - cl - co2) if not any(np.isnan([na, cl, co2])) else np.nan,
        "bun": _num(form.get("bun")), "creatinine": _num(form.get("creatinine")),
        "ca": _num(form.get("ca")), "glucose": _num(form.get("glucose")),
        "age": _num(form.get("age")), "gender_M": gender_m,
    }
    X = pd.DataFrame([[vals[f] for f in feats]], columns=feats)
    prob = float(b["model"].predict_proba(X)[0, 1])
    sv = np.asarray(b["explainer"].shap_values(X))
    if sv.ndim == 3:
        sv = sv[:, :, -1]
    contribs = [{"feature": f, "value": vals[f], "shap": float(sv[0, i])}
                for i, f in enumerate(feats)]
    contribs.sort(key=lambda d: -abs(d["shap"]))
    n_provided = int(sum(1 for v in vals.values() if not (isinstance(v, float) and np.isnan(v))))
    return {"probability": prob, "prevalence": b["prevalence"], "auc": b["auc"],
            "npv": b["npv"], "contributions": contribs, "n_provided": n_provided,
            "n_features": len(feats)}
