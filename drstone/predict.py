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


_COMPOSE = None


def _load_compose():
    global _COMPOSE
    if _COMPOSE is None:
        import joblib
        _COMPOSE = joblib.load(os.path.join(C.MODEL_DIR, "drstone_compose_model.pkl"))
    return _COMPOSE


def compose_assess(form: dict) -> dict:
    """Full assessment: composition distribution + acute (MET vs surgery) +
    prevention guidance. Inputs are routine ED data; missing values tolerated."""
    from drstone.recommendations import acute, prevention, DRAFT_NOTICE
    b = _load_compose()
    feats, classes = b["features"], b["classes"]
    na, cl, co2 = _num(form.get("na")), _num(form.get("cl")), _num(form.get("co2"))
    sex = str(form.get("sex", "")).strip().upper()
    gender_m = 1.0 if sex.startswith("M") else (0.0 if sex.startswith("F") else np.nan)
    labs = {k: _num(form.get(k)) for k in
            ("urine_ph", "na", "k", "cl", "co2", "bun", "creatinine", "ca", "glucose")}
    vals = {
        "hu_peak": _num(form.get("hu_peak")), "hu_mean": _num(form.get("hu_mean")),
        "hu_p95": _num(form.get("hu_p95")), "volume_mm3": _num(form.get("volume_mm3")),
        "urine_ph": labs["urine_ph"], "na": na, "k": _num(form.get("k")), "cl": cl, "co2": co2,
        "anion_gap": (na - cl - co2) if not any(np.isnan([na, cl, co2])) else np.nan,
        "bun": labs["bun"], "creatinine": labs["creatinine"], "ca": labs["ca"],
        "glucose": labs["glucose"], "age": _num(form.get("age")), "gender_M": gender_m,
    }
    X = pd.DataFrame([[vals.get(f, np.nan) for f in feats]], columns=feats)
    proba = b["model"].predict_proba(X)[0]
    dist = sorted([{"type": c, "p": float(p)} for c, p in zip(classes, proba)],
                  key=lambda d: -d["p"])
    top = [d["type"] for d in dist[:2] if d["p"] >= 0.15] or [dist[0]["type"]]
    diam = _num(form.get("stone_size_mm"))
    ac = acute(diameter_mm=(None if (isinstance(diam, float) and np.isnan(diam)) else diam),
               location=form.get("location", ""), creatinine=labs["creatinine"],
               infection=str(form.get("infection", "")).lower() in ("1", "true", "yes", "on"))
    return {"distribution": dist, "top": top, "acute": ac,
            "prevention": prevention(top, labs), "draft": DRAFT_NOTICE,
            "n_provided": int(sum(1 for v in vals.values()
                                  if not (isinstance(v, float) and np.isnan(v))))}


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
