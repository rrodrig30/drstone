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


_CAOX_CAP = False  # False = not loaded yet, None = unavailable, dict = loaded
CAP_ALERT_MIN_MASS = 0.34  # only screen for CaP when calcium is a real contender


def _load_caox_cap_head():
    """The CaOx-vs-CaP specialist head (HU+labs); None if not present."""
    global _CAOX_CAP
    if _CAOX_CAP is False:
        import joblib
        path = os.path.join(C.MODEL_DIR, "drstone_caox_cap_head.pkl")
        _CAOX_CAP = joblib.load(path) if os.path.exists(path) else None
    return _CAOX_CAP


def _refine_calcium(dist: list, vals: dict) -> tuple:
    """Re-partition the combined CaOx+CaP probability mass using the specialist
    head (a better calcium-subtype estimator than the 5-class model), leaving
    UA/Struvite/Other untouched. Returns (new_dist, info|None)."""
    head = _load_caox_cap_head()
    if head is None:
        return dist, None
    p = {d["type"]: d["p"] for d in dist}
    if "CaOx" not in p or "CaP" not in p:
        return dist, None
    mass = p["CaOx"] + p["CaP"]
    if mass <= 1e-6:
        return dist, None
    X = pd.DataFrame([[vals.get(f, np.nan) for f in head["features"]]],
                     columns=head["features"])
    cls = list(head["model"].classes_)
    j = cls.index(1) if 1 in cls else (len(cls) - 1)   # P(CaP); positive class = 1
    q_cap = float(head["model"].predict_proba(X)[0, j])
    before = (p["CaOx"], p["CaP"])
    p["CaP"] = mass * q_cap
    p["CaOx"] = mass * (1.0 - q_cap)
    new = sorted([{"type": t, "p": v} for t, v in p.items()], key=lambda d: -d["p"])
    # High-sensitivity CaP screen: fire only when calcium is a real contender, so
    # a clearly UA/struvite stone doesn't trigger a spurious CaP work-up.
    thr = float(head.get("threshold", 0.5))
    screen = bool(q_cap >= thr and mass >= CAP_ALERT_MIN_MASS)
    info = {"applied": True, "p_cap_given_calcium": q_cap,
            "calcium_mass": mass, "before": {"CaOx": before[0], "CaP": before[1]},
            "after": {"CaOx": p["CaOx"], "CaP": p["CaP"]},
            "cap_screen_positive": screen, "threshold": thr,
            "target_sens": head.get("threshold_target_sens")}
    return new, info


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
    # Cascade: defer the calcium-oxalate-vs-phosphate split to the specialist head.
    dist, calcium = _refine_calcium(dist, vals)
    top = [d["type"] for d in dist[:2] if d["p"] >= 0.15] or [dist[0]["type"]]
    diam = _num(form.get("stone_size_mm"))
    ac = acute(diameter_mm=(None if (isinstance(diam, float) and np.isnan(diam)) else diam),
               location=form.get("location", ""), creatinine=labs["creatinine"],
               infection=str(form.get("infection", "")).lower() in ("1", "true", "yes", "on"))
    return {"distribution": dist, "top": top, "acute": ac,
            "prevention": prevention(top, labs), "draft": DRAFT_NOTICE,
            "calcium_refined": bool(calcium),
            "cap_screen": (calcium if (calcium and calcium.get("cap_screen_positive")) else None),
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
