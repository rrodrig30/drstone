"""Derived acid-base / renal features from the existing BMP (no new labs).

These give the tree representational power it can't synthesize from axis-aligned
splits on the raw values: a hyperchloremia index, an explicit normal-anion-gap
(hyperchloremic) metabolic-acidosis composite — the distal-RTA / GI-bicarbonate-
loss signature that drives calcium-phosphate / calcium-oxalate stones — and
CKD-EPI eGFR.
"""

from __future__ import annotations

import numpy as np

DERIVED = ["cl_minus_co2", "nagma_score", "egfr"]


def egfr_ckdepi_2021(scr, age, gender_m):
    """CKD-EPI 2021 (race-free) eGFR; NaN if any input missing.
    scr mg/dL, age yr, gender_m 1=male/0=female."""
    try:
        scr = float(scr); age = float(age); gm = float(gender_m)
    except (TypeError, ValueError):
        return np.nan
    if np.isnan(scr) or np.isnan(age) or np.isnan(gm) or scr <= 0:
        return np.nan
    female = gm < 0.5
    kappa = 0.7 if female else 0.9
    alpha = -0.241 if female else -0.302
    r = scr / kappa
    egfr = (142 * (min(r, 1.0) ** alpha) * (max(r, 1.0) ** -1.200)
            * (0.9938 ** age) * (1.012 if female else 1.0))
    return float(egfr)


def add_derived(df):
    """Append DERIVED columns; expects cl, co2, anion_gap, creatinine, age,
    gender_M present (as in the modeling tables)."""
    out = df.copy()
    out["cl_minus_co2"] = out["cl"] - out["co2"]
    out["nagma_score"] = ((24.0 - out["co2"]).clip(lower=0)
                          - (out["anion_gap"] - 12.0).clip(lower=0))
    out["egfr"] = [egfr_ckdepi_2021(c, a, g) for c, a, g
                   in zip(out["creatinine"], out["age"], out["gender_M"])]
    return out


def derived_from_vals(vals: dict) -> dict:
    """Same three features from a single inference `vals` dict (predict path)."""
    cl, co2, ag = vals.get("cl"), vals.get("co2"), vals.get("anion_gap")
    def _f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan
    cl, co2, ag = _f(cl), _f(co2), _f(ag)
    nagma = (max(0.0, 24.0 - co2) - max(0.0, ag - 12.0)
             if not (np.isnan(co2) or np.isnan(ag)) else np.nan)
    return {"cl_minus_co2": (cl - co2) if not (np.isnan(cl) or np.isnan(co2)) else np.nan,
            "nagma_score": nagma,
            "egfr": egfr_ckdepi_2021(vals.get("creatinine"), vals.get("age"),
                                     vals.get("gender_M"))}
