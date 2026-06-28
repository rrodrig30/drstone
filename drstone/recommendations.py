"""Guideline-based recommendation engine (acute + prevention).

This is a DETERMINISTIC, guideline-cited knowledge layer — not a model. It turns
(predicted composition distribution + measured stone size/location + metabolic
context) into actionable guidance.

>>> DRAFT CLINICAL CONTENT — requires review and sign-off by the supervising
>>> urologist before any clinical use. Decision support / patient education only;
>>> does not replace clinical judgment, urology consultation, stone analysis, or a
>>> 24-hour urine metabolic evaluation. Sources: AUA Medical Management of Kidney
>>> Stones (2014, amended 2019); AUA Surgical Management of Stones (2016);
>>> EAU Urolithiasis Guidelines.
"""

from __future__ import annotations

import numpy as np

DRAFT_NOTICE = ("DRAFT — for clinician review. Decision support/education only; "
                "confirm with stone analysis, 24-hour urine, and urology as indicated.")

CITES = ("AUA Medical Mgmt of Kidney Stones 2014/2019; AUA Surgical Mgmt of Stones "
         "2016; EAU Urolithiasis Guidelines.")


# --------------------------------------------------------------------------
# Acute: medical expulsive therapy (MET) vs. intervention
# --------------------------------------------------------------------------
def acute(diameter_mm=None, location="", creatinine=None, infection=False):
    """Acute management guidance from stone size + location (+ red flags)."""
    loc = (location or "").lower()
    out = {"headline": "", "tier": "info", "details": [], "redflags": [], "cite": CITES}

    # Red flags (escalate regardless of size)
    if infection:
        out["redflags"].append("Suspected obstructing infected stone (fever/pyuria/↑WBC) — "
                                "UROLOGIC EMERGENCY: urgent decompression (ureteral stent or "
                                "percutaneous nephrostomy) + IV antibiotics. Do not delay.")
    if creatinine is not None and creatinine == creatinine and creatinine > 1.5:
        out["redflags"].append(f"Elevated creatinine ({creatinine:.1f}) — assess for "
                               "obstructive AKI / solitary or bilateral obstruction; urgent urology.")

    d = diameter_mm if (diameter_mm is not None and diameter_mm == diameter_mm) else None
    is_ureter = "ureter" in loc
    is_bladder = "bladder" in loc
    is_kidney = (("kidney" in loc) or ("renal" in loc)) and not is_ureter and not is_bladder

    if d is None:
        out["headline"] = "Stone size not measured — size + location determine MET vs. intervention."
        out["details"].append("Detect the stone (or enter max diameter and location) for size-based guidance.")
        return out

    if is_ureter or (not is_kidney and not is_bladder):
        if d <= 10:
            out["tier"] = "met"
            out["headline"] = f"Ureteral stone {d:.0f} mm — candidate for medical expulsive therapy (MET)."
            out["details"] = [
                "Offer MET: an α-blocker (e.g., tamsulosin 0.4 mg daily) improves passage of "
                "distal ureteral stones, especially 5–10 mm.",
                "Analgesia (NSAID first-line if renal function permits, ± opioid), antiemetic, hydration.",
                "Observe up to ~4–6 weeks for spontaneous passage; strain urine and send the stone for analysis.",
                "Stones ≤5 mm pass spontaneously in the majority; counsel expectantly.",
                "Return precautions: fever/chills, intractable pain or vomiting, or decreased urine output.",
            ]
        else:
            out["tier"] = "intervention"
            out["headline"] = f"Ureteral stone {d:.0f} mm — spontaneous passage unlikely; urology referral."
            out["details"] = [
                "Stones >10 mm have low spontaneous-passage rates — MET is not a substitute.",
                "Definitive options: ureteroscopy (URS, generally preferred) or shock-wave lithotripsy "
                "(SWL) in selected cases.",
            ]
    elif is_kidney:
        if d <= 10:
            out["tier"] = "surveillance"
            out["headline"] = f"Renal stone {d:.0f} mm — surveillance vs. treatment by symptoms/preference."
            out["details"] = [
                "Small asymptomatic renal stones (esp. lower pole) may be observed with periodic imaging.",
                "Treat if symptomatic, growing, obstructing, or per informed patient preference.",
            ]
        else:
            out["tier"] = "intervention"
            out["headline"] = f"Renal stone {d:.0f} mm — urology referral for definitive treatment."
            out["details"] = [
                "Options by size/anatomy: SWL or URS for ~10–20 mm; percutaneous nephrolithotomy "
                "(PCNL) for >20 mm, staghorn, or complex lower-pole stones.",
            ]
    elif is_bladder:
        out["tier"] = "intervention"
        out["headline"] = f"Bladder stone {d:.0f} mm — cystolitholapaxy; evaluate bladder outlet obstruction."
        out["details"] = ["Treat the underlying cause (e.g., BPH, retention, catheter)."]
    return out


# --------------------------------------------------------------------------
# Prevention: diet / meds / lifestyle by stone type (+ metabolic flags)
# --------------------------------------------------------------------------
PREVENTION = {
    "CaOx": {
        "label": "Calcium oxalate",
        "diet": [
            "Fluids: ≥2.5–3 L/day to achieve urine output >2.5 L/day (single most effective measure).",
            "Reduce sodium (<2,300 mg/day) — high sodium raises urinary calcium.",
            "NORMAL dietary calcium (1,000–1,200 mg/day) — do NOT restrict; low calcium increases oxalate absorption.",
            "Limit oxalate-rich foods if hyperoxaluric (spinach, rhubarb, nuts, beets, chocolate, tea).",
            "Moderate non-dairy animal protein; increase fruits/vegetables (citrate, potassium).",
        ],
        "meds": [
            "Thiazide diuretic for recurrent stones with hypercalciuria.",
            "Potassium citrate for hypocitraturia or low urine pH.",
            "Allopurinol for recurrent CaOx with hyperuricosuria and normal urine calcium.",
        ],
        "lifestyle": ["Weight management; DASH-style diet; avoid high-dose vitamin C supplements; "
                      "evaluate/treat bowel disease (enteric hyperoxaluria)."],
    },
    "CaP": {
        "label": "Calcium phosphate",
        "diet": ["Fluids ≥2.5–3 L/day; reduce sodium; normal dietary calcium."],
        "meds": [
            "Thiazide for hypercalciuria.",
            "Potassium citrate with CAUTION — it raises urine pH and can worsen calcium-phosphate "
            "stones; monitor urine pH.",
            "Evaluate and treat distal renal tubular acidosis and primary hyperparathyroidism.",
        ],
        "lifestyle": ["Work up secondary causes (dRTA, hyperparathyroidism, UTI); avoid over-alkalinization."],
    },
    "UA": {
        "label": "Uric acid",
        "diet": [
            "Fluids ≥2.5–3 L/day.",
            "Reduce purine-rich foods (red/organ meats, shellfish) and overall animal protein.",
            "Limit fructose and alcohol; increase fruits/vegetables.",
        ],
        "meds": [
            "Potassium citrate to alkalinize urine to pH 6.5–7.0 — often DISSOLVES existing uric-acid stones.",
            "Allopurinol if hyperuricemia/hyperuricosuria or refractory despite alkalinization.",
        ],
        "lifestyle": ["Weight loss; manage diabetes/metabolic syndrome and gout."],
    },
    "Struvite": {
        "label": "Struvite (infection stone)",
        "diet": ["Not diet-driven — prevention is infection control and complete stone clearance."],
        "meds": [
            "Culture-directed antibiotics to eradicate the urease-producing UTI.",
            "Acetohydroxamic acid (urease inhibitor) only in select refractory cases (toxicity-limited).",
        ],
        "lifestyle": [
            "COMPLETE surgical removal is the cornerstone — residual fragments harbor infection and regrow.",
            "Address predisposing factors: obstruction, indwelling catheters, neurogenic bladder.",
            "Urology referral.",
        ],
    },
    "Other": {
        "label": "Other / uncommon (incl. cystine)",
        "diet": ["Fluids — for suspected cystine, very high intake (>3–4 L/day); reduce sodium and animal protein."],
        "meds": ["For cystine: urinary alkalinization to pH >7; tiopronin or penicillamine in refractory cases."],
        "lifestyle": ["Consider cystine in young/recurrent stone formers (genetic); send stone analysis; "
                      "genetic/metabolic referral."],
    },
}


def metabolic_flags(labs: dict) -> list:
    """Notable spot-lab patterns (with the caveat that 24-h urine is definitive)."""
    f = []

    def num(k):
        try:
            x = float(labs.get(k))
            return x if x == x else None
        except (TypeError, ValueError):
            return None
    ph, co2, cl, na = num("urine_ph"), num("co2"), num("cl"), num("na")
    ca, glu, cr = num("ca"), num("glucose"), num("creatinine")
    if ph is not None and ph < 5.5:
        f.append("Persistently acidic urine (pH <5.5) — favors uric-acid stones.")
    if ph is not None and ph >= 6.8:
        f.append("Alkaline urine (pH ≥6.8) — consider calcium-phosphate or infection (struvite).")
    if co2 is not None and cl is not None and na is not None and co2 < 22 and (na - cl - co2) <= 10:
        f.append("Low bicarbonate with normal anion gap — possible distal RTA (calcium-phosphate risk).")
    if ca is not None and ca > 10.3:
        f.append("Elevated serum calcium — evaluate for primary hyperparathyroidism.")
    if glu is not None and glu > 140:
        f.append("Hyperglycemia — diabetes/insulin resistance lowers urine pH (uric-acid risk).")
    if cr is not None and cr > 1.5:
        f.append("Elevated creatinine — assess renal function/obstruction.")
    return f


def prevention(top_types, labs: dict) -> dict:
    """Prevention guidance for the most-likely type(s) + metabolic flags."""
    blocks = []
    for t in top_types:
        p = PREVENTION.get(t, PREVENTION["Other"])
        blocks.append({"type": t, "label": p["label"], "diet": p["diet"],
                       "meds": p["meds"], "lifestyle": p["lifestyle"]})
    return {
        "universal": "Universal: increase fluids to achieve >2.5 L urine/day; this benefits all stone types.",
        "blocks": blocks,
        "flags": metabolic_flags(labs),
        "workup": ("Obtain stone analysis (if a stone is retrieved/passed) and, for recurrent or "
                   "high-risk formers, a 24-hour urine (volume, calcium, oxalate, citrate, uric acid, "
                   "sodium, pH, supersaturations) — the definitive metabolic evaluation that spot labs "
                   "cannot replace."),
        "cite": CITES,
    }
