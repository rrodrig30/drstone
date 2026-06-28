"""Patient lookup (research / retrospective): MRN -> routine labs + curated CT path.

Auto-fills the predictor from the project tables — the same pattern a FHIR/HL7
adapter would use in production (pull labs + demographics by patient identifier).
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from drstone import config as C

_CACHE = None


def _norm(mrn) -> str:
    s = str(mrn).strip()
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s


def _load():
    global _CACHE
    if _CACHE is None:
        stones = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"),
                             dtype={"canonical_mrn": str})
        labs = (stones[["canonical_mrn", "urine_ph", "na", "k", "cl", "co2", "bun",
                        "creatinine", "ca", "glucose", "age", "gender"]]
                .drop_duplicates("canonical_mrn").set_index("canonical_mrn"))
        try:
            link = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"),
                               dtype={"canonical_mrn": str})
            dpath = (link.dropna(subset=["curated_dir"])
                     .drop_duplicates("canonical_mrn")
                     .set_index("canonical_mrn")["curated_dir"])
        except Exception:
            dpath = pd.Series(dtype=str)
        _CACHE = (labs, dpath)
    return _CACHE


def lookup(mrn: str) -> dict:
    labs, dpath = _load()
    key = _norm(mrn)
    if key not in labs.index:
        return {"found": False}
    r = labs.loc[key]
    if isinstance(r, pd.DataFrame):
        r = r.iloc[0]

    def g(col, ndigits=1):
        v = r[col]
        return None if pd.isna(v) else round(float(v), ndigits)

    sex = str(r["gender"]).upper()
    out = {"found": True, "labs": {
        "urine_ph": g("urine_ph"), "na": g("na", 0), "cl": g("cl", 0), "co2": g("co2", 0),
        "bun": g("bun", 0), "creatinine": g("creatinine"), "ca": g("ca"),
        "glucose": g("glucose", 0), "age": g("age", 0),
        "sex": "Male" if sex.startswith("M") else ("Female" if sex.startswith("F") else "")}}
    cur = dpath.get(key) if key in getattr(dpath, "index", []) else None
    out["dicom_path"] = cur if (cur and os.path.isdir(str(cur))) else None
    return out
