"""Phase 0 data engineering for Dr Stone.

Joins the project workbook into a clean per-stone analytic table:
  Sheet1 (clinical/labs)  ↔  COMP (composition labels)  ↔  MRN crossover
all consolidated on the canonical UT MRN, with composition normalized to
fractions + parent classes, the <120-day preop CT window applied, and the
single-stone primary cohort flagged.

Run:
    python -m drstone.data_ingest
Outputs (in drstone_data/):
    drstone_patients.csv   one row per patient (cleaned clinical/labs)
    drstone_stones.csv     one row per analyzed stone (labels + joined clinical)
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

from drstone import config as C

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------
# MRN handling
# --------------------------------------------------------------------------
def norm_mrn(x) -> str:
    """Normalize an MRN cell to a bare integer string ('' if unusable)."""
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s == "" or s.lower() in ("nan", "mrn"):
        return ""
    try:
        return str(int(float(s)))
    except (ValueError, TypeError):
        return s  # keep alphanumeric MRNs as-is


def build_crossover_map(xls: pd.ExcelFile) -> dict:
    """Map any site MRN -> canonical UT MRN, from the crossover sheet + Sheet1."""
    xmap: dict = {}
    # From the dedicated crossover sheet
    try:
        cx = pd.read_excel(xls, C.SHEET_CROSSOVER)
        ut_col = "UT MRN" if "UT MRN" in cx.columns else None
        uh_col = "UH MRN" if "UH MRN" in cx.columns else None
        first = cx.columns[0]
        for _, r in cx.iterrows():
            ut = norm_mrn(r.get(ut_col)) if ut_col else ""
            if not ut:
                continue
            for other in (r.get(first), r.get(uh_col) if uh_col else None):
                o = norm_mrn(other)
                if o:
                    xmap[o] = ut
            xmap[ut] = ut
    except Exception as e:  # crossover sheet optional
        print(f"  (crossover sheet not usable: {e})")
    # From Sheet1's own UT/UHS pairing
    s1 = pd.read_excel(xls, C.SHEET_CLINICAL)
    if "UT MRN" in s1.columns and "UHS MRN" in s1.columns:
        for _, r in s1.iterrows():
            ut = norm_mrn(r.get("UT MRN"))
            uh = norm_mrn(r.get("UHS MRN"))
            if ut:
                xmap.setdefault(ut, ut)
                if uh:
                    xmap.setdefault(uh, ut)
    return xmap


def to_canonical(mrn, xmap: dict) -> str:
    m = norm_mrn(mrn)
    return xmap.get(m, m)


# --------------------------------------------------------------------------
# Clinical / labs
# --------------------------------------------------------------------------
def load_clinical(xls: pd.ExcelFile, xmap: dict) -> pd.DataFrame:
    df = pd.read_excel(xls, C.SHEET_CLINICAL)
    df["canonical_mrn"] = df["UT MRN"].map(lambda x: to_canonical(x, xmap))
    # If UT MRN missing, fall back to UHS MRN via crossover
    miss = df["canonical_mrn"] == ""
    df.loc[miss, "canonical_mrn"] = df.loc[miss, "UHS MRN"].map(lambda x: to_canonical(x, xmap))

    # Gender normalization
    df["gender"] = (df["Gender"].astype(str).str.strip().str.lower()
                    .map({"female": "F", "male": "M"}))
    # Dates
    for c in ["DOS", "CT DOS", "DOB", "Date of lab"]:
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["age"] = (df["CT DOS"] - df["DOB"]).dt.days / 365.25
    # Numeric labs with plausibility nulling
    for c in C.LAB_NUMERIC:
        v = pd.to_numeric(df[c], errors="coerce")
        lo, hi = C.LAB_VALID_RANGE.get(c, (-np.inf, np.inf))
        v = v.where((v >= lo) & (v <= hi))
        df[c.lower().replace(" ", "_")] = v
    df["accession"] = df["UHS Accession#"]
    df["has_ct"] = df["accession"].notna()
    # Preop delta (days from CT to surgery): positive = CT before surgery
    df["preop_delta_days"] = (df["DOS"] - df["CT DOS"]).dt.days
    df["preop_ok"] = df["preop_delta_days"].between(-C.PREOP_AFTER_GRACE_DAYS, C.PREOP_WINDOW_DAYS)

    keep = ["canonical_mrn", "gender", "age", "accession", "has_ct",
            "DOS", "CT DOS", "Date of lab", "preop_delta_days", "preop_ok"] + \
           [c.lower().replace(" ", "_") for c in C.LAB_NUMERIC]
    df = df[df["canonical_mrn"] != ""][keep].copy()

    # One row per patient: prefer has_ct, then preop_ok, then most-complete labs
    lab_cols = [c.lower().replace(" ", "_") for c in C.LAB_NUMERIC]
    df["completeness"] = df[lab_cols].notna().sum(axis=1)
    df["score"] = (df["has_ct"].astype(int) * 100
                   + df["preop_ok"].fillna(False).astype(int) * 50
                   + df["completeness"])
    df = (df.sort_values("score", ascending=False)
            .drop_duplicates("canonical_mrn", keep="first")
            .drop(columns=["completeness", "score"]))
    return df


# --------------------------------------------------------------------------
# Composition labels
# --------------------------------------------------------------------------
def load_composition(xls: pd.ExcelFile, xmap: dict) -> pd.DataFrame:
    comp = pd.read_excel(xls, C.SHEET_COMP)
    idcol = comp.columns[0]
    # Handle merged MRN cells (forward fill) and drop header-artifact rows.
    comp[idcol] = comp[idcol].where(comp[idcol].map(norm_mrn) != "")  # blank artifacts -> NaN
    comp[idcol] = comp[idcol].ffill()

    def block_cols(suffix):
        return [(f if not suffix else f"{f}{suffix}") for f in C.COMP_BLOCK_FIELDS]

    rows = []
    for suffix in ("", ".1", ".2"):
        cols = block_cols(suffix)
        present = [c for c in cols if c in comp.columns]
        if len(present) < len(C.COMPONENTS):
            continue
        sub = comp[[idcol] + cols].copy()
        sub.columns = ["mrn", "mass_mg", "size_mm"] + C.COMPONENTS
        comp_vals = sub[C.COMPONENTS].apply(pd.to_numeric, errors="coerce")
        keep = comp_vals.fillna(0).sum(axis=1) > 0          # a real stone block
        sub = sub[keep]
        for c in C.COMPONENTS:
            sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.0)
        sub["mass_mg"] = pd.to_numeric(sub["mass_mg"], errors="coerce")
        sub["size_mm"] = pd.to_numeric(sub["size_mm"], errors="coerce")
        rows.append(sub)

    st = pd.concat(rows, ignore_index=True)
    st["canonical_mrn"] = st["mrn"].map(lambda x: to_canonical(x, xmap))
    st = st[st["canonical_mrn"] != ""].copy()

    # Fractions over fine components
    tot = st[C.COMPONENTS].sum(axis=1)
    for c in C.COMPONENTS:
        st[f"f_{c}"] = st[c] / tot
    # Parent-class fractions
    for parent in C.PARENT_CLASSES:
        kids = [c for c, p in C.PARENT_MAP.items() if p == parent]
        st[f"p_{parent}"] = st[[f"f_{k}" for k in kids]].sum(axis=1)
    pcols = [f"p_{p}" for p in C.PARENT_CLASSES]
    # Derived labels
    st["dominant_parent"] = st[pcols].idxmax(axis=1).str[2:]
    st["n_components"] = (st[pcols] > C.PRESENCE_THRESHOLD).sum(axis=1)
    st["mixed"] = st["n_components"] >= 2
    st["ua_any"] = st["p_UA"] > C.PRESENCE_THRESHOLD
    for p in C.PARENT_CLASSES:
        st[f"has_{p}"] = st[f"p_{p}"] > C.PRESENCE_THRESHOLD
    # Modeling class with rare parents folded into Other
    st["modeling_class"] = st["dominant_parent"].where(
        ~st["dominant_parent"].isin([r for r in C.RARE_FOLD_TO_OTHER]), "Other")

    # Sequential stone number per patient
    st = st.sort_values(["canonical_mrn"]).reset_index(drop=True)
    st["stone_no"] = st.groupby("canonical_mrn").cumcount() + 1
    n_per = st.groupby("canonical_mrn")["stone_no"].transform("max")
    st["single_stone_patient"] = n_per == 1
    return st


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------
def build(xls_path: str = None) -> tuple:
    xls_path = xls_path or C.WORKBOOK
    xls = pd.ExcelFile(xls_path, engine="openpyxl")
    print(f"Loading workbook: {xls_path}")
    xmap = build_crossover_map(xls)
    print(f"  crossover map entries: {len(xmap)}")
    clinical = load_clinical(xls, xmap)
    stones = load_composition(xls, xmap)
    print(f"  clinical patients: {len(clinical)} | stones with composition: {len(stones)} "
          f"({stones['canonical_mrn'].nunique()} patients)")

    analytic = stones.merge(clinical, on="canonical_mrn", how="left")
    analytic["has_label"] = True
    analytic["analyzable"] = analytic["has_ct"] & analytic["preop_ok"].fillna(False)
    analytic["complete_case"] = (analytic["analyzable"]
                                 & analytic["hemoglobin"].notna()
                                 & analytic["urine_ph"].notna())
    return clinical, stones, analytic


def summarize(clinical, stones, analytic) -> None:
    print("\n================ Dr Stone — analytic cohort ================")
    print(f"Patients (clinical):                 {len(clinical)}")
    print(f"  with CT accession:                 {int(clinical['has_ct'].sum())}")
    print(f"Stones with composition:             {len(stones)}")
    print(f"  single-stone patients:             {int(stones['single_stone_patient'].sum())} stones")
    print(f"Stones with CT + preop window OK:    {int(analytic['analyzable'].sum())}")
    print(f"  + Hb + pH (complete-case):         {int(analytic['complete_case'].sum())}")
    print(f"  single-stone & complete-case:      {int((analytic['complete_case'] & analytic['single_stone_patient']).sum())}")
    print(f"\nMixed vs pure (parent, >{int(C.PRESENCE_THRESHOLD*100)}%): "
          f"{stones['mixed'].value_counts().to_dict()}  ({100*stones['mixed'].mean():.0f}% mixed)")
    print(f"Dominant parent class:\n  {stones['dominant_parent'].value_counts().to_dict()}")
    print(f"Component presence:\n  " +
          ", ".join(f"{p}={int(stones[f'has_{p}'].sum())}" for p in C.PARENT_CLASSES))
    print(f"UA-any: {int(stones['ua_any'].sum())} | UA-dominant: {int((stones['dominant_parent']=='UA').sum())}")
    pre = analytic.loc[analytic['analyzable'], 'preop_delta_days']
    if len(pre):
        print(f"Preop CT->surgery days (analyzable): median={pre.median():.0f} "
              f"IQR=[{pre.quantile(.25):.0f},{pre.quantile(.75):.0f}]")


def main():
    clinical, stones, analytic = build()
    summarize(clinical, stones, analytic)
    pp = os.path.join(C.OUTPUT_DIR, "drstone_patients.csv")
    sp = os.path.join(C.OUTPUT_DIR, "drstone_stones.csv")
    clinical.to_csv(pp, index=False)
    analytic.to_csv(sp, index=False)
    print(f"\nWrote:\n  {pp}\n  {sp}")


if __name__ == "__main__":
    main()
