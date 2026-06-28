"""Curate the non-contrast axial CT series for the Dr Stone cohort.

Indexes the extracted DICOM archive (one header read per series), resolves each
series to a canonical UT MRN, classifies series as non-contrast / ORIGINAL /
axial / volume, selects the best non-contrast axial series per patient (kernel-
and thickness-aware), joins it to the per-stone analytic table, and copies the
selected series into a clean curated dataset on the Data drive.

Stones do not require contrast: only non-contrast CT is curated.

Run:
    python -m drstone.match_dicom            # index, select, link, COPY
    python -m drstone.match_dicom --no-copy  # tables only (no file copy)

Outputs (drstone_data/):
    drstone_series_index.csv     every CT series found + classification (audit)
    drstone_linkage.csv          per-stone -> selected non-contrast series + path
    drstone_curation_report.txt  counts / yield / exclusions
Curated DICOM copied under config.CURATED_DIR/<canonical_mrn>/<acc>_<series>/.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import math
import collections

import numpy as np
import pandas as pd
import pydicom

from drstone import config as C
from drstone.data_ingest import norm_mrn, build_crossover_map, to_canonical


# --------------------------------------------------------------------------
# Series-level metadata helpers
# --------------------------------------------------------------------------
def _is_dicom_file(name: str) -> bool:
    base = name.rsplit("/", 1)[-1]
    return base.lower().endswith(".dcm") or "." not in base


def _desc_has(desc: str, tokens) -> bool:
    """True if any token appears as a whole word/phrase (word-boundary match).

    Word-boundary matching is essential: a naive substring check for 'CE' would
    falsely fire on 'SPACE', 'SURFACE', etc., wrongly flagging non-contrast
    series as contrast.
    """
    d = desc.upper()
    for t in tokens:
        tt = t.strip().upper()
        if not tt:
            continue
        if re.search(r"(?<![A-Z0-9])" + re.escape(tt) + r"(?![A-Z0-9])", d):
            return True
    return False


def _is_axial(iop) -> bool:
    """Axial if the slice normal aligns with patient z (cross of row/col cosines)."""
    try:
        v = [float(x) for x in iop]
        if len(v) != 6:
            return False
        r, c = np.array(v[:3]), np.array(v[3:])
        n = np.cross(r, c)
        return abs(n[2]) >= C.AXIAL_NORMAL_TOL
    except (TypeError, ValueError):
        return False


def _kernel_is_soft(kernel: str) -> bool:
    k = (kernel or "").upper()
    if not k:
        return False
    return not any(h in k for h in C.SHARP_KERNEL_HINTS)


def index_series(root: str) -> pd.DataFrame:
    """Walk the archive; one representative header per series folder."""
    # Group candidate DICOM files by their containing directory (== series).
    series_files = collections.defaultdict(list)
    for dp, _dirs, files in os.walk(root):
        for f in files:
            if _is_dicom_file(f):
                series_files[dp].append(f)
    print(f"  series folders: {len(series_files):,}")

    rows = []
    for sdir, files in series_files.items():
        files_sorted = sorted(files)
        rep = os.path.join(sdir, files_sorted[len(files_sorted) // 2])  # middle slice
        try:
            ds = pydicom.dcmread(rep, stop_before_pixels=True, force=True)
        except Exception:
            continue
        rows.append({
            "series_dir": sdir,
            "n_instances": len(files),
            "patient_id": norm_mrn(getattr(ds, "PatientID", "")),
            "accession": str(getattr(ds, "AccessionNumber", "")).strip(),
            "series_uid": str(getattr(ds, "SeriesInstanceUID", "")),
            "modality": str(getattr(ds, "Modality", "")),
            "image_type0": (str(ds.ImageType[0]).upper()
                            if getattr(ds, "ImageType", None) else ""),
            "series_desc": str(getattr(ds, "SeriesDescription", "")),
            "contrast_agent": str(getattr(ds, "ContrastBolusAgent", "")).strip(),
            "iop": list(getattr(ds, "ImageOrientationPatient", []) or []),
            "slice_thickness": _to_float(getattr(ds, "SliceThickness", None)),
            "kvp": _to_float(getattr(ds, "KVP", None)),
            "kernel": str(getattr(ds, "ConvolutionKernel", "") or ""),
            "rows": _to_int(getattr(ds, "Rows", None)),
            "cols": _to_int(getattr(ds, "Columns", None)),
            "study_date": str(getattr(ds, "StudyDate", "")),
        })
    return pd.DataFrame(rows)


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def _to_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------
# Classification + patient resolution + selection
# --------------------------------------------------------------------------
def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_ct"] = df["modality"] == "CT"
    df["is_original"] = df["image_type0"] == "ORIGINAL"
    df["is_axial"] = df["iop"].map(_is_axial)
    df["contrast"] = (df["contrast_agent"].str.len() > 0) | \
                     df["series_desc"].map(lambda s: _desc_has(s, C.CONTRAST_DESC_TOKENS))
    df["is_noncontrast"] = ~df["contrast"]
    df["is_volume"] = df["n_instances"] >= C.MIN_AXIAL_SLICES
    df["soft_kernel"] = df["kernel"].map(_kernel_is_soft)
    # Reject non-image / rendered / report objects (or anything without pixel dims).
    df["nonimage"] = (df["series_desc"].map(lambda s: _desc_has(s, C.EXCLUDE_NONIMAGE_TOKENS))
                      | (df["image_type0"] == "SECONDARY")
                      | df["rows"].isna() | df["cols"].isna())
    # A usable non-contrast volume (HU is orientation-independent, so reformats
    # count) — quality is graded by tier, not by inclusion.
    df["usable"] = (df["is_ct"] & df["is_noncontrast"] & df["is_volume"] & ~df["nonimage"])
    # Quality tier (lower rank = better): axial native > axial reformat > non-axial reformat
    def tier(r):
        if not r["usable"]:
            return ("Z_excluded", 9)
        if r["is_axial"] and r["is_original"]:
            return ("A_axial_original", 0)
        if r["is_axial"]:
            return ("B_axial_derived", 1)
        return ("C_reformat", 2)
    tiers = df.apply(tier, axis=1)
    df["quality_tier"] = [t[0] for t in tiers]
    df["tier_rank"] = [t[1] for t in tiers]
    return df


def resolve_patients(df: pd.DataFrame, workbook: str) -> pd.DataFrame:
    s1 = pd.read_excel(workbook, C.SHEET_CLINICAL)
    ut = set(s1["UT MRN"].map(norm_mrn)) - {""}
    uh = set(s1["UHS MRN"].map(norm_mrn)) - {""}
    xmap = build_crossover_map(pd.ExcelFile(workbook))
    # accession -> canonical UT MRN
    acc_map = {}
    for _, r in s1.iterrows():
        a = str(r.get("UHS Accession#", "")).strip()
        if a and a.lower() != "nan":
            canon = norm_mrn(r.get("UT MRN")) or to_canonical(norm_mrn(r.get("UHS MRN")), xmap)
            if canon:
                acc_map[a] = canon

    def canon_for(pid, acc):
        if pid in ut:
            return pid
        if pid in uh:
            return to_canonical(pid, xmap)
        if acc and acc in acc_map:
            return acc_map[acc]
        return to_canonical(pid, xmap)  # best effort via crossover

    df = df.copy()
    df["canonical_mrn"] = [canon_for(p, a) for p, a in zip(df["patient_id"], df["accession"])]
    return df


def _selection_score(row) -> tuple:
    """Higher is better. Prefer (in order): better quality tier (axial native >
    axial reformat > reformat) > soft kernel (HU accuracy) > thinner slice >
    more instances > standard 512 matrix."""
    st = row["slice_thickness"]
    thin = -(st if (isinstance(st, float) and not math.isnan(st)) else 99.0)
    mat = 1 if row["cols"] == 512 else 0
    return (-row["tier_rank"], 1 if row["soft_kernel"] else 0, thin, row["n_instances"], mat)


def select_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    usable = df[df["usable"] & (df["canonical_mrn"] != "")].copy()
    if usable.empty:
        return usable.assign(selected=False)
    usable["score"] = usable.apply(_selection_score, axis=1)
    usable = usable.sort_values("score", ascending=False)
    sel = usable.drop_duplicates("canonical_mrn", keep="first").copy()
    sel["selected"] = True
    return sel


# --------------------------------------------------------------------------
# Linkage + copy
# --------------------------------------------------------------------------
def build_linkage(stones_csv: str, selected: pd.DataFrame) -> pd.DataFrame:
    stones = pd.read_csv(stones_csv, dtype={"canonical_mrn": str})
    selcols = ["canonical_mrn", "series_dir", "series_uid", "accession",
               "quality_tier", "is_axial", "slice_thickness", "kvp", "kernel",
               "soft_kernel", "n_instances", "rows", "cols", "study_date"]
    if "curated_dir" in selected.columns:
        selcols.append("curated_dir")
    sel = selected[selcols].rename(columns={
        "series_dir": "src_series_dir", "accession": "ct_accession",
        "n_instances": "ct_n_slices"})
    link = stones.merge(sel, on="canonical_mrn", how="left")
    link["has_noncontrast_ct"] = link["src_series_dir"].notna()
    link["no_image_reason"] = np.where(
        link["has_noncontrast_ct"], "",
        np.where(link["has_ct"].fillna(False), "no_noncontrast_axial_series", "no_imaging_folder"))
    return link


def _safe_series_token(uid: str) -> str:
    return (uid.split(".")[-1] or "series")[-10:]


def curate_copy(selected: pd.DataFrame, curated_dir: str, limit: int = None) -> pd.DataFrame:
    os.makedirs(curated_dir, exist_ok=True)
    dests = []
    rows = selected.itertuples(index=False)
    n = 0
    for r in rows:
        if limit and n >= limit:
            dests.append("")
            continue
        mrn = r.canonical_mrn
        token = f"{(r.accession or 'NA')}_{_safe_series_token(r.series_uid)}"
        dest = os.path.join(curated_dir, str(mrn), token)
        src = r.series_dir
        try:
            src_files = [f for f in os.listdir(src) if _is_dicom_file(f)]
            if os.path.isdir(dest) and len(os.listdir(dest)) >= len(src_files):
                dests.append(dest)  # already copied (idempotent)
                n += 1
                continue
            os.makedirs(dest, exist_ok=True)
            for f in src_files:
                shutil.copy2(os.path.join(src, f), os.path.join(dest, f))
            dests.append(dest)
        except Exception as e:
            print(f"  copy failed for {mrn}: {e}")
            dests.append("")
        n += 1
        if n % 25 == 0:
            print(f"    copied {n} series...")
    out = selected.copy()
    out["curated_dir"] = dests
    return out


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def write_report(idx, selected, link, path):
    cc = link[link["complete_case"] == True]
    lines = []
    L = lines.append
    L("Dr Stone — DICOM curation report")
    L("=" * 50)
    L(f"CT series indexed:                  {int((idx['is_ct']).sum())}")
    L(f"  non-contrast:                     {int((idx['is_ct'] & idx['is_noncontrast']).sum())}")
    L(f"  ORIGINAL axial volume (usable):   {int(idx['usable'].sum())}")
    L(f"Patients with a usable NC series:   {selected['canonical_mrn'].nunique()}")
    L("")
    L(f"Stones (rows in analytic table):    {len(link)}")
    L(f"  with non-contrast CT linked:      {int(link['has_noncontrast_ct'].sum())}")
    L(f"Complete-case stones:               {len(cc)}")
    L(f"  with non-contrast CT linked:      {int(cc['has_noncontrast_ct'].sum())}")
    L(f"  patients (complete-case+NC CT):   {cc.loc[cc['has_noncontrast_ct'],'canonical_mrn'].nunique()}")
    L("")
    L("Selected-series quality tier:")
    for k, v in selected["quality_tier"].value_counts().items():
        L(f"   {k}: {v}")
    L("Selected-series slice thickness (mm):")
    for k, v in selected["slice_thickness"].round(1).value_counts().sort_index().items():
        L(f"   {k:>4} mm : {v}")
    L("Selected-series kernel softness:")
    L(f"   soft/standard: {int(selected['soft_kernel'].sum())} | sharp/unknown: {int((~selected['soft_kernel']).sum())}")
    L("Reasons for no NC image (stones):")
    for k, v in link.loc[~link["has_noncontrast_ct"], "no_image_reason"].value_counts().items():
        L(f"   {k}: {v}")
    txt = "\n".join(lines)
    open(path, "w").write(txt)
    print("\n" + txt)


def main():
    do_copy = "--no-copy" not in sys.argv
    limit = None
    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])

    print(f"Indexing DICOM archive: {C.DICOM_ROOT}")
    idx = index_series(C.DICOM_ROOT)
    idx = classify(idx)
    idx = resolve_patients(idx, C.WORKBOOK)
    print(f"  CT series: {int(idx['is_ct'].sum())} | usable NC axial: {int(idx['usable'].sum())}")

    selected = select_per_patient(idx)
    print(f"  patients with a selected NC series: {len(selected)}")

    # Restrict the curated copy to cohort (labeled) patients.
    stones = pd.read_csv(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"),
                         dtype={"canonical_mrn": str})
    cohort = set(stones["canonical_mrn"])
    sel_cohort = selected[selected["canonical_mrn"].isin(cohort)].copy()
    print(f"  of which in the labeled cohort: {len(sel_cohort)}")

    if do_copy:
        print(f"Copying curated non-contrast series -> {C.CURATED_DIR}")
        sel_cohort = curate_copy(sel_cohort, C.CURATED_DIR, limit=limit)
        # carry curated_dir back onto the full selected frame for the linkage
        selected = selected.merge(
            sel_cohort[["canonical_mrn", "curated_dir"]], on="canonical_mrn", how="left")

    link = build_linkage(os.path.join(C.OUTPUT_DIR, "drstone_stones.csv"), selected)

    idx.drop(columns=["iop"]).to_csv(os.path.join(C.OUTPUT_DIR, "drstone_series_index.csv"), index=False)
    link.to_csv(os.path.join(C.OUTPUT_DIR, "drstone_linkage.csv"), index=False)
    write_report(idx, selected, link, os.path.join(C.OUTPUT_DIR, "drstone_curation_report.txt"))
    print(f"\nWrote series index, linkage table, and report to {C.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
