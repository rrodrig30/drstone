"""Configuration for Dr Stone (standalone)."""

from __future__ import annotations

import os

# --- Paths -----------------------------------------------------------------
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_dotenv(path: str) -> None:
    """Minimal, dependency-free .env loader. Real environment variables take
    precedence (setdefault), so .env only fills in what isn't already set."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


# Load local secrets/config (git-ignored) before any os.environ.get below.
_load_dotenv(os.path.join(REPO_ROOT, ".env"))
# Processed tables + intermediate outputs (git-ignored).
DATA_DIR = os.environ.get("DRSTONE_DATA_DIR", os.path.join(REPO_ROOT, "data"))
OUTPUT_DIR = DATA_DIR
# Deployable model artifacts (tracked in git): locked model + results.
MODEL_DIR = os.environ.get("DRSTONE_MODEL_DIR", os.path.join(REPO_ROOT, "models"))
WORKBOOK = os.path.join(DATA_DIR, "Kidney stone project (2).xlsx")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Extracted DICOM archive (the four image collections live under this root).
DICOM_ROOT = os.environ.get(
    "DRSTONE_DICOM_ROOT",
    "/run/media/exx/Data/Rodriguez_Stone_DICOM/Rodriguez Stone Project",
)
# Curated, model-ready dataset (non-contrast axial series only) on the Data drive.
CURATED_DIR = os.environ.get(
    "DRSTONE_CURATED_DIR", "/run/media/exx/Data/DrStone_Curated"
)

# --- Spreadsheet structure -------------------------------------------------
SHEET_CLINICAL = "Sheet1"
SHEET_COMP = "COMP"
SHEET_CROSSOVER = "MRN crossover"
CANONICAL_KEY = "UT MRN"          # patients span both sites; consolidate on UT MRN

# Clinical/lab columns in Sheet1 (BMP + Hb + spot urine pH)
LAB_NUMERIC = ["Hemoglobin", "Na", "K", "Cl", "CO2", "BUN",
               "Creatinine", "Ca", "Glucose", "Urine pH"]
# Plausibility ranges to null-out data-entry errors (e.g., CO2=222).
LAB_VALID_RANGE = {
    "Hemoglobin": (3.0, 25.0), "Na": (110, 170), "K": (1.5, 8.0),
    "Cl": (70, 140), "CO2": (5, 50), "BUN": (1, 200),
    "Creatinine": (0.1, 25.0), "Ca": (4.0, 16.0), "Glucose": (20, 1000),
    "Urine pH": (4.0, 9.0),
}

# --- Composition labels (COMP sheet) ---------------------------------------
# Fine-grained components reported by stone analysis.
COMPONENTS = ["COM", "COD", "CONOS", "CPB", "CPHA", "CPNOS", "UA", "CYS", "MAP", "Other"]
COMP_BLOCK_FIELDS = ["Mass (mg)", "Size (mm)"] + COMPONENTS   # one stone block

# Parent-class grouping (rare subtypes folded into parents per design decision).
PARENT_MAP = {
    "COM": "CaOx", "COD": "CaOx", "CONOS": "CaOx",
    "CPB": "CaP", "CPHA": "CaP", "CPNOS": "CaP",
    "UA": "UA", "MAP": "Struvite", "CYS": "Cystine", "Other": "Other",
}
PARENT_CLASSES = ["CaOx", "CaP", "UA", "Struvite", "Cystine", "Other"]

# A component counts as "present" at or above this fraction.
PRESENCE_THRESHOLD = 0.05

# Cystine is very rare (~8 stones); optionally fold into "Other" for modeling.
RARE_FOLD_TO_OTHER = ["Cystine"]

# --- Cohort definition -----------------------------------------------------
PREOP_WINDOW_DAYS = 120           # CT within 120 days before (or ~same encounter as) surgery
PREOP_AFTER_GRACE_DAYS = 7        # allow CT up to a week after DOS (same admission)

# --- DICOM series curation (non-contrast axial selection) ------------------
# Stones do not require contrast; the primary cohort uses non-contrast CT only.
MIN_AXIAL_SLICES = 20             # a real volume, not a scout/few-slice object
AXIAL_NORMAL_TOL = 0.95          # |z-component of slice normal| above this == axial

# A series is contrast-enhanced if ContrastBolusAgent is set OR the description
# matches these (word-boundary) tokens. Tag is authoritative; text is backup.
CONTRAST_DESC_TOKENS = ["CE", "CONTRAST", "WITH IV", "W IV", "W/IV",
                        "WITH CONTRAST", "POST CONTRAST", "ARTERIAL", "VENOUS",
                        "DELAYED", "NEPHROGRAPHIC", "UROGRAM", "EXCRETORY"]
# Non-image / rendered / report series to reject outright (never usable as a
# volume). NOTE: coronal/sagittal/MPR reformats are NOT here — a non-contrast
# reformat is a usable volume (HU is orientation-independent) and is the only
# archived series for some patients; it is kept as a lower-quality tier.
# NOTE: do NOT use a bare 'PROTOCOL' token — it wrongly catches 'STONE PROTOCOL'
# (the ideal non-contrast stone series). Target the report objects specifically.
EXCLUDE_NONIMAGE_TOKENS = ["SCOUT", "TOPOGRAM", "LOCALIZER", "DOSE", "REPORT",
                           "SUMMARY", "PRESENTATION", "KEY IMAGE", "MIP", "VR",
                           "3D", "AVG", "PATIENT PROTOCOL", "SURVIEW"]
# Reconstruction-kernel softness: soft/standard kernels give the most accurate
# stone HU (sharp/bone kernels edge-enhance and inflate HU). Used as a selection
# tiebreaker and always recorded for downstream HU harmonization.
SHARP_KERNEL_HINTS = ["BONE", "LUNG", "SHARP", "EDGE", "FC3", "FC5", "FC6", "FC07",
                      "FC08", "B70", "B60", "B50", "YA", "YB", "YC", "YD", "D40", "I70"]

os.makedirs(OUTPUT_DIR, exist_ok=True)
