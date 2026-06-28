# Dr Stone

**Predicting uric‑acid kidney stones from routine non‑contrast CT and emergency‑department labs.**

Most patients with symptomatic nephrolithiasis present to the ED, get a **non‑contrast stone‑protocol CT (NCCT)**, and have a basic metabolic panel + urinalysis. Dual‑energy CT — which can characterize stone composition — is **not routinely available**. Dr Stone reads the data the ED already has (stone attenuation + routine labs) and returns a **calibrated probability that a stone is uric‑acid**, the one composition whose management differs fundamentally (medical dissolution vs. intervention).

## Headline result (honest)

A deliberately lean, interpretable model on routine data:

- **AUROC 0.78 (95% CI 0.68–0.87)** for uric‑acid vs. non‑uric‑acid (repeated patient‑level grouped CV + patient‑clustered bootstrap; 319 stones / 223 patients; 11% UA).
- **NPV 0.97** at high sensitivity — reliably *rules out* uric acid.
- **Positive net clinical benefit across the full plausible threshold range** (decision‑curve analysis).
- Well calibrated (Brier 0.086).

Four more elaborate approaches were pre‑specified, tested, and **rejected**: Hb‑anchored HU calibration, a self‑supervised 3D CNN, HU‑distribution‑shape radiomics, and a larger multi‑stone cohort. The single‑energy composition signal is carried by **attenuation and urine chemistry (pH, acid–base / anion gap)**; added complexity overfit. See `docs/01_DrStone_Methods.md`.

## Front‑end

```bash
uvicorn web.app:app --host 127.0.0.1 --port 8200
# open http://127.0.0.1:8200/drstone
```

Workflow: point at a DICOM series → **detect & pick the stone** (auto‑measures HU) → add routine labs (or leave blank — missing values tolerated) → **calibrated UA probability + per‑case SHAP rationale**.

## Pipeline (`drstone/`)

| Module | Role |
|---|---|
| `data_ingest.py` | join the project workbook (clinical ↔ composition ↔ MRN crossover) |
| `match_dicom.py` | curate the **non‑contrast** axial series per patient |
| `stone_segmentation.py` | segment stone(s) (TotalSegmentator region + high‑HU components) |
| `stone_features.py` | native HU‑distribution + shape features |
| `match_stones.py` | multi‑stone rank matching (volume ↔ mass) |
| `calibration.py` | Hb‑anchored HU calibration (tested; not used in the locked model) |
| `interpolate.py`, `cnn.py` | 3D ROI resample + SSL CNN (tested; rejected) |
| `modeling.py`, `modeling2.py` | model development + group ablation |
| `lock_model.py` | **the locked model** with CIs + decision‑curve / calibration |
| `predict.py` | point‑of‑care inference (loads `models/`, per‑case SHAP) |
| `measure_cli.py` | isolated‑subprocess stone HU measurement for the web app |
| `web.py` | the `/drstone` routes (form + predict + auto‑measure + stone picker) |

The locked model is in `models/` (tracked). Processed tables and the source workbook live in `data/` (git‑ignored; may contain PHI). DICOM archives stay on the data drive (set via `DRSTONE_DICOM_ROOT` / `DRSTONE_CURATED_DIR`).

## Reproduce

```bash
pip install -r requirements.txt
python -m drstone.data_ingest        # build analytic tables (needs data/ workbook)
python -m drstone.match_dicom        # curate non-contrast series (needs DICOM)
python -m drstone.stone_features     # segment + features
python -m drstone.match_stones       # multi-stone matching
python -m drstone.lock_model         # lock model + CIs + figures
```

## Limitations

Retrospective, two‑site, modest UA count (wide CI); spot (not 24‑hour) urine chemistry; glucose as a diabetes proxy; rank‑based multi‑stone matching. Single‑energy CT cannot reliably separate calcium‑oxalate from calcium‑phosphate — this tool targets the actionable uric‑acid distinction. **Decision support, not a substitute for stone analysis. Requires prospective/external validation.**

---
*Extracted from the PRISM renal‑imaging codebase; Dr Stone is a self‑contained project.*
