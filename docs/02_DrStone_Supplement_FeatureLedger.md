# Supplementary Material — A Pre-Registered Ledger of Feature Families for Single-Energy CT Stone-Composition Prediction: What Works, and What Does Not

## S1. Rationale

The clinically actionable question in single-energy non-contrast CT (NCCT) stone imaging is whether stone composition can be estimated well enough, from data already collected at presentation, to guide definitive management. The dominant decisions are uric-acid vs. non-uric-acid (dissolvable vs. not) and, among calcium stones, **calcium oxalate (CaOx) vs. calcium phosphate (CaP)** — a distinction with metabolic and preventive consequences. Struvite is not a modeling target: infection stones are a clinical diagnosis (alkaline, turbid, malodorous urine at pH 7–8.5; dysuria, flank pain, recurrent UTI, frequently impending sepsis) that any emergency clinician makes without a model.

This supplement reports, in one place, **every feature family we tested** — the hypothesis and mechanistic rationale that motivated each, the extraction/derivation method, the evaluation, the result, and the disposition — with the successes grouped alongside the failures. Negative results are reported deliberately and in full: publishing only positive findings leaves others to repeatedly expend resources on avenues that do not bear fruit. Our aim is to delineate the boundary of what single-energy NCCT plus routinely available data can and cannot resolve, so that subsequent work is directed at the informative gaps (Section S4).

Unless stated otherwise, evaluation is **repeated stratified patient-grouped 5-fold cross-validation** with **patient-clustered bootstrap 95% CIs**; the calcium-subtype tests use a dedicated CaOx-vs-CaP binary head (positive class = CaP) with the deltas expressed against the **HU + routine-labs** baseline (AUROC 0.710). Cohort: 319 laboratory-analyzed stones from 223 patients (CaOx 107, CaP 144, UA 36, struvite 27, other 5).

## S2. Master ledger (Table S1)

| # | Domain | Feature family | Hypothesis / rationale | Key result | Verdict |
|---|--------|----------------|------------------------|-----------|---------|
| **Retained** ||||||
| R1 | Model | **Lean UA model** (stone peak/mean HU; urine pH; CO₂/Cl/anion gap; BUN, Cr, Ca, glucose; age, sex) | UA is acid-driven and low-HU; routine data should identify it | **AUROC 0.781 (0.677–0.867)**, NPV 0.97, Brier 0.086; positive net benefit across 10–50% thresholds | **Deployed** |
| R2 | Model | **CaOx-vs-CaP head, HU + routine labs** | Acid–base chemistry (pH, RTA pattern, Ca) separates the calcium subtypes better than HU alone | HU 0.665 → **HU+labs 0.710 (0.637–0.778)**; beats 5-class CaP OvR (0.674) | **Deployed** (cascade + high-sensitivity CaP screen) |
| R3 | Imaging | **Nephrocalcinosis** (parenchymal calcific foci) | dRTA / hyperparathyroidism / MSK cause parenchymal Ca deposition and CaP stones | Biology clean (CaP frac 10× CaOx) but head Δ +0.013 (CI crosses 0); outlier-driven | **Radiologic flag, not a model feature** |
| **Rejected — imaging** ||||||
| F1 | Imaging | **Hb-anchored HU calibration** (air + aortic-blood anchors) | Normalize scanner HU offset to sharpen composition-HU mapping | No same-composition variance reduction (scanner offset ~10 HU ≪ ~220 HU biological spread) | Rejected |
| F2 | Imaging | **Self-supervised 3D CNN** (frozen-encoder deep features) | Learned volumetric features may capture texture beyond handcrafted HU | Underperformed; deep features did not cluster by composition; degraded fusion | Rejected |
| F3 | Imaging | **HU-distribution-shape radiomics** (skew, kurtosis, entropy, GMM modes, Shapiro) | Multi-component / zoned stones have distinctive HU histograms | Reduced cross-validated AUC | Rejected |
| F4 | Cohort | **Multi-stone N-lift** (volume↔mass rank matching, 131→319) | More labeled stones per patient → power | Neutral (kept for N; no lift) | Rejected (as a lift) |
| F5 | Imaging | **Stone density gradient** (core/periphery HU ratio) | CaOx retains a dense core; CaP tapers | CaOx 2.86 vs CaP 2.80 (gap 0.06 vs SD 1.2); head Δ −0.013; SHAP 13/17 | Rejected |
| F6 | Labs | **Normal-AG (hyperchloremic) acidosis composite + hyperchloremia index** | Explicit dRTA/GI-loss signature the tree may not synthesize | Δ −0.000 (CI −0.009,+0.008); redundant with raw CO₂/Cl/AG | Rejected |
| F7 | Labs | **eGFR (CKD-EPI 2021)** | CKD shifts mineral metabolism toward CaP | Δ −0.009; SHAP rank 7 yet OOF-flat (redundant with Cr/age/sex) | Rejected |
| F8 | Imaging | **Opportunistic vertebral BMD** (L1–L3 trabecular HU) | Hypercalciuria/dRTA ↔ low BMD ↔ calcium stones | Full-cohort +0.028 but complete-case crosses 0; flat by-composition (187 vs 194); 35% missing (FOV) | Rejected |
| F9 | Imaging | **Hepatic steatosis** (liver−spleen HU) | Gut dysbiosis raises enteric oxalate (CaOx) and hepatic fat → fatty liver > in CaOx | Steatosis CaOx 13% vs CaP 17% (p=0.71, reversed); Δ −0.006; signal is UA (already captured) | Rejected |
| F10 | Imaging | **Sarcopenia / muscle quality** (paraspinal+psoas HU, volume) | Body composition indexes metabolic phenotype | Δ −0.023 (P 0.03, worse); muscle-volume SHAP rank 5 while hurting (body-size confound) | Rejected |
| F11 | Imaging | **Aortic calcification** (Agatston-style, denoised) | Ca×PO₄ / CKD-MBD mineral axis (no serum PO₄/PTH in data) → CaP | Age-adjusted coef −0.10 (CI −0.41,+0.21); CaOx ≥ CaP (330 vs 249); head Δ +0.001 | Rejected |
| F12 | Imaging | **Bladder wall fraction / volume** | Chronic cystitis/obstruction → struvite | Bladder-only Δ +0.017 (P 0.79); distension-confounded | Rejected |
| F13 | Imaging | **Prostate calcification** | Chronic prostatitis → struvite | SHAP rank 1/22 but a **sex proxy** (79% M / 1% F); non-significant; lowers 5-class struvite AUC 0.686→0.626 | Rejected (leak) |

## S3. Detail by family

### A. Retained models (R1–R2) and the one flag (R3)
The **lean UA model** (R1) is the primary clinical product: at a high-sensitivity operating point it rules out uric acid with NPV 0.97 and shows positive decision-curve net benefit across the plausible threshold range. The **CaOx-vs-CaP head** (R2) is the only feature *addition* that improved the calcium-subtype decision: adding the routine metabolic panel to stone HU raised AUROC from 0.665 to 0.710 — the discriminating information for the calcium split is **chemical, not morphological** (Main Figs 6–7). **Nephrocalcinosis** (R3) is the sole imaging feature with clean, correct biology (mean parenchymal-calcification fraction ~10× higher in CaP than CaOx; prevalence CaOx 21% → CaP 33%); it is retained as a **deterministic radiologic flag** that prompts a CaP/dRTA/hyperparathyroidism work-up when substantial foci are auto-detected, because as a learned feature it is carried by a handful of high-burden cases and does not survive outlier removal (Δ +0.013 → +0.004).

### B. Rejected imaging features (F1–F5, F8–F13)
These span three mechanistic bets. **HU fidelity** (F1 calibration, F2 CNN, F3 shape radiomics, F5 density gradient): none beat handcrafted peak/mean HU — the composition signal in attenuation is low-dimensional and adding HU-derived complexity overfits at this N. **Bone/mineral-metabolism surrogates** (F8 BMD, F11 aortic calcification): both index axes (bone loss, vascular calcium) that are dominated by **age** — already modeled — and, for vascular calcium, index the hypercalciuria/CaOx phenotype as much as the phosphate/CaP one, netting to null. **Systemic/metabolic surrogates** (F9 hepatic steatosis, F10 sarcopenia, F12–F13 bladder/prostate): each re-encodes the metabolic-syndrome/urine-pH axis (F9's steatosis signal is real but belongs to UA, already captured) or, worse, a confound — F13 prostate calcification is a **sex indicator** (present in 79% of males, 1% of females) that the model exploits, ranked first by SHAP yet non-significant and harmful to its intended target class.

### C. Rejected lab/derived features (F6–F7)
The routine chemistry is fully exploited by the raw analytes: an explicit normal-anion-gap-acidosis composite (F6) and CKD-EPI eGFR (F7) are deterministic functions of values already in the model (CO₂/Cl/anion gap; creatinine/age/sex) and add nothing out-of-fold, despite SHAP attending to eGFR (rank 7).

## S4. Methodological through-line and interpretation

Three recurring findings organize the ledger:

1. **Feature importance is not predictive value.** Repeatedly, a feature drew high SHAP attention while adding nothing (eGFR, rank 7) or actively hurting (muscle volume, rank 5; prostate calcification, rank 1) out-of-fold. Adjudication was therefore on grouped-CV ΔAUROC with patient-clustered bootstrap CIs, never on importance. Figure S1 shows every family's Δ against the HU+labs baseline: **all confidence intervals cross zero.**

2. **Apparent gains require verification.** The one nominally significant full-cohort result (nephrocalcinosis + BMD, +0.038) dissolved under complete-case analysis (removing the 35% of scans without lumbar spine in the field of view: +0.024, CI crossing zero) and outlier removal. The bladder/prostate "gain" (+0.039) was a sex-proxy leak. Every candidate was subjected to complete-case, age-adjusted, and/or outlier-robustness checks before disposition.

3. **The information boundary.** Across thirteen rejected families spanning HU fidelity, mineral-metabolism surrogates, systemic-metabolic surrogates, and derived chemistry, single-energy composition prediction is **saturated by stone attenuation and urine/acid–base chemistry**. The residual CaOx-vs-CaP ceiling (~0.71) is not a modeling limitation to be engineered away with additional CT-derived or routine-lab features; it reflects the absence of the **stone-specific analytes** that define the two supersaturation states — most directly **urinary oxalate** (the swing driver of CaOx supersaturation) and its modifiers (magnesium, citrate), which are not present in routine data and are not inferable from single-energy attenuation.

## S5. A second endpoint — multicomponent detection (an exhaustive negative)

Recasting the problem around management decisions (main text §3.7) raises a distinct, coarser question grounded in stone pathophysiology: most stones are **multicomponent** (68% here), laminating a nidus with a shell of different composition — so **can any routine‑data channel detect *that a stone is mixed*, without resolving what it is mixed of?** This is a lower bar than composition subtyping (heterogeneity, not identity) with better‑balanced classes, and it was tested exhaustively. Every feature family of the study — plus a dedicated **peak‑deconvolution** family added for this question (per‑stone 1‑ vs 2‑Gaussian mixture fits: peak separation, Sarle's bimodality coefficient, skewness, and the 2‑vs‑1 BIC evidence) — was evaluated against the multicomponent label under the same patient‑grouped CV with clustered bootstrap.

**Table S2. Multicomponent detection by channel** (multicomponent = second‑component fraction ≥ 10%; AUROC, patient‑clustered bootstrap 95% CI; Figure S4).

| Channel | n feat | AUROC (95% CI) | Verdict |
|---------|-------:|----------------|---------|
| Stone HU (base) | 5 | 0.552 (0.480–0.625) | null |
| HU‑shape radiomics | 10 | 0.553 (0.482–0.619) | null |
| Density gradient | 1 | 0.426 (0.353–0.496) | below chance |
| Hb‑calibrated HU | 3 | 0.422 (0.356–0.491) | below chance |
| CNN deep features | 64 | 0.480 (0.413–0.545) | null |
| Labs / urine | 10 | 0.484 (0.415–0.549) | null |
| Derived acid–base + eGFR | 3 | 0.433 (0.360–0.507) | null |
| Demographics | 2 | 0.435 (0.367–0.504) | null |
| Nephrocalcinosis | 4 | 0.357 (0.290–0.428) | below chance |
| Vertebral BMD | 2 | 0.459 (0.390–0.527) | null |
| Hepatic steatosis | 3 | 0.569 (0.488–0.649) | null |
| Sarcopenia | 2 | 0.481 (0.407–0.559) | null |
| Aortic calcification | 3 | 0.427 (0.360–0.494) | below chance |
| Bladder / prostate | 3 | 0.414 (0.345–0.485) | below chance |
| **Peak deconvolution** | 5 | 0.527 (0.455–0.598) | null (0.375 at ≥25%) |
| **All combined** | 115 | 0.626 (0.557–0.695) | **UA‑purity confound** |

**No channel detects multicomponent structure.** Every individual family's interval crosses chance, several below it. The single apparent gain — the 115‑feature combined model at **0.626 (0.557–0.695)** — is a **uric‑acid‑purity confound**, not multicomponent detection: uric‑acid stones are **39% multicomponent vs. 72% for non‑UA**, so predicting "mixed" is largely predicting "not‑UA," which the model already does (Node A). Two orthogonal controls expose it: restricting to calcium stones (removing the UA axis) drops the combined model to **0.549 (non‑significant)**, and a clean ≥25% mixedness threshold drops it to **0.491 (chance)**.

**Peak deconvolution** — the most direct attempt, actively fitting two overlapping Gaussians and measuring their separation and skewness (two unequal overlapping peaks sum to a skewed composite) — was itself at chance (0.53, size‑robust set excluding the voxel‑count‑confounded BIC), *below* chance at the clean threshold (0.375), added nothing to the HU‑shape moments (0.526→0.541), and was matched by a **stone‑volume‑only baseline** — what little varied was stone size, not bimodality. At clinical slice thickness (≈3 mm) partial‑volume averaging erases the two‑peak structure. The lamination is histologically real but not resolvable on single‑energy clinical CT: **the multicomponent architecture that characterizes most stones is undetectable by any channel a standard ED visit provides** — a reinforcement, at a coarser and better‑powered endpoint, of the composition‑subtyping boundary in Table S1.

## S6. Figures

[[FIGURE: FigS1_forest.png]]
**Figure S1.** Forest plot of ΔAUROC for each feature family added to the HU+labs CaOx-vs-CaP head (patient-clustered bootstrap 95% CI). All intervals cross zero; the only genuine improvement was adding the routine metabolic panel to stone HU (baseline, AUROC 0.710). Grey = rejected; orange = apparent gain traced to a confound/leak (bladder/prostate = sex proxy); blue = retained as a radiologic flag (nephrocalcinosis).

[[FIGURE: FigS2_confusion.png]]
**Figure S2.** Five-class composition confusion matrix (row-normalized), patient-level cross-validation.

[[FIGURE: FigS3_decision_nodes.png]]
**Figure S3.** Decision-node performance (left) and the multicomponent threshold sweep (right). Routine data resolves the dissolvable decision (Node A, uric acid vs. non-UA) and a modest calcium-phosphate metabolic flag (Node C2), but multicomponent detection (Node C1) sits at chance and falls below it as the mixedness definition tightens.

[[FIGURE: FigS4_multicomponent.png]]
**Figure S4.** Exhaustive multicomponent detection — every feature family plus peak deconvolution against the single- vs. multi-component endpoint (thr 0.10; patient-clustered bootstrap 95% CI). All families cross or sit below chance; the lone combined "gain" (orange) is a uric-acid-purity confound (see Table S2).
- Main-text figures cross-referenced: Fig 1 UA ROC; Fig 2 decision-curve; Fig 3 calibration; Fig 4 UA SHAP; Fig 5 density-gradient per-class AUC; Fig 6 CaOx-vs-CaP head ROC by feature set; Fig 7 CaP-screen operating point.
