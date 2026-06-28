# Predicting Uric‑Acid Kidney Stones from Routine Non‑Contrast CT and Emergency‑Department Labs

## Abstract

Most patients with symptomatic nephrolithiasis present to the emergency department (ED) and undergo a **non‑contrast stone‑protocol CT (NCCT)**; nearly all also have a basic metabolic panel and urinalysis. Dual‑energy CT (DECT), which can characterize stone composition in vivo, is **not routinely available** in ED practice. We asked whether the data already collected on these patients — stone attenuation on NCCT plus routine labs — can identify **uric‑acid (UA) stones**, the one composition whose management differs fundamentally (medical dissolution by urinary alkalinization vs. intervention). Using 319 surgically‑analyzed stones from 223 patients with paired NCCT and contemporaneous labs, a lean, interpretable model achieved an **AUROC of 0.781 (95% CI 0.677–0.867)** for UA vs. non‑UA under repeated patient‑level cross‑validation with patient‑clustered bootstrap. At a high‑sensitivity operating point the model **reliably rules out UA (NPV 0.97)**, and decision‑curve analysis shows **positive net clinical benefit across the full plausible threshold range (~10–50%)**. Notably, more elaborate approaches — internal HU calibration, a 3D CNN, HU‑distribution‑shape radiomics, and a larger multi‑stone cohort — were each tested and **did not improve** on this lean model. The composition signal in single‑energy CT is carried by **attenuation and urine chemistry**, and a tool built on **data the ED already has** is the deployable contribution.

## 1. Background and aim

Stone composition guides management. **Uric‑acid stones dissolve** with urinary alkalinization and are often managed medically, whereas calcium and struvite stones are not. Identifying UA non‑invasively, at presentation, could spare intervention and direct metabolic therapy. DECT addresses this but is concentrated in tertiary centers and is essentially **unavailable for routine ED use**, where the imaging is single‑energy NCCT. Our explicit aim is a **real‑world decision‑support method using only routinely available data** — NCCT stone attenuation and routine ED labs — not a method that presumes specialized acquisition.

## 2. Methods

### 2.1 Cohort and data
Patients with surgically retrieved, laboratory‑analyzed stones (FTIR/X‑ray diffraction component percentages) and a pre‑operative NCCT within 120 days were assembled from two sites, consolidated on a single medical‑record identifier. Composition was mapped to parent classes (calcium oxalate, calcium phosphate, uric acid, struvite, cystine), and the binary endpoint was **UA‑dominant vs. non‑UA**. Routine labs at the time of CT were extracted: basic metabolic panel, urine pH, and demographics.

### 2.2 Imaging curation (non‑contrast only)
From each study we selected a **non‑contrast** CT series (contrast‑enhanced and rendered/reformatted‑only series excluded by DICOM contrast tags and acquisition type), preferring native axial reconstructions and falling back to non‑contrast reformats where that was the only volume archived. HU is orientation‑independent, so reformats are valid for attenuation measurement.

### 2.3 Stone segmentation and features
Stones were auto‑segmented as high‑attenuation components within the kidney/bladder region (TotalSegmentator organ masks), excluding skeleton and great‑vessel calcification, with a size filter. Per stone we measured native‑voxel HU statistics (**peak, mean, percentiles**), volume, and — in extended analyses — HU‑distribution shape (skewness, kurtosis, number of histogram modes, normality) and morphology. The **locked model uses a deliberately lean feature set**, all routinely available in the ED:

> stone **peak and mean HU**; **urine pH**; **serum CO₂ (bicarbonate), chloride, anion gap** (acid–base axis); BUN, creatinine, calcium, glucose; **age, sex**.

Monotonic constraints encode known chemistry (higher HU → less likely UA; higher urine pH → less likely UA).

### 2.4 Locked model and validation
The classifier is a regularized gradient‑boosted tree (handles missing labs natively — important for real‑world data). Validation is **repeated (25×) stratified, patient‑level grouped 5‑fold cross‑validation** with averaged out‑of‑fold probabilities; **uncertainty is a patient‑clustered bootstrap (1000×) 95% CI**. We report discrimination (AUROC), a high‑sensitivity operating point (sensitivity/specificity/PPV/NPV), **calibration (Brier score)**, and **decision‑curve analysis** (net benefit vs. treat‑all/treat‑none) as the clinical‑utility readout.

### 2.5 Pre‑specified alternatives (tested, not assumed)
To avoid over‑engineering, four elaborations were pre‑specified and tested against the lean model: (i) an **Hb‑anchored internal HU calibration** (air + aortic‑blood anchors); (ii) a **self‑supervised 3D CNN** (frozen‑encoder deep features); (iii) **HU‑distribution‑shape radiomics**; and (iv) a **larger multi‑stone cohort** via volume↔mass matching.

## 3. Results

### 3.1 Cohort
319 clean‑label stones from 223 patients; **UA prevalence 11%**. Composition was dominated by calcium phosphate and calcium oxalate, with uric acid and struvite minorities and cystine rare.

### 3.2 Discrimination and operating characteristics
[[FIGURE: Fig1_roc.png]]
**Figure 1. UA‑stone detection (NCCT + ED labs).** AUROC **0.781 (95% CI 0.677–0.867)**. At a high‑sensitivity threshold: **sensitivity 0.92 (0.81–1.00)**, specificity 0.33 (0.25–0.39), PPV 0.15, **NPV 0.97** — a low predicted probability reliably excludes UA. Because UA prevalence is low, the model is best used as a **calibrated probability that updates clinical suspicion**, not a hard classifier.

### 3.3 Clinical utility
[[FIGURE: Fig2_dca.png]]
**Figure 2. Decision‑curve analysis.** The model yields **positive net benefit across the entire clinically plausible threshold range (~10–50%)**, exceeding both "treat all as UA" (which loses benefit beyond ~12%) and "treat none." This is the key real‑world result: for any clinician whose action threshold for "consider UA" falls in this range, the model adds value over default strategies.

[[FIGURE: Fig3_calibration.png]]
**Figure 3. Calibration.** Predicted UA probabilities track observed frequencies (**Brier 0.086**), supporting use of the continuous score for shared decision‑making.

### 3.4 What drives the prediction
[[FIGURE: Fig4_shap.png]]
**Figure 4. Feature importance (SHAP).** The model leans on **urine pH** and **stone HU**, with the **acid–base axis (bicarbonate, chloride, anion gap)** prominent — consistent with stone chemistry (UA forms in persistently acidic urine; the normal‑anion‑gap/alkaline‑urine pattern of distal renal tubular acidosis favors calcium‑phosphate stones). Directions are chemically correct (low pH and low HU push toward UA).

### 3.5 Elaborations that did not help
None of the four pre‑specified alternatives improved on the lean model: Hb‑anchored calibration gave no reduction in same‑composition HU variance (the scanner offset, ~10 HU, is dwarfed by ~220 HU of biological spread); the 3D CNN underperformed and degraded fusion (deep features did not cluster by composition); HU‑distribution‑shape features **reduced** cross‑validated AUC; and the multi‑stone N‑lift (131→319) was **neutral**. The single‑energy composition signal is captured by attenuation and chemistry, and added complexity overfit at this scale.

## 4. Discussion

The clinically actionable target in single‑energy CT is **UA vs. non‑UA**, and a model using only **routine ED data** discriminates it with an AUROC of ~0.78 and, importantly, **positive net clinical benefit** across realistic decision thresholds. The high NPV makes it well suited to **ruling out** the dissolvable stone and to **flagging** candidates for alkalinization or metabolic work‑up. Calcium‑oxalate vs. calcium‑phosphate separation remains poor — an intrinsic limit of single‑energy attenuation, not of the model — which is exactly where DECT, when available, retains an advantage; our method is complementary, targeting the decision that single‑energy data can support.

The chief value is **deployability and honesty**: every input is already collected on ED stone patients, and four tempting elaborations were tested and rejected, guarding against the over‑fitting that inflates many radiomics reports.

**Limitations.** Retrospective, two‑site, modest UA count (n=36) → wide CI; spot (not 24‑hour) urine chemistry; no explicit diabetes/HbA1c field (entered via glucose); rank‑based multi‑stone matching; and the need for **prospective, external validation**. The model is decision‑support, not a replacement for stone analysis.

## 5. Conclusion
Attenuation on routine non‑contrast CT, combined with routine ED labs centered on urine pH and the acid–base axis, identifies uric‑acid stones with clinically useful, well‑calibrated performance and net benefit across realistic thresholds — using data the emergency department already has, without dual‑energy CT. The deliberately lean, interpretable model outperformed calibration, deep‑learning, radiomic‑shape, and larger‑cohort elaborations, and is a credible basis for prospective evaluation as real‑world decision support in stone disease.
