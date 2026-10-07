# DeepEVI

## Deep learning–driven modelling of tumour–immune dynamics and T-cell exhaustion in breast cancer

**DeepEVI** is a reproducible Nextflow DSL2 research pipeline combining single-cell transcriptomics, tumour-microenvironment biology, statistical genomics, machine learning, graph deep learning, external cohort validation and explainable AI.

> **Scientific scope:** DeepEVI is a graph-learned continuous **exhaustion-associated T-cell state index**. It is **not RNA velocity**, not a temporal trajectory, and not a causal or clinical prediction model.

---

## Why this repository exists

This project is designed as a complete computational research workflow rather than a collection of notebooks. It demonstrates:

- **Biology:** breast-cancer TME, T-cell states, exhaustion/dysfunction, Treg/Tfh/cytotoxic biology
- **Bioinformatics:** scRNA-seq QC, normalization, HVG selection, PCA, integration, AnnData, TCGA/GDC processing
- **Statistics:** held-out evaluation, non-parametric testing, survival modelling, effect sizes, sensitivity analyses
- **AI/ML:** representation learning, supervised multi-task learning, Ridge molecular projection
- **Deep learning:** sparse graph convolution and latent-state modelling in PyTorch
- **GNN methodology:** state-space KNN graph, split-specific adjacency, graph diagnostics and ablations
- **Explainable AI:** gradient×input, graph-edge attribution, molecular and gene-level decomposition
- **Reproducibility:** Nextflow DSL2, Conda environments, modular processes, frozen manifests, provenance and integrity audits
- **Scientific validation:** held-out cells plus independent TCGA-BRCA biological validation

---

## End-to-end workflow

The canonical scientific workflow is orchestrated by **one Nextflow DSL2 entrypoint**:

```text
GSE176078
   |
01 provenance
   |
02 QC
   |
03 normalization
   |
04 cohort assembly
   |
05 HVG + PCA
   |
06 Harmony candidate
   |------ 06A quantitative evaluation
   |------ 06B standardized benchmark
   |
07A TME / T-cell validation
   |
07B T-cell expression programs
   |
07C expression-state landscape + KNN graph
   |
08A Deep-EVI
   |------ 08B characterization
   |------ 08C ablation
   |------ 08D graph diagnostics
   |------ 08E robustness
   |
FROZEN MODEL
   |
   +---- 09A frozen TCGA molecular surrogate
   |        |
   |       09B TCGA projection
   |        |--- 09C survival
   |        |--- 09D bulk composition
   |        |--- 09E biological concordance
   |
   +---- 10A held-out benchmark
   |        |--- 10A-5 state-axis analysis
   |        |--- 10A-5B overlap-controlled analysis
   |
   +---- 10B independent TCGA-BRCA validation
   |
   +---- 11 frozen-model XAI
```

**Canonical execution:** `main.nf` orchestrates the complete computational chain.

**Checkpoint execution:** `main_step*.nf` files reproduce individual frozen stages for debugging, review and checkpoint-level reproducibility. They are secondary interfaces, not separate scientific pipelines.

## Primary cohort

**GSE176078**

- 26 breast-cancer samples
- 100,064 cells
- 29,733 genes
- 11 ER+
- 5 HER2+
- 10 TNBC
- 35,214 curated T cells
- 9,864 held-out T cells from five held-out samples

The raw human genomic data are **not redistributed in this repository**.

---

## Deep-EVI model

Deep-EVI models six T-cell expression programs:

1. T-cell identity
2. CD8 cytotoxicity
3. Treg state
4. Tfh state
5. activation/effector state
6. exhaustion/dysfunction

A 30-nearest-neighbour graph represents local T-cell state structure.

The frozen model uses:

- sparse graph convolution
- hidden dimension = 64
- latent dimension = 32
- LayerNorm
- GELU
- dropout = 0.15
- program reconstruction
- exhaustion prediction
- activation prediction

### Leakage controls

The pipeline enforces:

- sample-level train/validation/test splitting
- graph edges restricted within split
- train-only feature scaling
- train-only target standardization
- train-only EVI orientation/scaling
- no test-set training
- no test-set model selection

---

## Frozen held-out results

On 9,864 held-out T cells:

| Metric | Result |
|---|---:|
| Spearman vs exhaustion target | **0.9498** |
| Pearson vs exhaustion target | **0.9925** |
| RMSE | **0.2055** |
| MAE | **0.1372** |
| R² | **0.9727** |

These are associations with a training-derived exhaustion target. They are **not independent biological validation**.

---

## Benchmark Study

### Objective

The benchmark asks a specific question:

> **Does the frozen Deep-EVI score recover an exhaustion-associated T-cell state on cells that were never used for model fitting or model selection, and does it behave consistently with predefined external T-cell state references?**

The benchmark is deliberately separated from model training and from the external TCGA-BRCA validation.

### Benchmark design

Step 10A evaluates the **frozen** Deep-EVI model on the exact held-out population from Step 08A:

- **9,864 held-out T cells**
- **5 held-out samples**
- **6 predefined reference signatures**
- Deep-EVI **not retrained**
- test cells **not used for training**
- test cells **not used for model selection**
- TCGA data **not used**
- reference scores **not refit**
- reference-gene overlap with Deep-EVI construction genes explicitly audited
- patient-level inference avoided because only five held-out samples are available

### Held-out reference benchmark

| Representation / reference | Spearman rho | Pearson r |
|---|---:|---:|
| **Deep-EVI vs exhaustion target** | **0.949793** | **0.992500** |
| Terminal-exhaustion reference | **0.970577** | **0.981107** |
| Sade-Feldman dysfunctional reference | 0.884936 | 0.883639 |
| Van der Leun 2022 reference | 0.787964 | 0.881975 |
| Progenitor-TPEX reference | **-0.373697** | **-0.374716** |
| Cytotoxic control | 0.312118 | 0.429753 |
| Memory-progenitor reference | **-0.420898** | **-0.436254** |

The directionality is biologically coherent: Deep-EVI is strongly aligned with terminal-exhaustion/dysfunction references and inversely related to progenitor/memory-oriented references.

### Subset concordance

A representative CD8/LAG3 exhaustion-associated state showed an AUC of **0.910133** for Deep-EVI on the held-out population. This supports state-level concordance beyond the single scalar exhaustion target.

### Overlap-controlled analysis

Reference-gene overlap was explicitly quantified because several published exhaustion signatures share genes with the Deep-EVI construction programs. The overlap-controlled analysis showed that part of the raw state-axis association is explained by direct molecular overlap, while a **residual positive relationship remains after construction genes are removed**.

This is important scientifically: the benchmark supports **construct validity and robustness**, but it is not described as independent biological validation.

### What the benchmark establishes

The benchmark supports the conclusion that Deep-EVI:

1. generalizes to unseen cells/samples under a leakage-controlled split;
2. recovers the intended exhaustion-associated molecular axis;
3. agrees with independent reference definitions of terminal exhaustion/dysfunction;
4. moves in the expected opposite direction to progenitor/memory-oriented states;
5. retains signal after explicit construction-gene overlap control.

It does **not** establish causality, temporal progression, RNA velocity, or clinical utility.

---

## Independent TCGA-BRCA validation

The frozen GSE176078-derived molecular surrogate was projected into TCGA-BRCA without retraining or refitting.

- 1,095 expression cases
- 1,083 matched participants for immune-subtype validation
- represented subtypes: C1, C2, C3, C4, C6
- C5 was absent

### Global test

Kruskal-Wallis:

- H = **276.6231**
- p = **1.19 × 10⁻⁵⁸**

### Pre-specified contrast

C2+C3 versus C4+C6:

- n = 582 vs 132
- rank-biserial effect = **0.5357**
- p = **6.78 × 10⁻²²**

This is the principal independent biological validation result of the frozen TCGA-compatible surrogate.

---

## Explainable AI

Step 11 operates on the frozen model and does not retrain it.

### Program attribution

Gradient×input attribution:

| Program | Global attribution |
|---|---:|
| Exhaustion/dysfunction | **78.0%** |
| Tfh | 6.43% |
| CD8 cytotoxic | 5.58% |
| T-cell identity | 3.67% |
| Treg | 3.62% |
| Activation/effector | 2.69% |

### Molecular attribution

The frozen TCGA molecular surrogate is dominated by the exhaustion/dysfunction component (**86.7%** of absolute molecular attribution).

The gene-level decomposition contains 33 genes, with high reconstructed weights including CXCL13, PDCD1, TIGIT, CTLA4, TOX, TOX2, LAG3, HAVCR2 and ENTPD1.

These are model attributions, not causal biological effects.

---

## Scientific Interpretation, Study Answer & Results

### Scientific question

**Can a graph-learned representation of T-cell transcriptional state provide a reproducible, biologically interpretable index of exhaustion across breast-cancer single-cell data, while remaining testable on held-out cells and an external TCGA-BRCA cohort?**

### Study design

The study begins with **100,064 cells from 26 breast-cancer samples** in GSE176078 and identifies a curated T-cell compartment of **35,214 cells**. Six complementary biological programs—T-cell identity, CD8 cytotoxicity, Treg state, Tfh state, activation/effector state, and exhaustion/dysfunction—are used to define the T-cell state space.

A sample-level split and split-restricted 30-nearest-neighbour graph are then used to train Deep-EVI. The model learns a latent representation through sparse graph convolution and multi-task objectives. A frozen molecular surrogate is subsequently derived for external TCGA-BRCA projection.

### Scientific answer

**Deep-EVI successfully recovers a continuous exhaustion-associated T-cell state that generalizes to held-out cells and shows concordance with predefined exhaustion reference signatures.** The evidence is strongest for construct validity and external biological concordance rather than causal or clinical prediction.

### Main results

**1. T-cell state modelling**

The 35,214-cell T-cell compartment forms a structured transcriptional state landscape spanning identity, cytotoxicity, regulatory/Tfh states, activation and exhaustion-associated programs. The graph representation captures local relationships between cells while preserving the sample-level separation required for leakage control.

**2. Held-out model performance**

On 9,864 held-out T cells from five held-out samples:

- Spearman correlation with the exhaustion target: **0.949793**
- Pearson correlation with the exhaustion target: **0.992500**
- RMSE: **0.2055**
- MAE: **0.1372**
- R²: **0.9727**

These values demonstrate recovery of the targeted exhaustion-associated molecular axis. Because the exhaustion program was an auxiliary training target, these metrics are **not independent biological validation**.

**3. External reference concordance**

The frozen score correlates strongly with a predefined terminal-exhaustion reference (Spearman **0.970577**) and dysfunctional-T-cell references, while showing negative associations with progenitor/memory-oriented references. The overlap-controlled analysis further tests whether the signal survives removal of construction genes.

**4. Independent TCGA-BRCA validation**

A frozen 33-gene molecular surrogate was projected into **1,095 TCGA-BRCA expression cases** without refitting against TCGA outcomes or immune-subtype labels. For the final immune-subtype analysis, **1,083 participants** were matched across five represented subtypes (C1, C2, C3, C4 and C6; C5 was absent).

The global subtype comparison was highly significant:

- Kruskal–Wallis H = **276.623117**
- p = **1.191353 × 10⁻⁵⁸**

The pre-specified C2+C3 versus C4+C6 contrast yielded:

- n = **582 vs 132**
- directional rank-biserial effect = **0.535666**
- p = **6.778021 × 10⁻²²**

This provides the principal independent biological validation of the frozen molecular surrogate against an external immune-state classification.

**5. Clinical interpretation**

Unadjusted TCGA survival analyses show an association with outcome, but the association attenuates after adjustment for age and stage and is sensitive to bulk immune composition. Therefore, this repository **does not claim independent prognostic utility or clinical prediction**.

**6. Biological interpretation and XAI**

The frozen model attributes most of its program-level signal to the exhaustion/dysfunction component. Molecular decomposition highlights exhaustion-associated genes including **PDCD1, LAG3, TIGIT, HAVCR2, CTLA4, TOX, TOX2, ENTPD1 and CXCL13**. These are model-attribution results and should be interpreted as biologically informative hypotheses rather than causal effects.

### Overall scientific conclusion

> **Deep-EVI provides a reproducible graph-learned representation of exhaustion-associated T-cell state in breast cancer. Its strongest evidence is the combination of leakage-controlled held-out construct benchmarking, agreement with external exhaustion-state references, and independent TCGA-BRCA immune-subtype concordance. The framework supports biological interpretation and biomarker hypothesis generation, but it does not establish temporal exhaustion, causality, or clinical utility.**

---

## Reproducibility architecture

The repository is organized in four computational layers:

```text
main.nf
   |
workflows/
   |
modules/
   |
bin/
   |
envs/ + conf/ + config/
```

### Canonical entrypoint

```bash
nextflow run main.nf -profile conda,workstation \
  --primary_raw_dir /path/to/GSE176078_RAW \
  --tcga_raw /path/to/tcga_brca/raw/star_counts \
  --tcga_query /path/to/TCGA-BRCA_STAR_Counts_query.json \
  --tcga_clinical /path/to/TCGA-BRCA_clinical.tsv \
  --tcga_manifest /path/to/TCGA-BRCA_STAR_Counts_manifest.tsv \
  --immune_subtypes /path/to/Subtype_Immune_Model_Based.txt \
  --reference_signatures config/10a_benchmark_signatures.tsv \
  --outdir results \
  -resume
```

The primary and TCGA human genomic inputs remain external user-supplied data. The repository contains code, scientific definitions, compact summaries, manifests and provenance rather than redistributed raw genomic data.

### Phase architecture

| Phase | Nextflow workflow | Purpose |
|---|---|---|
| 01 | `workflows/01_ingestion.nf` | provenance and input inventory |
| 02 | `workflows/02_preprocessing.nf` | QC, normalization, cohort assembly |
| 03 | `workflows/03_integration.nf` | HVG/PCA, Harmony and integration evaluation |
| 04 | `workflows/04_tcell_analysis.nf` | biological validation and T-cell state modelling |
| 05 | `workflows/05_deep_evi.nf` | Deep-EVI and model diagnostics |
| 06 | `workflows/06_tcga_validation.nf` | frozen TCGA bridge and bulk analyses |
| 07 | `workflows/07_heldout_validation.nf` | held-out and independent validation |
| 08 | `workflows/08_xai.nf` | frozen-model explainability |
| 09 | `workflows/09_audit.nf` | integrity auditing |

## Scientific safeguards

The frozen workflow explicitly tracks:

- RNA velocity: **not claimed**
- temporal trajectory: **not claimed**
- causal inference: **not claimed**
- clinical prediction: **not claimed**
- test-set training: **false**
- test-set model selection: **false**
- TCGA-driven model refitting: **false**
- TCGA signature refitting during validation: **false**
- immune-subtype labels used for training: **false**
- immune-subtype labels used for model selection: **false**

### Master audit

```text
PASS : 82
WARN : 1
FAIL : 0

PIPELINE INTEGRITY: PASS
```

The single warning is the absence of a published Step 1 artifact under `results_step1`; the Step 1 production inventory itself recorded 26 archives and 0 failures.

---

## Repository release policy

This repository is a **reproducible research-code portfolio**, not a raw-data repository.

Do not commit:

- raw GEO archives
- raw TCGA/GDC expression data
- FASTQ/BAM files
- `work/`
- `.nextflow/`
- large H5AD intermediates
- GDC binaries
- credentials
- machine-specific absolute paths

Use accession numbers, manifests, checksums, configuration and compact result summaries to make the analysis reproducible without redistributing controlled human genomic data.

---

## Scientific limitations

1. Exhaustion is an explicit model target, so target recovery is not independent biological validation.
2. GSE176078 count matrices do not provide spliced/unspliced layers for conventional RNA velocity.
3. The state landscape is an expression-state proxy, not a temporal trajectory.
4. Bulk TCGA molecular scores can be influenced by immune-cell abundance and tissue composition.
5. TCGA immune-subtype validation contains five represented classes; C5 is absent.
6. External validation demonstrates biological concordance, not causality or clinical utility.
7. XAI explains model behaviour and does not establish causal mechanisms.

---

## Portfolio focus

This project demonstrates a complete research stack:

**Cancer biology → scRNA-seq → statistical genomics → Python/Scanpy → Nextflow DSL2 → Conda reproducibility → PyTorch → GNNs → representation learning → model diagnostics → TCGA/GDC → external validation → XAI → scientific provenance**

The goal is to make the entire computational chain inspectable, reproducible and scientifically defensible.

---

## Author

**Payal Mishra**  
Bioinformatics | Computational Biology | AI/ML | Deep Learning | Reproducible Genomics

GitHub: https://github.com/payalmishra1107
