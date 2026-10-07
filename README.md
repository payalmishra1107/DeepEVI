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
