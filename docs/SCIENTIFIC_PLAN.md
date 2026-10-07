# Deep-EVI Scientific Plan

## Objective

Model tumour–immune heterogeneity in GSE176078 breast-cancer single-cell transcriptomes and derive a graph-learned continuous exhaustion-associated T-cell state index.

## Phase A — Cohort foundation

1. Inventory the 26 GSE176078 archives and preserve sample provenance.
2. Perform conservative per-cell QC.
3. Normalize each sample independently.
4. Assemble the cohort with sample-level metadata.
5. Select highly variable genes and compute PCA.
6. Generate a Harmony candidate representation.
7. Quantitatively compare batch correction against biological conservation before using the representation downstream.

## Phase B — Biological state construction

1. Validate major TME populations.
2. Curate T cells.
3. Compute six predefined expression programs:
   - T-cell identity
   - CD8 cytotoxicity
   - Treg
   - Tfh
   - activation/effector
   - exhaustion/dysfunction
4. Construct a 30-nearest-neighbour T-cell state graph.
5. Characterize local state gradients.

The state landscape is explicitly an expression-state proxy. GSE176078 count matrices do not provide spliced/unspliced layers for conventional RNA velocity.

## Phase C — Deep-EVI

Train a sparse graph convolution model using:

- six program inputs
- sample-level train/validation/test splitting
- split-specific graph edges
- train-only feature scaling
- program reconstruction
- exhaustion prediction
- activation prediction
- graph smoothness regularization

The primary model output is a continuous exhaustion-associated state index.

## Phase D — Model diagnostics

Before external interpretation:

- characterize the score
- perform ablations
- diagnose graph contribution
- evaluate non-target associations and construct robustness

These analyses do not convert the score into a causal or temporal quantity.

## Phase E — Frozen TCGA bridge

Use training cells only to fit a transparent Ridge bridge from the six frozen expression programs to Deep-EVI.

Freeze:

- program definitions
- scaling
- coefficients
- gene-level molecular weights
- model provenance

No TCGA outcome or immune-subtype information is used for fitting.

## Phase F — Validation

### 10A: held-out primary-cohort validation

Use the exact held-out samples from Deep-EVI.

Benchmark against predefined reference state/exhaustion signatures and state-axis constructs. Report gene overlap with Deep-EVI construction programs.

### 10B: independent biological validation

Project the frozen TCGA-compatible surrogate into TCGA-BRCA and compare it with published immune-subtype labels.

This is biological concordance in an independent cohort, not clinical prediction or causal inference.

## Phase G — XAI

Apply attribution methods only to the frozen Deep-EVI model:

- program gradient×input
- graph-edge attribution
- molecular attribution
- deterministic gene-level decomposition

Attribution describes model behaviour; it is not a causal intervention.

## Phase H — Integrity

The final computational release should contain:

- Nextflow source
- Conda environments
- scientific configuration
- frozen manifests
- compact result summaries
- execution provenance
- integrity audit
- explicit limitations
