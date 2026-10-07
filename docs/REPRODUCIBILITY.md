# Reproducibility

## Canonical execution model

The repository is now organized around one authoritative Nextflow DSL2 graph:

```text
main.nf
  -> workflows/
  -> modules/
  -> bin/
  -> envs/ + conf/ + config/
```

Run the complete analysis with:

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

The exact raw/TCGA locations are intentionally external to the repository because human genomic data are not redistributed.

## Pipeline layers

### 1. Canonical orchestration

`main.nf` defines the scientific dependency graph and passes outputs between phases.

### 2. Phase workflows

`workflows/` groups the methodology into reproducible scientific phases:

- ingestion
- preprocessing
- integration
- T-cell analysis
- Deep-EVI
- TCGA validation
- held-out/external validation
- XAI
- audit

### 3. Reusable processes

`modules/` contains the actual Nextflow process contracts.

### 4. Scientific implementations

`bin/` contains the production Python implementations invoked by the processes.

## Checkpoint interfaces

The `main_step*.nf` files are retained for:

- checkpoint reproduction
- debugging
- review
- isolated reruns
- reproducing frozen analytical stages

They are secondary interfaces. They do not replace the canonical `main.nf` dependency graph.

Steps 1–3 now have explicit checkpoint entrypoints, eliminating the previous ambiguity where Steps 2–3 were hidden inside `main.nf`.

## Dependency isolation

Each analytical stage declares a dedicated Conda environment under `envs/`.

This separates:

- Scanpy/AnnData preprocessing
- integration metrics
- T-cell state analysis
- PyTorch graph learning
- TCGA processing
- validation
- XAI

## Provenance

Nextflow execution records:

- execution report
- timeline
- trace
- work directories
- task-level command provenance

The scientific pipeline additionally produces:

- input inventory
- QC summaries
- integration metrics
- model split manifests
- frozen model checkpoints
- TCGA case inventories
- held-out validation manifests
- XAI reports
- integrity audit outputs

## Reproducibility safeguards

The pipeline preserves the project's key safeguards:

- sample-level train/validation/test splitting
- split-specific graph construction
- train-only scaling
- train-only EVI orientation/scaling
- no test-set model selection
- frozen TCGA molecular bridge
- no TCGA-driven refitting
- independent 10B validation
- frozen-model XAI

## Scientific interpretation boundaries

Reproducibility does not change the scientific scope:

- 07C is an expression-state landscape, not RNA velocity.
- Deep-EVI is an exhaustion-associated state index, not a temporal or causal quantity.
- High agreement with the model's exhaustion target is not independent validation.
- 10B is the principal independent biological validation.
- XAI explains model behaviour and does not establish causal mechanisms.

## Data policy

Human genomic data are excluded from Git history. Users retrieve GSE176078 and TCGA-BRCA from their original sources and provide them as explicit pipeline inputs.

## Frozen release policy

Frozen analyses should not be silently modified.

Changes to:

- model architecture
- feature definitions
- cohorts
- validation endpoints
- frozen thresholds
- scientific interpretation

should be versioned as a new release.
