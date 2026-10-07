# Reproducibility

## Execution model

The intended execution model is:

```bash
nextflow run main.nf -profile conda,workstation
```

Frozen downstream entry points are exposed as `main_step*.nf`.

## Environment isolation

Each analytical stage uses an explicit Conda environment under `envs/`. This keeps single-cell processing, integration metrics, PyTorch Deep-EVI, TCGA analyses and XAI dependencies isolated.

## Provenance

The pipeline records:
- input inventory
- sample counts
- cell/gene counts
- pipeline traces
- stage summaries
- frozen validation manifests
- integrity audit results

## Data policy

Human genomic data are excluded from Git history. Users retrieve GSE176078 and TCGA-BRCA from their original sources.

## Frozen release

The computational analysis is frozen. Changes to model architecture, features, cohorts or validation endpoints should be released as a new version rather than silently modifying the frozen result set.
