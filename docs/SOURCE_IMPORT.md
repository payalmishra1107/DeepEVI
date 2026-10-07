# Production Source Import

This public repository is being assembled from the frozen Deep-EVI analysis.

The documentation and compact result layer are already published. The next repository layer is the exact production Nextflow/Python source used to generate the frozen analysis.

## Required source layer

The release should contain the actual:

- Nextflow DSL2 entry points
- `modules/*.nf`
- `bin/*.py`
- `envs/*.yml`
- `conf/*.config`
- `config/*`
- relevant test/smoke files

## Data exclusion

The source import must exclude:

- raw GSE176078 archives
- raw TCGA/GDC expression data
- FASTQ/BAM files
- large H5AD intermediates
- `work/`
- `.nextflow/`
- credentials
- machine-specific absolute paths

## Important

The repository must preserve the exact production implementation rather than replacing it with a simplified demonstration implementation. This distinction is important for scientific reproducibility and for accurately demonstrating Nextflow, bioinformatics and AI/ML engineering skills.
