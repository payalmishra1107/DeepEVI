# DeepEVI Pipeline Architecture

## Design principle

DeepEVI is organized as a Nextflow DSL2 workflow so that data provenance, computational environments, resource allocation, intermediate artifacts and stage boundaries are explicit.

## Stage map

| Stage | Purpose | Core methods |
|---|---|---|
| 01 | Ingestion/inventory | archive validation, provenance |
| 02 | QC | cell/gene QC, mitochondrial metrics |
| 03 | Normalization | library-size normalization, log1p |
| 04 | Cohort assembly | AnnData concatenation, metadata validation |
| 05 | Feature/latent space | HVG selection, PCA |
| 06/06A | Integration | Harmony candidate + quantitative evaluation |
| 07A | Biological validation | TME/T-cell composition and neighbourhood preservation |
| 07B | T-cell programs | expression-derived state scores |
| 07C | State landscape | six-dimensional state graph |
| 08A–08E | Deep-EVI | GNN training, characterization, ablation, diagnostics |
| 09A–09E | TCGA bridge | frozen molecular surrogate, projection, statistical analyses |
| 10A | Held-out benchmark | reference-state comparison |
| 10B | External validation | published TCGA immune-subtype endpoint |
| 11 | XAI | program, graph, molecular and gene attribution |

## Engineering principles

- DSL2 module separation
- stage-specific Conda environments
- explicit resource labels
- resumable execution
- sample-level leakage prevention
- frozen downstream checkpoints
- artifact-aware integrity audit
- no raw human genomic data in Git
- compact provenance/result summaries in Git

## Scientific safeguards

The pipeline distinguishes:
- supervised target recovery from independent validation
- expression-state landscapes from RNA velocity
- model interpretation from causal inference
- TCGA biological validation from clinical prediction

The frozen master audit reports 82 PASS, 1 WARN and 0 FAIL.
