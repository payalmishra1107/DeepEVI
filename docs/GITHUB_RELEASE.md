# GitHub Release v1.0.0 — Frozen Computational Analysis

## Deep-EVI breast-cancer single-cell modelling

This release freezes the reproducible computational analysis of the Deep-EVI project.

### Highlights

- 100,064 GSE176078 cells
- 35,214 curated T cells
- graph-learned Deep-EVI state index
- strict sample-level held-out evaluation
- 1,095 TCGA-BRCA molecular projections
- 1,083-participant independent immune-subtype validation
- explainable Deep-EVI at program, graph, molecular and gene levels
- master pipeline audit: 82 PASS / 1 WARN / 0 FAIL

### Principal validation result

The frozen TCGA-compatible surrogate showed significant differences across the five represented TCGA immune subtypes (Kruskal-Wallis H=276.6231, p=1.19e-58).

The pre-specified C2+C3 versus C4+C6 contrast yielded rank-biserial effect size 0.5357 (p=6.78e-22).

### Scientific scope

Deep-EVI is an exhaustion-associated T-cell state index. It is not RNA velocity, a temporal trajectory, a causal score, or a clinical prediction model.

### Recommended release contents

Include the source code, configuration, documentation, compact summaries and frozen manifests. Exclude raw human genomic data and large intermediate files.
