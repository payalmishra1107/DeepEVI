# Scientific Scope and Terminology

## Deep-EVI

Deep-EVI is a graph-learned continuous exhaustion-associated T-cell state index.

## What Deep-EVI is not

- It is not RNA velocity.
- It is not a temporal trajectory.
- It is not a causal score.
- It is not an independently validated clinical prediction model.

## Validation hierarchy

1. Held-out evaluation tests recovery of the supervised exhaustion-associated target.
2. Ablation and graph diagnostics test model construction and robustness.
3. Frozen TCGA projection creates a bulk-compatible molecular surrogate.
4. TCGA-BRCA immune-subtype comparison provides independent external biological validation.
5. XAI explains frozen model behaviour but does not establish causality.

## Interpretation rule

High held-out correlation with exhaustion is expected because exhaustion is an auxiliary training target. The principal independent biological evidence is the frozen TCGA-BRCA validation against a published immune-subtype classification not used for model construction or selection.
