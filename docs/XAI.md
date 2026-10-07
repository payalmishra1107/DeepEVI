# Explainable AI

## Overview

Step 11 interprets the frozen Deep-EVI model. It does not modify the model.

## 11A — Program attribution

Gradient×input attribution was computed for the six input programs on the frozen held-out test cells.

The exhaustion/dysfunction program accounted for 77.999% of global mean-absolute program attribution.

## 11B — Graph attribution

Gradient×normalized-edge-weight attribution quantified the local influence of state-space graph connections.

The graph contains T-cell transcriptional state neighbourhoods generated from the six-dimensional program representation. These edges are not ligand-receptor interactions.

## 11C — Molecular attribution

The frozen 09A Ridge model was decomposed into its six program coefficients.

Absolute attribution fractions:

- exhaustion/dysfunction: 86.714%
- Tfh: 5.972%
- Treg: 3.494%
- CD8 cytotoxic: 2.109%
- activation/effector: 0.872%
- T-cell identity: 0.841%

## 11D — Gene-level decomposition

The six frozen program definitions were deterministically converted into a 33-gene weight vector. Overlapping genes inherited the summed contributions from the programs in which they occurred.

Highest reconstructed weights included CXCL13, PDCD1, TIGIT, CTLA4, TOX, TOX2, LAG3, HAVCR2 and ENTPD1.

This is an attribution/decomposition of the frozen molecular bridge, not a newly fitted gene signature.

## Reproduction

The production XAI run was reproduced against the validated development run.

Maximum absolute differences in summary metrics were below 2.4 × 10⁻⁷, and the 33-gene molecular weights matched exactly.

## What this does not claim

- no causal interpretation
- no independent biological validation from XAI
- no SHAP analysis
- no GNNExplainer analysis
- no RNA velocity
- no model retraining
