# Deep-EVI Interpretation Framework

## What Deep-EVI represents

Deep-EVI is a continuous, graph-learned **exhaustion-associated T-cell state index**.

Higher values are oriented toward the learned exhaustion/dysfunction program.

## What it does not represent

Deep-EVI is not:

- RNA velocity
- a temporal trajectory
- a causal effect
- a clinical prediction score
- a treatment-response predictor
- an independently validated exhaustion biomarker by target correlation alone

## Interpreting the primary-cohort model

The exhaustion program is an explicit supervised target.

Therefore:

> Strong Deep-EVI–exhaustion-target agreement demonstrates successful recovery of the training-defined construct, but is not independent biological validation.

This distinction must remain in the README, manuscript and presentations.

## Interpreting the state landscape

The 07C graph describes proximity in expression-program space.

Local gradients can be interpreted as state-space structure, but not as direction through biological time.

## Interpreting TCGA

The TCGA molecular score is a frozen molecular surrogate of cell-level Deep-EVI.

It is not the original GNN output.

The independent TCGA-BRCA immune-subtype analysis is the principal external biological validation because the subtype labels are not used to train or select the model.

## Interpreting survival analyses

The observed univariate survival association does not establish independent prognostic value after adjustment.

The adjusted analyses are therefore reported as exploratory/clinical-context analyses rather than as a validated clinical predictor.

## Interpreting XAI

Program and molecular attributions identify components that contribute to model output.

They should be described as:

- model attribution
- feature contribution
- learned representation contribution

They should not be described as:

- causal drivers
- mechanistic proof
- therapeutic targets

## Reporting rule

Every scientific claim should be traceable to:

1. a defined pipeline stage,
2. a frozen input/output contract,
3. a quantitative result,
4. an explicit interpretation boundary.
