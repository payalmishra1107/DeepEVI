# Results

## Cohort and T-cell compartment

The GSE176078 cohort contained 100,064 cells across 26 breast-cancer samples. The curated T-cell compartment contained 35,214 cells.

The study included 11 ER+, 5 HER2+ and 10 TNBC samples.

## Deep-EVI held-out performance

The held-out set contained 9,864 T cells from five samples.

Deep-EVI showed:

- exhaustion Spearman ρ = 0.9498
- exhaustion Pearson r = 0.9925
- RMSE = 0.2055
- MAE = 0.1372
- R² = 0.9727

These metrics reflect association with a training-derived exhaustion target and therefore should not be presented as independent biological validation.

## TCGA projection

The frozen molecular surrogate was projected to 1,095 TCGA-BRCA expression cases.

The external immune-subtype validation matched 1,083 participants. Five immune subtypes were represented.

### Global test

Kruskal-Wallis:

- H = 276.6231
- p = 1.19 × 10⁻⁵⁸

### Primary contrast

C2+C3 versus C4+C6:

- 582 versus 132 participants
- rank-biserial effect size = 0.5357
- p = 6.78 × 10⁻²²

This result is the principal independent biological validation of the frozen TCGA-compatible molecular surrogate.

## XAI

Gradient×input attribution of the frozen Deep-EVI model identified exhaustion/dysfunction as the dominant program-level contributor:

- exhaustion/dysfunction: 78.0%
- Tfh: 6.43%
- CD8 cytotoxic: 5.58%
- T-cell identity: 3.67%
- Treg: 3.62%
- activation/effector: 2.69%

The frozen TCGA molecular surrogate was similarly dominated by the exhaustion/dysfunction coefficient, accounting for 86.7% of absolute molecular attribution.

## Interpretation

Together, the results support a reproducible graph-learned representation of exhaustion-associated T-cell state with external biological concordance in TCGA-BRCA.

The evidence does not establish temporal directionality, RNA velocity, causality or clinical prediction.
