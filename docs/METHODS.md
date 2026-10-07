# Methods

## Study design

The study used GSE176078 as the primary single-cell breast-cancer cohort and TCGA-BRCA as an external cohort. The computational workflow was implemented in Nextflow DSL2.

The primary cohort contained 100,064 cells from 26 samples spanning ER+, HER2+ and TNBC disease. The curated T-cell compartment contained 35,214 cells.

## Preprocessing

Raw per-sample sparse count matrices were inventoried, quality controlled, normalized and assembled into a cohort-level AnnData object.

Normalization consisted of library-size normalization followed by log1p transformation while retaining the original count matrix in `adata.layers["counts"]`.

Highly variable gene selection and PCA were performed before integration evaluation.

## Integration

Harmony was retained as a candidate representation rather than automatically selected solely by an aggregate integration score. Batch correction and biological conservation were evaluated quantitatively using standardized scIB-metrics-style benchmarking and then checked against TME/T-cell biological structure.

## T-cell state modelling

T cells were defined using the curated `celltype_major == "T-cells"` annotation. Six expression-derived programs were calculated:

- T-cell identity
- CD8 cytotoxicity
- Treg
- Tfh
- activation/effector
- exhaustion/dysfunction

The exhaustion program used multiple genes and was not represented by a single-marker definition.

A 30-nearest-neighbour state graph was constructed over the six-dimensional program representation. The resulting state landscape was used as the input structure for Deep-EVI.

## Deep-EVI

The model used sparse graph convolution with two graph-convolution layers, LayerNorm, GELU and dropout. The model jointly reconstructed program state and predicted standardized exhaustion and activation targets.

The primary split was performed at the sample level. Graph edges were restricted to cells belonging to the same split. Feature scaling, target standardization and EVI orientation/scaling were fitted using training data only.

## Held-out evaluation

Five samples were reserved for held-out evaluation, containing 9,864 curated T cells.

The primary held-out metrics included Spearman correlation, Pearson correlation, RMSE, MAE and R² for the exhaustion-associated target.

Because exhaustion was an auxiliary training target, these metrics quantify recovery of a supervised molecular state and are not independent biological validation.

## TCGA molecular bridge

A frozen Ridge regression model was fitted using training cells only to predict the raw Deep-EVI representation from the six program scores. The program coefficients were deterministically expanded into a 33-gene molecular surrogate using the exact program gene definitions.

TCGA-BRCA expression files were projected into this frozen molecular surrogate without refitting.

## External validation

The frozen TCGA surrogate was compared with a published TCGA immune model-based subtype classification. GDC case identifiers were mapped to TCGA participant barcodes before joining the external subtype labels.

The primary endpoint was the global Kruskal-Wallis test across represented immune subtypes. A pre-specified contrast compared C2+C3 against C4+C6.

No immune-subtype labels were used for model training, model selection, feature selection or cutoff selection.

## XAI

Step 11 interpreted the frozen Deep-EVI model using:

1. gradient×input program attribution,
2. gradient×normalized graph-edge attribution,
3. frozen 09A molecular attribution,
4. deterministic gene-level decomposition of the frozen 33-gene molecular surrogate.

The XAI stage did not retrain the model or refit the molecular surrogate.

## Scientific terminology

Deep-EVI is described as a graph-learned continuous exhaustion-associated T-cell state index.

It is not described as RNA velocity, a temporal trajectory, a causal score, or an independently validated clinical predictor.
