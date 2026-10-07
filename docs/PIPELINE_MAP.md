# DeepEVI Pipeline Map

This is the authoritative map of executable stages in the repository.

## Repository conventions

- `main_step*.nf`: independently executable stage entrypoints.
- `modules/`: reusable Nextflow processes.
- `bin/`: production Python implementations.
- `envs/`: stage-specific Conda environments.
- `conf/`: resource and workstation configuration.
- `config/`: fixed scientific definitions.
- `docs/`: methods, results, reproducibility, XAI and pipeline documentation.
- `tests/`: repository-level smoke tests.
- `results/`: compact result summaries only; raw genomic data and large intermediates are not committed.

## Stage map

| Stage | Purpose | Entrypoint | Module / implementation |
|---|---|---|---|
| 01 | Input inventory / provenance | `main_step1.nf` | `modules/01_ingest.nf` |
| 02 | Per-sample QC | `main.nf` | `modules/02_qc.nf` |
| 03 | Normalization | `main.nf` | `modules/03_normalization.nf` |
| 04 | Cohort assembly | `main_step4.nf` | `modules/04_cohort_assembly.nf` |
| 05 | HVG + PCA | `main_step5.nf` | `modules/05_integration.nf` |
| 06 | Harmony candidate | `main_step6.nf` | `modules/06_harmony.nf` |
| 06A | Quantitative integration evaluation | `main_step6a.nf` | `modules/06a_evaluation.nf` |
| 06B | Integration benchmark | `main_step6b.nf` | `modules/06b_benchmark.nf` |
| 07A | TME / T-cell validation | `main_step7a.nf` | `modules/07a_biological_validation.nf` |
| 07B | T-cell expression programs | `main_step7b.nf` | `modules/07b_tcell_state.nf` |
| 07C | Expression-state landscape + KNN graph | `main_step7c.nf` | `modules/07c_tcell_trajectory.nf` |
| 08A | Deep-EVI training | `main_step8a.nf` | `modules/08_deep_evi.nf` |
| 08B | Deep-EVI characterization | `main_step8b.nf` | `modules/08b_deep_evi_characterization.nf` |
| 08C | Ablation | `main_step8c.nf` | `modules/08c_deep_evi_ablation.nf` |
| 08D | Graph diagnostics | `main_step8d.nf` | `modules/08d_deep_evi_graph_diagnostic.nf` |
| 08E | Robustness / construct analysis | `main_step8e.nf` | `modules/08e_deep_evi_independent_validation.nf` |
| 09A | Frozen TCGA molecular surrogate | `main_step9a.nf` | `modules/09a_tcga_signature.nf` |
| 09B | TCGA-BRCA projection | `main_step9b.nf` | `main_step9b.nf` + `bin/run_tcga_brca_09b.py` |
| 09C | Survival / clinical analysis | `main_step9c.nf` | `bin/run_tcga_brca_09c_clinical_validation.py` |
| 09D | Bulk-composition analysis | `main_step9d.nf` | `modules/09d_tcga_bulk_composition.nf` |
| 09E | Biological concordance | `main_step9e.nf` | `modules/09e_tcga_biological_validation.nf` |
| 10A held-out | Frozen held-out benchmark | `main_step10a.nf` | `modules/10a_heldout_benchmark.nf` + `bin/run_10a_heldout_benchmark.py` |\n| 10A preparation/freeze | Held-out expression, signature scoring and freeze | `main_step10a_freeze.nf` | `bin/build_10a_test_expression.py`, `bin/score_10a_reference_signatures.py`, `modules/10a_freeze.nf` |
| 10A-5 | State-axis sensitivity | `main_step10a5.nf` | `modules/10a5_state_axis.nf` |
| 10A-5B | Overlap-controlled sensitivity | `main_step10a5b.nf` | `modules/10a5b_overlap_controlled.nf` |
| 10B | Independent TCGA immune-subtype validation | `main_step10b.nf` | `modules/10b_tcga_immune_validation.nf` |
| 11A–C | Frozen Deep-EVI XAI | `main_step11.nf` | `modules/11_deep_evi_xai.nf` |
| 11D | Gene-level attribution | `main_step11.nf` | `modules/11_deep_evi_xai.nf` |

## Important architecture note

`main.nf` is deliberately the preprocessing entrypoint for the currently implemented Steps 2–3. It is **not** a monolithic Steps 1–11 launcher. Downstream frozen stages are exposed individually through `main_step*.nf`.

This avoids silently mixing frozen analytical checkpoints and makes provenance easier to audit.

## Scientific boundaries

- 07C is an expression-state landscape, not RNA velocity.
- Deep-EVI is an exhaustion-associated state index, not a temporal or causal quantity.
- 10A is held-out construct/state benchmarking.
- 10B is the principal independent biological validation.
- 11 explains frozen model behaviour and does not establish causality.

## Adding a stage

When a new analytical stage is added:

1. add/update its reusable module;
2. add a dedicated `main_step*.nf` entrypoint when independently runnable;
3. add the matching Conda environment;
4. document the input/output contract here;
5. add a representative path to `tests/test_repository_layout.py`;
6. update the README only after the executable structure is stable.
