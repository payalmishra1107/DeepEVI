# DeepEVI Pipeline Architecture

This document defines the canonical reproducibility architecture.

## 1. Execution hierarchy

DeepEVI has four layers:

1. **Canonical orchestration** — `main.nf`
2. **Phase workflows** — `workflows/*.nf`
3. **Reusable Nextflow processes** — `modules/*.nf`
4. **Scientific implementations** — `bin/*.py`

The `main_step*.nf` files are checkpoint/debug interfaces. They are not the primary architecture.

## 2. Canonical scientific graph

```text
GSE176078
   |
01 provenance / inventory
   |
02 QC
   |
03 normalization
   |
04 cohort assembly
   |
05 HVG + PCA
   |
06 Harmony candidate
   |------------------06A quantitative       06B standardized
integration evaluation benchmark
   \------------------/
             |
07A TME / T-cell biological validation
             |
07B T-cell expression programs
             |
07C expression-state landscape + KNN graph
             |
08A Deep-EVI
             |
     +-------+--------+---------+
     |       |        |         |
    08B     08C      08D       08E
character. ablation graph      robustness
                         diagnostic
             |
          FROZEN
       Deep-EVI model
             |
      +------+---------+
      |                |
     09A              10A
 TCGA molecular    held-out benchmark
    bridge              |
      |             10A-5 / 10A-5B
      |
     09B
 TCGA-BRCA projection
      |
   +--+----+-----+
   |       |     |
  09C     09D   09E
 survival bulk  biological
         composition concordance
      |
     10B
 independent TCGA-BRCA
 immune-subtype validation
      |
     11
 frozen-model XAI
```

## 3. Canonical entrypoint

The public reproducibility command is:

```bash
nextflow run main.nf -profile conda,workstation \
  --primary_raw_dir /path/to/GSE176078_RAW \
  --tcga_raw /path/to/tcga_brca/raw/star_counts \
  --tcga_query /path/to/TCGA-BRCA_STAR_Counts_query.json \
  --tcga_clinical /path/to/TCGA-BRCA_clinical.tsv \
  --tcga_manifest /path/to/TCGA-BRCA_STAR_Counts_manifest.tsv \
  --multifile_audit /path/to/multifile_case_file_level_scores.csv \
  --immune_subtypes /path/to/Subtype_Immune_Model_Based.txt \
  --reference_signatures config/10a_benchmark_signatures.tsv \
  --outdir results \
  -resume
```

Human genomic data are supplied by the user and are not redistributed by this repository.

## 4. Phase workflows

| Phase | Workflow | Scientific responsibility |
|---|---|---|
| 01 | `workflows/01_ingestion.nf` | accession/input inventory and provenance |
| 02 | `workflows/02_preprocessing.nf` | QC, normalization, cohort assembly |
| 03 | `workflows/03_integration.nf` | HVG/PCA, Harmony candidate, quantitative evaluation |
| 04 | `workflows/04_tcell_analysis.nf` | TME validation, T-cell programs, state landscape |
| 05 | `workflows/05_deep_evi.nf` | Deep-EVI and robustness/diagnostic analyses |
| 06 | `workflows/06_tcga_validation.nf` | frozen molecular bridge and TCGA analyses |
| 07 | `workflows/07_heldout_validation.nf` | held-out benchmarking and 10B validation |
| 08 | `workflows/08_xai.nf` | frozen-model explainability |
| 09 | `workflows/09_audit.nf` | integrity audit |

## 5. Stage/checkpoint entrypoints

The individual `main_step*.nf` files remain available for checkpoint reproduction, debugging and review.

Steps 1–11 now have explicit checkpoint interfaces, including:

- `main_step1.nf`
- `main_step2.nf`
- `main_step3.nf`
- `main_step4.nf`
- `main_step5.nf`
- `main_step6.nf`
- `main_step6a.nf`
- `main_step6b.nf`
- `main_step7a.nf`
- `main_step7b.nf`
- `main_step7c.nf`
- `main_step8a.nf` through `main_step8e.nf`
- `main_step9a.nf` through `main_step9e.nf`
- `main_step10a.nf`, `main_step10a5.nf`, `main_step10a5b.nf`, `main_step10a_freeze.nf`, `main_step10b.nf`
- `main_step11.nf`

## 6. Reproducibility boundaries

### Primary cohort

All cell-level model development originates from GSE176078.

### Deep-EVI freeze

The Deep-EVI model and the six-program-to-TCGA molecular bridge are frozen before TCGA biological validation.

### Held-out validation

10A uses the exact held-out sample split produced by 08A. Test cells are not used for training or model selection.

### Independent validation

10B uses an independent TCGA-BRCA cohort and published immune-subtype labels. TCGA labels are not used to train, refit, select features or optimize cutoffs.

### Explainability

Step 11 operates on the frozen Deep-EVI model. It explains model behaviour and does not establish causality.

## 7. Scientific interpretation boundaries

- 07C is an expression-state landscape, not RNA velocity.
- Deep-EVI is a continuous exhaustion-associated state index, not a temporal trajectory.
- The exhaustion target is model-derived; high recovery against that target is not independent biological validation.
- 10B is the principal independent biological validation.
- XAI is interpretive, not causal.

## 8. Design rule

New analytical work must enter through:

```text
main.nf
  -> workflows/
  -> modules/
  -> bin/
  -> envs/
```

A notebook or manually executed Python script must not become a required step in the scientific chain.
