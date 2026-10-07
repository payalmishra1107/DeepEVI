nextflow.enable.dsl=2

/*
 * Deep-EVI Step 09C
 *
 * Clinical/prognostic validation of the frozen TCGA-BRCA
 * molecular surrogate.
 *
 * No signature refitting.
 * No outcome-derived cutoff.
 * No GNN retraining.
 */

params.scores =
    "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_scores.csv"

params.survival =
    "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_survival_summary.csv"

params.clinical =
    "${baseDir}/../tcga_brca/clinical/TCGA-BRCA_clinical.tsv"

params.multifile_audit =
    "${baseDir}/results_step9b/multifile_audit/multifile_case_file_level_scores.csv"


process TCGA_BRCA_09C {

    tag "TCGA-BRCA-09C"

    publishDir "${baseDir}/results_step9c/clinical_validation",
        mode: 'copy',
        overwrite: true

    cpus 4
    memory '8 GB'
    time '4h'
    maxForks 1

    conda "${baseDir}/envs/deep_evi_tcga_clinical.yml"

    input:
    path scores
    path survival
    path clinical
    path multifile_audit

    output:
    path "*.json", emit: reports
    path "*.csv", emit: tables

    script:
    """
    python ${baseDir}/bin/run_tcga_brca_09c_clinical_validation.py \
        --scores ${scores} \
        --survival ${survival} \
        --clinical ${clinical} \
        --multifile-audit ${multifile_audit} \
        --outdir .
    """
}


workflow {

    scores_ch = Channel.fromPath(
        params.scores,
        checkIfExists: true
    )

    survival_ch = Channel.fromPath(
        params.survival,
        checkIfExists: true
    )

    clinical_ch = Channel.fromPath(
        params.clinical,
        checkIfExists: true
    )

    multifile_ch = Channel.fromPath(
        params.multifile_audit,
        checkIfExists: true
    )

    TCGA_BRCA_09C(
        scores_ch,
        survival_ch,
        clinical_ch,
        multifile_ch
    )
}