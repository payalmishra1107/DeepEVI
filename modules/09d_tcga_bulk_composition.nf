nextflow.enable.dsl=2

process TCGA_BRCA_09D {
    tag "TCGA-BRCA-09D-bulk-composition"

    publishDir "${baseDir}/results_step9d/bulk_composition",
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

    output:
    path "*.json", emit: reports
    path "*.csv", emit: tables

    script:
    """
    python ${baseDir}/bin/run_tcga_brca_09d_bulk_composition.py \
        --scores ${scores} \
        --survival ${survival} \
        --clinical ${clinical} \
        --outdir .
    """
}

workflow {
    scores_ch = Channel.fromPath(params.scores, checkIfExists: true)
    survival_ch = Channel.fromPath(params.survival, checkIfExists: true)
    clinical_ch = Channel.fromPath(params.clinical, checkIfExists: true)

    TCGA_BRCA_09D(scores_ch, survival_ch, clinical_ch)
}