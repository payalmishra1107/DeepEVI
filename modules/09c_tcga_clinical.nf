process TCGA_BRCA_09C {
    tag "TCGA-BRCA-09C clinical analysis"
    publishDir "${params.outdir}/step9c/clinical_validation", mode: 'copy', overwrite: true
    cpus 4
    memory '8 GB'
    time '4h'
    maxForks 1
    conda "${projectDir}/envs/deep_evi_tcga_clinical.yml"
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
    python ${projectDir}/bin/run_tcga_brca_09c_clinical_validation.py \
        --scores ${scores} \
        --survival ${survival} \
        --clinical ${clinical} \
        --multifile-audit ${multifile_audit} \
        --outdir .
    """
}
