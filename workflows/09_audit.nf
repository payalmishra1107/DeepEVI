nextflow.enable.dsl=2
process AUDIT_PIPELINE {
    tag "DeepEVI master integrity audit"
    publishDir "${params.outdir}/audit", mode: 'copy', overwrite: true
    conda "\${projectDir}/envs/ingestion.yml"
    input:
    path repo_root
    output:
    path "audit_pipeline_integrity.txt", emit: report
    script:
    """
    cd ${repo_root}
    python bin/audit_pipeline_integrity.py > audit_pipeline_integrity.txt
    """
}
workflow AUDIT {
    take:
    repo_root
    main:
    AUDIT_PIPELINE(repo_root)
    emit:
    report = AUDIT_PIPELINE.out.report
}
