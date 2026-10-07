process EVALUATE_INTEGRATION {

    tag "GSE176078 pre-vs-Harmony integration evaluation"

    publishDir "${params.outdir}/metrics",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_integration_metrics.csv"

    publishDir "${params.outdir}/reports",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_integration_evaluation.json"

    conda "${projectDir}/envs/integration_eval.yml"

    input:
    path integrated

    output:
    path "GSE176078_integration_metrics.csv", emit: metrics
    path "GSE176078_integration_evaluation.json", emit: report

    script:
    """
    python ${projectDir}/bin/evaluate_integration.py \
        --input ${integrated} \
        --output-csv GSE176078_integration_metrics.csv \
        --output-json GSE176078_integration_evaluation.json \
        --batch-key ${params.batch_key} \
        --label-key ${params.label_key}
    """
}