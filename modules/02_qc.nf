process EXTRACT_QC_SAMPLE {

    tag { sample_id }

    publishDir "${params.outdir}/h5ad", mode: 'copy', overwrite: true,
        saveAs: { "filtered_${sample_id}.h5ad" }

    publishDir "${params.outdir}/qc_metrics", mode: 'copy', overwrite: true,
        pattern: "sample_qc_metrics.csv",
        saveAs: { "${sample_id}_qc_metrics.csv" }

    publishDir "${params.outdir}/qc_summary", mode: 'copy', overwrite: true,
        pattern: "*_qc_summary.json",
        saveAs: { "${sample_id}_qc_summary.json" }

    conda "${projectDir}/envs/qc.yml"

    input:
    tuple val(sample_id), path(archive)

    output:
    tuple val(sample_id), path("sample.h5ad"), emit: h5ad
    path "sample_qc_metrics.csv", emit: qc
    path "*_qc_summary.json", emit: summary

    script:
    """
    python ${projectDir}/bin/process_sample.py \
        --archive ${archive} \
        --sample ${sample_id} \
        --min-genes ${params.min_genes} \
        --max-pct-mito ${params.max_pct_mito} \
        --h5ad sample.h5ad \
        --qc sample_qc_metrics.csv \
        --summary sample_qc_summary.json

    mv sample_qc_summary.json ${sample_id}_qc_summary.json
    """
}


process AGGREGATE_QC {

    tag "GSE176078 cohort QC aggregation"

    publishDir "${params.outdir}/cohort_qc", mode: 'copy', overwrite: true

    conda "${projectDir}/envs/qc.yml"

    input:
    path summaries

    output:
    path "GSE176078_cohort_qc_summary.csv", emit: cohort_csv
    path "GSE176078_cohort_qc_summary.json", emit: cohort_json

    script:
    """
    mkdir summaries

    for f in *.json; do
        cp "\$f" "summaries/\$f"
    done

    python ${projectDir}/bin/aggregate_qc.py \
        --input summaries \
        --csv GSE176078_cohort_qc_summary.csv \
        --json GSE176078_cohort_qc_summary.json
    """
}