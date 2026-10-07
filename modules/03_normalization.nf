process NORMALIZE_SAMPLE_V2 {

    tag { sample_id }

    publishDir "${params.outdir}/normalized",
        mode: 'copy',
        overwrite: true,
        pattern: "*_normalized.h5ad",
        saveAs: { "${sample_id}_normalized.h5ad" }

    publishDir "${params.outdir}/normalization_summary",
        mode: 'copy',
        overwrite: true,
        pattern: "*_normalization_summary.json",
        saveAs: { "${sample_id}_normalization_summary.json" }

    conda "${projectDir}/envs/normalization.yml"

    input:
    tuple val(sample_id), path(h5ad)

    output:
    tuple val(sample_id), path("*_normalized.h5ad"), emit: normalized
    tuple val(sample_id), path("*_normalization_summary.json"), emit: summary

    script:
    normalized_file = "${sample_id}_normalized.h5ad"
    summary_file = "${sample_id}_normalization_summary.json"

    """
    python ${projectDir}/bin/normalize_sample.py \
        --input ${h5ad} \
        --sample-id ${sample_id} \
        --output ${normalized_file} \
        --summary ${summary_file} \
        --target-sum ${params.target_sum}
    """
}