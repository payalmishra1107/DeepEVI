process TCELL_TRAJECTORY {
    tag "GSE176078 T-cell state landscape"

    publishDir "${params.outdir}/trajectory",
        mode: 'copy',
        overwrite: true,
        pattern: "*.csv"

    publishDir "${params.outdir}/trajectory",
        mode: 'copy',
        overwrite: true,
        pattern: "*.json"

    conda "${projectDir}/envs/tcell_trajectory.yml"

    input:
    path integrated
    path scores

    output:
    path "trajectory/*.csv", emit: csv_outputs
    path "trajectory/*.json", emit: json_outputs

    script:
    """
    mkdir -p trajectory

    python ${projectDir}/bin/build_tcell_trajectory.py \
        --input ${integrated} \
        --scores ${scores} \
        --output-dir trajectory \
        --sample-key ${params.batch_key} \
        --subtype-key ${params.subtype_key} \
        --subset-key ${params.subset_key} \
        --n-components ${params.n_components} \
        --n-neighbors ${params.n_neighbors}
    """
}