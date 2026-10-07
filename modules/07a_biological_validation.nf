process BIOLOGICAL_VALIDATION {

    tag "GSE176078 TME/T-cell biological validation"

    publishDir "${params.outdir}/validation",
        mode: 'copy',
        overwrite: true

    conda "${projectDir}/envs/biological_validation.yml"

    input:
    path integrated

    output:
    path "validation/*", emit: validation

    script:
    """
    python ${projectDir}/bin/validate_tme_tcell.py \
        --input ${integrated} \
        --output-dir validation \
        --batch-key ${params.batch_key} \
        --subtype-key ${params.subtype_key} \
        --n-neighbors ${params.n_neighbors}
    """
}