process TCELL_STATE_ANALYSIS {

    tag "GSE176078 T-cell expression-state analysis"

    publishDir "${params.outdir}/tcell_state",
        mode: 'copy',
        overwrite: true,
        pattern: "tcell_state/*"

    conda "${projectDir}/envs/tcell_state.yml"

    input:
    path integrated

    output:
    path "tcell_state/*", emit: state_results

    script:
    """
    mkdir -p tcell_state

    python ${projectDir}/bin/analyze_tcell_state.py \
        --input ${integrated} \
        --output-dir tcell_state \
        --celltype-key ${params.celltype_key} \
        --tcell-label ${params.tcell_label} \
        --sample-key ${params.batch_key} \
        --subtype-key ${params.subtype_key} \
        --subset-key ${params.subset_key}
    """
}