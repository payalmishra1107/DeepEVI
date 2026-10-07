process DEEP_EVI_XAI {
    tag "11A_C_frozen_xai"

    label 'DEEP_EVI_XAI'

    publishDir "${projectDir}/results_step11/xai",
        mode: 'copy',
        overwrite: true

    conda "${projectDir}/envs/deepevi_xai.yml"

    input:
    path scores
    path landscape
    path edges
    path split_manifest
    path model
    path signature

    output:
    path "GSE176078_11_program_attribution.csv"
    path "GSE176078_11_program_attribution_summary.csv"
    path "GSE176078_11_graph_edge_attribution.csv"
    path "GSE176078_11_graph_edge_summary.csv"
    path "GSE176078_11_molecular_attribution.csv"
    path "GSE176078_11_cell_level_results.csv"
    path "GSE176078_11_report.json"

    script:
    """
    python ${projectDir}/bin/run_deep_evi_xai.py \
        --scores ${scores} \
        --landscape ${landscape} \
        --edges ${edges} \
        --splitManifest ${split_manifest} \
        --model ${model} \
        --signature ${signature} \
        --outdir .
    """
}


process DEEP_EVI_XAI_11D {
    tag "11D_frozen_gene_attribution"

    label 'DEEP_EVI_XAI'

    publishDir "${projectDir}/results_step11/gene_attribution",
        mode: 'copy',
        overwrite: true

    conda "${projectDir}/envs/deepevi_xai.yml"

    input:
    path signature

    output:
    path "GSE176078_11D_gene_molecular_attribution.csv"
    path "GSE176078_11D_gene_program_contributions.csv"
    path "GSE176078_11D_program_molecular_attribution.csv"
    path "GSE176078_11D_gene_attribution_report.json"

    script:
    """
    python ${projectDir}/bin/build_11d_gene_attribution.py \
        --signature ${signature} \
        --outdir .
    """
}