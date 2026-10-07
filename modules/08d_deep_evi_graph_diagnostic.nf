process DEEP_EVI_GRAPH_DIAGNOSTIC {

    tag "GSE176078 Deep-EVI graph architecture diagnostic"

    publishDir "${params.outdir}/diagnostic",
        mode: 'copy',
        overwrite: true

    conda "${projectDir}/envs/deep_evi_ablation.yml"

    input:
    path landscape
    path edges
    path scores
    path split_manifest

    output:
    path "*.csv", emit: csv
    path "*.json", emit: json
    path "*.pt", emit: models

    script:
    """
    python ${projectDir}/bin/run_deep_evi_graph_diagnostic.py \
        --landscape ${landscape} \
        --edges ${edges} \
        --scores ${scores} \
        --split-manifest ${split_manifest} \
        --output-dir . \
        --epochs ${params.diagnostic_epochs} \
        --patience ${params.diagnostic_patience} \
        --hidden-dim ${params.diagnostic_hidden_dim} \
        --latent-dim ${params.diagnostic_latent_dim} \
        --dropout ${params.diagnostic_dropout} \
        --learning-rate ${params.diagnostic_learning_rate} \
        --weight-decay ${params.diagnostic_weight_decay} \
        --seed ${params.seed}
    """
}