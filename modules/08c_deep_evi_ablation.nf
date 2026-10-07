process DEEP_EVI_ABLATION {

    tag "GSE176078 Deep-EVI ablation and independence validation"

    publishDir "${params.outdir}/ablation",
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
    python ${projectDir}/bin/run_deep_evi_ablation.py \
        --landscape ${landscape} \
        --edges ${edges} \
        --scores ${scores} \
        --split-manifest ${split_manifest} \
        --output-dir . \
        --epochs ${params.ablation_epochs} \
        --patience ${params.ablation_patience} \
        --hidden-dim ${params.ablation_hidden_dim} \
        --latent-dim ${params.ablation_latent_dim} \
        --dropout ${params.ablation_dropout} \
        --learning-rate ${params.ablation_learning_rate} \
        --weight-decay ${params.ablation_weight_decay} \
        --seed ${params.seed}
    """
}