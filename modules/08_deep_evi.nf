process DEEP_EVI {

    tag "GSE176078 Deep-EVI graph learning"

    publishDir "${params.outdir}/deep_evi",
        mode: 'copy',
        overwrite: true,
        pattern: "*.csv"

    publishDir "${params.outdir}/deep_evi",
        mode: 'copy',
        overwrite: true,
        pattern: "*.json"

    publishDir "${params.outdir}/deep_evi",
        mode: 'copy',
        overwrite: true,
        pattern: "*.pt"

    conda "${projectDir}/envs/deep_evi.yml"

    input:
    path landscape
    path edges
    path scores

    output:
    path "*.csv", emit: csv
    path "*.json", emit: json
    path "*.pt", emit: model

    script:
    """
    python ${projectDir}/bin/train_deep_evi.py \
        --landscape ${landscape} \
        --edges ${edges} \
        --scores ${scores} \
        --output-dir . \
        --hidden-dim ${params.deep_evi_hidden_dim} \
        --latent-dim ${params.deep_evi_latent_dim} \
        --dropout ${params.deep_evi_dropout} \
        --learning-rate ${params.deep_evi_learning_rate} \
        --weight-decay ${params.deep_evi_weight_decay} \
        --epochs ${params.deep_evi_epochs} \
        --patience ${params.deep_evi_patience} \
        --seed ${params.seed}
    """
}