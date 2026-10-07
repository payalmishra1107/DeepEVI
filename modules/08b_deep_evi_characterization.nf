process DEEP_EVI_CHARACTERIZATION {

    tag "GSE176078 Deep-EVI biological characterization"

    publishDir "${params.outdir}/deep_evi_characterization",
        mode: 'copy',
        overwrite: true

    conda "${projectDir}/envs/deep_evi_characterization.yml"

    input:
    path scores

    output:
    path "*.csv", emit: csv
    path "*.json", emit: json

    script:
    """
    python ${projectDir}/bin/characterize_deep_evi.py \
        --scores ${scores} \
        --output-dir . \
        --high-evi-quantile ${params.high_evi_quantile}
    """
}