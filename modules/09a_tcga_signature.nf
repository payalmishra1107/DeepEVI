process BUILD_TCGA_SIGNATURE {

    tag "GSE176078 09A frozen TCGA-compatible Deep-EVI signature"

    conda "${projectDir}/envs/deep_evi_tcga_signature.yml"

    publishDir "${params.outdir}/signature",
        mode: 'copy',
        overwrite: true

    input:
    path scores
    path state_scores
    path split_manifest

    output:
    path "*.csv"
    path "*.json"

    script:

    """
    set -euo pipefail

    python ${projectDir}/bin/build_deep_evi_tcga_signature.py \\
        --scores ${scores} \\
        --state_scores ${state_scores} \\
        --split_manifest ${split_manifest} \\
        --outdir .
    """
}