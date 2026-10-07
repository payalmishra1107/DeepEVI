process DEEP_EVI_INDEPENDENT_VALIDATION {

    tag "GSE176078 08E independent biological robustness"

    conda "${projectDir}/envs/deep_evi_independent_validation.yml"

    publishDir "${params.outdir}/validation",
        mode: 'copy',
        overwrite: true

    input:
    path landscape
    path scores
    path state_scores
    path split_manifest

    output:
    path "*.csv"
    path "*.json"

    script:

    """
    set -euo pipefail

    python ${projectDir}/bin/validate_deep_evi_independent.py \\
        --scores ${scores} \\
        --state_scores ${state_scores} \\
        --split_manifest ${split_manifest} \\
        --landscape ${landscape} \\
        --outdir . \\
        --cohort GSE176078
    """
}