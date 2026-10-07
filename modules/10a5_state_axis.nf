process ANALYZE_10A5_STATE_AXIS {

    tag "10A5_state_axis"

    label 'DEEP_EVI_10A5'

    conda "${projectDir}/envs/deep_evi_10a_state_axis.yml"

    input:
    path deep_evi_scores
    path reference_scores
    path state_scores

    output:
    path "GSE176078_10A5_cell_state_axis.csv"
    path "GSE176078_10A5_associations.csv"
    path "GSE176078_10A5_partial_associations.csv"
    path "GSE176078_10A5_state_bin_summary.csv"
    path "GSE176078_10A5_subset_state_summary.csv"
    path "GSE176078_10A5_named_state_summary.csv"
    path "GSE176078_10A5_sample_summary.csv"
    path "GSE176078_10A5_sample_level_association.csv"
    path "GSE176078_10A5_subset_monotonicity.csv"
    path "GSE176078_10A5_report.json"

    script:
    """
    python ${projectDir}/bin/analyze_10a_state_axis.py \\
        --deep-evi-scores ${deep_evi_scores} \\
        --reference-scores ${reference_scores} \\
        --state-scores ${state_scores} \\
        --outdir .
    """
}