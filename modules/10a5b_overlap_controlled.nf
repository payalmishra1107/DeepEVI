process ANALYZE_10A5B_OVERLAP_CONTROLLED {

    tag "10A5B_overlap_controlled"

    label 'DEEP_EVI_10A5B'

    conda "${projectDir}/envs/deep_evi_10a5b_overlap.yml"

    input:
    path expression
    path deep_evi_scores
    path state_scores

    output:
    path "GSE176078_10A5B_overlap_controlled_associations.csv"
    path "GSE176078_10A5B_overlap_controlled_cell_scores.csv"
    path "GSE176078_10A5B_partial_associations.csv"
    path "GSE176078_10A5B_report.json"
    path "GSE176078_10A5B_sample_summary.csv"
    path "GSE176078_10A5B_signature_inventory.csv"
    path "GSE176078_10A5B_state_bins.csv"
    path "GSE176078_10A5B_subset_auc.csv"

    script:
    """
    python ${projectDir}/bin/analyze_10a5b_overlap_controlled.py \\
        --expression ${expression} \\
        --deep-evi ${deep_evi_scores} \\
        --state ${state_scores} \\
        --output-dir .
    """
}