process RUN_10A_HELDOUT_BENCHMARK {

    tag "10A_heldout_benchmark"

    conda "${projectDir}/envs/deep_evi_10a_benchmark.yml"

    input:
    path deep_evi_scores
    path state_scores
    path reference_scores
    path reference_metadata

    output:
    path "GSE176078_10A_test_cell_scores.csv"
    path "GSE176078_10A_target_correlations.csv"
    path "GSE176078_10A_program_correlations.csv"
    path "GSE176078_10A_pairwise_correlations.csv"
    path "GSE176078_10A_subset_auc.csv"
    path "GSE176078_10A_sample_summary.csv"
    path "GSE176078_10A_report.json"

    script:
    """
    python ${projectDir}/bin/run_10a_heldout_benchmark.py \
        --deep-evi-scores ${deep_evi_scores} \
        --state-scores ${state_scores} \
        --reference-scores ${reference_scores} \
        --reference-metadata ${reference_metadata} \
        --outdir .
    """
}
