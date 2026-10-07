process RUN_10A_HELDOUT_BENCHMARK {

    tag "10A_heldout_benchmark"

    conda "${projectDir}/envs/deep_evi_10a_benchmark.yml"

    input:
    path deep_evi_scores
    path state_scores
    path reference_scores
    path reference_metadata

    output:
    path "GSE176078_10A_benchmark_cell_table.csv", emit: cell_table
    path "GSE176078_10A_exhaustion_target_associations.csv", emit: exhaustion_associations
    path "GSE176078_10A_program_associations.csv", emit: program_associations
    path "GSE176078_10A_deep_evi_reference_associations.csv", emit: reference_associations
    path "GSE176078_10A_subset_concordance.csv", emit: subset_concordance
    path "GSE176078_10A_sample_summary.csv", emit: sample_summary
    path "GSE176078_10A_report.json", emit: report
    path "GSE176078_10A_reference_metadata.csv", emit: reference_metadata

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
