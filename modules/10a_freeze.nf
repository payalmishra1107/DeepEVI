process FREEZE_10A_BENCHMARK {

    tag "10A_frozen_benchmark"

    label 'DEEP_EVI_10A_FREEZE'

    input:
    path test_expression
    path reference_scores
    path reference_metadata
    path benchmark_report
    path state_axis_associations
    path state_axis_partial
    path overlap_associations
    path overlap_partial
    path overlap_report

    output:
    path "GSE176078_10A_frozen_manifest.json"
    path "GSE176078_10A_freeze_summary.txt"

    script:
    """
    python ${projectDir}/bin/freeze_10a_benchmark.py \\
        --test-expression ${test_expression} \\
        --reference-scores ${reference_scores} \\
        --reference-metadata ${reference_metadata} \\
        --benchmark-report ${benchmark_report} \\
        --state-axis-associations ${state_axis_associations} \\
        --state-axis-partial ${state_axis_partial} \\
        --overlap-associations ${overlap_associations} \\
        --overlap-partial ${overlap_partial} \\
        --overlap-report ${overlap_report} \\
        --outdir .
    """
}