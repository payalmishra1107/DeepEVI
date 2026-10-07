nextflow.enable.dsl=2

include {
    FREEZE_10A_BENCHMARK
} from './modules/10a_freeze.nf'


params.test_expression = "${projectDir}/results_step10a/test_expression/GSE176078_10A_test_Tcells.h5ad"

params.reference_scores = "${projectDir}/results_step10a/reference_scores_v2/GSE176078_10A_reference_signature_scores.csv"

params.reference_metadata = "${projectDir}/results_step10a/reference_scores_v2/GSE176078_10A_reference_signature_metadata.csv"

params.benchmark_report = "${projectDir}/results_step10a/benchmark_v2/GSE176078_10A_report.json"

params.state_axis_associations = "${projectDir}/results_step10a5/GSE176078_10A5_associations.csv"

params.state_axis_partial = "${projectDir}/results_step10a5/GSE176078_10A5_partial_associations.csv"

params.overlap_associations = "${projectDir}/results_step10a5b/overlap_controlled/GSE176078_10A5B_overlap_controlled_associations.csv"

params.overlap_partial = "${projectDir}/results_step10a5b/overlap_controlled/GSE176078_10A5B_partial_associations.csv"

params.overlap_report = "${projectDir}/results_step10a5b/overlap_controlled/GSE176078_10A5B_report.json"


workflow {

    test_expression_ch = Channel.fromPath(
        params.test_expression,
        checkIfExists: true
    )

    reference_scores_ch = Channel.fromPath(
        params.reference_scores,
        checkIfExists: true
    )

    reference_metadata_ch = Channel.fromPath(
        params.reference_metadata,
        checkIfExists: true
    )

    benchmark_report_ch = Channel.fromPath(
        params.benchmark_report,
        checkIfExists: true
    )

    state_axis_assoc_ch = Channel.fromPath(
        params.state_axis_associations,
        checkIfExists: true
    )

    state_axis_partial_ch = Channel.fromPath(
        params.state_axis_partial,
        checkIfExists: true
    )

    overlap_assoc_ch = Channel.fromPath(
        params.overlap_associations,
        checkIfExists: true
    )

    overlap_partial_ch = Channel.fromPath(
        params.overlap_partial,
        checkIfExists: true
    )

    overlap_report_ch = Channel.fromPath(
        params.overlap_report,
        checkIfExists: true
    )

    FREEZE_10A_BENCHMARK(
        test_expression_ch,
        reference_scores_ch,
        reference_metadata_ch,
        benchmark_report_ch,
        state_axis_assoc_ch,
        state_axis_partial_ch,
        overlap_assoc_ch,
        overlap_partial_ch,
        overlap_report_ch
    )
}