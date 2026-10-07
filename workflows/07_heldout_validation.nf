nextflow.enable.dsl=2
include { BUILD_10A_TEST_EXPRESSION } from '../modules/10a_test_expression.nf'
include { SCORE_10A_REFERENCE_SIGNATURES } from '../modules/10a_reference_scoring.nf'
include { RUN_10A_HELDOUT_BENCHMARK } from '../modules/10a_heldout_benchmark.nf'
include { ANALYZE_10A5_STATE_AXIS } from '../modules/10a5_state_axis.nf'
include { ANALYZE_10A5B_OVERLAP_CONTROLLED } from '../modules/10a5b_overlap_controlled.nf'
include { FREEZE_10A_BENCHMARK } from '../modules/10a_freeze.nf'
include { VALIDATE_10B_TCGA_IMMUNE_SUBTYPES } from '../modules/10b_tcga_immune_validation.nf'

workflow HELDOUT_AND_EXTERNAL_VALIDATION {
    take:
    source_h5ad
    deep_scores
    state_scores
    split_manifest
    reference_signatures
    tcga_scores
    tcga_case_inventory
    immune_subtypes

    main:
    BUILD_10A_TEST_EXPRESSION(source_h5ad, split_manifest)

    SCORE_10A_REFERENCE_SIGNATURES(
        BUILD_10A_TEST_EXPRESSION.out.expression,
        reference_signatures
    )

    RUN_10A_HELDOUT_BENCHMARK(
        deep_scores,
        state_scores,
        SCORE_10A_REFERENCE_SIGNATURES.out.scores,
        SCORE_10A_REFERENCE_SIGNATURES.out.metadata
    )

    ANALYZE_10A5_STATE_AXIS(
        deep_scores,
        SCORE_10A_REFERENCE_SIGNATURES.out.scores,
        state_scores
    )

    ANALYZE_10A5B_OVERLAP_CONTROLLED(
        BUILD_10A_TEST_EXPRESSION.out.expression,
        deep_scores,
        state_scores
    )

    FREEZE_10A_BENCHMARK(
        BUILD_10A_TEST_EXPRESSION.out.expression,
        SCORE_10A_REFERENCE_SIGNATURES.out.scores,
        SCORE_10A_REFERENCE_SIGNATURES.out.metadata,
        RUN_10A_HELDOUT_BENCHMARK.out.report.first(),
        ANALYZE_10A5_STATE_AXIS.out.filter { it.name == 'GSE176078_10A5_associations.csv' }.first(),
        ANALYZE_10A5_STATE_AXIS.out.filter { it.name == 'GSE176078_10A5_partial_associations.csv' }.first(),
        ANALYZE_10A5B_OVERLAP_CONTROLLED.out.filter { it.name == 'GSE176078_10A5B_overlap_controlled_associations.csv' }.first(),
        ANALYZE_10A5B_OVERLAP_CONTROLLED.out.filter { it.name == 'GSE176078_10A5B_partial_associations.csv' }.first(),
        ANALYZE_10A5B_OVERLAP_CONTROLLED.out.filter { it.name == 'GSE176078_10A5B_report.json' }.first()
    )

    VALIDATE_10B_TCGA_IMMUNE_SUBTYPES(
        tcga_scores,
        immune_subtypes,
        tcga_case_inventory
    )

    emit:
    test_expression = BUILD_10A_TEST_EXPRESSION.out.expression
    benchmark = RUN_10A_HELDOUT_BENCHMARK.out
    state_axis = ANALYZE_10A5_STATE_AXIS.out
    overlap_controlled = ANALYZE_10A5B_OVERLAP_CONTROLLED.out
    frozen_manifest = FREEZE_10A_BENCHMARK.out
    independent_tcga = VALIDATE_10B_TCGA_IMMUNE_SUBTYPES.out
}
