nextflow.enable.dsl=2
include { FEATURE_PCA_COHORT } from '../modules/05_integration.nf'
include { HARMONY_INTEGRATION } from '../modules/06_harmony.nf'
include { EVALUATE_INTEGRATION } from '../modules/06a_evaluation.nf'
include { BENCHMARK_INTEGRATION } from '../modules/06b_benchmark.nf'
workflow INTEGRATION {
    take:
    cohort
    main:
    FEATURE_PCA_COHORT(cohort)
    HARMONY_INTEGRATION(FEATURE_PCA_COHORT.out.latent)
    EVALUATE_INTEGRATION(HARMONY_INTEGRATION.out.integrated)
    BENCHMARK_INTEGRATION(HARMONY_INTEGRATION.out.integrated)
    emit:
    latent = FEATURE_PCA_COHORT.out.latent
    integrated = HARMONY_INTEGRATION.out.integrated
    integration_metrics = EVALUATE_INTEGRATION.out
    integration_benchmark = BENCHMARK_INTEGRATION.out
}
