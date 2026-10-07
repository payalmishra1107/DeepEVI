nextflow.enable.dsl=2

/*
 * Deep-EVI Step 10A
 * Frozen held-out benchmark.
 */

params {
    deep_evi_scores = null
    state_scores = null
    reference_scores = null
    reference_metadata = null
    outdir = "results_step10a"
}

include {
    RUN_10A_HELDOUT_BENCHMARK
} from './modules/10a_heldout_benchmark.nf'

workflow {
    if (!params.deep_evi_scores) error "Missing --deep_evi_scores"
    if (!params.state_scores) error "Missing --state_scores"
    if (!params.reference_scores) error "Missing --reference_scores"
    if (!params.reference_metadata) error "Missing --reference_metadata"

    deep_ch = Channel.fromPath(params.deep_evi_scores, checkIfExists: true)
    state_ch = Channel.fromPath(params.state_scores, checkIfExists: true)
    reference_ch = Channel.fromPath(params.reference_scores, checkIfExists: true)
    metadata_ch = Channel.fromPath(params.reference_metadata, checkIfExists: true)

    RUN_10A_HELDOUT_BENCHMARK(
        deep_ch,
        state_ch,
        reference_ch,
        metadata_ch
    )
}
