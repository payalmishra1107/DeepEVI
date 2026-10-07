nextflow.enable.dsl=2

params {
    scores = null
    state_scores = null
    splitManifest = null
    landscape = null
    outdir = "results_step8e"
}

include {
    DEEP_EVI_INDEPENDENT_VALIDATION
} from './modules/08e_deep_evi_independent_validation.nf'


workflow {

    if (!params.scores) {
        error "Missing --scores"
    }

    if (!params.state_scores) {
        error "Missing --state_scores"
    }

    if (!params.splitManifest) {
        error "Missing --splitManifest"
    }

    if (!params.landscape) {
        error "Missing --landscape"
    }

    scores_ch = Channel.fromPath(
        params.scores,
        checkIfExists: true
    )

    state_scores_ch = Channel.fromPath(
        params.state_scores,
        checkIfExists: true
    )

    split_ch = Channel.fromPath(
        params.splitManifest,
        checkIfExists: true
    )

    landscape_ch = Channel.fromPath(
        params.landscape,
        checkIfExists: true
    )

    DEEP_EVI_INDEPENDENT_VALIDATION(
        landscape_ch,
        scores_ch,
        state_scores_ch,
        split_ch
    )
}