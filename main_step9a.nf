nextflow.enable.dsl=2

params {
    scores = null
    state_scores = null
    splitManifest = null
    outdir = "results_step9a"
}

include {
    BUILD_TCGA_SIGNATURE
} from './modules/09a_tcga_signature.nf'


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

    BUILD_TCGA_SIGNATURE(
        scores_ch,
        state_scores_ch,
        split_ch
    )
}