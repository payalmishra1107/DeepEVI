nextflow.enable.dsl=2

params {
    scores = null

    outdir = "results_step8b"

    high_evi_quantile = 0.75
}

include {
    DEEP_EVI_CHARACTERIZATION
} from './modules/08b_deep_evi_characterization.nf'


workflow {

    if (!params.scores) {
        error "Missing --scores"
    }

    scores_ch = Channel.fromPath(
        params.scores,
        checkIfExists: true
    )

    DEEP_EVI_CHARACTERIZATION(scores_ch)
}