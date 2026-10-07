nextflow.enable.dsl=2

params {
    landscape = null
    edges = null
    scores = null

    outdir = "results_step8a"

    deep_evi_hidden_dim = 64
    deep_evi_latent_dim = 32
    deep_evi_dropout = 0.15
    deep_evi_learning_rate = 0.001
    deep_evi_weight_decay = 0.0001
    deep_evi_epochs = 300
    deep_evi_patience = 30

    seed = 20260922
}

include {
    DEEP_EVI
} from './modules/08_deep_evi.nf'

workflow {

    if (!params.landscape) {
        error "Missing --landscape"
    }

    if (!params.edges) {
        error "Missing --edges"
    }

    if (!params.scores) {
        error "Missing --scores"
    }

    landscape_ch = Channel.fromPath(
        params.landscape,
        checkIfExists: true
    )

    edges_ch = Channel.fromPath(
        params.edges,
        checkIfExists: true
    )

    scores_ch = Channel.fromPath(
        params.scores,
        checkIfExists: true
    )

    DEEP_EVI(
        landscape_ch,
        edges_ch,
        scores_ch
    )
}