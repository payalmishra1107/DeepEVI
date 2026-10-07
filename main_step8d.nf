nextflow.enable.dsl=2

params {
    landscape = null
    edges = null
    scores = null
    splitManifest = null

    outdir = "results_step8d"

    diagnostic_epochs = 150
    diagnostic_patience = 20

    diagnostic_hidden_dim = 64
    diagnostic_latent_dim = 32
    diagnostic_dropout = 0.15
    diagnostic_learning_rate = 0.001
    diagnostic_weight_decay = 0.0001

    seed = 20260924
}

include {
    DEEP_EVI_GRAPH_DIAGNOSTIC
} from './modules/08d_deep_evi_graph_diagnostic.nf'

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

    if (!params.splitManifest) {
        error "Missing --splitManifest"
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

    split_ch = Channel.fromPath(
        params.splitManifest,
        checkIfExists: true
    )

    DEEP_EVI_GRAPH_DIAGNOSTIC(
        landscape_ch,
        edges_ch,
        scores_ch,
        split_ch
    )
}