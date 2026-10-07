nextflow.enable.dsl=2

params {
    landscape = null
    edges = null
    scores = null
    splitManifest = null

    outdir = "results_step8c"

    ablation_epochs = 150
    ablation_patience = 20

    ablation_hidden_dim = 64
    ablation_latent_dim = 32
    ablation_dropout = 0.15
    ablation_learning_rate = 0.001
    ablation_weight_decay = 0.0001

    seed = 20260924
}

include {
    DEEP_EVI_ABLATION
} from './modules/08c_deep_evi_ablation.nf'

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

    DEEP_EVI_ABLATION(
        landscape_ch,
        edges_ch,
        scores_ch,
        split_ch
    )
}