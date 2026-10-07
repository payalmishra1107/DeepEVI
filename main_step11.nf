nextflow.enable.dsl=2

include {
    DEEP_EVI_XAI
    DEEP_EVI_XAI_11D
} from './modules/11_deep_evi_xai.nf'


params.scores = "${projectDir}/results_step8a/deep_evi/GSE176078_deep_evi_scores.csv"
params.landscape = "${projectDir}/results_step7c/trajectory/trajectory/tcell_state_landscape.csv"
params.edges = "${projectDir}/results_step7c/trajectory/trajectory/tcell_state_knn_edges.csv"
params.splitManifest = "${projectDir}/results_step8a/deep_evi/GSE176078_deep_evi_split_manifest.csv"
params.model = "${projectDir}/results_step8a/deep_evi/GSE176078_deep_evi_model.pt"
params.signature = "${projectDir}/results_step9a/signature/GSE176078_09A_frozen_signature.json"

workflow {

    scores_ch = Channel.fromPath(
        params.scores,
        checkIfExists: true
    )

    landscape_ch = Channel.fromPath(
        params.landscape,
        checkIfExists: true
    )

    edges_ch = Channel.fromPath(
        params.edges,
        checkIfExists: true
    )

    split_manifest_ch = Channel.fromPath(
        params.splitManifest,
        checkIfExists: true
    )

    model_ch = Channel.fromPath(
        params.model,
        checkIfExists: true
    )

    signature_ch = Channel.fromPath(
        params.signature,
        checkIfExists: true
    )

    /*
     * 11A–C:
     * Frozen GNN explainability
     */
    DEEP_EVI_XAI(
        scores_ch,
        landscape_ch,
        edges_ch,
        split_manifest_ch,
        model_ch,
        signature_ch
    )

    /*
     * 11D:
     * Frozen 09A gene-level molecular attribution
     *
     * This is deterministic reconstruction from the
     * frozen signature and does not refit the model.
     */
    DEEP_EVI_XAI_11D(
        signature_ch
    )
}