nextflow.enable.dsl=2

include {
    ANALYZE_10A5_STATE_AXIS
} from './modules/10a5_state_axis.nf'


params.deep_evi_scores = "${projectDir}/results_step8a/deep_evi/GSE176078_deep_evi_scores.csv"
params.reference_scores = "${projectDir}/results_step10a/reference_scores_v2/GSE176078_10A_reference_signature_scores.csv"
params.state_scores = "${projectDir}/results_step7b/tcell_state/tcell_state/tcell_expression_program_scores.csv"


workflow {

    deep_evi_ch = Channel.fromPath(
        params.deep_evi_scores,
        checkIfExists: true
    )

    reference_ch = Channel.fromPath(
        params.reference_scores,
        checkIfExists: true
    )

    state_ch = Channel.fromPath(
        params.state_scores,
        checkIfExists: true
    )

    ANALYZE_10A5_STATE_AXIS(
        deep_evi_ch,
        reference_ch,
        state_ch
    )
}