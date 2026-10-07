nextflow.enable.dsl=2

include {
    ANALYZE_10A5B_OVERLAP_CONTROLLED
} from './modules/10a5b_overlap_controlled.nf'


params.expression = "${projectDir}/results_step10a/test_expression/GSE176078_10A_test_Tcells.h5ad"
params.deep_evi   = "${projectDir}/results_step8a/deep_evi/GSE176078_deep_evi_scores.csv"
params.state      = "${projectDir}/results_step7b/tcell_state/tcell_state/tcell_expression_program_scores.csv"


workflow {

    expression_ch = Channel.fromPath(
        params.expression,
        checkIfExists: true
    )

    deep_evi_ch = Channel.fromPath(
        params.deep_evi,
        checkIfExists: true
    )

    state_ch = Channel.fromPath(
        params.state,
        checkIfExists: true
    )

    ANALYZE_10A5B_OVERLAP_CONTROLLED(
        expression_ch,
        deep_evi_ch,
        state_ch
    )
}