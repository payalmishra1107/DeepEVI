nextflow.enable.dsl=2
include { BIOLOGICAL_VALIDATION } from '../modules/07a_biological_validation.nf'
include { TCELL_STATE_ANALYSIS } from '../modules/07b_tcell_state.nf'
include { TCELL_TRAJECTORY } from '../modules/07c_tcell_trajectory.nf'
workflow TCELL_ANALYSIS {
    take:
    integrated
    main:
    BIOLOGICAL_VALIDATION(integrated)
    TCELL_STATE_ANALYSIS(integrated)
    state_scores = TCELL_STATE_ANALYSIS.out.state_results
        .filter { it.name == 'tcell_expression_program_scores.csv' }
    TCELL_TRAJECTORY(integrated, state_scores)
    emit:
    biological = BIOLOGICAL_VALIDATION.out
    state = TCELL_STATE_ANALYSIS.out
    trajectory = TCELL_TRAJECTORY.out
}
