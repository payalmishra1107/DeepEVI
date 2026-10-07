nextflow.enable.dsl=2

params {
    integrated = null
    scores = null

    outdir = "results_step7c"

    batch_key = "sample_id"
    subtype_key = "subtype"
    subset_key = "celltype_subset"

    n_components = 10
    n_neighbors = 30
}

include {
    TCELL_TRAJECTORY
} from './modules/07c_tcell_trajectory.nf'


workflow {

    if (!params.integrated) {
        error """
        Missing --integrated.

        Example:

        nextflow run main_step7c.nf \\
          -profile conda,workstation \\
          --integrated results_step6/integrated/GSE176078_harmony_candidate.h5ad \\
          --scores results_step7b/tcell_state/tcell_state/tcell_expression_program_scores.csv \\
          --outdir results_step7c
        """
    }

    if (!params.scores) {
        error """
        Missing --scores.

        Expected Step 7B output:

        results_step7b/tcell_state/tcell_state/tcell_expression_program_scores.csv
        """
    }

    integrated_ch = Channel.fromPath(
        params.integrated,
        checkIfExists: true
    )

    scores_ch = Channel.fromPath(
        params.scores,
        checkIfExists: true
    )

    TCELL_TRAJECTORY(
        integrated_ch,
        scores_ch
    )
}