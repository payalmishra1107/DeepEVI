nextflow.enable.dsl=2

/*
 * Deep-EVI / GSE176078
 *
 * STEP 6A
 * Quantitative integration evaluation
 *
 * Compares:
 *   X_pca_pre_harmony
 *   X_harmony
 *
 * Batch variable:
 *   sample_id
 *
 * Biological label:
 *   celltype_major
 */

params {
    integrated = null
    outdir = "results_step6_evaluation"
    batch_key = "sample_id"
    label_key = "celltype_major"
}

include {
    EVALUATE_INTEGRATION
} from './modules/06a_evaluation.nf'

workflow {

    if (!params.integrated) {
        error """
        Missing --integrated.

        Example:

        nextflow run main_step6a.nf \\
          -profile conda,workstation \\
          --integrated results_step6/integrated/GSE176078_harmony_candidate.h5ad \\
          --outdir results_step6_evaluation
        """
    }

    integrated_ch = Channel.fromPath(
        params.integrated,
        checkIfExists: true
    )

    EVALUATE_INTEGRATION(
        integrated_ch
    )
}