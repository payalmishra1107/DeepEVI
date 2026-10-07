nextflow.enable.dsl=2

/*
 * Deep-EVI / GSE176078
 *
 * STEP 5
 * Feature selection + PCA / pre-integration latent representation
 *
 * Input:
 *   GSE176078_cohort.h5ad
 *
 * Output:
 *   GSE176078_step5_latent.h5ad
 */

params {
    cohort = null
    outdir = "results_step5"

    n_hvgs = 3000
    n_pcs = 50
}

include {
    FEATURE_PCA_COHORT
} from './modules/05_integration.nf'


workflow {

    if (!params.cohort) {
        error """
        Missing --cohort.

        Example:

        nextflow run main_step5.nf \\
          -profile conda,workstation \\
          --cohort results_step4/cohort/GSE176078_cohort.h5ad \\
          --outdir results_step5
        """
    }

    cohort_ch = Channel.fromPath(
        params.cohort,
        checkIfExists: true
    )

    FEATURE_PCA_COHORT(
        cohort_ch
    )
}