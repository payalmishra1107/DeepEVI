nextflow.enable.dsl=2

/*
 * Deep-EVI / GSE176078
 *
 * STEP 6
 * Candidate sample-aware integration
 *
 * Input:
 *   GSE176078_step5_latent.h5ad
 *
 * Output:
 *   GSE176078_harmony_candidate.h5ad
 *
 * IMPORTANT:
 *   This is a candidate integration.
 *   Quantitative evaluation is required before
 *   declaring the representation biologically appropriate.
 */

params {
    latent = null
    outdir = "results_step6"

    batch_key = "sample_id"
    n_pcs = 50
}

include {
    HARMONY_INTEGRATION
} from './modules/06_harmony.nf'


workflow {

    if (!params.latent) {
        error """
        Missing --latent.

        Example:

        nextflow run main_step6.nf \\
          -profile conda,workstation \\
          --latent results_step5/latent/GSE176078_step5_latent.h5ad \\
          --outdir results_step6
        """
    }

    latent_ch = Channel.fromPath(
        params.latent,
        checkIfExists: true
    )

    HARMONY_INTEGRATION(
        latent_ch
    )
}