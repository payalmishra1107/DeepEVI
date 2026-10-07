nextflow.enable.dsl=2

/*
 * Deep-EVI / GSE176078
 *
 * STEP 4 — Cohort Assembly
 *
 * Input:
 *   results_step3/normalized/*_normalized.h5ad
 *
 * Output:
 *   GSE176078_cohort.h5ad
 */

params {
    normalized_dir = null
    outdir = "results_step4"
}

include {
    ASSEMBLE_COHORT
} from './modules/04_cohort_assembly.nf'


workflow {

    if (!params.normalized_dir) {
        error """
        Missing --normalized_dir.

        Example:

        nextflow run main_step4.nf \\
          -profile conda,workstation \\
          --normalized_dir results_step3/normalized \\
          --outdir results_step4
        """
    }

    normalized_h5ads = Channel.fromPath(
        "${params.normalized_dir}/*_normalized.h5ad",
        checkIfExists: true
    )

    ASSEMBLE_COHORT(
        normalized_h5ads.collect()
    )
}