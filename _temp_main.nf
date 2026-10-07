nextflow.enable.dsl=2

/*
 * Deep-EVI / GSE176078
 *
 * STEP 2 — per-sample extraction + QC
 * STEP 3 — per-sample normalization
 *
 * Raw input:
 *   GSM*_CID*.tar.gz
 *
 * Step 2 output:
 *   filtered_<GSM>_<CID>.h5ad
 *
 * Step 3 output:
 *   <GSM>_<CID>_normalized.h5ad
 */

params {
    primary_raw_dir = null
    outdir = "results_step2"

    min_genes = 200
    max_pct_mito = 20.0

    target_sum = 10000.0
}

include {
    EXTRACT_QC_SAMPLE
    AGGREGATE_QC
} from './modules/02_qc.nf'

include {
    NORMALIZE_SAMPLE_V2
} from './modules/03_normalization.nf'

include {
    ASSEMBLE_COHORT
} from './modules/04_cohort_assembly.nf'
workflow {

    if (!params.primary_raw_dir) {
        error """
        Missing --primary_raw_dir.

        Example:

        nextflow run main.nf \\
          -profile conda,workstation \\
          --primary_raw_dir ~/deepevi/manual_downloads/GSE176078_RAW \\
          --outdir results_step2 \\
          -resume
        """
    }

    /*
     * Build an explicit tuple:
     *
     * (sample_id, archive)
     *
     * This preserves the biological sample identity
     * throughout the Nextflow workflow.
     */
    archives = Channel.fromPath(
        "${params.primary_raw_dir}/GSM*.tar.gz",
        checkIfExists: true
    ).map { archive ->
        def sample_id = archive.baseName.replaceFirst(/\.tar$/, '')
        tuple(sample_id, archive)
    }

    /*
     * STEP 2
     * Per-sample extraction and QC.
     */
    EXTRACT_QC_SAMPLE(archives)

    /*
     * Aggregate QC summaries across all samples.
     */
    AGGREGATE_QC(
        EXTRACT_QC_SAMPLE.out.summary.collect()
    )

    /*
     * STEP 3
     * Normalize each QC-filtered H5AD.
     *
     * The sample ID is passed directly with
     * the H5AD instead of being inferred from
     * the temporary filename.
     */
    NORMALIZE_SAMPLE_V2(
        EXTRACT_QC_SAMPLE.out.h5ad
    )
}