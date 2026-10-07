nextflow.enable.dsl=2

/*
 * Deep-EVI / GSE176078
 *
 * STEP 1 — Input inventory and provenance validation
 *
 * Input:
 *   --primary_raw_dir  Directory containing GSM*.tar.gz archives
 *
 * Output:
 *   results_step1/inventory/GSE176078_inventory.csv
 */

params {
    primary_raw_dir = null
    outdir = "results_step1"
}

include {
    VALIDATE_GSE176078
} from './modules/01_ingest.nf'

workflow {
    if (!params.primary_raw_dir) {
        error """
        Missing --primary_raw_dir.

        Example:

        nextflow run main_step1.nf \
          -profile conda,workstation \
          --primary_raw_dir ~/deepevi/manual_downloads/GSE176078_RAW \
          --outdir results_step1
        """
    }

    raw_dir_ch = Channel.fromPath(
        params.primary_raw_dir,
        type: 'dir',
        checkIfExists: true
    )

    VALIDATE_GSE176078(raw_dir_ch)
}
