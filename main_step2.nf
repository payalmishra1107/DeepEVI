nextflow.enable.dsl=2

/*
 * STEP 2 — Per-sample extraction and QC.
 * Canonical end-to-end execution is main.nf.
 */
params {
    primary_raw_dir = null
    outdir = "results_step2"
    min_genes = 200
    max_pct_mito = 20.0
}
include { EXTRACT_QC_SAMPLE; AGGREGATE_QC } from './modules/02_qc.nf'
workflow {
    if (!params.primary_raw_dir) error "Missing --primary_raw_dir"
    archives = Channel.fromPath(
        "${params.primary_raw_dir}/GSM*.tar.gz",
        checkIfExists: true
    ).map { archive ->
        def sample_id = archive.baseName.replaceFirst(/\.tar$/, '')
        tuple(sample_id, archive)
    }
    EXTRACT_QC_SAMPLE(archives)
    AGGREGATE_QC(EXTRACT_QC_SAMPLE.out.summary.collect())
}
