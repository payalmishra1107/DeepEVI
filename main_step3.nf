nextflow.enable.dsl=2

/*
 * STEP 3 — Per-sample normalization.
 * Input is the published Step-2 filtered H5AD directory.
 * Canonical end-to-end execution is main.nf.
 */
params {
    qc_h5ad_dir = null
    outdir = "results_step3"
    target_sum = 10000.0
}
include { NORMALIZE_SAMPLE_V2 } from './modules/03_normalization.nf'
workflow {
    if (!params.qc_h5ad_dir) error "Missing --qc_h5ad_dir"
    h5ads = Channel.fromPath(
        "${params.qc_h5ad_dir}/filtered_*.h5ad",
        checkIfExists: true
    ).map { h5ad ->
        def name = h5ad.baseName
        def sample_id = name.replaceFirst(/^filtered_/, '')
        tuple(sample_id, h5ad)
    }
    NORMALIZE_SAMPLE_V2(h5ads)
}
