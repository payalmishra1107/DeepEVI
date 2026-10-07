nextflow.enable.dsl=2

/*
 * Step 09B checkpoint interface.
 * Canonical execution is workflows/06_tcga_validation.nf.
 */

params.tcga_raw = "${baseDir}/../tcga_brca/raw/star_counts"
params.tcga_clinical = "${baseDir}/../tcga_brca/clinical/TCGA-BRCA_clinical.tsv"
params.tcga_manifest = "${baseDir}/../tcga_brca/manifests/TCGA-BRCA_STAR_Counts_manifest.tsv"
params.tcga_query = "${baseDir}/../tcga_brca/manifests/TCGA-BRCA_STAR_Counts_query.json"
params.signature_dir = "${baseDir}/results_step9a/signature"

include { TCGA_BRCA_09B } from './modules/09b_tcga_projection.nf'

workflow {
    raw_ch = Channel.fromPath(params.tcga_raw, type: 'dir', checkIfExists: true).first()
    query_ch = Channel.fromPath(params.tcga_query, checkIfExists: true).first()
    clinical_ch = Channel.fromPath(params.tcga_clinical, checkIfExists: true).first()
    manifest_ch = Channel.fromPath(params.tcga_manifest, checkIfExists: true).first()
    signature_ch = Channel.fromPath(params.signature_dir, type: 'dir', checkIfExists: true).first()

    TCGA_BRCA_09B(
        raw_ch,
        query_ch,
        clinical_ch,
        manifest_ch,
        signature_ch
    )
}
