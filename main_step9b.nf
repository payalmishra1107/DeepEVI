nextflow.enable.dsl=2

/*
 * Deep-EVI Step 09B
 * TCGA-BRCA molecular surrogate validation
 *
 * Frozen inputs:
 *   - GSE176078 09A gene signature
 *   - TCGA-BRCA STAR-Counts expression files
 *   - TCGA-BRCA clinical metadata
 *   - TCGA-BRCA GDC manifest
 *   - TCGA-BRCA GDC query JSON
 *
 * The 09A signature is NOT refit on TCGA.
 */

params.tcga_raw = "${baseDir}/../tcga_brca/raw/star_counts"

params.tcga_clinical = "${baseDir}/../tcga_brca/clinical/TCGA-BRCA_clinical.tsv"

params.tcga_manifest = "${baseDir}/../tcga_brca/manifests/TCGA-BRCA_STAR_Counts_manifest.tsv"

params.tcga_query = "${baseDir}/../tcga_brca/manifests/TCGA-BRCA_STAR_Counts_query.json"

params.signature_dir = "${baseDir}/results_step9a/signature"


process TCGA_BRCA_09B {

    tag "TCGA-BRCA-09B"

    cpus 4
    memory '10 GB'
    time '8h'
    maxForks 1

    conda "${baseDir}/envs/deep_evi_tcga_validation.yml"

    input:
    path raw_dir
    path query_json
    path clinical_tsv
    path manifest_tsv
    path signature_dir

    output:
    path "*.json", emit: reports
    path "*.csv", emit: tables

    script:
    """
    python ${baseDir}/bin/run_tcga_brca_09b.py \
        --raw-dir ${raw_dir} \
        --query-json ${query_json} \
        --clinical-tsv ${clinical_tsv} \
        --manifest-tsv ${manifest_tsv} \
        --signature-dir ${signature_dir} \
        --outdir .
    """
}


workflow {

    raw_ch = Channel.fromPath(
        params.tcga_raw,
        type: 'dir',
        checkIfExists: true
    )

    query_ch = Channel.fromPath(
        params.tcga_query,
        checkIfExists: true
    )

    clinical_ch = Channel.fromPath(
        params.tcga_clinical,
        checkIfExists: true
    )

    manifest_ch = Channel.fromPath(
        params.tcga_manifest,
        checkIfExists: true
    )

    signature_ch = Channel.fromPath(
        params.signature_dir,
        type: 'dir',
        checkIfExists: true
    )

    TCGA_BRCA_09B(
        raw_ch,
        query_ch,
        clinical_ch,
        manifest_ch,
        signature_ch
    )
}