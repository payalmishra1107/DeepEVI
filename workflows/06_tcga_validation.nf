nextflow.enable.dsl=2
include { BUILD_TCGA_SIGNATURE } from '../modules/09a_tcga_signature.nf'
include { TCGA_BRCA_09B } from '../modules/09b_tcga_projection.nf'
include { TCGA_BRCA_09C } from '../modules/09c_tcga_clinical.nf'
include { TCGA_BRCA_09D } from '../modules/09d_tcga_bulk_composition.nf'
include { TCGA_BRCA_09E } from '../modules/09e_tcga_biological_validation.nf'

workflow TCGA_VALIDATION {
    take:
    deep_scores
    state_scores
    split_manifest
    tcga_raw
    tcga_query
    tcga_clinical
    tcga_manifest
    multifile_audit

    main:
    BUILD_TCGA_SIGNATURE(deep_scores, state_scores, split_manifest)

    signature_json = BUILD_TCGA_SIGNATURE.out
        .filter { it.name == 'GSE176078_09A_frozen_signature.json' }
        .first()

    signature_dir = signature_json.parent

    TCGA_BRCA_09B(
        tcga_raw,
        tcga_query,
        tcga_clinical,
        tcga_manifest,
        Channel.value(signature_dir)
    )

    tcga_scores = TCGA_BRCA_09B.out.tables
        .filter { it.name == 'GSE176078_09B_TCGA_BRCA_scores.csv' }
        .first()
    tcga_survival = TCGA_BRCA_09B.out.tables
        .filter { it.name == 'GSE176078_09B_TCGA_BRCA_survival_summary.csv' }
        .first()
    tcga_inventory = TCGA_BRCA_09B.out.tables
        .filter { it.name == 'GSE176078_09B_TCGA_BRCA_case_inventory.csv' }
        .first()
    multifile = TCGA_BRCA_09B.out.tables
        .filter { it.name == 'multifile_case_file_level_scores.csv' }
        .first()

    TCGA_BRCA_09C(tcga_scores, tcga_survival, tcga_clinical, multifile_audit)
    TCGA_BRCA_09D(tcga_scores, tcga_survival, tcga_clinical)

    pathways = Channel.fromPath("${projectDir}/config/09e_validation_pathways.tsv", checkIfExists: true).first()
    TCGA_BRCA_09E(tcga_raw, tcga_scores, tcga_inventory, tcga_clinical, pathways)

    emit:
    signature = signature_json
    projection_scores = tcga_scores
    projection_survival = tcga_survival
    projection_inventory = tcga_inventory
    projection_multifile = multifile
    clinical = TCGA_BRCA_09C.out
    composition = TCGA_BRCA_09D.out
    concordance = TCGA_BRCA_09E.out
}
