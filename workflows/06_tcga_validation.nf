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
    signature_dir = BUILD_TCGA_SIGNATURE.out
        .filter { it.name == 'GSE176078_09A_frozen_signature.json' }
        .map { it.parent }
    TCGA_BRCA_09B(tcga_raw, tcga_query, tcga_clinical, tcga_manifest, signature_dir)
    tcga_scores = TCGA_BRCA_09B.out.tables.filter { it.name == 'GSE176078_09B_TCGA_BRCA_scores.csv' }
    tcga_survival = TCGA_BRCA_09B.out.tables.filter { it.name == 'GSE176078_09B_TCGA_BRCA_survival_summary.csv' }
    tcga_inventory = TCGA_BRCA_09B.out.tables.filter { it.name == 'GSE176078_09B_TCGA_BRCA_case_inventory.csv' }
    TCGA_BRCA_09C(tcga_scores, tcga_survival, tcga_clinical, multifile_audit)
    TCGA_BRCA_09D(tcga_scores, tcga_survival, tcga_clinical)
    pathways = Channel.fromPath("${projectDir}/config/09e_validation_pathways.tsv", checkIfExists: true)
    TCGA_BRCA_09E(tcga_raw, tcga_scores, tcga_inventory, tcga_clinical, pathways)
    emit:
    signature = BUILD_TCGA_SIGNATURE.out
    projection = TCGA_BRCA_09B.out
    clinical = TCGA_BRCA_09C.out
    composition = TCGA_BRCA_09D.out
    concordance = TCGA_BRCA_09E.out
}
