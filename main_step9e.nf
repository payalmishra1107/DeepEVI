nextflow.enable.dsl=2

params.tcga_raw = "${baseDir}/../tcga_brca/raw/star_counts"
params.tcga_clinical = "${baseDir}/../tcga_brca/clinical/TCGA-BRCA_clinical.tsv"
params.tcga_scores = "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_scores.csv"
params.tcga_case_inventory = "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_case_inventory.csv"
params.pathways = "${baseDir}/config/09e_validation_pathways.tsv"

include {
    TCGA_BRCA_09E
} from './modules/09e_tcga_biological_validation.nf'


workflow {

    raw_ch = Channel.fromPath(
        params.tcga_raw,
        type: 'dir',
        checkIfExists: true
    )

    scores_ch = Channel.fromPath(
        params.tcga_scores,
        checkIfExists: true
    )

    inventory_ch = Channel.fromPath(
        params.tcga_case_inventory,
        checkIfExists: true
    )

    clinical_ch = Channel.fromPath(
        params.tcga_clinical,
        checkIfExists: true
    )

    pathways_ch = Channel.fromPath(
        params.pathways,
        checkIfExists: true
    )

    TCGA_BRCA_09E(
        scores_ch,
        raw_ch,
        inventory_ch,
        clinical_ch,
        pathways_ch
    )
}