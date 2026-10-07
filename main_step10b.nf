nextflow.enable.dsl=2

include {
    VALIDATE_10B_TCGA_IMMUNE_SUBTYPES
} from './modules/10b_tcga_immune_validation.nf'


params.scores = "${projectDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_scores.csv"

params.immune_subtypes = "${projectDir}/tcga_brca/metadata/immune_subtypes/Subtype_Immune_Model_Based.txt"

params.case_inventory = "${projectDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_case_inventory.csv"


workflow {

    scores_ch = Channel.fromPath(
        params.scores,
        checkIfExists: true
    )

    immune_subtypes_ch = Channel.fromPath(
        params.immune_subtypes,
        checkIfExists: true
    )

    case_inventory_ch = Channel.fromPath(
        params.case_inventory,
        checkIfExists: true
    )

    VALIDATE_10B_TCGA_IMMUNE_SUBTYPES(
        scores_ch,
        immune_subtypes_ch,
        case_inventory_ch
    )
}