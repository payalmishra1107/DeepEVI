nextflow.enable.dsl=2

params.scores =
    "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_scores.csv"

params.survival =
    "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_survival_summary.csv"

params.clinical =
    "${baseDir}/../tcga_brca/clinical/TCGA-BRCA_clinical.tsv"

include {
    TCGA_BRCA_09D
} from './modules/09d_tcga_bulk_composition.nf'

workflow {
    scores_ch = Channel.fromPath(params.scores, checkIfExists: true)
    survival_ch = Channel.fromPath(params.survival, checkIfExists: true)
    clinical_ch = Channel.fromPath(params.clinical, checkIfExists: true)

    TCGA_BRCA_09D(
        scores_ch,
        survival_ch,
        clinical_ch
    )
}