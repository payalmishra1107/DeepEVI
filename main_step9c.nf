nextflow.enable.dsl=2

/*
 * Step 09C checkpoint interface.
 * Canonical execution is workflows/06_tcga_validation.nf.
 */

params.scores = "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_scores.csv"
params.survival = "${baseDir}/results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_survival_summary.csv"
params.clinical = "${baseDir}/../tcga_brca/clinical/TCGA-BRCA_clinical.tsv"
params.multifile_audit = "${baseDir}/results_step9b/tcga_validation/multifile_case_file_level_scores.csv"

include { TCGA_BRCA_09C } from './modules/09c_tcga_clinical.nf'

workflow {
    TCGA_BRCA_09C(
        Channel.fromPath(params.scores, checkIfExists: true).first(),
        Channel.fromPath(params.survival, checkIfExists: true).first(),
        Channel.fromPath(params.clinical, checkIfExists: true).first(),
        Channel.fromPath(params.multifile_audit, checkIfExists: true).first()
    )
}
