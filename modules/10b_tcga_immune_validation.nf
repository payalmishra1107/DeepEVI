process VALIDATE_10B_TCGA_IMMUNE_SUBTYPES {

    tag "10B_TCGA_immune_subtypes"

    label 'DEEP_EVI_10B'

    publishDir "${projectDir}/results_step10b/validation",
        mode: 'copy',
        overwrite: true

    conda "${projectDir}/envs/deep_evi_10b_tcga_immune.yml"

    input:
    path scores
    path immune_subtypes
    path case_inventory

    output:
    path "TCGA_BRCA_10B_case_level_scores.csv"
    path "TCGA_BRCA_10B_subtype_summary.csv"
    path "TCGA_BRCA_10B_global_test.csv"
    path "TCGA_BRCA_10B_pairwise_tests.csv"
    path "TCGA_BRCA_10B_primary_contrast.csv"
    path "TCGA_BRCA_10B_sensitivity_single_vs_multifile.csv"
    path "TCGA_BRCA_10B_report.json"

    script:
    """
    python ${projectDir}/bin/validate_10b_tcga_immune_subtypes.py \
        --scores ${scores} \
        --immune-subtypes ${immune_subtypes} \
        --case-inventory ${case_inventory} \
        --outdir .
    """
}