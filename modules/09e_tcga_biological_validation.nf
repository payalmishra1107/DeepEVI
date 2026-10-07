process TCGA_BRCA_09E {
    tag "TCGA-BRCA-09E-biological-concordance"

    cpus 4
    memory '10 GB'
    time '8h'
    maxForks 1

    conda "${baseDir}/envs/deepevi_tcga_biological_validation.yml"

    publishDir "${baseDir}/results_step9e/tcga_biological_concordance",
        mode: 'copy',
        overwrite: true

    input:
    path scores
    path expression_dir
    path case_inventory
    path clinical
    path pathways

    output:
    path "GSE176078_09E_report.json", emit: report
    path "09E_analysis_cohort.csv", emit: analysis_cohort
    path "09E_pathway_correlations.csv", emit: pathway_correlations
    path "09E_immune_context_correlations.csv", emit: immune_context_correlations
    path "09E_pathway_metadata.csv", emit: pathway_metadata
    path "09E_pathway_case_scores.csv", emit: pathway_case_scores
    path "09E_immune_context_case_scores.csv", emit: immune_context_case_scores

    script:
    """
    python ${baseDir}/bin/run_tcga_brca_09e_biological_validation.py \
        --scores ${scores} \
        --expression-dir ${expression_dir} \
        --case-inventory ${case_inventory} \
        --clinical ${clinical} \
        --pathways ${pathways} \
        --outdir .
    """
}