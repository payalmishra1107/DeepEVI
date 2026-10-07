process FEATURE_PCA_COHORT {

    tag "GSE176078 feature selection and PCA"

    publishDir "${params.outdir}/latent",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_step5_latent.h5ad"

    publishDir "${params.outdir}/summary",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_step5_summary.json"

    conda "${projectDir}/envs/integration.yml"

    input:
    path cohort

    output:
    path "GSE176078_step5_latent.h5ad", emit: latent
    path "GSE176078_step5_summary.json", emit: summary

    script:
    """
    python ${projectDir}/bin/integrate_cohort.py \
        --input ${cohort} \
        --output GSE176078_step5_latent.h5ad \
        --summary GSE176078_step5_summary.json \
        --n-hvgs ${params.n_hvgs} \
        --n-pcs ${params.n_pcs}
    """
}