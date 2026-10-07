process ASSEMBLE_COHORT {

    tag "GSE176078 cohort assembly"

    publishDir "${params.outdir}/cohort",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_cohort.h5ad"

    publishDir "${params.outdir}/assembly_summary",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_cohort_assembly_summary.json"

    conda "${projectDir}/envs/cohort_assembly.yml"

    input:
    path normalized_h5ads

    output:
    path "GSE176078_cohort.h5ad", emit: cohort
    path "GSE176078_cohort_assembly_summary.json", emit: summary

    script:
    """
    mkdir normalized_inputs

    cp *_normalized.h5ad normalized_inputs/

    python ${projectDir}/bin/assemble_cohort.py \
        --input-dir normalized_inputs \
        --output GSE176078_cohort.h5ad \
        --summary GSE176078_cohort_assembly_summary.json
    """
}