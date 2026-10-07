process HARMONY_INTEGRATION {

    tag "GSE176078 Harmony candidate integration"

    publishDir "${params.outdir}/integrated",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_harmony_candidate.h5ad"

    publishDir "${params.outdir}/summary",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_harmony_summary.json"

    conda "${projectDir}/envs/harmony.yml"

    input:
    path latent

    output:
    path "GSE176078_harmony_candidate.h5ad", emit: integrated
    path "GSE176078_harmony_summary.json", emit: summary

    script:
    """
    python ${projectDir}/bin/harmony_integrate.py \
        --input ${latent} \
        --output GSE176078_harmony_candidate.h5ad \
        --summary GSE176078_harmony_summary.json \
        --batch-key ${params.batch_key} \
        --n-pcs ${params.n_pcs}
    """
}