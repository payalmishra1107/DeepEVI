process SCORE_10A_REFERENCE_SIGNATURES {
    tag "10A-2 frozen reference signature scoring"
    conda "\${projectDir}/envs/deep_evi_10a_benchmark.yml"
    input:
    path input_h5ad
    path signatures
    output:
    path "GSE176078_10A_reference_signature_scores.csv", emit: scores
    path "GSE176078_10A_reference_signature_metadata.csv", emit: metadata
    path "GSE176078_10A_10A2_report.json", emit: report
    script:
    """
    python \${projectDir}/bin/score_10a_reference_signatures.py \
        --input-h5ad ${input_h5ad} \
        --signatures ${signatures} \
        --outdir .
    """
}
