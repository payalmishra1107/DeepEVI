process TCGA_BRCA_09B {
    tag "TCGA-BRCA-09B frozen molecular surrogate projection"
    cpus 4
    memory '10 GB'
    time '8h'
    maxForks 1
    conda "\${projectDir}/envs/deep_evi_tcga_validation.yml"
    input:
    path raw_dir
    path query_json
    path clinical_tsv
    path manifest_tsv
    path signature_dir
    output:
    path "*.json", emit: reports
    path "*.csv", emit: tables
    script:
    """
    python \${projectDir}/bin/run_tcga_brca_09b.py \
        --raw-dir ${raw_dir} \
        --query-json ${query_json} \
        --clinical-tsv ${clinical_tsv} \
        --manifest-tsv ${manifest_tsv} \
        --signature-dir ${signature_dir} \
        --outdir .
    """
}
