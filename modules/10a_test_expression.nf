process BUILD_10A_TEST_EXPRESSION {
    tag "10A-1 frozen held-out test expression"
    conda "\${projectDir}/envs/deep_evi_10a_benchmark.yml"
    input:
    path input_h5ad
    path split_manifest
    output:
    path "GSE176078_10A_test_Tcells.h5ad", emit: expression
    path "GSE176078_10A_test_expression_report.json", emit: report
    script:
    """
    python \${projectDir}/bin/build_10a_test_expression.py \
        --input-h5ad ${input_h5ad} \
        --split-manifest ${split_manifest} \
        --out-h5ad GSE176078_10A_test_Tcells.h5ad \
        --out-report GSE176078_10A_test_expression_report.json
    """
}
