process BENCHMARK_INTEGRATION {

    tag "GSE176078 standardized integration benchmark"

    publishDir "${params.outdir}/benchmark",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_standardized_integration_metrics.csv"

    publishDir "${params.outdir}/benchmark",
        mode: 'copy',
        overwrite: true,
        pattern: "GSE176078_standardized_integration_summary.json"

    publishDir "${params.outdir}/plots",
        mode: 'copy',
        overwrite: true,
        pattern: "*.png"

    conda "${projectDir}/envs/integration_benchmark.yml"

    input:
    path integrated

    output:
    path "GSE176078_standardized_integration_metrics.csv",
        emit: metrics

    path "GSE176078_standardized_integration_summary.json",
        emit: summary

    path "plots/*.png",
        emit: plots

    script:
    """
    export JAX_PLATFORMS=cpu
    export CUDA_VISIBLE_DEVICES=""
    export TF_CPP_MIN_LOG_LEVEL=2

    mkdir -p plots

    python ${projectDir}/bin/benchmark_integration.py \
        --input ${integrated} \
        --output-csv GSE176078_standardized_integration_metrics.csv \
        --output-json GSE176078_standardized_integration_summary.json \
        --plot-dir plots \
        --batch-key ${params.batch_key} \
        --label-key ${params.label_key} \
        --n-jobs ${params.n_jobs}
    """
}