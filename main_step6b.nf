nextflow.enable.dsl=2

params {
    integrated = null
    outdir = "results_step6b"
    batch_key = "sample_id"
    label_key = "celltype_major"
    n_jobs = 2
}

include {
    BENCHMARK_INTEGRATION
} from './modules/06b_benchmark.nf'

workflow {

    if (!params.integrated) {
        error """
        Missing --integrated.

        Example:

        nextflow run main_step6b.nf \\
          -profile conda,workstation \\
          --integrated results_step6/integrated/GSE176078_harmony_candidate.h5ad \\
          --outdir results_step6b
        """
    }

    integrated_ch = Channel.fromPath(
        params.integrated,
        checkIfExists: true
    )

    BENCHMARK_INTEGRATION(
        integrated_ch
    )
}