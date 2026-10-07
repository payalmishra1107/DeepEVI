nextflow.enable.dsl=2

params {
    integrated = null
    outdir = "results_step7a"
    batch_key = "sample_id"
    subtype_key = "subtype"
    n_neighbors = 30
}

include {
    BIOLOGICAL_VALIDATION
} from './modules/07a_biological_validation.nf'

workflow {

    if (!params.integrated) {
        error """
        Missing --integrated.

        Example:

        nextflow run main_step7a.nf \\
          -profile conda,workstation \\
          --integrated results_step6/integrated/GSE176078_harmony_candidate.h5ad \\
          --outdir results_step7a
        """
    }

    integrated_ch = Channel.fromPath(
        params.integrated,
        checkIfExists: true
    )

    BIOLOGICAL_VALIDATION(
        integrated_ch
    )
}