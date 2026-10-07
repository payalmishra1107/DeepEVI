nextflow.enable.dsl=2

params {
    integrated = null
    outdir = "results_step7b"

    celltype_key = "celltype_major"
    tcell_label = "T-cells"
    batch_key = "sample_id"
    subtype_key = "subtype"
    subset_key = "celltype_subset"
}

include {
    TCELL_STATE_ANALYSIS
} from './modules/07b_tcell_state.nf'


workflow {

    if (!params.integrated) {
        error """
        Missing --integrated.

        Example:

        nextflow run main_step7b.nf \\
          -profile conda,workstation \\
          --integrated results_step6/integrated/GSE176078_harmony_candidate.h5ad \\
          --outdir results_step7b
        """
    }

    integrated_ch = Channel.fromPath(
        params.integrated,
        checkIfExists: true
    )

    TCELL_STATE_ANALYSIS(
        integrated_ch
    )
}