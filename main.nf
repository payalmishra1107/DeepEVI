nextflow.enable.dsl=2

/*
 * Deep-EVI canonical end-to-end workflow.
 *
 * main.nf is the authoritative reproducibility entrypoint.
 * main_step*.nf are checkpoint/debug interfaces.
 */

params {
    primary_raw_dir = null
    tcga_raw = null
    tcga_query = null
    tcga_clinical = null
    tcga_manifest = null
    reference_signatures = "${projectDir}/config/10a_benchmark_signatures.tsv"
    immune_subtypes = null
    multifile_audit = null

    outdir = "results"
    run_tcga = true
    run_heldout = true
    run_xai = true
    run_audit = false
}

include { INGESTION } from './workflows/01_ingestion.nf'
include { PREPROCESSING } from './workflows/02_preprocessing.nf'
include { INTEGRATION } from './workflows/03_integration.nf'
include { TCELL_ANALYSIS } from './workflows/04_tcell_analysis.nf'
include { DEEP_EVI_PHASE } from './workflows/05_deep_evi.nf'
include { TCGA_VALIDATION } from './workflows/06_tcga_validation.nf'
include { HELDOUT_AND_EXTERNAL_VALIDATION } from './workflows/07_heldout_validation.nf'
include { XAI } from './workflows/08_xai.nf'
include { AUDIT } from './workflows/09_audit.nf'

workflow {
    if (!params.primary_raw_dir) {
        error "Missing --primary_raw_dir"
    }

    INGESTION(params.primary_raw_dir)
    PREPROCESSING(params.primary_raw_dir)

    INTEGRATION(PREPROCESSING.out.cohort)
    TCELL_ANALYSIS(INTEGRATION.out.integrated)

    trajectory_landscape = TCELL_ANALYSIS.out.trajectory
        .filter { it.name == 'tcell_state_landscape.csv' }
        .first()
    trajectory_edges = TCELL_ANALYSIS.out.trajectory
        .filter { it.name == 'tcell_state_knn_edges.csv' }
        .first()
    state_scores = TCELL_ANALYSIS.out.state
        .filter { it.name == 'tcell_expression_program_scores.csv' }
        .first()

    DEEP_EVI_PHASE(
        trajectory_landscape,
        trajectory_edges,
        state_scores,
        state_scores
    )

    deep_scores = DEEP_EVI_PHASE.out.scores.first()
    deep_model = DEEP_EVI_PHASE.out.model.first()
    deep_split = DEEP_EVI_PHASE.out.split.first()

    if (params.run_tcga) {
        required = [
            params.tcga_raw,
            params.tcga_query,
            params.tcga_clinical,
            params.tcga_manifest,
            params.multifile_audit
        ]
        if (required.any { !it }) {
            error "TCGA execution requires --tcga_raw, --tcga_query, --tcga_clinical, --tcga_manifest and --multifile_audit"
        }

        TCGA_VALIDATION(
            deep_scores,
            state_scores,
            deep_split,
            Channel.fromPath(params.tcga_raw, type: 'dir', checkIfExists: true).first(),
            Channel.fromPath(params.tcga_query, checkIfExists: true).first(),
            Channel.fromPath(params.tcga_clinical, checkIfExists: true).first(),
            Channel.fromPath(params.tcga_manifest, checkIfExists: true).first(),
            Channel.fromPath(params.multifile_audit, checkIfExists: true).first()
        )
    }

    if (params.run_heldout) {
        if (!file(params.reference_signatures).exists()) {
            error "Missing --reference_signatures: ${params.reference_signatures}"
        }
        if (!params.run_tcga) {
            error "Held-out/external validation requires TCGA 09B outputs; keep --run_tcga true."
        }
        if (!params.immune_subtypes) {
            error "Held-out/external validation requires --immune_subtypes for Step 10B."
        }

        tcga_scores = TCGA_VALIDATION.out.projection
            .filter { it.name == 'TCGA_BRCA_09B_TCGA_BRCA_scores.csv' }
            .first()
        tcga_inventory = TCGA_VALIDATION.out.projection
            .filter { it.name == 'TCGA_BRCA_09B_TCGA_BRCA_case_inventory.csv' }
            .first()

        HELDOUT_AND_EXTERNAL_VALIDATION(
            INTEGRATION.out.latent.first(),
            deep_scores,
            state_scores,
            deep_split,
            Channel.fromPath(params.reference_signatures, checkIfExists: true).first(),
            tcga_scores,
            tcga_inventory,
            Channel.fromPath(params.immune_subtypes, checkIfExists: true).first()
        )
    }

    if (params.run_xai) {
        if (!params.run_tcga) {
            error "XAI requires the frozen 09A signature; keep --run_tcga true."
        }

        signature = TCGA_VALIDATION.out.signature
            .filter { it.name == 'GSE176078_09A_frozen_signature.json' }
            .first()

        XAI(
            deep_scores,
            trajectory_landscape,
            trajectory_edges,
            deep_split,
            deep_model,
            signature
        )
    }

    if (params.run_audit) {
        AUDIT(Channel.fromPath(projectDir, type: 'dir', checkIfExists: true).first())
    }
}
