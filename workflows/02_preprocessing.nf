nextflow.enable.dsl=2
include { EXTRACT_QC_SAMPLE; AGGREGATE_QC } from '../modules/02_qc.nf'
include { NORMALIZE_SAMPLE_V2 } from '../modules/03_normalization.nf'
include { ASSEMBLE_COHORT } from '../modules/04_cohort_assembly.nf'
workflow PREPROCESSING {
    take:
    raw_dir
    main:
    archives = Channel.fromPath("${raw_dir}/GSM*.tar.gz", checkIfExists: true)
        .map { archive ->
            def sample_id = archive.baseName.replaceFirst(/\.tar$/, '')
            tuple(sample_id, archive)
        }
    EXTRACT_QC_SAMPLE(archives)
    AGGREGATE_QC(EXTRACT_QC_SAMPLE.out.summary.collect())
    NORMALIZE_SAMPLE_V2(EXTRACT_QC_SAMPLE.out.h5ad)
    normalized_files = NORMALIZE_SAMPLE_V2.out.normalized.map { it[1] }.collect()
    ASSEMBLE_COHORT(normalized_files)
    emit:
    qc_h5ad = EXTRACT_QC_SAMPLE.out.h5ad
    qc_summary = AGGREGATE_QC.out
    normalized = NORMALIZE_SAMPLE_V2.out.normalized
    cohort = ASSEMBLE_COHORT.out.cohort
}
