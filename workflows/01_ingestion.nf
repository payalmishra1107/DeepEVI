nextflow.enable.dsl=2
include { VALIDATE_GSE176078 } from '../modules/01_ingest.nf'
workflow INGESTION {
    take:
    raw_dir
    main:
    raw_dir_ch = Channel.fromPath(raw_dir, type: 'dir', checkIfExists: true)
    VALIDATE_GSE176078(raw_dir_ch)
    emit:
    inventory = VALIDATE_GSE176078.out
}
