process VALIDATE_GSE176078 {

    tag "GSE176078 archive validation"

    publishDir "${params.outdir}/inventory",
        mode: 'copy',
        overwrite: true

    conda "${projectDir}/envs/ingestion.yml"

    input:
    path raw_dir

    output:
    path "GSE176078_inventory.csv"

    script:
    """
    python ${projectDir}/bin/validate_and_inventory.py \
        --input ${raw_dir} \
        --output GSE176078_inventory.csv
    """
}