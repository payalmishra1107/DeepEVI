nextflow.enable.dsl=2
include { DEEP_EVI_XAI; DEEP_EVI_XAI_11D } from '../modules/11_deep_evi_xai.nf'
workflow XAI {
    take:
    scores
    landscape
    edges
    split_manifest
    model
    signature
    main:
    DEEP_EVI_XAI(scores, landscape, edges, split_manifest, model, signature)
    DEEP_EVI_XAI_11D(signature)
    emit:
    xai = DEEP_EVI_XAI.out
    gene_attribution = DEEP_EVI_XAI_11D.out
}
