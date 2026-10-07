nextflow.enable.dsl=2
include { DEEP_EVI } from '../modules/08_deep_evi.nf'
include { DEEP_EVI_CHARACTERIZATION } from '../modules/08b_deep_evi_characterization.nf'
include { DEEP_EVI_ABLATION } from '../modules/08c_deep_evi_ablation.nf'
include { DEEP_EVI_GRAPH_DIAGNOSTIC } from '../modules/08d_deep_evi_graph_diagnostic.nf'
include { DEEP_EVI_INDEPENDENT_VALIDATION } from '../modules/08e_deep_evi_independent_validation.nf'

workflow DEEP_EVI_PHASE {
    take:
    landscape
    edges
    state_scores

    main:
    DEEP_EVI(landscape, edges, state_scores)

    deep_scores = DEEP_EVI.out.csv
        .filter { it.name == 'GSE176078_deep_evi_scores.csv' }
        .first()
    deep_model = DEEP_EVI.out.model.first()
    deep_split = DEEP_EVI.out.csv
        .filter { it.name.toLowerCase().contains('split') && it.name.toLowerCase().contains('manifest') }
        .first()

    DEEP_EVI_CHARACTERIZATION(deep_scores)
    DEEP_EVI_ABLATION(landscape, edges, deep_scores, deep_split)
    DEEP_EVI_GRAPH_DIAGNOSTIC(landscape, edges, deep_scores, deep_split)

    DEEP_EVI_INDEPENDENT_VALIDATION(landscape, deep_scores, state_scores, deep_split)

    emit:
    scores = deep_scores
    model = deep_model
    split = deep_split
    characterization = DEEP_EVI_CHARACTERIZATION.out
    ablation = DEEP_EVI_ABLATION.out
    graph_diagnostic = DEEP_EVI_GRAPH_DIAGNOSTIC.out
    robustness = DEEP_EVI_INDEPENDENT_VALIDATION.out
}
