#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--input", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--sample-key", default="sample_id")
    p.add_argument("--subtype-key", default="subtype")
    p.add_argument("--subset-key", default="celltype_subset")

    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--n-neighbors", type=int, default=30)

    return p.parse_args()


def clean(series):
    return series.astype("string").fillna("NA").str.strip()


def main():

    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 7C — T-CELL STATE LANDSCAPE / TRAJECTORY PROXY")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Load full integrated cohort
    # ------------------------------------------------------------------

    adata_full = ad.read_h5ad(args.input)

    print(
        f"Input cohort: "
        f"{adata_full.n_obs} cells × {adata_full.n_vars} genes"
    )

    # ------------------------------------------------------------------
    # 2. Validate required metadata
    # ------------------------------------------------------------------

    required_obs = [
        args.sample_key,
        args.subtype_key,
        args.subset_key,
        "celltype_major",
    ]

    missing_obs = [
        key for key in required_obs
        if key not in adata_full.obs.columns
    ]

    if missing_obs:
        raise ValueError(
            f"Missing required observation keys: {missing_obs}"
        )

    # ------------------------------------------------------------------
    # 3. Load Step 7B T-cell score table
    #
    # IMPORTANT:
    # The score table contains only the 35,214 curated T cells.
    # The integrated H5AD contains all 100,064 cells.
    # ------------------------------------------------------------------

    scores = pd.read_csv(args.scores)

    print(f"Score table: {len(scores)} cells")

    if "cell_id" not in scores.columns:
        raise ValueError(
            "Step 7B score table must contain a cell_id column."
        )

    missing_programs = [
        col for col in PROGRAMS
        if col not in scores.columns
    ]

    if missing_programs:
        raise ValueError(
            f"Missing required program scores: {missing_programs}"
        )

    scores["cell_id"] = scores["cell_id"].astype(str)

    if not scores["cell_id"].is_unique:
        raise ValueError(
            "Step 7B score table contains duplicate cell IDs."
        )

    cohort_ids = pd.Index(
        adata_full.obs_names.astype(str)
    )

    score_ids = pd.Index(
        scores["cell_id"]
    )

    # ------------------------------------------------------------------
    # 4. Verify that EVERY scored T cell exists in the cohort
    # ------------------------------------------------------------------

    missing_from_cohort = score_ids.difference(cohort_ids)

    if len(missing_from_cohort) > 0:

        raise ValueError(
            f"{len(missing_from_cohort)} scored T-cell IDs are "
            f"missing from the integrated cohort. "
            f"First missing IDs: "
            f"{missing_from_cohort[:10].tolist()}"
        )

    # ------------------------------------------------------------------
    # 5. Align the full H5AD to exactly the Step 7B T-cell cells
    #
    # The ordering is explicitly controlled by the score table.
    # ------------------------------------------------------------------

    scores = (
        scores
        .set_index("cell_id")
        .loc[score_ids]
        .reset_index()
    )

    adata = adata_full[
        score_ids
    ].copy()

    aligned_ids = pd.Index(
        adata.obs_names.astype(str)
    )

    if len(aligned_ids) != len(scores):
        raise ValueError(
            "Post-alignment cell count mismatch."
        )

    if not np.array_equal(
        aligned_ids.to_numpy(),
        scores["cell_id"].to_numpy(),
    ):
        raise ValueError(
            "Post-alignment cell ordering mismatch between "
            "integrated H5AD and Step 7B score table."
        )

    print(
        f"Aligned T-cell cohort: {adata.n_obs} cells"
    )

    # ------------------------------------------------------------------
    # 6. Confirm that the aligned cells are actually curated T cells
    # ------------------------------------------------------------------

    aligned_major = clean(
        adata.obs["celltype_major"]
    )

    n_curated_t = int(
        (aligned_major == "T-cells").sum()
    )

    if n_curated_t != adata.n_obs:

        non_t = (
            aligned_major
            .value_counts()
            .to_dict()
        )

        raise ValueError(
            "Step 7B score table contains cells that are not "
            f"celltype_major == 'T-cells'. "
            f"Aligned annotation counts: {non_t}"
        )

    # ------------------------------------------------------------------
    # 7. Validate program score matrix
    # ------------------------------------------------------------------

    X_program = scores[
        PROGRAMS
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(X_program).all():
        raise ValueError(
            "Program score matrix contains NaN or Inf."
        )

    # ------------------------------------------------------------------
    # 8. Standardize program scores
    # ------------------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X_program
    ).astype(np.float32)

    # ------------------------------------------------------------------
    # 9. PCA state representation
    # ------------------------------------------------------------------

    n_components = min(
        args.n_components,
        X_scaled.shape[1],
        X_scaled.shape[0] - 1,
    )

    pca = PCA(
        n_components=n_components,
        svd_solver="full",
        random_state=42,
    )

    X_state = pca.fit_transform(
        X_scaled
    ).astype(np.float32)

    state_columns = [
        f"state_pc{i + 1}"
        for i in range(n_components)
    ]

    # ------------------------------------------------------------------
    # 10. Build kNN graph in T-cell state space
    # ------------------------------------------------------------------

    k = min(
        args.n_neighbors,
        X_state.shape[0] - 1,
    )

    nn = NearestNeighbors(
        n_neighbors=k + 1,
        metric="euclidean",
        algorithm="auto",
        n_jobs=2,
    )

    nn.fit(X_state)

    distances, indices = nn.kneighbors(
        X_state
    )

    # Remove self-neighbor
    distances = distances[:, 1:]
    indices = indices[:, 1:]

    # ------------------------------------------------------------------
    # 11. Extract exhaustion and activation programs
    # ------------------------------------------------------------------

    exhaustion = scores[
        "exhaustion_dysfunction_score"
    ].to_numpy(
        dtype=np.float32
    )

    activation = scores[
        "activation_effector_score"
    ].to_numpy(
        dtype=np.float32
    )

    # ------------------------------------------------------------------
    # 12. Local state gradients
    #
    # This is an expression-state trajectory proxy.
    # It is NOT RNA velocity.
    # ------------------------------------------------------------------

    delta_exhaustion = (
        exhaustion[indices]
        - exhaustion[:, None]
    )

    delta_activation = (
        activation[indices]
        - activation[:, None]
    )

    weights = 1.0 / (
        distances + 1e-6
    )

    weight_sum = np.sum(
        weights,
        axis=1,
    )

    exhaustion_gradient = (
        np.sum(
            weights * delta_exhaustion,
            axis=1,
        )
        / weight_sum
    )

    activation_gradient = (
        np.sum(
            weights * delta_activation,
            axis=1,
        )
        / weight_sum
    )

    # ------------------------------------------------------------------
    # 13. Local exhaustion structure
    # ------------------------------------------------------------------

    local_exhaustion = exhaustion[
        indices
    ]

    exhaustion_local_mean = (
        local_exhaustion.mean(axis=1)
    )

    exhaustion_local_std = (
        local_exhaustion.std(axis=1)
    )

    local_stability = 1.0 / (
        1.0 + exhaustion_local_std
    )

    # ------------------------------------------------------------------
    # 14. Orient the first state axis toward exhaustion
    # ------------------------------------------------------------------

    exhaustion_axis = X_state[:, 0].copy()

    correlation = np.corrcoef(
        exhaustion_axis,
        exhaustion,
    )[0, 1]

    if (
        np.isfinite(correlation)
        and correlation < 0
    ):

        exhaustion_axis *= -1.0
        X_state[:, 0] *= -1.0

        correlation = -correlation

    # ------------------------------------------------------------------
    # 15. Cell-level trajectory/state table
    # ------------------------------------------------------------------

    result = pd.DataFrame({
        "cell_id": aligned_ids.to_numpy(),

        args.sample_key:
            clean(
                adata.obs[
                    args.sample_key
                ]
            ).to_numpy(),

        args.subtype_key:
            clean(
                adata.obs[
                    args.subtype_key
                ]
            ).to_numpy(),

        args.subset_key:
            clean(
                adata.obs[
                    args.subset_key
                ]
            ).to_numpy(),

        "exhaustion_dysfunction_score":
            exhaustion,

        "activation_effector_score":
            activation,

        "exhaustion_gradient":
            exhaustion_gradient,

        "activation_gradient":
            activation_gradient,

        "local_exhaustion_mean":
            exhaustion_local_mean,

        "local_exhaustion_std":
            exhaustion_local_std,

        "local_state_stability":
            local_stability,

        "exhaustion_state_axis":
            exhaustion_axis,
    })

    for i, column in enumerate(
        state_columns
    ):
        result[column] = X_state[:, i]

    result.to_csv(
        out / "tcell_state_landscape.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 16. kNN state graph
    # ------------------------------------------------------------------

    rows = []

    for i in range(
        len(aligned_ids)
    ):

        for j in range(k):

            neighbor = indices[i, j]

            rows.append({
                "source_cell":
                    aligned_ids[i],

                "target_cell":
                    aligned_ids[neighbor],

                "distance":
                    float(
                        distances[i, j]
                    ),

                "exhaustion_source":
                    float(
                        exhaustion[i]
                    ),

                "exhaustion_target":
                    float(
                        exhaustion[neighbor]
                    ),

                "delta_exhaustion":
                    float(
                        exhaustion[neighbor]
                        - exhaustion[i]
                    ),

                "delta_activation":
                    float(
                        activation[neighbor]
                        - activation[i]
                    ),
            })

    graph = pd.DataFrame(rows)

    graph.to_csv(
        out / "tcell_state_knn_edges.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 17. Annotation summaries
    # ------------------------------------------------------------------

    summary_columns = [
        "exhaustion_dysfunction_score",
        "activation_effector_score",
        "exhaustion_gradient",
        "activation_gradient",
        "local_state_stability",
        "exhaustion_state_axis",
    ]

    annotation_summary = (
        result
        .groupby(args.subset_key)[
            summary_columns
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )

    annotation_summary.to_csv(
        out / "tcell_state_by_annotation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 18. Subtype summaries
    # ------------------------------------------------------------------

    subtype_summary = (
        result
        .groupby(args.subtype_key)[
            summary_columns
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )

    subtype_summary.to_csv(
        out / "tcell_state_by_subtype.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 19. Sample summaries
    # ------------------------------------------------------------------

    sample_summary = (
        result
        .groupby(args.sample_key)[
            summary_columns
        ]
        .agg(
            [
                "count",
                "mean",
                "median",
                "std",
            ]
        )
        .reset_index()
    )

    sample_summary.to_csv(
        out / "tcell_state_by_sample.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 20. Pre-Harmony and Harmony T-cell embeddings
    #
    # adata has already been restricted to the exact 35,214 T cells,
    # so NO SECOND T-cell MASK is applied here.
    # ------------------------------------------------------------------

    if "X_pca_pre_harmony" not in adata.obsm:
        raise ValueError(
            "Missing X_pca_pre_harmony in integrated H5AD."
        )

    if "X_harmony" not in adata.obsm:
        raise ValueError(
            "Missing X_harmony in integrated H5AD."
        )

    pre_t = np.asarray(
        adata.obsm[
            "X_pca_pre_harmony"
        ],
        dtype=np.float32,
    )

    harmony_t = np.asarray(
        adata.obsm[
            "X_harmony"
        ],
        dtype=np.float32,
    )

    if pre_t.shape[0] != len(result):
        raise ValueError(
            "Pre-Harmony T-cell embedding row count "
            "does not match trajectory table."
        )

    if harmony_t.shape[0] != len(result):
        raise ValueError(
            "Harmony T-cell embedding row count "
            "does not match trajectory table."
        )

    if not np.isfinite(pre_t).all():
        raise ValueError(
            "X_pca_pre_harmony contains NaN/Inf."
        )

    if not np.isfinite(harmony_t).all():
        raise ValueError(
            "X_harmony contains NaN/Inf."
        )

    # ------------------------------------------------------------------
    # 21. Representation variance summary
    # ------------------------------------------------------------------

    def embedding_pca_variance(X):

        X_scaled_embedding = (
            StandardScaler()
            .fit_transform(X)
        )

        n = min(
            20,
            X_scaled_embedding.shape[1],
            X_scaled_embedding.shape[0] - 1,
        )

        embedding_pca = PCA(
            n_components=n,
            svd_solver="randomized",
            random_state=42,
        )

        embedding_pca.fit(
            X_scaled_embedding
        )

        return float(
            np.sum(
                embedding_pca
                .explained_variance_ratio_
            )
        )

    representation_summary = {
        "t_cells":
            int(len(result)),

        "pre_harmony_dimensions":
            int(pre_t.shape[1]),

        "harmony_dimensions":
            int(harmony_t.shape[1]),

        "pre_harmony_finite":
            bool(np.isfinite(pre_t).all()),

        "harmony_finite":
            bool(np.isfinite(harmony_t).all()),

        "program_state_dimensions":
            int(X_state.shape[1]),

        "program_state_exhaustion_axis_correlation":
            float(correlation),

        "program_state_explained_variance":
            [
                float(x)
                for x in pca.explained_variance_ratio_
            ],

        "pre_harmony_variance_first20":
            embedding_pca_variance(pre_t),

        "harmony_variance_first20":
            embedding_pca_variance(harmony_t),
    }

    with open(
        out / "representation_comparison.json",
        "w",
    ) as f:

        json.dump(
            representation_summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 22. Trajectory proxy summary
    # ------------------------------------------------------------------

    directional_summary = {
        "definition": (
            "Local expression-state trajectory proxy based on "
            "neighbor-to-neighbor changes in exhaustion and "
            "activation program scores."
        ),

        "not_rna_velocity":
            True,

        "cells":
            int(len(result)),

        "neighbors_per_cell":
            int(k),

        "mean_exhaustion_gradient":
            float(
                np.mean(
                    exhaustion_gradient
                )
            ),

        "median_exhaustion_gradient":
            float(
                np.median(
                    exhaustion_gradient
                )
            ),

        "mean_activation_gradient":
            float(
                np.mean(
                    activation_gradient
                )
            ),

        "median_activation_gradient":
            float(
                np.median(
                    activation_gradient
                )
            ),
    }

    with open(
        out / "trajectory_proxy_summary.json",
        "w",
    ) as f:

        json.dump(
            directional_summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 23. Alignment/provenance report
    # ------------------------------------------------------------------

    alignment_report = {
        "full_cohort_cells":
            int(adata_full.n_obs),

        "scored_t_cells":
            int(len(scores)),

        "aligned_t_cells":
            int(adata.n_obs),

        "alignment_complete":
            True,

        "score_ids_all_present_in_cohort":
            True,

        "cell_order_verified":
            True,

        "curated_t_cell_annotation_count":
            n_curated_t,
    }

    with open(
        out / "tcell_alignment_report.json",
        "w",
    ) as f:

        json.dump(
            alignment_report,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 24. Final report
    # ------------------------------------------------------------------

    report = {
        "cohort":
            "GSE176078",

        "step":
            "07C_tcell_state_landscape",

        "input_file":
            Path(args.input).name,

        "score_file":
            Path(args.scores).name,

        "full_cohort_cells":
            int(adata_full.n_obs),

        "tcell_cells":
            int(len(result)),

        "programs":
            PROGRAMS,

        "state_dimensions":
            int(n_components),

        "neighbors":
            int(k),

        "representations":
            [
                "X_pca_pre_harmony",
                "X_harmony",
            ],

        "alignment":
            alignment_report,

        "outputs":
            [
                "tcell_state_landscape.csv",
                "tcell_state_knn_edges.csv",
                "tcell_state_by_annotation.csv",
                "tcell_state_by_subtype.csv",
                "tcell_state_by_sample.csv",
                "representation_comparison.json",
                "trajectory_proxy_summary.json",
                "tcell_alignment_report.json",
            ],

        "scientific_policy": {

            "trajectory": (
                "The trajectory quantity is an "
                "expression-state trajectory proxy "
                "derived from local program-score "
                "gradients."
            ),

            "rna_velocity": (
                "No RNA velocity is inferred because "
                "the input does not contain "
                "spliced/unspliced velocity layers."
            ),

            "deep_evi": (
                "The state landscape and graph are "
                "preparation for the subsequent "
                "Deep-EVI model and are not themselves "
                "a trained Deep-EVI index."
            ),

            "integration_selection": (
                "Pre-Harmony and Harmony representations "
                "are retained for downstream "
                "evidence-based comparison."
            ),
        },

        "status":
            "complete",
    }

    with open(
        out / "GSE176078_step7c_report.json",
        "w",
    ) as f:

        json.dump(
            report,
            f,
            indent=2,
        )

    print("=" * 80)
    print("STEP 7C COMPLETE")
    print(f"Full cohort: {adata_full.n_obs}")
    print(f"T cells: {len(result)}")
    print(f"State dimensions: {n_components}")
    print(f"kNN neighbors: {k}")
    print("=" * 80)


if __name__ == "__main__":
    main()