#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Force CPU before importing scib-metrics/JAX.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

from scib_metrics.benchmark import (
    BatchCorrection,
    BioConservation,
    Benchmarker,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Step 6B: standardized scIB-metrics benchmarking "
            "of pre-Harmony and Harmony embeddings."
        )
    )

    p.add_argument("--input", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--plot-dir", required=True)

    p.add_argument("--batch-key", default="sample_id")
    p.add_argument("--label-key", default="celltype_major")

    p.add_argument("--n-jobs", type=int, default=2)

    return p.parse_args()


def validate_input(adata, args):

    required_obs = [
        args.batch_key,
        args.label_key,
    ]

    for key in required_obs:
        if key not in adata.obs.columns:
            raise RuntimeError(
                f"Missing required obs column: {key}"
            )

    required_embeddings = [
        "X_pca_pre_harmony",
        "X_harmony",
    ]

    for key in required_embeddings:
        if key not in adata.obsm:
            raise RuntimeError(
                f"Missing required embedding: {key}"
            )

    for key in required_embeddings:

        X = np.asarray(
            adata.obsm[key],
            dtype=np.float32
        )

        if X.ndim != 2:
            raise RuntimeError(
                f"{key} is not a 2D embedding."
            )

        if X.shape[0] != adata.n_obs:
            raise RuntimeError(
                f"{key} cell count mismatch."
            )

        if not np.isfinite(X).all():
            raise RuntimeError(
                f"{key} contains NaN or Inf."
            )

        print(
            f"{key}: "
            f"{X.shape[0]} cells × {X.shape[1]} dimensions"
        )


def save_umap(
    adata,
    embedding_key,
    color_key,
    output_path,
):

    temp = adata.copy()

    sc.pp.neighbors(
        temp,
        use_rep=embedding_key,
        n_neighbors=30,
        metric="euclidean",
    )

    sc.tl.umap(
        temp,
        random_state=42,
    )

    sc.pl.umap(
        temp,
        color=color_key,
        show=False,
        frameon=False,
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close("all")


def main():

    args = parse_args()

    input_path = Path(args.input)
    csv_path = Path(args.output_csv)
    json_path = Path(args.output_json)
    plot_dir = Path(args.plot_dir)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("STEP 6B — STANDARDIZED INTEGRATION BENCHMARK")
    print("=" * 78)

    print(
        f"JAX_PLATFORMS="
        f"{os.environ.get('JAX_PLATFORMS')}"
    )

    print(
        f"CUDA_VISIBLE_DEVICES="
        f"{os.environ.get('CUDA_VISIBLE_DEVICES')}"
    )

    try:
        import jax

        print(
            f"JAX version: {jax.__version__}"
        )

        print(
            f"JAX devices: {jax.devices()}"
        )

    except Exception as exc:

        print(
            f"JAX device check unavailable: {exc}"
        )

    print()

    print(
        f"Reading: {input_path}"
    )

    adata = ad.read_h5ad(
        input_path
    )

    print(
        f"Input: "
        f"{adata.n_obs} cells × "
        f"{adata.n_vars} genes"
    )

    validate_input(
        adata,
        args,
    )

    # The current scib-metrics benchmark expects normalized,
    # non-integrated data in adata.X and uses HVGs for its
    # internal preparation when needed.
    if "highly_variable" not in adata.var.columns:
        raise RuntimeError(
            "adata.var['highly_variable'] is missing."
        )

    hvg_count = int(
        np.asarray(
            adata.var["highly_variable"],
            dtype=bool,
        ).sum()
    )

    print(
        f"Highly variable genes: {hvg_count}"
    )

    if hvg_count == 0:
        raise RuntimeError(
            "No highly variable genes found."
        )

    # Explicit names for the benchmark.
    adata.obsm["Pre_Harmony"] = np.asarray(
        adata.obsm["X_pca_pre_harmony"],
        dtype=np.float32,
    )

    adata.obsm["Harmony"] = np.asarray(
        adata.obsm["X_harmony"],
        dtype=np.float32,
    )

    print()
    print("=" * 78)
    print("INITIALIZING SCIB-METRICS BENCHMARK")
    print("=" * 78)

    # Use the standardized scib-metrics metric sets.
    bio_metrics = BioConservation(
        isolated_labels=True,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=True,
        silhouette_label=True,
        clisi_knn=True,
    )

    batch_metrics = BatchCorrection(
        bras=True,
        ilisi_knn=True,
        kbet_per_label=True,
        graph_connectivity=True,
        pcr_comparison=True,
        sbee=False,
    )

    benchmarker = Benchmarker(
        adata,
        batch_key=args.batch_key,
        label_key=args.label_key,
        embedding_obsm_keys=[
            "Pre_Harmony",
            "Harmony",
        ],
        bio_conservation_metrics=bio_metrics,
        batch_correction_metrics=batch_metrics,
        pre_integrated_embedding_obsm_key="Pre_Harmony",
        n_jobs=args.n_jobs,
        progress_bar=True,
        solver="arpack",
    )

    print()
    print("=" * 78)
    print("RUNNING SCIB-METRICS BENCHMARK")
    print("=" * 78)

    benchmarker.benchmark()

    print()
    print("=" * 78)
    print("EXTRACTING RESULTS")
    print("=" * 78)

    results = benchmarker.get_results(
        min_max_scale=False,
        clean_names=True,
    )

    print(
        results.to_string()
    )

    # Save the complete benchmark table.
    results_reset = (
        results
        .reset_index()
        .rename(
            columns={
                "Embedding": "embedding"
            }
        )
    )

    results_reset.to_csv(
        csv_path,
        index=False,
    )

    # Also save a machine-readable JSON summary.
    summary = {
        "cohort": "GSE176078",
        "step": "06B_standardized_integration_benchmark",
        "input_file": input_path.name,
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "hvg_count": hvg_count,
        "batch_key": args.batch_key,
        "biological_label": args.label_key,
        "embeddings": [
            "Pre_Harmony",
            "Harmony",
        ],
        "scib_metrics_version": "0.6.1",
        "benchmark_components": {
            "bio_conservation": [
                "isolated_labels",
                "kmeans_nmi",
                "kmeans_ari",
                "silhouette_label",
                "clisi_knn",
            ],
            "batch_correction": [
                "bras",
                "ilisi_knn",
                "kbet_per_label",
                "graph_connectivity",
                "pcr_comparison",
            ],
        },
        "aggregate_scores_are_diagnostic_only": True,
        "selection_rule": (
            "No single aggregate score is used to automatically "
            "select the downstream representation. Selection "
            "requires joint assessment of batch correction, "
            "biological conservation, and downstream biological "
            "validity for tumour-immune and T-cell analyses."
        ),
        "status": "benchmark_complete",
    }

    with open(
        json_path,
        "w",
    ) as fh:

        json.dump(
            summary,
            fh,
            indent=2,
        )

    print()
    print("=" * 78)
    print("GENERATING UMAP DIAGNOSTICS")
    print("=" * 78)

    embedding_keys = {
        "pre_harmony": "Pre_Harmony",
        "harmony": "Harmony",
    }

    colors = [
        args.batch_key,
        args.label_key,
    ]

    # Subtype is evaluated if present.
    if "subtype" in adata.obs.columns:
        colors.append("subtype")

    for emb_name, emb_key in embedding_keys.items():

        for color_key in colors:

            output_path = (
                plot_dir
                / f"{emb_name}_umap_{color_key}.png"
            )

            print(
                f"Creating: {output_path}"
            )

            save_umap(
                adata,
                emb_key,
                color_key,
                output_path,
            )

    print()
    print("=" * 78)
    print("STEP 6B SUCCESS")
    print("=" * 78)

    print(
        f"Benchmark CSV: {csv_path}"
    )

    print(
        f"Benchmark JSON: {json_path}"
    )

    print(
        f"Plots: {plot_dir}"
    )


if __name__ == "__main__":
    main()