#!/usr/bin/env python3

from __future__ import annotations

import os

# Force CPU execution before importing libraries that may initialize JAX/XLA.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scib_metrics


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantitatively evaluate pre/post Harmony integration."
    )

    p.add_argument("--input", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--output-json", required=True)

    p.add_argument(
        "--batch-key",
        default="sample_id"
    )

    p.add_argument(
        "--label-key",
        default="celltype_major"
    )

    return p.parse_args()


def evaluate_embedding(
    X,
    batches,
    labels,
    name
):

    print("\n" + "=" * 70)
    print(f"EVALUATING: {name}")
    print("=" * 70)

    # ------------------------------------------------------------
    # Nearest-neighbor graph
    # ------------------------------------------------------------

    print("Building kNN graph...")

    neighbors = scib_metrics.nearest_neighbors.pynndescent(
        X,
        n_neighbors=30
    )

    # ------------------------------------------------------------
    # Batch correction metrics
    # ------------------------------------------------------------

    print("Computing iLISI...")

    ilisi = scib_metrics.ilisi_knn(
        neighbors,
        batches
    )

    print("Computing kBET...")

    kbet_result = scib_metrics.kbet(
        neighbors,
        batches
    )

    kbet = float(
        kbet_result[0]
        if isinstance(kbet_result, tuple)
        else kbet_result
    )

    print("Computing graph connectivity...")

    # Graph connectivity requires labels.
    graph_connectivity = (
        scib_metrics.graph_connectivity(
            neighbors,
            labels
        )
    )

    # ------------------------------------------------------------
    # Biological conservation
    # ------------------------------------------------------------

    print("Computing cell-type ASW...")

    celltype_asw = scib_metrics.silhouette_label(
        X,
        labels
    )

    print("Computing cLISI...")

    clisi = scib_metrics.clisi_knn(
        neighbors,
        labels
    )

    print("Computing NMI/ARI...")

    kmeans_result = (
        scib_metrics.nmi_ari_cluster_labels_kmeans(
            X,
            labels
        )
    )

    if isinstance(kmeans_result, dict):

        nmi = float(
            kmeans_result["nmi"]
        )

        ari = float(
            kmeans_result["ari"]
        )

    else:

        nmi = float(
            kmeans_result[0]
        )

        ari = float(
            kmeans_result[1]
        )

    return {
        "representation": name,
        "cells": int(X.shape[0]),
        "dimensions": int(X.shape[1]),

        "batch_ilisi": float(ilisi),

        "batch_kbet": float(kbet),

        "batch_graph_connectivity": float(
            graph_connectivity
        ),

        "biology_celltype_asw": float(
            celltype_asw
        ),

        "biology_clisi": float(
            clisi
        ),

        "biology_nmi": float(nmi),

        "biology_ari": float(ari)
    }


def main():

    args = parse_args()

    input_path = Path(args.input)
    csv_path = Path(args.output_csv)
    json_path = Path(args.output_json)

    print("=" * 70)
    print("STEP 6A — INTEGRATION EVALUATION")
    print("=" * 70)

    print("Execution configuration:")
    print(f"  JAX_PLATFORMS={os.environ.get('JAX_PLATFORMS')}")
    print(f"  CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")

    try:
       import jax
       print(f"  JAX version={jax.__version__}")
       print(f"  JAX devices={jax.devices()}")
    except Exception as exc:
       print(f"  JAX device check unavailable: {exc}")

    adata = ad.read_h5ad(input_path)

    print(
        f"Input: {adata.n_obs} cells × "
        f"{adata.n_vars} genes"
    )

    # ------------------------------------------------------------
    # Validate metadata
    # ------------------------------------------------------------

    if args.batch_key not in adata.obs.columns:
        raise RuntimeError(
            f"Missing batch key: {args.batch_key}"
        )

    if args.label_key not in adata.obs.columns:
        raise RuntimeError(
            f"Missing biological label: {args.label_key}"
        )

    batches = (
        adata.obs[args.batch_key]
        .astype(str)
        .to_numpy()
    )

    labels = (
        adata.obs[args.label_key]
        .astype(str)
        .to_numpy()
    )

    # ------------------------------------------------------------
    # Validate embeddings
    # ------------------------------------------------------------

    required = [
        "X_pca_pre_harmony",
        "X_harmony"
    ]

    for key in required:

        if key not in adata.obsm:

            raise RuntimeError(
                f"Missing embedding: {key}"
            )

    X_pre = np.asarray(
        adata.obsm["X_pca_pre_harmony"],
        dtype=np.float32
    )

    X_post = np.asarray(
        adata.obsm["X_harmony"],
        dtype=np.float32
    )

    if X_pre.shape != X_post.shape:

        raise RuntimeError(
            f"Embedding shape mismatch: "
            f"{X_pre.shape} vs {X_post.shape}"
        )

    if np.isnan(X_pre).any():
        raise RuntimeError(
            "NaN detected in pre-Harmony embedding."
        )

    if np.isnan(X_post).any():
        raise RuntimeError(
            "NaN detected in Harmony embedding."
        )

    # ------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------

    pre = evaluate_embedding(
        X_pre,
        batches,
        labels,
        "pre_harmony"
    )

    post = evaluate_embedding(
        X_post,
        batches,
        labels,
        "harmony"
    )

    results = pd.DataFrame(
        [pre, post]
    )

    # ------------------------------------------------------------
    # Delta calculations
    # ------------------------------------------------------------

    metric_columns = [
        "batch_ilisi",
        "batch_kbet",
        "batch_graph_connectivity",
        "biology_celltype_asw",
        "biology_clisi",
        "biology_nmi",
        "biology_ari"
    ]

    delta = {
        "representation": "harmony_minus_pre",
        "cells": int(adata.n_obs),
        "dimensions": int(X_post.shape[1])
    }

    for metric in metric_columns:

        delta[metric] = (
            float(post[metric])
            -
            float(pre[metric])
        )

    results = pd.concat(
        [
            results,
            pd.DataFrame([delta])
        ],
        ignore_index=True
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = {
        "cohort": "GSE176078",

        "step":
            "06A_integration_evaluation",

        "input_file":
            input_path.name,

        "cells":
            int(adata.n_obs),

        "dimensions":
            int(X_pre.shape[1]),

        "batch_key":
            args.batch_key,

        "biological_label":
            args.label_key,

        "representations": [
            "pre_harmony",
            "harmony"
        ],

        "metrics": [
            "iLISI",
            "kBET",
            "graph_connectivity",
            "celltype_ASW",
            "cLISI",
            "NMI",
            "ARI"
        ],

        "interpretation":
            "Higher batch-mixing metrics indicate greater "
            "sample mixing; biological metrics are evaluated "
            "for preservation rather than maximization of "
            "batch mixing alone.",

        "status":
            "evaluation_complete"
    }

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    json_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results.to_csv(
        csv_path,
        index=False
    )

    with open(json_path, "w") as fh:

        json.dump(
            summary,
            fh,
            indent=2
        )

    print("\n" + "=" * 70)
    print("INTEGRATION RESULTS")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("STEP 6A SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()