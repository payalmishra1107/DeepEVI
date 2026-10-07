#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 5: feature selection and sample-aware latent representation."
    )

    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)

    p.add_argument(
        "--n-hvgs",
        type=int,
        default=3000
    )

    p.add_argument(
        "--n-pcs",
        type=int,
        default=50
    )

    return p.parse_args()


def main():

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    print("=" * 70)
    print("STEP 5 — FEATURE SELECTION + LATENT REPRESENTATION")
    print("=" * 70)

    print(f"Input: {input_path}")

    adata = ad.read_h5ad(input_path)

    print(
        f"Input shape: "
        f"{adata.n_obs} cells × {adata.n_vars} genes"
    )

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise RuntimeError("Input AnnData object is empty.")

    if "counts" not in adata.layers:
        raise RuntimeError(
            "Required layers['counts'] is missing."
        )

    if "sample_id" not in adata.obs.columns:
        raise RuntimeError(
            "Required obs['sample_id'] is missing."
        )

    if not adata.var_names.is_unique:
        raise RuntimeError(
            "Gene identifiers are not unique."
        )

    if not adata.obs_names.is_unique:
        adata.obs_names_make_unique()

    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(
            np.asarray(
                adata.X,
                dtype=np.float32
            )
        )

    adata.X = adata.X.tocsr().astype(np.float32)

    # ------------------------------------------------------------
    # Highly variable genes
    # ------------------------------------------------------------

    print("\nSelecting highly variable genes...")

    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=args.n_hvgs,
        flavor="seurat",
        layer=None,
        inplace=True
    )

    hvg_mask = adata.var["highly_variable"].to_numpy()

    hvg_count = int(hvg_mask.sum())

    if hvg_count == 0:
        raise RuntimeError(
            "No highly variable genes were selected."
        )

    print(f"Selected HVGs: {hvg_count}")

    # ------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------

    print("\nRunning PCA...")

    n_pcs = min(
        args.n_pcs,
        hvg_count - 1
    )

    if n_pcs < 2:
        raise RuntimeError(
            "Insufficient HVGs for PCA."
        )

    sc.pp.pca(
    adata,
    n_comps=n_pcs,
    use_highly_variable=True,
    zero_center=False,
    svd_solver="arpack"

    )

    if "X_pca" not in adata.obsm:
        raise RuntimeError(
            "PCA failed: X_pca missing."
        )

    # ------------------------------------------------------------
    # Sample-aware representation
    # ------------------------------------------------------------

    # At this stage we preserve PCA as the primary latent
    # representation. Sample identity is retained explicitly
    # for downstream integration/modeling rather than being
    # removed blindly.

    adata.obsm["X_latent"] = (
        np.asarray(
            adata.obsm["X_pca"],
            dtype=np.float32
        )
    )

    # ------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------

    adata.uns["deepevi_step"] = (
        "05_feature_selection_pca"
    )

    adata.uns["latent_representation"] = {
        "method": "PCA",
        "n_pcs": int(n_pcs),
        "n_hvgs": int(hvg_count),
        "sample_aware": True,
        "sample_variable": "sample_id",
        "integration_status": "pre_integration_latent_space"
    }

    adata.uns["source_cohort"] = "GSE176078"

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    X_pca = adata.obsm["X_pca"]

    if np.isnan(X_pca).any():
        raise RuntimeError(
            "NaN detected in PCA representation."
        )

    if np.isinf(X_pca).any():
        raise RuntimeError(
            "Inf detected in PCA representation."
        )

    if X_pca.shape != (adata.n_obs, n_pcs):
        raise RuntimeError(
            f"Unexpected PCA shape: {X_pca.shape}"
        )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    sample_count = int(
        adata.obs["sample_id"].nunique()
    )

    summary = {
        "cohort": "GSE176078",
        "step": "05_feature_selection_pca",
        "input_file": input_path.name,
        "output_file": output_path.name,
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "samples": sample_count,
        "hvg_count": hvg_count,
        "requested_hvgs": int(args.n_hvgs),
        "n_pcs": int(n_pcs),
        "X_sparse": sp.issparse(adata.X),
        "counts_layer_present": (
            "counts" in adata.layers
        ),
        "sample_variable": "sample_id",
        "latent_representation": "X_latent",
        "integration_status": "pre_integration_latent_space",
        "pca_shape": list(X_pca.shape),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nWriting Step 5 H5AD...")

    adata.write_h5ad(
        output_path,
        compression="gzip"
    )

    with open(summary_path, "w") as fh:
        json.dump(
            summary,
            fh,
            indent=2
        )

    print(json.dumps(summary, indent=2))

    print("\n" + "=" * 70)
    print("STEP 5 SUCCESS")
    print(
        f"{adata.n_obs} cells × "
        f"{adata.n_vars} genes"
    )
    print(f"HVGs: {hvg_count}")
    print(f"PCA dimensions: {n_pcs}")
    print("Latent representation: X_latent")
    print("=" * 70)


if __name__ == "__main__":
    main()