#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import harmonypy as hm
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 6: candidate Harmony integration of GSE176078."
    )

    p.add_argument(
        "--input",
        required=True,
        help="Input Step 5 H5AD."
    )

    p.add_argument(
        "--output",
        required=True,
        help="Output integrated H5AD."
    )

    p.add_argument(
        "--summary",
        required=True,
        help="Output integration summary JSON."
    )

    p.add_argument(
        "--batch-key",
        default="sample_id",
        help="Metadata variable used for sample-aware integration."
    )

    p.add_argument(
        "--n-pcs",
        type=int,
        default=50,
        help="Number of PCA dimensions used by Harmony."
    )

    return p.parse_args()


def main():

    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    print("=" * 70)
    print("STEP 6 — CANDIDATE HARMONY INTEGRATION")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load input
    # ------------------------------------------------------------

    print(f"Input: {input_path}")

    adata = ad.read_h5ad(input_path)

    print(
        f"Input: {adata.n_obs} cells × "
        f"{adata.n_vars} genes"
    )

    # ------------------------------------------------------------
    # Required input validation
    # ------------------------------------------------------------

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise RuntimeError(
            "Input AnnData object is empty."
        )

    if "X_pca" not in adata.obsm:
        raise RuntimeError(
            "X_pca is missing. Step 5 must be completed first."
        )

    if args.batch_key not in adata.obs.columns:
        raise RuntimeError(
            f"Missing batch variable: {args.batch_key}"
        )

    if not adata.obs_names.is_unique:
        raise RuntimeError(
            "Cell identifiers are not unique."
        )

    if not adata.var_names.is_unique:
        raise RuntimeError(
            "Gene identifiers are not unique."
        )

    if "counts" not in adata.layers:
        raise RuntimeError(
            "Required layers['counts'] is missing."
        )

    # ------------------------------------------------------------
    # PCA input
    # ------------------------------------------------------------

    X_pca = np.asarray(
        adata.obsm["X_pca"],
        dtype=np.float32
    )

    n_pcs = min(
        args.n_pcs,
        X_pca.shape[1]
    )

    X_pca = X_pca[:, :n_pcs]

    expected_pca_shape = (
        adata.n_obs,
        n_pcs
    )

    if X_pca.shape != expected_pca_shape:
        raise RuntimeError(
            f"Unexpected PCA shape: "
            f"{X_pca.shape}; "
            f"expected {expected_pca_shape}"
        )

    if np.isnan(X_pca).any():
        raise RuntimeError(
            "NaN detected in PCA representation."
        )

    if np.isinf(X_pca).any():
        raise RuntimeError(
            "Inf detected in PCA representation."
        )

    # ------------------------------------------------------------
    # Batch/sample information
    # ------------------------------------------------------------

    batch_values = (
        adata.obs[args.batch_key]
        .astype(str)
    )

    n_batches = int(
        batch_values.nunique()
    )

    if n_batches < 2:
        raise RuntimeError(
            "Integration requires at least two batches."
        )

    print(f"PCA dimensions: {n_pcs}")
    print(
        f"Integration variable: "
        f"{args.batch_key}"
    )
    print(
        f"Number of samples: "
        f"{n_batches}"
    )

    # ------------------------------------------------------------
    # Harmony
    # ------------------------------------------------------------

    print("\nRunning Harmony...")

    harmony_result = hm.run_harmony(
        X_pca,
        adata.obs,
        vars_use=[args.batch_key],
        ncores=2,
        verbose=True
    )

    # harmonypy versions may expose Z_corr with either
    # cells × dimensions or dimensions × cells orientation.
    #
    # Normalize the orientation explicitly to:
    #
    #     cells × PCA dimensions
    #
    X_harmony_raw = np.asarray(
        harmony_result.Z_corr,
        dtype=np.float32
    )

    print(
        f"Raw Harmony embedding shape: "
        f"{X_harmony_raw.shape}"
    )

    expected_shape = (
        adata.n_obs,
        n_pcs
    )

    transposed_shape = (
        n_pcs,
        adata.n_obs
    )

    if X_harmony_raw.shape == expected_shape:

        X_harmony = X_harmony_raw

    elif X_harmony_raw.shape == transposed_shape:

        X_harmony = X_harmony_raw.T

    else:

        raise RuntimeError(
            f"Unexpected Harmony shape: "
            f"{X_harmony_raw.shape}; "
            f"expected either "
            f"{expected_shape} or "
            f"{transposed_shape}"
        )

    print(
        f"Harmony embedding shape: "
        f"{X_harmony.shape}"
    )

    # ------------------------------------------------------------
    # Harmony validation
    # ------------------------------------------------------------

    if X_harmony.shape != expected_shape:
        raise RuntimeError(
            f"Final Harmony shape mismatch: "
            f"{X_harmony.shape}; "
            f"expected {expected_shape}"
        )

    if np.isnan(X_harmony).any():
        raise RuntimeError(
            "NaN detected in Harmony embedding."
        )

    if np.isinf(X_harmony).any():
        raise RuntimeError(
            "Inf detected in Harmony embedding."
        )

    # ------------------------------------------------------------
    # Store representations
    # ------------------------------------------------------------

    # Preserve the original PCA representation.
    adata.obsm["X_pca_pre_harmony"] = (
        X_pca.copy()
    )

    # Store Harmony representation.
    adata.obsm["X_harmony"] = (
        X_harmony.copy()
    )

    # ------------------------------------------------------------
    # Provenance
    # ------------------------------------------------------------

    adata.uns["deepevi_step"] = (
        "06_candidate_harmony_integration"
    )

    adata.uns["integration"] = {

        "method": "Harmony",

        "implementation": "harmonypy",

        "input_representation":
            "X_pca",

        "output_representation":
            "X_harmony",

        "batch_variable":
            args.batch_key,

        "n_pcs":
            int(n_pcs),

        "n_batches":
            int(n_batches),

        "status":
            "candidate_pending_quantitative_evaluation"
    }

    adata.uns["integration_note"] = (
        "Harmony is treated as a candidate integration "
        "method. Quantitative assessment of sample mixing "
        "and biological conservation is required before "
        "using X_harmony as the definitive integrated space."
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    summary = {

        "cohort":
            "GSE176078",

        "step":
            "06_candidate_harmony_integration",

        "input_file":
            input_path.name,

        "output_file":
            output_path.name,

        "cells":
            int(adata.n_obs),

        "genes":
            int(adata.n_vars),

        "samples":
            int(n_batches),

        "batch_variable":
            args.batch_key,

        "input_representation":
            "X_pca",

        "output_representation":
            "X_harmony",

        "n_pcs":
            int(n_pcs),

        "pca_shape":
            list(X_pca.shape),

        "harmony_shape":
            list(X_harmony.shape),

        "counts_layer_present":
            "counts" in adata.layers,

        "status":
            "candidate_pending_quantitative_evaluation"
    }

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nWriting integrated H5AD...")

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

    print(json.dumps(
        summary,
        indent=2
    ))

    print("\n" + "=" * 70)
    print("STEP 6 SUCCESS")
    print(
        f"{adata.n_obs} cells × "
        f"{n_pcs} integrated dimensions"
    )
    print("Representation: X_harmony")
    print(
        "Status: candidate pending "
        "quantitative evaluation"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()