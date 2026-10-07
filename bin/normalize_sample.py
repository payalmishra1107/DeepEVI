#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import scipy.sparse as sp


def parse_args():
    p = argparse.ArgumentParser(
        description="Normalize one Deep-EVI GSE176078 sample."
    )

    p.add_argument("--input", required=True)
    p.add_argument("--sample-id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--summary", required=True)
    p.add_argument(
        "--target-sum",
        type=float,
        default=10000.0
    )

    return p.parse_args()


def main():

    args = parse_args()

    sample_id = args.sample_id

    input_path = Path(args.input)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    adata = ad.read_h5ad(input_path)

    if adata.n_obs == 0 or adata.n_vars == 0:
        raise RuntimeError(
            f"Empty AnnData object: cells={adata.n_obs}, genes={adata.n_vars}"
        )

    # Ensure sparse CSR representation.
    if sp.issparse(adata.X):
        counts = adata.X.tocsr().astype(np.float32)
    else:
        counts = sp.csr_matrix(
            np.asarray(adata.X, dtype=np.float32)
        )

    # Preserve the original count matrix.
    adata.layers["counts"] = counts.copy()

    # Library size per cell.
    library_size = np.asarray(
        counts.sum(axis=1)
    ).ravel().astype(np.float64)

    if np.any(library_size <= 0):
        bad = int(np.sum(library_size <= 0))
        raise RuntimeError(
            f"Found {bad} cells with zero total counts."
        )

    # Normalize each cell to target_sum.
    scale = args.target_sum / library_size

    normalized = counts.multiply(
        scale[:, None]
    ).tocsr()

    # Sparse-safe log1p.
    normalized.data = np.log1p(
        normalized.data
    ).astype(np.float32)

    adata.X = normalized

    # Record provenance.
    adata.uns["sample_id"] = sample_id
    adata.uns["deepevi_step"] = "03_normalization"

    adata.uns["normalization"] = {
        "method": "library_size_normalization_then_log1p",
        "target_sum": args.target_sum,
        "input_file": input_path.name,
        "raw_counts_layer": "counts"
    }

    # Useful normalization diagnostics.
    normalized_totals = np.asarray(
        adata.X.sum(axis=1)
    ).ravel()

    summary = {
        "sample_id": sample_id,
        "input_file": input_path.name,
        "output_file": output_path.name,
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "raw_nonzero": int(counts.nnz),
        "target_sum": float(args.target_sum),
        "raw_library_size_median": float(
            np.median(library_size)
        ),
        "raw_library_size_min": float(
            np.min(library_size)
        ),
        "raw_library_size_max": float(
            np.max(library_size)
        ),
        "normalized_log1p_total_median": float(
            np.median(normalized_totals)
        ),
        "counts_layer_present": "counts" in adata.layers,
        "normalization_method":
            "library_size_normalization_then_log1p"
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

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


if __name__ == "__main__":
    main()