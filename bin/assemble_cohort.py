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
        description="Assemble normalized GSE176078 H5AD samples into one cohort."
    )

    p.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing normalized H5AD files."
    )

    p.add_argument(
        "--output",
        required=True,
        help="Output cohort H5AD."
    )

    p.add_argument(
        "--summary",
        required=True,
        help="Output assembly summary JSON."
    )

    return p.parse_args()


def main():

    args = parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    files = sorted(
        input_dir.glob("*_normalized.h5ad")
    )

    if not files:
        raise RuntimeError(
            f"No normalized H5AD files found in {input_dir}"
        )

    print(f"Found {len(files)} normalized H5AD files.")

    sample_ids = []
    cell_counts = []
    gene_counts = []

    reference_genes = None
    reference_var = None

    adatas = []

    for f in files:

        print(f"Reading: {f.name}")

        a = ad.read_h5ad(f)

        if a.n_obs == 0 or a.n_vars == 0:
            raise RuntimeError(
                f"Empty AnnData object: {f}"
            )

        sample_id = a.uns.get("sample_id")

        if sample_id is None:
            raise RuntimeError(
                f"Missing sample_id in {f}"
            )

        sample_id = str(sample_id)

        if sample_id in sample_ids:
            raise RuntimeError(
                f"Duplicate sample_id detected: {sample_id}"
            )

        genes = np.asarray(
            a.var_names.astype(str)
        )

        if len(np.unique(genes)) != len(genes):
            raise RuntimeError(
                f"Duplicate gene identifiers in {f}"
            )

        if reference_genes is None:

            reference_genes = genes.copy()

            reference_var = a.var.copy()

        else:

            if not np.array_equal(
                reference_genes,
                genes
            ):
                raise RuntimeError(
                    f"Gene ordering mismatch in {f}"
                )

        if "counts" not in a.layers:
            raise RuntimeError(
                f"Missing counts layer in {f}"
            )

        if a.layers["counts"].shape != a.shape:
            raise RuntimeError(
                f"Counts layer shape mismatch in {f}"
            )

        sample_ids.append(sample_id)
        cell_counts.append(int(a.n_obs))
        gene_counts.append(int(a.n_vars))

        adatas.append(a)

    print("All individual samples passed validation.")

    print("Concatenating cohort...")

    cohort = ad.concat(
        adatas,
        axis=0,
        join="outer",
        merge="same",
        index_unique=None
    )

    # Explicitly preserve the biological sample identity.
    if "sample_id" not in cohort.obs.columns:
        cohort.obs["sample_id"] = np.concatenate(
            [
                np.repeat(sample_id, n_cells)
                for sample_id, n_cells
                in zip(sample_ids, cell_counts)
            ]
        )

    cohort.obs["sample_id"] = (
        cohort.obs["sample_id"]
        .astype(str)
    )

    # Ensure cell identifiers are unique.
    if not cohort.obs_names.is_unique:
        cohort.obs_names_make_unique()

    # Validate final cohort dimensions.
    expected_cells = sum(cell_counts)

    if cohort.n_obs != expected_cells:
        raise RuntimeError(
            f"Cell count mismatch: "
            f"expected {expected_cells}, "
            f"got {cohort.n_obs}"
        )

    if cohort.n_vars != len(reference_genes):
        raise RuntimeError(
            f"Gene count mismatch: "
            f"expected {len(reference_genes)}, "
            f"got {cohort.n_vars}"
        )

    if "counts" not in cohort.layers:
        raise RuntimeError(
            "Final cohort is missing counts layer."
        )

    # Store cohort-level provenance.
    cohort.uns["deepevi_step"] = (
        "04_cohort_assembly"
    )

    cohort.uns["cohort_name"] = (
        "GSE176078"
    )

    cohort.uns["assembly"] = {
        "method": "anndata_concat",
        "axis": "cells",
        "join": "outer",
        "merge": "same",
        "input_samples": len(files),
        "input_files": [
            f.name for f in files
        ],
        "sample_ids": sample_ids,
        "total_cells": int(cohort.n_obs),
        "genes": int(cohort.n_vars),
        "counts_layer": "counts"
    }

    cohort.uns["source_samples"] = sample_ids

    # Verify sample counts after concatenation.
    observed_sample_counts = (
        cohort.obs["sample_id"]
        .value_counts()
        .to_dict()
    )

    expected_sample_counts = dict(
        zip(sample_ids, cell_counts)
    )

    if observed_sample_counts != expected_sample_counts:
        raise RuntimeError(
            "Sample-level cell counts changed "
            "during cohort assembly."
        )

    # Validate matrix integrity.
    if sp.issparse(cohort.X):

        if np.isnan(cohort.X.data).any():
            raise RuntimeError(
                "NaN detected in cohort X."
            )

        if np.isinf(cohort.X.data).any():
            raise RuntimeError(
                "Inf detected in cohort X."
            )

    else:

        if np.isnan(cohort.X).any():
            raise RuntimeError(
                "NaN detected in cohort X."
            )

        if np.isinf(cohort.X).any():
            raise RuntimeError(
                "Inf detected in cohort X."
            )

    summary = {
        "cohort": "GSE176078",
        "step": "04_cohort_assembly",
        "method": "anndata_concat",
        "samples": int(len(files)),
        "total_cells": int(cohort.n_obs),
        "genes": int(cohort.n_vars),
        "expected_cells": int(expected_cells),
        "sample_ids": sample_ids,
        "sample_cell_counts": expected_sample_counts,
        "counts_layer_present": (
            "counts" in cohort.layers
        ),
        "X_sparse": sp.issparse(cohort.X),
        "obs_columns": list(
            cohort.obs.columns
        ),
        "var_columns": list(
            cohort.var.columns
        ),
        "uns_keys": list(
            cohort.uns.keys()
        ),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Writing cohort H5AD...")

    cohort.write_h5ad(
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

    print(
        f"SUCCESS: "
        f"{cohort.n_obs} cells × "
        f"{cohort.n_vars} genes"
    )


if __name__ == "__main__":
    main()