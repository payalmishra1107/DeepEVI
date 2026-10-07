#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


def load_table(path):
    path = Path(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-h5ad",
        required=True
    )
    parser.add_argument(
        "--split-manifest",
        required=True
    )
    parser.add_argument(
        "--out-h5ad",
        required=True
    )
    parser.add_argument(
        "--out-report",
        required=True
    )

    args = parser.parse_args()

    input_h5ad = Path(args.input_h5ad)
    manifest_path = Path(args.split_manifest)
    output_h5ad = Path(args.out_h5ad)
    report_path = Path(args.out_report)

    output_h5ad.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Load source expression matrix
    # ------------------------------------------------------------

    adata = ad.read_h5ad(input_h5ad)

    adata.obs_names = adata.obs_names.astype(str)
    adata.var_names = adata.var_names.astype(str)

    if adata.obs_names.duplicated().any():
        raise RuntimeError(
            "Duplicate cell IDs in source H5AD."
        )

    # ------------------------------------------------------------
    # Load the exact 08A split
    # ------------------------------------------------------------

    manifest = load_table(manifest_path)

    required = {
        "cell_id",
        "sample_id",
        "split"
    }

    missing = required - set(manifest.columns)

    if missing:
        raise RuntimeError(
            f"Split manifest missing columns: {missing}"
        )

    manifest["cell_id"] = (
        manifest["cell_id"]
        .astype(str)
    )

    manifest["sample_id"] = (
        manifest["sample_id"]
        .astype(str)
    )

    manifest["split"] = (
        manifest["split"]
        .astype(str)
        .str.lower()
    )

    # ------------------------------------------------------------
    # Exact test population
    # ------------------------------------------------------------

    test_manifest = manifest[
        manifest["split"] == "test"
    ].copy()

    test_manifest = (
        test_manifest
        .drop_duplicates("cell_id")
        .reset_index(drop=True)
    )

    if len(test_manifest) != 9864:
        raise RuntimeError(
            f"Expected 9864 test cells from 08A, "
            f"found {len(test_manifest)}."
        )

    test_ids = test_manifest["cell_id"].tolist()

    missing_cells = sorted(
        set(test_ids) -
        set(adata.obs_names)
    )

    if missing_cells:
        raise RuntimeError(
            f"{len(missing_cells)} test cells are missing "
            f"from the Step 5 H5AD."
        )

    # ------------------------------------------------------------
    # Preserve exact manifest order
    # ------------------------------------------------------------

    test = adata[
        test_ids,
        :
    ].copy()

    # ------------------------------------------------------------
    # Verify identity
    # ------------------------------------------------------------

    if list(test.obs_names) != test_ids:
        raise RuntimeError(
            "Test-cell ordering does not match split manifest."
        )

    if "celltype_major" in test.obs.columns:

        bad = (
            test.obs["celltype_major"]
            .astype(str)
            != "T-cells"
        ).sum()

        if bad:
            raise RuntimeError(
                f"{bad} test cells are not annotated as T-cells."
            )

    # Add authoritative split metadata
    manifest_index = (
        test_manifest
        .set_index("cell_id")
    )

    test.obs["sample_id"] = [
        manifest_index.loc[c, "sample_id"]
        for c in test.obs_names
    ]

    test.obs["benchmark_split"] = "test"

    # ------------------------------------------------------------
    # Confirm normalized expression
    # ------------------------------------------------------------

    if test.X is None:
        raise RuntimeError(
            "Test H5AD contains no expression matrix."
        )

    if not np.isfinite(
        test.X.data
        if hasattr(test.X, "data")
        else test.X
    ).all():

        raise RuntimeError(
            "Non-finite values found in test expression."
        )

    # ------------------------------------------------------------
    # Write
    # ------------------------------------------------------------

    test.uns["benchmark_provenance"] = {
        "source": str(input_h5ad),
        "split_manifest": str(manifest_path),
        "split": "test",
        "test_cells": int(test.n_obs),
        "genes": int(test.n_vars),
        "source_h5ad_unchanged": True,
        "deep_evi_retrained": False,
        "test_used_for_model_selection": False,
        "purpose": (
            "Frozen held-out gene-level expression artifact "
            "for 10A Deep-EVI benchmarking."
        )
    }

    test.write_h5ad(
        output_h5ad,
        compression="gzip"
    )

    report = {
        "status": "complete",
        "source_cells": int(adata.n_obs),
        "source_genes": int(adata.n_vars),
        "test_cells": int(test.n_obs),
        "test_genes": int(test.n_vars),
        "test_samples": int(
            test.obs["sample_id"].nunique()
        ),
        "all_test_cells_tcells": True,
        "expression_finite": True,
        "deep_evi_retrained": False,
        "test_used_for_model_selection": False,
        "rna_velocity": False
    }

    report_path.write_text(
        json.dumps(
            report,
            indent=2
        )
    )

    print("=" * 80)
    print("10A-1 HELD-OUT TEST EXPRESSION")
    print("=" * 80)
    print(f"Source cells:       {adata.n_obs}")
    print(f"Source genes:       {adata.n_vars}")
    print(f"Test cells:         {test.n_obs}")
    print(f"Test genes:         {test.n_vars}")
    print(
        f"Test samples:       "
        f"{test.obs['sample_id'].nunique()}"
    )
    print("All test cells T-cells: TRUE")
    print("STATUS: PASS")


if __name__ == "__main__":
    main()