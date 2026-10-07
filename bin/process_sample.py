#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.io


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--archive", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--min-genes", type=int, default=200)
    p.add_argument("--max-pct-mito", type=float, default=20.0)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--qc", required=True)
    p.add_argument("--summary", required=True)
    return p.parse_args()


def clean_id(x):
    return str(x).strip()


def find_one(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {name}, found {len(matches)}"
        )
    return matches[0]


def read_vector(path: Path):
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    if df.shape[1] < 1:
        raise RuntimeError(f"Empty identifier file: {path}")
    return df.iloc[:, 0].map(clean_id).to_numpy()


def main():
    args = get_args()
    archive = Path(args.archive).resolve()

    with tempfile.TemporaryDirectory(prefix="deepevi_qc_") as tmpdir:
        tmp = Path(tmpdir)

        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(tmp)

        matrix_file = find_one(tmp, "count_matrix_sparse.mtx")
        genes_file = find_one(tmp, "count_matrix_genes.tsv")
        barcodes_file = find_one(tmp, "count_matrix_barcodes.tsv")
        metadata_file = find_one(tmp, "metadata.csv")

        matrix = scipy.io.mmread(matrix_file).tocsr()

        genes = read_vector(genes_file)
        barcodes = read_vector(barcodes_file)

        if matrix.shape != (len(genes), len(barcodes)):
            raise RuntimeError(
                "Matrix dimensions do not match genes/barcodes: "
                f"matrix={matrix.shape}, genes={len(genes)}, "
                f"barcodes={len(barcodes)}"
            )

        if len(set(genes)) != len(genes):
            raise RuntimeError("Duplicate gene identifiers detected.")

        if len(set(barcodes)) != len(barcodes):
            raise RuntimeError("Duplicate cell barcodes detected.")

        # Matrix is genes x cells. AnnData uses cells x genes.
        X = matrix.transpose().tocsr().astype(np.float32)

        obs = pd.read_csv(metadata_file, dtype=str)

        # First CSV column is the cell identifier because the file was
        # exported with the row names as an unnamed first column.
        obs = obs.rename(columns={obs.columns[0]: "barcode"})
        obs["barcode"] = obs["barcode"].map(clean_id)

        if obs["barcode"].duplicated().any():
            raise RuntimeError("Duplicate barcodes in metadata.csv.")

        expression_barcodes = pd.Index(barcodes)
        metadata_barcodes = pd.Index(obs["barcode"])

        missing_metadata = expression_barcodes.difference(metadata_barcodes)
        extra_metadata = metadata_barcodes.difference(expression_barcodes)

        if len(missing_metadata) or len(extra_metadata):
            raise RuntimeError(
                "Barcode/metadata mismatch: "
                f"missing_metadata={len(missing_metadata)}, "
                f"extra_metadata={len(extra_metadata)}"
            )

        obs = obs.set_index("barcode").loc[expression_barcodes].copy()

        for col in ["nCount_RNA", "nFeature_RNA", "percent.mito"]:
            if col in obs.columns:
                obs[col] = pd.to_numeric(obs[col], errors="coerce")

        var = pd.DataFrame(index=genes)
        var["gene_name"] = genes

        adata = ad.AnnData(X=X, obs=obs, var=var)
        adata.obs_names = barcodes
        adata.var_names = genes

        # Recompute QC directly from the expression matrix.
        adata.obs["n_genes_by_counts"] = np.asarray(
            (adata.X > 0).sum(axis=1)
        ).ravel()

        adata.obs["total_counts"] = np.asarray(
            adata.X.sum(axis=1)
        ).ravel()

        mito_mask = adata.var_names.str.upper().str.startswith("MT-")

        if mito_mask.sum() == 0:
            adata.obs["pct_counts_mt"] = np.nan
            mito_status = "NO_MT_GENES_DETECTED"
        else:
            mito_counts = np.asarray(
                adata[:, mito_mask].X.sum(axis=1)
            ).ravel()

            total = adata.obs["total_counts"].to_numpy()

            adata.obs["pct_counts_mt"] = np.divide(
                mito_counts * 100.0,
                total,
                out=np.zeros_like(mito_counts, dtype=float),
                where=total > 0,
            )

            mito_status = "COMPUTED"

        # Conservative QC flag. We retain all cells in the H5AD at this stage
        # and do not irreversibly discard cells before cohort-level review.
        qc_pass = (
            adata.obs["n_genes_by_counts"] >= args.min_genes
        )

        if mito_status == "COMPUTED":
            qc_pass &= (
                adata.obs["pct_counts_mt"] < args.max_pct_mito
            )

        adata.obs["qc_pass_conservative"] = qc_pass.to_numpy()

        # Provenance.
        adata.uns["deepevi_step"] = "02_qc"
        adata.uns["sample_id"] = args.sample
        adata.uns["source_archive"] = archive.name
        adata.uns["qc_thresholds"] = {
            "min_genes": args.min_genes,
            "max_pct_mito": args.max_pct_mito,
        }
        adata.uns["mitochondrial_status"] = mito_status

        qc_cols = [
            c for c in [
                "n_genes_by_counts",
                "total_counts",
                "pct_counts_mt",
                "qc_pass_conservative",
                "nCount_RNA",
                "nFeature_RNA",
                "percent.mito",
                "subtype",
                "celltype_major",
                "celltype_minor",
                "celltype_subset",
            ]
            if c in adata.obs.columns
        ]

        qc = adata.obs[qc_cols].copy()
        qc.insert(0, "barcode", qc.index)
        qc.insert(0, "sample_id", args.sample)
        qc.to_csv(args.qc, index=False)

        passed = int(adata.obs["qc_pass_conservative"].sum())

        summary = {
            "sample_id": args.sample,
            "gsm": args.sample.split("_")[0],
            "archive": archive.name,
            "input_cells": int(adata.n_obs),
            "input_genes": int(adata.n_vars),
            "input_nonzero": int(adata.X.nnz),
            "conservative_qc_pass_cells": passed,
            "conservative_qc_fail_cells": int(adata.n_obs - passed),
            "conservative_retention_fraction": (
                float(passed / adata.n_obs) if adata.n_obs else 0.0
            ),
            "min_genes": args.min_genes,
            "max_pct_mito": args.max_pct_mito,
            "mitochondrial_status": mito_status,
            "metadata_rows": int(obs.shape[0]),
            "metadata_columns": list(obs.columns),
        }

        # We intentionally save the complete sample matrix plus QC flags.
        # Irreversible filtering will be decided after cohort QC review.
        adata.write_h5ad(args.h5ad, compression="gzip")

        with open(args.summary, "w") as fh:
            json.dump(summary, fh, indent=2)

        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()