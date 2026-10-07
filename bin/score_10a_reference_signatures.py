#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


DEEP_EVI_CONSTRUCTION_GENES = {
    "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2",
    "CD8A", "CD8B", "NKG7", "GNLY", "GZMB",
    "GZMH", "PRF1", "CTSW",
    "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IL7R",
    "CXCL13", "PDCD1", "ICOS", "CXCR5",
    "IFNG", "TNF", "IL2", "CD69", "HLA-DRA",
    "HLA-DRB1",
    "LAG3", "HAVCR2", "TOX", "TOX2", "ENTPD1"
}


def zscore(x):
    x = np.asarray(x, dtype=float)

    mean = np.nanmean(x)
    std = np.nanstd(x)

    if not np.isfinite(std) or std == 0:
        return np.zeros_like(x, dtype=float)

    return (x - mean) / std


def main():

    parser = argparse.ArgumentParser(
        description="Score reference T-cell gene sets on frozen 10A test cells."
    )

    parser.add_argument(
        "--input-h5ad",
        required=True
    )

    parser.add_argument(
        "--signatures",
        required=True
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # Load expression
    # ------------------------------------------------------------

    adata = ad.read_h5ad(
        args.input_h5ad
    )

    adata.var_names = (
        adata.var_names
        .astype(str)
    )

    if adata.n_obs != 9864:
        raise RuntimeError(
            f"Expected 9864 test cells, "
            f"found {adata.n_obs}"
        )

    if adata.n_vars != 29733:
        raise RuntimeError(
            f"Expected 29733 genes, "
            f"found {adata.n_vars}"
        )

    # ------------------------------------------------------------
    # Load signature definitions
    # ------------------------------------------------------------

    signatures = pd.read_csv(
        args.signatures,
        sep="\t"
    )

    required = {
        "signature_id",
        "signature_name",
        "category",
        "genes",
        "source_reference"
    }

    missing = required - set(signatures.columns)

    if missing:
        raise RuntimeError(
            f"Signature definition missing columns: {missing}"
        )

    gene_to_index = {
        gene: i
        for i, gene in enumerate(adata.var_names)
    }

    score_columns = []
    metadata = []
    score_matrix = {}

    # ------------------------------------------------------------
    # Score each reference gene set
    # ------------------------------------------------------------

    for _, row in signatures.iterrows():

        signature_id = str(
            row["signature_id"]
        )

        genes = [
            g.strip()
            for g in str(row["genes"]).split(";")
            if g.strip()
        ]

        present = [
            gene
            for gene in genes
            if gene in gene_to_index
        ]

        missing_genes = [
            gene
            for gene in genes
            if gene not in gene_to_index
        ]

        if len(present) == 0:
            raise RuntimeError(
                f"No genes from signature "
                f"{signature_id} were found."
            )

        indices = [
            gene_to_index[g]
            for g in present
        ]

        values = adata[:, indices].X

        if hasattr(values, "toarray"):
            values = values.toarray()

        values = np.asarray(
            values,
            dtype=float
        )

        raw_score = np.mean(
            values,
            axis=1
        )

        standardized_score = zscore(
            raw_score
        )

        column = (
            "reference_"
            + signature_id.lower()
        )

        score_matrix[column] = (
            standardized_score
        )

        score_columns.append(column)

        overlap = sorted(
            set(present) &
            DEEP_EVI_CONSTRUCTION_GENES
        )

        metadata.append({
            "signature_id": signature_id,
            "signature_name": row["signature_name"],
            "category": row["category"],
            "source_reference": row["source_reference"],
            "genes_requested": len(genes),
            "genes_present": len(present),
            "genes_missing": len(missing_genes),
            "missing_genes": ";".join(
                missing_genes
            ),
            "deep_evi_construction_overlap_count":
                len(overlap),
            "deep_evi_construction_overlap":
                ";".join(overlap)
        })

    # ------------------------------------------------------------
    # Output cell-level scores
    # ------------------------------------------------------------

    score_df = pd.DataFrame(
        {
            "cell_id":
                adata.obs_names.astype(str)
        }
    )

    if "sample_id" in adata.obs.columns:
        score_df["sample_id"] = (
            adata.obs["sample_id"]
            .astype(str)
            .values
        )

    if "celltype_subset" in adata.obs.columns:
        score_df["celltype_subset"] = (
            adata.obs["celltype_subset"]
            .astype(str)
            .values
        )

    if "subtype" in adata.obs.columns:
        score_df["subtype"] = (
            adata.obs["subtype"]
            .astype(str)
            .values
        )

    for column in score_columns:
        score_df[column] = score_matrix[column]

    score_df.to_csv(
        outdir /
        "GSE176078_10A_reference_signature_scores.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Output metadata
    # ------------------------------------------------------------

    metadata_df = pd.DataFrame(
        metadata
    )

    metadata_df.to_csv(
        outdir /
        "GSE176078_10A_reference_signature_metadata.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    report = {
        "cohort": "GSE176078",
        "step": "10A_reference_signature_scoring",
        "status": "complete",
        "test_cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "reference_signatures": int(
            len(signatures)
        ),
        "deep_evi_retrained": False,
        "test_used_for_model_selection": False,
        "rna_velocity": False,
        "tcga_used": False,
        "construction_gene_overlap_reported": True,
        "gene_score_method":
            "mean expression followed by test-cohort z-standardization",
        "scientific_warning":
            "Reference signatures are benchmark constructs. "
            "Gene overlap with Deep-EVI construction programs "
            "is explicitly reported and must be considered "
            "when interpreting concordance."
    }

    with open(
        outdir /
        "GSE176078_10A_10A2_report.json",
        "w"
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2
        )

    print("=" * 80)
    print("10A-2 REFERENCE SIGNATURE SCORING")
    print("=" * 80)

    print(
        f"Test cells:           {adata.n_obs}"
    )

    print(
        f"Genes:                 {adata.n_vars}"
    )

    print(
        f"Signatures:            {len(signatures)}"
    )

    print("\nSignature metadata:")

    print(
        metadata_df[
            [
                "signature_id",
                "genes_requested",
                "genes_present",
                "deep_evi_construction_overlap_count"
            ]
        ].to_string(index=False)
    )

    print("\nSTATUS: PASS")


if __name__ == "__main__":
    main()