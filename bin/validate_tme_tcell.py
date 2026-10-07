#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from pynndescent import NNDescent


def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "Step 7A: TME and T-cell biological validation "
            "of pre-Harmony and Harmony representations."
        )
    )

    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)

    p.add_argument("--batch-key", default="sample_id")
    p.add_argument("--subtype-key", default="subtype")

    p.add_argument(
        "--annotation-keys",
        nargs="+",
        default=[
            "celltype_major",
            "celltype_minor",
            "celltype_subset",
        ],
    )

    p.add_argument(
        "--n-neighbors",
        type=int,
        default=30,
    )

    return p.parse_args()


def clean_series(series):

    return (
        series
        .astype("string")
        .fillna("NA")
        .str.strip()
    )


def annotation_inventory(adata, output_dir, keys):

    inventory_rows = []

    for key in keys:

        if key not in adata.obs.columns:
            inventory_rows.append(
                {
                    "annotation_key": key,
                    "present": False,
                    "n_unique": 0,
                    "top_values": "",
                }
            )
            continue

        values = clean_series(
            adata.obs[key]
        )

        counts = (
            values
            .value_counts()
            .head(30)
        )

        inventory_rows.append(
            {
                "annotation_key": key,
                "present": True,
                "n_unique": int(values.nunique()),
                "top_values": "; ".join(
                    f"{idx}={value}"
                    for idx, value in counts.items()
                ),
            }
        )

        counts_df = (
            counts
            .rename_axis(key)
            .reset_index(name="cell_count")
        )

        counts_df.to_csv(
            output_dir
            / f"annotation_{key}_counts.csv",
            index=False,
        )

    pd.DataFrame(
        inventory_rows
    ).to_csv(
        output_dir / "annotation_inventory.csv",
        index=False,
    )


def detect_t_cells(adata, annotation_keys):

    patterns = [
        r"\bt[\s_-]*cell\b",
        r"\bt[\s_-]*lymph",
        r"\bcd4\b",
        r"\bcd8\b",
        r"\btreg\b",
        r"\btcr\b",
        r"\bcytotoxic[\s_-]*t\b",
    ]

    combined = pd.Series(
        "",
        index=adata.obs_names,
        dtype=str,
    )

    for key in annotation_keys:

        if key not in adata.obs.columns:
            continue

        values = clean_series(
            adata.obs[key]
        )

        combined = (
            combined
            + " "
            + values
        )

    pattern = "|".join(patterns)

    tcell_flag = (
        combined
        .str.lower()
        .str.contains(
            pattern,
            regex=True,
            na=False,
        )
    )

    return tcell_flag


def composition_by_sample(
    adata,
    sample_key,
    label_key,
    output_path,
):

    if (
        sample_key not in adata.obs.columns
        or label_key not in adata.obs.columns
    ):
        return

    df = adata.obs[
        [sample_key, label_key]
    ].copy()

    df[sample_key] = clean_series(
        df[sample_key]
    )

    df[label_key] = clean_series(
        df[label_key]
    )

    counts = (
        df
        .groupby(
            [sample_key, label_key],
            observed=True,
        )
        .size()
        .rename("cell_count")
        .reset_index()
    )

    totals = (
        counts
        .groupby(
            sample_key,
            observed=True,
        )["cell_count"]
        .sum()
        .rename("sample_total")
        .reset_index()
    )

    counts = counts.merge(
        totals,
        on=sample_key,
        how="left",
    )

    counts["fraction"] = (
        counts["cell_count"]
        / counts["sample_total"]
    )

    counts.to_csv(
        output_path,
        index=False,
    )


def subtype_composition(
    adata,
    sample_key,
    subtype_key,
    output_path,
):

    if (
        sample_key not in adata.obs.columns
        or subtype_key not in adata.obs.columns
    ):
        return

    df = adata.obs[
        [sample_key, subtype_key]
    ].copy()

    df[sample_key] = clean_series(
        df[sample_key]
    )

    df[subtype_key] = clean_series(
        df[subtype_key]
    )

    result = (
        df
        .groupby(
            [sample_key, subtype_key],
            observed=True,
        )
        .size()
        .rename("cell_count")
        .reset_index()
    )

    totals = (
        result
        .groupby(
            sample_key,
            observed=True,
        )["cell_count"]
        .sum()
        .rename("sample_total")
        .reset_index()
    )

    result = result.merge(
        totals,
        on=sample_key,
        how="left",
    )

    result["fraction"] = (
        result["cell_count"]
        / result["sample_total"]
    )

    result.to_csv(
        output_path,
        index=False,
    )


def tcell_summary(
    adata,
    tcell_flag,
    sample_key,
    subtype_key,
    output_dir,
):

    summary = {
        "total_cells": int(adata.n_obs),
        "t_cells": int(tcell_flag.sum()),
        "t_cell_fraction": float(
            tcell_flag.mean()
        ),
    }

    if sample_key in adata.obs.columns:

        by_sample = pd.DataFrame(
            {
                sample_key:
                    clean_series(
                        adata.obs[sample_key]
                    ),
                "t_cell": tcell_flag,
            }
        )

        sample_result = (
            by_sample
            .groupby(
                sample_key,
                observed=True,
            )["t_cell"]
            .agg(
                t_cell_count="sum",
                total_cells="size",
            )
            .reset_index()
        )

        sample_result[
            "t_cell_fraction"
        ] = (
            sample_result["t_cell_count"]
            / sample_result["total_cells"]
        )

        sample_result.to_csv(
            output_dir
            / "tcell_by_sample.csv",
            index=False,
        )

    if subtype_key in adata.obs.columns:

        by_subtype = pd.DataFrame(
            {
                subtype_key:
                    clean_series(
                        adata.obs[subtype_key]
                    ),
                "t_cell": tcell_flag,
            }
        )

        subtype_result = (
            by_subtype
            .groupby(
                subtype_key,
                observed=True,
            )["t_cell"]
            .agg(
                t_cell_count="sum",
                total_cells="size",
            )
            .reset_index()
        )

        subtype_result[
            "t_cell_fraction"
        ] = (
            subtype_result["t_cell_count"]
            / subtype_result["total_cells"]
        )

        subtype_result.to_csv(
            output_dir
            / "tcell_by_subtype.csv",
            index=False,
        )

    with open(
        output_dir / "tcell_summary.json",
        "w",
    ) as fh:

        json.dump(
            summary,
            fh,
            indent=2,
        )

    return summary


def knn_indices(
    X,
    n_neighbors,
):

    index = NNDescent(
        X,
        n_neighbors=n_neighbors + 1,
        metric="euclidean",
        random_state=42,
        n_jobs=2,
    )

    indices, distances = (
        index.neighbor_graph
    )

    return indices[:, 1:]


def neighbor_overlap(
    X_a,
    X_b,
    n_neighbors,
):

    print(
        "Building pre-Harmony T-cell kNN..."
    )

    knn_a = knn_indices(
        X_a,
        n_neighbors,
    )

    print(
        "Building Harmony T-cell kNN..."
    )

    knn_b = knn_indices(
        X_b,
        n_neighbors,
    )

    overlaps = np.empty(
        X_a.shape[0],
        dtype=np.float32,
    )

    for i in range(
        X_a.shape[0]
    ):

        a = set(
            knn_a[i]
        )

        b = set(
            knn_b[i]
        )

        union = a | b

        if not union:
            overlaps[i] = 1.0
        else:
            overlaps[i] = (
                len(a & b)
                / len(union)
            )

    return {
        "mean_jaccard": float(
            np.mean(overlaps)
        ),
        "median_jaccard": float(
            np.median(overlaps)
        ),
        "q25_jaccard": float(
            np.quantile(
                overlaps,
                0.25,
            )
        ),
        "q75_jaccard": float(
            np.quantile(
                overlaps,
                0.75,
            )
        ),
    }


def tcell_neighbor_analysis(
    adata,
    tcell_flag,
    n_neighbors,
    output_dir,
):

    n_tcells = int(
        tcell_flag.sum()
    )

    if n_tcells < 100:

        return {
            "status": "insufficient_t_cells",
            "t_cells": n_tcells,
        }

    X_pre = np.asarray(
        adata.obsm[
            "X_pca_pre_harmony"
        ][tcell_flag],
        dtype=np.float32,
    )

    X_harmony = np.asarray(
        adata.obsm[
            "X_harmony"
        ][tcell_flag],
        dtype=np.float32,
    )

    result = neighbor_overlap(
        X_pre,
        X_harmony,
        n_neighbors,
    )

    result.update(
        {
            "status": "complete",
            "t_cells": n_tcells,
            "neighbors": n_neighbors,
        }
    )

    with open(
        output_dir
        / "tcell_neighbor_preservation.json",
        "w",
    ) as fh:

        json.dump(
            result,
            fh,
            indent=2,
        )

    return result


def detect_exhaustion_annotations(
    adata,
    annotation_keys,
):

    rows = []

    pattern = re.compile(
        r"exhaust|dysfunction|dysfunctional|"
        r"exhausted|terminal",
        re.IGNORECASE,
    )

    for key in annotation_keys:

        if key not in adata.obs.columns:
            continue

        values = (
            clean_series(
                adata.obs[key]
            )
            .unique()
        )

        matches = [
            value
            for value in values
            if pattern.search(value)
        ]

        rows.append(
            {
                "annotation_key": key,
                "exhaustion_related_labels": (
                    "; ".join(
                        sorted(matches)
                    )
                ),
                "n_matches": len(matches),
            }
        )

    return pd.DataFrame(rows)


def main():

    args = parse_args()

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("STEP 7A — TME / T-CELL BIOLOGICAL VALIDATION")
    print("=" * 78)

    adata = ad.read_h5ad(
        args.input
    )

    print(
        f"Input: {adata.n_obs} cells × "
        f"{adata.n_vars} genes"
    )

    required_embeddings = [
        "X_pca_pre_harmony",
        "X_harmony",
    ]

    for key in required_embeddings:

        if key not in adata.obsm:
            raise RuntimeError(
                f"Missing embedding: {key}"
            )

        X = np.asarray(
            adata.obsm[key],
            dtype=np.float32,
        )

        if not np.isfinite(X).all():
            raise RuntimeError(
                f"{key} contains NaN/Inf."
            )

    annotation_inventory(
        adata,
        output_dir,
        args.annotation_keys,
    )

    # Major cell-type composition.
    for key in args.annotation_keys:

        composition_by_sample(
            adata,
            args.batch_key,
            key,
            output_dir
            / f"composition_by_sample_{key}.csv",
        )

    subtype_composition(
        adata,
        args.batch_key,
        args.subtype_key,
        output_dir
        / "subtype_composition_by_sample.csv",
    )

    # Detect T-cell populations directly from available annotations.
    tcell_flag = detect_t_cells(
        adata,
        args.annotation_keys,
    )

    adata.obs[
        "_deepevi_tcell_flag"
    ] = tcell_flag

    summary = tcell_summary(
        adata,
        tcell_flag,
        args.batch_key,
        args.subtype_key,
        output_dir,
    )

    exhaustion_annotations = (
        detect_exhaustion_annotations(
            adata,
            args.annotation_keys,
        )
    )

    exhaustion_annotations.to_csv(
        output_dir
        / "exhaustion_annotation_inventory.csv",
        index=False,
    )

    neighbor_result = (
        tcell_neighbor_analysis(
            adata,
            tcell_flag,
            args.n_neighbors,
            output_dir,
        )
    )

    final_report = {
        "cohort": "GSE176078",
        "step": "07A_tme_tcell_biological_validation",
        "input_file": Path(
            args.input
        ).name,
        "cells": int(adata.n_obs),
        "genes": int(adata.n_vars),
        "batch_key": args.batch_key,
        "subtype_key": args.subtype_key,
        "annotation_keys": args.annotation_keys,
        "tcell_summary": summary,
        "tcell_neighbor_preservation":
            neighbor_result,
        "exhaustion_annotation_detection":
            exhaustion_annotations.to_dict(
                orient="records"
            ),
        "representations": [
            "X_pca_pre_harmony",
            "X_harmony",
        ],
        "interpretation_policy": (
            "No latent representation is selected "
            "automatically. T-cell neighborhood "
            "preservation, annotation structure, "
            "sample structure, subtype structure, "
            "and exhaustion-related evidence are "
            "evaluated jointly."
        ),
        "status": "validation_complete",
    }

    with open(
        output_dir
        / "GSE176078_step7a_report.json",
        "w",
    ) as fh:

        json.dump(
            final_report,
            fh,
            indent=2,
        )

    print()
    print("=" * 78)
    print("STEP 7A SUCCESS")
    print("=" * 78)

    print(
        f"T cells detected: "
        f"{summary['t_cells']} / "
        f"{summary['total_cells']}"
    )

    print(
        f"T-cell fraction: "
        f"{summary['t_cell_fraction']:.4f}"
    )

    print(
        f"Output directory: "
        f"{output_dir}"
    )


if __name__ == "__main__":
    main()