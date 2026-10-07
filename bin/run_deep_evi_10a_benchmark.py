#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser(
        description="10A Deep-EVI held-out benchmark"
    )

    p.add_argument("--deep-evi-scores", required=True)
    p.add_argument("--state-scores", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--signatures", required=True)
    p.add_argument("--outdir", required=True)

    return p.parse_args()


def load_table(path):
    path = Path(path)

    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")

    return pd.read_csv(path)


def zscore(x):
    x = np.asarray(x, dtype=float)

    mu = np.nanmean(x)
    sd = np.nanstd(x)

    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x)

    return (x - mu) / sd


def mean_expression_score(
    state_df,
    genes,
    available_genes,
):
    present = [
        g for g in genes
        if g in available_genes
    ]

    if not present:
        return np.full(len(state_df), np.nan), present

    values = state_df[present].to_numpy(dtype=float)

    return np.nanmean(values, axis=1), present


def main():

    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    deep = load_table(args.deep_evi_scores)
    state = load_table(args.state_scores)
    manifest = load_table(args.split_manifest)
    signatures = load_table(args.signatures)

    required_deep = {
        "cell_id",
        "split",
        "deep_evi_raw",
    }

    required_state = {
        "cell_id",
        "exhaustion_dysfunction_score",
    }

    required_manifest = {
        "cell_id",
        "sample_id",
        "split",
    }

    if not required_deep.issubset(deep.columns):
        raise RuntimeError(
            f"Deep-EVI score table missing columns: "
            f"{required_deep - set(deep.columns)}"
        )

    if not required_state.issubset(state.columns):
        raise RuntimeError(
            f"State score table missing columns: "
            f"{required_state - set(state.columns)}"
        )

    if not required_manifest.issubset(manifest.columns):
        raise RuntimeError(
            f"Split manifest missing columns: "
            f"{required_manifest - set(manifest.columns)}"
        )

    deep = deep.copy()
    state = state.copy()
    manifest = manifest.copy()

    deep["cell_id"] = deep["cell_id"].astype(str)
    state["cell_id"] = state["cell_id"].astype(str)
    manifest["cell_id"] = manifest["cell_id"].astype(str)

    if deep["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell IDs in Deep-EVI table.")

    if state["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell IDs in state-score table.")

    if manifest["cell_id"].duplicated().any():
        manifest = (
            manifest
            .drop_duplicates("cell_id")
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Exact held-out test population
    # ------------------------------------------------------------------

    test_manifest = manifest[
        manifest["split"].astype(str).str.lower() == "test"
    ].copy()

    test_ids = set(test_manifest["cell_id"])

    if not test_ids:
        raise RuntimeError("No test cells found.")

    deep_test = deep[
        deep["cell_id"].isin(test_ids)
    ].copy()

    state_test = state[
        state["cell_id"].isin(test_ids)
    ].copy()

    if len(deep_test) != len(test_ids):
        raise RuntimeError(
            "Deep-EVI test-cell alignment is incomplete."
        )

    if len(state_test) != len(test_ids):
        raise RuntimeError(
            "State-score test-cell alignment is incomplete."
        )

    deep_test = deep_test.set_index("cell_id")
    state_test = state_test.set_index("cell_id")

    common_ids = sorted(
        test_ids &
        set(deep_test.index) &
        set(state_test.index)
    )

    if len(common_ids) != len(test_ids):
        raise RuntimeError(
            "Test-cell intersection is incomplete."
        )

    deep_test = deep_test.loc[common_ids]
    state_test = state_test.loc[common_ids]

    benchmark = pd.DataFrame(index=common_ids)

    benchmark["sample_id"] = test_manifest.set_index(
        "cell_id"
    ).loc[common_ids, "sample_id"].values

    benchmark["deep_evi"] = deep_test["deep_evi_raw"].astype(float)

    benchmark["direct_exhaustion_target"] = (
        state_test["exhaustion_dysfunction_score"]
        .astype(float)
        .values
    )

    # ------------------------------------------------------------------
    # Determine available expression-like columns
    #
    # Step 7B score table contains program scores, not raw genes.
    # Therefore the benchmark must use a gene-level table if available.
    # The state-score table is NOT silently treated as raw expression.
    # ------------------------------------------------------------------

    gene_columns = [
        c for c in state_test.columns
        if c.isupper() and c.isalpha()
    ]

    if not gene_columns:
        raise RuntimeError(
            "No gene-level expression columns were found in the "
            "provided state-score table. 10A requires a gene-level "
            "T-cell expression matrix for published signature scoring."
        )

    available_genes = set(gene_columns)

    # ------------------------------------------------------------------
    # Score published signatures
    # ------------------------------------------------------------------

    metadata_rows = []
    score_columns = []

    for _, row in signatures.iterrows():

        sid = str(row["signature_id"])
        name = str(row["signature_name"])
        category = str(row["category"])

        genes = [
            g.strip()
            for g in str(row["genes"]).split(";")
            if g.strip()
        ]

        score, present = mean_expression_score(
            state_test.reset_index(),
            genes,
            available_genes,
        )

        col = f"benchmark_{sid.lower()}"

        benchmark[col] = zscore(score)

        score_columns.append(col)

        metadata_rows.append({
            "signature_id": sid,
            "signature_name": name,
            "category": category,
            "source": row["source"],
            "genes_requested": len(genes),
            "genes_present": len(present),
            "genes_missing": len(
                set(genes) - set(present)
            ),
            "present_gene_list": ";".join(present),
        })

    metadata = pd.DataFrame(metadata_rows)

    # ------------------------------------------------------------------
    # Correlation benchmark
    # ------------------------------------------------------------------

    results = []

    target = benchmark["direct_exhaustion_target"].to_numpy()

    for col in [
        "deep_evi",
        "direct_exhaustion_target",
    ] + score_columns:

        x = benchmark[col].to_numpy(dtype=float)

        mask = (
            np.isfinite(x) &
            np.isfinite(target)
        )

        if mask.sum() < 10:
            continue

        rho, rho_p = spearmanr(
            x[mask],
            target[mask]
        )

        r, r_p = pearsonr(
            x[mask],
            target[mask]
        )

        results.append({
            "representation": col,
            "n_test_cells": int(mask.sum()),
            "spearman_rho": float(rho),
            "spearman_pvalue": float(rho_p),
            "pearson_r": float(r),
            "pearson_pvalue": float(r_p),
        })

    correlation_df = pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Deep-EVI versus each benchmark
    # ------------------------------------------------------------------

    pairwise = []

    deep_values = benchmark["deep_evi"].to_numpy(dtype=float)

    for col in score_columns:

        x = benchmark[col].to_numpy(dtype=float)

        mask = (
            np.isfinite(deep_values) &
            np.isfinite(x)
        )

        if mask.sum() < 10:
            continue

        rho, p = spearmanr(
            deep_values[mask],
            x[mask]
        )

        pairwise.append({
            "representation": col,
            "n_test_cells": int(mask.sum()),
            "deep_evi_spearman_rho": float(rho),
            "deep_evi_spearman_pvalue": float(p),
        })

    pairwise_df = pd.DataFrame(pairwise)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    benchmark.reset_index(names="cell_id").to_csv(
        outdir / "GSE176078_10A_test_cell_scores.csv",
        index=False,
    )

    correlation_df.to_csv(
        outdir / "GSE176078_10A_benchmark_correlations.csv",
        index=False,
    )

    pairwise_df.to_csv(
        outdir / "GSE176078_10A_deep_evi_pairwise.csv",
        index=False,
    )

    metadata.to_csv(
        outdir / "GSE176078_10A_signature_metadata.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    report = {
        "cohort": "GSE176078",
        "step": "10A_deep_evi_held_out_benchmark",
        "status": "complete",
        "test_cells": len(common_ids),
        "benchmark_signatures": len(signatures),
        "deep_evi_retrained": False,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "rna_velocity": False,
        "tcga_data_used": False,
        "patient_level_inference": False,
        "direct_exhaustion_target": True,
        "published_signature_gene_level_scoring": True,
        "scientific_interpretation":
            "10A compares the frozen Deep-EVI representation "
            "against predefined published/reference T-cell "
            "state signatures on the held-out GSE176078 test "
            "cells without retraining or test-set model selection."
    }

    with open(
        outdir / "GSE176078_10A_report.json",
        "w",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )

    print("=" * 80)
    print("10A DEEP-EVI HELD-OUT BENCHMARK")
    print("=" * 80)
    print(f"Test cells:              {len(common_ids)}")
    print(f"Benchmark signatures:    {len(signatures)}")
    print("Deep-EVI retrained:      FALSE")
    print("Test used for training:  FALSE")
    print("Model selection on test: FALSE")
    print("RNA velocity:            FALSE")
    print("=" * 80)

    print("\nBenchmark correlations:")
    print(
        correlation_df.to_string(index=False)
    )

    print("\nSTATUS: PASS — 10A benchmark completed.")


if __name__ == "__main__":
    main()