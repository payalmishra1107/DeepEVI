#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr


def zscore(x):
    x = np.asarray(x, dtype=float)
    sd = np.nanstd(x, ddof=0)

    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(x, dtype=float)

    return (x - np.nanmean(x)) / sd


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)

    if keep.sum() < 3:
        return {
            "n": int(keep.sum()),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }

    rho, rho_p = spearmanr(x[keep], y[keep])
    r, r_p = pearsonr(x[keep], y[keep])

    return {
        "n": int(keep.sum()),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "pearson_r": float(r),
        "pearson_p": float(r_p),
    }


def residualize(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)

    keep = np.isfinite(y) & np.isfinite(x)

    residual = np.full(len(y), np.nan, dtype=float)

    if keep.sum() < 3:
        return residual

    X = np.column_stack([
        np.ones(keep.sum()),
        x[keep],
    ])

    beta, *_ = np.linalg.lstsq(
        X,
        y[keep],
        rcond=None,
    )

    residual[keep] = y[keep] - X @ beta

    return residual


def partial_spearman(y, x, control):
    y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    c = pd.Series(control).rank(method="average").to_numpy(dtype=float)

    keep = np.isfinite(y) & np.isfinite(x) & np.isfinite(c)

    if keep.sum() < 3:
        return {
            "n": int(keep.sum()),
            "partial_spearman_rho": np.nan,
            "partial_spearman_p": np.nan,
        }

    ry = residualize(y[keep], c[keep])
    rx = residualize(x[keep], c[keep])

    rho, p = spearmanr(rx, ry)

    return {
        "n": int(keep.sum()),
        "partial_spearman_rho": float(rho),
        "partial_spearman_p": float(p),
    }


def load_csv(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{name} not found: {path}"
        )

    df = pd.read_csv(path)

    if "cell_id" not in df.columns:
        raise ValueError(
            f"{name} does not contain required cell_id column"
        )

    df["cell_id"] = df["cell_id"].astype(str)

    if df["cell_id"].duplicated().any():
        dup = int(df["cell_id"].duplicated().sum())

        raise ValueError(
            f"{name} contains {dup} duplicated cell IDs"
        )

    return df


def main():

    parser = argparse.ArgumentParser(
        description=(
            "10A-5 held-out Deep-EVI "
            "progenitor-to-terminal state-axis analysis"
        )
    )

    parser.add_argument(
        "--deep-evi-scores",
        required=True,
    )

    parser.add_argument(
        "--reference-scores",
        required=True,
    )

    parser.add_argument(
        "--state-scores",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 90)
    print("10A-5 DEEP-EVI HELD-OUT STATE-AXIS ANALYSIS")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    deep_all = load_csv(
        args.deep_evi_scores,
        "Deep-EVI scores",
    )

    refs = load_csv(
        args.reference_scores,
        "reference scores",
    )

    state = load_csv(
        args.state_scores,
        "T-cell state scores",
    )

    print(
        f"Deep-EVI rows (all splits): {len(deep_all)}"
    )

    print(
        f"Reference rows:             {len(refs)}"
    )

    print(
        f"State-score rows:           {len(state)}"
    )

    # ------------------------------------------------------------------
    # Validate split structure
    # ------------------------------------------------------------------

    if "split" not in deep_all.columns:
        raise ValueError(
            "Deep-EVI scores do not contain split column"
        )

    split_values = sorted(
        deep_all["split"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    expected_splits = [
        "test",
        "train",
        "validation",
    ]

    if split_values != expected_splits:
        raise ValueError(
            "Unexpected Deep-EVI split values: "
            f"{split_values}"
        )

    split_counts = (
        deep_all["split"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Deep-EVI split counts:")
    print(split_counts.to_string())

    # ------------------------------------------------------------------
    # CRITICAL FIX:
    # Deep-EVI CSV contains all cells.
    # Select frozen held-out test cells here.
    # ------------------------------------------------------------------

    deep = deep_all[
        deep_all["split"].astype(str) == "test"
    ].copy()

    print()
    print(
        f"Frozen test Deep-EVI rows:   {len(deep)}"
    )

    if len(deep) != 9864:
        raise ValueError(
            "Expected exactly 9864 frozen test cells; "
            f"observed {len(deep)}"
        )

    if deep["cell_id"].duplicated().any():
        raise ValueError(
            "Test Deep-EVI subset contains duplicated cell IDs"
        )

    # ------------------------------------------------------------------
    # Required columns
    # ------------------------------------------------------------------

    required_refs = [
        "reference_progenitor_tpex",
        "reference_terminal_exhaustion",
    ]

    missing_refs = [
        c
        for c in required_refs
        if c not in refs.columns
    ]

    if missing_refs:
        raise ValueError(
            "Missing required reference columns: "
            + ", ".join(missing_refs)
        )

    required_deep = [
        "deep_evi_raw",
        "sample_id",
        "subtype",
        "celltype_subset",
        "split",
    ]

    missing_deep = [
        c
        for c in required_deep
        if c not in deep.columns
    ]

    if missing_deep:
        raise ValueError(
            "Missing required Deep-EVI columns: "
            + ", ".join(missing_deep)
        )

    required_state = [
        "exhaustion_dysfunction_score",
        "tcell_identity_score",
        "cd8_cytotoxic_score",
        "treg_score",
        "tfh_score",
        "activation_effector_score",
    ]

    missing_state = [
        c
        for c in required_state
        if c not in state.columns
    ]

    if missing_state:
        raise ValueError(
            "Missing required state-score columns: "
            + ", ".join(missing_state)
        )

    # ------------------------------------------------------------------
    # Reference score must be exactly the frozen test set
    # ------------------------------------------------------------------

    test_ids = set(
        deep["cell_id"].astype(str)
    )

    reference_ids = set(
        refs["cell_id"].astype(str)
    )

    if test_ids != reference_ids:
        missing_reference = sorted(
            test_ids - reference_ids
        )

        extra_reference = sorted(
            reference_ids - test_ids
        )

        raise ValueError(
            "Reference scores are not exactly aligned "
            "to the frozen test set. "
            f"Missing reference cells={len(missing_reference)}; "
            f"extra reference cells={len(extra_reference)}"
        )

    # ------------------------------------------------------------------
    # Merge exact test cells
    # ------------------------------------------------------------------

    merged = (
        deep[
            [
                "cell_id",
                "sample_id",
                "subtype",
                "celltype_subset",
                "split",
                "deep_evi_raw",
            ]
        ]
        .merge(
            refs[
                [
                    "cell_id",
                    "reference_progenitor_tpex",
                    "reference_terminal_exhaustion",
                ]
            ],
            on="cell_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            state[
                [
                    "cell_id",
                    "exhaustion_dysfunction_score",
                    "tcell_identity_score",
                    "cd8_cytotoxic_score",
                    "treg_score",
                    "tfh_score",
                    "activation_effector_score",
                ]
            ],
            on="cell_id",
            how="inner",
            validate="one_to_one",
        )
    )

    if len(merged) != len(deep):
        raise ValueError(
            "Cell alignment failure: "
            f"merged={len(merged)} "
            f"expected={len(deep)}"
        )

    if merged["cell_id"].duplicated().any():
        raise ValueError(
            "Merged dataset contains duplicated cell IDs"
        )

    # ------------------------------------------------------------------
    # Construct state axis
    #
    # Higher = relatively more terminal-exhaustion-associated
    # and less progenitor-associated.
    # ------------------------------------------------------------------

    merged["progenitor_tpex_z"] = zscore(
        merged["reference_progenitor_tpex"]
    )

    merged["terminal_exhaustion_z"] = zscore(
        merged["reference_terminal_exhaustion"]
    )

    merged["terminal_minus_progenitor_axis"] = (
        merged["terminal_exhaustion_z"]
        - merged["progenitor_tpex_z"]
    )

    merged["state_axis_z"] = zscore(
        merged["terminal_minus_progenitor_axis"]
    )

    # ------------------------------------------------------------------
    # Cell-level output
    # ------------------------------------------------------------------

    cell_columns = [
        "cell_id",
        "sample_id",
        "subtype",
        "celltype_subset",
        "split",
        "deep_evi_raw",
        "reference_progenitor_tpex",
        "reference_terminal_exhaustion",
        "progenitor_tpex_z",
        "terminal_exhaustion_z",
        "terminal_minus_progenitor_axis",
        "state_axis_z",
        "exhaustion_dysfunction_score",
        "tcell_identity_score",
        "cd8_cytotoxic_score",
        "treg_score",
        "tfh_score",
        "activation_effector_score",
    ]

    merged[cell_columns].to_csv(
        outdir /
        "GSE176078_10A5_cell_state_axis.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Main associations
    # ------------------------------------------------------------------

    association_rows = []

    targets = {
        "state_axis": merged["state_axis_z"],
        "terminal_exhaustion": merged[
            "terminal_exhaustion_z"
        ],
        "progenitor_tpex": merged[
            "progenitor_tpex_z"
        ],
        "direct_exhaustion_target": merged[
            "exhaustion_dysfunction_score"
        ],
    }

    for target_name, values in targets.items():

        result = safe_corr(
            merged["deep_evi_raw"],
            values,
        )

        association_rows.append(
            {
                "representation": "Deep-EVI",
                "target": target_name,
                **result,
            }
        )

    association_df = pd.DataFrame(
        association_rows
    )

    association_df.to_csv(
        outdir /
        "GSE176078_10A5_associations.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Partial associations
    # ------------------------------------------------------------------

    partial_terminal = partial_spearman(
        merged["state_axis_z"],
        merged["deep_evi_raw"],
        merged["terminal_exhaustion_z"],
    )

    partial_progenitor = partial_spearman(
        merged["state_axis_z"],
        merged["deep_evi_raw"],
        merged["progenitor_tpex_z"],
    )

    partial_df = pd.DataFrame(
        [
            {
                "analysis": (
                    "Deep-EVI vs state axis "
                    "controlling terminal score"
                ),
                **partial_terminal,
            },
            {
                "analysis": (
                    "Deep-EVI vs state axis "
                    "controlling progenitor score"
                ),
                **partial_progenitor,
            },
        ]
    )

    partial_df.to_csv(
        outdir /
        "GSE176078_10A5_partial_associations.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Descriptive state bins
    # ------------------------------------------------------------------

    q33, q67 = np.quantile(
        merged["state_axis_z"].dropna(),
        [1 / 3, 2 / 3],
    )

    merged["state_bin"] = pd.cut(
        merged["state_axis_z"],
        bins=[
            -np.inf,
            q33,
            q67,
            np.inf,
        ],
        labels=[
            "progenitor_enriched",
            "transitional",
            "terminal_enriched",
        ],
        include_lowest=True,
    )

    state_bin_summary = (
        merged
        .groupby(
            "state_bin",
            observed=False,
        )
        .agg(
            cells=("cell_id", "size"),
            samples=("sample_id", "nunique"),
            deep_evi_mean=(
                "deep_evi_raw",
                "mean",
            ),
            deep_evi_median=(
                "deep_evi_raw",
                "median",
            ),
            deep_evi_sd=(
                "deep_evi_raw",
                "std",
            ),
            progenitor_mean=(
                "progenitor_tpex_z",
                "mean",
            ),
            terminal_mean=(
                "terminal_exhaustion_z",
                "mean",
            ),
            state_axis_mean=(
                "state_axis_z",
                "mean",
            ),
        )
        .reset_index()
    )

    state_bin_summary.to_csv(
        outdir /
        "GSE176078_10A5_state_bin_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # T-cell subset summaries
    # ------------------------------------------------------------------

    subset_summary = (
        merged
        .groupby(
            "celltype_subset",
            dropna=False,
        )
        .agg(
            cells=("cell_id", "size"),
            samples=("sample_id", "nunique"),
            deep_evi_mean=(
                "deep_evi_raw",
                "mean",
            ),
            deep_evi_median=(
                "deep_evi_raw",
                "median",
            ),
            state_axis_mean=(
                "state_axis_z",
                "mean",
            ),
            progenitor_mean=(
                "progenitor_tpex_z",
                "mean",
            ),
            terminal_mean=(
                "terminal_exhaustion_z",
                "mean",
            ),
        )
        .reset_index()
    )

    subset_summary["state_axis_rank"] = (
        subset_summary[
            "state_axis_mean"
        ].rank(
            method="average"
        )
    )

    subset_summary = subset_summary.sort_values(
        "state_axis_mean"
    )

    subset_summary.to_csv(
        outdir /
        "GSE176078_10A5_subset_state_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Named states
    # ------------------------------------------------------------------

    named_states = [
        "T_cells_c2_CD4+_T-regs_FOXP3",
        "T_cells_c3_CD4+_Tfh_CXCL13",
        "T_cells_c8_CD8+_LAG3",
        "T_cells_c7_CD8+_IFNG",
        "T_cells_c0_CD4+_CCR7",
        "T_cells_c1_CD4+_IL7R",
        "T_cells_c4_CD8+_ZFP36",
        "T_cells_c5_CD8+_GZMK",
        "T_cells_c11_MKI67",
    ]

    named = subset_summary[
        subset_summary[
            "celltype_subset"
        ].isin(named_states)
    ].copy()

    named.to_csv(
        outdir /
        "GSE176078_10A5_named_state_summary.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Sample-level descriptive summary
    # ------------------------------------------------------------------

    sample_summary = (
        merged
        .groupby(
            [
                "sample_id",
                "subtype",
            ],
            dropna=False,
        )
        .agg(
            cells=("cell_id", "size"),
            deep_evi_mean=(
                "deep_evi_raw",
                "mean",
            ),
            deep_evi_median=(
                "deep_evi_raw",
                "median",
            ),
            deep_evi_sd=(
                "deep_evi_raw",
                "std",
            ),
            state_axis_mean=(
                "state_axis_z",
                "mean",
            ),
            state_axis_median=(
                "state_axis_z",
                "median",
            ),
            progenitor_mean=(
                "progenitor_tpex_z",
                "mean",
            ),
            terminal_mean=(
                "terminal_exhaustion_z",
                "mean",
            ),
        )
        .reset_index()
    )

    sample_summary.to_csv(
        outdir /
        "GSE176078_10A5_sample_summary.csv",
        index=False,
    )

    sample_corr = safe_corr(
        sample_summary["deep_evi_mean"],
        sample_summary["state_axis_mean"],
    )

    pd.DataFrame(
        [
            {
                "unit": "sample",
                **sample_corr,
            }
        ]
    ).to_csv(
        outdir /
        "GSE176078_10A5_sample_level_association.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Subset monotonicity
    # ------------------------------------------------------------------

    subset_order = subset_summary.sort_values(
        "state_axis_mean"
    )

    monotonic_rows = []

    for name, values in {
        "Deep-EVI": subset_order[
            "deep_evi_mean"
        ],
        "terminal_exhaustion_reference": subset_order[
            "terminal_mean"
        ],
        "progenitor_TPEX_reference": subset_order[
            "progenitor_mean"
        ],
    }.items():

        result = safe_corr(
            subset_order[
                "state_axis_mean"
            ],
            values,
        )

        monotonic_rows.append(
            {
                "quantity": name,
                **result,
            }
        )

    monotonic_df = pd.DataFrame(
        monotonic_rows
    )

    monotonic_df.to_csv(
        outdir /
        "GSE176078_10A5_subset_monotonicity.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Final audit
    # ------------------------------------------------------------------

    audit = {
        "cohort": "GSE176078",
        "step": "10A5_heldout_state_axis_analysis",
        "status": "complete",
        "all_deep_evi_cells": int(
            len(deep_all)
        ),
        "test_cells": int(
            len(deep)
        ),
        "samples": int(
            merged["sample_id"].nunique()
        ),
        "subtypes": sorted(
            merged["subtype"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "state_axis_definition": (
            "z(terminal_exhaustion_reference) - "
            "z(progenitor_TPEX_reference)"
        ),
        "higher_state_axis": (
            "relatively more terminal-exhaustion-associated "
            "and less progenitor-associated"
        ),
        "test_split_only": True,
        "deep_evi_retrained": False,
        "test_used_for_training": False,
        "test_used_for_model_selection": False,
        "reference_scores_refit": False,
        "cutoff_optimization": False,
        "rna_velocity": False,
        "independent_biological_validation": False,
        "quantile_bins_descriptive_only": True,
        "sample_level_inference_descriptive_only": True,
        "partial_association_descriptive_only": True,
        "main_associations": association_df.to_dict(
            orient="records"
        ),
        "partial_associations": partial_df.to_dict(
            orient="records"
        ),
        "sample_level_association": sample_corr,
        "state_bin_cutpoints": {
            "q33": float(q33),
            "q67": float(q67),
        },
    }

    with open(
        outdir /
        "GSE176078_10A5_report.json",
        "w",
    ) as handle:
        json.dump(
            audit,
            handle,
            indent=2,
            allow_nan=True,
        )

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------

    print()
    print("=" * 90)
    print("10A-5 TEST-SET ALIGNMENT")
    print("=" * 90)

    print(
        f"All Deep-EVI cells: {len(deep_all)}"
    )

    print(
        f"Frozen test cells:  {len(deep)}"
    )

    print(
        f"Reference cells:    {len(refs)}"
    )

    print(
        f"Merged cells:       {len(merged)}"
    )

    print(
        "Alignment:           PASS"
    )

    print()
    print("=" * 90)
    print("10A-5 MAIN ASSOCIATIONS")
    print("=" * 90)

    print(
        association_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("10A-5 PARTIAL ASSOCIATIONS")
    print("=" * 90)

    print(
        partial_df.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("10A-5 STATE-BIN SUMMARY")
    print("=" * 90)

    print(
        state_bin_summary.to_string(
            index=False
        )
    )

    print()
    print("=" * 90)
    print("STATUS: PASS — 10A-5 state-axis analysis completed.")
    print("=" * 90)


if __name__ == "__main__":
    main()