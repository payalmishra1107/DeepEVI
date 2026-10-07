#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


def load_csv(path):
    return pd.read_csv(path)


def safe_corr(x, y, method="spearman"):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan, int(mask.sum())

    if np.nanstd(x[mask]) == 0 or np.nanstd(y[mask]) == 0:
        return np.nan, np.nan, int(mask.sum())

    if method == "spearman":
        r, p = spearmanr(x[mask], y[mask])
    else:
        r, p = pearsonr(x[mask], y[mask])

    return float(r), float(p), int(mask.sum())


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--deep-evi-scores",
        required=True
    )

    parser.add_argument(
        "--state-scores",
        required=True
    )

    parser.add_argument(
        "--reference-scores",
        required=True
    )

    parser.add_argument(
        "--reference-metadata",
        required=True
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Load
    # ------------------------------------------------------------

    deep = load_csv(args.deep_evi_scores)
    state = load_csv(args.state_scores)
    reference = load_csv(args.reference_scores)
    metadata = load_csv(args.reference_metadata)

    for name, df in {
        "deep": deep,
        "state": state,
        "reference": reference
    }.items():

        if "cell_id" not in df.columns:
            raise RuntimeError(
                f"{name} table lacks cell_id"
            )

        df["cell_id"] = df["cell_id"].astype(str)

        if df["cell_id"].duplicated().any():
            raise RuntimeError(
                f"Duplicate cell IDs in {name} table"
            )

    # ------------------------------------------------------------
    # Exact held-out Deep-EVI cells
    # ------------------------------------------------------------

    if "split" not in deep.columns:
        raise RuntimeError(
            "Deep-EVI table lacks split column."
        )

    deep_test = deep[
        deep["split"].astype(str).str.lower() == "test"
    ].copy()

    if len(deep_test) != 9864:
        raise RuntimeError(
            f"Expected 9864 Deep-EVI test cells, "
            f"found {len(deep_test)}."
        )

    test_ids = set(deep_test["cell_id"])

    state_test = state[
        state["cell_id"].isin(test_ids)
    ].copy()

    ref_test = reference[
        reference["cell_id"].isin(test_ids)
    ].copy()

    if len(state_test) != 9864:
        raise RuntimeError(
            f"Expected 9864 state-score cells, "
            f"found {len(state_test)}."
        )

    if len(ref_test) != 9864:
        raise RuntimeError(
            f"Expected 9864 reference-score cells, "
            f"found {len(ref_test)}."
        )

    # ------------------------------------------------------------
    # Merge exact population
    # ------------------------------------------------------------

    deep_columns = [
        "cell_id",
        "sample_id",
        "subtype",
        "celltype_subset",
        "deep_evi_raw"
    ]

    deep_columns = [
        c for c in deep_columns
        if c in deep_test.columns
    ]

    merged = deep_test[
        deep_columns
    ].copy()

    state_columns = [
        "cell_id",
        "tcell_identity_score",
        "cd8_cytotoxic_score",
        "treg_score",
        "tfh_score",
        "activation_effector_score",
        "exhaustion_dysfunction_score"
    ]

    missing_state = (
        set(state_columns) -
        set(state_test.columns)
    )

    if missing_state:
        raise RuntimeError(
            f"Missing state columns: {missing_state}"
        )

    merged = merged.merge(
        state_test[state_columns],
        on="cell_id",
        how="inner",
        validate="one_to_one"
    )

    ref_columns = [
        c for c in ref_test.columns
        if c.startswith("reference_")
    ]

    if not ref_columns:
        raise RuntimeError(
            "No reference signature columns found."
        )

    merged = merged.merge(
        ref_test[
            ["cell_id"] + ref_columns
        ],
        on="cell_id",
        how="inner",
        validate="one_to_one"
    )

    if len(merged) != 9864:
        raise RuntimeError(
            f"Merged benchmark has {len(merged)} cells; "
            f"expected 9864."
        )

    # ------------------------------------------------------------
    # Representations
    # ------------------------------------------------------------

    representations = {
        "Deep-EVI":
            "deep_evi_raw",

        "Direct exhaustion target":
            "exhaustion_dysfunction_score"
    }

    for col in ref_columns:
        representations[col] = col

    # ------------------------------------------------------------
    # 1. Association with original exhaustion target
    #
    # This is NOT independent validation.
    # ------------------------------------------------------------

    target = merged[
        "exhaustion_dysfunction_score"
    ].to_numpy(dtype=float)

    target_rows = []

    for name, col in representations.items():

        x = merged[col].to_numpy(dtype=float)

        rho, rho_p, n = safe_corr(
            x,
            target,
            "spearman"
        )

        r, r_p, _ = safe_corr(
            x,
            target,
            "pearson"
        )

        target_rows.append({
            "representation": name,
            "n_cells": n,
            "spearman_rho": rho,
            "spearman_pvalue": rho_p,
            "pearson_r": r,
            "pearson_pvalue": r_p,
            "independent_validation": False
        })

    target_df = pd.DataFrame(target_rows)

    # ------------------------------------------------------------
    # 2. Relationships with non-exhaustion programs
    # ------------------------------------------------------------

    biological_programs = [
        "tcell_identity_score",
        "cd8_cytotoxic_score",
        "treg_score",
        "tfh_score",
        "activation_effector_score"
    ]

    program_rows = []

    for representation, col in representations.items():

        x = merged[col].to_numpy(dtype=float)

        for program in biological_programs:

            y = merged[
                program
            ].to_numpy(dtype=float)

            rho, p, n = safe_corr(
                x,
                y,
                "spearman"
            )

            program_rows.append({
                "representation":
                    representation,
                "biological_program":
                    program,
                "n_cells": n,
                "spearman_rho": rho,
                "spearman_pvalue": p
            })

    program_df = pd.DataFrame(
        program_rows
    )

    # ------------------------------------------------------------
    # 3. Reference relationship to Deep-EVI
    # ------------------------------------------------------------

    deep_values = merged[
        "deep_evi_raw"
    ].to_numpy(dtype=float)

    pairwise_rows = []

    for col in ref_columns:

        rho, p, n = safe_corr(
            deep_values,
            merged[col].to_numpy(dtype=float),
            "spearman"
        )

        pairwise_rows.append({
            "reference_score": col,
            "n_cells": n,
            "deep_evi_spearman_rho": rho,
            "deep_evi_spearman_pvalue": p
        })

    pairwise_df = pd.DataFrame(
        pairwise_rows
    )

    # ------------------------------------------------------------
    # 4. Predefined subset discrimination
    #
    # These are annotation-concordance diagnostics,
    # not independent biological validation.
    # ------------------------------------------------------------

    subset_rows = []

    if "celltype_subset" in merged.columns:

        subset_series = (
            merged["celltype_subset"]
            .fillna("")
            .astype(str)
        )

        subset_definitions = {
            "CD8_LAG3":
                subset_series.str.contains(
                    "CD8\\+_LAG3",
                    regex=True
                ),

            "Tfh_CXCL13":
                subset_series.str.contains(
                    "Tfh_CXCL13",
                    regex=False
                ),

            "Treg_FOXP3":
                subset_series.str.contains(
                    "T-regs_FOXP3",
                    regex=False
                ),

            "CD4_CCR7":
                subset_series.str.contains(
                    "CD4\\+_CCR7",
                    regex=True
                ),

            "CD4_IL7R":
                subset_series.str.contains(
                    "CD4\\+_IL7R",
                    regex=True
                ),

            "Cycling_MKI67":
                subset_series.str.contains(
                    "MKI67",
                    regex=False
                )
        }

        for state_name, mask in subset_definitions.items():

            y = mask.astype(int).to_numpy()

            positives = int(y.sum())
            negatives = int(len(y) - positives)

            if positives == 0 or negatives == 0:
                continue

            for representation, col in representations.items():

                x = merged[
                    col
                ].to_numpy(dtype=float)

                finite = np.isfinite(x)

                if len(
                    np.unique(y[finite])
                ) < 2:
                    continue

                auc = roc_auc_score(
                    y[finite],
                    x[finite]
                )

                subset_rows.append({
                    "state": state_name,
                    "representation":
                        representation,
                    "n_cells":
                        int(finite.sum()),
                    "positive_cells":
                        int(y[finite].sum()),
                    "negative_cells":
                        int(
                            finite.sum() -
                            y[finite].sum()
                        ),
                    "auc":
                        float(auc),
                    "interpretation":
                        "annotation_concordance_only"
                })

    subset_df = pd.DataFrame(
        subset_rows
    )

    # ------------------------------------------------------------
    # 5. Sample-level summaries
    #
    # Only five held-out samples; descriptive.
    # ------------------------------------------------------------

    sample_rows = []

    if "sample_id" in merged.columns:

        for sample_id, group in merged.groupby(
            "sample_id"
        ):

            row = {
                "sample_id":
                    sample_id,
                "n_cells":
                    int(len(group))
            }

            if "subtype" in group.columns:
                subtype_values = (
                    group["subtype"]
                    .dropna()
                    .astype(str)
                    .unique()
                )

                row["subtype"] = (
                    subtype_values[0]
                    if len(subtype_values)
                    else ""
                )

            for representation, col in representations.items():

                safe_name = (
                    representation
                    .lower()
                    .replace(" ", "_")
                    .replace("-", "_")
                )

                row[
                    f"{safe_name}_mean"
                ] = float(
                    group[col].mean()
                )

                row[
                    f"{safe_name}_median"
                ] = float(
                    group[col].median()
                )

            sample_rows.append(row)

    sample_df = pd.DataFrame(
        sample_rows
    )

    # ------------------------------------------------------------
    # 6. Metadata overlap integration
    # ------------------------------------------------------------

    metadata_out = metadata.copy()

    if (
        "deep_evi_construction_overlap_count"
        not in metadata_out.columns
    ):
        raise RuntimeError(
            "Reference metadata lacks construction overlap audit."
        )

    # ------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------

    merged.to_csv(
        outdir /
        "GSE176078_10A_benchmark_cell_table.csv",
        index=False
    )

    target_df.to_csv(
        outdir /
        "GSE176078_10A_exhaustion_target_associations.csv",
        index=False
    )

    program_df.to_csv(
        outdir /
        "GSE176078_10A_program_associations.csv",
        index=False
    )

    pairwise_df.to_csv(
        outdir /
        "GSE176078_10A_deep_evi_reference_associations.csv",
        index=False
    )

    subset_df.to_csv(
        outdir /
        "GSE176078_10A_subset_concordance.csv",
        index=False
    )

    sample_df.to_csv(
        outdir /
        "GSE176078_10A_sample_summary.csv",
        index=False
    )

    metadata_out.to_csv(
        outdir /
        "GSE176078_10A_reference_metadata.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    report = {
        "cohort":
            "GSE176078",

        "step":
            "10A_held_out_reference_benchmark",

        "status":
            "complete",

        "test_cells":
            int(len(merged)),

        "test_samples":
            int(
                merged["sample_id"].nunique()
            )
            if "sample_id" in merged.columns
            else None,

        "reference_signatures":
            int(len(ref_columns)),

        "deep_evi_retrained":
            False,

        "reference_scores_refit":
            False,

        "test_used_for_training":
            False,

        "test_used_for_model_selection":
            False,

        "rna_velocity":
            False,

        "tcga_used":
            False,

        "independent_biological_validation":
            False,

        "construction_gene_overlap_reported":
            True,

        "cell_level_statistics":
            True,

        "sample_level_statistics":
            "descriptive_only_due_to_five_test_samples",

        "scientific_interpretation":
            (
                "10A benchmarks the frozen Deep-EVI score "
                "against predefined reference T-cell state "
                "gene-set scores on the exact held-out 08A "
                "test population. Correlation with the original "
                "exhaustion target is a construct benchmark and "
                "not independent validation because exhaustion "
                "was an auxiliary Deep-EVI training target. "
                "Reference gene overlap with Deep-EVI "
                "construction is explicitly retained."
            )
    }

    with open(
        outdir /
        "GSE176078_10A_report.json",
        "w"
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2
        )

    # ------------------------------------------------------------
    # Console
    # ------------------------------------------------------------

    print("=" * 80)
    print("10A-3 DEEP-EVI HELD-OUT REFERENCE BENCHMARK")
    print("=" * 80)

    print(
        f"Test cells:           {len(merged)}"
    )

    if "sample_id" in merged.columns:
        print(
            f"Test samples:         "
            f"{merged['sample_id'].nunique()}"
        )

    print(
        f"Reference signatures: {len(ref_columns)}"
    )

    print("\nExhaustion-target associations:")
    print(
        target_df[
            [
                "representation",
                "spearman_rho",
                "pearson_r"
            ]
        ].to_string(index=False)
    )

    if not subset_df.empty:

        print("\nSubset concordance:")
        print(
            subset_df[
                [
                    "state",
                    "representation",
                    "positive_cells",
                    "auc"
                ]
            ].to_string(index=False)
        )

    print("\nScientific safeguards:")
    print("  Deep-EVI retrained: FALSE")
    print("  Test used for training: FALSE")
    print("  Test model selection: FALSE")
    print("  Reference overlap reported: TRUE")
    print("  Independent validation: FALSE")
    print("  RNA velocity: FALSE")

    print(
        "\nSTATUS: PASS — 10A held-out benchmark completed."
    )


if __name__ == "__main__":
    main()