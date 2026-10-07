#!/usr/bin/env python

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, kruskal
from sklearn.linear_model import LinearRegression


PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]

REQUIRED_COLUMNS = [
    "cell_id",
    "sample_id",
    "subtype",
    "celltype_subset",
    "split",
    *PROGRAMS,
    "deep_evi_raw",
    "deep_evi_exhaustion_prediction",
    "deep_evi_activation_prediction",
]


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    value = spearmanr(x[mask], y[mask]).statistic
    return float(value) if np.isfinite(value) else np.nan


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    value = pearsonr(x[mask], y[mask]).statistic
    return float(value) if np.isfinite(value) else np.nan


def summarize_group(df, group_col, value_col="deep_evi_raw"):
    rows = []

    for group, g in df.groupby(group_col, dropna=False):
        values = pd.to_numeric(g[value_col], errors="coerce").dropna()

        if len(values) == 0:
            continue

        rows.append({
            group_col: group,
            "n_cells": int(len(values)),
            "mean": float(values.mean()),
            "median": float(values.median()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "q05": float(values.quantile(0.05)),
            "q25": float(values.quantile(0.25)),
            "q75": float(values.quantile(0.75)),
            "q95": float(values.quantile(0.95)),
            "min": float(values.min()),
            "max": float(values.max()),
        })

    return pd.DataFrame(rows)


def summarize_sample(df, high_threshold):
    rows = []

    for sample, g in df.groupby("sample_id", dropna=False):
        evi = pd.to_numeric(g["deep_evi_raw"], errors="coerce")
        valid = np.isfinite(evi)

        if valid.sum() == 0:
            continue

        evi = evi[valid]

        row = {
            "sample_id": sample,
            "subtype": g["subtype"].iloc[0],
            "n_t_cells": int(len(evi)),
            "deep_evi_mean": float(evi.mean()),
            "deep_evi_median": float(evi.median()),
            "deep_evi_std": float(evi.std(ddof=1)) if len(evi) > 1 else 0.0,
            "deep_evi_q25": float(evi.quantile(0.25)),
            "deep_evi_q75": float(evi.quantile(0.75)),
            "high_evi_threshold": float(high_threshold),
            "high_evi_cells": int((evi >= high_threshold).sum()),
            "high_evi_fraction": float((evi >= high_threshold).mean()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def program_correlations(df):
    rows = []

    for program in PROGRAMS:
        rows.append({
            "program": program,
            "spearman_deep_evi": safe_spearman(
                df["deep_evi_raw"], df[program]
            ),
            "pearson_deep_evi": safe_pearson(
                df["deep_evi_raw"], df[program]
            ),
            "spearman_n": int(
                np.isfinite(df["deep_evi_raw"]).astype(int).mul(
                    np.isfinite(df[program]).astype(int)
                ).sum()
            ),
        })

    return pd.DataFrame(rows)


def partial_correlation_evi_activation(df):
    """
    Quantify whether Deep-EVI retains association with activation
    after linearly accounting for the exhaustion target.

    This is an audit, not proof of biological independence.
    """
    cols = [
        "deep_evi_raw",
        "exhaustion_dysfunction_score",
        "activation_effector_score",
    ]

    x = df[cols].apply(pd.to_numeric, errors="coerce").dropna()

    if len(x) < 10:
        return {
            "n": int(len(x)),
            "partial_pearson_evi_activation_given_exhaustion": np.nan,
        }

    exhaustion = x["exhaustion_dysfunction_score"].to_numpy().reshape(-1, 1)

    evi_model = LinearRegression().fit(
        exhaustion,
        x["deep_evi_raw"].to_numpy()
    )

    activation_model = LinearRegression().fit(
        exhaustion,
        x["activation_effector_score"].to_numpy()
    )

    evi_residual = (
        x["deep_evi_raw"].to_numpy()
        - evi_model.predict(exhaustion)
    )

    activation_residual = (
        x["activation_effector_score"].to_numpy()
        - activation_model.predict(exhaustion)
    )

    return {
        "n": int(len(x)),
        "partial_pearson_evi_activation_given_exhaustion": safe_pearson(
            evi_residual,
            activation_residual,
        ),
    }


def subtype_statistics(df):
    rows = []

    subtypes = [
        x for x in ["ER+", "HER2+", "TNBC"]
        if x in set(df["subtype"].dropna().unique())
    ]

    for subtype in subtypes:
        values = pd.to_numeric(
            df.loc[df["subtype"] == subtype, "deep_evi_raw"],
            errors="coerce"
        ).dropna()

        rows.append({
            "subtype": subtype,
            "n_cells": int(len(values)),
            "mean_deep_evi": float(values.mean()),
            "median_deep_evi": float(values.median()),
            "std_deep_evi": float(values.std(ddof=1))
            if len(values) > 1 else 0.0,
            "high_evi_fraction": np.nan,
        })

    return pd.DataFrame(rows)


def subtype_kruskal(df):
    groups = []

    for subtype in ["ER+", "HER2+", "TNBC"]:
        values = pd.to_numeric(
            df.loc[df["subtype"] == subtype, "deep_evi_raw"],
            errors="coerce"
        ).dropna()

        if len(values) > 0:
            groups.append((subtype, values))

    if len(groups) < 2:
        return {
            "test": "Kruskal-Wallis",
            "groups": [g[0] for g in groups],
            "statistic": np.nan,
            "pvalue": np.nan,
        }

    statistic, pvalue = kruskal(*(g[1].to_numpy() for g in groups))

    return {
        "test": "Kruskal-Wallis",
        "groups": [g[0] for g in groups],
        "statistic": float(statistic),
        "pvalue": float(pvalue),
        "note": (
            "Cell-level p-value is descriptive and does not account for "
            "patient-level clustering. Patient-level analyses should be "
            "used for confirmatory inference."
        ),
    }


def test_set_program_correlations(df):
    test = df[df["split"] == "test"].copy()

    rows = []

    for program in PROGRAMS:
        rows.append({
            "program": program,
            "split": "test",
            "n_cells": int(len(test)),
            "spearman_deep_evi": safe_spearman(
                test["deep_evi_raw"],
                test[program],
            ),
            "pearson_deep_evi": safe_pearson(
                test["deep_evi_raw"],
                test[program],
            ),
        })

    return pd.DataFrame(rows)


def annotation_summary(df, high_threshold):
    work = df.copy()
    work["high_deep_evi"] = (
        work["deep_evi_raw"] >= high_threshold
    )

    rows = []

    for annotation, g in work.groupby(
        "celltype_subset",
        dropna=False
    ):
        values = pd.to_numeric(
            g["deep_evi_raw"],
            errors="coerce"
        ).dropna()

        if len(values) == 0:
            continue

        rows.append({
            "celltype_subset": annotation,
            "n_cells": int(len(values)),
            "deep_evi_mean": float(values.mean()),
            "deep_evi_median": float(values.median()),
            "deep_evi_std": float(values.std(ddof=1))
            if len(values) > 1 else 0.0,
            "deep_evi_q25": float(values.quantile(0.25)),
            "deep_evi_q75": float(values.quantile(0.75)),
            "high_evi_cells": int(g["high_deep_evi"].sum()),
            "high_evi_fraction": float(g["high_deep_evi"].mean()),
        })

    return pd.DataFrame(rows).sort_values(
        "deep_evi_mean",
        ascending=False
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--high-evi-quantile", type=float, default=0.75)

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)

    missing = [
        c for c in REQUIRED_COLUMNS
        if c not in scores.columns
    ]

    if missing:
        raise RuntimeError(
            "Missing required columns: " + ", ".join(missing)
        )

    if scores["cell_id"].duplicated().any():
        raise RuntimeError("Duplicate cell_id values detected.")

    if scores["cell_id"].isna().any():
        raise RuntimeError("Missing cell_id values detected.")

    if len(scores) != 35214:
        raise RuntimeError(
            f"Expected 35214 T cells, observed {len(scores)}."
        )

    if scores["subtype"].isna().any():
        raise RuntimeError("Missing subtype values detected.")

    if scores["split"].isna().any():
        raise RuntimeError("Missing split assignments detected.")

    evi = pd.to_numeric(
        scores["deep_evi_raw"],
        errors="coerce"
    )

    if not np.isfinite(evi).all():
        raise RuntimeError("Deep-EVI contains NaN or infinite values.")

    # Training-derived threshold to avoid using test distribution
    train_evi = evi[scores["split"] == "train"]

    if len(train_evi) == 0:
        raise RuntimeError("No training cells found.")

    high_threshold = float(
        train_evi.quantile(args.high_evi_quantile)
    )

    scores["high_deep_evi"] = (
        scores["deep_evi_raw"] >= high_threshold
    )

    # ------------------------------------------------------------------
    # 1. Overall summary
    # ------------------------------------------------------------------

    overall = {
        "cells": int(len(scores)),
        "samples": int(scores["sample_id"].nunique()),
        "subtypes": sorted(
            scores["subtype"].dropna().unique().tolist()
        ),
        "deep_evi_mean": float(evi.mean()),
        "deep_evi_median": float(evi.median()),
        "deep_evi_std": float(evi.std(ddof=1)),
        "deep_evi_q05": float(evi.quantile(0.05)),
        "deep_evi_q25": float(evi.quantile(0.25)),
        "deep_evi_q75": float(evi.quantile(0.75)),
        "deep_evi_q95": float(evi.quantile(0.95)),
        "high_evi_threshold_training_q75": high_threshold,
        "high_evi_cells": int(scores["high_deep_evi"].sum()),
        "high_evi_fraction": float(scores["high_deep_evi"].mean()),
    }

    with open(
        outdir / "GSE176078_deep_evi_overall_summary.json",
        "w"
    ) as handle:
        json.dump(overall, handle, indent=2)

    # ------------------------------------------------------------------
    # 2. Program correlations
    # ------------------------------------------------------------------

    program_corr = program_correlations(scores)
    program_corr.to_csv(
        outdir / "GSE176078_deep_evi_program_correlations.csv",
        index=False,
    )

    test_program_corr = test_set_program_correlations(scores)
    test_program_corr.to_csv(
        outdir / "GSE176078_deep_evi_test_program_correlations.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 3. Partial correlation audit
    # ------------------------------------------------------------------

    partial = partial_correlation_evi_activation(scores)

    with open(
        outdir / "GSE176078_deep_evi_independence_audit.json",
        "w"
    ) as handle:
        json.dump(
            {
                "deep_evi_exhaustion_relationship": {
                    "all_cells_spearman": safe_spearman(
                        scores["deep_evi_raw"],
                        scores["exhaustion_dysfunction_score"],
                    ),
                    "all_cells_pearson": safe_pearson(
                        scores["deep_evi_raw"],
                        scores["exhaustion_dysfunction_score"],
                    ),
                    "test_cells_spearman": safe_spearman(
                        scores.loc[
                            scores["split"] == "test",
                            "deep_evi_raw"
                        ],
                        scores.loc[
                            scores["split"] == "test",
                            "exhaustion_dysfunction_score"
                        ],
                    ),
                },
                "partial_activation_audit": partial,
                "interpretation": (
                    "Deep-EVI was trained with the exhaustion-dysfunction "
                    "program as an auxiliary target. Therefore a strong "
                    "correlation with that target is expected and is not "
                    "independent biological validation. This audit "
                    "quantifies the relationship with the remaining "
                    "programs and activation after accounting for the "
                    "exhaustion-associated target."
                ),
            },
            handle,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 4. T-cell subset characterization
    # ------------------------------------------------------------------

    annotation = annotation_summary(
        scores,
        high_threshold,
    )

    annotation.to_csv(
        outdir / "GSE176078_deep_evi_by_tcell_subset.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 5. Breast cancer subtype characterization
    # ------------------------------------------------------------------

    subtype = summarize_group(
        scores,
        "subtype",
        "deep_evi_raw",
    )

    subtype["high_evi_threshold"] = high_threshold

    subtype_counts = (
        scores.groupby("subtype")
        .agg(
            n_cells=("cell_id", "size"),
            high_evi_cells=("high_deep_evi", "sum"),
        )
        .reset_index()
    )

    subtype = subtype.merge(
        subtype_counts,
        on="subtype",
        how="left",
        suffixes=("", "_counts"),
    )

    subtype["high_evi_fraction"] = (
        subtype["high_evi_cells"]
        / subtype["n_cells"]
    )

    subtype.to_csv(
        outdir / "GSE176078_deep_evi_by_subtype.csv",
        index=False,
    )

    with open(
        outdir / "GSE176078_deep_evi_subtype_statistics.json",
        "w"
    ) as handle:
        json.dump(
            subtype_kruskal(scores),
            handle,
            indent=2,
        )

    # ------------------------------------------------------------------
    # 6. Patient/sample heterogeneity
    # ------------------------------------------------------------------

    sample = summarize_sample(
        scores,
        high_threshold,
    )

    sample.to_csv(
        outdir / "GSE176078_deep_evi_by_sample.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 7. Split-specific summaries
    # ------------------------------------------------------------------

    split_summary = summarize_group(
        scores,
        "split",
        "deep_evi_raw",
    )

    split_summary.to_csv(
        outdir / "GSE176078_deep_evi_by_split.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 8. Program state distributions by T-cell subset
    # ------------------------------------------------------------------

    program_by_annotation = (
        scores.groupby("celltype_subset")[PROGRAMS]
        .agg(["mean", "median"])
        .reset_index()
    )

    program_by_annotation.to_csv(
        outdir / "GSE176078_tcell_programs_by_subset_for_deep_evi.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 9. High-EVI cells
    # ------------------------------------------------------------------

    high_evi = scores[
        scores["high_deep_evi"]
    ][[
        "cell_id",
        "sample_id",
        "subtype",
        "celltype_subset",
        "split",
        "deep_evi_raw",
        "deep_evi_exhaustion_prediction",
        "deep_evi_activation_prediction",
        "exhaustion_dysfunction_score",
        "activation_effector_score",
    ]].copy()

    high_evi = high_evi.sort_values(
        "deep_evi_raw",
        ascending=False
    )

    high_evi.to_csv(
        outdir / "GSE176078_high_deep_evi_cells.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 10. Final machine-readable report
    # ------------------------------------------------------------------

    report = {
        "cohort": "GSE176078",
        "step": "08B_deep_evi_characterization",
        "status": "complete",
        "cells": int(len(scores)),
        "samples": int(scores["sample_id"].nunique()),
        "subtypes": sorted(
            scores["subtype"].unique().tolist()
        ),
        "deep_evi_definition": (
            "Graph-learned continuous T-cell state index from Step 08A."
        ),
        "not_rna_velocity": True,
        "training_target_dependence": {
            "exhaustion_dysfunction_score_is_training_target": True,
            "strong_correlation_with_target_is_expected": True,
            "independent_biological_validation": False,
        },
        "high_evi_threshold": {
            "quantile": float(args.high_evi_quantile),
            "fit_on": "training_cells_only",
            "threshold": high_threshold,
        },
        "outputs": [
            "GSE176078_deep_evi_overall_summary.json",
            "GSE176078_deep_evi_program_correlations.csv",
            "GSE176078_deep_evi_test_program_correlations.csv",
            "GSE176078_deep_evi_independence_audit.json",
            "GSE176078_deep_evi_by_tcell_subset.csv",
            "GSE176078_deep_evi_by_subtype.csv",
            "GSE176078_deep_evi_subtype_statistics.json",
            "GSE176078_deep_evi_by_sample.csv",
            "GSE176078_deep_evi_by_split.csv",
            "GSE176078_tcell_programs_by_subset_for_deep_evi.csv",
            "GSE176078_high_deep_evi_cells.csv",
        ],
    }

    with open(
        outdir / "GSE176078_step8b_report.json",
        "w"
    ) as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()