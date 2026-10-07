#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def load_json(path):
    with open(path, "r") as handle:
        return json.load(handle)


def find_row(df, target):
    rows = df[df["target"] == target]
    if rows.empty:
        raise ValueError(f"Target not found: {target}")
    return rows.iloc[0]


def find_partial(df, text):
    rows = df[df["analysis"].str.contains(text, regex=False)]
    if rows.empty:
        raise ValueError(f"Partial association not found: {text}")
    return rows.iloc[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-expression", required=True)
    p.add_argument("--reference-scores", required=True)
    p.add_argument("--reference-metadata", required=True)
    p.add_argument("--benchmark-report", required=True)
    p.add_argument("--state-axis-associations", required=True)
    p.add_argument("--state-axis-partial", required=True)
    p.add_argument("--overlap-associations", required=True)
    p.add_argument("--overlap-partial", required=True)
    p.add_argument("--overlap-report", required=True)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    benchmark = load_json(args.benchmark_report)
    overlap_report = load_json(args.overlap_report)

    test_expression = Path(args.test_expression)
    reference_scores = Path(args.reference_scores)
    reference_metadata = Path(args.reference_metadata)

    state_assoc = pd.read_csv(args.state_axis_associations)
    state_partial = pd.read_csv(args.state_axis_partial)
    overlap_assoc = pd.read_csv(args.overlap_associations)
    overlap_partial = pd.read_csv(args.overlap_partial)

    state_axis = find_row(state_assoc, "state_axis")
    terminal = find_row(state_assoc, "terminal_exhaustion")
    progenitor = find_row(state_assoc, "progenitor_tpex")
    direct_target = find_row(state_assoc, "direct_exhaustion_target")

    original_partial_terminal = find_partial(
        state_partial, "controlling terminal score"
    )
    original_partial_progenitor = find_partial(
        state_partial, "controlling progenitor score"
    )

    controlled_axis = find_row(
        overlap_assoc, "controlled_state_axis"
    )
    controlled_terminal = find_row(
        overlap_assoc, "controlled_terminal_reference"
    )
    controlled_progenitor = find_row(
        overlap_assoc, "controlled_progenitor_tpex"
    )

    controlled_partial_terminal = find_partial(
        overlap_partial, "controlled_state_axis_control_terminal_reference"
    )
    controlled_partial_progenitor = find_partial(
        overlap_partial, "controlled_state_axis_control_progenitor"
    )

    manifest = {
        "cohort": "GSE176078",
        "stage": "10A_frozen_heldout_deep_evi_benchmark",
        "status": "complete",

        "test_population": {
            "cells": int(benchmark["test_cells"]),
            "samples": int(benchmark["test_samples"]),
            "source": test_expression.name
        },

        "reference_benchmark": {
            "version": "v2",
            "reference_signatures": int(benchmark["reference_signatures"]),
            "scores": reference_scores.name,
            "metadata": reference_metadata.name,
            "benchmark_report": Path(args.benchmark_report).name
        },

        "state_axis_validation": {
            "step": "10A-5",
            "production_validated": True,
            "test_cells": int(state_axis["n"]),
            "state_axis_spearman": float(state_axis["spearman_rho"]),
            "state_axis_pearson": float(state_axis["pearson_r"]),
            "terminal_exhaustion_spearman": float(
                terminal["spearman_rho"]
            ),
            "progenitor_tpex_spearman": float(
                progenitor["spearman_rho"]
            ),
            "direct_exhaustion_target_spearman": float(
                direct_target["spearman_rho"]
            ),
            "partial_state_axis_control_terminal": float(
                original_partial_terminal["partial_spearman_rho"]
            ),
            "partial_state_axis_control_progenitor": float(
                original_partial_progenitor["partial_spearman_rho"]
            ),
            "rna_velocity": False,
            "independent_biological_validation": False
        },

        "overlap_controlled_sensitivity": {
            "step": "10A-5B",
            "production_validated": True,
            "test_cells": int(controlled_axis["n"]),
            "deep_evi_construction_genes_removed": 33,
            "controlled_state_axis_spearman": float(
                controlled_axis["spearman_rho"]
            ),
            "controlled_state_axis_pearson": float(
                controlled_axis["pearson_r"]
            ),
            "controlled_terminal_reference_spearman": float(
                controlled_terminal["spearman_rho"]
            ),
            "controlled_progenitor_tpex_spearman": float(
                controlled_progenitor["spearman_rho"]
            ),
            "controlled_partial_state_axis_control_terminal": float(
                controlled_partial_terminal["partial_spearman_rho"]
            ),
            "controlled_partial_state_axis_control_progenitor": float(
                controlled_partial_progenitor["partial_spearman_rho"]
            ),
            "original_terminal_reference_controlled_score":
                "not_estimable",
            "rna_velocity": False
        },

        "safeguards": {
            "deep_evi_retrained": False,
            "test_used_for_training": False,
            "test_used_for_model_selection": False,
            "reference_scores_refit": False,
            "cutoff_optimization": False,
            "tcga_used": False,
            "independent_biological_validation": False,
            "rna_velocity": False,
            "sample_level_inference": "descriptive_only"
        },

        "interpretation": {
            "primary":
                "Deep-EVI generalizes to the held-out GSE176078 "
                "T-cell population and shows a strong association "
                "with a predefined progenitor-to-terminal exhaustion "
                "state axis.",
            "overlap_sensitivity":
                "After removal of the 33 genes used in Deep-EVI "
                "construction, the state-axis association remains "
                "positive but is substantially attenuated, indicating "
                "residual robustness while demonstrating that direct "
                "molecular overlap contributed to the magnitude of "
                "the original association.",
            "scope":
                "Held-out construct/state-axis validation and "
                "overlap-controlled sensitivity analysis; not "
                "independent biological validation and not RNA velocity."
        }
    }

    with open(
        outdir / "GSE176078_10A_frozen_manifest.json", "w"
    ) as handle:
        json.dump(manifest, handle, indent=2)

    summary = f"""GSE176078 10A FROZEN BENCHMARK
================================

Test cells: {manifest["test_population"]["cells"]}
Test samples: {manifest["test_population"]["samples"]}

10A-5 original state-axis Spearman:
{manifest["state_axis_validation"]["state_axis_spearman"]:.12f}

10A-5 terminal-exhaustion Spearman:
{manifest["state_axis_validation"]["terminal_exhaustion_spearman"]:.12f}

10A-5 progenitor/TPEX Spearman:
{manifest["state_axis_validation"]["progenitor_tpex_spearman"]:.12f}

10A-5B overlap-controlled state-axis Spearman:
{manifest["overlap_controlled_sensitivity"]["controlled_state_axis_spearman"]:.12f}

Deep-EVI construction genes removed:
33

RNA velocity:
FALSE

Independent biological validation:
FALSE

Deep-EVI retrained:
FALSE

Test used for training:
FALSE

Test used for model selection:
FALSE

STATUS:
FROZEN
"""

    with open(
        outdir / "GSE176078_10A_freeze_summary.txt", "w"
    ) as handle:
        handle.write(summary)


if __name__ == "__main__":
    main()