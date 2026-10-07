#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score


PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]

NON_TARGET_PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
]

KEY_SUBSETS = {
    "Treg_FOXP3": ["T_cells_c2_CD4+_T-regs_FOXP3"],
    "Tfh_CXCL13": ["T_cells_c3_CD4+_Tfh_CXCL13"],
    "CD8_LAG3": ["T_cells_c8_CD8+_LAG3"],
    "CD8_IFNG": ["T_cells_c7_CD8+_IFNG"],
    "CD4_CCR7": ["T_cells_c0_CD4+_CCR7"],
    "CD4_IL7R": ["T_cells_c1_CD4+_IL7R"],
    "CD8_ZFP36": ["T_cells_c4_CD8+_ZFP36"],
    "CD8_GZMK": ["T_cells_c5_CD8+_GZMK"],
    "Cycling_MKI67": ["T_cells_c11_MKI67"],
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 08E independent biological robustness evaluation."
    )
    p.add_argument("--scores", required=True)
    p.add_argument("--state_scores", required=True)
    p.add_argument("--split_manifest", required=True)
    p.add_argument("--landscape", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--cohort", default="GSE176078")
    return p.parse_args()


def load_table(path, name):
    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(f"{name} is empty: {path}")

    if "cell_id" not in df.columns:
        raise RuntimeError(
            f"{name} missing required column: cell_id"
        )

    if df["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{name} contains duplicated cell_id values: {path}"
        )

    return df


def finite_pair(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    return x[mask], y[mask], int(mask.sum())


def correlation(x, y):
    x, y, n = finite_pair(x, y)

    if n < 3:
        return {
            "n": n,
            "spearman": np.nan,
            "pearson": np.nan,
        }

    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "n": n,
            "spearman": np.nan,
            "pearson": np.nan,
        }

    sr = spearmanr(x, y)
    pr = pearsonr(x, y)

    return {
        "n": n,
        "spearman": float(sr.statistic),
        "pearson": float(pr.statistic),
    }


def safe_auc(y_true, score):
    y_true = np.asarray(y_true).astype(int)
    score = np.asarray(score, dtype=float)

    mask = np.isfinite(score)

    y_true = y_true[mask]
    score = score[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, score))


def validate_manifest(manifest, sample_ids):

    required = {"sample_id", "split"}

    missing = required - set(manifest.columns)

    if missing:
        raise RuntimeError(
            f"Split manifest missing columns: {sorted(missing)}"
        )

    allowed = {"train", "validation", "test"}

    bad = (
        set(manifest["split"].dropna().astype(str).unique())
        - allowed
    )

    if bad:
        raise RuntimeError(
            f"Unexpected split values: {sorted(bad)}"
        )

    sample_map = (
        manifest[["sample_id", "split"]]
        .drop_duplicates()
    )

    ambiguous = (
        sample_map
        .groupby("sample_id")["split"]
        .nunique()
    )

    ambiguous = ambiguous[ambiguous > 1]

    if len(ambiguous):
        raise RuntimeError(
            "Samples assigned to multiple splits: "
            + ", ".join(
                ambiguous.index.astype(str).tolist()
            )
        )

    missing_samples = sorted(
        set(sample_ids)
        - set(sample_map["sample_id"])
    )

    if missing_samples:
        raise RuntimeError(
            "Samples missing from split manifest: "
            + ", ".join(map(str, missing_samples))
        )

    return sample_map


def main():

    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=" * 90)
    print("08E DEEP-EVI INDEPENDENT BIOLOGICAL ROBUSTNESS VALIDATION")
    print("=" * 90)

    # ---------------------------------------------------------------
    # Load all required artifacts
    # ---------------------------------------------------------------

    scores = load_table(
        args.scores,
        "Deep-EVI score table"
    )

    state = load_table(
        args.state_scores,
        "T-cell state score table"
    )

    landscape = load_table(
        args.landscape,
        "T-cell state landscape"
    )

    manifest = pd.read_csv(
        args.split_manifest
    )

    print(f"Deep-EVI score rows: {len(scores):,}")
    print(f"State-score rows: {len(state):,}")
    print(f"Landscape rows: {len(landscape):,}")

    required_scores = {
        "sample_id",
        "subtype",
        "celltype_subset",
        "split",
        "deep_evi_raw",
        "deep_evi_exhaustion_prediction",
        "deep_evi_activation_prediction",
    }

    missing_scores = (
        required_scores - set(scores.columns)
    )

    if missing_scores:
        raise RuntimeError(
            "Deep-EVI score table missing columns: "
            + ", ".join(sorted(missing_scores))
        )

    missing_programs = (
        set(PROGRAMS) - set(state.columns)
    )

    if missing_programs:
        raise RuntimeError(
            "State score table missing program columns: "
            + ", ".join(sorted(missing_programs))
        )

    # ---------------------------------------------------------------
    # Verify landscape population
    # ---------------------------------------------------------------

    landscape_ids = set(
        landscape["cell_id"]
    )

    score_ids = set(
        scores["cell_id"]
    )

    missing_landscape = score_ids - landscape_ids

    if missing_landscape:
        raise RuntimeError(
            f"{len(missing_landscape)} Deep-EVI cells are absent "
            "from Step 7C landscape."
        )

    print(
        "Landscape alignment: PASS; "
        f"{len(score_ids):,} Deep-EVI cells represented."
    )

    # ---------------------------------------------------------------
    # Exact state-score alignment
    # ---------------------------------------------------------------

    state_ids = set(
        state["cell_id"]
    )

    missing_state = score_ids - state_ids

    if missing_state:
        raise RuntimeError(
            f"{len(missing_state)} Deep-EVI cells are absent "
            "from Step 7B state-score table."
        )

    state_index = state.set_index(
        "cell_id"
    )

    ordered_state = (
        state_index
        .loc[scores["cell_id"].tolist()]
        .reset_index()
    )

    if not np.array_equal(
        ordered_state["cell_id"].to_numpy(),
        scores["cell_id"].to_numpy()
    ):
        raise RuntimeError(
            "Cell alignment failed."
        )

    df = scores.copy()

    for col in PROGRAMS:
        df[col] = ordered_state[col].to_numpy()

    # ---------------------------------------------------------------
    # Split validation
    # ---------------------------------------------------------------

    sample_map = validate_manifest(
        manifest,
        df["sample_id"].unique()
    )

    split_lookup = dict(
        zip(
            sample_map["sample_id"],
            sample_map["split"]
        )
    )

    df["validated_split"] = (
        df["sample_id"].map(split_lookup)
    )

    if df["validated_split"].isna().any():
        raise RuntimeError(
            "Some cells lack a validated train/validation/test split."
        )

    mismatch = (
        df["split"].astype(str)
        != df["validated_split"].astype(str)
    )

    if mismatch.any():
        raise RuntimeError(
            f"Split mismatch in {int(mismatch.sum())} cells."
        )

    print(
        "Split validation: PASS; "
        f"train={(df.validated_split == 'train').sum():,}, "
        f"validation={(df.validated_split == 'validation').sum():,}, "
        f"test={(df.validated_split == 'test').sum():,}"
    )

    test = df[
        df["validated_split"] == "test"
    ].copy()

    # ---------------------------------------------------------------
    # Global program associations
    # ---------------------------------------------------------------

    rows = []

    for target in PROGRAMS:

        result = correlation(
            df["deep_evi_raw"],
            df[target]
        )

        rows.append({
            "representation": "deep_evi_raw",
            "target": target,
            **result,
        })

    pd.DataFrame(rows).to_csv(
        outdir
        / "GSE176078_08E_global_program_associations.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Split-specific associations
    # ---------------------------------------------------------------

    rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        sub = df[
            df["validated_split"] == split_name
        ]

        for target in PROGRAMS:

            result = correlation(
                sub["deep_evi_raw"],
                sub[target]
            )

            rows.append({
                "split": split_name,
                "target": target,
                **result,
            })

    pd.DataFrame(rows).to_csv(
        outdir
        / "GSE176078_08E_split_program_associations.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Test non-exhaustion program associations
    # ---------------------------------------------------------------

    rows = []

    for target in NON_TARGET_PROGRAMS:

        result = correlation(
            test["deep_evi_raw"],
            test[target]
        )

        rows.append({
            "split": "test",
            "target": target,
            **result,
            "evaluation_role":
                "non_exhaustion_biological_association",
        })

    pd.DataFrame(rows).to_csv(
        outdir
        / "GSE176078_08E_test_non_target_associations.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Annotation-level AUC
    # ---------------------------------------------------------------

    rows = []

    for subset in sorted(
        df["celltype_subset"]
        .dropna()
        .astype(str)
        .unique()
    ):

        y_all = (
            df["celltype_subset"]
            .astype(str)
            .eq(subset)
            .astype(int)
        )

        test_mask = (
            df["validated_split"] == "test"
        )

        y_test = y_all[test_mask]

        rows.append({
            "celltype_subset": subset,
            "n_all": len(df),
            "n_subset_all": int(y_all.sum()),
            "n_test": int(test_mask.sum()),
            "n_subset_test": int(y_test.sum()),
            "auc_all": safe_auc(
                y_all,
                df["deep_evi_raw"]
            ),
            "auc_test": safe_auc(
                y_test,
                df.loc[
                    test_mask,
                    "deep_evi_raw"
                ]
            ),
        })

    pd.DataFrame(rows).to_csv(
        outdir
        / "GSE176078_08E_subset_auc.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Prespecified key biological states
    # ---------------------------------------------------------------

    rows = []

    for state_name, labels in KEY_SUBSETS.items():

        y = (
            df["celltype_subset"]
            .astype(str)
            .isin(labels)
            .astype(int)
        )

        test_mask = (
            df["validated_split"] == "test"
        )

        test_y = y[test_mask]

        positive = df.loc[
            y.astype(bool),
            "deep_evi_raw"
        ]

        negative = df.loc[
            ~y.astype(bool),
            "deep_evi_raw"
        ]

        test_positive = df.loc[
            test_mask & y.astype(bool),
            "deep_evi_raw"
        ]

        test_negative = df.loc[
            test_mask & ~y.astype(bool),
            "deep_evi_raw"
        ]

        rows.append({
            "state": state_name,
            "labels": ";".join(labels),
            "n_all_positive": int(y.sum()),
            "n_test_positive": int(test_y.sum()),
            "auc_all": safe_auc(
                y,
                df["deep_evi_raw"]
            ),
            "auc_test": safe_auc(
                test_y,
                df.loc[
                    test_mask,
                    "deep_evi_raw"
                ]
            ),
            "mean_evi_all_positive":
                float(positive.mean())
                if len(positive) else np.nan,
            "mean_evi_all_negative":
                float(negative.mean())
                if len(negative) else np.nan,
            "mean_evi_test_positive":
                float(test_positive.mean())
                if len(test_positive) else np.nan,
            "mean_evi_test_negative":
                float(test_negative.mean())
                if len(test_negative) else np.nan,
        })

    pd.DataFrame(rows).to_csv(
        outdir
        / "GSE176078_08E_key_state_validation.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Annotation summaries
    # ---------------------------------------------------------------

    (
        df.groupby(
            ["celltype_subset", "validated_split"],
            observed=True
        )
        .agg(
            cells=("cell_id", "size"),
            mean_deep_evi=("deep_evi_raw", "mean"),
            median_deep_evi=("deep_evi_raw", "median"),
            std_deep_evi=("deep_evi_raw", "std"),
            mean_exhaustion=(
                "exhaustion_dysfunction_score",
                "mean"
            ),
            mean_activation=(
                "activation_effector_score",
                "mean"
            ),
        )
        .reset_index()
        .to_csv(
            outdir
            / "GSE176078_08E_annotation_summary.csv",
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Subtype summaries
    # ---------------------------------------------------------------

    (
        df.groupby(
            ["subtype", "validated_split"],
            observed=True
        )
        .agg(
            cells=("cell_id", "size"),
            samples=("sample_id", "nunique"),
            mean_deep_evi=("deep_evi_raw", "mean"),
            median_deep_evi=("deep_evi_raw", "median"),
            std_deep_evi=("deep_evi_raw", "std"),
            mean_exhaustion=(
                "exhaustion_dysfunction_score",
                "mean"
            ),
        )
        .reset_index()
        .to_csv(
            outdir
            / "GSE176078_08E_subtype_summary.csv",
            index=False
        )
    )

    # ---------------------------------------------------------------
    # Sample-level descriptive analysis
    # ---------------------------------------------------------------

    rows = []

    for sample_id, sub in df.groupby(
        "sample_id",
        observed=True
    ):

        exhaustion_corr = correlation(
            sub["deep_evi_raw"],
            sub["exhaustion_dysfunction_score"]
        )

        row = {
            "sample_id": sample_id,
            "subtype": str(
                sub["subtype"].iloc[0]
            ),
            "split": str(
                sub["validated_split"].iloc[0]
            ),
            "cells": len(sub),
            "mean_deep_evi": float(
                sub["deep_evi_raw"].mean()
            ),
            "median_deep_evi": float(
                sub["deep_evi_raw"].median()
            ),
            "mean_exhaustion": float(
                sub["exhaustion_dysfunction_score"].mean()
            ),
            "evi_exhaustion_spearman":
                exhaustion_corr["spearman"],
            "evi_exhaustion_pearson":
                exhaustion_corr["pearson"],
        }

        for target in NON_TARGET_PROGRAMS:

            result = correlation(
                sub["deep_evi_raw"],
                sub[target]
            )

            row[
                f"{target}_spearman"
            ] = result["spearman"]

        rows.append(row)

    sample_summary = pd.DataFrame(rows)

    sample_summary.to_csv(
        outdir
        / "GSE176078_08E_sample_level_descriptive.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Patient/sample-level test analysis
    # ---------------------------------------------------------------

    test_sample = sample_summary[
        sample_summary["split"] == "test"
    ]

    rows = []

    if len(test_sample) >= 3:

        result = correlation(
            test_sample["mean_deep_evi"],
            test_sample["mean_exhaustion"]
        )

        rows.append({
            "split": "test",
            "n_samples": len(test_sample),
            "target": "mean_exhaustion",
            **result,
            "interpretation":
                "descriptive_only_small_test_sample",
        })

    pd.DataFrame(rows).to_csv(
        outdir
        / "GSE176078_08E_patient_level_test.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Training-only EVI high-state threshold
    # ---------------------------------------------------------------

    train_evi = df.loc[
        df["validated_split"] == "train",
        "deep_evi_raw"
    ]

    threshold = float(
        train_evi.quantile(0.75)
    )

    df["evi_high_train_q75"] = (
        df["deep_evi_raw"] >= threshold
    )

    rows = []

    for split_name in [
        "train",
        "validation",
        "test",
    ]:

        sub = df[
            df["validated_split"] == split_name
        ]

        for target in PROGRAMS:

            high = sub.loc[
                sub["evi_high_train_q75"],
                target
            ]

            low = sub.loc[
                ~sub["evi_high_train_q75"],
                target
            ]

            rows.append({
                "split": split_name,
                "target": target,
                "training_q75_threshold": threshold,
                "high_cells": len(high),
                "low_cells": len(low),
                "high_mean":
                    float(high.mean())
                    if len(high) else np.nan,
                "low_mean":
                    float(low.mean())
                    if len(low) else np.nan,
                "mean_difference":
                    float(high.mean() - low.mean())
                    if len(high) and len(low)
                    else np.nan,
            })

    pd.DataFrame(rows).to_csv(
        outdir
        / "GSE176078_08E_high_low_state_comparison.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Independence / target-dependence audit
    # ---------------------------------------------------------------

    test_exhaustion = correlation(
        test["deep_evi_raw"],
        test["exhaustion_dysfunction_score"]
    )

    test_non_target = {}

    for target in NON_TARGET_PROGRAMS:

        test_non_target[target] = correlation(
            test["deep_evi_raw"],
            test[target]
        )

    independence = {
        "test_cells": int(len(test)),
        "test_samples": int(
            test["sample_id"].nunique()
        ),
        "training_target_association": {
            "target":
                "exhaustion_dysfunction_score",
            **test_exhaustion,
        },
        "test_non_target_associations":
            test_non_target,
        "training_q75_evi_threshold":
            threshold,
        "model_retrained": False,
        "test_used_for_model_selection": False,
        "interpretation": (
            "The exhaustion association is expected because "
            "exhaustion_dysfunction_score was a training target "
            "in Step 08A. Associations with other T-cell programs "
            "and annotated states are evaluated without retraining, "
            "but are not fully independent external validation "
            "because those programs contributed to the learned "
            "representation."
        ),
    }

    with open(
        outdir
        / "GSE176078_08E_independence_audit.json",
        "w"
    ) as fh:
        json.dump(
            independence,
            fh,
            indent=2
        )

    # ---------------------------------------------------------------
    # Final report
    # ---------------------------------------------------------------

    report = {
        "cohort": args.cohort,
        "step":
            "08E_deep_evi_independent_biological_robustness",
        "status": "complete",
        "cells": int(len(df)),
        "samples": int(
            df["sample_id"].nunique()
        ),
        "train_cells": int(
            (df["validated_split"] == "train").sum()
        ),
        "validation_cells": int(
            (df["validated_split"] == "validation").sum()
        ),
        "test_cells": int(
            (df["validated_split"] == "test").sum()
        ),
        "test_samples": int(
            test["sample_id"].nunique()
        ),
        "source_artifacts": {
            "deep_evi": args.scores,
            "state_scores": args.state_scores,
            "landscape": args.landscape,
            "split_manifest": args.split_manifest,
        },
        "model_retrained": False,
        "test_used_for_model_selection": False,
        "training_target": "exhaustion_dysfunction_score",
        "rna_velocity": False,
        "external_cohort_validation": False,
        "independent_biological_validation": False,
        "purpose": (
            "Post-hoc biological robustness and construct "
            "evaluation of the already-trained Deep-EVI."
        ),
        "interpretation_policy": (
            "08E does not establish RNA velocity or external "
            "biological validation. It evaluates associations "
            "between the fixed Deep-EVI and non-exhaustion "
            "T-cell programs and curated T-cell states without "
            "retraining or test-set model selection."
        ),
    }

    with open(
        outdir
        / "GSE176078_08E_report.json",
        "w"
    ) as fh:
        json.dump(
            report,
            fh,
            indent=2
        )

    print("=" * 90)
    print("08E COMPLETE")
    print("=" * 90)
    print(f"Cells: {len(df):,}")
    print(f"Samples: {df['sample_id'].nunique()}")
    print(f"Test cells: {len(test):,}")
    print(
        f"Test samples: {test['sample_id'].nunique()}"
    )
    print("Model retrained: NO")
    print("Test-set model selection: NO")
    print("RNA velocity: NO")
    print(
        f"Training-only EVI Q75: {threshold:.6f}"
    )
    print("=" * 90)


if __name__ == "__main__":
    main()