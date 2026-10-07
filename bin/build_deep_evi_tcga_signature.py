#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]

# These are the exact Step 7B program definitions.
PROGRAM_GENES = {
    "tcell_identity_score": [
        "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2"
    ],
    "cd8_cytotoxic_score": [
        "CD8A", "CD8B", "NKG7", "GNLY",
        "GZMB", "GZMH", "PRF1", "CTSW"
    ],
    "treg_score": [
        "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IL7R"
    ],
    "tfh_score": [
        "CXCL13", "PDCD1", "ICOS", "CXCR5"
    ],
    "activation_effector_score": [
        "IFNG", "TNF", "IL2", "CD69",
        "HLA-DRA", "HLA-DRB1"
    ],
    "exhaustion_dysfunction_score": [
        "PDCD1", "LAG3", "TIGIT", "HAVCR2",
        "CTLA4", "TOX", "TOX2", "ENTPD1", "CXCL13"
    ],
}


def parse_args():

    p = argparse.ArgumentParser(
        description=(
            "Freeze a TCGA-compatible molecular surrogate "
            "of the GSE176078 Deep-EVI."
        )
    )

    p.add_argument(
        "--scores",
        required=True
    )

    p.add_argument(
        "--state_scores",
        required=True
    )

    p.add_argument(
        "--split_manifest",
        required=True
    )

    p.add_argument(
        "--outdir",
        required=True
    )

    return p.parse_args()


def load_csv(path, name):

    df = pd.read_csv(path)

    if df.empty:
        raise RuntimeError(
            f"{name} is empty: {path}"
        )

    if "cell_id" not in df.columns:
        raise RuntimeError(
            f"{name} lacks cell_id: {path}"
        )

    if df["cell_id"].duplicated().any():
        raise RuntimeError(
            f"{name} contains duplicated cell_id values."
        )

    return df


def main():

    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 90)
    print("09A DEEP-EVI TCGA MOLECULAR SURROGATE FREEZING")
    print("=" * 90)

    scores = load_csv(
        args.scores,
        "Deep-EVI score table"
    )

    state = load_csv(
        args.state_scores,
        "T-cell state score table"
    )

    manifest = pd.read_csv(
        args.split_manifest
    )

    required_scores = {
        "cell_id",
        "sample_id",
        "deep_evi_raw",
        "split",
    }

    missing = (
        required_scores
        - set(scores.columns)
    )

    if missing:
        raise RuntimeError(
            "Deep-EVI score table missing: "
            + ", ".join(sorted(missing))
        )

    missing_programs = (
        set(PROGRAMS)
        - set(state.columns)
    )

    if missing_programs:
        raise RuntimeError(
            "State-score table missing: "
            + ", ".join(sorted(missing_programs))
        )

    # ---------------------------------------------------------------
    # Validate sample-level split
    # ---------------------------------------------------------------

    if not {
        "sample_id",
        "split"
    }.issubset(manifest.columns):

        raise RuntimeError(
            "Split manifest must contain sample_id and split."
        )

    sample_map = (
        manifest[
            ["sample_id", "split"]
        ]
        .drop_duplicates()
    )

    ambiguous = (
        sample_map
        .groupby("sample_id")["split"]
        .nunique()
    )

    ambiguous = ambiguous[
        ambiguous > 1
    ]

    if len(ambiguous):

        raise RuntimeError(
            "Samples assigned to multiple splits: "
            + ", ".join(
                ambiguous.index.astype(str)
            )
        )

    split_lookup = dict(
        zip(
            sample_map["sample_id"],
            sample_map["split"]
        )
    )

    scores["validated_split"] = (
        scores["sample_id"]
        .map(split_lookup)
    )

    if scores["validated_split"].isna().any():

        raise RuntimeError(
            "Some Deep-EVI cells are absent "
            "from the split manifest."
        )

    if (
        scores["split"].astype(str)
        != scores["validated_split"].astype(str)
    ).any():

        raise RuntimeError(
            "Deep-EVI split column disagrees "
            "with the sample-level split manifest."
        )

    # ---------------------------------------------------------------
    # Exact cell alignment
    # ---------------------------------------------------------------

    state_index = state.set_index(
        "cell_id"
    )

    missing_state = (
        set(scores["cell_id"])
        - set(state_index.index)
    )

    if missing_state:

        raise RuntimeError(
            f"{len(missing_state)} Deep-EVI cells "
            "are missing from state scores."
        )

    state_ordered = (
        state_index
        .loc[scores["cell_id"]]
        .reset_index()
    )

    for program in PROGRAMS:

        scores[program] = (
            state_ordered[program]
            .to_numpy()
        )

    # ---------------------------------------------------------------
    # Training cells ONLY
    #
    # This is critical:
    # TCGA signature coefficients are frozen exclusively using
    # training cells. Validation and test cells never determine
    # the molecular signature.
    # ---------------------------------------------------------------

    train = scores[
        scores["validated_split"] == "train"
    ].copy()

    validation = scores[
        scores["validated_split"] == "validation"
    ].copy()

    test = scores[
        scores["validated_split"] == "test"
    ].copy()

    print(
        f"Training cells: {len(train):,}"
    )

    print(
        f"Validation cells: {len(validation):,}"
    )

    print(
        f"Test cells: {len(test):,}"
    )

    print(
        f"Training samples: "
        f"{train['sample_id'].nunique()}"
    )

    # ---------------------------------------------------------------
    # Standardize program inputs using training cells only
    # ---------------------------------------------------------------

    scaler = StandardScaler()

    X_train = scaler.fit_transform(
        train[PROGRAMS]
    )

    X_validation = scaler.transform(
        validation[PROGRAMS]
    )

    X_test = scaler.transform(
        test[PROGRAMS]
    )

    y_train = (
        train["deep_evi_raw"]
        .to_numpy(dtype=float)
    )

    # ---------------------------------------------------------------
    # Fit regularized linear bridge
    #
    # This is deliberately a simple, transparent bridge.
    # It converts the six frozen biological programs into a
    # TCGA-compatible scalar representation.
    #
    # It is NOT the original GNN.
    # ---------------------------------------------------------------

    model = Ridge(
        alpha=1.0,
        fit_intercept=True
    )

    model.fit(
        X_train,
        y_train
    )

    train_pred = model.predict(
        X_train
    )

    validation_pred = model.predict(
        X_validation
    )

    test_pred = model.predict(
        X_test
    )

    # ---------------------------------------------------------------
    # Evaluate bridge performance
    # ---------------------------------------------------------------

    def corr(a, b):

        from scipy.stats import spearmanr, pearsonr

        sr = spearmanr(a, b)
        pr = pearsonr(a, b)

        return {
            "spearman": float(
                sr.statistic
            ),
            "pearson": float(
                pr.statistic
            ),
        }

    bridge_metrics = {
        "train": corr(
            train_pred,
            y_train
        ),
        "validation": corr(
            validation_pred,
            validation[
                "deep_evi_raw"
            ].to_numpy(float)
        ),
        "test": corr(
            test_pred,
            test[
                "deep_evi_raw"
            ].to_numpy(float)
        ),
    }

    # ---------------------------------------------------------------
    # Program coefficients
    # ---------------------------------------------------------------

    coefficient_table = pd.DataFrame({
        "program": PROGRAMS,
        "coefficient_standardized":
            model.coef_,
        "training_scaler_mean":
            scaler.mean_,
        "training_scaler_std":
            scaler.scale_,
    })

    coefficient_table.to_csv(
        outdir
        / "GSE176078_09A_program_coefficients.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Create a gene-level molecular signature.
    #
    # If a gene occurs in multiple programs, its weights are summed.
    # The program coefficients are divided by the number of genes
    # in each program so that large programs do not automatically
    # receive greater weight.
    # ---------------------------------------------------------------

    gene_weights = {}

    for program, coefficient in zip(
        PROGRAMS,
        model.coef_
    ):

        genes = PROGRAM_GENES[
            program
        ]

        per_gene = (
            float(coefficient)
            / float(len(genes))
        )

        for gene in genes:

            gene_weights[gene] = (
                gene_weights.get(gene, 0.0)
                + per_gene
            )

    gene_rows = []

    for gene, weight in sorted(
        gene_weights.items()
    ):

        contributing = [
            program
            for program in PROGRAMS
            if gene in PROGRAM_GENES[program]
        ]

        gene_rows.append({
            "gene_symbol": gene,
            "gene_weight":
                float(weight),
            "programs":
                ";".join(contributing),
            "n_programs":
                len(contributing),
        })

    gene_table = pd.DataFrame(
        gene_rows
    )

    gene_table.to_csv(
        outdir
        / "GSE176078_09A_gene_signature.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Save frozen model parameters
    # ---------------------------------------------------------------

    frozen_model = {
        "cohort": "GSE176078",
        "source_step": "08A",
        "bridge_step": "09A",
        "representation": (
            "six Step-7B T-cell expression programs"
        ),
        "target": "deep_evi_raw",
        "model": "Ridge",
        "alpha": 1.0,
        "programs": PROGRAMS,
        "training_only": True,
        "training_cells": len(train),
        "training_samples":
            int(train["sample_id"].nunique()),
        "training_scaler_mean":
            scaler.mean_.tolist(),
        "training_scaler_std":
            scaler.scale_.tolist(),
        "coefficients":
            model.coef_.tolist(),
        "intercept":
            float(model.intercept_),
        "bridge_metrics":
            bridge_metrics,
        "gene_signature_rule": (
            "Program coefficient divided by number of "
            "genes in that program; overlapping gene "
            "weights are summed."
        ),
        "important_definition": (
            "The resulting TCGA score is a frozen "
            "TCGA-compatible molecular surrogate of "
            "cell-level Deep-EVI. It is not the original "
            "single-cell GNN output."
        ),
    }

    with open(
        outdir
        / "GSE176078_09A_frozen_signature.json",
        "w"
    ) as fh:

        json.dump(
            frozen_model,
            fh,
            indent=2
        )

    # ---------------------------------------------------------------
    # Bridge predictions
    # ---------------------------------------------------------------

    prediction_rows = []

    for frame, split_name, predictions in [
        (
            train,
            "train",
            train_pred
        ),
        (
            validation,
            "validation",
            validation_pred
        ),
        (
            test,
            "test",
            test_pred
        ),
    ]:

        tmp = pd.DataFrame({
            "cell_id":
                frame["cell_id"].to_numpy(),
            "sample_id":
                frame["sample_id"].to_numpy(),
            "split":
                split_name,
            "deep_evi_original":
                frame[
                    "deep_evi_raw"
                ].to_numpy(),
            "deep_evi_tcga_surrogate":
                predictions,
        })

        prediction_rows.append(
            tmp
        )

    pd.concat(
        prediction_rows,
        ignore_index=True
    ).to_csv(
        outdir
        / "GSE176078_09A_bridge_predictions.csv",
        index=False
    )

    # ---------------------------------------------------------------
    # Report
    # ---------------------------------------------------------------

    report = {
        "cohort": "GSE176078",
        "step":
            "09A_freeze_tcga_compatible_deep_evi_signature",
        "status": "complete",
        "training_cells": len(train),
        "validation_cells": len(validation),
        "test_cells": len(test),
        "training_samples":
            int(train["sample_id"].nunique()),
        "validation_samples":
            int(validation["sample_id"].nunique()),
        "test_samples":
            int(test["sample_id"].nunique()),
        "program_count": len(PROGRAMS),
        "gene_count": len(gene_table),
        "model": "Ridge",
        "alpha": 1.0,
        "fit_using_training_cells_only": True,
        "validation_used_for_fitting": False,
        "test_used_for_fitting": False,
        "original_gnn_reused_directly": False,
        "tcga_score_definition":
            "frozen molecular surrogate",
        "rna_velocity": False,
        "external_validation": False,
        "bridge_metrics":
            bridge_metrics,
        "scientific_warning": (
            "The TCGA-compatible score is not identical "
            "to the cell-level Deep-EVI GNN output. "
            "It is a frozen molecular surrogate trained "
            "only on the GSE176078 training partition."
        ),
    }

    with open(
        outdir
        / "GSE176078_09A_report.json",
        "w"
    ) as fh:

        json.dump(
            report,
            fh,
            indent=2
        )

    print("\n" + "=" * 90)
    print("09A COMPLETE")
    print("=" * 90)

    print(
        f"Frozen genes: {len(gene_table)}"
    )

    print(
        "Bridge performance:"
    )

    for split_name, metrics in bridge_metrics.items():

        print(
            f"  {split_name}: "
            f"Spearman={metrics['spearman']:.4f}, "
            f"Pearson={metrics['pearson']:.4f}"
        )

    print(
        "\nTCGA score is a FROZEN MOLECULAR SURROGATE "
        "of cell-level Deep-EVI."
    )

    print(
        "Fit using training cells only: YES"
    )

    print("=" * 90)


if __name__ == "__main__":
    main()