#!/usr/bin/env python3

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# General utilities
# =============================================================================

def require_columns(df, columns, name):
    missing = [
        column
        for column in columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


def zscore_columns(X, mean=None, std=None):
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if mean is None:
        mean = X.mean(
            axis=0,
            keepdims=True,
        )

    if std is None:
        std = X.std(
            axis=0,
            keepdims=True,
        )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    X_scaled = (
        X - mean
    ) / std

    return (
        X_scaled.astype(np.float32),
        mean.astype(np.float32),
        std.astype(np.float32),
    )


def apply_zscore(X, mean, std):
    X = np.asarray(
        X,
        dtype=np.float32,
    )

    std = np.where(
        std < 1e-8,
        1.0,
        std,
    )

    return (
        (X - mean) / std
    ).astype(np.float32)


def safe_corr(x, y):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    y = np.asarray(
        y,
        dtype=np.float64,
    )

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    if mask.sum() < 3:
        return float("nan")

    x = x[mask]
    y = y[mask]

    if (
        np.std(x) < 1e-12
        or np.std(y) < 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(x, y)[0, 1]
    )


def safe_rmse(y_true, y_pred):
    return float(
        math.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def safe_mae(y_true, y_pred):
    return float(
        mean_absolute_error(
            y_true,
            y_pred,
        )
    )


def safe_r2(y_true, y_pred):
    try:
        return float(
            r2_score(
                y_true,
                y_pred,
            )
        )
    except Exception:
        return float("nan")


# =============================================================================
# Sample-level splitting
# =============================================================================

def make_sample_split(
    metadata,
    train_fraction,
    validation_fraction,
    seed,
):
    """
    Split complete samples rather than individual cells.

    This prevents cells from the same sample/patient appearing
    in more than one model split.
    """

    sample_table = (
        metadata[
            [
                "sample_id",
                "subtype",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    samples = (
        sample_table[
            "sample_id"
        ]
        .astype(str)
        .to_numpy()
    )

    subtypes = (
        sample_table[
            "subtype"
        ]
        .astype(str)
        .to_numpy()
    )

    if len(samples) < 6:
        raise ValueError(
            "At least 6 samples are required "
            "for train/validation/test splitting."
        )

    test_fraction = (
        1.0
        - train_fraction
        - validation_fraction
    )

    if test_fraction <= 0:
        raise ValueError(
            "train_fraction + validation_fraction "
            "must be less than 1."
        )

    try:
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=(
                1.0 - train_fraction
            ),
            random_state=seed,
            stratify=subtypes,
        )

    except ValueError:
        train_samples, temp_samples = train_test_split(
            samples,
            test_size=(
                1.0 - train_fraction
            ),
            random_state=seed,
            shuffle=True,
        )

    temp_fraction = (
        validation_fraction
        / (
            validation_fraction
            + test_fraction
        )
    )

    subtype_map = {
        str(sample): str(subtype)
        for sample, subtype
        in zip(samples, subtypes)
    }

    temp_subtypes = np.array(
        [
            subtype_map[
                str(sample)
            ]
            for sample in temp_samples
        ]
    )

    try:
        validation_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(
                1.0 - temp_fraction
            ),
            random_state=seed,
            stratify=temp_subtypes,
        )

    except ValueError:
        validation_samples, test_samples = train_test_split(
            temp_samples,
            test_size=(
                1.0 - temp_fraction
            ),
            random_state=seed,
            shuffle=True,
        )

    train_set = set(
        map(str, train_samples)
    )

    validation_set = set(
        map(str, validation_samples)
    )

    test_set = set(
        map(str, test_samples)
    )

    if train_set & validation_set:
        raise RuntimeError(
            "Sample leakage between train and validation."
        )

    if train_set & test_set:
        raise RuntimeError(
            "Sample leakage between train and test."
        )

    if validation_set & test_set:
        raise RuntimeError(
            "Sample leakage between validation and test."
        )

    split = pd.Series(
        "train",
        index=metadata.index,
        dtype="object",
    )

    split[
        metadata[
            "sample_id"
        ]
        .astype(str)
        .isin(validation_set)
    ] = "validation"

    split[
        metadata[
            "sample_id"
        ]
        .astype(str)
        .isin(test_set)
    ] = "test"

    return (
        split.to_numpy(),
        sorted(train_set),
        sorted(validation_set),
        sorted(test_set),
    )


# =============================================================================
# Graph construction
# =============================================================================

def build_local_graph(
    edges,
    cell_ids,
):
    """
    Build a graph using only edges whose source and target cells
    are both inside the current split.

    This prevents cross-split graph leakage.
    """

    cell_ids = [
        str(x)
        for x in cell_ids
    ]

    local_index = {
        cell_id: i
        for i, cell_id
        in enumerate(cell_ids)
    }

    source = (
        edges[
            "source_cell"
        ]
        .astype(str)
    )

    target = (
        edges[
            "target_cell"
        ]
        .astype(str)
    )

    mask = (
        source.isin(local_index)
        &
        target.isin(local_index)
    )

    internal_edges = (
        edges.loc[
            mask
        ]
        .copy()
    )

    if len(internal_edges) == 0:
        raise ValueError(
            "No internal graph edges remain "
            "after split-specific filtering."
        )

    src = np.array(
        [
            local_index[x]
            for x in internal_edges[
                "source_cell"
            ]
        ],
        dtype=np.int64,
    )

    dst = np.array(
        [
            local_index[x]
            for x in internal_edges[
                "target_cell"
            ]
        ],
        dtype=np.int64,
    )

    distances = (
        internal_edges[
            "distance"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    finite = np.isfinite(
        distances
    )

    src = src[finite]
    dst = dst[finite]
    distances = distances[finite]

    if len(distances) == 0:
        raise ValueError(
            "No finite graph distances remain."
        )

    positive = distances[
        distances > 0
    ]

    if len(positive) > 0:
        scale = float(
            np.median(
                positive
            )
        )
    else:
        scale = 1.0

    if scale <= 1e-12:
        scale = 1.0

    weights = np.exp(
        -distances / scale
    ).astype(
        np.float32
    )

    n = len(cell_ids)

    src_all = np.concatenate(
        [
            src,
            np.arange(n),
        ]
    )

    dst_all = np.concatenate(
        [
            dst,
            np.arange(n),
        ]
    )

    weight_all = np.concatenate(
        [
            weights,
            np.ones(
                n,
                dtype=np.float32,
            ),
        ]
    )

    adjacency = sp.coo_matrix(
        (
            weight_all,
            (
                src_all,
                dst_all,
            ),
        ),
        shape=(
            n,
            n,
        ),
        dtype=np.float32,
    ).tocsr()

    row_sum = np.asarray(
        adjacency.sum(
            axis=1
        )
    ).reshape(-1)

    row_sum[
        row_sum <= 1e-12
    ] = 1.0

    adjacency = (
        sp.diags(
            1.0 / row_sum
        )
        @ adjacency
    ).tocsr()

    return (
        adjacency,
        internal_edges,
    )


def scipy_to_torch_sparse(
    matrix,
    device,
):
    matrix = matrix.tocoo()

    indices = torch.tensor(
        np.vstack(
            [
                matrix.row,
                matrix.col,
            ]
        ),
        dtype=torch.long,
        device=device,
    )

    values = torch.tensor(
        matrix.data,
        dtype=torch.float32,
        device=device,
    )

    return torch.sparse_coo_tensor(
        indices,
        values,
        size=matrix.shape,
        device=device,
    ).coalesce()


# =============================================================================
# Graph neural network
# =============================================================================

class GraphConv(nn.Module):

    def __init__(
        self,
        in_dim,
        out_dim,
    ):
        super().__init__()

        self.linear = nn.Linear(
            in_dim,
            out_dim,
        )

    def forward(
        self,
        x,
        adjacency,
    ):
        propagated = torch.sparse.mm(
            adjacency,
            x,
        )

        return self.linear(
            propagated
        )


class DeepEVIModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        output_dim,
        dropout,
    ):
        super().__init__()

        self.conv1 = GraphConv(
            input_dim,
            hidden_dim,
        )

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.conv2 = GraphConv(
            hidden_dim,
            latent_dim,
        )

        self.norm2 = nn.LayerNorm(
            latent_dim
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.program_decoder = nn.Sequential(
            nn.Linear(
                latent_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                output_dim,
            ),
        )

        self.exhaustion_head = nn.Linear(
            latent_dim,
            1,
        )

        self.activation_head = nn.Linear(
            latent_dim,
            1,
        )

    def encode(
        self,
        x,
        adjacency,
    ):

        h = self.conv1(
            x,
            adjacency,
        )

        h = self.norm1(
            h
        )

        h = F.gelu(
            h
        )

        h = self.dropout(
            h
        )

        z = self.conv2(
            h,
            adjacency,
        )

        z = self.norm2(
            z
        )

        z = F.gelu(
            z
        )

        z = self.dropout(
            z
        )

        return z

    def forward(
        self,
        x,
        adjacency,
    ):

        z = self.encode(
            x,
            adjacency,
        )

        program_prediction = (
            self.program_decoder(
                z
            )
        )

        exhaustion_prediction = (
            self.exhaustion_head(
                z
            )
            .squeeze(-1)
        )

        activation_prediction = (
            self.activation_head(
                z
            )
            .squeeze(-1)
        )

        return (
            z,
            program_prediction,
            exhaustion_prediction,
            activation_prediction,
        )


# =============================================================================
# Graph smoothness
# =============================================================================

def graph_smoothness_loss(
    z,
    adjacency,
):
    coalesced = (
        adjacency
        .coalesce()
    )

    indices = (
        coalesced.indices()
    )

    values = (
        coalesced.values()
    )

    src = indices[0]
    dst = indices[1]

    valid = (
        src != dst
    )

    if valid.sum() == 0:
        return torch.tensor(
            0.0,
            device=z.device,
        )

    src = src[valid]
    dst = dst[valid]
    values = values[valid]

    differences = (
        z[src]
        - z[dst]
    )

    squared = (
        differences
        * differences
    ).sum(
        dim=1
    )

    weighted = (
        values
        * squared
    )

    denominator = (
        values.sum()
        + 1e-8
    )

    return (
        weighted.sum()
        / denominator
    )


# =============================================================================
# Main
# =============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--landscape",
        required=True,
    )

    parser.add_argument(
        "--edges",
        required=True,
    )

    parser.add_argument(
        "--scores",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260922,
    )

    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.15,
    )

    args = parser.parse_args()

    # =========================================================================
    # Reproducibility
    # =========================================================================

    set_seed(
        args.seed
    )

    # =========================================================================
    # Output
    # =========================================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # CPU
    # =========================================================================

    device = torch.device(
        "cpu"
    )

    torch.set_num_threads(
        min(
            4,
            os.cpu_count() or 1,
        )
    )

    print("=" * 80)
    print("DEEPEVI GRAPH REPRESENTATION LEARNING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")

    # =========================================================================
    # Load Step 7B + Step 7C inputs
    # =========================================================================

    landscape = pd.read_csv(
        args.landscape
    )

    edges = pd.read_csv(
        args.edges
    )

    scores = pd.read_csv(
        args.scores
    )

    # =========================================================================
    # Validate Step 7C landscape
    # =========================================================================

    require_columns(
        landscape,
        [
            "cell_id",
            "sample_id",
            "subtype",
            "celltype_subset",
        ],
        "T-cell state landscape",
    )

    # =========================================================================
    # Validate Step 7C graph
    # =========================================================================

    require_columns(
        edges,
        [
            "source_cell",
            "target_cell",
            "distance",
            "delta_exhaustion",
            "delta_activation",        ],
        "T-cell KNN graph",
    )

    # =========================================================================
    # Validate Step 7B scores
    # =========================================================================

    require_columns(
        scores,
        [
            "cell_id",
            "sample_id",
            "subtype",
            "celltype_major",
            "celltype_subset",
        ] + PROGRAMS,
        "T-cell expression program scores",
    )

    print(
        f"Landscape rows: {len(landscape)}"
    )

    print(
        f"Graph edges: {len(edges)}"
    )

    print(
        f"Score rows: {len(scores)}"
    )

    # =========================================================================
    # Normalize IDs
    # =========================================================================

    landscape = landscape.copy()
    edges = edges.copy()
    scores = scores.copy()

    landscape["cell_id"] = (
        landscape[
            "cell_id"
        ]
        .astype(str)
        .str.strip()
    )

    edges["source_cell"] = (
        edges[
            "source_cell"
        ]
        .astype(str)
        .str.strip()
    )

    edges["target_cell"] = (
        edges[
            "target_cell"
        ]
        .astype(str)
        .str.strip()
    )

    scores["cell_id"] = (
        scores[
            "cell_id"
        ]
        .astype(str)
        .str.strip()
    )

    # =========================================================================
    # Check unique cell IDs
    # =========================================================================

    if landscape[
        "cell_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate cell IDs in Step 7C landscape."
        )

    if scores[
        "cell_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate cell IDs in Step 7B score table."
        )

    # =========================================================================
    # Step 7B ↔ Step 7C alignment
    # =========================================================================

    landscape_ids = set(
        landscape[
            "cell_id"
        ]
    )

    score_ids = set(
        scores[
            "cell_id"
        ]
    )

    missing_scores = (
        landscape_ids
        - score_ids
    )

    extra_scores = (
        score_ids
        - landscape_ids
    )

    if missing_scores:

        examples = sorted(
            missing_scores
        )[:10]

        raise ValueError(
            "Step 7B score table is missing "
            f"{len(missing_scores)} cells from Step 7C. "
            f"Examples: {examples}"
        )

    if extra_scores:

        examples = sorted(
            extra_scores
        )[:10]

        raise ValueError(
            "Step 7B score table contains "
            f"{len(extra_scores)} cells absent from Step 7C. "
            f"Examples: {examples}"
        )

    # =========================================================================
    # IMPORTANT:
    # Step 7B is the authoritative source for all six program scores.
    #
    # Step 7C already contains activation/exhaustion values because they were
    # used in the state-landscape construction. We remove ALL overlapping
    # program columns before merging so pandas does not create _x/_y columns.
    # =========================================================================

    landscape = landscape.drop(
        columns=[
            column
            for column in PROGRAMS
            if column in landscape.columns
        ],
        errors="ignore",
    )

    score_features = scores[
        [
            "cell_id"
        ] + PROGRAMS
    ].copy()

    landscape = landscape.merge(
        score_features,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    if landscape[
        PROGRAMS
    ].isna().any().any():

        missing_counts = (
            landscape[
                PROGRAMS
            ]
            .isna()
            .sum()
            .to_dict()
        )

        raise ValueError(
            "Missing program scores remain after "
            "Step 7B/7C alignment: "
            f"{missing_counts}"
        )

    print(
        "Program scores aligned: "
        f"{len(score_features)} cells"
    )

    # =========================================================================
    # Verify metadata consistency
    # =========================================================================

    metadata_check = landscape[
        [
            "cell_id",
            "sample_id",
            "subtype",
            "celltype_subset",
        ]
    ].merge(
        scores[
            [
                "cell_id",
                "sample_id",
                "subtype",
                "celltype_subset",
            ]
        ],
        on="cell_id",
        how="left",
        validate="one_to_one",
        suffixes=(
            "_landscape",
            "_scores",
        ),
    )

    sample_mismatch = (
        metadata_check[
            "sample_id_landscape"
        ].astype(str)
        !=
        metadata_check[
            "sample_id_scores"
        ].astype(str)
    )

    subtype_mismatch = (
        metadata_check[
            "subtype_landscape"
        ].astype(str)
        !=
        metadata_check[
            "subtype_scores"
        ].astype(str)
    )

    subset_mismatch = (
        metadata_check[
            "celltype_subset_landscape"
        ].astype(str)
        !=
        metadata_check[
            "celltype_subset_scores"
        ].astype(str)
    )

    if sample_mismatch.any():
        raise ValueError(
            "Sample IDs differ between Step 7B and Step 7C."
        )

    if subtype_mismatch.any():
        raise ValueError(
            "Subtype labels differ between Step 7B and Step 7C."
        )

    if subset_mismatch.any():
        raise ValueError(
            "Celltype subset labels differ between Step 7B and Step 7C."
        )

    # =========================================================================
    # Graph coverage
    # =========================================================================

    all_landscape_cells = set(
        landscape[
            "cell_id"
        ]
    )

    missing_source = (
        set(
            edges[
                "source_cell"
            ]
        )
        - all_landscape_cells
    )

    missing_target = (
        set(
            edges[
                "target_cell"
            ]
        )
        - all_landscape_cells
    )

    if missing_source:

        raise ValueError(
            "Graph contains source cells absent from Step 7C: "
            f"{len(missing_source)}"
        )

    if missing_target:

        raise ValueError(
            "Graph contains target cells absent from Step 7C: "
            f"{len(missing_target)}"
        )

    # =========================================================================
    # Curated T-cell compartment
    # =========================================================================

    major = (
        scores[
            [
                "cell_id",
                "celltype_major",
            ]
        ]
        .copy()
    )

    major[
        "celltype_major"
    ] = (
        major[
            "celltype_major"
        ]
        .astype(str)
        .str.strip()
    )

    landscape = landscape.merge(
        major,
        on="cell_id",
        how="left",
        validate="one_to_one",
    )

    tcell_mask = (
        landscape[
            "celltype_major"
        ]
        == "T-cells"
    )

    if not tcell_mask.any():

        raise ValueError(
            "No curated T cells were identified."
        )

    tcell_count = int(
        tcell_mask.sum()
    )

    print(
        f"Curated T cells: {tcell_count}"
    )

    if tcell_count != 35214:

        print(
            "WARNING: expected 35214 curated "
            f"T cells but found {tcell_count}."
        )

    # =========================================================================
    # Restrict to curated T cells
    # =========================================================================

    landscape = (
        landscape.loc[
            tcell_mask
        ]
        .reset_index(
            drop=True
        )
    )

    cell_ids = (
        landscape[
            "cell_id"
        ]
        .astype(str)
        .tolist()
    )

    cell_set = set(
        cell_ids
    )

    edges = edges[
        edges[
            "source_cell"
        ].isin(cell_set)
        &
        edges[
            "target_cell"
        ].isin(cell_set)
    ].copy()

    if len(edges) == 0:

        raise ValueError(
            "No graph edges remain after restricting "
            "to curated T cells."
        )

    print(
        f"T-cell graph edges: {len(edges)}"
    )

    # =========================================================================
    # Feature matrix
    # =========================================================================

    X_raw = landscape[
        PROGRAMS
    ].to_numpy(
        dtype=np.float32
    )

    if not np.isfinite(
        X_raw
    ).all():

        raise ValueError(
            "Non-finite values detected in program scores."
        )

    # =========================================================================
    # Sample-level train/validation/test split
    # =========================================================================

    (
        split_labels,
        train_samples,
        validation_samples,
        test_samples,
    ) = make_sample_split(
        landscape[
            [
                "sample_id",
                "subtype",
            ]
        ],
        args.train_fraction,
        args.validation_fraction,
        args.seed,
    )

    landscape[
        "_split"
    ] = split_labels

    train_mask = (
        landscape[
            "_split"
        ].to_numpy()
        == "train"
    )

    validation_mask = (
        landscape[
            "_split"
        ].to_numpy()
        == "validation"
    )

    test_mask = (
        landscape[
            "_split"
        ].to_numpy()
        == "test"
    )

    # =========================================================================
    # Train-only feature scaling
    # =========================================================================

    X_scaled_train, feature_mean, feature_std = (
        zscore_columns(
            X_raw[
                train_mask
            ]
        )
    )

    X_scaled = apply_zscore(
        X_raw,
        feature_mean,
        feature_std,
    )

    # =========================================================================
    # Targets
    # =========================================================================

    exhaustion_target_raw = (
        landscape[
            "exhaustion_dysfunction_score"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    activation_target_raw = (
        landscape[
            "activation_effector_score"
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    exhaustion_train_mean = float(
        exhaustion_target_raw[
            train_mask
        ].mean()
    )

    exhaustion_train_std = float(
        exhaustion_target_raw[
            train_mask
        ].std()
    )

    if exhaustion_train_std < 1e-8:
        exhaustion_train_std = 1.0

    activation_train_mean = float(
        activation_target_raw[
            train_mask
        ].mean()
    )

    activation_train_std = float(
        activation_target_raw[
            train_mask
        ].std()
    )

    if activation_train_std < 1e-8:
        activation_train_std = 1.0

    exhaustion_target = (
        (
            exhaustion_target_raw
            - exhaustion_train_mean
        )
        / exhaustion_train_std
    ).astype(
        np.float32
    )

    activation_target = (
        (
            activation_target_raw
            - activation_train_mean
        )
        / activation_train_std
    ).astype(
        np.float32
    )

    # =========================================================================
    # Split manifest
    # =========================================================================

    split_manifest = landscape[
        [
            "cell_id",
            "sample_id",
            "subtype",
            "celltype_subset",
            "_split",
        ]
    ].copy()

    split_manifest = split_manifest.rename(
        columns={
            "_split": "split"
        }
    )

    split_manifest.to_csv(
        output_dir
        / "GSE176078_deep_evi_split_manifest.csv",
        index=False,
    )

    print(
        f"Train samples: {len(train_samples)}"
    )

    print(
        f"Validation samples: {len(validation_samples)}"
    )

    print(
        f"Test samples: {len(test_samples)}"
    )

    print(
        f"Train cells: {int(train_mask.sum())}"
    )

    print(
        f"Validation cells: {int(validation_mask.sum())}"
    )

    print(
        f"Test cells: {int(test_mask.sum())}"
    )

    # =========================================================================
    # Build split-specific graphs
    # =========================================================================

    split_graphs = {}

    for split_name, mask in [
        ("train", train_mask),
        ("validation", validation_mask),
        ("test", test_mask),
    ]:

        split_ids = (
            landscape.loc[
                mask,
                "cell_id",
            ]
            .astype(str)
            .tolist()
        )

        adjacency, internal_edges = (
            build_local_graph(
                edges,
                split_ids,
            )
        )

        split_graphs[
            split_name
        ] = {
            "ids": split_ids,
            "adjacency": adjacency,
            "internal_edges": internal_edges,
        }

        print(
            f"{split_name} graph: "
            f"{len(internal_edges)} internal edges, "
            f"{adjacency.nnz} adjacency entries"
        )

    # =========================================================================
    # Tensor preparation
    # =========================================================================

    train_indices = np.where(
        train_mask
    )[0]

    validation_indices = np.where(
        validation_mask
    )[0]

    test_indices = np.where(
        test_mask
    )[0]

    X_train_tensor = torch.tensor(
        X_scaled[
            train_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    X_validation_tensor = torch.tensor(
        X_scaled[
            validation_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    X_test_tensor = torch.tensor(
        X_scaled[
            test_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    programs_train = X_train_tensor
    programs_validation = X_validation_tensor
    programs_test = X_test_tensor

    y_train = torch.tensor(
        exhaustion_target[
            train_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    y_validation = torch.tensor(
        exhaustion_target[
            validation_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    y_test = torch.tensor(
        exhaustion_target[
            test_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    a_train = torch.tensor(
        activation_target[
            train_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    a_validation = torch.tensor(
        activation_target[
            validation_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    a_test = torch.tensor(
        activation_target[
            test_indices
        ],
        dtype=torch.float32,
        device=device,
    )

    A_train = scipy_to_torch_sparse(
        split_graphs[
            "train"
        ][
            "adjacency"
        ],
        device,
    )

    A_validation = scipy_to_torch_sparse(
        split_graphs[
            "validation"
        ][
            "adjacency"
        ],
        device,
    )

    A_test = scipy_to_torch_sparse(
        split_graphs[
            "test"
        ][
            "adjacency"
        ],
        device,
    )

    # =========================================================================
    # Model
    # =========================================================================

    model = DeepEVIModel(
        input_dim=len(PROGRAMS),
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        output_dim=len(PROGRAMS),
        dropout=args.dropout,
    ).to(
        device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # =========================================================================
    # Training
    # =========================================================================

    history = []

    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    best_state = None

    epochs_without_improvement = 0

    print("=" * 80)
    print("TRAINING")
    print("=" * 80)

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            z_train,
            program_pred_train,
            exhaustion_pred_train,
            activation_pred_train,
        ) = model(
            X_train_tensor,
            A_train,
        )

        program_loss = F.mse_loss(
            program_pred_train,
            programs_train,
        )

        exhaustion_loss = F.mse_loss(
            exhaustion_pred_train,
            y_train,
        )

        activation_loss = F.mse_loss(
            activation_pred_train,
            a_train,
        )

        smoothness_loss = (
            graph_smoothness_loss(
                z_train,
                A_train,
            )
        )

        total_loss = (
            1.00
            * program_loss
            + 0.25
            * exhaustion_loss
            + 0.10
            * activation_loss
            + 0.05
            * smoothness_loss
        )

        total_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0,
        )

        optimizer.step()

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        model.eval()

        with torch.no_grad():

            (
                z_validation,
                program_pred_validation,
                exhaustion_pred_validation,
                activation_pred_validation,
            ) = model(
                X_validation_tensor,
                A_validation,
            )

            validation_program_loss = F.mse_loss(
                program_pred_validation,
                programs_validation,
            )

            validation_exhaustion_loss = F.mse_loss(
                exhaustion_pred_validation,
                y_validation,
            )

            validation_activation_loss = F.mse_loss(
                activation_pred_validation,
                a_validation,
            )

            validation_smoothness_loss = (
                graph_smoothness_loss(
                    z_validation,
                    A_validation,
                )
            )

            validation_total_loss = (
                1.00
                * validation_program_loss
                + 0.25
                * validation_exhaustion_loss
                + 0.10
                * validation_activation_loss
                + 0.05
                * validation_smoothness_loss
            )

        train_loss_value = float(
            total_loss.item()
        )

        validation_loss_value = float(
            validation_total_loss.item()
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss_value,
                "train_program_loss": float(
                    program_loss.item()
                ),
                "train_exhaustion_loss": float(
                    exhaustion_loss.item()
                ),
                "train_activation_loss": float(
                    activation_loss.item()
                ),
                "train_smoothness_loss": float(
                    smoothness_loss.item()
                ),
                "validation_loss": validation_loss_value,
                "validation_program_loss": float(
                    validation_program_loss.item()
                ),
                "validation_exhaustion_loss": float(
                    validation_exhaustion_loss.item()
                ),
                "validation_activation_loss": float(
                    validation_activation_loss.item()
                ),
                "validation_smoothness_loss": float(
                    validation_smoothness_loss.item()
                ),
            }
        )

        print(
            f"Epoch {epoch:04d} | "
            f"train={train_loss_value:.6f} | "
            f"validation={validation_loss_value:.6f}"
        )

        if (
            validation_loss_value
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss_value
            )

            best_epoch = epoch

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }
            epochs_without_improvement = 0

        else:

            epochs_without_improvement += 1

        if (
            epochs_without_improvement
            >= args.patience
        ):

            print(
                f"Early stopping at epoch {epoch}."
            )

            break

    if best_state is None:

        raise RuntimeError(
            "No valid model checkpoint was produced."
        )

    model.load_state_dict(
        best_state
    )

    # =========================================================================
    # Test prediction
    # =========================================================================

    model.eval()

    with torch.no_grad():

        (
            z_train,
            program_pred_train,
            exhaustion_pred_train,
            activation_pred_train,
        ) = model(
            X_train_tensor,
            A_train,
        )

        (
            z_validation,
            program_pred_validation,
            exhaustion_pred_validation,
            activation_pred_validation,
        ) = model(
            X_validation_tensor,
            A_validation,
        )

        (
            z_test,
            program_pred_test,
            exhaustion_pred_test,
            activation_pred_test,
        ) = model(
            X_test_tensor,
            A_test,
        )

    train_exhaustion_prediction = (
        exhaustion_pred_train
        .detach()
        .cpu()
        .numpy()
    )

    validation_exhaustion_prediction = (
        exhaustion_pred_validation
        .detach()
        .cpu()
        .numpy()
    )

    test_exhaustion_prediction = (
        exhaustion_pred_test
        .detach()
        .cpu()
        .numpy()
    )

    train_activation_prediction = (
        activation_pred_train
        .detach()
        .cpu()
        .numpy()
    )

    validation_activation_prediction = (
        activation_pred_validation
        .detach()
        .cpu()
        .numpy()
    )

    test_activation_prediction = (
        activation_pred_test
        .detach()
        .cpu()
        .numpy()
    )

    # =========================================================================
    # Target arrays
    # =========================================================================

    y_train_np = (
        y_train
        .detach()
        .cpu()
        .numpy()
    )

    y_validation_np = (
        y_validation
        .detach()
        .cpu()
        .numpy()
    )

    y_test_np = (
        y_test
        .detach()
        .cpu()
        .numpy()
    )

    a_train_np = (
        a_train
        .detach()
        .cpu()
        .numpy()
    )

    a_validation_np = (
        a_validation
        .detach()
        .cpu()
        .numpy()
    )

    a_test_np = (
        a_test
        .detach()
        .cpu()
        .numpy()
    )

    # =========================================================================
    # Orient learned exhaustion axis
    # =========================================================================

    train_orientation_corr = safe_corr(
        train_exhaustion_prediction,
        y_train_np,
    )

    if (
        np.isfinite(
            train_orientation_corr
        )
        and train_orientation_corr >= 0
    ):
        orientation_sign = 1.0
    else:
        orientation_sign = -1.0

    train_oriented = (
        orientation_sign
        * train_exhaustion_prediction
    )

    validation_oriented = (
        orientation_sign
        * validation_exhaustion_prediction
    )

    test_oriented = (
        orientation_sign
        * test_exhaustion_prediction
    )

    # =========================================================================
    # Training-only EVI scaling
    # =========================================================================

    evi_mean = float(
        train_oriented.mean()
    )

    evi_std = float(
        train_oriented.std()
    )

    if evi_std < 1e-8:
        evi_std = 1.0

    # =========================================================================
    # Full-cell EVI arrays
    # =========================================================================

    deep_evi = np.zeros(
        len(landscape),
        dtype=np.float32,
    )

    deep_evi[
        train_indices
    ] = (
        train_oriented
        - evi_mean
    ) / evi_std

    deep_evi[
        validation_indices
    ] = (
        validation_oriented
        - evi_mean
    ) / evi_std

    deep_evi[
        test_indices
    ] = (
        test_oriented
        - evi_mean
    ) / evi_std

    exhaustion_prediction = np.zeros(
        len(landscape),
        dtype=np.float32,
    )

    exhaustion_prediction[
        train_indices
    ] = train_exhaustion_prediction

    exhaustion_prediction[
        validation_indices
    ] = validation_exhaustion_prediction

    exhaustion_prediction[
        test_indices
    ] = test_exhaustion_prediction

    activation_prediction = np.zeros(
        len(landscape),
        dtype=np.float32,
    )

    activation_prediction[
        train_indices
    ] = train_activation_prediction

    activation_prediction[
        validation_indices
    ] = validation_activation_prediction

    activation_prediction[
        test_indices
    ] = test_activation_prediction

    # =========================================================================
    # Prediction table
    # =========================================================================

    predictions = landscape[
        [
            "cell_id",
            "sample_id",
            "subtype",
            "celltype_subset",
        ]
    ].copy()

    predictions[
        "split"
    ] = landscape[
        "_split"
    ].to_numpy()

    predictions[
        "tcell_identity_score"
    ] = landscape[
        "tcell_identity_score"
    ].to_numpy()

    predictions[
        "cd8_cytotoxic_score"
    ] = landscape[
        "cd8_cytotoxic_score"
    ].to_numpy()

    predictions[
        "treg_score"
    ] = landscape[
        "treg_score"
    ].to_numpy()

    predictions[
        "tfh_score"
    ] = landscape[
        "tfh_score"
    ].to_numpy()

    predictions[
        "activation_effector_score"
    ] = landscape[
        "activation_effector_score"
    ].to_numpy()

    predictions[
        "exhaustion_dysfunction_score"
    ] = landscape[
        "exhaustion_dysfunction_score"
    ].to_numpy()

    predictions[
        "deep_evi_raw"
    ] = deep_evi

    predictions[
        "deep_evi_exhaustion_prediction"
    ] = exhaustion_prediction

    predictions[
        "deep_evi_activation_prediction"
    ] = activation_prediction

    # =========================================================================
    # Latent representation
    # =========================================================================

    latent = np.zeros(
        (
            len(landscape),
            args.latent_dim,
        ),
        dtype=np.float32,
    )

    latent[
        train_indices
    ] = (
        z_train
        .detach()
        .cpu()
        .numpy()
    )

    latent[
        validation_indices
    ] = (
        z_validation
        .detach()
        .cpu()
        .numpy()
    )

    latent[
        test_indices
    ] = (
        z_test
        .detach()
        .cpu()
        .numpy()
    )

    latent_columns = [
        f"deep_evi_latent_{i + 1}"
        for i in range(
            args.latent_dim
        )
    ]

    latent_df = pd.DataFrame(
        latent,
        columns=latent_columns,
    )

    latent_df.insert(
        0,
        "cell_id",
        landscape[
            "cell_id"
        ].to_numpy(),
    )

    latent_df.insert(
        1,
        "sample_id",
        landscape[
            "sample_id"
        ].to_numpy(),
    )

    latent_df.insert(
        2,
        "subtype",
        landscape[
            "subtype"
        ].to_numpy(),
    )

    latent_df.insert(
        3,
        "split",
        landscape[
            "_split"
        ].to_numpy(),
    )

    # =========================================================================
    # Metrics
    # =========================================================================

    validation_exhaustion_corr = safe_corr(
        validation_oriented,
        y_validation_np,
    )

    test_exhaustion_corr = safe_corr(
        test_oriented,
        y_test_np,
    )

    validation_activation_corr = safe_corr(
        validation_activation_prediction,
        a_validation_np,
    )

    test_activation_corr = safe_corr(
        test_activation_prediction,
        a_test_np,
    )

    test_exhaustion_rmse = safe_rmse(
        y_test_np,
        test_oriented,
    )

    test_exhaustion_mae = safe_mae(
        y_test_np,
        test_oriented,
    )

    test_exhaustion_r2 = safe_r2(
        y_test_np,
        test_oriented,
    )

    # =========================================================================
    # Save cell-level results
    # =========================================================================

    predictions.to_csv(
        output_dir
        / "GSE176078_deep_evi_scores.csv",
        index=False,
    )

    latent_df.to_csv(
        output_dir
        / "GSE176078_deep_evi_latent.csv",
        index=False,
    )

    pd.DataFrame(
        history
    ).to_csv(
        output_dir
        / "GSE176078_deep_evi_training_history.csv",
        index=False,
    )

    # =========================================================================
    # Sample summary
    # =========================================================================

    sample_summary = (
        predictions
        .groupby(
            "sample_id",
            as_index=False,
        )
        .agg(
            n_cells=(
                "cell_id",
                "size",
            ),
            mean_deep_evi=(
                "deep_evi_raw",
                "mean",
            ),
            median_deep_evi=(
                "deep_evi_raw",
                "median",
            ),
            mean_exhaustion_score=(
                "exhaustion_dysfunction_score",
                "mean",
            ),
            mean_activation_score=(
                "activation_effector_score",
                "mean",
            ),
        )
    )

    sample_summary.to_csv(
        output_dir
        / "GSE176078_deep_evi_by_sample.csv",
        index=False,
    )

    # =========================================================================
    # Subtype summary
    # =========================================================================

    subtype_summary = (
        predictions
        .groupby(
            "subtype",
            as_index=False,
        )
        .agg(
            n_cells=(
                "cell_id",
                "size",
            ),
            mean_deep_evi=(
                "deep_evi_raw",
                "mean",
            ),
            median_deep_evi=(
                "deep_evi_raw",
                "median",
            ),
            mean_exhaustion_score=(
                "exhaustion_dysfunction_score",
                "mean",
            ),
            mean_activation_score=(
                "activation_effector_score",
                "mean",
            ),
        )
    )

    subtype_summary.to_csv(
        output_dir
        / "GSE176078_deep_evi_by_subtype.csv",
        index=False,
    )

    # =========================================================================
    # Annotation summary
    # =========================================================================

    annotation_summary = (
        predictions
        .groupby(
            "celltype_subset",
            as_index=False,
        )
        .agg(
            n_cells=(
                "cell_id",
                "size",
            ),
            mean_deep_evi=(
                "deep_evi_raw",
                "mean",
            ),
            median_deep_evi=(
                "deep_evi_raw",
                "median",
            ),
            mean_exhaustion_score=(
                "exhaustion_dysfunction_score",
                "mean",
            ),
            mean_activation_score=(
                "activation_effector_score",
                "mean",
            ),
        )
    )

    annotation_summary.to_csv(
        output_dir
        / "GSE176078_deep_evi_by_annotation.csv",
        index=False,
    )

    # =========================================================================
    # Save model
    # =========================================================================

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "programs": PROGRAMS,
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "dropout": args.dropout,
            "orientation_sign": orientation_sign,
            "evi_mean": evi_mean,
            "evi_std": evi_std,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "seed": args.seed,
            "best_epoch": best_epoch,
        },
        output_dir
        / "GSE176078_deep_evi_model.pt",
    )

    # =========================================================================
    # Report
    # =========================================================================

    report = {
        "cohort": "GSE176078",

        "step": "08A_deep_evi",

        "scientific_definition": (
            "Deep-EVI is a graph-learned continuous T-cell "
            "state index trained from six expression-derived "
            "biological programs and a sample-restricted "
            "30-nearest-neighbor T-cell graph."
        ),

        "not_rna_velocity": True,

        "velocity_warning": (
            "This implementation does not infer RNA velocity. "
            "The GSE176078 processed matrices used here do not "
            "contain the spliced/unspliced layers required for "
            "true RNA velocity estimation."
        ),

        "cells": int(
            len(landscape)
        ),

        "curated_t_cells": int(
            tcell_count
        ),

        "program_dimensions": len(
            PROGRAMS
        ),

        "programs": PROGRAMS,

        "landscape_input": str(
            args.landscape
        ),

        "graph_input": str(
            args.edges
        ),

        "score_input": str(
            args.scores
        ),

        "graph_edges_total": int(
            len(edges)
        ),

        "sample_split": {
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "train_cells": int(
                train_mask.sum()
            ),
            "validation_cells": int(
                validation_mask.sum()
            ),
            "test_cells": int(
                test_mask.sum()
            ),
        },

        "graph_split": {
            "train_internal_edges": int(
                len(
                    split_graphs[
                        "train"
                    ][
                        "internal_edges"
                    ]
                )
            ),
            "validation_internal_edges": int(
                len(
                    split_graphs[
                        "validation"
                    ][
                        "internal_edges"
                    ]
                )
            ),
            "test_internal_edges": int(
                len(
                    split_graphs[
                        "test"
                    ][
                        "internal_edges"
                    ]
                )
            ),
            "train_adjacency_nnz": int(
                split_graphs[
                    "train"
                ][
                    "adjacency"
                ].nnz
            ),
            "validation_adjacency_nnz": int(
                split_graphs[
                    "validation"
                ][
                    "adjacency"
                ].nnz
            ),
            "test_adjacency_nnz": int(
                split_graphs[
                    "test"
                ][
                    "adjacency"
                ].nnz
            ),
        },

        "model": {
            "hidden_dim": args.hidden_dim,
            "latent_dim": args.latent_dim,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "max_epochs": args.epochs,
            "patience": args.patience,
            "best_epoch": int(
                best_epoch
            ),
        },

        "validation_metrics": {
            "exhaustion_correlation": validation_exhaustion_corr,
            "activation_correlation": validation_activation_corr,
        },

        "test_metrics": {
            "exhaustion_correlation": test_exhaustion_corr,
            "exhaustion_rmse": test_exhaustion_rmse,
            "exhaustion_mae": test_exhaustion_mae,
            "exhaustion_r2": test_exhaustion_r2,
            "activation_correlation": test_activation_corr,
        },

        "orientation": {
            "training_exhaustion_correlation": train_orientation_corr,
            "orientation_sign": orientation_sign,
            "higher_deep_evi_direction": (
                "higher learned exhaustion-associated state"
                if orientation_sign > 0
                else "lower learned exhaustion-associated state"
            ),
        },

        "training": {
            "best_validation_loss": float(
                best_validation_loss
            ),
            "epochs_completed": len(
                history
            ),
        },

        "leakage_controls": {
            "sample_level_split": True,
            "cross_split_graph_edges_removed": True,
            "test_used_for_model_selection": False,
            "feature_scaling_fit_on_training_only": True,
            "evi_standardization_fit_on_training_only": True,
        },

        "data_interface": {
            "step_7b_scores_are_authoritative": True,
            "step_7c_landscape_provides_state_geometry": True,
            "step_7c_graph_provides_neighbor_structure": True,
            "program_score_join_key": "cell_id",
            "aligned_cells": int(
                len(score_features)
            ),
        },

        "status": "complete",
    }

    with open(
        output_dir
        / "GSE176078_deep_evi_report.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    print("=" * 80)
    print("DEEP-EVI COMPLETE")
    print("=" * 80)

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation loss: "
        f"{best_validation_loss:.6f}"
    )

    print(
        f"Validation exhaustion correlation: "
        f"{validation_exhaustion_corr:.6f}"
    )

    print(
        f"Test exhaustion correlation: "
        f"{test_exhaustion_corr:.6f}"
    )

    print(
        f"Test exhaustion RMSE: "
        f"{test_exhaustion_rmse:.6f}"
    )

    print(
        f"Test exhaustion MAE: "
        f"{test_exhaustion_mae:.6f}"
    )

    print(
        f"Test exhaustion R2: "
        f"{test_exhaustion_r2:.6f}"
    )

    print(
        f"Test activation correlation: "
        f"{test_activation_corr:.6f}"
    )

    print(
        "Outputs written to:",
        output_dir,
    )


if __name__ == "__main__":
    main()