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
            "delta_activation",