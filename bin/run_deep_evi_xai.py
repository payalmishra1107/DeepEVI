#!/usr/bin/env python3

"""
Step 11 — Explainable Deep-EVI

Frozen-model explainability for the production GSE176078 Deep-EVI model.

Scientific safeguards
---------------------
1. Uses the frozen Step-8A checkpoint exactly as supplied.
2. Does NOT retrain the model.
3. Does NOT refit the 09A molecular surrogate.
4. Does NOT use TCGA immune-subtype labels for model construction.
5. Reconstructs the held-out test graph only.
6. Performs a strict frozen-model reproduction check before XAI.
7. Deep-EVI is NOT RNA velocity.
8. Molecular attribution is performed on the frozen 09A Ridge surrogate.
9. GNN attribution is performed on the frozen Step-8A model.
10. XAI results are interpretive and do not constitute causal inference.

Outputs
-------
GSE176078_11_program_attribution.csv
GSE176078_11_program_attribution_summary.csv
GSE176078_11_graph_edge_attribution.csv
GSE176078_11_graph_edge_summary.csv
GSE176078_11_molecular_attribution.csv
GSE176078_11_molecular_attribution_summary.csv
GSE176078_11_report.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------

PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]

REQUIRED_SCORE_COLUMNS = [
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

REQUIRED_CHECKPOINT_KEYS = [
    "model_state_dict",
    "programs",
    "hidden_dim",
    "latent_dim",
    "dropout",
    "orientation_sign",
    "evi_mean",
    "evi_std",
    "feature_mean",
    "feature_std",
    "seed",
    "best_epoch",
]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def fail(message: str) -> None:
    raise RuntimeError(message)


def json_safe(value):
    """Convert numpy / torch scalar types to JSON-safe Python values."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().tolist()
    return value


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        return float("nan")

    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")

    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    result = spearmanr(x, y)
    return float(result.statistic)


def summarize_vector(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "q25": float(np.quantile(x, 0.25)),
        "q75": float(np.quantile(x, 0.75)),
    }


# ---------------------------------------------------------------------
# Frozen Deep-EVI architecture
# ---------------------------------------------------------------------

class GraphConv(nn.Module):
    """
    Frozen Step-8A graph convolution.

    Message passing:
        A X W

    where A is a sparse row-normalized adjacency matrix.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ) -> torch.Tensor:

        propagated = torch.sparse.mm(adj, x)

        return self.linear(propagated)


class DeepEVIModel(nn.Module):
    """
    Exact architecture represented by the frozen production checkpoint.

    Checkpoint-verified dimensions:
        input      = 6
        hidden     = 64
        latent     = 32

    Heads:
        exhaustion_head = Linear(32, 1)
        activation_head = Linear(32, 1)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.conv1 = GraphConv(
            in_dim,
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
            nn.Dropout(dropout),
            nn.Linear(
                hidden_dim,
                in_dim,
            ),
        )

        # These are explicitly verified against
        # the production checkpoint.
        self.exhaustion_head = nn.Linear(
            latent_dim,
            1,
        )

        self.activation_head = nn.Linear(
            latent_dim,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
    ):

        h = self.conv1(
            x,
            adj,
        )

        h = self.norm1(h)
        h = F.gelu(h)
        h = self.dropout(h)

        z = self.conv2(
            h,
            adj,
        )

        z = self.norm2(z)
        z = F.gelu(z)

        reconstructed_programs = self.program_decoder(
            z
        )

        exhaustion = self.exhaustion_head(
            z
        ).squeeze(1)

        activation = self.activation_head(
            z
        ).squeeze(1)

        return (
            z,
            reconstructed_programs,
            exhaustion,
            activation,
        )


# ---------------------------------------------------------------------
# Frozen graph reconstruction
# ---------------------------------------------------------------------

def build_test_adjacency(
    edges: pd.DataFrame,
    test_ids: List[str],
) -> Tuple[torch.Tensor, pd.DataFrame, float]:

    local_index = {
        cell_id: i
        for i, cell_id in enumerate(test_ids)
    }

    source = edges["source_cell"].astype(str)
    target = edges["target_cell"].astype(str)

    internal_mask = (
        source.isin(local_index)
        &
        target.isin(local_index)
    )

    internal_edges = edges.loc[
        internal_mask,
        [
            "source_cell",
            "target_cell",
            "distance",
        ],
    ].copy()

    if len(internal_edges) == 0:
        fail(
            "No internal test-set graph edges were found."
        )

    rows = (
        internal_edges["source_cell"]
        .map(local_index)
        .to_numpy(np.int64)
    )

    cols = (
        internal_edges["target_cell"]
        .map(local_index)
        .to_numpy(np.int64)
    )

    distances = (
        internal_edges["distance"]
        .to_numpy(np.float32)
    )

    finite = np.isfinite(
        distances
    )

    rows = rows[finite]
    cols = cols[finite]
    distances = distances[finite]

    positive_distances = distances[
        distances > 0
    ]

    if len(positive_distances) == 0:
        distance_scale = 1.0
    else:
        distance_scale = float(
            np.median(
                positive_distances
            )
        )

    if not np.isfinite(distance_scale) or distance_scale <= 0:
        distance_scale = 1.0

    weights = np.exp(
        -distances / distance_scale
    ).astype(np.float32)

    n_cells = len(test_ids)

    # Add one self-loop per cell.
    rows = np.concatenate(
        [
            rows,
            np.arange(n_cells),
        ]
    )

    cols = np.concatenate(
        [
            cols,
            np.arange(n_cells),
        ]
    )

    weights = np.concatenate(
        [
            weights,
            np.ones(
                n_cells,
                dtype=np.float32,
            ),
        ]
    )

    raw_adjacency = sp.coo_matrix(
        (
            weights,
            (
                rows,
                cols,
            ),
        ),
        shape=(
            n_cells,
            n_cells,
        ),
        dtype=np.float32,
    )

    raw_adjacency.sum_duplicates()

    degree = np.asarray(
        raw_adjacency.sum(
            axis=1
        )
    ).ravel()

    degree[
        degree <= 0
    ] = 1.0

    normalized_adjacency = (
        sp.diags(
            1.0 / degree
        )
        .dot(
            raw_adjacency
        )
        .tocoo()
    )

    indices = torch.tensor(
        np.vstack(
            [
                normalized_adjacency.row,
                normalized_adjacency.col,
            ]
        ),
        dtype=torch.long,
    )

    values = torch.tensor(
        normalized_adjacency.data,
        dtype=torch.float32,
    )

    adjacency = torch.sparse_coo_tensor(
        indices,
        values,
        size=normalized_adjacency.shape,
    ).coalesce()

    return (
        adjacency,
        internal_edges,
        distance_scale,
    )


# ---------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------

def load_frozen_checkpoint(
    model_path: Path,
) -> Tuple[
    DeepEVIModel,
    Dict,
]:

    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    missing = [
        key
        for key in REQUIRED_CHECKPOINT_KEYS
        if key not in checkpoint
    ]

    if missing:
        fail(
            "Frozen checkpoint is missing required keys: "
            + ", ".join(missing)
        )

    checkpoint_programs = list(
        checkpoint["programs"]
    )

    if checkpoint_programs != PROGRAMS:
        fail(
            "Checkpoint program definition does not match "
            "the frozen Step-8A program order.\n"
            f"Checkpoint: {checkpoint_programs}\n"
            f"Expected:   {PROGRAMS}"
        )

    model = DeepEVIModel(
        in_dim=len(PROGRAMS),
        hidden_dim=int(
            checkpoint["hidden_dim"]
        ),
        latent_dim=int(
            checkpoint["latent_dim"]
        ),
        dropout=float(
            checkpoint["dropout"]
        ),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    return model, checkpoint


# ---------------------------------------------------------------------
# Program-level gradient × input attribution
# ---------------------------------------------------------------------

def compute_program_attribution(
    model: DeepEVIModel,
    X: torch.Tensor,
    adjacency: torch.Tensor,
) -> Tuple[
    np.ndarray,
    np.ndarray,
]:

    # A fresh leaf tensor is required for gradient attribution.
    x = X.detach().clone()
    x.requires_grad_(True)

    (
        latent,
        reconstructed_programs,
        exhaustion,
        activation,
    ) = model(
        x,
        adjacency,
    )

    target = exhaustion.sum()

    model.zero_grad(
        set_to_none=True
    )

    if x.grad is not None:
        x.grad.zero_()

    target.backward()

    gradients = (
        x.grad
        .detach()
        .cpu()
        .numpy()
    )

    values = (
        x.detach()
        .cpu()
        .numpy()
    )

    attribution = (
        gradients * values
    )

    return (
        attribution,
        gradients,
    )


# ---------------------------------------------------------------------
# Graph edge attribution
# ---------------------------------------------------------------------

def compute_graph_edge_attribution(
    model: DeepEVIModel,
    X: torch.Tensor,
    adjacency: torch.Tensor,
) -> np.ndarray:

    """
    Gradient × edge-weight attribution.

    The graph structure is held fixed except for the
    normalized adjacency values.

    This is an interpretive local attribution and
    does not constitute causal graph intervention.
    """

    adj = adjacency.detach().clone()
    adj.requires_grad_(True)

    (
        latent,
        reconstructed_programs,
        exhaustion,
        activation,
    ) = model(
        X.detach(),
        adj,
    )

    target = exhaustion.sum()

    model.zero_grad(
        set_to_none=True
    )

    if adj.grad is not None:
        adj.grad.zero_()

    target.backward()

    gradients = (
        adj.grad
        .detach()
        .cpu()
        .values()
        .numpy()
    )

    values = (
        adj.detach()
        .cpu()
        .values()
        .numpy()
    )

    attribution = (
        gradients * values
    )

    return attribution


# ---------------------------------------------------------------------
# Molecular surrogate attribution
# ---------------------------------------------------------------------

def load_frozen_signature(
    signature_path: Path,
) -> Dict:

    with open(
        signature_path,
        "r",
        encoding="utf-8",
    ) as handle:

        signature = json.load(
            handle
        )

    required = [
        "cohort",
        "source_step",
        "bridge_step",
        "programs",
        "coefficients",
        "gene_signature_rule",
        "important_definition",
    ]

    missing = [
        key
        for key in required
        if key not in signature
    ]

    if missing:
        fail(
            "Frozen 09A signature is missing: "
            + ", ".join(missing)
        )

    if signature["programs"] != PROGRAMS:
        fail(
            "Frozen 09A program order does not match "
            "Step-8A programs."
        )

    if not bool(
        signature.get(
            "training_only",
            False,
        )
    ):
        fail(
            "Frozen 09A signature does not declare "
            "training_only=true."
        )

    return signature


def build_program_molecular_attribution(
    signature: Dict,
) -> pd.DataFrame:

    coefficients = np.asarray(
        signature["coefficients"],
        dtype=float,
    )

    if len(coefficients) != len(
        PROGRAMS
    ):
        fail(
            "09A coefficient count does not match "
            "the six frozen programs."
        )

    rows = []

    for program, coefficient in zip(
        PROGRAMS,
        coefficients,
    ):

        rows.append(
            {
                "program": program,
                "ridge_coefficient": float(
                    coefficient
                ),
                "absolute_coefficient": float(
                    abs(coefficient)
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result["absolute_coefficient_fraction"] = (
        result["absolute_coefficient"]
        /
        result["absolute_coefficient"].sum()
    )

    result = result.sort_values(
        "absolute_coefficient",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result["rank"] = (
        np.arange(len(result))
        + 1
    )

    return result


def build_gene_molecular_attribution(
    signature: Dict,
) -> pd.DataFrame:

    """
    The frozen 09A gene-signature rule is:

        program coefficient / number of genes
        with overlapping genes summed.

    The actual gene-level weights are stored in the
    frozen signature only if supplied explicitly.

    This function therefore requires the signature to
    contain `gene_weights` for gene-level attribution.

    If gene_weights are absent, only program-level
    molecular attribution is produced.
    """

    gene_weights = signature.get(
        "gene_weights"
    )

    if gene_weights is None:
        return pd.DataFrame(
            columns=[
                "gene",
                "molecular_weight",
                "absolute_weight",
                "absolute_weight_fraction",
                "rank",
            ]
        )

    rows = []

    for gene, weight in gene_weights.items():
        rows.append(
            {
                "gene": str(gene),
                "molecular_weight": float(
                    weight
                ),
                "absolute_weight": float(
                    abs(weight)
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    total = result[
        "absolute_weight"
    ].sum()

    if total > 0:
        result[
            "absolute_weight_fraction"
        ] = (
            result["absolute_weight"]
            / total
        )
    else:
        result[
            "absolute_weight_fraction"
        ] = 0.0

    result = result.sort_values(
        "absolute_weight",
        ascending=False,
    ).reset_index(
        drop=True
    )

    result["rank"] = (
        np.arange(len(result))
        + 1
    )

    return result


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 11 Explainable Deep-EVI "
            "using the frozen Step-8A checkpoint."
        )
    )

    parser.add_argument(
        "--scores",
        required=True,
    )

    parser.add_argument(
        "--landscape",
        required=True,
    )

    parser.add_argument(
        "--edges",
        required=True,
    )

    parser.add_argument(
        "--splitManifest",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--signature",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------
    # Load frozen inputs
    # -------------------------------------------------------------

    scores = pd.read_csv(
        args.scores
    )

    landscape = pd.read_csv(
        args.landscape
    )

    edges = pd.read_csv(
        args.edges
    )

    split_manifest = pd.read_csv(
        args.splitManifest
    )

    signature = load_frozen_signature(
        Path(args.signature)
    )

    # Normalize IDs.
    scores["cell_id"] = (
        scores["cell_id"]
        .astype(str)
    )

    landscape["cell_id"] = (
        landscape["cell_id"]
        .astype(str)
    )

    edges["source_cell"] = (
        edges["source_cell"]
        .astype(str)
    )

    edges["target_cell"] = (
        edges["target_cell"]
        .astype(str)
    )

    # -------------------------------------------------------------
    # Validate score schema
    # -------------------------------------------------------------

    missing_score_columns = [
        column
        for column in REQUIRED_SCORE_COLUMNS
        if column not in scores.columns
    ]

    if missing_score_columns:
        fail(
            "Step-8A score table is missing columns: "
            + ", ".join(
                missing_score_columns
            )
        )

    # -------------------------------------------------------------
    # Select held-out test cells
    # -------------------------------------------------------------

    test = scores.loc[
        scores["split"].astype(str)
        == "test"
    ].copy()

    if len(test) == 0:
        fail(
            "No held-out test cells were found."
        )

    # IMPORTANT:
    #
    # The diagnostic established that score-table order and