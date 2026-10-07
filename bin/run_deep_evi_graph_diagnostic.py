#!/usr/bin/env python

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr
from sklearn.preprocessing import StandardScaler


ALL_PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def spearman_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    return float(spearmanr(x[mask], y[mask]).statistic)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    if mask.sum() < 3:
        return np.nan

    return float(pearsonr(x[mask], y[mask]).statistic)


def load_split_manifest(path):
    df = pd.read_csv(path)

    required = {"sample_id", "split"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Split manifest missing required columns: {sorted(missing)}"
        )

    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str).str.strip()
    df["split"] = df["split"].astype(str).str.strip().str.lower()

    allowed = {"train", "validation", "test"}
    invalid = set(df["split"].unique()) - allowed

    if invalid:
        raise RuntimeError(
            f"Invalid split labels: {sorted(invalid)}"
        )

    sample_split_counts = (
        df.groupby("sample_id")["split"]
        .nunique()
    )

    conflicting = sample_split_counts[
        sample_split_counts > 1
    ]

    if len(conflicting) > 0:
        details = (
            df[df["sample_id"].isin(conflicting.index)]
            .groupby("sample_id")["split"]
            .unique()
            .to_dict()
        )

        raise RuntimeError(
            "Sample-level leakage detected: "
            f"{details}"
        )

    sample_map = (
        df[["sample_id", "split"]]
        .drop_duplicates()
        .sort_values(["split", "sample_id"])
        .reset_index(drop=True)
    )

    if sample_map["sample_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate sample_id remains after manifest collapsing."
        )

    observed = set(sample_map["split"].unique())
    missing_partitions = allowed - observed

    if missing_partitions:
        raise RuntimeError(
            "Missing split partitions: "
            f"{sorted(missing_partitions)}"
        )

    counts = sample_map["split"].value_counts().to_dict()

    print(
        "Validated split manifest: "
        f"{len(sample_map)} samples; "
        f"train={counts.get('train', 0)}, "
        f"validation={counts.get('validation', 0)}, "
        f"test={counts.get('test', 0)}"
    )

    return dict(
        zip(
            sample_map["sample_id"],
            sample_map["split"],
        )
    )


def build_graph(
    edges,
    cell_to_idx,
    allowed_cells,
    normalization="symmetric",
):
    edges = edges[
        edges["source_cell"].isin(allowed_cells)
        &
        edges["target_cell"].isin(allowed_cells)
    ].copy()

    src = (
        edges["source_cell"]
        .map(cell_to_idx)
        .to_numpy()
    )

    dst = (
        edges["target_cell"]
        .map(cell_to_idx)
        .to_numpy()
    )

    distance = pd.to_numeric(
        edges["distance"],
        errors="coerce",
    ).to_numpy()

    valid = (
        np.isfinite(distance)
        &
        (src >= 0)
        &
        (dst >= 0)
    )

    src = src[valid]
    dst = dst[valid]
    distance = distance[valid]

    positive = distance[distance > 0]

    scale = (
        float(np.median(positive))
        if len(positive) > 0
        else 1.0
    )

    weights = np.exp(
        -distance / max(scale, 1e-8)
    )

    n = len(allowed_cells)

    if len(src) == 0:
        A = torch.sparse_coo_tensor(
            torch.empty((2, 0), dtype=torch.long),
            torch.empty((0,), dtype=torch.float32),
            size=(n, n),
        ).coalesce()
    else:
        A = torch.sparse_coo_tensor(
            np.vstack([src, dst]),
            weights.astype(np.float32),
            size=(n, n),
        ).coalesce()

    eye = torch.eye(n).to_sparse()
    A = (A + eye).coalesce()

    rows = A.indices()[0]
    cols = A.indices()[1]
    values = A.values()

    degree = torch.zeros(n)
    degree.index_add_(0, rows, values)

    if normalization == "symmetric":
        inv_degree = (
            degree
            .clamp_min(1e-8)
            .pow(-0.5)
        )

        norm_values = (
            values
            * inv_degree[rows]
            * inv_degree[cols]
        )

    elif normalization == "row":
        inv_degree = (
            degree
            .clamp_min(1e-8)
            .pow(-1.0)
        )

        norm_values = values * inv_degree[rows]

    else:
        raise ValueError(
            f"Unknown normalization: {normalization}"
        )

    A = torch.sparse_coo_tensor(
        A.indices(),
        norm_values,
        size=(n, n),
    ).coalesce()

    return A, len(src)


class GraphConv(nn.Module):

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        return self.linear(
            torch.sparse.mm(adj, x)
        )


class DiagnosticModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        dropout,
        output_dim,
        use_graph=True,
    ):
        super().__init__()

        self.use_graph = use_graph

        self.conv1 = GraphConv(
            input_dim,
            hidden_dim,
        )

        self.conv2 = GraphConv(
            hidden_dim,
            latent_dim,
        )

        self.mlp1 = nn.Linear(
            input_dim,
            hidden_dim,
        )

        self.mlp2 = nn.Linear(
            hidden_dim,
            latent_dim,
        )

        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

        self.program_decoder = nn.Linear(
            latent_dim,
            output_dim,
        )

        self.exhaustion_head = nn.Linear(
            latent_dim,
            1,
        )

        self.activation_head = nn.Linear(
            latent_dim,
            1,
        )

    def encode(self, x, adj):

        if self.use_graph:
            h = self.conv1(x, adj)
            h = F.gelu(h)
            h = self.dropout(h)
            z = self.conv2(h, adj)

        else:
            h = self.mlp1(x)
            h = F.gelu(h)
            h = self.dropout(h)
            z = self.mlp2(h)

        return self.norm(z)

    def forward(self, x, adj):

        z = self.encode(x, adj)

        program = self.program_decoder(z)

        exhaustion = (
            self.exhaustion_head(z)
            .squeeze(-1)
        )

        activation = (
            self.activation_head(z)
            .squeeze(-1)
        )

        return (
            z,
            program,
            exhaustion,
            activation,
        )


def train_variant(
    X,
    program_target,
    exhaustion_target,
    activation_target,
    adj,
    train_idx,
    val_idx,
    test_idx,
    variant,
    use_graph,
    use_smoothness,
    epochs,
    patience,
    hidden_dim,
    latent_dim,
    dropout,
    lr,
    weight_decay,
    seed,
):
    set_seed(seed)

    device = torch.device("cpu")

    X_t = torch.tensor(
        X,
        dtype=torch.float32,
        device=device,
    )

    program_t = torch.tensor(
        program_target,
        dtype=torch.float32,
        device=device,
    )

    exhaustion_t = torch.tensor(
        exhaustion_target,
        dtype=torch.float32,
        device=device,
    )

    activation_t = torch.tensor(
        activation_target,
        dtype=torch.float32,
        device=device,
    )

    model = DiagnosticModel(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        output_dim=program_target.shape[1],
        use_graph=use_graph,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Training-only graph for smoothness.
    train_mask = torch.zeros(
        X.shape[0],
        dtype=torch.bool,
        device=device,
    )

    train_mask[
        torch.as_tensor(
            train_idx,
            dtype=torch.long,
            device=device,
        )
    ] = True

    if (
        use_graph
        and use_smoothness
        and adj._nnz() > 0
    ):
        adj_indices = adj.indices()

        train_edge_mask = (
            train_mask[adj_indices[0]]
            &
            train_mask[adj_indices[1]]
        )

        train_adj = torch.sparse_coo_tensor(
            adj_indices[:, train_edge_mask],
            adj.values()[train_edge_mask],
            size=adj.shape,
            dtype=adj.dtype,
            device=device,
        ).coalesce()

    else:
        train_adj = torch.sparse_coo_tensor(
            torch.empty(
                (2, 0),
                dtype=torch.long,
                device=device,
            ),
            torch.empty(
                (0,),
                dtype=torch.float32,
                device=device,
            ),
            size=adj.shape,
            dtype=torch.float32,
            device=device,
        ).coalesce()

    best_state = None
    best_val = np.inf
    best_epoch = 0
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):

        model.train()
        optimizer.zero_grad()

        z, program_pred, exhaustion_pred, activation_pred = model(
            X_t,
            adj,
        )

        program_loss = F.mse_loss(
            program_pred[train_idx],
            program_t[train_idx],
        )

        exhaustion_loss = F.mse_loss(
            exhaustion_pred[train_idx],
            exhaustion_t[train_idx],
        )

        activation_loss = F.mse_loss(
            activation_pred[train_idx],
            activation_t[train_idx],
        )

        smooth_loss = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        )

        if (
            use_smoothness
            and train_adj._nnz() > 0
        ):
            rows = train_adj.indices()[0]
            cols = train_adj.indices()[1]

            smooth_loss = (
                (z[rows] - z[cols])
                .pow(2)
                .mean()
            )

        loss = (
            program_loss
            + 0.25 * exhaustion_loss
            + 0.10 * activation_loss
            + (
                0.05 * smooth_loss
                if use_smoothness
                else 0.0
            )
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        model.eval()

        with torch.no_grad():

            (
                z_val,
                program_val,
                exhaustion_val,
                activation_val,
            ) = model(
                X_t,
                adj,
            )

            val_program = F.mse_loss(
                program_val[val_idx],
                program_t[val_idx],
            )

            val_exhaustion = F.mse_loss(
                exhaustion_val[val_idx],
                exhaustion_t[val_idx],
            )

            val_activation = F.mse_loss(
                activation_val[val_idx],
                activation_t[val_idx],
            )

            val_loss = (
                val_program
                + 0.25 * val_exhaustion
                + 0.10 * val_activation
            )

        val_value = float(
            val_loss.item()
        )

        history.append({
            "variant": variant,
            "epoch": epoch,
            "train_loss": float(loss.item()),
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
                smooth_loss.item()
            ),
            "train_smoothness_edges": (
                int(train_adj._nnz())
                if use_smoothness
                else 0
            ),
            "validation_loss": val_value,
            "validation_program_loss": float(
                val_program.item()
            ),
            "validation_exhaustion_loss": float(
                val_exhaustion.item()
            ),
            "validation_activation_loss": float(
                val_activation.item()
            ),
        })

        if val_value < best_val - 1e-7:
            best_val = val_value
            best_epoch = epoch
            no_improve = 0

            best_state = {
                k: v.detach()
                .cpu()
                .clone()
                for k, v in model.state_dict().items()
            }
        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"No valid checkpoint for {variant}"
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():

        z, program_pred, exhaustion_pred, activation_pred = model(
            X_t,
            adj,
        )

    z = z.cpu().numpy()

    exhaustion_pred = (
        exhaustion_pred
        .cpu()
        .numpy()
    )

    activation_pred = (
        activation_pred
        .cpu()
        .numpy()
    )

    sign = np.sign(
        spearman_corr(
            z[train_idx, 0],
            exhaustion_target[train_idx],
        )
    )

    if not np.isfinite(sign) or sign == 0:
        sign = 1.0

    evi = z[:, 0] * sign

    metrics = {
        "variant": variant,
        "best_epoch": float(best_epoch),
        "epochs_completed": float(len(history)),
        "best_validation_loss": float(best_val),
        "orientation_sign": float(sign),

        "train_evi_exhaustion_spearman":
            spearman_corr(
                evi[train_idx],
                exhaustion_target[train_idx],
            ),

        "train_evi_exhaustion_pearson":
            pearson_corr(
                evi[train_idx],
                exhaustion_target[train_idx],
            ),

        "validation_evi_exhaustion_spearman":
            spearman_corr(
                evi[val_idx],
                exhaustion_target[val_idx],
            ),

        "validation_evi_exhaustion_pearson":
            pearson_corr(
                evi[val_idx],
                exhaustion_target[val_idx],
            ),

        "test_evi_exhaustion_spearman":
            spearman_corr(
                evi[test_idx],
                exhaustion_target[test_idx],
            ),

        "test_evi_exhaustion_pearson":
            pearson_corr(
                evi[test_idx],
                exhaustion_target[test_idx],
            ),

        "train_evi_activation_spearman":
            spearman_corr(
                evi[train_idx],
                activation_target[train_idx],
            ),

        "validation_evi_activation_spearman":
            spearman_corr(
                evi[val_idx],
                activation_target[val_idx],
            ),

        "test_evi_activation_spearman":
            spearman_corr(
                evi[test_idx],
                activation_target[test_idx],
            ),

        "graph_used": bool(use_graph),
        "smoothness_used": bool(use_smoothness),

        "training_smoothness_edges": (
            int(train_adj._nnz())
            if use_smoothness
            else 0
        ),
    }

    return (
        metrics,
        pd.DataFrame(history),
        evi,
        z,
        model,
    )


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
        "--split-manifest",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=20,
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
        default=0.001,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260924,
    )

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    set_seed(args.seed)

    landscape = pd.read_csv(
        args.landscape
    )

    edges = pd.read_csv(
        args.edges
    )

    scores = pd.read_csv(
        args.scores
    )

    split_map = load_split_manifest(
        args.split_manifest
    )

    # ----------------------------------------------------------
    # SCORE TABLE VALIDATION
    # ----------------------------------------------------------

    required_scores = {
        "cell_id",
        "sample_id",
        "subtype",
        "celltype_subset",
    }

    required_scores.update(
        ALL_PROGRAMS
    )

    missing = (
        required_scores
        - set(scores.columns)
    )

    if missing:
        raise RuntimeError(
            "Score table missing columns: "
            f"{sorted(missing)}"
        )

    scores = scores.copy()

    scores["cell_id"] = (
        scores["cell_id"]
        .astype(str)
        .str.strip()
    )

    scores["sample_id"] = (
        scores["sample_id"]
        .astype(str)
        .str.strip()
    )

    # The Step 8A score table is already the curated
    # T-cell population. Do not require celltype_major here.

    if len(scores) != 35214:
        raise RuntimeError(
            "Expected the Step 8A score table to contain "
            f"35,214 curated T cells; found {len(scores)}"
        )

    if scores["cell_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate cell_id values in Step 8A score table."
        )

    # ----------------------------------------------------------
    # LANDSCAPE VALIDATION
    # ----------------------------------------------------------

    if "cell_id" not in landscape.columns:
        raise RuntimeError(
            "Landscape is missing cell_id."
        )

    landscape["cell_id"] = (
        landscape["cell_id"]
        .astype(str)
        .str.strip()
    )

    if landscape["cell_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate cell_id values in Step 7C landscape."
        )

    # ----------------------------------------------------------
    # ALIGNMENT
    # ----------------------------------------------------------

    merged = landscape.merge(
        scores[
            [
                "cell_id",
                "sample_id",
                "subtype",
                "celltype_subset",
            ]
            + ALL_PROGRAMS
        ],
        on="cell_id",
        how="inner",
        suffixes=("", "_scores"),
    )

    if len(merged) != 35214:
        raise RuntimeError(
            "Landscape/score alignment failed: "
            f"expected 35214, found {len(merged)}"
        )

    if merged["cell_id"].duplicated().any():
        raise RuntimeError(
            "Aligned dataset contains duplicate cell_id values."
        )

    print(
        "Curated T-cell alignment: PASS; "
        f"{len(merged)} cells"
    )

    # ----------------------------------------------------------
    # FEATURES
    # ----------------------------------------------------------

    raw_X = (
        merged[ALL_PROGRAMS]
        .to_numpy(dtype=np.float32)
    )

    if not np.isfinite(raw_X).all():
        raise RuntimeError(
            "Non-finite program values detected."
        )

    exhaustion = (
        merged[
            "exhaustion_dysfunction_score"
        ]
        .to_numpy(dtype=np.float32)
    )

    activation = (
        merged[
            "activation_effector_score"
        ]
        .to_numpy(dtype=np.float32)
    )

    # ----------------------------------------------------------
    # SAMPLE-LEVEL SPLIT
    # ----------------------------------------------------------

    sample_ids = (
        merged["sample_id"]