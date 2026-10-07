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
from scipy.stats import spearmanr, pearsonr, kruskal
from sklearn.preprocessing import StandardScaler


ALL_PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
    "exhaustion_dysfunction_score",
]

NO_EXHAUSTION_PROGRAMS = [
    "tcell_identity_score",
    "cd8_cytotoxic_score",
    "treg_score",
    "tfh_score",
    "activation_effector_score",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 3:
        return np.nan

    return float(spearmanr(x[m], y[m]).statistic)


def pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)

    if m.sum() < 3:
        return np.nan

    return float(pearsonr(x[m], y[m]).statistic)


def load_split_manifest(path):
    """
    Load the Step 8A split manifest and return a validated
    sample_id -> split mapping.

    The manifest may be represented either as:
      1. one row per sample, or
      2. one row per cell, with repeated sample_id values.

    In both cases, the function requires every sample to have exactly
    one split assignment. A sample appearing in multiple splits is
    treated as a hard leakage-control failure.
    """
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

    if df["sample_id"].isna().any() or (df["sample_id"] == "").any():
        raise RuntimeError("Split manifest contains empty sample_id values.")

    allowed_splits = {"train", "validation", "test"}
    observed_splits = set(df["split"].unique())
    invalid_splits = observed_splits - allowed_splits

    if invalid_splits:
        raise RuntimeError(
            f"Invalid split labels in manifest: {sorted(invalid_splits)}. "
            f"Allowed labels: {sorted(allowed_splits)}"
        )

    # Validate that each sample has exactly one split.
    sample_split_counts = (
        df.groupby("sample_id")["split"]
        .nunique()
    )

    conflicting = sample_split_counts[sample_split_counts > 1]

    if len(conflicting) > 0:
        details = (
            df[df["sample_id"].isin(conflicting.index)]
            .groupby("sample_id")["split"]
            .unique()
            .to_dict()
        )
        raise RuntimeError(
            "Sample-level leakage detected: one or more sample_id values "
            f"occur in multiple splits: {details}"
        )

    # Collapse cell-level manifests to one row per sample.
    sample_map_df = (
        df[["sample_id", "split"]]
        .drop_duplicates()
        .sort_values(["split", "sample_id"])
        .reset_index(drop=True)
    )

    if sample_map_df["sample_id"].duplicated().any():
        raise RuntimeError(
            "Internal manifest validation failed: duplicate sample_id "
            "remains after collapsing the manifest."
        )

    split_counts = sample_map_df["split"].value_counts().to_dict()

    # For this cohort we expect all three partitions.
    missing_partitions = allowed_splits - set(split_counts)
    if missing_partitions:
        raise RuntimeError(
            "Split manifest is missing required partition(s): "
            f"{sorted(missing_partitions)}"
        )

    print(
        "Validated split manifest: "
        f"{len(sample_map_df)} unique samples; "
        f"train={split_counts.get('train', 0)}, "
        f"validation={split_counts.get('validation', 0)}, "
        f"test={split_counts.get('test', 0)}"
    )

    return dict(
        zip(
            sample_map_df["sample_id"],
            sample_map_df["split"],
        )
    )


def build_graph(edges, cell_to_idx, allowed_cells):
    edges = edges[
        edges["source_cell"].isin(allowed_cells)
        & edges["target_cell"].isin(allowed_cells)
    ].copy()

    src = edges["source_cell"].map(cell_to_idx).to_numpy()
    dst = edges["target_cell"].map(cell_to_idx).to_numpy()

    distance = pd.to_numeric(
        edges["distance"], errors="coerce"
    ).to_numpy()

    valid = (
        np.isfinite(distance)
        & (src >= 0)
        & (dst >= 0)
    )

    src = src[valid]
    dst = dst[valid]
    distance = distance[valid]

    positive = distance[distance > 0]

    if len(positive) == 0:
        scale = 1.0
    else:
        scale = float(np.median(positive))

    weights = np.exp(-distance / max(scale, 1e-8))

    n = len(allowed_cells)

    A = torch.sparse_coo_tensor(
        np.vstack([src, dst]),
        weights.astype(np.float32),
        size=(n, n),
    ).coalesce()

    # Add self-loops.
    eye = torch.eye(n).to_sparse()

    A = (A + eye).coalesce()

    row = A.indices()[0]
    values = A.values()

    degree = torch.zeros(n)
    degree.index_add_(0, row, values)

    inv_degree = degree.clamp_min(1e-8).pow(-0.5)

    norm_values = (
        values
        * inv_degree[row]
        * inv_degree[A.indices()[1]]
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
        x = torch.sparse.mm(adj, x)
        return self.linear(x)


class GraphModel(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        latent_dim,
        dropout,
        output_dim,
        use_graph=True,
        exhaustion_head=True,
    ):
        super().__init__()

        self.use_graph = use_graph
        self.exhaustion_head_enabled = exhaustion_head

        self.conv1 = GraphConv(input_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, latent_dim)

        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        self.mlp2 = nn.Linear(hidden_dim, latent_dim)

        self.norm = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

        self.program_decoder = nn.Linear(
            latent_dim,
            output_dim,
        )

        self.activation_head = nn.Linear(
            latent_dim,
            1,
        )

        if exhaustion_head:
            self.exhaustion_head = nn.Linear(
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

        z = self.norm(z)

        return z

    def forward(self, x, adj):

        z = self.encode(x, adj)

        program = self.program_decoder(z)
        activation = self.activation_head(z).squeeze(-1)

        if self.exhaustion_head_enabled:
            exhaustion = self.exhaustion_head(z).squeeze(-1)
        else:
            exhaustion = None

        return z, program, exhaustion, activation


def train_model(
    X,
    program_target,
    exhaustion_target,
    activation_target,
    adj,
    train_idx,
    val_idx,
    test_idx,
    variant,
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

    use_graph = variant != "no_graph"
    use_exhaustion = variant != "no_exhaustion"

    model = GraphModel(
        input_dim=X.shape[1],
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        output_dim=program_target.shape[1],
        use_graph=use_graph,
        exhaustion_head=use_exhaustion,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------------
    # TRAINING-ONLY GRAPH FOR SMOOTHNESS REGULARIZATION
    # --------------------------------------------------------------
    # The model receives a split-isolated adjacency during forward
    # passes, but the smoothness loss must be restricted explicitly
    # to training cells. This prevents validation/test graph structure
    # from contributing to the optimization objective.
    train_mask = torch.zeros(
        X.shape[0],
        dtype=torch.bool,
        device=device,
    )
    train_mask[torch.as_tensor(train_idx, dtype=torch.long, device=device)] = True

    if use_graph and adj._nnz() > 0:
        adj_indices = adj.indices()
        train_edge_mask = (
            train_mask[adj_indices[0]]
            & train_mask[adj_indices[1]]
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
            torch.empty((2, 0), dtype=torch.long, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
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

        activation_loss = F.mse_loss(
            activation_pred[train_idx],
            activation_t[train_idx],
        )

        smooth_loss = torch.tensor(
            0.0,
            dtype=torch.float32,
            device=device,
        )

        if use_graph and train_adj._nnz() > 0:

            train_rows = train_adj.indices()[0]
            train_cols = train_adj.indices()[1]

            smooth_loss = (
                (z[train_rows] - z[train_cols]).pow(2).mean()
            )

        loss = (
            program_loss
            + 0.10 * activation_loss
            + 0.05 * smooth_loss
        )

        exhaustion_loss = torch.tensor(
            0.0,
            dtype=torch.float32,
        )

        if use_exhaustion:
            exhaustion_loss = F.mse_loss(
                exhaustion_pred[train_idx],
                exhaustion_t[train_idx],
            )

            loss = loss + 0.25 * exhaustion_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            5.0,
        )

        optimizer.step()

        model.eval()

        with torch.no_grad():

            z, program_pred, exhaustion_pred, activation_pred = model(
                X_t,
                adj,
            )

            val_program = F.mse_loss(
                program_pred[val_idx],
                program_t[val_idx],
            )

            val_activation = F.mse_loss(
                activation_pred[val_idx],
                activation_t[val_idx],
            )

            val_loss = (
                val_program
                + 0.10 * val_activation
            )

            if use_exhaustion:
                val_exhaustion = F.mse_loss(
                    exhaustion_pred[val_idx],
                    exhaustion_t[val_idx],
                )

                val_loss = (
                    val_loss
                    + 0.25 * val_exhaustion
                )
            else:
                val_exhaustion = torch.tensor(
                    0.0
                )

            if use_graph:
                val_smooth = smooth_loss
            else:
                val_smooth = torch.tensor(
                    0.0
                )

        val_value = float(val_loss.item())

        history.append({
            "epoch": epoch,
            "train_loss": float(loss.item()),
            "train_program_loss": float(program_loss.item()),
            "train_exhaustion_loss": float(exhaustion_loss.item()),
            "train_activation_loss": float(
                activation_loss.item()
            ),
            "train_smoothness_loss": float(
                smooth_loss.item()
            ),
            "train_smoothness_edges": int(
                train_adj._nnz()
            ) if use_graph else 0,
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
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            no_improve = 0

        else:
            no_improve += 1

        if no_improve >= patience:
            break

    if best_state is None:
        raise RuntimeError(
            f"No valid checkpoint generated for {variant}"
        )

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():

        z, program_pred, exhaustion_pred, activation_pred = model(
            X_t,
            adj,
        )

    z = z.cpu().numpy()
    activation_pred = activation_pred.cpu().numpy()

    if exhaustion_pred is not None:
        exhaustion_pred = exhaustion_pred.cpu().numpy()

    # Orient latent axis using training-only exhaustion association.
    # This does not alter model fitting.
    if latent_dim >= 1:

        train_latent = z[train_idx, 0]

        sign = np.sign(
            corr(
                train_latent,
                exhaustion_target[train_idx],
            )
        )

        if not np.isfinite(sign) or sign == 0:
            sign = 1.0

        evi = z[:, 0] * sign

    else:
        sign = 1.0
        evi = z[:, 0]

    metrics = {
        "variant": variant,
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "best_validation_loss": float(best_val),
        "orientation_sign": float(sign),
    }

    for split_name, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:

        metrics[
            f"{split_name}_evi_exhaustion_spearman"
        ] = corr(
            evi[indices],
            exhaustion_target[indices],
        )

        metrics[
            f"{split_name}_evi_exhaustion_pearson"
        ] = pearson(
            evi[indices],
            exhaustion_target[indices],
        )

        metrics[
            f"{split_name}_evi_activation_spearman"
        ] = corr(
            evi[indices],
            activation_target[indices],
        )

    return model, evi, z, activation_pred, exhaustion_pred, history, metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--landscape", required=True)
    parser.add_argument("--edges", required=True)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=20260924)

    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    landscape = pd.read_csv(args.landscape)
    edges = pd.read_csv(args.edges)

    split_map = load_split_manifest(
        args.split_manifest
    )

    required = [
        "cell_id",
        "sample_id",
        "subtype",
        "celltype_subset",
        "split",
        *ALL_PROGRAMS,
    ]

    missing = [
        x for x in required
        if x not in scores.columns
    ]

    if missing:
        raise RuntimeError(
            "Scores missing columns: "
            + ", ".join(missing)
        )

    if len(scores) != 35214:
        raise RuntimeError(
            f"Expected 35214 cells, found {len(scores)}"
        )

    if scores["cell_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate cell IDs detected."
        )

    if set(scores["sample_id"]) != set(split_map):
        raise RuntimeError(
            "Current score samples do not match 08A split manifest."
        )

    expected_split = scores["sample_id"].map(split_map)

    if not (
        expected_split.to_numpy()
        == scores["split"].astype(str).to_numpy()
    ).all():
        raise RuntimeError(
            "Score split assignments differ from 08A split manifest."
        )

    # Align landscape to score table.
    if set(scores["cell_id"]) != set(
        landscape["cell_id"]
    ):
        raise RuntimeError(
            "Landscape and score cell IDs differ."
        )

    landscape = landscape.set_index(
        "cell_id"
    ).loc[
        scores["cell_id"]
    ].reset_index()

    # Ensure curated T-cell compartment.
    if "celltype_major" in landscape.columns:
        major = landscape["celltype_major"].astype(str)

        if not (major == "T-cells").all():
            raise RuntimeError(
                "Landscape contains non-curated T cells."
            )

    # Align program values.
    score_values = scores[ALL_PROGRAMS].apply(
        pd.to_numeric,
        errors="coerce",
    )

    if score_values.isna().any().any():
        raise RuntimeError(
            "Program score table contains NaN."
        )

    # Split indices.
    train_idx = np.where(
        scores["split"].to_numpy() == "train"
    )[0]

    val_idx = np.where(
        scores["split"].to_numpy() == "validation"
    )[0]

    test_idx = np.where(
        scores["split"].to_numpy() == "test"
    )[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            "One or more splits contain zero cells."
        )

    # Train-only scaling.
    scaler_all = StandardScaler()

    X_all = scaler_all.fit_transform(
        score_values.to_numpy()[train_idx]
    )

    X = np.zeros_like(
        score_values.to_numpy(),
        dtype=np.float32,
    )

    X[train_idx] = X_all

    X[val_idx] = (
        score_values.to_numpy()[val_idx]
        - scaler_all.mean_
    ) / scaler_all.scale_

    X[test_idx] = (
        score_values.to_numpy()[test_idx]
        - scaler_all.mean_
    ) / scaler_all.scale_

    # Standardized targets based only on training data.
    exhaustion = score_values[
        "exhaustion_dysfunction_score"
    ].to_numpy()

    activation = score_values[
        "activation_effector_score"
    ].to_numpy()

    ex_mean = exhaustion[train_idx].mean()
    ex_std = exhaustion[train_idx].std()

    ac_mean = activation[train_idx].mean()
    ac_std = activation[train_idx].std()

    exhaustion_z = (
        (exhaustion - ex_mean)
        / max(ex_std, 1e-8)
    )

    activation_z = (
        (activation - ac_mean)
        / max(ac_std, 1e-8)
    )

    # Build graph separately for each split.
    cell_to_global = dict(
        zip(
            scores["cell_id"],
            range(len(scores)),
        )
    )

    graph_edges = {}

    for split_name, indices in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:

        cells = set(
            scores.iloc[indices]["cell_id"]
        )

        sub_edges = edges[
            edges["source_cell"].isin(cells)
            & edges["target_cell"].isin(cells)
        ].copy()

        graph_edges[split_name] = sub_edges

    # For model training we create one full adjacency with
    # cross-split edges removed. The model never gets cross-split
    # information.
    adjacency, internal_edges = build_graph(
        edges,
        cell_to_global,
        set(scores["cell_id"]),
    )

    # The function above only retained edges with both endpoints
    # present. Explicitly verify that no cross-split edges remain.
    src_split = scores.set_index(
        "cell_id"
    )["split"]

    edge_check = edges[
        edges["source_cell"].isin(src_split.index)
        & edges["target_cell"].isin(src_split.index)
    ].copy()

    cross = (
        edge_check["source_cell"].map(src_split)
        != edge_check["target_cell"].map(src_split)
    )

    if cross.any():
        # Rebuild strictly within split.
        valid_edges = edge_check.loc[
            ~cross
        ].copy()

        adjacency, internal_edges = build_graph(
            valid_edges,
            cell_to_global,
            set(scores["cell_id"]),
        )

    # --------------------------------------------------------------
    # Variant-specific input matrices
    # --------------------------------------------------------------

    variant_inputs = {
        "full": ALL_PROGRAMS,
        "no_exhaustion": NO_EXHAUSTION_PROGRAMS,
        "no_graph": ALL_PROGRAMS,
    }

    results = []
    histories = []
    latent_rows = []

    for variant, programs in variant_inputs.items():

        cols = [
            ALL_PROGRAMS.index(p)
            for p in programs
        ]

        X_variant = X[:, cols]

        program_target = (
            score_values[programs]
            .to_numpy()
        )

        # Standardize program reconstruction targets using
        # training cells only.
        pmean = program_target[train_idx].mean(
            axis=0
        )

        pstd = program_target[train_idx].std(
            axis=0
        )

        program_target = (
            program_target - pmean
        ) / np.maximum(pstd, 1e-8)

        model, evi, latent, activation_pred, exhaustion_pred, history, metrics = train_model(
            X_variant,
            program_target,
            exhaustion_z,
            activation_z,
            adjacency,
            train_idx,
            val_idx,
            test_idx,
            variant,
            args.epochs,
            args.patience,
            args.hidden_dim,
            args.latent_dim,
            args.dropout,
            args.learning_rate,
            args.weight_decay,
            args.seed,
        )

        metrics["input_programs"] = ",".join(programs)
        metrics["graph_used"] = variant != "no_graph"
        metrics["exhaustion_supervision"] = variant != "no_exhaustion"

        results.append(metrics)

        for row in history:
            row["variant"] = variant
            histories.append(row)

        for i in range(len(scores)):

            latent_rows.append({
                "variant": variant,
                "cell_id": scores.iloc[i]["cell_id"],
                "sample_id": scores.iloc[i]["sample_id"],
                "subtype": scores.iloc[i]["subtype"],
                "split": scores.iloc[i]["split"],
                "deep_evi_raw": float(evi[i]),
                "activation_prediction": float(
                    activation_pred[i]
                ),
                "exhaustion_prediction": (
                    float(exhaustion_pred[i])
                    if exhaustion_pred is not None
                    else np.nan
                ),
            })

        torch.save(
            {
                "variant": variant,
                "state_dict": model.state_dict(),                "input_programs": programs,
                "seed": args.seed,
            },
            outdir / f"GSE176078_08C_{variant}.pt",
        )

    # --------------------------------------------------------------
    # Direct baseline
    # --------------------------------------------------------------

    direct_test = scores[
        scores["split"] == "test"
    ]

    baseline = {
        "variant": "direct_exhaustion_baseline",
        "input_programs": "exhaustion_dysfunction_score",
        "graph_used": False,
        "exhaustion_supervision": False,
        "test_evi_exhaustion_spearman": 1.0,
        "test_evi_exhaustion_pearson": 1.0,
        "note": (
            "This is the direct target baseline and is not an "
            "independent model. It defines the reference level "
            "that a learned EVI must be interpreted against."
        ),
    }

    results.append(baseline)

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        outdir / "GSE176078_08C_ablation_metrics.csv",
        index=False,
    )

    history_df = pd.DataFrame(histories)

    history_df.to_csv(
        outdir / "GSE176078_08C_training_history.csv",
        index=False,
    )

    latent_df = pd.DataFrame(latent_rows)

    latent_df.to_csv(
        outdir / "GSE176078_08C_variant_scores.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Patient-level summaries
    # --------------------------------------------------------------

    patient_rows = []

    for variant in latent_df["variant"].unique():

        if variant == "direct_exhaustion_baseline":
            continue

        v = latent_df[
            latent_df["variant"] == variant
        ]

        for sample_id, g in v.groupby(
            "sample_id"
        ):

            patient_rows.append({
                "variant": variant,
                "sample_id": sample_id,
                "subtype": g["subtype"].iloc[0],
                "split": g["split"].iloc[0],
                "n_cells": len(g),
                "mean_deep_evi": float(
                    g["deep_evi_raw"].mean()
                ),
                "median_deep_evi": float(
                    g["deep_evi_raw"].median()
                ),
                "std_deep_evi": float(
                    g["deep_evi_raw"].std(ddof=1)
                ) if len(g) > 1 else 0.0,
            })

    patient_df = pd.DataFrame(patient_rows)

    patient_df.to_csv(
        outdir / "GSE176078_08C_patient_level_summary.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Patient-level subtype analysis on test samples
    # --------------------------------------------------------------

    patient_test = patient_df[
        patient_df["split"] == "test"
    ].copy()

    subtype_rows = []

    for variant in patient_test["variant"].unique():

        v = patient_test[
            patient_test["variant"] == variant
        ]

        groups = []

        for subtype in [
            "ER+",
            "HER2+",
            "TNBC",
        ]:

            values = v.loc[
                v["subtype"] == subtype,
                "mean_deep_evi",
            ].dropna()

            if len(values):
                groups.append(
                    (subtype, values)
                )

        if len(groups) >= 2:

            stat, pvalue = kruskal(
                *[
                    x[1].to_numpy()
                    for x in groups
                ]
            )

        else:
            stat = np.nan
            pvalue = np.nan

        subtype_rows.append({
            "variant": variant,
            "test_patient_n": int(len(v)),
            "kruskal_statistic": (
                float(stat)
                if np.isfinite(stat)
                else np.nan
            ),
            "kruskal_pvalue": (
                float(pvalue)
                if np.isfinite(pvalue)
                else np.nan
            ),
            "subtypes_present": ",".join(
                x[0] for x in groups
            ),
        })

    pd.DataFrame(subtype_rows).to_csv(
        outdir / "GSE176078_08C_patient_level_subtype_test.csv",
        index=False,
    )

    # --------------------------------------------------------------
    # Report
    # --------------------------------------------------------------

    report = {
        "cohort": "GSE176078",
        "step": "08C_deep_evi_ablation_independence",
        "status": "complete",
        "cells": int(len(scores)),
        "samples": int(scores["sample_id"].nunique()),
        "sample_split_reused_from_08A": True,
        "variants": [
            {
                "name": "full",
                "programs": ALL_PROGRAMS,
                "graph": True,
                "exhaustion_supervision": True,
            },
            {
                "name": "no_exhaustion",
                "programs": NO_EXHAUSTION_PROGRAMS,
                "graph": True,
                "exhaustion_supervision": False,
            },
            {
                "name": "no_graph",
                "programs": ALL_PROGRAMS,
                "graph": False,
                "exhaustion_supervision": True,
            },
            {
                "name": "direct_exhaustion_baseline",
                "programs": [
                    "exhaustion_dysfunction_score"
                ],
                "graph": False,
                "exhaustion_supervision": False,
            },
        ],
        "leakage_controls": {
            "sample_level_split": True,
            "same_08A_split_manifest": True,
            "cross_split_graph_edges_removed": True,
            "feature_scaling_training_only": True,
            "target_scaling_training_only": True,
            "no_exhaustion_variant_uses_exhaustion_as_input": True,
            "no_exhaustion_variant_has_exhaustion_training_loss": True,
            "test_used_for_model_selection": False,
            "evi_orientation_training_only": True,
        },
        "scientific_interpretation": (
            "08C evaluates whether graph learning and multivariate "
            "T-cell state representation contribute information "
            "beyond direct reconstruction of the exhaustion-associated "
            "training target. The no-exhaustion variant removes the "
            "exhaustion score from both inputs and supervision."
        ),
    }

    with open(
        outdir / "GSE176078_08C_report.json",
        "w",
    ) as handle:
        json.dump(
            report,
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()