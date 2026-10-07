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
        merged["sample_id"]        .astype(str)
        .to_numpy()
    )

    unknown_samples = (
        set(sample_ids)
        - set(split_map)
    )

    if unknown_samples:
        raise RuntimeError(
            "Samples absent from split manifest: "
            f"{sorted(unknown_samples)}"
        )

    split = np.array([
        split_map[s]
        for s in sample_ids
    ])

    train_idx = np.where(
        split == "train"
    )[0]

    val_idx = np.where(
        split == "validation"
    )[0]

    test_idx = np.where(
        split == "test"
    )[0]

    if min(
        len(train_idx),
        len(val_idx),
        len(test_idx),
    ) == 0:
        raise RuntimeError(
            "One or more partitions is empty."
        )

    print(
        "Cell split: "
        f"train={len(train_idx)}, "
        f"validation={len(val_idx)}, "
        f"test={len(test_idx)}"
    )

    # ----------------------------------------------------------
    # TRAIN-ONLY FEATURE STANDARDIZATION
    # ----------------------------------------------------------

    feature_scaler = StandardScaler()

    feature_scaler.fit(
        raw_X[train_idx]
    )

    X = feature_scaler.transform(
        raw_X
    ).astype(np.float32)

    # ----------------------------------------------------------
    # TRAIN-ONLY TARGET STANDARDIZATION
    # ----------------------------------------------------------

    def train_zscore(y):

        mean = float(
            np.mean(y[train_idx])
        )

        std = float(
            np.std(y[train_idx])
        )

        if std < 1e-8:
            std = 1.0

        return (
            (y - mean) / std
        ).astype(np.float32)

    exhaustion_scaled = train_zscore(
        exhaustion
    )

    activation_scaled = train_zscore(
        activation
    )

    program_target = X.copy()

    # ----------------------------------------------------------
    # RAW GRAPH / SPLIT ISOLATION AUDIT
    # ----------------------------------------------------------
    #
    # The Step 7C KNN graph is constructed over the complete
    # curated T-cell population. Therefore cross-split edges
    # are expected to exist in the raw edge table.
    #
    # They must NOT be treated as leakage by themselves.
    # The leakage control is that these edges are removed before
    # the graph is supplied to each model.
    #

    cell_ids = (
        merged["cell_id"]
        .astype(str)
        .tolist()
    )

    split_by_cell = {
        cell_ids[i]: split[i]
        for i in range(len(cell_ids))
    }

    edge_source = (
        edges["source_cell"]
        .astype(str)
        .str.strip()
    )

    edge_target = (
        edges["target_cell"]
        .astype(str)
        .str.strip()
    )

    valid_raw_edges = (
        edge_source.isin(split_by_cell)
        &
        edge_target.isin(split_by_cell)
    )

    audit_edges = edges.loc[
        valid_raw_edges,
        ["source_cell", "target_cell"]
    ].copy()

    source_split = (
        audit_edges["source_cell"]
        .map(split_by_cell)
    )

    target_split = (
        audit_edges["target_cell"]
        .map(split_by_cell)
    )

    raw_cross_split_edges = int(
        (
            source_split
            != target_split
        ).sum()
    )

    raw_within_split_edges = int(
        (
            source_split
            == target_split
        ).sum()
    )

    raw_edges_checked = int(
        len(audit_edges)
    )

    print(
        "Raw graph audit: "
        f"edges_checked={raw_edges_checked}, "
        f"within_split={raw_within_split_edges}, "
        f"cross_split={raw_cross_split_edges}"
    )

    if raw_edges_checked == 0:
        raise RuntimeError(
            "No usable graph edges were found after "
            "matching edge cell IDs to the curated T-cell "
            "population."
        )

    # ----------------------------------------------------------
    # SPLIT-ISOLATED GRAPHS
    # ----------------------------------------------------------

    graph_by_split = {}

    for split_name, idx in [
        ("train", train_idx),
        ("validation", val_idx),
        ("test", test_idx),
    ]:

        allowed = set(
            cell_ids[i]
            for i in idx
        )

        sub_edges = edges[
            edges["source_cell"].astype(str).isin(
                allowed
            )
            &
            edges["target_cell"].astype(str).isin(
                allowed
            )
        ].copy()

        local_ids = [
            cell_ids[i]
            for i in idx
        ]

        local_map = {
            cid: j
            for j, cid in enumerate(local_ids)
        }

        A_sym, n_edges = build_graph(
            sub_edges,
            local_map,
            allowed,
            normalization="symmetric",
        )

        A_row, _ = build_graph(
            sub_edges,
            local_map,
            allowed,
            normalization="row",
        )

        graph_by_split[split_name] = {
            "indices": idx,
            "symmetric": A_sym,
            "row": A_row,
            "edges": n_edges,
        }

        print(
            f"{split_name} graph: "
            f"{n_edges} original edges; "
            f"{A_sym._nnz()} adjacency entries"
        )

    # ----------------------------------------------------------
    # GLOBAL BLOCK-DIAGONAL ADJACENCY
    # ----------------------------------------------------------

    def assemble_global_adjacency(
        normalization
    ):

        rows = []
        cols = []
        vals = []

        for split_name in [
            "train",
            "validation",
            "test",
        ]:

            idx = graph_by_split[
                split_name
            ]["indices"]

            A = graph_by_split[
                split_name
            ][normalization]

            if A._nnz() == 0:
                continue

            local = A.coalesce()

            r = (
                local.indices()[0]
                .cpu()
                .numpy()
            )

            c = (
                local.indices()[1]
                .cpu()
                .numpy()
            )

            v = (
                local.values()
                .cpu()
                .numpy()
            )

            rows.append(idx[r])
            cols.append(idx[c])
            vals.append(v)

        if not rows:

            return torch.sparse_coo_tensor(
                torch.empty(
                    (2, 0),
                    dtype=torch.long,
                ),
                torch.empty(
                    (0,),
                    dtype=torch.float32,
                ),
                size=(
                    len(merged),
                    len(merged),
                ),
            ).coalesce()

        return torch.sparse_coo_tensor(
            np.vstack([
                np.concatenate(rows),
                np.concatenate(cols),
            ]),
            np.concatenate(vals).astype(
                np.float32
            ),
            size=(
                len(merged),
                len(merged),
            ),
        ).coalesce()

    # ----------------------------------------------------------
    # FINAL GRAPH ISOLATION AUDIT
    # ----------------------------------------------------------
    #
    # Every model adjacency is assembled from the three
    # split-specific graphs above. Therefore the final
    # block-diagonal adjacency contains no cross-split edges.
    #

    retained_split_edges = int(
        sum(
            graph_by_split[name]["edges"]
            for name in [
                "train",
                "validation",
                "test",
            ]
        )
    )

    if (
        retained_split_edges
        != raw_within_split_edges
    ):
        raise RuntimeError(
            "Split graph construction mismatch: "
            f"raw within-split edges="
            f"{raw_within_split_edges}, "
            f"retained="
            f"{retained_split_edges}"
        )

    print(
        "Split-isolation audit: PASS; "
        f"raw_cross_split_edges="
        f"{raw_cross_split_edges}; "
        f"retained_within_split_edges="
        f"{retained_split_edges}"
    )

    # ----------------------------------------------------------
    # DIAGNOSTIC VARIANTS
    # ----------------------------------------------------------

    variants = [
        {
            "name": "no_graph",
            "graph": False,
            "smoothness": False,
            "normalization": "symmetric",
        },
        {
            "name": "graph_no_smoothness",
            "graph": True,
            "smoothness": False,
            "normalization": "symmetric",
        },
        {
            "name": "graph_current",
            "graph": True,
            "smoothness": True,
            "normalization": "symmetric",
        },
        {
            "name": "graph_row_normalized",
            "graph": True,
            "smoothness": True,
            "normalization": "row",
        },
    ]

    all_metrics = []
    all_history = []

    for variant in variants:

        print(
            "\n=================================================="
        )

        print(
            f"Running 08D variant: {variant['name']}"
        )

        print(
            "=================================================="
        )

        adj = assemble_global_adjacency(
            variant["normalization"]
        )

        (
            metrics,
            history,
            evi,
            latent,
            model,
        ) = train_variant(
            X=X,
            program_target=program_target,
            exhaustion_target=exhaustion_scaled,
            activation_target=activation_scaled,
            adj=adj,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            variant=variant["name"],
            use_graph=variant["graph"],
            use_smoothness=variant["smoothness"],
            epochs=args.epochs,
            patience=args.patience,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim,
            dropout=args.dropout,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
        )

        metrics["normalization"] = (
            variant["normalization"]
        )

        all_metrics.append(metrics)
        all_history.append(history)

        score_df = merged[
            [
                "cell_id",
                "sample_id",
                "subtype",
                "celltype_subset",
            ]
        ].copy()

        score_df["split"] = split
        score_df["deep_evi"] = evi
        score_df["exhaustion_target"] = exhaustion
        score_df["activation_target"] = activation

        score_df.to_csv(
            outdir
            / (
                "GSE176078_08D_"
                f"{variant['name']}_scores.csv"
            ),
            index=False,
        )

        torch.save(
            model.state_dict(),
            outdir
            / (
                "GSE176078_08D_"
                f"{variant['name']}_model.pt"
            ),
        )

    metrics_df = pd.DataFrame(
        all_metrics
    )

    history_df = pd.concat(
        all_history,
        ignore_index=True,
    )

    metrics_df.to_csv(
        outdir
        / "GSE176078_08D_graph_diagnostic_metrics.csv",
        index=False,
    )

    history_df.to_csv(
        outdir
        / "GSE176078_08D_training_history.csv",
        index=False,
    )

    # ----------------------------------------------------------
    # REPORT
    # ----------------------------------------------------------

    report = {
        "cohort": "GSE176078",
        "step": "08D_deep_evi_graph_architecture_diagnostic",
        "status": "complete",
        "cells": int(len(merged)),
        "samples": int(
            merged["sample_id"].nunique()
        ),

        "train_cells": int(len(train_idx)),
        "validation_cells": int(len(val_idx)),
        "test_cells": int(len(test_idx)),

        "variants": [
            {
                "name": "no_graph",
                "graph": False,
                "smoothness": False,
                "normalization": "not_applicable",
            },
            {
                "name": "graph_no_smoothness",
                "graph": True,
                "smoothness": False,
                "normalization": "symmetric",
            },
            {
                "name": "graph_current",
                "graph": True,
                "smoothness": True,
                "normalization": "symmetric",
            },
            {
                "name": "graph_row_normalized",
                "graph": True,
                "smoothness": True,
                "normalization": "row",
            },
        ],

        "sample_split_reused": True,

        "leakage_controls": {
            "sample_level_split": True,
            "same_08A_split_manifest": True,
            "cross_split_graph_edges_removed": True,
            "raw_cross_split_edges_expected_and_removed": True,
            "raw_cross_split_edges": raw_cross_split_edges,
            "raw_within_split_edges": raw_within_split_edges,
            "retained_within_split_edges": retained_split_edges,
            "final_model_graph_cross_split_edges": 0,
            "feature_scaling_training_only": True,
            "target_scaling_training_only": True,
            "training_smoothness_edges_only": True,
            "test_used_for_model_selection": False,
            "evi_orientation_training_only": True,
        },

        "data_provenance": {
            "curated_tcell_population": (
                "Step 8A score table"
            ),
            "cell_alignment_key": "cell_id",
            "curated_tcell_count": 35214,
            "celltype_major_required": False,
        },

        "scientific_purpose": (
            "08D isolates whether the performance "
            "difference observed in 08C is attributable "
            "to graph message passing, the graph "
            "smoothness regularizer, or adjacency "
            "normalization."
        ),

        "interpretation_policy": (
            "This diagnostic does not establish RNA "
            "velocity or independent biological validation. "
            "It evaluates the contribution of graph "
            "architecture to a learned exhaustion-associated "
            "T-cell state index."
        ),
    }

    with open(
        outdir / "GSE176078_08D_report.json",
        "w",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    print(
        "\n=================================================="
    )

    print(
        "08D completed successfully."
    )

    print(
        "=================================================="
    )

    print(
        metrics_df[
            [
                "variant",
                "test_evi_exhaustion_spearman",
                "test_evi_exhaustion_pearson",
                "test_evi_activation_spearman",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()