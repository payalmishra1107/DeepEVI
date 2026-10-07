#!/usr/bin/env python3

"""
10A-5B OVERLAP-CONTROLLED STATE-AXIS SENSITIVITY

Uses the frozen 9,864-cell held-out GSE176078 H5AD expression matrix.

The Deep-EVI model is NOT retrained.

The analysis tests whether Deep-EVI's association with a
progenitor-to-terminal exhaustion state axis persists after removing
genes directly used in the Deep-EVI construction from the external
reference signatures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score


# ============================================================================
# Exact Step 7B Deep-EVI construction programs
# ============================================================================

DEEP_EVI_PROGRAMS = {
    "tcell_identity": [
        "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2"
    ],
    "cd8_cytotoxic": [
        "CD8A", "CD8B", "NKG7", "GNLY",
        "GZMB", "GZMH", "PRF1", "CTSW"
    ],
    "treg": [
        "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IL7R"
    ],
    "tfh": [
        "CXCL13", "PDCD1", "ICOS", "CXCR5"
    ],
    "activation_effector": [
        "IFNG", "TNF", "IL2", "CD69", "HLA-DRA", "HLA-DRB1"
    ],
    "exhaustion_dysfunction": [
        "PDCD1", "LAG3", "TIGIT", "HAVCR2",
        "CTLA4", "TOX", "TOX2", "ENTPD1", "CXCL13"
    ],
}


# ============================================================================
# Original external reference signatures
# ============================================================================

REFERENCE_SIGNATURES = {
    "SADE_FELDMAN_DYSFUNCTIONAL": [
        "LAG3", "PDCD1", "HAVCR2",
        "TIGIT", "CD38", "ENTPD1"
    ],

    "VAN_DER_LEUN_22": [
        "LAYN", "ITGAE", "PDCD1", "CTLA4",
        "HAVCR2", "LAG3", "TIGIT", "CXCL13",
        "CD38", "ENTPD1", "CDK1", "HSPH1",
        "CCNB1", "HSPB1", "CDK4", "GZMB",
        "TOX", "IFNG", "MIR155HG", "TNFRSF9",
        "RB1"
    ],

    "PROGENITOR_TPEX": [
        "TCF7", "SLAMF6", "CXCR5",
        "CCR7", "SELL", "IL7R"
    ],

    "TERMINAL_EXHAUSTION": [
        "HAVCR2", "LAG3", "PDCD1",
        "TOX", "ENTPD1", "CXCL13", "TIGIT"
    ],

    "CYTOTOXIC_CONTROL": [
        "GZMA", "GZMB", "GZMH",
        "PRF1", "GNLY", "NKG7", "CTSW"
    ],

    "MEMORY_PROGENITOR": [
        "TCF7", "IL7R", "CCR7",
        "SELL", "LEF1", "MALAT1"
    ],
}


# ============================================================================
# Utilities
# ============================================================================

def clean_gene(gene):
    return str(gene).strip().upper()


def zscore(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)

    if not np.isfinite(std) or std < 1e-12:
        return np.zeros_like(values)

    return (values - mean) / std


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)

    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return {
            "n": int(len(x)),
            "spearman_rho": np.nan,
            "spearman_p": np.nan,
            "pearson_r": np.nan,
            "pearson_p": np.nan,
        }

    sr = spearmanr(x, y)
    pr = pearsonr(x, y)

    return {
        "n": int(len(x)),
        "spearman_rho": float(sr.statistic),
        "spearman_p": float(sr.pvalue),
        "pearson_r": float(pr.statistic),
        "pearson_p": float(pr.pvalue),
    }


def partial_spearman(x, y, control):
    """
    Partial Spearman correlation using rank transformation followed
    by residualization against the control variable.
    """

    x = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    y = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    c = pd.Series(control).rank(method="average").to_numpy(dtype=float)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(c)
    )

    x = x[mask]
    y = y[mask]
    c = c[mask]

    if len(x) < 4:
        return {
            "n": int(len(x)),
            "partial_spearman_rho": np.nan,
            "partial_spearman_p": np.nan,
        }

    design = np.column_stack([
        np.ones(len(c)),
        c,
    ])

    bx = np.linalg.lstsq(
        design,
        x,
        rcond=None,
    )[0]

    by = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )[0]

    rx = x - design @ bx
    ry = y - design @ by

    result = pearsonr(rx, ry)

    return {
        "n": int(len(x)),
        "partial_spearman_rho": float(result.statistic),
        "partial_spearman_p": float(result.pvalue),
    }


def safe_auc(y_true, score):
    y_true = np.asarray(y_true)
    score = np.asarray(score, dtype=float)

    mask = np.isfinite(score)

    y_true = y_true[mask]
    score = score[mask]

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(
        roc_auc_score(
            y_true,
            score,
        )
    )


def build_controlled_signature(
    signature,
    deep_evi_union,
):
    signature = [
        clean_gene(g)
        for g in signature
    ]

    deep_evi_union = set(
        clean_gene(g)
        for g in deep_evi_union
    )

    retained = [
        g for g in signature
        if g not in deep_evi_union
    ]

    removed = [
        g for g in signature
        if g in deep_evi_union
    ]

    return retained, removed


def expression_score(
    adata,
    gene_indices,
    chunk_size=512,
):
    """
    Mean expression across selected genes from an H5AD.

    Reads observations in chunks so that a dense-backed H5AD does not
    require materializing the complete 9,864 x 29,733 matrix at once.
    """

    n_cells = adata.n_obs
    scores = np.empty(
        n_cells,
        dtype=np.float64,
    )

    for start in range(
        0,
        n_cells,
        chunk_size,
    ):
        stop = min(
            start + chunk_size,
            n_cells,
        )

        block = adata[
            start:stop,
            gene_indices,
        ].X

        if hasattr(block, "toarray"):
            block = block.toarray()

        block = np.asarray(
            block,
            dtype=np.float64,
        )

        if block.ndim != 2:
            raise ValueError(
                "Unexpected expression block dimensionality: "
                f"{block.shape}"
            )

        scores[start:stop] = np.mean(
            block,
            axis=1,
        )

    return scores


# ============================================================================
# Main
# ============================================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--expression",
        required=True,
    )

    parser.add_argument(
        "--deep-evi",
        required=True,
    )

    parser.add_argument(
        "--state",
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )

    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print("10A-5B OVERLAP-CONTROLLED STATE-AXIS SENSITIVITY")
    print("=" * 80)

    # ========================================================================
    # 1. Deep-EVI construction union
    # ========================================================================

    deep_evi_union = sorted({
        clean_gene(g)
        for genes in DEEP_EVI_PROGRAMS.values()
        for g in genes
    })

    print(
        f"Deep-EVI construction union: "
        f"{len(deep_evi_union)} genes"
    )

    # ========================================================================
    # 2. Load frozen held-out H5AD
    # ========================================================================

    print("Loading frozen held-out H5AD...")

    adata = ad.read_h5ad(
        args.expression,
        backed="r",
    )

    print(
        f"Expression cells: {adata.n_obs}"
    )

    print(
        f"Expression genes: {adata.n_vars}"
    )

    if adata.n_obs != 9864:
        raise ValueError(
            f"Expected 9864 held-out cells; "
            f"found {adata.n_obs}"
        )

    # ------------------------------------------------------------------------
    # Gene names
    # ------------------------------------------------------------------------

    var_names = [
        clean_gene(g)
        for g in adata.var_names
    ]

    if len(set(var_names)) != len(var_names):
        raise ValueError(
            "Expression H5AD contains duplicate gene names."
        )

    gene_lookup = {
        gene: i
        for i, gene in enumerate(var_names)
    }

    # ------------------------------------------------------------------------
    # Cell IDs
    # ------------------------------------------------------------------------

    cell_ids = [
        str(x)
        for x in adata.obs_names
    ]

    if len(set(cell_ids)) != len(cell_ids):
        raise ValueError(
            "Expression H5AD contains duplicate cell IDs."
        )

    # ------------------------------------------------------------------------
    # Read expression into memory
    #
    # 9,864 x 29,733 is approximately 293M entries.
    # The matrix is expected to be sparse.
    # ------------------------------------------------------------------------

    X = adata.X

    if sparse.issparse(X):
        X = X.tocsr()
    else:
        X = np.asarray(X)

    print(
        f"Sparse expression matrix: "
        f"{sparse.issparse(X)}"
    )

    # ========================================================================
    # 3. Load Deep-EVI
    # ========================================================================

    deep = pd.read_csv(
        args.deep_evi
    )

    deep["cell_id"] = (
        deep["cell_id"]
        .astype(str)
    )

    required_deep = [
        "cell_id",
        "split",
        "deep_evi_raw",
    ]

    missing = [
        c for c in required_deep
        if c not in deep.columns
    ]

    if missing:
        raise ValueError(
            f"Deep-EVI table missing columns: {missing}"
        )

    split_counts = (
        deep["split"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    print(
        "Deep-EVI split counts:"
    )

    for key, value in split_counts.items():
        print(
            f"  {key}: {value}"
        )

    test_deep = deep.loc[
        deep["split"].astype(str).str.lower()
        == "test"
    ].copy()

    if len(test_deep) != 9864:
        raise ValueError(
            f"Expected 9864 frozen test Deep-EVI rows; "
            f"found {len(test_deep)}"
        )

    # ========================================================================
    # 4. Load Step 7B state scores
    # ========================================================================

    state = pd.read_csv(
        args.state
    )

    state["cell_id"] = (
        state["cell_id"]
        .astype(str)
    )

    required_state = [
        "cell_id",
        "exhaustion_dysfunction_score",
        "tcell_identity_score",
        "cd8_cytotoxic_score",
        "treg_score",
        "tfh_score",
        "activation_effector_score",
    ]

    missing = [
        c for c in required_state
        if c not in state.columns
    ]

    if missing:
        raise ValueError(
            f"Step 7B state table missing columns: {missing}"
        )

    # ========================================================================
    # 5. Strict cell alignment
    # ========================================================================

    test_ids = set(
        test_deep["cell_id"]
    )

    expression_ids = set(
        cell_ids
    )

    state_ids = set(
        state["cell_id"]
    )

    if test_ids != expression_ids:
        missing_expression = (
            test_ids - expression_ids
        )

        extra_expression = (
            expression_ids - test_ids
        )

        raise ValueError(
            "Frozen Deep-EVI test IDs do not exactly match "
            "the held-out H5AD IDs. "
            f"Missing expression IDs: {len(missing_expression)}; "
            f"extra expression IDs: {len(extra_expression)}"
        )

    if not test_ids.issubset(state_ids):
        raise ValueError(
            "Some frozen test cells are absent "
            "from Step 7B state scores."
        )

    # ========================================================================
    # 6. Align Deep-EVI metadata
    # ========================================================================

    metadata_columns = [
        "cell_id",
        "deep_evi_raw",
    ]

    for column in [
        "sample_id",
        "subtype",
        "celltype_subset",
    ]:
        if column in test_deep.columns:
            metadata_columns.append(
                column
            )

    deep_test = (
        test_deep[
            metadata_columns
        ]
        .set_index("cell_id")
        .loc[cell_ids]
        .reset_index()
    )

    if not np.array_equal(
        deep_test["cell_id"].to_numpy(),
        np.asarray(cell_ids),
    ):
        raise ValueError(
            "Deep-EVI test cells could not be aligned "
            "to the H5AD order."
        )

    state_test = (
        state[
            required_state
        ]
        .set_index("cell_id")
        .loc[cell_ids]
        .reset_index()
    )

    # ========================================================================
    # 7. Build analysis table
    # ========================================================================

    merged = deep_test.merge(
        state_test,
        on="cell_id",
        how="inner",
        validate="one_to_one",
    )

    if len(merged) != 9864:
        raise ValueError(
            f"Expected 9864 aligned rows; "
            f"found {len(merged)}"
        )

    print()
    print("TEST-SET ALIGNMENT")
    print(
        f"Frozen test cells: {len(test_deep)}"
    )
    print(
        f"H5AD cells:        {len(cell_ids)}"
    )
    print(
        f"Aligned cells:     {len(merged)}"
    )
    print(
        "Alignment: PASS"
    )

    # ========================================================================
    # 8. Construct original and overlap-controlled signatures
    # ========================================================================

    signature_inventory = []
    controlled_signatures = {}

    for name, genes in REFERENCE_SIGNATURES.items():

        original = [
            clean_gene(g)
            for g in genes
        ]

        controlled, removed = (
            build_controlled_signature(
                original,
                deep_evi_union,
            )
        )

        controlled_signatures[name] = controlled

        signature_inventory.append(
            {
                "signature": name,
                "version": "original",
                "n_requested": len(original),
                "n_retained": len(original),
                "n_removed_deep_evi_overlap": 0,
                "genes": ";".join(original),
                "removed_genes": "",
                "overlap_controlled": False,
            }
        )

        signature_inventory.append(
            {
                "signature": name,
                "version": "overlap_controlled",
                "n_requested": len(original),
                "n_retained": len(controlled),
                "n_removed_deep_evi_overlap": len(removed),
                "genes": ";".join(controlled),
                "removed_genes": ";".join(removed),
                "overlap_controlled": True,
            }
        )

        print()
        print(
            f"{name}:"
        )
        print(
            f"  Original: {len(original)}"
        )
        print(
            f"  Controlled: {len(controlled)}"
        )
        print(
            f"  Removed: {', '.join(removed) if removed else 'NONE'}"
        )

    signature_inventory_df = pd.DataFrame(
        signature_inventory
    )

    signature_inventory_df.to_csv(
        out
        / "GSE176078_10A5B_signature_inventory.csv",
        index=False,
    )

    # ========================================================================
    # 9. Score expression references
    # ========================================================================

    score_columns = {}

    for name, genes in REFERENCE_SIGNATURES.items():

        original_genes = [
            clean_gene(g)
            for g in genes
        ]

        missing_original = [
            g
            for g in original_genes
            if g not in gene_lookup
        ]

        if missing_original:
            raise ValueError(
                f"{name} original signature has "
                f"missing genes in H5AD: "
                f"{missing_original}"
            )

        original_indices = [
            gene_lookup[g]
            for g in original_genes
        ]

        original_score = expression_score(
            adata,
            original_indices,
        )

        original_column = (
            f"{name}__original"
        )

        score_columns[
            original_column
        ] = original_score

        controlled_genes = (
            controlled_signatures[name]
        )

        if len(controlled_genes) < 2:

            score_columns[
                f"{name}__overlap_controlled"
            ] = np.full(
                len(cell_ids),
                np.nan,
            )

            continue

        missing_controlled = [
            g
            for g in controlled_genes
            if g not in gene_lookup
        ]

        if missing_controlled:
            raise ValueError(
                f"{name} controlled signature has "
                f"missing genes: {missing_controlled}"
            )

        controlled_indices = [
            gene_lookup[g]
            for g in controlled_genes
        ]

        controlled_score = expression_score(
            adata,
            controlled_indices,
        )

        score_columns[
            f"{name}__overlap_controlled"
        ] = controlled_score

    for column, values in score_columns.items():
        merged[column] = values

    # ========================================================================
    # 10. State axes
    #
    # The original 7-gene terminal exhaustion signature is retained for
    # comparison, but its overlap-controlled version has zero genes because
    # all seven genes were directly used in the Deep-EVI construction.
    #
    # Therefore:
    #   - original axis = original terminal - original progenitor
    #   - controlled axis = controlled Van der Leun - controlled progenitor
    #
    # The controlled Van der Leun construct is explicitly a derived
    # overlap-controlled sensitivity reference, not a new published signature.
    # ========================================================================

    merged[
        "terminal_exhaustion_original_z"
    ] = zscore(
        merged[
            "TERMINAL_EXHAUSTION__original"
        ]
    )

    merged[
        "progenitor_tpex_original_z"
    ] = zscore(
        merged[
            "PROGENITOR_TPEX__original"
        ]
    )

    merged[
        "state_axis_original"
    ] = (
        merged[
            "terminal_exhaustion_original_z"
        ]
        -
        merged[
            "progenitor_tpex_original_z"
        ]
    )

    # ------------------------------------------------------------------------
    # Controlled state axis
    #
    # Use the non-empty overlap-controlled Van der Leun reference as the
    # terminal/dysfunction component.
    # ------------------------------------------------------------------------

    merged[
        "controlled_terminal_reference_z"
    ] = zscore(
        merged[
            "VAN_DER_LEUN_22__overlap_controlled"
        ]
    )

    merged[
        "controlled_progenitor_tpex_z"
    ] = zscore(
        merged[
            "PROGENITOR_TPEX__overlap_controlled"
        ]
    )

    merged[
        "state_axis_controlled"
    ] = (
        merged[
            "controlled_terminal_reference_z"
        ]
        -
        merged[
            "controlled_progenitor_tpex_z"
        ]
    )

    # ========================================================================
    # 11. Main correlations
    # ========================================================================

    targets = {
        "original_state_axis":
            merged["state_axis_original"],

        "controlled_state_axis":
            merged["state_axis_controlled"],

        "original_terminal_exhaustion":
            merged[
                "TERMINAL_EXHAUSTION__original"
            ],

        "controlled_terminal_reference":
            merged[
                "VAN_DER_LEUN_22__overlap_controlled"
            ],

        "original_progenitor_tpex":
            merged[
                "PROGENITOR_TPEX__original"
            ],

        "controlled_progenitor_tpex":
            merged[
                "PROGENITOR_TPEX__overlap_controlled"
            ],

        "direct_exhaustion_training_target":
            merged[
                "exhaustion_dysfunction_score"
            ],
    }

    association_rows = []

    for target_name, values in targets.items():

        result = safe_corr(
            merged["deep_evi_raw"],
            values,
        )

        association_rows.append(
            {
                "representation": "Deep-EVI",
                "target": target_name,
                **result,
            }
        )

    association_df = pd.DataFrame(
        association_rows
    )

    association_df.to_csv(
        out
        / "GSE176078_10A5B_overlap_controlled_associations.csv",
        index=False,
    )

    # ========================================================================
    # 12. Partial correlations
    # ========================================================================

    partial_specs = [
        (
            "original_state_axis_control_terminal",
            merged["state_axis_original"],
            merged[
                "TERMINAL_EXHAUSTION__original"
            ],
        ),
        (
            "original_state_axis_control_progenitor",
            merged["state_axis_original"],
            merged[
                "PROGENITOR_TPEX__original"
            ],
        ),
        (
            "controlled_state_axis_control_terminal_reference",
            merged["state_axis_controlled"],
            merged[
                "VAN_DER_LEUN_22__overlap_controlled"
            ],
        ),
        (
            "controlled_state_axis_control_progenitor",
            merged["state_axis_controlled"],
            merged[
                "PROGENITOR_TPEX__overlap_controlled"
            ],
        ),
    ]

    partial_rows = []

    for name, axis, control in partial_specs:

        result = partial_spearman(
            merged["deep_evi_raw"],
            axis,
            control,
        )

        partial_rows.append(
            {
                "analysis": name,
                **result,
            }
        )

    partial_df = pd.DataFrame(
        partial_rows
    )