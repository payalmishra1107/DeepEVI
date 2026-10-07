#!/usr/bin/env python3

import argparse
import json
import os
import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--celltype-key", default="celltype_major")
    p.add_argument("--tcell-label", default="T-cells")
    p.add_argument("--sample-key", default="sample_id")
    p.add_argument("--subtype-key", default="subtype")
    p.add_argument("--subset-key", default="celltype_subset")
    p.add_argument("--min-program-genes", type=int, default=3)
    return p.parse_args()


def clean_series(s):
    return s.astype("string").fillna("NA").str.strip()


def gene_lookup(var_names):
    return {str(g).upper(): i for i, g in enumerate(var_names)}


def available_genes(var_names, genes):
    lookup = gene_lookup(var_names)
    found = []
    missing = []

    for g in genes:
        gu = g.upper()
        if gu in lookup:
            found.append((g, lookup[gu]))
        else:
            missing.append(g)

    return found, missing


def get_matrix(adata, indices):
    X = adata.X[indices]

    if sparse.issparse(X):
        return X.toarray().astype(np.float32)

    return np.asarray(X, dtype=np.float32)


def zscore(values):
    values = np.asarray(values, dtype=np.float32)

    mu = np.nanmean(values)
    sd = np.nanstd(values)

    if not np.isfinite(sd) or sd == 0:
        return np.zeros_like(values, dtype=np.float32)

    return ((values - mu) / sd).astype(np.float32)


def score_program(X, gene_indices):
    if not gene_indices:
        return None

    vals = X[:, gene_indices].mean(axis=1)
    return np.asarray(vals, dtype=np.float32)


def safe_group_summary(df, score_columns, group_key):
    rows = []

    for group, g in df.groupby(group_key, dropna=False):
        row = {group_key: str(group), "n_cells": int(len(g))}

        for col in score_columns:
            vals = pd.to_numeric(g[col], errors="coerce")
            row[f"{col}_mean"] = float(vals.mean())
            row[f"{col}_median"] = float(vals.median())
            row[f"{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    args = parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 7B — T-CELL STATE / EXHAUSTION PROGRAM ANALYSIS")
    print("=" * 80)

    adata = ad.read_h5ad(args.input)

    print(f"Input: {adata.n_obs} cells × {adata.n_vars} genes")

    required = [
        args.celltype_key,
        args.sample_key,
        args.subtype_key,
        args.subset_key,
    ]

    missing_keys = [k for k in required if k not in adata.obs.columns]

    if missing_keys:
        raise ValueError(f"Missing required obs keys: {missing_keys}")

    if "X_pca_pre_harmony" not in adata.obsm:
        raise ValueError("Missing X_pca_pre_harmony")

    if "X_harmony" not in adata.obsm:
        raise ValueError("Missing X_harmony")

    # ------------------------------------------------------------------
    # 1. Curated T-cell compartment
    # ------------------------------------------------------------------

    major = clean_series(adata.obs[args.celltype_key])

    tmask = major.eq(args.tcell_label).to_numpy()

    if not tmask.any():
        raise ValueError(
            f"No cells found with {args.celltype_key} == {args.tcell_label}"
        )

    t_idx = np.flatnonzero(tmask)

    tdata = adata[tmask].copy()

    print(f"Curated T cells: {len(tdata)}")

    # ------------------------------------------------------------------
    # 2. Document Step 7A discrepancy
    # ------------------------------------------------------------------

    regex_patterns = [
        r"\bt[\s_-]*cells?\b",
        r"\bt[\s_-]*lymph",
        r"\bcd4\b",
        r"\bcd8\b",
        r"\btreg\b",
        r"\btcr\b",
        r"\bcytotoxic[\s_-]*t\b",
    ]

    regex = re.compile("|".join(regex_patterns), flags=re.IGNORECASE)

    regex_mask = major.str.contains(regex, na=False).to_numpy()

    discrepancy = {
        "curated_major_t_cells": int(tmask.sum()),
        "regex_detected_t_cells": int(regex_mask.sum()),
        "difference": int(tmask.sum() - regex_mask.sum()),
        "policy": (
            "celltype_major == T-cells is the primary curated T-cell compartment; "
            "regex detection is retained only as a sensitivity check."
        ),
    }

    with open(out / "tcell_compartment_definition.json", "w") as f:
        json.dump(discrepancy, f, indent=2)

    # ------------------------------------------------------------------
    # 3. Define expression programs
    # ------------------------------------------------------------------

    programs = {
        "tcell_identity": [
            "CD3D",
            "CD3E",
            "CD3G",
            "TRBC1",
            "TRBC2",
        ],
        "cd8_cytotoxic": [
            "CD8A",
            "CD8B",
            "NKG7",
            "GNLY",
            "GZMB",
            "GZMH",
            "PRF1",
            "CTSW",
        ],
        "treg": [
            "FOXP3",
            "IL2RA",
            "CTLA4",
            "TIGIT",
            "IL7R",
        ],
        "tfh": [
            "CXCL13",
            "PDCD1",
            "ICOS",
            "CXCR5",
        ],
        "activation_effector": [
            "IFNG",
            "TNF",
            "IL2",
            "CD69",
            "HLA-DRA",
            "HLA-DRB1",
        ],
        "exhaustion_dysfunction": [
            "PDCD1",
            "LAG3",
            "TIGIT",
            "HAVCR2",
            "CTLA4",
            "TOX",
            "TOX2",
            "ENTPD1",
            "CXCL13",
        ],
    }

    lookup = gene_lookup(tdata.var_names)

    program_inventory = {}
    usable_programs = {}

    for name, genes in programs.items():
        found = []
        missing = []

        for gene in genes:
            gu = gene.upper()

            if gu in lookup:
                found.append(gene)
            else:
                missing.append(gene)

        program_inventory[name] = {
            "requested_genes": genes,
            "available_genes": found,
            "missing_genes": missing,
            "n_available": len(found),
        }

        if len(found) >= args.min_program_genes:
            usable_programs[name] = found

    with open(out / "program_gene_inventory.json", "w") as f:
        json.dump(program_inventory, f, indent=2)

    # ------------------------------------------------------------------
    # 4. Calculate expression scores
    # ------------------------------------------------------------------

    X = get_matrix(tdata, np.arange(tdata.n_obs))

    score_columns = []

    for program_name, genes in usable_programs.items():

        indices = [lookup[g.upper()] for g in genes]

        raw = score_program(X, indices)

        raw_name = f"{program_name}_score"
        z_name = f"{program_name}_z"

        tdata.obs[raw_name] = raw
        tdata.obs[z_name] = zscore(raw)

        score_columns.append(raw_name)

    if "exhaustion_dysfunction_score" not in tdata.obs.columns:
        raise ValueError(
            "Exhaustion/dysfunction program has fewer than the required "
            f"{args.min_program_genes} genes available."
        )

    # ------------------------------------------------------------------
    # 5. Add curated annotations
    # ------------------------------------------------------------------

    for key in [
        args.sample_key,
        args.subtype_key,
        args.celltype_key,
        args.subset_key,
    ]:
        tdata.obs[key] = clean_series(tdata.obs[key])

    # ------------------------------------------------------------------
    # 6. Save cell-level T-cell state table
    # ------------------------------------------------------------------

    cell_table = tdata.obs[
        [
            args.sample_key,
            args.subtype_key,
            args.celltype_key,
            args.subset_key,
        ]
        + [c for c in tdata.obs.columns if c.endswith("_score")]
        + [c for c in tdata.obs.columns if c.endswith("_z")]
    ].copy()

    cell_table.insert(0, "cell_id", tdata.obs_names.astype(str))

    cell_table.to_csv(
        out / "tcell_expression_program_scores.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 7. Sample-level summaries
    # ------------------------------------------------------------------

    sample_df = safe_group_summary(
        cell_table,
        score_columns,
        args.sample_key,
    )

    sample_df.to_csv(
        out / "tcell_programs_by_sample.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 8. Subtype-level summaries
    # ------------------------------------------------------------------

    subtype_df = safe_group_summary(
        cell_table,
        score_columns,
        args.subtype_key,
    )

    subtype_df.to_csv(
        out / "tcell_programs_by_subtype.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 9. T-cell subset summaries
    # ------------------------------------------------------------------

    subset_df = safe_group_summary(
        cell_table,
        score_columns,
        args.subset_key,
    )

    subset_df.to_csv(
        out / "tcell_programs_by_annotation.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # 10. Pre-Harmony vs Harmony T-cell embedding inventory
    # ------------------------------------------------------------------

    pre = np.asarray(tdata.obsm["X_pca_pre_harmony"])
    harmony = np.asarray(tdata.obsm["X_harmony"])

    embedding_summary = {
        "t_cells": int(tdata.n_obs),
        "pre_harmony_dimensions": int(pre.shape[1]),
        "harmony_dimensions": int(harmony.shape[1]),
        "pre_harmony_finite": bool(np.isfinite(pre).all()),
        "harmony_finite": bool(np.isfinite(harmony).all()),
    }

    with open(out / "tcell_embedding_inventory.json", "w") as f:
        json.dump(embedding_summary, f, indent=2)

    # ------------------------------------------------------------------
    # 11. Program correlations
    # ------------------------------------------------------------------

    corr_cols = [
        c for c in score_columns
        if c in cell_table.columns
    ]

    corr = cell_table[corr_cols].corr(method="spearman")

    corr.to_csv(
        out / "tcell_program_spearman_correlations.csv"
    )

    # ------------------------------------------------------------------
    # 12. Final report
    # ------------------------------------------------------------------

    report = {
        "cohort": "GSE176078",
        "step": "07B_tcell_expression_program_analysis",
        "input_file": Path(args.input).name,
        "total_cohort_cells": int(adata.n_obs),
        "curated_t_cells": int(tdata.n_obs),
        "curated_t_cell_fraction": float(tdata.n_obs / adata.n_obs),
        "tcell_compartment_definition": discrepancy,
        "programs": program_inventory,
        "score_columns": score_columns,
        "representations": [
            "X_pca_pre_harmony",
            "X_harmony",
        ],
        "scientific_policy": {
            "exhaustion_definition": (
                "Expression-derived multi-gene exhaustion/dysfunction "
                "program score; not an annotation-derived label."
            ),
            "velocity_claim": (
                "No RNA velocity is inferred from these count matrices. "
                "These are expression-state scores only."
            ),
            "single_marker_policy": (
                "Individual markers such as LAG3 or PDCD1 are not treated "
                "as sufficient evidence of exhaustion."
            ),
            "integration_selection": (
                "No representation is automatically selected. "
                "Biological evidence is evaluated jointly."
            ),
        },
        "status": "complete",
    }

    with open(out / "GSE176078_step7b_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=" * 80)
    print("STEP 7B COMPLETE")
    print(f"Curated T cells: {tdata.n_obs}")
    print(f"Usable programs: {len(usable_programs)}")
    print("=" * 80)


if __name__ == "__main__":
    main()