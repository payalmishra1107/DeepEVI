#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


DEEPEVI_PROGRAM_GENES = {
    "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2",
    "CD8A", "CD8B", "NKG7", "GNLY", "GZMB", "GZMH", "PRF1", "CTSW",
    "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IL7R",
    "CXCL13", "PDCD1", "ICOS", "CXCR5",
    "IFNG", "TNF", "IL2", "CD69", "HLA-DRA", "HLA-DRB1",
    "LAG3", "HAVCR2", "TOX", "TOX2", "ENTPD1",
}


def load_table(path):
    path = Path(path)

    if not path.exists():
        raise RuntimeError(f"Missing input: {path}")

    if path.name.endswith(".tsv") or path.name.endswith(".tsv.gz"):
        sep = "\t"
    else:
        sep = ","

    df = pd.read_csv(
        path,
        sep=sep,
        dtype=str,
        low_memory=False,
    )

    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_scores(path):
    df = load_table(path)

    required = {
        "case_id",
        "deep_evi_tcga_surrogate",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"09B score table missing required columns: {sorted(missing)}"
        )

    df["case_id"] = df["case_id"].astype(str)

    df["deep_evi_tcga_surrogate"] = pd.to_numeric(
        df["deep_evi_tcga_surrogate"],
        errors="coerce",
    )

    df = df[
        [
            "case_id",
            "deep_evi_tcga_surrogate",
        ]
    ].drop_duplicates("case_id")

    return df


def load_inventory(path):
    df = load_table(path)

    required = {
        "file_id",
        "filename",
        "case_id",
        "submitter_id",
        "genes_found",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"09B case inventory missing required columns: {sorted(missing)}"
        )

    df["file_id"] = df["file_id"].astype(str)
    df["filename"] = df["filename"].astype(str)
    df["case_id"] = df["case_id"].astype(str)
    df["submitter_id"] = df["submitter_id"].astype(str)

    return df


def load_pathways(path):
    df = load_table(path)

    required = {
        "pathway_id",
        "pathway_name",
        "source",
        "source_version",
        "genes",
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Pathway definition missing columns: {sorted(missing)}"
        )

    records = []

    for _, row in df.iterrows():

        raw = str(row["genes"])

        if ";" in raw:
            genes = raw.split(";")
        else:
            genes = raw.split(",")

        genes = sorted(
            {
                g.strip().upper()
                for g in genes
                if g.strip()
            }
        )

        original_genes = genes

        independent_genes = [
            g
            for g in genes
            if g not in DEEPEVI_PROGRAM_GENES
        ]

        records.append(
            {
                "pathway_id": str(row["pathway_id"]),
                "pathway_name": str(row["pathway_name"]),
                "source": str(row["source"]),
                "source_version": str(row["source_version"]),
                "original_genes": original_genes,
                "genes": independent_genes,
                "original_gene_count": len(original_genes),
                "independent_gene_count": len(independent_genes),
                "deepevi_genes_removed": len(
                    set(original_genes) & DEEPEVI_PROGRAM_GENES
                ),
            }
        )

    return records


def locate_star_header(path):
    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        for i, line in enumerate(handle):

            if (
                "gene_name" in line
                and "tpm_unstranded" in line
            ):
                return i

    return None


def read_required_expression(path, required_genes):
    header_idx = locate_star_header(path)

    if header_idx is None:
        raise RuntimeError(
            f"Could not find STAR-Counts header in {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="replace",
    ) as handle:

        lines = handle.readlines()

    header = lines[header_idx].rstrip("\n").split("\t")

    try:
        gene_name_idx = header.index("gene_name")
        tpm_idx = header.index("tpm_unstranded")
    except ValueError as exc:
        raise RuntimeError(
            f"Required STAR columns missing in {path}"
        ) from exc

    rows = []

    for line in lines[header_idx + 1:]:

        line = line.rstrip("\n")

        if not line or line.startswith("#"):
            continue

        fields = line.split("\t")

        if len(fields) <= max(
            gene_name_idx,
            tpm_idx,
        ):
            continue

        gene = fields[gene_name_idx].strip().upper()

        if gene not in required_genes:
            continue

        try:
            tpm = float(fields[tpm_idx])
        except ValueError:
            continue

        if not np.isfinite(tpm):
            continue

        rows.append(
            (
                gene,
                tpm,
            )
        )

    if not rows:
        raise RuntimeError(
            f"No requested genes found in STAR file: {path}"
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "gene",
            "tpm",
        ],
    )

    # Average duplicate gene symbols.
    df = (
        df.groupby("gene", as_index=True)["tpm"]
        .mean()
    )

    return np.log2(df + 1.0)


def score_genes(expression, genes):
    available = [
        gene
        for gene in genes
        if gene in expression.index
    ]

    if not available:
        return np.nan, 0

    values = expression.loc[available]

    return float(values.mean()), len(available)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--scores",
        required=True,
    )

    parser.add_argument(
        "--expression-dir",
        required=True,
    )

    parser.add_argument(
        "--case-inventory",
        required=True,
    )

    parser.add_argument(
        "--clinical",
        required=True,
    )

    parser.add_argument(
        "--pathways",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    expression_dir = Path(args.expression_dir)

    if not expression_dir.exists():
        raise RuntimeError(
            f"Expression directory missing: {expression_dir}"
        )

    scores = load_scores(args.scores)
    inventory = load_inventory(args.case_inventory)
    clinical = load_table(args.clinical)
    pathway_records = load_pathways(args.pathways)

    if "case_id" not in clinical.columns:
        raise RuntimeError(
            "Clinical table missing case_id."
        )

    clinical["case_id"] = clinical["case_id"].astype(str)

    # ------------------------------------------------------------
    # Validate 09B inventory
    # ------------------------------------------------------------

    if inventory["file_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate file_id detected in 09B case inventory."
        )

    if inventory["filename"].duplicated().any():
        raise RuntimeError(
            "Duplicate filename detected in 09B case inventory."
        )

    if inventory["case_id"].isna().any():
        raise RuntimeError(
            "Missing case_id in 09B case inventory."
        )

    # ------------------------------------------------------------
    # Validate physical expression files against inventory
    # ------------------------------------------------------------

    required_files = []

    missing_files = []

    for _, row in inventory.iterrows():

        expected = (
            expression_dir
            / row["file_id"]
            / row["filename"]
        )

        required_files.append(expected)

        if not expected.is_file():
            missing_files.append(str(expected))

    if missing_files:
        raise RuntimeError(
            "Missing expression files referenced by 09B inventory. "
            f"Count={len(missing_files)}; first="
            f"{missing_files[0]}"
        )

    # ------------------------------------------------------------
    # Validate score/inventory relationship
    # ------------------------------------------------------------

    inventory_cases = set(
        inventory["case_id"]
    )

    score_cases = set(
        scores["case_id"]
    )

    missing_score_cases = inventory_cases - score_cases

    if missing_score_cases:
        raise RuntimeError(
            f"{len(missing_score_cases)} inventory cases missing "
            "from 09B score table."
        )

    # ------------------------------------------------------------
    # Clinical overlap
    # ------------------------------------------------------------

    clinical_cases = set(
        clinical["case_id"]
    )

    analysis_cases = (
        score_cases
        & clinical_cases
    )

    if not analysis_cases:
        raise RuntimeError(
            "No TCGA case overlap between score and clinical tables."
        )

    # ------------------------------------------------------------
    # Independent pathway genes
    # ------------------------------------------------------------

    pathway_metadata = []

    required_genes = set()

    for pathway in pathway_records:

        if pathway["independent_gene_count"] == 0:
            raise RuntimeError(
                f"Pathway {pathway['pathway_id']} has zero "
                "independent genes after Deep-EVI gene exclusion."
            )

        required_genes.update(
            pathway["genes"]
        )

        pathway_metadata.append(
            {
                "pathway_id": pathway["pathway_id"],
                "pathway_name": pathway["pathway_name"],
                "source": pathway["source"],
                "source_version": pathway["source_version"],
                "original_gene_count": pathway[
                    "original_gene_count"
                ],
                "independent_gene_count": pathway[
                    "independent_gene_count"
                ],
                "deepevi_genes_removed": pathway[
                    "deepevi_genes_removed"
                ],
            }
        )

    # ------------------------------------------------------------
    # Context genes
    #
    # These are explicitly contextual and NOT independent
    # validation endpoints.
    # ------------------------------------------------------------

    tcell_context_genes = {
        "CD3D",
        "CD3E",
        "CD3G",
        "TRBC1",
        "TRBC2",
    }

    cytotoxic_context_genes = {
        "NKG7",
        "GNLY",
        "GZMB",
        "GZMH",
        "PRF1",
        "CTSW",
    }

    required_genes.update(
        tcell_context_genes
    )

    required_genes.update(
        cytotoxic_context_genes
    )

    # ------------------------------------------------------------
    # Process every expression file
    # ------------------------------------------------------------

    pathway_file_records = []
    context_file_records = []

    processed_files = 0

    for _, row in inventory.iterrows():

        expression_file = (
            expression_dir
            / row["file_id"]
            / row["filename"]
        )

        expression = read_required_expression(
            expression_file,
            required_genes,
        )

        processed_files += 1

        for pathway in pathway_records:

            score, n_available = score_genes(
                expression,
                pathway["genes"],
            )

            pathway_file_records.append(
                {
                    "case_id": row["case_id"],
                    "file_id": row["file_id"],
                    "filename": row["filename"],
                    "pathway_id": pathway["pathway_id"],
                    "pathway_name": pathway["pathway_name"],
                    "source": pathway["source"],
                    "source_version": pathway["source_version"],
                    "pathway_score": score,
                    "genes_available": n_available,
                    "genes_defined_independent": pathway[
                        "independent_gene_count"
                    ],
                }
            )

        tcell_score, tcell_n = score_genes(
            expression,
            sorted(tcell_context_genes),
        )

        cytotoxic_score, cytotoxic_n = score_genes(
            expression,
            sorted(cytotoxic_context_genes),
        )

        context_file_records.append(
            {
                "case_id": row["case_id"],
                "file_id": row["file_id"],
                "filename": row["filename"],
                "tcell_context_score": tcell_score,
                "tcell_context_genes_available": tcell_n,
                "cytotoxic_context_score": cytotoxic_score,
                "cytotoxic_context_genes_available": cytotoxic_n,
            }
        )

    if processed_files != len(inventory):
        raise RuntimeError(
            "Not all 09B inventory expression files were processed."
        )

    pathway_file_df = pd.DataFrame(
        pathway_file_records
    )

    context_file_df = pd.DataFrame(
        context_file_records
    )

    # ------------------------------------------------------------
    # Case-level aggregation
    #
    # Exactly the same arithmetic mean approach as 09B.
    # ------------------------------------------------------------

    pathway_case = (
        pathway_file_df
        .groupby(
            [
                "case_id",
                "pathway_id",
                "pathway_name",
                "source",
                "source_version",
            ],
            as_index=False,
        )
        .agg(
            pathway_score=(
                "pathway_score",
                "mean",
            ),
            n_expression_files=(
                "file_id",
                "nunique",
            ),
            mean_genes_available=(
                "genes_available",
                "mean",
            ),
        )
    )

    context_case = (
        context_file_df
        .groupby(
            "case_id",
            as_index=False,
        )
        .agg(
            tcell_context_score=(
                "tcell_context_score",
                "mean",
            ),
            cytotoxic_context_score=(
                "cytotoxic_context_score",
                "mean",
            ),
            n_expression_files=(
                "file_id",
                "nunique",
            ),
        )
    )

    # ------------------------------------------------------------
    # Pathway correlations
    # ------------------------------------------------------------

    correlations = []

    for pathway_id, group in pathway_case.groupby(
        "pathway_id",
        sort=False,
    ):

        merged = scores.merge(
            group[
                [
                    "case_id",
                    "pathway_score",
                ]
            ],
            on="case_id",
            how="inner",
        )

        merged = merged.dropna(
            subset=[
                "deep_evi_tcga_surrogate",
                "pathway_score",
            ]
        )

        if len(merged) >= 5:

            rho, pvalue = spearmanr(
                merged["deep_evi_tcga_surrogate"],
                merged["pathway_score"],
            )

        else:

            rho = np.nan
            pvalue = np.nan

        info = next(
            x
            for x in pathway_records
            if x["pathway_id"] == pathway_id
        )

        correlations.append(
            {
                "pathway_id": pathway_id,
                "pathway_name": info["pathway_name"],
                "source": info["source"],
                "source_version": info["source_version"],
                "n_cases": len(merged),
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
            }
        )

    pathway_correlations = pd.DataFrame(
        correlations
    )

    # ------------------------------------------------------------
    # Context correlations
    # ------------------------------------------------------------

    context_analysis = scores.merge(
        context_case,
        on="case_id",
        how="inner",
    )

    context_correlations = []

    for variable in [
        "tcell_context_score",
        "cytotoxic_context_score",
    ]:

        subset = context_analysis.dropna(
            subset=[
                "deep_evi_tcga_surrogate",
                variable,
            ]
        )

        if len(subset) >= 5:

            rho, pvalue = spearmanr(
                subset["deep_evi_tcga_surrogate"],
                subset[variable],
            )

        else:

            rho = np.nan
            pvalue = np.nan

        context_correlations.append(
            {
                "variable": variable,
                "n_cases": len(subset),
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                "interpretation": (
                    "contextual_immune_association"
                ),
            }
        )

    context_correlations = pd.DataFrame(
        context_correlations
    )

    # ------------------------------------------------------------
    # Analysis cohort
    # ------------------------------------------------------------

    analysis_cohort = (
        scores[
            [
                "case_id",
                "deep_evi_tcga_surrogate",
            ]
        ]
        .merge(
            clinical,
            on="case_id",
            how="inner",
        )
    )

    # ------------------------------------------------------------
    # Report
    # ------------------------------------------------------------

    multi_file_cases = int(
        (
            inventory
            .groupby("case_id")
            .size()
            .gt(1)
        )
        .sum()
    )

    report = {
        "cohort": "TCGA-BRCA",

        "step": (
            "09E_TCGA_biological_concordance_"
            "construct_validation"
        ),

        "status": "complete",

        "patient_level_unit": True,

        "score_cases": int(
            scores["case_id"].nunique()
        ),

        "clinical_cases": int(
            clinical["case_id"].nunique()
        ),

        "analysis_cases": int(
            analysis_cohort["case_id"].nunique()
        ),

        "expression_files": int(
            len(inventory)
        ),

        "processed_expression_files": int(
            processed_files
        ),

        "multi_file_cases": multi_file_cases,

        "validation_pathways": len(
            pathway_records
        ),

        "deep_evi_program_gene_count": len(
            DEEPEVI_PROGRAM_GENES
        ),

        "deep_evi_genes_removed_from_validation": True,

        "deep_evi_retrained": False,

        "signature_refit_on_tcga": False,

        "validation_cutoff_optimized": False,

        "rna_velocity": False,

        "external_cohort_projection": True,

        "independent_biological_validation": False,

        "clinical_outcomes_used_for_pathway_definition": False,

        "tcga_outcomes_used_for_signature_refitting": False,

        "multi_file_aggregation": (
            "Arithmetic mean across expression files "
            "belonging to the same TCGA case, matching "
            "the 09B aggregation convention."
        ),

        "scientific_interpretation": (
            "09E evaluates biological concordance of the "
            "frozen GSE176078-derived Deep-EVI molecular "
            "surrogate in TCGA-BRCA. Because TCGA-BRCA "
            "is the same external cohort used for the "
            "09B-09D projection and clinical characterization, "
            "this analysis is not independent biological "
            "validation. Genes used in the six Deep-EVI "
            "input programs are excluded from the predefined "
            "validation pathway gene sets to reduce "
            "circularity."
        ),
    }

    # ------------------------------------------------------------
    # Outputs
    # ------------------------------------------------------------

    analysis_cohort.to_csv(
        outdir / "09E_analysis_cohort.csv",
        index=False,
    )

    pathway_correlations.to_csv(
        outdir / "09E_pathway_correlations.csv",
        index=False,
    )

    context_correlations.to_csv(
        outdir / "09E_immune_context_correlations.csv",
        index=False,
    )

    pd.DataFrame(
        pathway_metadata
    ).to_csv(
        outdir / "09E_pathway_metadata.csv",
        index=False,
    )

    pathway_case.to_csv(
        outdir / "09E_pathway_case_scores.csv",
        index=False,
    )

    context_case.to_csv(
        outdir / "09E_immune_context_case_scores.csv",
        index=False,
    )

    with open(
        outdir / "GSE176078_09E_report.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            report,
            handle,
            indent=2,
        )

    print("=" * 80)
    print(
        "09E TCGA-BRCA BIOLOGICAL CONCORDANCE / "
        "CONSTRUCT VALIDATION"
    )
    print("=" * 80)

    print(
        f"09B score cases:       {scores['case_id'].nunique()}"
    )

    print(
        f"Clinical cases:        {clinical['case_id'].nunique()}"
    )

    print(
        f"Analysis cases:        {analysis_cohort['case_id'].nunique()}"
    )

    print(
        f"Expression files:      {len(inventory)}"
    )

    print(
        f"Processed files:       {processed_files}"
    )

    print(
        f"Multi-file cases:      {multi_file_cases}"
    )

    print(
        f"Validation pathways:   {len(pathway_records)}"
    )

    print()
    print("Pathway correlations:")

    print(
        pathway_correlations[
            [
                "pathway_id",
                "n_cases",
                "spearman_rho",
                "spearman_pvalue",
            ]
        ].to_string(index=False)
    )

    print()
    print("Scientific safeguards:")
    print("  Frozen TCGA surrogate: TRUE")
    print("  Signature refit: FALSE")
    print("  Deep-EVI retrained: FALSE")
    print("  Deep-EVI construction genes removed: TRUE")
    print("  TCGA outcome-driven pathway selection: FALSE")
    print("  RNA velocity: FALSE")
    print("  Independent biological validation: FALSE")
    print("  TCGA biological concordance: TRUE")

    print()
    print(
        "STATUS: PASS — 09E analysis completed successfully."
    )


if __name__ == "__main__":
    main()