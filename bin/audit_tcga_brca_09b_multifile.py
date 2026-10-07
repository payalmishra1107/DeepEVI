#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TCELL_GENES = [
    "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2"
]

CYTOTOXIC_GENES = [
    "CD8A", "CD8B", "NKG7", "GNLY",
    "GZMB", "GZMH", "PRF1", "CTSW"
]

EXHAUSTION_GENES = [
    "PDCD1", "LAG3", "TIGIT", "HAVCR2",
    "CTLA4", "TOX", "TOX2", "ENTPD1", "CXCL13"
]


def read_signature(signature_dir):
    files = list(
        Path(signature_dir).glob("*gene_signature.csv")
    )

    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one gene signature CSV; "
            f"found {len(files)} in {signature_dir}"
        )

    df = pd.read_csv(files[0])

    required = {"gene_symbol", "gene_weight"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Signature missing columns: {sorted(missing)}"
        )

    df["gene_symbol"] = df["gene_symbol"].astype(str)
    df["gene_weight"] = pd.to_numeric(
        df["gene_weight"],
        errors="raise"
    )

    if df["gene_symbol"].duplicated().any():
        raise RuntimeError(
            "Duplicate genes in frozen 09A signature"
        )

    if len(df) != 33:
        raise RuntimeError(
            f"Expected 33 signature genes; found {len(df)}"
        )

    return dict(
        zip(
            df["gene_symbol"],
            df["gene_weight"]
        )
    )


def read_query_metadata(query_json):
    with open(query_json) as fh:
        data = json.load(fh)

    rows = []

    for hit in data["data"]["hits"]:

        file_id = hit["id"]
        cases = hit.get("cases", [])

        if not cases:
            raise RuntimeError(
                f"No case metadata for file {file_id}"
            )

        case = cases[0]

        sample_types = sorted(
            {
                str(sample.get("sample_type"))
                for sample in case.get("samples", [])
                if sample.get("sample_type") is not None
            }
        )

        rows.append(
            {
                "file_id": file_id,
                "case_id": case.get("case_id"),
                "submitter_id": case.get("submitter_id"),
                "sample_types": ";".join(sample_types),
            }
        )

    df = pd.DataFrame(rows)

    if df["file_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate file IDs in query JSON"
        )

    return df


def read_manifest(path):
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str
    )

    required = {"id", "filename"}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Manifest missing columns: {sorted(missing)}"
        )

    return df[
        ["id", "filename"]
    ].rename(
        columns={"id": "file_id"}
    )


def find_expression_header(fh):
    """
    Skip GDC metadata/comment lines and locate the actual
    STAR-Counts tabular header.

    Typical GDC file beginning:

        # gene-model: GENCODE v36
        gene_id gene_name gene_type unstranded ...

    Some files may contain several metadata lines.
    """

    metadata_lines = []

    while True:
        line = fh.readline()

        if not line:
            raise RuntimeError(
                "Reached EOF before finding STAR-Counts header"
            )

        stripped = line.strip()

        if not stripped:
            continue

        # GDC metadata/comment line.
        if stripped.startswith("#"):
            metadata_lines.append(stripped)
            continue

        header = stripped.split()

        # We recognize the actual expression table by the
        # presence of one of these expression columns.
        if (
            "tpm_unstranded" in header
            or "tpm" in header
        ):
            return header

        # Otherwise continue searching. This protects against
        # additional metadata lines that do not begin with #.
        metadata_lines.append(stripped)


def resolve_gene_column(header):
    if "gene_name" in header:
        return "gene_name"

    if "gene_id" in header:
        return "gene_id"

    raise RuntimeError(
        "STAR-Counts header contains neither gene_name nor gene_id. "
        f"Header={header}"
    )


def normalize_gene_name(gene):
    gene = str(gene)

    # Remove Ensembl version suffix only when applicable.
    if gene.startswith("ENSG") and "." in gene:
        gene = gene.split(".", 1)[0]

    return gene


def read_tpm_file(path, genes_needed):

    path = Path(path)
    values = {}

    with open(
        path,
        "r",
        errors="replace"
    ) as fh:

        header = find_expression_header(fh)

        gene_column = resolve_gene_column(header)

        if "tpm_unstranded" in header:
            value_column = "tpm_unstranded"
        elif "tpm" in header:
            value_column = "tpm"
        else:
            raise RuntimeError(
                f"No TPM column in {path}"
            )

        gene_idx = header.index(gene_column)
        value_idx = header.index(value_column)

        for line in fh:

            parts = line.rstrip("\n").split()

            if len(parts) <= max(
                gene_idx,
                value_idx
            ):
                continue

            gene = normalize_gene_name(
                parts[gene_idx]
            )

            if gene not in genes_needed:
                continue

            try:
                tpm = float(parts[value_idx])
            except ValueError:
                continue

            if not np.isfinite(tpm):
                continue

            values[gene] = np.log2(
                tpm + 1.0
            )

    return values


def mean_present(values, genes):

    present = [
        values[g]
        for g in genes
        if g in values
    ]

    if not present:
        return np.nan

    return float(np.mean(present))


def score_file(path, signature):

    values = read_tpm_file(
        path,
        set(signature)
    )

    missing = sorted(
        set(signature) - set(values)
    )

    score = float(
        sum(
            signature[g] * values[g]
            for g in values
        )
    )

    return {
        "score": score,
        "tcell_context_score": mean_present(
            values,
            TCELL_GENES
        ),
        "cytotoxic_context_score": mean_present(
            values,
            CYTOTOXIC_GENES
        ),
        "exhaustion_context_score": mean_present(
            values,
            EXHAUSTION_GENES
        ),
        "genes_found": len(values),
        "missing_genes": ";".join(missing),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Audit the 11 TCGA-BRCA cases with multiple "
            "GDC STAR-Counts expression files."
        )
    )

    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--query-json", required=True)
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--signature-dir", required=True)
    parser.add_argument("--inventory-csv", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    outdir = Path(args.outdir)

    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    signature = read_signature(
        args.signature_dir
    )

    query = read_query_metadata(
        args.query_json
    )

    manifest = read_manifest(
        args.manifest_tsv
    )

    metadata = query.merge(
        manifest,
        on="file_id",
        how="inner",
        validate="one_to_one"
    )

    duplicate_counts = (
        metadata
        .groupby("case_id")
        .size()
        .rename("n_files")
        .reset_index()
    )

    duplicate_cases = duplicate_counts[
        duplicate_counts["n_files"] > 1
    ].copy()

    if duplicate_cases.empty:
        raise RuntimeError(
            "No multi-file cases found"
        )

    rows = []

    for _, case_row in duplicate_cases.iterrows():

        case_id = case_row["case_id"]

        case_files = metadata[
            metadata["case_id"] == case_id
        ]

        for _, file_row in case_files.iterrows():

            filename = file_row["filename"]

            candidates = list(
                raw_dir.rglob(filename)
            )

            if len(candidates) != 1:
                raise RuntimeError(
                    f"Expected one downloaded file for "
                    f"{filename}; found {len(candidates)}"
                )

            result = score_file(
                candidates[0],
                signature
            )

            rows.append(
                {
                    "case_id": case_id,
                    "submitter_id": file_row[
                        "submitter_id"
                    ],
                    "file_id": file_row[
                        "file_id"
                    ],
                    "filename": filename,
                    "sample_types": file_row[
                        "sample_types"
                    ],
                    **result,
                }
            )

    per_file = pd.DataFrame(rows)

    summary_rows = []

    for case_id, group in per_file.groupby(
        "case_id"
    ):

        scores = group["score"].astype(float).to_numpy()

        mean_score = float(np.mean(scores))
        median_score = float(np.median(scores))
        std_score = float(np.std(scores, ddof=0))

        summary_rows.append(
            {
                "case_id": case_id,
                "submitter_id": group[
                    "submitter_id"
                ].iloc[0],
                "n_expression_files": len(group),
                "score_mean": mean_score,
                "score_median": median_score,
                "score_min": float(np.min(scores)),
                "score_max": float(np.max(scores)),
                "score_range": float(
                    np.max(scores) - np.min(scores)
                ),
                "score_std": std_score,
                "score_cv": (
                    float(
                        std_score / abs(mean_score)
                    )
                    if mean_score != 0
                    else np.nan
                ),
                "sample_types": ";".join(
                    sorted(
                        set(
                            group["sample_types"]
                        )
                    )
                ),
                "all_files_have_33_genes": bool(
                    group["genes_found"].eq(33).all()
                ),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    existing_scores = pd.read_csv(
        Path(args.inventory_csv).parent
        / "GSE176078_09B_TCGA_BRCA_scores.csv",
        dtype={"case_id": str}
    )

    comparison = existing_scores[
        [
            "case_id",
            "deep_evi_tcga_surrogate"
        ]
    ].merge(
        summary[
            [
                "case_id",
                "score_mean",
                "score_median"
            ]
        ],
        on="case_id",
        how="inner",
        validate="one_to_one"
    )

    comparison[
        "pipeline_minus_file_mean"
    ] = (
        comparison["deep_evi_tcga_surrogate"]
        - comparison["score_mean"]
    )

    comparison[
        "pipeline_matches_file_mean"
    ] = np.isclose(
        comparison["deep_evi_tcga_surrogate"],
        comparison["score_mean"],
        rtol=1e-6,
        atol=1e-8
    )

    per_file.to_csv(
        outdir
        / "multifile_case_file_level_scores.csv",
        index=False
    )

    summary.to_csv(
        outdir
        / "multifile_case_aggregation_summary.csv",
        index=False
    )

    comparison.to_csv(
        outdir
        / "multifile_pipeline_vs_filemean.csv",
        index=False
    )

    report = {
        "status": "complete",
        "multi_file_case_count": int(
            len(summary)
        ),
        "multi_file_expression_file_count": int(
            len(per_file)
        ),
        "signature_gene_count": int(
            len(signature)
        ),
        "all_files_have_complete_signature": bool(
            per_file[
                "genes_found"
            ].eq(33).all()
        ),
        "pipeline_matches_arithmetic_file_mean": bool(
            comparison[
                "pipeline_matches_file_mean"
            ].all()
        ),
        "maximum_absolute_pipeline_vs_file_mean_difference": float(
            comparison[
                "pipeline_minus_file_mean"
            ].abs().max()
        ),
        "mean_score_range_across_multi_file_cases": float(
            summary["score_range"].mean()
        ),
        "maximum_score_range": float(
            summary["score_range"].max()
        ),
        "mean_score_cv": float(
            summary["score_cv"].mean()
        ),
        "clinical_outcomes_used": False,
        "signature_refit": False,
        "interpretation": (
            "Technical audit only. The frozen 09A molecular "
            "signature is applied independently to each "
            "expression file belonging to a multi-file case. "
            "No TCGA clinical outcome is used."
        ),
    }

    with open(
        outdir / "multifile_audit_report.json",
        "w"
    ) as fh:
        json.dump(
            report,
            fh,
            indent=2
        )

    print(
        json.dumps(
            report,
            indent=2
        )
    )

    print("\nMULTI-FILE CASE SUMMARY")
    print(
        summary.to_string(index=False)
    )


if __name__ == "__main__":
    main()