#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Fixed molecular/context gene sets
# ---------------------------------------------------------------------

T_CELL_CONTEXT_GENES = [
    "CD3D",
    "CD3E",
    "CD3G",
    "TRBC1",
    "TRBC2",
]

CYTOTOXIC_CONTEXT_GENES = [
    "CD8A",
    "CD8B",
    "NKG7",
    "GNLY",
    "GZMB",
    "GZMH",
    "PRF1",
    "CTSW",
]

EXHAUSTION_CONTEXT_GENES = [
    "PDCD1",
    "LAG3",
    "TIGIT",
    "HAVCR2",
    "CTLA4",
    "TOX",
    "TOX2",
    "ENTPD1",
    "CXCL13",
]


# ---------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------

def normalize_gene_name(value):
    """
    Normalize GENCODE-style gene identifiers.

    Example:
        ENSG00000000003.15 -> ENSG00000000003

    The signature is gene-symbol based, so the function returns the
    supplied gene symbol after stripping whitespace and version suffixes.
    """
    if value is None:
        return ""

    value = str(value).strip()

    if "." in value and value.startswith("ENSG"):
        value = value.split(".", 1)[0]

    return value


def read_signature(signature_dir):
    signature_dir = Path(signature_dir)

    candidates = sorted(
        signature_dir.glob("*09A_gene_signature.csv")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No 09A gene signature found in {signature_dir}"
        )

    signature_path = candidates[0]

    df = pd.read_csv(signature_path)

    required = {
        "gene_symbol",
        "gene_weight",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Signature missing required columns: {sorted(missing)}"
        )

    df["gene_symbol"] = (
        df["gene_symbol"]
        .astype(str)
        .str.strip()
    )

    df["gene_weight"] = pd.to_numeric(
        df["gene_weight"],
        errors="coerce",
    )

    if df["gene_weight"].isna().any():
        raise ValueError(
            "Signature contains non-numeric gene weights."
        )

    if df["gene_symbol"].duplicated().any():
        raise ValueError(
            "09A signature contains duplicated gene symbols."
        )

    if not np.isfinite(
        df["gene_weight"].to_numpy()
    ).all():
        raise ValueError(
            "09A signature contains non-finite weights."
        )

    return signature_path, df


# ---------------------------------------------------------------------
# GDC manifest
# ---------------------------------------------------------------------

def read_manifest(path):
    """
    Read the two-column GDC download manifest.

    The manifest intentionally contains only:
        id
        filename

    Case-level metadata are obtained separately from the
    GDC query JSON.
    """

    path = Path(path)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "id",
        "filename",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"GDC manifest missing required columns: "
            f"{sorted(missing)}"
        )

    mapping = {}

    for _, row in df.iterrows():

        file_id = str(
            row["id"]
        ).strip()

        filename = str(
            row["filename"]
        ).strip()

        if not file_id:
            continue

        mapping[file_id] = {
            "filename": filename,
        }

    if not mapping:
        raise ValueError(
            f"No records found in GDC manifest: {path}"
        )

    return mapping


def read_query_metadata(path):
    """
    Read the GDC query JSON and construct:

        file UUID -> case_id -> submitter_id -> sample metadata

    The query JSON is the authoritative source for file-level
    biological metadata.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"GDC query JSON not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    if not isinstance(payload, dict):
        raise ValueError(
            "GDC query JSON is not a JSON object."
        )

    data = payload.get("data")

    if not isinstance(data, dict):
        raise ValueError(
            "GDC query JSON does not contain a 'data' object."
        )

    hits = data.get("hits")

    if not isinstance(hits, list):
        raise ValueError(
            "GDC query JSON does not contain a 'data.hits' list."
        )

    mapping = {}

    for hit in hits:

        if not isinstance(hit, dict):
            continue

        file_id = str(
            hit.get("id", "")
        ).strip()

        if not file_id:
            continue

        file_name = str(
            hit.get("file_name", "")
        ).strip()

        cases = hit.get(
            "cases",
            [],
        )

        if not isinstance(cases, list) or not cases:
            raise ValueError(
                f"No case metadata found for GDC file: "
                f"{file_id}"
            )

        case = cases[0]

        if not isinstance(case, dict):
            raise ValueError(
                f"Invalid case metadata for GDC file: "
                f"{file_id}"
            )

        case_id = str(
            case.get("case_id", "")
        ).strip()

        submitter_id = str(
            case.get("submitter_id", "")
        ).strip()

        if not case_id:
            raise ValueError(
                f"Missing case_id for GDC file: "
                f"{file_id}"
            )

        if not submitter_id:
            raise ValueError(
                f"Missing submitter_id for GDC file: "
                f"{file_id}"
            )

        samples = case.get(
            "samples",
            [],
        )

        sample_types = []

        if isinstance(samples, list):

            for sample in samples:

                if not isinstance(
                    sample,
                    dict,
                ):
                    continue

                sample_type = str(
                    sample.get(
                        "sample_type",
                        "",
                    )
                ).strip()

                if sample_type:
                    sample_types.append(
                        sample_type
                    )

        mapping[file_id] = {
            "case_id": case_id,
            "submitter_id": submitter_id,
            "filename": file_name,
            "sample_types": sample_types,
        }

    if not mapping:
        raise ValueError(
            f"No file metadata found in GDC query JSON: {path}"
        )

    return mapping

def discover_expression_files(raw_dir):

    raw_dir = Path(raw_dir)

    patterns = [
        "*augmented_star_gene_counts.tsv",
        "*.rna_seq.augmented_star_gene_counts.tsv",
    ]

    files = set()

    for pattern in patterns:
        files.update(
            raw_dir.rglob(pattern)
        )

    files = sorted(files)

    if not files:
        raise FileNotFoundError(
            f"No augmented STAR-Counts files found in {raw_dir}"
        )

    return files


# ---------------------------------------------------------------------
# UUID extraction
# ---------------------------------------------------------------------

def extract_gdc_uuid(path):

    path = Path(path)

    for parent in path.parents:

        name = parent.name

        if re.fullmatch(
            r"[0-9a-fA-F]{8}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{4}-"
            r"[0-9a-fA-F]{12}",
            name,
        ):
            return name

    raise ValueError(
        f"Could not identify GDC UUID directory for {path}"
    )


# ---------------------------------------------------------------------
# STAR Counts parser
# ---------------------------------------------------------------------

def parse_star_counts(path, required_genes):
    rows = []

    with open(path, encoding="utf-8", errors="replace") as handle:

        header = None

        for raw in handle:

            line = raw.rstrip("\n\r")

            if not line:
                continue

            if line.startswith("gene_id"):
                header = line.split()
                break

        if header is None:
            raise ValueError(
                f"Could not find STAR header in {path}"
            )

        positions = {
            name: i
            for i, name in enumerate(header)
        }

        if "gene_name" not in positions:
            raise ValueError(
                f"gene_name column missing in {path}"
            )

        if "tpm_unstranded" in positions:
            tpm_column = "tpm_unstranded"
        elif "tpm" in positions:
            tpm_column = "tpm"
        else:
            raise ValueError(
                f"No TPM column found in {path}. "
                f"Available columns: {header}"
            )

        gene_pos = positions["gene_name"]
        tpm_pos = positions[tpm_column]

        required = set(required_genes)

        for raw in handle:

            parts = raw.rstrip("\n\r").split()

            if len(parts) <= max(gene_pos, tpm_pos):
                continue

            gene = normalize_gene_name(
                parts[gene_pos]
            )

            if gene not in required:
                continue

            try:
                value = float(parts[tpm_pos])
            except (ValueError, TypeError):
                continue

            if not np.isfinite(value):
                continue

            rows.append((gene, value))

    values = defaultdict(float)

    for gene, value in rows:
        values[gene] += value

    return dict(values)

# ---------------------------------------------------------------------
# Clinical data
# ---------------------------------------------------------------------

def read_clinical(path):

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required = {
        "case_id",
        "submitter_id",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            f"Clinical table missing: {sorted(missing)}"
        )

    return df


# ---------------------------------------------------------------------
# Numeric helper
# ---------------------------------------------------------------------

def safe_float(value):

    try:
        x = float(value)

        if np.isfinite(x):
            return x

    except (
        ValueError,
        TypeError,
    ):
        pass

    return np.nan


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw-dir",
        required=True,
    )

    parser.add_argument(
        "--query-json",
        required=True,
    )

    parser.add_argument(
        "--clinical-tsv",
        required=True,
    )

    parser.add_argument(
        "--manifest-tsv",
        required=True,
    )

    parser.add_argument(
        "--signature-dir",
        required=True,
    )

    parser.add_argument(
        "--outdir",
        required=True,
    )

    args = parser.parse_args()

    raw_dir = Path(
        args.raw_dir
    )

    query_json = Path(
        args.query_json
    )

    clinical_tsv = Path(
        args.clinical_tsv
    )

    signature_dir = Path(
        args.signature_dir
    )

    outdir = Path(
        args.outdir
    )

    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "STEP 09B — TCGA-BRCA FROZEN DEEP-EVI VALIDATION"
    )
    print("=" * 80)

    # --------------------------------------------------------------
    # Signature
    # --------------------------------------------------------------

    signature_path, signature_df = read_signature(
        signature_dir
    )

    signature_genes = (
        signature_df["gene_symbol"]
        .tolist()
    )

    print(
        "Frozen signature:",
        signature_path,
    )

    print(
        "Signature genes:",
        len(signature_genes),
    )

    # --------------------------------------------------------------
    # Manifest
    # --------------------------------------------------------------

    manifest_path = Path(
        args.manifest_tsv
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"TCGA STAR-Counts manifest not found: "
            f"{manifest_path}"
        )

    manifest_mapping = read_manifest(
        manifest_path
    )

    query_metadata = read_query_metadata(
        query_json
    )

    print(
        "Manifest:",
        manifest_path,
    )

    print(
        "Manifest file records:",
        len(manifest_mapping),
    )

    print(
        "GDC query metadata records:",
        len(query_metadata),
    )

    manifest_ids = set(
        manifest_mapping
    )

    query_ids = set(
        query_metadata
    )

    missing_query_metadata = (
        manifest_ids - query_ids
    )

    unexpected_query_metadata = (
        query_ids - manifest_ids
    )

    if missing_query_metadata:
        raise ValueError(
            "Manifest files missing from GDC query JSON: "
            f"{len(missing_query_metadata)}"
        )

    if unexpected_query_metadata:
        raise ValueError(
            "GDC query JSON contains file IDs absent "
            "from manifest: "
            f"{len(unexpected_query_metadata)}"
        )

    print(
        "Manifest/query UUID agreement: PASS"
    )

    # --------------------------------------------------------------
    # Expression files
    # --------------------------------------------------------------

    expression_files = discover_expression_files(
        raw_dir
    )

    print(
        "Downloaded STAR-Counts files:",
        len(expression_files),
    )

    if len(expression_files) != len(
        manifest_mapping
    ):
        raise ValueError(
            "Downloaded expression file count does not "
            "match manifest record count."
        )

    # --------------------------------------------------------------
    # Clinical
    # --------------------------------------------------------------

    clinical = read_clinical(
        clinical_tsv
    )

    print(
        "Clinical cases:",
        len(clinical),
    )

    print(
        "Unique clinical cases:",
        clinical["case_id"].nunique(),
    )

    # --------------------------------------------------------------
    # Gene sets
    # --------------------------------------------------------------

    all_context_genes = sorted(
        set(
            signature_genes
            + T_CELL_CONTEXT_GENES
            + CYTOTOXIC_CONTEXT_GENES
            + EXHAUSTION_CONTEXT_GENES
        )
    )

    # --------------------------------------------------------------
    # Parse expression
    # --------------------------------------------------------------

    case_vectors = defaultdict(list)

    file_inventory = []

    for index, path in enumerate(
        expression_files,
        start=1,
    ):

        uuid = extract_gdc_uuid(
            path
        )

        manifest_metadata = manifest_mapping.get(
            uuid
        )

        if manifest_metadata is None:
            raise ValueError(
                "Downloaded UUID not found in manifest: "
                f"{uuid}"
            )

        metadata = query_metadata.get(
            uuid
        )

        if metadata is None:
            raise ValueError(
                "Downloaded UUID not found in GDC query metadata: "
                f"{uuid}"
            )

        manifest_filename = manifest_metadata[
            "filename"
        ]

        if (
            manifest_filename
            and manifest_filename != path.name
        ):
            raise ValueError(
                "Downloaded filename does not match "
                "manifest filename for UUID "
                f"{uuid}: "
                f"{path.name} != {manifest_filename}"
            )

        values = parse_star_counts(
            path,
            all_context_genes,
        )

        # Explicit molecular scale:
        # log2(TPM_unstranded + 1)
        transformed = {
            gene: math.log2(
                max(value, 0.0) + 1.0
            )
            for gene, value in values.items()
        }

        case_vectors[
            metadata["case_id"]
        ].append(
            transformed
        )

        file_inventory.append(
            {
                "file_id": uuid,
                "filename": path.name,
                "case_id": metadata[
                    "case_id"
                ],
                "submitter_id": metadata[
                    "submitter_id"
                ],
                "genes_found": len(
                    set(signature_genes)
                    & set(values)
                ),
            }
        )

        if index % 100 == 0:
            print(
                f"Processed "
                f"{index}/{len(expression_files)}"
            )

    # --------------------------------------------------------------
    # Patient-level aggregation
    # --------------------------------------------------------------

    score_rows = []

    for case_id, vectors in case_vectors.items():

        genes = sorted(
            set().union(
                *[
                    set(v.keys())
                    for v in vectors
                ]
            )
        )

        aggregated = {}

        for gene in genes:

            values = [
                v[gene]
                for v in vectors
                if gene in v
            ]

            if values:
                aggregated[gene] = float(
                    np.mean(values)
                )

        # ----------------------------------------------------------
        # Frozen molecular surrogate
        # ----------------------------------------------------------

        signature_score = 0.0

        for _, row in signature_df.iterrows():

            gene = row[
                "gene_symbol"
            ]

            weight = float(
                row["gene_weight"]
            )

            expression = aggregated.get(
                gene,
                np.nan,
            )

            if np.isfinite(expression):
                signature_score += (
                    weight * expression
                )

        # ----------------------------------------------------------
        # Context scores
        # ----------------------------------------------------------

        def mean_gene_score(
            geneset
        ):

            values = [
                aggregated[g]
                for g in geneset
                if g in aggregated
            ]

            if not values:
                return np.nan

            return float(
                np.mean(values)
            )

        tcell_context = mean_gene_score(
            T_CELL_CONTEXT_GENES
        )

        cytotoxic_context = mean_gene_score(
            CYTOTOXIC_CONTEXT_GENES
        )

        exhaustion_context = mean_gene_score(
            EXHAUSTION_CONTEXT_GENES
        )

        score_rows.append(
            {
                "case_id": case_id,
                "deep_evi_tcga_surrogate": signature_score,
                "tcell_context_score": tcell_context,
                "cytotoxic_context_score": cytotoxic_context,
                "exhaustion_context_score": exhaustion_context,
                "n_expression_files": len(vectors),
                "n_signature_genes_detected": sum(
                    gene in aggregated
                    for gene in signature_genes
                ),
            }
        )

    scores = pd.DataFrame(
        score_rows
    )

    # --------------------------------------------------------------
    # Merge clinical
    # --------------------------------------------------------------

    merged = scores.merge(
        clinical,
        on="case_id",
        how="inner",
        suffixes=(
            "",
            "_clinical",
        ),
    )

    # --------------------------------------------------------------
    # Survival endpoint
    # --------------------------------------------------------------

    def choose_time(row):

        death = safe_float(
            row.get(
                "days_to_death",
                "",
            )
        )

        followup = safe_float(
            row.get(
                "days_to_last_follow_up",
                "",
            )
        )

        latest = safe_float(
            row.get(
                "latest_followup_days",
                "",
            )
        )

        if np.isfinite(death):
            return death

        if np.isfinite(followup):
            return followup

        if np.isfinite(latest):
            return latest