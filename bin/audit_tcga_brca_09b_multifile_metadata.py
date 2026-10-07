#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import pandas as pd


def main():

    parser = argparse.ArgumentParser(
        description="Audit GDC metadata for TCGA-BRCA multi-file cases"
    )

    parser.add_argument("--query-json", required=True)
    parser.add_argument("--manifest-tsv", required=True)
    parser.add_argument("--multifile-csv", required=True)
    parser.add_argument("--outdir", required=True)

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(args.query_json) as fh:
        query = json.load(fh)

    hits = query["data"]["hits"]

    manifest = pd.read_csv(
        args.manifest_tsv,
        sep="\t",
        dtype=str
    )

    multifile = pd.read_csv(
        args.multifile_csv,
        dtype=str
    )

    target_file_ids = set(
        multifile["file_id"]
    )

    rows = []

    for hit in hits:

        file_id = hit["id"]

        if file_id not in target_file_ids:
            continue

        cases = hit.get("cases", [])

        if not cases:
            rows.append(
                {
                    "file_id": file_id,
                    "case_id": None,
                    "submitter_id": None,
                    "file_name": hit.get("file_name"),
                    "data_type": hit.get("data_type"),
                    "access": hit.get("access"),
                    "workflow_type": (
                        hit.get("analysis", {})
                        .get("workflow_type")
                    ),
                    "sample_count": 0,
                    "sample_types": "",
                    "sample_ids": "",
                    "sample_submitter_ids": "",
                    "sample_is_ffpe": "",
                    "sample_portions": "",
                    "sample_analytes": "",
                }
            )
            continue

        case = cases[0]

        samples = case.get(
            "samples",
            []
        )

        sample_types = []
        sample_ids = []
        sample_submitter_ids = []
        sample_is_ffpe = []
        sample_portions = []
        sample_analytes = []

        for sample in samples:

            sample_types.append(
                str(
                    sample.get(
                        "sample_type",
                        ""
                    )
                )
            )

            sample_ids.append(
                str(
                    sample.get(
                        "sample_id",
                        ""
                    )
                )
            )

            sample_submitter_ids.append(
                str(
                    sample.get(
                        "submitter_id",
                        ""
                    )
                )
            )

            sample_is_ffpe.append(
                str(
                    sample.get(
                        "is_ffpe",
                        ""
                    )
                )
            )

            sample_portions.append(
                str(
                    sample.get(
                        "portion",
                        ""
                    )
                )
            )

            sample_analytes.append(
                str(
                    sample.get(
                        "analyte",
                        ""
                    )
                )
            )

        rows.append(
            {
                "file_id": file_id,
                "case_id": case.get(
                    "case_id"
                ),
                "submitter_id": case.get(
                    "submitter_id"
                ),
                "file_name": hit.get(
                    "file_name"
                ),
                "data_type": hit.get(
                    "data_type"
                ),
                "access": hit.get(
                    "access"
                ),
                "workflow_type": (
                    hit.get("analysis", {})
                    .get("workflow_type")
                ),
                "file_size": hit.get(
                    "file_size"
                ),
                "sample_count": len(
                    samples
                ),
                "sample_types": ";".join(
                    sorted(
                        set(sample_types)
                    )
                ),
                "sample_ids": ";".join(
                    sample_ids
                ),
                "sample_submitter_ids": ";".join(
                    sample_submitter_ids
                ),
                "sample_is_ffpe": ";".join(
                    sample_is_ffpe
                ),
                "sample_portions": ";".join(
                    sample_portions
                ),
                "sample_analytes": ";".join(
                    sample_analytes
                ),
            }
        )

    result = pd.DataFrame(rows)

    result = result.sort_values(
        ["submitter_id", "file_id"]
    )

    result.to_csv(
        outdir
        / "multifile_gdc_metadata.csv",
        index=False
    )

    # Case-level summary
    case_summary = (
        result
        .groupby(
            [
                "case_id",
                "submitter_id"
            ],
            dropna=False
        )
        .agg(
            n_files=("file_id", "count"),
            sample_count_total=(
                "sample_count",
                "sum"
            ),
            file_sizes=(
                "file_size",
                lambda x: ";".join(
                    str(v)
                    for v in x
                )
            ),
            sample_types=(
                "sample_types",
                lambda x: ";".join(
                    sorted(
                        set(
                            ";".join(
                                str(v)
                                for v in x
                                if pd.notna(v)
                            ).split(";")
                        )
                    )
                )
            ),
            sample_ids=(
                "sample_ids",
                lambda x: ";".join(
                    sorted(
                        set(
                            ";".join(
                                str(v)
                                for v in x
                                if pd.notna(v)
                            ).split(";")
                        )
                    )
                )
            ),
            sample_portions=(
                "sample_portions",
                lambda x: ";".join(
                    sorted(
                        set(
                            ";".join(
                                str(v)
                                for v in x
                                if pd.notna(v)
                            ).split(";")
                        )
                    )
                )
            ),
            sample_analytes=(
                "sample_analytes",
                lambda x: ";".join(
                    sorted(
                        set(
                            ";".join(
                                str(v)
                                for v in x
                                if pd.notna(v)
                            ).split(";")
                        )
                    )
                )
            ),
        )
        .reset_index()
    )

    case_summary.to_csv(
        outdir
        / "multifile_case_metadata_summary.csv",
        index=False
    )

    print(
        f"Multi-file expression files inspected: "
        f"{len(result)}"
    )

    print(
        f"Multi-file cases inspected: "
        f"{case_summary.shape[0]}"
    )

    print("\nFILE-LEVEL METADATA")
    print(
        result.to_string(
            index=False
        )
    )

    print("\nCASE-LEVEL METADATA")
    print(
        case_summary.to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()