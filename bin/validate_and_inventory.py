#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import tarfile
from pathlib import Path


REQUIRED_FILES = {
    "count_matrix_sparse.mtx",
    "count_matrix_genes.tsv",
    "count_matrix_barcodes.tsv",
    "metadata.csv",
}


def main():

    parser = argparse.ArgumentParser(
        description="Validate GSE176078 sample archives."
    )

    parser.add_argument(
        "--input",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output = Path(args.output)

    archives = sorted(
        input_dir.glob("GSM*.tar.gz")
    )

    rows = []

    for archive in archives:

        sample_id = archive.name.replace(
            ".tar.gz",
            ""
        )

        found = set()
        error = ""

        try:

            with tarfile.open(
                archive,
                "r:gz"
            ) as tar:

                for member in tar.getmembers():

                    name = Path(
                        member.name
                    ).name

                    if name in REQUIRED_FILES:
                        found.add(name)

        except Exception as exc:

            error = str(exc)

        missing = REQUIRED_FILES - found

        rows.append({
            "sample_id": sample_id,
            "archive": archive.name,
            "required_files_found": len(found),
            "missing_files": ";".join(
                sorted(missing)
            ),
            "status":
                "PASS"
                if not missing and not error
                else "FAIL",
            "error": error
        })

    output.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output,
        "w",
        newline=""
    ) as fh:

        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "sample_id",
                "archive",
                "required_files_found",
                "missing_files",
                "status",
                "error"
            ]
        )

        writer.writeheader()
        writer.writerows(rows)

    failed = sum(
        row["status"] == "FAIL"
        for row in rows
    )

    print(
        f"input={input_dir}"
    )

    print(
        f"gsm_archives={len(archives)}"
    )

    print(
        f"failed_archives={failed}"
    )

    print(
        "status="
        + (
            "PASS"
            if failed == 0
            else "FAIL"
        )
    )


if __name__ == "__main__":
    main()