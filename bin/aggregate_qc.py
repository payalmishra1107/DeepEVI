#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--json", required=True)
    args = p.parse_args()

    root = Path(args.input)
    files = sorted(root.glob("*.json"))

    if not files:
        raise SystemExit("No per-sample QC JSON files found.")

    rows = []

    for f in files:
        with open(f) as fh:
            rows.append(json.load(fh))

    df = pd.DataFrame(rows).sort_values("sample_id")

    required = {
        "sample_id",
        "gsm",
        "input_cells",
        "input_genes",
        "conservative_qc_pass_cells",
        "conservative_qc_fail_cells",
        "conservative_retention_fraction",
    }

    missing = required.difference(df.columns)

    if missing:
        raise SystemExit(
            f"Missing required summary fields: {sorted(missing)}"
        )

    df.to_csv(args.csv, index=False)

    cohort = {
        "samples": int(len(df)),
        "total_input_cells": int(df["input_cells"].sum()),
        "total_conservative_pass_cells": int(
            df["conservative_qc_pass_cells"].sum()
        ),
        "total_conservative_fail_cells": int(
            df["conservative_qc_fail_cells"].sum()
        ),
        "overall_conservative_retention_fraction": (
            float(
                df["conservative_qc_pass_cells"].sum()
                / df["input_cells"].sum()
            )
            if df["input_cells"].sum()
            else 0.0
        ),
        "samples_with_mito_qc": int(
            (df["mitochondrial_status"] == "COMPUTED").sum()
        ),
        "samples_without_detectable_mito_genes": int(
            (df["mitochondrial_status"] == "NO_MT_GENES_DETECTED").sum()
        ),
        "sample_ids": df["sample_id"].tolist(),
    }

    with open(args.json, "w") as fh:
        json.dump(cohort, fh, indent=2)


if __name__ == "__main__":
    main()