#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROGRAM_GENES = {
    "tcell_identity_score": [
        "CD3D", "CD3E", "CD3G", "TRBC1", "TRBC2"
    ],

    "cd8_cytotoxic_score": [
        "CD8A", "CD8B", "NKG7", "GNLY",
        "GZMB", "GZMH", "PRF1", "CTSW"
    ],

    "treg_score": [
        "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IL7R"
    ],

    "tfh_score": [
        "CXCL13", "PDCD1", "ICOS", "CXCR5"
    ],

    "activation_effector_score": [
        "IFNG", "TNF", "IL2", "CD69",
        "HLA-DRA", "HLA-DRB1"
    ],

    "exhaustion_dysfunction_score": [
        "PDCD1", "LAG3", "TIGIT", "HAVCR2",
        "CTLA4", "TOX", "TOX2", "ENTPD1",
        "CXCL13"
    ],
}


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--signature",
        required=True
    )

    parser.add_argument(
        "--outdir",
        required=True
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        args.signature,
        "r",
        encoding="utf-8"
    ) as f:
        signature = json.load(f)

    programs = signature["programs"]
    coefficients = np.asarray(
        signature["coefficients"],
        dtype=float
    )

    if list(programs) != list(
        PROGRAM_GENES.keys()
    ):
        raise RuntimeError(
            "Frozen 09A program order does not match "
            "the Step-7B gene definitions."
        )

    if len(coefficients) != len(programs):
        raise RuntimeError(
            "Coefficient count does not match programs."
        )

    rows = []

    # Exact frozen rule:
    #
    # program coefficient / number of genes
    #
    # overlapping genes are summed.
    for program, coefficient in zip(
        programs,
        coefficients
    ):

        genes = PROGRAM_GENES[program]

        per_gene_weight = (
            float(coefficient)
            / float(len(genes))
        )

        for gene in genes:

            rows.append(
                {
                    "gene": gene,
                    "program": program,
                    "program_coefficient": float(
                        coefficient
                    ),
                    "program_gene_count": len(
                        genes
                    ),
                    "gene_weight_contribution": (
                        per_gene_weight
                    ),
                }
            )

    contribution_df = pd.DataFrame(
        rows
    )

    gene_df = (
        contribution_df
        .groupby(
            "gene",
            as_index=False
        )
        .agg(
            molecular_weight=(
                "gene_weight_contribution",
                "sum"
            ),
            contributing_programs=(
                "program",
                lambda x: ";".join(
                    sorted(set(x))
                )
            ),
            contributing_program_count=(
                "program",
                "nunique"
            )
        )
    )

    gene_df[
        "absolute_weight"
    ] = np.abs(
        gene_df["molecular_weight"]
    )

    total_abs = gene_df[
        "absolute_weight"
    ].sum()

    gene_df[
        "absolute_weight_fraction"
    ] = (
        gene_df["absolute_weight"]
        / total_abs
    )

    gene_df = gene_df.sort_values(
        "absolute_weight",
        ascending=False
    ).reset_index(
        drop=True
    )

    gene_df["rank"] = (
        np.arange(
            len(gene_df)
        ) + 1
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    expected_union = set()

    for genes in PROGRAM_GENES.values():
        expected_union.update(genes)

    if len(expected_union) != 33:
        raise RuntimeError(
            f"Expected 33 unique genes, found "
            f"{len(expected_union)}."
        )

    if len(gene_df) != 33:
        raise RuntimeError(
            f"Gene attribution contains "
            f"{len(gene_df)} genes instead of 33."
        )

    # Verify that summed gene weights reproduce the
    # total program coefficients.
    program_total = (
        contribution_df
        .groupby("program")[
            "gene_weight_contribution"
        ]
        .sum()
        .to_dict()
    )

    for program, coefficient in zip(
        programs,
        coefficients
    ):

        reconstructed = (
            program_total[program]
        )

        if not np.isclose(
            reconstructed,
            coefficient,
            rtol=1e-8,
            atol=1e-10
        ):
            raise RuntimeError(
                f"Gene-weight reconstruction failed "
                f"for {program}: "
                f"{reconstructed} != {coefficient}"
            )

    # ---------------------------------------------------------
    # Program summary
    # ---------------------------------------------------------

    program_df = pd.DataFrame(
        {
            "program": programs,
            "ridge_coefficient": coefficients,
            "gene_count": [
                len(PROGRAM_GENES[p])
                for p in programs
            ]
        }
    )

    program_df[
        "absolute_coefficient"
    ] = np.abs(
        program_df[
            "ridge_coefficient"
        ]
    )

    program_df[
        "absolute_coefficient_fraction"
    ] = (
        program_df[
            "absolute_coefficient"
        ]
        /
        program_df[
            "absolute_coefficient"
        ].sum()
    )

    program_df = program_df.sort_values(
        "absolute_coefficient",
        ascending=False
    ).reset_index(
        drop=True
    )

    program_df["rank"] = (
        np.arange(
            len(program_df)
        ) + 1
    )

    # ---------------------------------------------------------
    # Report
    # ---------------------------------------------------------

    report = {
        "step": "11D_frozen_gene_level_molecular_attribution",
        "status": "complete",
        "source_step": "09A",
        "model": signature.get(
            "model",
            "Ridge"
        ),
        "alpha": signature.get(
            "alpha"
        ),
        "training_only": signature.get(
            "training_only",
            False
        ),
        "signature_refit": False,
        "gene_weight_rule": (
            "program coefficient divided by program "
            "gene count; overlapping gene contributions summed"
        ),
        "program_count": len(programs),
        "gene_count": len(gene_df),
        "genes": sorted(
            gene_df["gene"].tolist()
        ),
        "programs": programs,
        "independent_validation": False,
        "causal_inference": False,
        "rna_velocity": False
    }

    gene_df.to_csv(
        outdir /
        "GSE176078_11D_gene_molecular_attribution.csv",
        index=False
    )

    contribution_df.to_csv(
        outdir /
        "GSE176078_11D_gene_program_contributions.csv",
        index=False
    )

    program_df.to_csv(
        outdir /
        "GSE176078_11D_program_molecular_attribution.csv",
        index=False
    )

    with open(
        outdir /
        "GSE176078_11D_gene_attribution_report.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            report,
            f,
            indent=2
        )

    print("=" * 80)
    print("STEP 11D GENE-LEVEL MOLECULAR ATTRIBUTION COMPLETE")
    print("=" * 80)
    print(
        f"Programs: {len(programs)}"
    )
    print(
        f"Unique genes: {len(gene_df)}"
    )
    print(
        "Signature refitted: NO"
    )
    print(
        "Gene weights reconstructed from frozen 09A rule: YES"
    )
    print("=" * 80)


if __name__ == "__main__":
    main()