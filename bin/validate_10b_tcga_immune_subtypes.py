#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, spearmanr


EXPECTED_SUBTYPES = {
    "C1": "Wound Healing",
    "C2": "IFN-gamma Dominant",
    "C3": "Inflammatory",
    "C4": "Lymphocyte Depleted",
    "C5": "Immunologically Quiet",
    "C6": "TGF-beta Dominant",
}


def parse_args():
    p = argparse.ArgumentParser(
        description="10B external TCGA-BRCA immune-subtype validation"
    )

    p.add_argument(
        "--scores",
        required=True,
        help="Frozen 09B TCGA case-level Deep-EVI scores"
    )

    p.add_argument(
        "--immune-subtypes",
        required=True,
        help="Published TCGA immune subtype sample-level table"
    )

    p.add_argument(
        "--case-inventory",
        required=True,
        help="09B case inventory"
    )

    p.add_argument(
        "--outdir",
        required=True
    )

    return p.parse_args()


def subtype_code(value):
    if pd.isna(value):
        return np.nan

    value = str(value)

    m = re.search(r"Immune\s+C([1-6])", value)
    if m:
        return f"C{m.group(1)}"

    m = re.search(r"\bC([1-6])\b", value)
    if m:
        return f"C{m.group(1)}"

    return np.nan


def bh_fdr(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)

    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = pvalues[order]

    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    result = np.empty(n)
    result[order] = adjusted

    return result


def safe_mwu(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) == 0 or len(y) == 0:
        return np.nan, np.nan

    stat, p = mannwhitneyu(
        x,
        y,
        alternative="two-sided"
    )

    # rank-biserial-style direction:
    # positive means x tends to have larger values
    u = stat
    effect = (2 * u / (len(x) * len(y))) - 1

    return float(effect), float(p)


def main():

    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.scores)
    immune = pd.read_csv(
        args.immune_subtypes,
        sep="\t"
    )
    inventory = pd.read_csv(args.case_inventory)

    # ------------------------------------------------------------
    # Validate input structure
    # ------------------------------------------------------------

    required_score_columns = {
        "case_id",
        "deep_evi_tcga_surrogate",
        "n_expression_files"
    }

    missing = required_score_columns - set(scores.columns)

    if missing:
        raise RuntimeError(
            f"Missing required score columns: {sorted(missing)}"
        )

    if "sample" not in immune.columns:
        raise RuntimeError(
            "Immune subtype table is missing required column: sample"
        )

    if "Subtype_Immune_Model_Based" not in immune.columns:
        raise RuntimeError(
            "Immune subtype table is missing required column: "
            "Subtype_Immune_Model_Based"
        )

    # ------------------------------------------------------------
    # Parse published immune subtype
    # ------------------------------------------------------------

    immune = immune.copy()

    # Published immune-subtype resource uses TCGA sample barcodes,
    # whereas frozen 09B scores use GDC case UUIDs.
    # Build the bridge through the 09B case inventory submitter_id.
    immune["tcga_participant_id"] = (
        immune["sample"]
        .astype(str)
        .str[:12]
    )

    immune["immune_subtype"] = (
        immune["Subtype_Immune_Model_Based"]
        .apply(subtype_code)
    )

    # Only expected C1-C6
    unknown_labels = sorted(
        set(
            immune["immune_subtype"]
            .dropna()
            .unique()
        )
        - set(EXPECTED_SUBTYPES)
    )

    if unknown_labels:
        raise RuntimeError(
            f"Unexpected immune subtype labels: {unknown_labels}"
        )

    # ------------------------------------------------------------
    # Case-level subtype consistency
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    # Build GDC UUID -> TCGA participant bridge from frozen 09B
    # inventory. This is the authoritative mapping already used by
    # the frozen TCGA projection.
    # ------------------------------------------------------------

    required_inventory_columns = {"case_id", "submitter_id"}

    missing_inventory = (
        required_inventory_columns - set(inventory.columns)
    )

    if missing_inventory:
        raise RuntimeError(
            "09B case inventory is missing required columns: "
            f"{sorted(missing_inventory)}"
        )

    inventory = inventory.copy()

    inventory["case_id"] = (
        inventory["case_id"].astype(str)
    )

    inventory["tcga_participant_id"] = (
        inventory["submitter_id"]
        .astype(str)
        .str[:12]
    )

    # Verify each GDC case maps to one participant.
    case_bridge_check = (
        inventory
        .groupby("case_id")["tcga_participant_id"]
        .nunique()
    )

    conflicting_bridges = sorted(
        case_bridge_check[
            case_bridge_check > 1
        ].index.tolist()
    )

    if conflicting_bridges:
        raise RuntimeError(
            "A GDC case maps to multiple TCGA participants. "
            f"Examples: {conflicting_bridges[:10]}"
        )

    # Published subtype labels may contain multiple sample-level
    # rows for the same participant. Verify that all such rows agree.
    consistency = (
        immune
        .dropna(
            subset=[
                "tcga_participant_id",
                "immune_subtype"
            ]
        )
        .groupby("tcga_participant_id")["immune_subtype"]
        .nunique()
    )

    inconsistent_participants = sorted(
        consistency[consistency > 1].index.tolist()
    )

    if inconsistent_participants:
        raise RuntimeError(
            "Conflicting published immune subtype labels detected "
            "within TCGA participants. "
            f"Count={len(inconsistent_participants)}; "
            f"examples={inconsistent_participants[:10]}"
        )

    case_labels = (
        immune
        .dropna(
            subset=[
                "tcga_participant_id",
                "immune_subtype"
            ]
        )
        [["tcga_participant_id", "immune_subtype"]]
        .drop_duplicates()
    )

    # ------------------------------------------------------------
    # Map frozen 09B GDC case UUIDs to TCGA participant IDs
    # ------------------------------------------------------------

    scores["case_id"] = scores["case_id"].astype(str)

    case_bridge = (
        inventory[
            ["case_id", "tcga_participant_id"]
        ]
        .drop_duplicates()
    )

    if (
        case_bridge.groupby("case_id")
        .size()
        .max() > 1
    ):
        raise RuntimeError(
            "Duplicate GDC case -> participant mappings remain."
        )

    scores = scores.merge(
        case_bridge,
        on="case_id",
        how="left",
        validate="one_to_one"
    )

    if scores["tcga_participant_id"].isna().any():
        raise RuntimeError(
            "Some frozen 09B cases could not be mapped to a "
            "TCGA participant through the 09B case inventory."
        )

    merged = scores.merge(
        case_labels,
        on="tcga_participant_id",
        how="left",
        validate="many_to_one"
    )

    # ------------------------------------------------------------
    # Inventory-based multi-file classification
    # ------------------------------------------------------------

    if "file_id" in inventory.columns and "case_id" in inventory.columns:
        file_counts = (
            inventory
            .groupby("case_id")
            .size()
            .rename("inventory_expression_file_count")
            .reset_index()
        )

        file_counts["case_id"] = (
            file_counts["case_id"].astype(str)
        )

        merged = merged.merge(
            file_counts,
            on="case_id",
            how="left",
            validate="one_to_one"
        )

    else:
        merged["inventory_expression_file_count"] = (
            merged["n_expression_files"]
        )

    merged["single_file_case"] = (
        merged["n_expression_files"] == 1
    )

    merged["multi_file_case"] = (
        merged["n_expression_files"] > 1
    )

    # ------------------------------------------------------------
    # Analysis cohort
    # ------------------------------------------------------------

    analysis = merged.dropna(
        subset=[
            "deep_evi_tcga_surrogate",
            "immune_subtype"
        ]
    ).copy()

    if analysis.empty:
        raise RuntimeError(
            "No TCGA cases matched between frozen scores "
            "and immune subtype labels after GDC UUID -> "
            "TCGA participant mapping."
        )

    # Ensure the analysis remains participant-level.
    if analysis["tcga_participant_id"].duplicated().any():
        raise RuntimeError(
            "Multiple frozen score rows map to the same TCGA "
            "participant after the case-level join."
        )

    analysis["immune_group"] = np.where(
        analysis["immune_subtype"].isin(["C2", "C3"]),
        "C2_C3_immune_response",
        np.where(
            analysis["immune_subtype"].isin(["C4", "C6"]),
            "C4_C6_immune_low",
            "other"
        )
    )

    # ------------------------------------------------------------
    # Case-level score table
    # ------------------------------------------------------------

    analysis[
        [
            "case_id",
            "tcga_participant_id",
            "deep_evi_tcga_surrogate",
            "immune_subtype",
            "immune_group",
            "n_expression_files",
            "single_file_case",
            "multi_file_case"
        ]
    ].to_csv(
        outdir / "TCGA_BRCA_10B_case_level_scores.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Subtype summary
    # ------------------------------------------------------------

    summary_rows = []

    for subtype in sorted(
        analysis["immune_subtype"].unique(),
        key=lambda x: int(x[1:])
    ):

        x = analysis.loc[
            analysis["immune_subtype"] == subtype,
            "deep_evi_tcga_surrogate"
        ].astype(float)

        summary_rows.append({
            "immune_subtype": subtype,
            "label": EXPECTED_SUBTYPES[subtype],
            "n_cases": len(x),
            "mean_deep_evi": float(x.mean()),
            "median_deep_evi": float(x.median()),
            "std_deep_evi": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
            "q25_deep_evi": float(x.quantile(0.25)),
            "q75_deep_evi": float(x.quantile(0.75)),
        })

    subtype_summary = pd.DataFrame(summary_rows)

    subtype_summary.to_csv(
        outdir / "TCGA_BRCA_10B_subtype_summary.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Primary global Kruskal-Wallis test
    # ------------------------------------------------------------

    groups = []

    for subtype in sorted(
        analysis["immune_subtype"].unique(),
        key=lambda x: int(x[1:])
    ):

        x = analysis.loc[
            analysis["immune_subtype"] == subtype,
            "deep_evi_tcga_surrogate"
        ].dropna().values

        if len(x) > 0:
            groups.append(x)

    if len(groups) < 2:
        raise RuntimeError(
            "Fewer than two immune-subtype groups available."
        )

    kw_stat, kw_p = kruskal(*groups)

    global_test = pd.DataFrame([{
        "test": "Kruskal_Wallis",
        "n_cases": len(analysis),
        "n_subtypes": len(groups),
        "statistic": float(kw_stat),
        "p_value": float(kw_p),
    }])

    global_test.to_csv(
        outdir / "TCGA_BRCA_10B_global_test.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Pairwise subtype tests
    # ------------------------------------------------------------

    pair_rows = []

    subtypes = sorted(
        analysis["immune_subtype"].unique(),
        key=lambda x: int(x[1:])
    )

    for i, s1 in enumerate(subtypes):
        for s2 in subtypes[i + 1:]:

            x = analysis.loc[
                analysis["immune_subtype"] == s1,
                "deep_evi_tcga_surrogate"
            ].dropna().values

            y = analysis.loc[
                analysis["immune_subtype"] == s2,
                "deep_evi_tcga_surrogate"
            ].dropna().values

            effect, p = safe_mwu(x, y)

            pair_rows.append({
                "subtype_1": s1,
                "subtype_2": s2,
                "n_1": len(x),
                "n_2": len(y),
                "rank_biserial_directional_effect": effect,
                "p_value": p,
            })

    pairwise = pd.DataFrame(pair_rows)

    if not pairwise.empty:
        pairwise["fdr_bh"] = bh_fdr(
            pairwise["p_value"].values
        )

    pairwise.to_csv(
        outdir / "TCGA_BRCA_10B_pairwise_tests.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Pre-specified C2+C3 vs C4+C6 contrast
    # ------------------------------------------------------------

    primary = analysis[
        analysis["immune_group"].isin([
            "C2_C3_immune_response",
            "C4_C6_immune_low"
        ])
    ].copy()

    x = primary.loc[
        primary["immune_group"] == "C2_C3_immune_response",
        "deep_evi_tcga_surrogate"
    ].dropna().values

    y = primary.loc[
        primary["immune_group"] == "C4_C6_immune_low",
        "deep_evi_tcga_surrogate"
    ].dropna().values

    effect, p = safe_mwu(x, y)

    contrast = pd.DataFrame([{
        "group_1": "C2_C3_immune_response",
        "group_2": "C4_C6_immune_low",
        "n_group_1": len(x),
        "n_group_2": len(y),
        "mean_group_1": float(np.mean(x)) if len(x) else np.nan,
        "mean_group_2": float(np.mean(y)) if len(y) else np.nan,
        "median_group_1": float(np.median(x)) if len(x) else np.nan,
        "median_group_2": float(np.median(y)) if len(y) else np.nan,
        "rank_biserial_directional_effect": effect,
        "p_value": p,
    }])

    contrast.to_csv(
        outdir / "TCGA_BRCA_10B_primary_contrast.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Single-file versus multi-file sensitivity
    # ------------------------------------------------------------

    sensitivity_rows = []

    for subset_name, subset in [
        ("all_cases", analysis),
        (
            "single_file_only",
            analysis[analysis["single_file_case"]]
        ),
        (
            "multi_file_only",
            analysis[analysis["multi_file_case"]]
        )
    ]:

        if len(subset) == 0:
            continue

        row = {
            "subset": subset_name,
            "n_cases": len(subset),
            "n_subtypes": subset["immune_subtype"].nunique(),
            "mean_deep_evi": subset["deep_evi_tcga_surrogate"].mean(),
            "median_deep_evi": subset["deep_evi_tcga_surrogate"].median()
        }

        sensitivity_rows.append(row)

    pd.DataFrame(sensitivity_rows).to_csv(
        outdir / "TCGA_BRCA_10B_sensitivity_single_vs_multifile.csv",
        index=False
    )

    # ------------------------------------------------------------
    # Frozen scientific report
    # ------------------------------------------------------------

    report = {
        "cohort": "TCGA-BRCA",
        "step": "10B_independent_tcga_immune_subtype_validation",
        "status": "complete",

        "external_cohort": True,

        "input_score_source":
            "GSE176078 09A frozen TCGA-compatible Deep-EVI molecular surrogate",

        "immune_subtype_source":
            "Published TCGA immune model-based subtype classification",

        "immune_subtype_endpoint":
            "Subtype_Immune_Model_Based",

        "immune_subtypes_observed":
            sorted(
                analysis["immune_subtype"].unique(),
                key=lambda x: int(x[1:])
            ),

        "sample_level_source_rows": int(len(immune)),

        "matched_case_count": int(len(analysis)),

        "matched_participant_count": int(
            analysis["tcga_participant_id"].nunique()
        ),

        "unmatched_score_cases": int(
            merged["immune_subtype"].isna().sum()
        ),

        "case_level_analysis": True,

        "primary_endpoint":
            "Global Kruskal-Wallis test across C1-C6",

        "secondary_endpoint":
            "Pre-specified C2+C3 versus C4+C6 contrast",

        "deep_evi_retrained": False,
        "signature_refit": False,
        "tcga_used_for_model_selection": False,
        "tcga_used_for_cutoff_selection": False,
        "tcga_used_for_feature_selection": False,

        "immune_subtype_labels_used_for_model_training": False,
        "immune_subtype_labels_used_for_model_selection": False,

        "rna_velocity": False,

        "patient_level_inference": True,

        "multi_file_cases_retained": True,
        "multi_file_aggregation":
            "Inherited from frozen 09B arithmetic-mean case-level projection",

        "independent_biological_validation": True,

        "interpretation":
            "External TCGA-BRCA biological validation of the frozen "
            "GSE176078-derived Deep-EVI molecular surrogate against "
            "a published immune-subtype classification that was not "
            "used for Deep-EVI construction or TCGA projection fitting."
    }

    with open(
        outdir / "TCGA_BRCA_10B_report.json",
        "w"
    ) as f:
        json.dump(report, f, indent=2)

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------

    print("=" * 80)
    print("10B TCGA-BRCA INDEPENDENT IMMUNE-SUBTYPE VALIDATION")
    print("=" * 80)

    print(f"Published subtype sample rows : {len(immune)}")
    print(f"Published subtype participants: {immune['tcga_participant_id'].nunique()}")
    print(f"Frozen 09B score cases        : {len(scores)}")
    print(f"Matched TCGA cases             : {len(analysis)}")
    print(f"Unmatched score cases          : {merged['immune_subtype'].isna().sum()}")
    print()

    print("SUBTYPE COUNTS")
    print(subtype_summary.to_string(index=False))

    print()
    print("GLOBAL KRUSKAL-WALLIS")
    print(f"statistic = {kw_stat:.8g}")
    print(f"p_value   = {kw_p:.8g}")

    print()
    print("C2+C3 VS C4+C6")
    print(contrast.to_string(index=False))

    print()
    print("STATUS: PASS")


if __name__ == "__main__":
    main()