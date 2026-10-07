#!/usr/bin/env python3

from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone

ROOT = Path(".")
RESULTS = ROOT / "results_step10b" / "validation"
IMMUNE_META = (
    ROOT
    / "tcga_brca"
    / "metadata"
    / "immune_subtypes"
)

REPORT = RESULTS / "TCGA_BRCA_10B_report.json"
GLOBAL = RESULTS / "TCGA_BRCA_10B_global_test.csv"
CONTRAST = RESULTS / "TCGA_BRCA_10B_primary_contrast.csv"
SUBTYPE = RESULTS / "TCGA_BRCA_10B_subtype_summary.csv"
PAIRWISE = RESULTS / "TCGA_BRCA_10B_pairwise_tests.csv"
SENSITIVITY = (
    RESULTS / "TCGA_BRCA_10B_sensitivity_single_vs_multifile.csv"
)
CASE_LEVEL = RESULTS / "TCGA_BRCA_10B_case_level_scores.csv"

SOURCE_GZ = (
    IMMUNE_META / "Subtype_Immune_Model_Based.txt.gz"
)
SOURCE_SHA = (
    IMMUNE_META / "Subtype_Immune_Model_Based.txt.gz.sha256"
)
SOURCE_TXT = (
    IMMUNE_META / "Subtype_Immune_Model_Based.txt"
)

FREEZE_DIR = ROOT / "results_step10b" / "frozen_10b"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


required = [
    REPORT,
    GLOBAL,
    CONTRAST,
    SUBTYPE,
    PAIRWISE,
    SENSITIVITY,
    CASE_LEVEL,
    SOURCE_GZ,
    SOURCE_SHA,
    SOURCE_TXT,
]

missing = [str(p) for p in required if not p.exists()]

if missing:
    raise SystemExit(
        "Cannot freeze 10B. Missing required artifacts:\n"
        + "\n".join(missing)
    )


with open(REPORT) as f:
    report = json.load(f)


# Confirm the production report has the expected completion state.
if report.get("status") != "complete":
    raise SystemExit(
        f"10B report status is not complete: "
        f"{report.get('status')}"
    )

if not report.get("independent_biological_validation", False):
    raise SystemExit(
        "10B report does not declare independent biological validation."
    )


# Source checksum.
source_sha256 = sha256(SOURCE_GZ)


# Read the checksum file without assuming its exact formatting.
checksum_text = SOURCE_SHA.read_text().strip()

manifest = {
    "cohort": "TCGA-BRCA",
    "step": "10B_independent_tcga_immune_subtype_validation",
    "freeze_status": "FROZEN",
    "freeze_timestamp_utc": datetime.now(
        timezone.utc
    ).isoformat(),

    "external_cohort": True,
    "independent_biological_validation": True,

    "input_score_source": (
        "GSE176078 09A frozen TCGA-compatible "
        "Deep-EVI molecular surrogate"
    ),

    "immune_subtype_source": (
        "Published TCGA immune model-based subtype classification"
    ),

    "immune_subtype_endpoint": "Subtype_Immune_Model_Based",

    "source_artifact": {
        "compressed_file": str(SOURCE_GZ),
        "decompressed_file": str(SOURCE_TXT),
        "sha256_file": str(SOURCE_SHA),
        "computed_sha256": source_sha256,
        "recorded_sha256_file_contents": checksum_text,
    },

    "analysis": {
        "sample_level_source_rows": report.get(
            "sample_level_source_rows"
        ),
        "matched_case_count": report.get(
            "matched_case_count"
        ),
        "matched_participant_count": report.get(
            "matched_participant_count"
        ),
        "unmatched_score_cases": report.get(
            "unmatched_score_cases"
        ),
        "case_level_analysis": True,
        "patient_level_inference": True,

        "observed_subtypes": report.get(
            "immune_subtypes_observed"
        ),

        "c5_observed": "C5" in report.get(
            "immune_subtypes_observed", []
        ),

        "primary_endpoint": (
            "Global Kruskal-Wallis test across observed "
            "TCGA immune subtypes"
        ),

        "secondary_endpoint": (
            "Pre-specified C2+C3 versus C4+C6 contrast"
        ),
    },

    "model_freezing_and_leakage_controls": {
        "deep_evi_retrained": False,
        "signature_refit": False,
        "tcga_used_for_model_selection": False,
        "tcga_used_for_cutoff_selection": False,
        "tcga_used_for_feature_selection": False,
        "immune_subtype_labels_used_for_model_training": False,
        "immune_subtype_labels_used_for_model_selection": False,
        "rna_velocity": False,
    },

    "multi_file_handling": {
        "multi_file_cases_retained": True,
        "aggregation": (
            "Arithmetic mean across expression files belonging "
            "to the same TCGA case, inherited from frozen 09B."
        ),
    },

    "interpretation": (
        "External TCGA-BRCA biological validation of the frozen "
        "GSE176078-derived Deep-EVI molecular surrogate against "
        "a published immune-subtype classification that was not "
        "used for Deep-EVI construction or TCGA projection fitting."
    ),

    "scientific_scope": {
        "independent_biological_validation": True,
        "clinical_prediction": False,
        "causal_inference": False,
        "rna_velocity": False,
        "temporal_trajectory": False,
        "patient_level_inference": True,
    },

    "artifact_inventory": {
        "report": str(REPORT),
        "case_level_scores": str(CASE_LEVEL),
        "subtype_summary": str(SUBTYPE),
        "global_test": str(GLOBAL),
        "pairwise_tests": str(PAIRWISE),
        "primary_contrast": str(CONTRAST),
        "single_vs_multifile_sensitivity": str(SENSITIVITY),
    },
}


FREEZE_DIR.mkdir(parents=True, exist_ok=True)

out = FREEZE_DIR / "GSE176078_10B_frozen_manifest.json"

with open(out, "w") as f:
    json.dump(manifest, f, indent=2)

print("=" * 80)
print("10B FREEZE")
print("=" * 80)
print(f"Status : FROZEN")
print(f"Output : {out}")
print(f"SHA256 : {source_sha256}")
print(
    f"Matched participants : "
    f"{manifest['analysis']['matched_participant_count']}"
)
print(
    f"Observed subtypes : "
    f"{manifest['analysis']['observed_subtypes']}"
)
print("STATUS: PASS")