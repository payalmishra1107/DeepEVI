#!/usr/bin/env python3

import json
import time
from pathlib import Path

import requests


GDC_URL = "https://api.gdc.cancer.gov/cases"

OUT_DIR = Path(
    Path.home()
    / "deepevi"
    / "tcga_brca"
    / "clinical"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

PROJECT = "TCGA-BRCA"
PAGE_SIZE = 500


FILTERS = {
    "op": "in",
    "content": {
        "field": "cases.project.project_id",
        "value": [PROJECT],
    },
}


FIELDS = [
    "case_id",
    "submitter_id",
    "project.project_id",
    "primary_site",
    "disease_type",

    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.days_to_birth",
    "demographic.age_at_index",
    "demographic.sex_at_birth",

    "diagnoses.age_at_diagnosis",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.days_to_last_known_disease_status",
    "diagnoses.ajcc_pathologic_stage",
    "diagnoses.ajcc_pathologic_t",
    "diagnoses.ajcc_pathologic_n",
    "diagnoses.ajcc_pathologic_m",
    "diagnoses.ajcc_clinical_stage",
    "diagnoses.classification_of_tumor",
    "diagnoses.primary_diagnosis",

    "follow_ups.days_to_follow_up",
]


def get_nested(obj, path):
    current = obj

    for part in path.split("."):
        if isinstance(current, list):
            if not current:
                return None
            current = current[0]

        if not isinstance(current, dict):
            return None

        current = current.get(part)

        if current is None:
            return None

    if isinstance(current, list):
        return current[0] if current else None

    return current


def first_diagnosis(case):
    diagnoses = case.get("diagnoses", [])

    if not diagnoses:
        return {}

    # Prefer primary disease if explicitly marked.
    for diagnosis in diagnoses:
        if diagnosis.get("diagnosis_is_primary_disease") is True:
            return diagnosis

    return diagnoses[0]


def first_followup(case):
    followups = case.get("follow_ups", [])

    if not followups:
        return {}

    # Select the latest numerical follow-up when possible.
    valid = [
        x for x in followups
        if x.get("days_to_follow_up") is not None
    ]

    if valid:
        return max(
            valid,
            key=lambda x: float(x["days_to_follow_up"])
        )

    return followups[0]


def flatten_case(case):
    diagnosis = first_diagnosis(case)
    followup = first_followup(case)

    row = {
        "case_id": case.get("case_id"),
        "submitter_id": case.get("submitter_id"),
        "project_id": (
            case.get("project", {}) or {}
        ).get("project_id"),
        "primary_site": case.get("primary_site"),
        "disease_type": case.get("disease_type"),

        "vital_status": (
            case.get("demographic", {}) or {}
        ).get("vital_status"),

        "days_to_death": (
            case.get("demographic", {}) or {}
        ).get("days_to_death"),

        "days_to_birth": (
            case.get("demographic", {}) or {}
        ).get("days_to_birth"),

        "age_at_index": (
            case.get("demographic", {}) or {}
        ).get("age_at_index"),

        "sex_at_birth": (
            case.get("demographic", {}) or {}
        ).get("sex_at_birth"),

        "age_at_diagnosis": diagnosis.get(
            "age_at_diagnosis"
        ),

        "days_to_last_follow_up": diagnosis.get(
            "days_to_last_follow_up"
        ),

        "days_to_last_known_disease_status": diagnosis.get(
            "days_to_last_known_disease_status"
        ),

        "ajcc_pathologic_stage": diagnosis.get(
            "ajcc_pathologic_stage"
        ),

        "ajcc_pathologic_t": diagnosis.get(
            "ajcc_pathologic_t"
        ),

        "ajcc_pathologic_n": diagnosis.get(
            "ajcc_pathologic_n"
        ),

        "ajcc_pathologic_m": diagnosis.get(
            "ajcc_pathologic_m"
        ),

        "ajcc_clinical_stage": diagnosis.get(
            "ajcc_clinical_stage"
        ),

        "classification_of_tumor": diagnosis.get(
            "classification_of_tumor"
        ),

        "primary_diagnosis": diagnosis.get(
            "primary_diagnosis"
        ),

        "latest_followup_days": followup.get(
            "days_to_follow_up"
        ),
    }

    return row


def main():

    print("=" * 80)
    print("TCGA-BRCA CLINICAL DATA ACQUISITION")
    print("=" * 80)

    rows = []
    raw_cases = []

    offset = 0

    while True:

        params = {
            "filters": json.dumps(FILTERS),
            "fields": ",".join(FIELDS),
            "format": "JSON",
            "size": PAGE_SIZE,
            "from": offset,
        }

        print(
            f"Requesting cases {offset + 1}"
            f"-{offset + PAGE_SIZE} ..."
        )

        response = requests.get(
            GDC_URL,
            params=params,
            timeout=120,
        )

        response.raise_for_status()

        payload = response.json()

        hits = payload["data"]["hits"]
        pagination = payload["data"]["pagination"]

        raw_cases.extend(hits)

        for case in hits:
            rows.append(flatten_case(case))

        total = pagination["total"]

        offset += len(hits)

        print(
            f"  received {len(hits)} "
            f"(total {offset}/{total})"
        )

        if offset >= total or not hits:
            break

        time.sleep(0.2)

    # Deduplicate by case ID.
    unique = {}

    for row in rows:
        unique[row["case_id"]] = row

    rows = list(unique.values())

    raw_path = OUT_DIR / "TCGA-BRCA_cases_raw.json"

    with open(raw_path, "w") as handle:
        json.dump(
            raw_cases,
            handle,
            indent=2,
        )

    import csv

    table_path = OUT_DIR / "TCGA-BRCA_clinical.tsv"

    columns = [
        "case_id",
        "submitter_id",
        "project_id",
        "primary_site",
        "disease_type",
        "vital_status",
        "days_to_death",
        "days_to_birth",
        "age_at_index",
        "sex_at_birth",
        "age_at_diagnosis",
        "days_to_last_follow_up",
        "days_to_last_known_disease_status",
        "latest_followup_days",
        "ajcc_pathologic_stage",
        "ajcc_pathologic_t",
        "ajcc_pathologic_n",
        "ajcc_pathologic_m",
        "ajcc_clinical_stage",
        "classification_of_tumor",
        "primary_diagnosis",
    ]

    with open(
        table_path,
        "w",
        newline="",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            delimiter="\t",
        )

        writer.writeheader()

        for row in sorted(
            rows,
            key=lambda x: x["submitter_id"] or "",
        ):
            writer.writerow(row)

    metadata = {
        "project": PROJECT,
        "endpoint": GDC_URL,
        "case_count_returned": len(raw_cases),
        "unique_case_count": len(rows),
        "page_size": PAGE_SIZE,
        "fields": FIELDS,
    }

    metadata_path = (
        OUT_DIR
        / "TCGA-BRCA_clinical_metadata.json"
    )

    with open(metadata_path, "w") as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print()
    print("Clinical cases:", len(rows))
    print("Raw JSON:", raw_path)
    print("Clinical TSV:", table_path)
    print("Metadata:", metadata_path)
    print()
    print("=" * 80)
    print("CLINICAL ACQUISITION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()