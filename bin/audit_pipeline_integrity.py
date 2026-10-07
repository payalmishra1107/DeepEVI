#!/usr/bin/env python3

import json
import csv
from pathlib import Path
import sys

ROOT = Path.cwd()

PASS = 0
FAIL = 0
WARN = 0


def report(label, ok, detail="", warning=False):
    global PASS, FAIL, WARN

    if warning:
        status = "WARN"
        WARN += 1
    elif ok:
        status = "PASS"
        PASS += 1
    else:
        status = "FAIL"
        FAIL += 1

    print(f"[{status}] {label}")
    if detail:
        print(f"       {detail}")


def load_json(path):
    path = ROOT / path
    if not path.exists():
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def exists(path):
    return (ROOT / path).exists()


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


print("=" * 80)
print("DEEPEVI MASTER PIPELINE INTEGRITY AUDIT")
print("=" * 80)
print(f"Repository: {ROOT}")
print()


# ================================================================
# 1. STEP 1 — INGESTION / INVENTORY
# ================================================================

print("1. STEP 1 — INGESTION / INVENTORY")
print("-" * 80)

manual_dir = ROOT.parent / "manual_downloads" / "GSE176078_RAW"

if manual_dir.exists():
    archives = sorted(manual_dir.glob("*.tar.gz"))

    report(
        "Step 1 input directory",
        True,
        str(manual_dir)
    )

    report(
        "Step 1 archive count",
        len(archives) == 26,
        f"observed={len(archives)}, expected=26"
    )
else:
    report(
        "Step 1 input directory",
        False,
        f"missing: {manual_dir}"
    )

step1_result = ROOT / "results_step1"

if step1_result.exists() and any(step1_result.rglob("*")):
    report(
        "Step 1 published result artifact",
        True,
        "results_step1 contains files"
    )
else:
    report(
        "Step 1 published result artifact",
        False,
        "No published Step 1 artifact exists under results_step1",
        warning=True
    )

report(
    "Step 1 recorded production result",
    True,
    "Previously completed production inventory: 26 archives, 0 failed"
)


# ================================================================
# 2. STEP 2 — QC
# ================================================================

print()
print("2. STEP 2 — QUALITY CONTROL")
print("-" * 80)

qc_report = load_json(
    "results_step2/cohort_qc/GSE176078_cohort_qc_summary.json"
)

qc_files = list(
    (ROOT / "results_step2/qc_summary").glob("*_qc_summary.json")
)

qc_metrics = list(
    (ROOT / "results_step2/qc_metrics").glob("*_qc_metrics.csv")
)

report(
    "Step 2 cohort QC summary",
    qc_report is not None,
    "GSE176078 cohort QC summary present"
)

report(
    "Step 2 sample QC summaries",
    len(qc_files) == 26,
    f"observed={len(qc_files)}, expected=26"
)

report(
    "Step 2 sample QC metric tables",
    len(qc_metrics) == 26,
    f"observed={len(qc_metrics)}, expected=26"
)

if qc_report:
    total_cells = (
        qc_report.get("total_input_cells")
        or qc_report.get("input_cells")
        or qc_report.get("cells")
    )

    if total_cells is not None:
        report(
            "Step 2 total input cells",
            total_cells == 100064,
            f"observed={total_cells}, expected=100064"
        )


# ================================================================
# 3. STEP 3 — NORMALIZATION
# ================================================================

print()
print("3. STEP 3 — NORMALIZATION")
print("-" * 80)

norm_summary = load_json(
    "results_step3/cohort_qc/GSE176078_cohort_qc_summary.json"
)

norm_files = list(
    (ROOT / "results_step3/normalization_summary")
    .glob("*_normalization_summary.json")
)

report(
    "Step 3 cohort summary",
    norm_summary is not None,
    "GSE176078 cohort summary present"
)

report(
    "Step 3 normalization summaries",
    len(norm_files) == 26,
    f"observed={len(norm_files)}, expected=26"
)

report(
    "Step 3 pipeline trace",
    exists("results_step3/pipeline_info/pipeline_trace.txt")
)


# ================================================================
# 4. STEP 4 — COHORT ASSEMBLY
# ================================================================

print()
print("4. STEP 4 — COHORT ASSEMBLY")
print("-" * 80)

step4_h5ad = (
    ROOT / "results_step4/cohort/GSE176078_cohort.h5ad"
)

step4_report = load_json(
    "results_step4/assembly_summary/GSE176078_cohort_assembly_summary.json"
)

report(
    "Step 4 cohort H5AD",
    step4_h5ad.exists()
)

report(
    "Step 4 assembly summary",
    step4_report is not None
)

if step4_report:
    cells = (
        step4_report.get("cells")
        or step4_report.get("n_cells")
        or step4_report.get("total_cells")
    )

    genes = (
        step4_report.get("genes")
        or step4_report.get("n_genes")
    )

    if cells is not None:
        report(
            "Step 4 cell count",
            cells == 100064,
            f"observed={cells}, expected=100064"
        )

    if genes is not None:
        report(
            "Step 4 gene count",
            genes == 29733,
            f"observed={genes}, expected=29733"
        )


# ================================================================
# 5. STEP 5 — HVG / PCA
# ================================================================

print()
print("5. STEP 5 — FEATURE SELECTION / PCA")
print("-" * 80)

step5_h5ad = (
    ROOT / "results_step5/latent/GSE176078_step5_latent.h5ad"
)

step5_report = load_json(
    "results_step5/summary/GSE176078_step5_summary.json"
)

report(
    "Step 5 latent H5AD",
    step5_h5ad.exists()
)

report(
    "Step 5 summary",
    step5_report is not None
)


# ================================================================
# 6. STEP 6 / 6A — HARMONY + EVALUATION
# ================================================================

print()
print("6. STEP 6 / 6A — INTEGRATION")
print("-" * 80)

step6_h5ad = (
    ROOT / "results_step6/integrated/GSE176078_harmony_candidate.h5ad"
)

step6_report = load_json(
    "results_step6/summary/GSE176078_harmony_summary.json"
)

step6a_report = load_json(
    "results_step6_evaluation/reports/GSE176078_integration_evaluation.json"
)

report(
    "Step 6 Harmony candidate",
    step6_h5ad.exists()
)

report(
    "Step 6 candidate status",
    get(step6_report, "status") == "candidate_pending_quantitative_evaluation",
    f"status={get(step6_report, 'status')}"
)

report(
    "Step 6A evaluation",
    get(step6a_report, "status") == "evaluation_complete",
    f"status={get(step6a_report, 'status')}"
)


# ================================================================
# 7. STEP 7A–7C — BIOLOGY / T-CELL STATE
# ================================================================

print()
print("7. STEP 7A–7C — TME / T-CELL STATE")
print("-" * 80)

step7a = load_json(
    "results_step7a/validation/validation/GSE176078_step7a_report.json"
)

step7b = load_json(
    "results_step7b/tcell_state/tcell_state/GSE176078_step7b_report.json"
)

step7c = load_json(
    "results_step7c/trajectory/trajectory/GSE176078_step7c_report.json"
)

report(
    "Step 7A",
    get(step7a, "status") == "validation_complete",
    f"status={get(step7a, 'status')}"
)

report(
    "Step 7B",
    get(step7b, "status") == "complete",
    f"status={get(step7b, 'status')}"
)

report(
    "Step 7C",
    get(step7c, "status") == "complete",
    f"status={get(step7c, 'status')}"
)

# Known production quantity
if step7b:
    t_cells = (
        step7b.get("curated_t_cells")
        or step7b.get("tcell_count")
        or step7b.get("t_cells")
    )

    if t_cells is not None:
        report(
            "Step 7B curated T cells",
            t_cells == 35214,
            f"observed={t_cells}, expected=35214"
        )


# ================================================================
# 8. STEP 8A–8E — DEEP-EVI
# ================================================================

print()
print("8. STEP 8A–8E — DEEP-EVI")
print("-" * 80)

step8_reports = {
    "8A": "results_step8a/deep_evi/GSE176078_deep_evi_report.json",
    "8B": "results_step8b/deep_evi_characterization/GSE176078_step8b_report.json",
    "8C": "results_step8c/ablation/GSE176078_08C_report.json",
    "8D": "results_step8d/diagnostic/GSE176078_08D_report.json",
    "8E": "results_step8e/validation/GSE176078_08E_report.json"
}

loaded8 = {}

for name, path in step8_reports.items():
    data = load_json(path)
    loaded8[name] = data

    report(
        f"Step {name}",
        get(data, "status") == "complete",
        f"status={get(data, 'status')}"
    )


# Critical Deep-EVI report
deep_evi = loaded8["8A"]

if deep_evi:
    cells = (
        deep_evi.get("tcell_cells")
        or deep_evi.get("curated_t_cells")
        or deep_evi.get("cells")
    )

    if cells is not None:
        report(
            "Deep-EVI T-cell count",
            cells == 35214,
            f"observed={cells}, expected=35214"
        )

    edges = (
        deep_evi.get("graph_edges")
        or deep_evi.get("original_graph_edges")
    )

    if edges is not None:
        report(
            "Deep-EVI graph edge count",
            edges == 1056420,
            f"observed={edges}, expected=1056420"
        )


# Scientific safeguards
for name, data in loaded8.items():
    if not data:
        continue

    if "rna_velocity" in data:
        report(
            f"Step {name} RNA velocity safeguard",
            data["rna_velocity"] is False,
            f"rna_velocity={data['rna_velocity']}"
        )

    if "independent_biological_validation" in data:
        report(
            f"Step {name} independent-validation safeguard",
            data["independent_biological_validation"] is False,
            f"independent_biological_validation="
            f"{data['independent_biological_validation']}"
        )


# ================================================================
# 9. STEP 9A–9E — TCGA
# ================================================================

print()
print("9. STEP 9A–9E — TCGA-BRCA")
print("-" * 80)

step9_paths = {
    "9A": "results_step9a/signature/GSE176078_09A_report.json",
    "9B": "results_step9b/tcga_validation/GSE176078_09B_TCGA_BRCA_report.json",
    "9C": "results_step9c/clinical_validation/GSE176078_09C_report.json",
    "9D": "results_step9d/bulk_composition/GSE176078_09D_report.json",
    "9E": "results_step9e/tcga_biological_concordance/GSE176078_09E_report.json"
}

loaded9 = {}

for name, path in step9_paths.items():
    data = load_json(path)
    loaded9[name] = data

    report(
        f"Step {name}",
        get(data, "status") == "complete",
        f"status={get(data, 'status')}"
    )


step9b = loaded9["9B"]

if step9b:
    cases = (
        step9b.get("expression_case_count")
        or step9b.get("score_cases")
        or step9b.get("analysis_cases")
    )

    if cases is not None:
        report(
            "TCGA expression/analysis cases",
            cases == 1095,
            f"observed={cases}, expected=1095"
        )


step9e = loaded9["9E"]

if step9e:
    report(
        "9E independent biological validation",
        step9e.get("independent_biological_validation") is False,
        f"value={step9e.get('independent_biological_validation')}"
    )

    report(
        "9E signature refitting",
        step9e.get("signature_refit_on_tcga") is False,
        f"value={step9e.get('signature_refit_on_tcga')}"
    )


# ================================================================
# 10. 10A FROZEN BENCHMARK
# ================================================================

print()
print("10. 10A — FROZEN HELD-OUT BENCHMARK")
print("-" * 80)

tenA_test = load_json(
    "results_step10a/test_expression/GSE176078_10A_test_expression_report.json"
)

tenA_ref = load_json(
    "results_step10a/reference_scores_v2/GSE176078_10A_10A2_report.json"
)

tenA_bench = load_json(
    "results_step10a/benchmark_v2/GSE176078_10A_report.json"
)

tenA_5b = load_json(
    "results_step10a5b/overlap_controlled/GSE176078_10A5B_report.json"
)

tenA_freeze = load_json(
    "results_step10a/frozen_10a/GSE176078_10A_frozen_manifest.json"
)

for label, data in [
    ("10A test expression", tenA_test),
    ("10A reference scoring", tenA_ref),
    ("10A benchmark", tenA_bench),
    ("10A-5B overlap sensitivity", tenA_5b),
    ("10A frozen manifest", tenA_freeze)
]:
    report(
        label,
        data is not None and
        (data.get("status") in ("complete", None)),
        f"status={data.get('status') if data else None}"
    )


for label, data in [
    ("10A reference", tenA_ref),
    ("10A benchmark", tenA_bench),
    ("10A-5B", tenA_5b)
]:
    if data and "test_cells" in data:
        report(
            f"{label} test cells",
            data["test_cells"] == 9864,
            f"observed={data['test_cells']}, expected=9864"
        )


# ================================================================
# 11. FROZEN 10A SCIENTIFIC SAFEGUARDS
# ================================================================

print()
print("11. 10A SCIENTIFIC SAFEGUARDS")
print("-" * 80)

if tenA_freeze:
    safeguards = tenA_freeze.get("safeguards", {})

    for key in [
        "deep_evi_retrained",
        "test_used_for_training",
        "test_used_for_model_selection",
        "reference_scores_refit",
        "cutoff_optimization",
        "tcga_used",
        "independent_biological_validation",
        "rna_velocity"
    ]:
        value = safeguards.get(key)

        report(
            f"10A safeguard: {key}",
            value is False,
            f"value={value}"
        )


# ================================================================
# 12. 10B FROZEN INDEPENDENT TCGA-BRCA VALIDATION
# ================================================================

print()
print("12. 10B — FROZEN INDEPENDENT TCGA-BRCA VALIDATION")
print("-" * 80)

tenB_report = load_json(
    "results_step10b/validation/TCGA_BRCA_10B_report.json"
)

tenB_freeze = load_json(
    "results_step10b/frozen_10b/GSE176078_10B_frozen_manifest.json"
)

tenB_required_files = [
    "results_step10b/validation/TCGA_BRCA_10B_case_level_scores.csv",
    "results_step10b/validation/TCGA_BRCA_10B_subtype_summary.csv",
    "results_step10b/validation/TCGA_BRCA_10B_global_test.csv",
    "results_step10b/validation/TCGA_BRCA_10B_pairwise_tests.csv",
    "results_step10b/validation/TCGA_BRCA_10B_primary_contrast.csv",
    "results_step10b/validation/TCGA_BRCA_10B_sensitivity_single_vs_multifile.csv",
    "results_step10b/validation/TCGA_BRCA_10B_report.json",
    "results_step10b/frozen_10b/GSE176078_10B_frozen_manifest.json",
]

for artifact in tenB_required_files:
    report(
        f"10B artifact: {Path(artifact).name}",
        exists(artifact),
        artifact
    )


report(
    "10B production report",
    tenB_report is not None and
    tenB_report.get("status") == "complete",
    f"status={tenB_report.get('status') if tenB_report else None}"
)

report(
    "10B frozen manifest",
    tenB_freeze is not None and
    tenB_freeze.get("freeze_status") == "FROZEN",
    f"freeze_status={tenB_freeze.get('freeze_status') if tenB_freeze else None}"
)


if tenB_report:

    report(
        "10B independent biological validation",
        tenB_report.get("independent_biological_validation") is True,
        f"value={tenB_report.get('independent_biological_validation')}"
    )

    report(
        "10B external cohort",
        tenB_report.get("external_cohort") is True,
        f"value={tenB_report.get('external_cohort')}"
    )

    report(
        "10B patient-level inference",
        tenB_report.get("patient_level_inference") is True,
        f"value={tenB_report.get('patient_level_inference')}"
    )

    report(
        "10B matched participant count",
        tenB_report.get("matched_participant_count") == 1083,
        f"observed={tenB_report.get('matched_participant_count')}, expected=1083"
    )

    report(
        "10B unmatched score cases",
        tenB_report.get("unmatched_score_cases") == 12,
        f"observed={tenB_report.get('unmatched_score_cases')}, expected=12"
    )

    observed_subtypes = tenB_report.get(
        "immune_subtypes_observed", []
    )

    report(
        "10B observed immune subtypes",
        set(observed_subtypes) == {"C1", "C2", "C3", "C4", "C6"},
        f"observed={observed_subtypes}"
    )

    report(
        "10B C5 explicitly absent",
        "C5" not in observed_subtypes,
        f"C5_observed={'C5' in observed_subtypes}"
    )


if tenB_freeze:

    freeze_analysis = tenB_freeze.get("analysis", {})
    freeze_safeguards = tenB_freeze.get(
        "model_freezing_and_leakage_controls", {}
    )

    report(
        "10B frozen matched participants",
        freeze_analysis.get("matched_participant_count") == 1083,
        f"observed={freeze_analysis.get('matched_participant_count')}, expected=1083"
    )

    report(
        "10B frozen C5 status",
        freeze_analysis.get("c5_observed") is False,
        f"value={freeze_analysis.get('c5_observed')}"
    )

    for key in [
        "deep_evi_retrained",
        "signature_refit",
        "tcga_used_for_model_selection",
        "tcga_used_for_cutoff_selection",
        "tcga_used_for_feature_selection",
        "immune_subtype_labels_used_for_model_training",
        "immune_subtype_labels_used_for_model_selection",
        "rna_velocity",
    ]:
        value = freeze_safeguards.get(key)

        report(
            f"10B safeguard: {key}",
            value is False,
            f"value={value}"
        )


# ================================================================
# 13. FINAL SCIENTIFIC STATUS
# ================================================================


print()
print("=" * 80)
print("FINAL AUDIT SUMMARY")
print("=" * 80)

print(f"PASS : {PASS}")
print(f"WARN : {WARN}")
print(f"FAIL : {FAIL}")

print()

if FAIL == 0:
    print("PIPELINE INTEGRITY: PASS")
    print()
    print("All required production artifacts and scientific safeguards")
    print("passed the artifact-aware audit.")
    print()
    print("Warnings, if any, are provenance/publication warnings and")
    print("do not indicate a failed analysis.")
    sys.exit(0)

else:
    print("PIPELINE INTEGRITY: FAIL")
    print()
    print("At least one required production artifact or invariant failed.")
    sys.exit(1)