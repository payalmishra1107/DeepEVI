from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    ".gitignore",
    "nextflow.config",
    "main.nf",
    "main_step1.nf",
    "main_step2.nf",
    "main_step3.nf",
    "main_step4.nf",
    "main_step5.nf",
    "main_step6.nf",
    "main_step6a.nf",
    "main_step6b.nf",
    "main_step7a.nf",
    "main_step7b.nf",
    "main_step7c.nf",
    "main_step8a.nf",
    "main_step8b.nf",
    "main_step8c.nf",
    "main_step8d.nf",
    "main_step8e.nf",
    "main_step9a.nf",
    "main_step9b.nf",
    "main_step9c.nf",
    "main_step9d.nf",
    "main_step9e.nf",
    "main_step10a.nf",
    "main_step10a5.nf",
    "main_step10a5b.nf",
    "main_step10a_freeze.nf",
    "main_step10b.nf",
    "main_step11.nf",
    "workflows/01_ingestion.nf",
    "workflows/02_preprocessing.nf",
    "workflows/03_integration.nf",
    "workflows/04_tcell_analysis.nf",
    "workflows/05_deep_evi.nf",
    "workflows/06_tcga_validation.nf",
    "workflows/07_heldout_validation.nf",
    "workflows/08_xai.nf",
    "workflows/09_audit.nf",
    "modules/08_deep_evi.nf",
    "modules/09b_tcga_projection.nf",
    "modules/09c_tcga_clinical.nf",
    "modules/10a_test_expression.nf",
    "modules/10a_reference_scoring.nf",
    "modules/10a_heldout_benchmark.nf",
    "modules/10b_tcga_immune_validation.nf",
    "modules/11_deep_evi_xai.nf",
    "bin/train_deep_evi.py",
    "bin/run_deep_evi_xai.py",
    "envs/deep_evi.yml",
    "envs/deepevi_xai.yml",
    "conf/workstation.config",
    "docs/PIPELINE_MAP.md",
]

def test_production_layout():
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    assert not missing, f"Missing production files: {missing}"

if __name__ == "__main__":
    test_production_layout()
    print(f"PASS: {len(REQUIRED)} required production paths are present.")
