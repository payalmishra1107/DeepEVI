from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = [
    "main.nf",
    "main_step1.nf",
    "nextflow.config",
    "modules/08_deep_evi.nf",
    "modules/10b_tcga_immune_validation.nf",
    "modules/11_deep_evi_xai.nf",
    "bin/train_deep_evi.py",
    "bin/run_deep_evi_xai.py",
    "envs/deep_evi.yml",
    "envs/deepevi_xai.yml",
    "conf/workstation.config",
]

def test_production_layout():
    missing = [p for p in REQUIRED if not (ROOT / p).is_file()]
    assert not missing, f"Missing production files: {missing}"

if __name__ == "__main__":
    test_production_layout()
    print("PASS: production repository layout")
