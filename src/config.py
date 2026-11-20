from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATASET_DIR = ROOT_DIR / "dataset"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
AUDIT_DIR = ARTIFACTS_DIR / "dataset_audit"
IMAGE_SIZE = 256
RANDOM_SEED = 42
