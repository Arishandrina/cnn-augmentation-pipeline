from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42
BATCH_SIZE = 384
VAL_FRACTION = 0.1
EPOCHS = 15
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 2

CLASS_NAMES = [str(i) for i in range(10)]

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
(DATA_DIR).mkdir(parents=True, exist_ok=True)