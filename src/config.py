from pathlib import Path

# -------- Project Paths -------------------
BASE_DIR = Path(__file__).resolve().parents[1]

# -------- Paths to Data -------------------
TRAIN_DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"
TEST_DATA_PATH = BASE_DIR / "data" / "raw" / "test.csv"

# -------- Path to model location -------------
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_model.pkl"

# -------- Path to reports ------------------
REPORT_DIR = BASE_DIR / "reports"
METRICS_PATH = REPORT_DIR / "metrics.json"

# -------- Path to images -------------------
IMAGE_DIR = BASE_DIR / "images"

# -------- Some universal variables for the project
TARGET = "Survived"
RANDOM_STATE = 42