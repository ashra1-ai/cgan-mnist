# configuration used by scripts
from pathlib import Path
import torch

BASE_DIR = Path("..") if Path(".").name == "src" else Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SAMPLES_DIR = RESULTS_DIR / "samples"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

Z_DIM = 100
NUM_CLASSES = 10
IMG_SHAPE = (1, 28, 28)

LR = 2e-4
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42

SAVE_EVERY = 1
SAMPLE_COUNT = 25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WORKERS = 4