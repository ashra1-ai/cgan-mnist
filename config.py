# convenience root config (identical to src/config.py)
from pathlib import Path
import torch

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
SAMPLES_DIR = RESULTS_DIR / "samples"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
NOTEBOOKS_DIR = BASE_DIR / "notebooks"

# Model/training hyperparams (CPU-friendly)
Z_DIM = 100
NUM_CLASSES = 10
IMG_SHAPE = (1, 28, 28)

LR = 2e-4
BATCH_SIZE = 64
EPOCHS = 20
SEED = 42

# Logging / saving
SAVE_EVERY = 1         # save every N epochs
SAMPLE_COUNT = 25      # grid size (5x5)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 100      # print every N batches