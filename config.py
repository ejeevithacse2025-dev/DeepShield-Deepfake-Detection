import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Model configuration
# Check models/ directory first, fallback to root
MODEL_FILENAME = "deepshield_model_final.keras"
MODELS_DIR_PATH = BASE_DIR / "models" / MODEL_FILENAME
ROOT_MODEL_PATH = BASE_DIR / MODEL_FILENAME

if MODELS_DIR_PATH.exists():
    DEFAULT_MODEL_PATH = str(MODELS_DIR_PATH)
elif ROOT_MODEL_PATH.exists():
    DEFAULT_MODEL_PATH = str(ROOT_MODEL_PATH)
else:
    DEFAULT_MODEL_PATH = str(MODELS_DIR_PATH)

# Input image dimensions
INPUT_HEIGHT = 224
INPUT_WIDTH = 224
INPUT_CHANNELS = 3
INPUT_SIZE = (INPUT_HEIGHT, INPUT_WIDTH)

# Classification thresholds
# >= 0.5 -> FAKE, < 0.5 -> REAL
DEFAULT_THRESHOLD = 0.50

LABEL_FAKE = "FAKE"
LABEL_REAL = "REAL"

# Face detection parameters
FACE_PADDING_RATIO = 0.20  # 20% margin around detected face box
MIN_FACE_SIZE = (40, 40)

# Video analysis parameters
DEFAULT_MAX_VIDEO_FRAMES = 30
DEFAULT_VIDEO_SAMPLE_FPS = 1.0  # Sample 1 frame per second by default

# Temporary working directory for video frames
TEMP_DIR = BASE_DIR / "temp_uploads"
TEMP_DIR.mkdir(parents=True, exist_ok=True)
