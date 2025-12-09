import os
from pathlib import Path

# Base project directory (assuming config is in ia_detector/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory
DATA_DIR = BASE_DIR / "data"

# Ensure data dir exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Model Paths
ENSEMBLE_MODEL_PATH = DATA_DIR / "ensemble_model.pkl"
TFIDF_MODEL_PATH = DATA_DIR / "tfidf_model.pkl"
CACHE_DB_PATH = DATA_DIR / "detector_cache.db"
TRAINING_DATA_PATH = DATA_DIR / "gemini_training_data.json"

# API Keys (loaded from env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
