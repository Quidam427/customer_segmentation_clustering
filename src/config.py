"""Shared settings: where files live and the fixed random seed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

RAW_FILE = RAW_DIR / "online_retail_II.xlsx"            # original file from UCI
CACHE_FILE = RAW_DIR / "online_retail_II_combined.pkl"  # fast-loading copy, created on first run

RANDOM_STATE = 42
