"""Learned denoising for Monte Carlo path-traced renders."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CACHE_PATH = DATA_DIR / "views.npy"
