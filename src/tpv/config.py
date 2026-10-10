"""Project-wide paths and constants."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"      # PhysioNet EDF files (git-ignored)
MODELS_DIR = ROOT_DIR / "models"  # Trained pipelines (git-ignored)
PLOTS_DIR = ROOT_DIR / "plots"    # Generated figures

N_SUBJECTS = 109
STREAM_DELAY_S = 2.0
