"""Paths, constants, and the canonical Play schema for Engage Eight.

Everything in the pipeline reads/writes through these paths so the data and
model artifacts live in predictable, git-ignored locations.
"""
from __future__ import annotations

from pathlib import Path

# --- Directories -----------------------------------------------------------
ML_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ML_ROOT / "data"
ARTIFACTS_DIR = ML_ROOT / "artifacts"

RAW_DIR = DATA_DIR / "raw"          # raw nflverse pulls
PROCESSED_DIR = DATA_DIR / "processed"  # normalized + featurized parquet

for _d in (DATA_DIR, ARTIFACTS_DIR, RAW_DIR, PROCESSED_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Pipeline artifact paths
PLAYS_PARQUET = PROCESSED_DIR / "plays.parquet"        # canonical schema
FEATURES_PARQUET = PROCESSED_DIR / "features.parquet"  # model-ready features
MODEL_PATH = ARTIFACTS_DIR / "runpass_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# --- Canonical Play schema -------------------------------------------------
# Every data source (nflverse, manual charting CSV) normalizes into these
# columns. Fields the source can't provide are left null; LightGBM handles
# missingness natively.
CANONICAL_COLUMNS = [
    "play_id", "game_id", "season", "week",
    "offense_team", "defense_team",
    "quarter", "game_seconds_remaining", "half",
    "down", "ydstogo", "yardline_100", "hash",
    "score_differential", "posteam_timeouts",
    "personnel", "formation", "backfield", "motion_type",
    "play_type",          # run | pass  (the v1 target)
    "play_family", "pass_concept", "run_concept",
    "result_yards", "epa", "success", "explosive",
]

# --- Modeling constants ----------------------------------------------------
# Explosive play definition (yards gained). Coaches commonly use 12+ for runs,
# 16+ for passes; we use a single yardage threshold on the result for v1.
EXPLOSIVE_RUN_YARDS = 12
EXPLOSIVE_PASS_YARDS = 16

# Time-based split: hold out the most recent season(s) for honest evaluation.
# Anything <= TRAIN_THROUGH_SEASON trains; the rest is val/test.
DEFAULT_SEASONS = [2019, 2020, 2021, 2022, 2023]
