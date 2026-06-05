"""Feature engineering: canonical plays -> model-ready feature table.

Builds the situational features plus engineered "previous N plays" lag
features (the GBM's substitute for a sequence model). Reads
plays.parquet, writes features.parquet.

Run:
    python -m engage8.features
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config

# Feature columns the model consumes. Categorical columns are listed so the
# trainer can hand them to LightGBM as categoricals.
NUMERIC_FEATURES = [
    "down", "ydstogo", "yardline_100", "quarter", "half",
    "game_seconds_remaining", "score_differential", "posteam_timeouts",
    "is_goal_to_go", "is_red_zone", "is_two_minute", "is_third_and_long",
    "is_short_yardage", "field_zone", "prev1_was_pass", "prev2_was_pass",
    "prev3_pass_rate", "drive_play_index",
]
CATEGORICAL_FEATURES = [
    "offense_team", "formation", "personnel", "motion_type", "hash",
]
TARGET = "is_pass"          # 1 = pass, 0 = run
EXPLOSIVE_TARGET = "explosive"

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _field_zone(yardline_100: pd.Series) -> pd.Series:
    """Coarse field zone: 0=backed up,1=own,2=midfield,3=fringe,4=red zone."""
    bins = [-1, 10, 40, 60, 80, 100]
    return pd.cut(yardline_100, bins=bins, labels=[4, 3, 2, 1, 0]).astype("Int64")
    # note: yardline_100 counts down toward opponent goal, so small = red zone


def build_features(plays: pd.DataFrame) -> pd.DataFrame:
    df = plays.copy()

    # Target: pass vs run.
    df["is_pass"] = (df["play_type"] == "pass").astype("Int64")

    # --- Situational flags ---------------------------------------------
    df["is_goal_to_go"] = (df["yardline_100"] <= df["ydstogo"]).astype("Int64")
    df["is_red_zone"] = (df["yardline_100"] <= 20).astype("Int64")
    df["is_two_minute"] = (
        (df["game_seconds_remaining"] <= 120)
        | ((df["game_seconds_remaining"] > 1680)
           & (df["game_seconds_remaining"] <= 1800))
    ).astype("Int64")
    df["is_third_and_long"] = (
        (df["down"] == 3) & (df["ydstogo"] >= 7)
    ).astype("Int64")
    df["is_short_yardage"] = (df["ydstogo"] <= 2).astype("Int64")
    df["field_zone"] = _field_zone(df["yardline_100"])

    # --- Sequence / lag features (per drive within a game) -------------
    # Sort so lags are chronological. game_seconds_remaining decreases.
    df = df.sort_values(
        ["game_id", "offense_team", "game_seconds_remaining"],
        ascending=[True, True, False],
    ).reset_index(drop=True)

    grp = df.groupby(["game_id", "offense_team"], sort=False)
    df["drive_play_index"] = grp.cumcount().astype("Int64")
    is_pass = df["is_pass"]
    df["prev1_was_pass"] = grp["is_pass"].shift(1)
    df["prev2_was_pass"] = grp["is_pass"].shift(2)
    # Rolling pass rate over the previous 3 plays (excludes current).
    df["prev3_pass_rate"] = (
        grp["is_pass"].apply(
            lambda s: s.shift(1).rolling(3, min_periods=1).mean()
        ).reset_index(level=[0, 1], drop=True)
    )

    # Ensure categorical dtypes for LightGBM.
    for c in CATEGORICAL_FEATURES:
        if c in df.columns:
            df[c] = df[c].astype("category")

    keep = (
        ["play_id", "game_id", "season", "week"]
        + ALL_FEATURES
        + [TARGET, EXPLOSIVE_TARGET, "epa"]
    )
    keep = [c for c in keep if c in df.columns]
    out = df[keep]
    print(f"Built features: {len(out):,} rows, {len(ALL_FEATURES)} features")
    return out


def main() -> None:
    if not config.PLAYS_PARQUET.exists():
        raise SystemExit(
            f"{config.PLAYS_PARQUET} not found. Run `python -m engage8.extract` first."
        )
    plays = pd.read_parquet(config.PLAYS_PARQUET)
    feats = build_features(plays)
    feats.to_parquet(config.FEATURES_PARQUET, index=False)
    print(f"Wrote {config.FEATURES_PARQUET}")


if __name__ == "__main__":
    main()
