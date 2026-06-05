"""Tests that run on synthetic data, no network or LightGBM required.

    cd ml && python -m pytest        # or: python tests/test_pipeline.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engage8.config import CANONICAL_COLUMNS
from engage8.features import build_features, TARGET
from engage8.tendencies import tendency_table


def _synthetic_plays(n: int = 400, seed: int = 8) -> pd.DataFrame:
    """Make a plausible canonical plays frame for two seasons."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        down = int(rng.integers(1, 5))
        dist = int(rng.integers(1, 15))
        # Pass more likely on 3rd and long, which gives the model real signal.
        p_pass = 0.45 + (0.4 if (down == 3 and dist >= 7) else 0.0) - (0.2 if dist <= 2 else 0)
        is_pass = rng.random() < min(max(p_pass, 0.05), 0.95)
        yards = int(rng.integers(-3, 25))
        rows.append({
            "play_id": str(i), "game_id": f"g{i // 40}",
            "season": 2022 + (i % 2), "week": 1 + (i % 17),
            "offense_team": rng.choice(["KC", "SF"]), "defense_team": "OPP",
            "quarter": int(rng.integers(1, 5)),
            "game_seconds_remaining": int(rng.integers(0, 3600)),
            "half": 1, "down": down, "ydstogo": dist,
            "yardline_100": int(rng.integers(1, 99)), "hash": pd.NA,
            "score_differential": int(rng.integers(-14, 15)),
            "posteam_timeouts": 3, "personnel": pd.NA,
            "formation": rng.choice(["SHOTGUN", "SINGLEBACK", pd.NA]),
            "backfield": pd.NA, "motion_type": pd.NA,
            "play_type": "pass" if is_pass else "run",
            "play_family": pd.NA, "pass_concept": pd.NA, "run_concept": pd.NA,
            "result_yards": yards, "epa": float(rng.normal(0, 1)),
            "success": int(yards > 4), "explosive": int(yards >= 12),
        })
    return pd.DataFrame(rows)[CANONICAL_COLUMNS]


def test_build_features_has_target_and_flags():
    feats = build_features(_synthetic_plays())
    assert TARGET in feats.columns
    assert feats[TARGET].isin([0, 1]).all()
    # 3rd & long flag should be set correctly.
    assert "is_third_and_long" in feats.columns
    assert "field_zone" in feats.columns
    # Lag feature exists and is bounded.
    assert "prev3_pass_rate" in feats.columns


def test_lag_features_are_chronological():
    feats = build_features(_synthetic_plays())
    # drive_play_index starts at 0 within each game/offense group.
    assert feats["drive_play_index"].min() == 0


def test_tendency_table_sums_to_100():
    table = tendency_table(_synthetic_plays(), split="down_distance")
    assert not table.empty
    for _, r in table.iterrows():
        assert abs(r["run_pct"] + r["pass_pct"] - 100.0) < 0.5


def test_third_and_long_is_pass_heavy():
    plays = _synthetic_plays(n=1500)
    table = tendency_table(plays, split="down_distance")
    row = table[table["bucket"] == "3rd & long"]
    if not row.empty:
        assert row.iloc[0]["pass_pct"] > 55  # signal we baked in


if __name__ == "__main__":
    test_build_features_has_target_and_flags()
    test_lag_features_are_chronological()
    test_tendency_table_sums_to_100()
    test_third_and_long_is_pass_heavy()
    print("All tests passed")
