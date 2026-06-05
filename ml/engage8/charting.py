"""Manual charting — bring your own opponent.

For teams that aren't in nflverse (your actual Friday/Saturday opponent), chart
their plays into a simple CSV and load it into the same canonical schema the
tendency + prediction engine uses.

    python -m engage8.charting --template            # write a blank template
    python -m engage8.charting --load opponent.csv   # validate + summarize
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config

# The minimal columns a coach/analyst fills in while charting film.
CHARTING_COLUMNS = [
    "game_id", "offense_team", "defense_team",
    "quarter", "game_seconds_remaining",
    "down", "ydstogo", "yardline_100", "hash",
    "score_differential",
    "personnel", "formation", "backfield", "motion_type",
    "play_type", "play_family", "pass_concept", "run_concept",
    "result_yards",
]

TEMPLATE_PATH = config.DATA_DIR / "charting_template.csv"


def write_template(path=TEMPLATE_PATH) -> None:
    example = {
        "game_id": "2026_wk1_us_vs_central", "offense_team": "CENTRAL",
        "defense_team": "US", "quarter": 1, "game_seconds_remaining": 720,
        "down": 1, "ydstogo": 10, "yardline_100": 75, "hash": "M",
        "score_differential": 0, "personnel": "11", "formation": "Trips",
        "backfield": "Gun", "motion_type": "Jet", "play_type": "run",
        "play_family": "Inside Zone", "pass_concept": "", "run_concept": "IZ",
        "result_yards": 5,
    }
    pd.DataFrame([example], columns=CHARTING_COLUMNS).to_csv(path, index=False)
    print(f"Wrote charting template -> {path}")
    print("Fill one row per play, then: python -m engage8.charting --load <file>")


def load_charting(path) -> pd.DataFrame:
    """Validate a charting CSV and map it to the canonical Play schema."""
    df = pd.read_csv(path)
    missing = [c for c in ("down", "ydstogo", "yardline_100", "play_type") if c not in df.columns]
    if missing:
        raise SystemExit(f"Charting file missing required columns: {missing}")

    bad = ~df["play_type"].isin(["run", "pass"])
    if bad.any():
        print(f"  warning: dropping {bad.sum()} rows with non run/pass play_type")
        df = df[~bad]

    out = pd.DataFrame(index=df.index)
    for col in config.CANONICAL_COLUMNS:
        out[col] = df[col] if col in df.columns else pd.NA

    out["play_id"] = [f"chart_{i}" for i in range(len(out))]
    out["season"] = out["season"] if "season" in df.columns else 2026
    out["half"] = np.where(pd.to_numeric(out["quarter"], errors="coerce") <= 2, 1, 2)

    yards = pd.to_numeric(out["result_yards"], errors="coerce")
    thresh = np.where(out["play_type"] == "run",
                      config.EXPLOSIVE_RUN_YARDS, config.EXPLOSIVE_PASS_YARDS)
    out["explosive"] = (yards >= thresh).astype("Int64")
    out = out[config.CANONICAL_COLUMNS].reset_index(drop=True)
    print(f"Loaded {len(out):,} charted plays for "
          f"{out['offense_team'].dropna().unique().tolist()}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Manual opponent charting.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--template", action="store_true", help="write a blank CSV template")
    g.add_argument("--load", type=str, help="path to a filled charting CSV")
    args = p.parse_args()

    if args.template:
        write_template()
    else:
        plays = load_charting(args.load)
        from .tendencies import tendency_table
        print(tendency_table(plays).to_string(index=False))


if __name__ == "__main__":
    main()
