"""Predict run/pass for a single pre-snap situation.

Run:
    python -m engage8.predict --down 2 --distance 6 --yardline 38 \
        --quarter 2 --clock 252 --score-diff 3 --formation SHOTGUN

Returns a calibrated pass probability + a plain-English "why".
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config
from .features import _field_zone


def load_model():
    if not config.MODEL_PATH.exists():
        raise SystemExit(
            f"{config.MODEL_PATH} not found. Train first: python -m engage8.train"
        )
    import joblib
    return joblib.load(config.MODEL_PATH)


def _situation_row(features: list[str], **kw) -> pd.DataFrame:
    """Build a one-row feature frame from a pre-snap situation."""
    down = kw["down"]
    dist = kw["distance"]
    yard = kw["yardline_100"]
    clock = kw.get("game_seconds_remaining", 1800)
    quarter = kw.get("quarter", 2)

    row = {
        "down": down, "ydstogo": dist, "yardline_100": yard,
        "quarter": quarter, "half": 1 if quarter <= 2 else 2,
        "game_seconds_remaining": clock,
        "score_differential": kw.get("score_differential", 0),
        "posteam_timeouts": kw.get("posteam_timeouts", 3),
        "is_goal_to_go": int(yard <= dist),
        "is_red_zone": int(yard <= 20),
        "is_two_minute": int(clock <= 120 or 1680 < clock <= 1800),
        "is_third_and_long": int(down == 3 and dist >= 7),
        "is_short_yardage": int(dist <= 2),
        "field_zone": int(_field_zone(pd.Series([yard])).iloc[0]),
        # Lag features unknown for a cold one-off prediction -> neutral/NaN.
        "prev1_was_pass": kw.get("prev1_was_pass", pd.NA),
        "prev2_was_pass": kw.get("prev2_was_pass", pd.NA),
        "prev3_pass_rate": kw.get("prev3_pass_rate", pd.NA),
        "drive_play_index": kw.get("drive_play_index", pd.NA),
        "offense_team": kw.get("offense_team", pd.NA),
        "formation": kw.get("formation", pd.NA),
        "personnel": kw.get("personnel", pd.NA),
        "motion_type": kw.get("motion_type", pd.NA),
        "hash": kw.get("hash", pd.NA),
    }
    return pd.DataFrame([{f: row.get(f, pd.NA) for f in features}])


def predict_situation(artifact: dict, **kw) -> dict:
    feats = artifact["features"]
    cat = set(artifact["categorical_features"])
    X = _situation_row(feats, **kw)
    # Numeric features must be float (NaN ok — LightGBM handles missingness);
    # categoricals must be category dtype.
    for c in feats:
        if c in cat:
            X[c] = X[c].astype("category")
        else:
            X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")

    raw = float(artifact["model"].predict_proba(X)[:, 1][0])
    p_pass = float(artifact["calibrator"].predict([raw])[0])
    p_run = 1.0 - p_pass

    lean = "PASS" if p_pass >= 0.5 else "RUN"
    conf = max(p_pass, p_run)
    return {
        "pass_prob": round(p_pass, 3),
        "run_prob": round(p_run, 3),
        "lean": lean,
        "confidence": round(conf, 3),
        "why": _explain(kw, p_pass),
    }


def _explain(kw: dict, p_pass: float) -> str:
    down, dist = kw["down"], kw["distance"]
    bits = [f"{_ordinal(down)} & {dist}"]
    if kw["yardline_100"] <= 20:
        bits.append("red zone")
    if down == 3 and dist >= 7:
        bits.append("obvious passing down")
    if dist <= 2:
        bits.append("short yardage (run-leaning)")
    lean = "pass" if p_pass >= 0.5 else "run"
    return f"{', '.join(bits)} → {int(round(max(p_pass,1-p_pass)*100))}% {lean}."


def _ordinal(n: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}.get(n, f"{n}th")


def main() -> None:
    p = argparse.ArgumentParser(description="Predict run/pass for a situation.")
    p.add_argument("--down", type=int, required=True)
    p.add_argument("--distance", type=int, required=True)
    p.add_argument("--yardline", type=int, required=True,
                   help="Yards from opponent goal line (yardline_100)")
    p.add_argument("--quarter", type=int, default=2)
    p.add_argument("--clock", type=int, default=1800,
                   help="game_seconds_remaining (3600=kickoff, 0=end)")
    p.add_argument("--score-diff", type=int, default=0)
    p.add_argument("--formation", type=str, default=None)
    p.add_argument("--offense-team", type=str, default=None)
    args = p.parse_args()

    artifact = load_model()
    result = predict_situation(
        artifact,
        down=args.down, distance=args.distance, yardline_100=args.yardline,
        quarter=args.quarter, game_seconds_remaining=args.clock,
        score_differential=args.score_diff, formation=args.formation,
        offense_team=args.offense_team,
    )
    print(f"\n  Pass: {result['pass_prob']:.0%}   Run: {result['run_prob']:.0%}")
    print(f"  Lean: {result['lean']}  (confidence {result['confidence']:.0%})")
    print(f"  Why:  {result['why']}\n")


if __name__ == "__main__":
    main()
