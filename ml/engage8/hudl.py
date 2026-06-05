"""Hudl CSV adapter: map a Hudl breakdown export into the canonical schema.

Hudl breakdown/playlist exports vary team to team because the columns are
user-configured, so this adapter is deliberately tolerant: it normalizes the
header names, maps the common ones through an alias table, converts Hudl's
yard-line and hash conventions, and leaves anything it can't map null (the
model and tendency code both handle nulls).

    python -m engage8.hudl --template          # write a sample Hudl-style CSV
    python -m engage8.hudl --load export.csv    # validate + summarize
"""
from __future__ import annotations

import argparse
import re

import numpy as np
import pandas as pd

from . import config

# Normalized Hudl header -> intermediate field. Headers are matched after
# uppercasing, dropping '.', and collapsing whitespace (so "Dist." -> "DIST").
HUDL_ALIASES = {
    "DN": "down", "DOWN": "down",
    "DIST": "ydstogo", "DISTANCE": "ydstogo", "TO GO": "ydstogo", "TOGO": "ydstogo",
    "QTR": "quarter", "QUARTER": "quarter", "PERIOD": "quarter",
    "HASH": "hash", "HASH MARK": "hash", "HM": "hash",
    "OFF FORM": "formation", "OFF FORMATION": "formation", "FORMATION": "formation",
    "OFFENSIVE FORMATION": "formation", "FORM": "formation",
    "MOTION": "motion_type", "MO": "motion_type",
    "PERS": "personnel", "PERSONNEL": "personnel", "OFF PERS": "personnel",
    "PLAY TYPE": "play_type_raw", "TYPE": "play_type_raw",
    "RUN/PASS": "play_type_raw", "R/P": "play_type_raw", "PLAY CATEGORY": "play_type_raw",
    "OFF PLAY": "play_name", "PLAY": "play_name", "PLAY NAME": "play_name",
    "PLAY CALL": "play_name", "CONCEPT": "play_name",
    "GN/LS": "result_yards", "GN/LOSS": "result_yards", "GAIN/LOSS": "result_yards",
    "GAIN": "result_yards", "GN": "result_yards", "YARDS": "result_yards",
    "YDS": "result_yards", "RESULT YARDS": "result_yards",
    "YARD LN": "yardline_raw", "YARD LINE": "yardline_raw", "YDLN": "yardline_raw",
    "YD LN": "yardline_raw", "LOS": "yardline_raw", "BALL ON": "yardline_raw",
    "ODK": "odk", "PHASE": "odk",
    "OFF TEAM": "offense_team", "DEF TEAM": "defense_team",
}

# Hudl-ish headers used to auto-detect a Hudl file vs the plain charting CSV.
HUDL_SIGNAL_HEADERS = {"ODK", "OFF FORM", "GN/LS", "YARD LN", "DN", "OFF PLAY"}

TEMPLATE_PATH = config.DATA_DIR / "hudl_sample.csv"


def _norm_header(h) -> str:
    return " ".join(str(h).upper().replace(".", "").split())


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Return the column if present, else an all-null Series aligned to df."""
    if name in df.columns:
        return df[name]
    return pd.Series([pd.NA] * len(df), index=df.index)


def _to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def _parse_hash(value) -> object:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    s = str(value).strip().upper()
    if not s:
        return pd.NA
    if s[0] == "L":
        return "L"
    if s[0] == "R":
        return "R"
    if s[0] in ("M", "C") or s.startswith("MID"):
        return "M"
    return pd.NA


def _parse_personnel(value) -> object:
    """Pull a clean two-digit personnel grouping (e.g. '11', '21') if present."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    m = re.search(r"\d{2}", str(value))
    return m.group() if m else pd.NA


def _yl_from_signed(iv: int) -> object:
    """Yard line under Hudl's sign convention: negative=own, positive=opp."""
    a = abs(iv)
    if a == 0 or a > 50:
        return pd.NA
    if a == 50:
        return 50            # midfield
    return 100 - a if iv < 0 else a


def _parse_yardline(value) -> object:
    """Convert a Hudl yard-line value to yardline_100 (yards to opp goal).

    Hudl encodes territory by sign: a negative value (or "OWN N") is your own
    side, a positive/unsigned value (or "OPP N") is the opponent's side.
      -35 / OWN 35  -> 100 - 35 = 65
      +30 / OPP 30 / 30 -> 30
      50 / C / MID  -> 50 (midfield)
    Note: pandas often strips a leading '+', so an unsigned positive number is
    treated as opponent territory, matching the dominant Hudl convention.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NA
    # Numeric input: the sign is authoritative.
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _yl_from_signed(int(round(float(value))))

    s = str(value).strip().upper().replace(" ", "")
    if not s or s == "NAN":
        return pd.NA
    if s in ("C", "MID", "MIDFIELD"):
        return 50

    own = None
    if s.startswith("OWN") or s.startswith("-"):
        own = True
    elif s.startswith("OPP") or s.startswith("+"):
        own = False

    m = re.search(r"\d+", s)
    if not m:
        return pd.NA
    yard = int(m.group())
    if own is True:
        return _yl_from_signed(-yard)
    if own is False:
        return _yl_from_signed(yard)
    return _yl_from_signed(yard)  # unsigned -> opponent territory (convention)


def _parse_play_type(raw, play_name) -> object:
    """Resolve run vs pass from the play-type column, falling back to the name."""
    for text in (raw, play_name):
        if text is None or (isinstance(text, float) and pd.isna(text)):
            continue
        t = str(text).strip().upper()
        if not t:
            continue
        if t in ("PASS", "P") or "PASS" in t or "SCREEN" in t or "DROPBACK" in t \
                or "DROP BACK" in t or "PA " in t or "PLAY ACTION" in t:
            return "pass"
        if t in ("RUN", "R", "RUSH") or "RUN" in t or "RUSH" in t or "DRAW" in t \
                or "SWEEP" in t or "ZONE" in t or "POWER" in t or "COUNTER" in t:
            return "run"
    return pd.NA


def _clip(series: pd.Series, n: int) -> pd.Series:
    return series.map(lambda v: v if (v is None or pd.isna(v)) else str(v)[:n])


def load_hudl(path, default_offense_team: str | None = None) -> pd.DataFrame:
    """Load a Hudl breakdown CSV and map it to the canonical Play schema."""
    raw = pd.read_csv(path)

    colmap = {c: HUDL_ALIASES[_norm_header(c)] for c in raw.columns
              if _norm_header(c) in HUDL_ALIASES}
    # If the same intermediate field maps from two columns, keep the first.
    seen: set[str] = set()
    dedup = {}
    for src, dst in colmap.items():
        if dst not in seen:
            dedup[src] = dst
            seen.add(dst)
    work = raw.rename(columns=dedup)

    # Keep offensive plays when an ODK/phase column exists.
    if "odk" in work.columns:
        first = work["odk"].astype(str).str.strip().str.upper().str[:1]
        work = work[first == "O"]

    out = pd.DataFrame(index=work.index)
    out["down"] = _to_int(_col(work, "down"))
    out["ydstogo"] = _to_int(_col(work, "ydstogo"))
    out["quarter"] = _to_int(_col(work, "quarter"))
    out["yardline_100"] = _to_int(_col(work, "yardline_raw").map(_parse_yardline))
    out["hash"] = _col(work, "hash").map(_parse_hash)
    out["personnel"] = _col(work, "personnel").map(_parse_personnel)
    out["formation"] = _clip(_col(work, "formation"), 40)
    out["motion_type"] = _clip(_col(work, "motion_type"), 40)

    play_name = _col(work, "play_name")
    pt_raw = _col(work, "play_type_raw")
    out["play_type"] = [_parse_play_type(a, b) for a, b in zip(pt_raw, play_name)]
    out["play_family"] = _clip(play_name, 40)
    out["result_yards"] = _to_int(_col(work, "result_yards"))

    team = _col(work, "offense_team")
    if default_offense_team is not None:
        team = team.where(team.notna() & (team.astype(str) != ""), default_offense_team)
    out["offense_team"] = team
    out["defense_team"] = _col(work, "defense_team")

    # Keep only real run/pass plays with a down (the prediction target).
    out = out[out["play_type"].isin(["run", "pass"]) & out["down"].notna()]

    # Fill the remaining canonical columns and derive the simple ones.
    out["half"] = np.where(pd.to_numeric(out["quarter"], errors="coerce") <= 2, 1, 2)
    yards = pd.to_numeric(out["result_yards"], errors="coerce")
    thresh = np.where(out["play_type"] == "run",
                      config.EXPLOSIVE_RUN_YARDS, config.EXPLOSIVE_PASS_YARDS)
    out["explosive"] = (yards >= thresh).astype("Int64")
    out["play_id"] = [f"hudl_{i}" for i in range(len(out))]

    for col in config.CANONICAL_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[config.CANONICAL_COLUMNS].reset_index(drop=True)
    teams = [t for t in out["offense_team"].dropna().unique().tolist()]
    print(f"Loaded {len(out):,} Hudl offensive plays" + (f" for {teams}" if teams else ""))
    return out


def looks_like_hudl(path) -> bool:
    """Heuristic: does this CSV's header row look like a Hudl export?"""
    try:
        header = pd.read_csv(path, nrows=0).columns
    except Exception:
        return False
    norm = {_norm_header(c) for c in header}
    # The plain charting CSV uses exact lowercase canonical names.
    if "play_type" in set(header):
        return False
    return len(norm & HUDL_SIGNAL_HEADERS) >= 2


def write_template(path=TEMPLATE_PATH) -> None:
    rows = [
        ["1", "O", "1", "10", "-25", "L", "Trips Rt", "Inside Zone", "Run", "Jet", "11", "5", "1"],
        ["2", "O", "2", "5", "-30", "R", "Doubles", "Stick", "Pass", "None", "11", "8", "1"],
        ["3", "O", "3", "2", "+45", "M", "Heavy", "Power", "Run", "", "21", "1", "2"],
        ["4", "D", "1", "10", "+20", "L", "", "", "", "", "", "0", "2"],
        ["5", "O", "3", "9", "-40", "R", "Empty", "Four Verticals", "Pass", "", "10", "-2", "3"],
    ]
    cols = ["PLAY #", "ODK", "DN", "DIST", "YARD LN", "HASH", "OFF FORM",
            "OFF PLAY", "PLAY TYPE", "MOTION", "PERS", "GN/LS", "QTR"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)
    print(f"Wrote sample Hudl-style CSV -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Hudl breakdown CSV adapter.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--template", action="store_true", help="write a sample Hudl CSV")
    g.add_argument("--load", type=str, help="path to a Hudl export CSV")
    p.add_argument("--team", type=str, default=None, help="label offense_team for all rows")
    args = p.parse_args()

    if args.template:
        write_template()
    else:
        plays = load_hudl(args.load, default_offense_team=args.team)
        from .tendencies import tendency_table
        print(tendency_table(plays).to_string(index=False))


if __name__ == "__main__":
    main()
