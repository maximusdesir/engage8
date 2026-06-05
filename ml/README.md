# Engage Eight — ML pipeline

The data + model layer. Pulls open nflverse play-by-play, engineers pre-snap
features, and trains a calibrated run/pass model — plus a tendency engine that
runs on either nflverse data or your own hand-charted opponent.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11–3.12 ideal
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Pull & normalize nflverse pbp -> data/processed/plays.parquet
python -m engage8.extract --seasons 2019 2020 2021 2022 2023

# 2. Engineer features -> data/processed/features.parquet
python -m engage8.features

# 3. Train run/pass model (time-split + calibration) -> artifacts/
python -m engage8.train

# 4. Predict a situation
python -m engage8.predict --down 3 --distance 8 --yardline 45 --quarter 4 --clock 600

# 5. Inspect tendencies
python -m engage8.tendencies --team KC --split down_distance
```

## Bring your own opponent (manual charting)

For a team that isn't in nflverse (your actual opponent):

```bash
python -m engage8.charting --template          # writes data/charting_template.csv
# ...chart film, one row per play...
python -m engage8.charting --load my_opponent.csv   # tendencies on charted data
```

## Module map

| Module | Job |
|--------|-----|
| `config.py` | Paths + the canonical `Play` schema every source maps into |
| `extract.py` | nflverse pull → canonical schema |
| `features.py` | Situational + lag (prev-N-play) features |
| `train.py` | LightGBM run/pass, **time-split**, isotonic calibration, metrics |
| `predict.py` | Single-situation calibrated prediction + a "why" |
| `tendencies.py` | Run/pass + top-play aggregation by down/dist, zone, formation, hash |
| `charting.py` | Manual opponent charting CSV → canonical schema |

## Why these choices

- **LightGBM** — best on tabular pre-snap data, handles missing values
  natively (nflverse is full of nulls), trains in seconds, explainable.
- **Time-based split** — train on older seasons, test on the newest. A random
  split leaks the future and inflates accuracy; the #1 football-ML mistake.
- **Calibration** — a coach only trusts "63% pass" if it's right ~63% of the
  time. We report **Brier score**, not just accuracy.

## Tests

```bash
python -m pytest        # runs on synthetic data, no network/LightGBM needed
```

## Honest accuracy

Run/pass from situation: ~72–75%. Specific play family from public data is much
harder (~30–45% top-1) and only gets good once you add tagged
formation/motion/personnel — which is what the charting flow is for.
