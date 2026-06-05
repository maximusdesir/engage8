# Engage Eight ML pipeline

The data and model layer. It pulls open nflverse play-by-play, engineers pre-snap
features, and trains a calibrated run/pass model. It also includes a tendency
engine that runs on either nflverse data or your own hand-charted opponent.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Python 3.11 or 3.12
pip install -r requirements.txt
```

## Pipeline

```bash
# 1. Pull and normalize nflverse pbp into data/processed/plays.parquet
python -m engage8.extract --seasons 2019 2020 2021 2022 2023

# 2. Engineer features into data/processed/features.parquet
python -m engage8.features

# 3. Train the run/pass model (time-split + calibration) into artifacts/
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
# chart the film, one row per play
python -m engage8.charting --load my_opponent.csv   # tendencies on charted data
```

## Module map

| Module | Job |
|--------|-----|
| `config.py` | Paths and the canonical `Play` schema every source maps into |
| `extract.py` | nflverse pull into the canonical schema |
| `features.py` | Situational and lag (prev-N-play) features |
| `train.py` | LightGBM run/pass, time-split, isotonic calibration, metrics |
| `predict.py` | Single-situation calibrated prediction with a "why" |
| `tendencies.py` | Run/pass and top-play aggregation by down/dist, zone, formation, hash |
| `charting.py` | Manual opponent charting CSV into the canonical schema |

## Why these choices

- **LightGBM**: best on tabular pre-snap data, handles missing values natively
  (nflverse is full of nulls), trains in seconds, and is explainable.
- **Time-based split**: train on older seasons, test on the newest. A random split
  leaks the future and inflates accuracy, which is the most common football-ML
  mistake.
- **Calibration**: a coach only trusts "63% pass" if it is right about 63 percent
  of the time, so the model reports its Brier score, not just accuracy.

## Tests

```bash
python tests/test_pipeline.py   # runs on synthetic data, no network or LightGBM needed
```

## A note on accuracy

Run/pass from the situation lands around 72 to 75 percent. A specific play family
from public data is much harder (roughly 30 to 45 percent top-1) and only gets
good once you add tagged formation, motion, and personnel data, which is what the
charting flow is for.
