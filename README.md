# Engage Eight 🏈

> AI pre-snap prediction & defensive tendency engine.
> *"Eight in the box" — read the offense before the snap.*

Engage Eight ingests football play-by-play data and answers the two questions a
defensive coordinator asks before every snap:

1. **What is the offense most likely to do?** — a calibrated run/pass + explosive
   probability (and a play-leaning), given down, distance, field position, score,
   time, and formation/personnel when available.
2. **What are their tendencies?** — a tendency matrix sliced by down & distance,
   field zone, formation, and hash, so you can self-scout or scout an opponent.

It trains on free, open **nflverse** NFL play-by-play, and also lets you **chart a
real opponent by hand** so it works for teams that aren't in public datasets.

---

## Why this exists

Defensive coaches spend hundreds of hours charting film to build the probabilistic
model that elite players (Kuechly, Ray Lewis) hold in their heads. Engage Eight
automates the tedious part: aggregating tendencies and turning situation into a
calibrated prediction.

**Honest about accuracy.** Run/pass from situation lands around **72–75%**.
Predicting a *specific* play family from public data is much harder (~30–45%
top-1). The model reports its **calibration (Brier score)**, not just accuracy, and
is validated with **time-based splits** (train on past seasons, test on the most
recent) — never a random split, which leaks the future and inflates results.

---

## Project status

| Layer | Status |
|-------|--------|
| Data + ML pipeline (`ml/`) | ✅ Working end-to-end on real NFL data |
| API (`api/`) | ⏳ Planned |
| Web app (`web/`) | ⏳ Planned |

### Measured results (run/pass, 2019–2023)

Trained on 173,881 real run/pass plays (nflverse), **time-split**: train
2019–2021, calibrate on 2022, test on the fully held-out **2023** season.

| Metric | Value |
|--------|-------|
| Test-season accuracy | **69.7%** |
| Naive baseline (always guess majority) | 59.0% |
| ROC-AUC | 0.766 |
| Brier score (lower = better-calibrated) | 0.192 |

The model's top signals — offense identity, time remaining, field position,
score differential, down & distance — are exactly what a coordinator reads.
Example: it calls **89% pass** on 3rd-and-8 and **85% run** on 3rd-and-1 at the
2-yard line.

## Quickstart (ML pipeline)

```bash
cd ml
python -m venv .venv && source .venv/bin/activate   # use Python 3.11–3.12
pip install -r requirements.txt

# 1. Pull nflverse play-by-play (downloads to ml/data/)
python -m engage8.extract --seasons 2019 2020 2021 2022 2023

# 2. Normalize → canonical schema → engineer features
python -m engage8.features

# 3. Train the run/pass + explosive model (time-split, calibrated)
python -m engage8.train

# 4. Predict a situation
python -m engage8.predict --down 2 --distance 6 --yardline 38 \
    --quarter 2 --clock 252 --score-diff 3
```

## Repo layout

```
engage_eight/
├── ml/        # data pipeline + model training (start here)
├── api/       # FastAPI service (planned)
└── web/       # Next.js dashboard (planned)
```

## Data sources

- **[nflverse / nflfastR](https://github.com/nflverse)** via
  [`nfl_data_py`](https://github.com/cooperdff/nfl_data_py) — open NFL
  play-by-play, EPA, win probability. Free for public use.
- **Manual charting** — chart any team's plays into a CSV and run the same
  tendency + prediction engine on them.

## License

MIT — see [LICENSE](LICENSE).
