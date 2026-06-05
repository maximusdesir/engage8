# Engage Eight API

A FastAPI service that wraps the trained model and tendency engine from `../ml`
and exposes them over HTTP, with auth, teams, plays, and CSV charting upload.

## Run it

```bash
cd api
python -m venv .venv && source .venv/bin/activate   # Python 3.11 to 3.14
pip install -r requirements.txt

uvicorn app.main:app --reload
```

Then open the interactive docs at http://localhost:8000/docs.

The API reuses the model trained in `../ml`, so run the ML pipeline first (see
`../ml/README.md`) or the prediction endpoints return 503. It uses SQLite on disk
by default (`engage8.db`); set `ENGAGE8_DATABASE_URL` to a Postgres URL for
production.

## Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | no | Liveness check |
| POST | `/auth/signup` | no | Create a user |
| POST | `/auth/login` | no | Get a JWT |
| GET | `/auth/me` | yes | Current user |
| GET/POST | `/teams` | yes | List/create teams |
| GET | `/teams/{id}` | yes | One team |
| GET/POST | `/teams/{id}/opponents` | yes | Opponents for a team |
| GET/POST | `/plays` | yes | Filter/insert plays |
| POST | `/predict` | no | Run/pass probability for a situation |
| POST | `/recommend` | no | Predicted offense + ranked defensive calls |
| GET | `/tendencies` | no | Tendency matrix by down/dist, zone, formation, hash |
| POST | `/uploads` | no | Upload a charting CSV, ingest as plays |

## Layout

```
api/app/
├── main.py          # app factory, router registration, startup
├── config.py        # settings (env-driven)
├── deps.py          # get_db, get_current_user
├── core/security.py # bcrypt hashing + JWT
├── db/              # SQLAlchemy engine, session, models
├── schemas/         # Pydantic request/response models
├── services/        # prediction, recommendation, tendencies (reuse ../ml)
└── routers/         # auth, teams, plays, predict, tendencies, uploads
```

## Smoke test

```bash
python smoke_test.py    # spins up a TestClient and exercises every endpoint
```
