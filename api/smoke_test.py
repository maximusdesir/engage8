"""End-to-end smoke test for the Engage Eight API (uses FastAPI TestClient)."""
import os
import tempfile

# Use a throwaway DB so the test never touches a real one.
os.environ["ENGAGE8_DATABASE_URL"] = f"sqlite:///{tempfile.mktemp(suffix='.db')}"

from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db

init_db()  # create tables (lifespan startup only fires under a context manager)
c = TestClient(app)
ok = 0


def check(label, cond, extra=""):
    global ok
    mark = "PASS" if cond else "FAIL"
    if cond:
        ok += 1
    print(f"  [{mark}] {label} {extra}")
    assert cond, label


# 1. health
r = c.get("/health")
check("GET /health", r.status_code == 200, r.json())

# 2. auth: signup + login + me
r = c.post("/auth/signup", json={"email": "coach@engage8.app", "password": "pw12345"})
check("POST /auth/signup", r.status_code in (200, 201), r.status_code)

r = c.post("/auth/login", data={"username": "coach@engage8.app", "password": "pw12345"})
check("POST /auth/login", r.status_code == 200, r.status_code)
token = r.json()["access_token"]
auth = {"Authorization": f"Bearer {token}"}

r = c.get("/auth/me", headers=auth)
check("GET /auth/me", r.status_code == 200 and r.json()["email"] == "coach@engage8.app")

# 3. teams + opponents
r = c.post("/teams", json={"name": "My HS", "level": "hs", "season": 2026}, headers=auth)
check("POST /teams", r.status_code in (200, 201), r.status_code)
team_id = r.json()["id"]

r = c.get("/teams", headers=auth)
check("GET /teams", r.status_code == 200 and len(r.json()) == 1)

r = c.post(f"/teams/{team_id}/opponents", json={"name": "Central", "oc_name": "Smith"}, headers=auth)
check("POST /teams/{id}/opponents", r.status_code in (200, 201), r.status_code)

# 4. auth enforcement
r = c.get("/teams")
check("GET /teams without token -> 401", r.status_code == 401, r.status_code)

# 5. prediction (uses the trained ml model)
r = c.post("/predict", json={"down": 3, "distance": 8, "yardline_100": 45, "quarter": 4})
check("POST /predict", r.status_code == 200, r.json() if r.status_code == 200 else r.text)
if r.status_code == 200:
    body = r.json()
    print(f"        -> pass {body['pass_prob']:.0%} / run {body['run_prob']:.0%}, {body['lean']}")

# 6. recommendation
r = c.post("/recommend", json={"down": 3, "distance": 1, "yardline_100": 3})
check("POST /recommend", r.status_code == 200, r.status_code)
if r.status_code == 200:
    recs = r.json()["recommendations"]
    print(f"        -> top call: {recs[0]['front']} / {recs[0]['coverage']} / {recs[0]['pressure']}")

# 7. plays + tendencies
for pt, dn, dist in [("run", 1, 10), ("pass", 3, 8), ("run", 2, 2), ("pass", 3, 9)]:
    c.post("/plays", json={"offense_team": "CENTRAL", "down": dn, "ydstogo": dist,
                           "yardline_100": 50, "play_type": pt, "quarter": 1}, headers=auth)
r = c.get("/plays", headers=auth, params={"offense_team": "CENTRAL"})
check("GET /plays", r.status_code == 200 and len(r.json()) >= 4, len(r.json()))

r = c.get("/tendencies", params={"team": "CENTRAL", "split": "down_distance"})
check("GET /tendencies", r.status_code == 200, r.status_code)
if r.status_code == 200:
    print(f"        -> {len(r.json()['rows'])} tendency buckets")

r = c.get("/tendencies", params={"split": "not_a_split"})
check("GET /tendencies bad split -> 400", r.status_code == 400, r.status_code)

print(f"\n{ok} checks passed.")
