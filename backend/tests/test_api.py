"""
Smoke tests for the §06 contract and §11 edge cases. Uses a throwaway SQLite
file so the dev database is untouched.

Run from backend/:   python -m pytest tests -q
                     (or plain:  python tests/test_api.py)
"""

import os
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))
os.environ["DATABASE_URL"] = f"sqlite:///{BACKEND / 'tests' / '_test.db'}"

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402
from app.config import MAX_INPUT_CHARS, MODEL_VERSION  # noqa: E402

PHISH = (
    "Subject: Immediate Action Required: Your Account is at Risk "
    "Dear Valued Customer, We hope this message finds you well. Our system has "
    "detected unusual activity on your account. To secure your account, please "
    "verify your identity immediately by clicking the link below: "
    "https://secure-login-verify.com/verify?session=1 If you do not verify "
    "within 24 hours, your account will be suspended."
)
LEGIT = (
    "Hi all, sorry I've not got your copies of these to you yet. I'll follow "
    "this up after the meeting on Thursday — the handout is nearly done and "
    "I'll bring printouts. Cheers, Tom"
)
RESPONSE_KEYS = {"id", "label", "confidence", "probabilities", "explanation",
                 "model_version", "truncated", "created_at"}
LABELS = {"ai_phishing", "human_phishing", "legitimate"}


def run():
    db = BACKEND / "tests" / "_test.db"
    if db.exists():
        db.unlink()

    with TestClient(app) as c:
        # ---- health
        r = c.get("/api/health")
        assert r.status_code == 200, r.text
        assert r.json() == {"status": "ok", "model_version": MODEL_VERSION, "model_loaded": True}
        print("health          ok ", r.json())

        # ---- predict: exact response shape
        r = c.post("/api/predict", json={"text": PHISH})
        assert r.status_code == 200, r.text
        j = r.json()
        assert set(j) == RESPONSE_KEYS, set(j) ^ RESPONSE_KEYS
        assert j["label"] in LABELS
        assert set(j["probabilities"]) == LABELS
        assert abs(sum(j["probabilities"].values()) - 1.0) < 1e-6
        assert j["confidence"] == max(j["probabilities"].values())
        assert isinstance(j["explanation"]["top_features"], list) and j["explanation"]["top_features"]
        assert j["model_version"] == MODEL_VERSION
        assert j["truncated"] is False
        # spec example: "2026-09-14T18:22:03Z" — UTC, trailing Z, no fractional seconds
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", j["created_at"]), j["created_at"]
        first_id = j["id"]
        print("predict         ok ", j["label"], round(j["confidence"], 3), j["explanation"]["top_features"])

        r = c.post("/api/predict", json={"text": LEGIT})
        assert r.status_code == 200
        print("predict         ok ", r.json()["label"], round(r.json()["confidence"], 3),
              r.json()["explanation"]["top_features"])

        # ---- §11: empty / whitespace → 400
        for bad in ["", "   ", "\n\t "]:
            r = c.post("/api/predict", json={"text": bad})
            assert r.status_code == 400, (bad, r.status_code)
        print("empty -> 400    ok")

        # ---- §11: truncation
        long_text = PHISH * 20
        assert len(long_text) > MAX_INPUT_CHARS
        r = c.post("/api/predict", json={"text": long_text})
        assert r.status_code == 200
        assert r.json()["truncated"] is True
        long_id = r.json()["id"]
        d = c.get(f"/api/predictions/{long_id}").json()
        assert len(d["input_text"]) == MAX_INPUT_CHARS, len(d["input_text"])
        assert d["input_length"] == len(long_text)
        assert d["truncated"] is True
        print(f"truncation      ok  stored {len(d['input_text'])} of {d['input_length']} chars")

        # ---- history: newest first, paginated
        r = c.get("/api/predictions?page=1&page_size=2")
        assert r.status_code == 200
        p = r.json()
        assert p["total"] == 3 and p["page"] == 1 and p["page_size"] == 2
        assert len(p["items"]) == 2
        assert p["items"][0]["id"] == long_id  # newest first
        assert "input_text" not in p["items"][0]
        r = c.get("/api/predictions?page=2&page_size=2")
        assert len(r.json()["items"]) == 1 and r.json()["items"][0]["id"] == first_id
        print("history         ok  total=3, newest-first, paginated")

        # ---- one result / 404
        r = c.get(f"/api/predictions/{first_id}")
        assert r.status_code == 200 and r.json()["input_text"] == PHISH
        r = c.get("/api/predictions/00000000-0000-0000-0000-000000000000")
        assert r.status_code == 404
        r = c.get("/api/predictions/not-a-uuid")
        assert r.status_code == 422
        print("detail / 404    ok")

        # ---- stats
        r = c.get("/api/stats")
        assert r.status_code == 200
        s = r.json()
        assert s["total"] == 3 and set(s["by_label"]) == LABELS
        print("stats           ok ", s["by_label"])

        # ---- CORS: allowed origin echoed, unknown origin not
        r = c.options("/api/predict", headers={
            "Origin": "https://sairishitha-2787.github.io",
            "Access-Control-Request-Method": "POST",
        })
        assert r.headers.get("access-control-allow-origin") == "https://sairishitha-2787.github.io"
        r = c.options("/api/predict", headers={
            "Origin": "https://evil.example",
            "Access-Control-Request-Method": "POST",
        })
        assert "access-control-allow-origin" not in r.headers
        print("CORS allow-list ok")

    # Windows won't unlink while the pool holds the file open
    from app.db import engine
    engine.dispose()
    if db.exists():
        db.unlink()
    print("\nALL PASSED")


def test_api():
    run()


if __name__ == "__main__":
    run()
