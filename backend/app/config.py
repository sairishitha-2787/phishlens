"""
Environment configuration. Build spec §09 — every value below is set as an env
var on Render in production; the defaults here only exist so the app boots
locally with zero setup.
"""

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

# Neon hands out `postgres://` URLs; SQLAlchemy 2.x only accepts `postgresql://`.
# And a bare `postgresql://` makes SQLAlchemy pick psycopg2, which we don't
# install — pin the psycopg (v3) driver explicitly so Render doesn't crash with
# "No module named psycopg2" on the first query.
_raw_db = os.getenv("DATABASE_URL", f"sqlite:///{BACKEND_DIR / 'phishlens.db'}")
if _raw_db.startswith("postgres://"):
    _raw_db = "postgresql://" + _raw_db[len("postgres://"):]
if _raw_db.startswith("postgresql://"):
    _raw_db = "postgresql+psycopg://" + _raw_db[len("postgresql://"):]
DATABASE_URL = _raw_db

# §11: explicit allow-list, never "*". The Netlify URL is a placeholder until
# Vedha finalises hosting; localhost:5173 is Vite's default dev port.
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "ALLOWED_ORIGINS",
        "https://phishlens.netlify.app,http://localhost:5173",
    ).split(",")
    if o.strip()
]

# §05: the artifact being served, so every logged prediction is traceable.
MODEL_PATH = Path(os.getenv("MODEL_PATH", BACKEND_DIR / "ml" / "artifacts" / "model_v2.pkl"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "svm-tfidf-v2-20260911")

# §11: server-side truncation limit.
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "3000"))

# Explanation: how many terms to surface.
TOP_FEATURES = int(os.getenv("TOP_FEATURES", "5"))
