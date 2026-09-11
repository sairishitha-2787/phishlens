"""
Phishlens API — FastAPI app instance, CORS, router mounts.

Run locally:   uvicorn app.main:app --reload        (from backend/)
Docs:          http://localhost:8000/docs
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import ALLOWED_ORIGINS, MODEL_VERSION
from .db import init_db
from .models import inference
from .routers import health, predict, predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()          # one table, created if absent (§07)
    inference.load()   # model_v2.pkl, once, at startup (§05)
    yield


app = FastAPI(
    title="Phishlens API",
    version=MODEL_VERSION,
    description="3-way phishing classifier: legitimate / human_phishing / ai_phishing.",
    lifespan=lifespan,
)

# §11: explicit allow-list, never "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

app.include_router(predict.router)
app.include_router(predictions.router)
app.include_router(health.router)
