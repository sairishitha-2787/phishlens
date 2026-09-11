"""
SQLAlchemy engine, session, and the one table (build spec §07).

Column types are chosen to be native on Postgres (uuid, jsonb, timestamptz)
while still working on SQLite for local dev — the same ORM model runs in both.
No migrations framework: the schema is created on startup, and schema.sql
exists for applying it to Neon by hand.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column, DateTime, Float, Integer, String, Text, Uuid, JSON, create_engine, func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import DATABASE_URL

_is_sqlite = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if _is_sqlite else {},
    pool_pre_ping=not _is_sqlite,  # Neon autosuspends; ping before reuse
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def _now():
    return datetime.now(timezone.utc)


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Uuid, primary_key=True, default=uuid.uuid4)
    input_text = Column(Text, nullable=False)          # post-truncation (§11)
    input_length = Column(Integer, nullable=False)     # original length
    predicted_label = Column(String(32), nullable=False)
    confidence = Column(Float, nullable=False)
    probabilities = Column(JSON().with_variant(JSONB, "postgresql"), nullable=False)
    model_version = Column(String(64), nullable=False)
    created_at = Column(
        DateTime(timezone=True), nullable=False, default=_now,
        server_default=func.now(), index=True,
    )


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
