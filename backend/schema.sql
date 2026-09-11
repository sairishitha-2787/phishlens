-- Phishlens — build spec §07. One table, one index. Apply once to Neon:
--   psql "$DATABASE_URL" -f schema.sql
-- The app also runs create_all() at startup, so this is belt-and-braces for
-- anyone who wants the schema visible without booting the app.
-- No migrations framework on purpose (2-week build) — don't reach for Alembic.

CREATE TABLE IF NOT EXISTS predictions (
    id               uuid         PRIMARY KEY DEFAULT gen_random_uuid(),
    input_text       text         NOT NULL,   -- stored post-truncation (§11)
    input_length     integer      NOT NULL,   -- original length, pre-truncation
    predicted_label  varchar(32)  NOT NULL,   -- legitimate / human_phishing / ai_phishing
    confidence       real         NOT NULL,   -- 0–1
    probabilities    jsonb        NOT NULL,   -- full per-class breakdown
    model_version    varchar(64)  NOT NULL,   -- matches the served artifact
    created_at       timestamptz  NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_predictions_created_at ON predictions (created_at);
