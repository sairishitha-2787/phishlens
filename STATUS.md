# Phishlens — Team Status

**Read this first. Update this last.** Whatever AI tool you're using this session — Claude Code, ChatGPT, Gemini, whatever — paste this instruction at the start: "Read STATUS.md before doing anything." And at the end: "Update STATUS.md with what you just did and what's next."

One line per entry: `- [date] [R/P/V] did X`. Newest entries go on top of each list. Don't delete old "Done" entries — that log is the project's memory across all of you and all your different tools.

---

## Now — in progress

- R: backend is in this repo (`backend/`) with a Render Blueprint (`render.yaml`) — **deploy is the next action and needs a human with the Render + Neon accounts** (see Blocked). Then connect frontend (with P).
- P: (update this)
- V: (update this)

---

## Done — log, newest first

- 2026-09-11 R: fixed a deploy-blocking bug found while prepping Render — `postgresql://` made SQLAlchemy pick the psycopg2 driver, which isn't installed (we ship psycopg v3). Would have crashed on the first Neon query. `config.py` now pins `postgresql+psycopg://` for any Neon URL shape. Still only tested on SQLite locally; first real Postgres test happens at deploy.
- 2026-09-11 R: backend pushed to this repo under `backend/` (app/, ml/ incl. `model_v2.pkl`, schema.sql, requirements.txt, tests/) plus `render.yaml` Blueprint at repo root encoding the exact Render settings (root dir `backend`, `uvicorn app.main:app --host 0.0.0.0 --port $PORT`, Python 3.11.9, health check `/api/health`). Tests pass from the repo copy.
- 2026-09-11 R: backend verified end-to-end in a clean venv — `pip install -r requirements.txt` OK, `pytest tests/ -v` 1 passed / 0 failed, `uvicorn app.main:app --reload` boots in ~3s. Hit `POST /api/predict` with 3 real held-out test emails (Chase phish from Nazario, SquirrelMail list reply from CEAS, GPT phish): all 3 classified correctly (confidences 0.79 / 0.81 / 0.77). `/api/health`, `/api/stats`, `/api/predictions` all 200; empty input -> 400. Only fix needed: `pytest` was missing from `requirements.txt`. Known nits, not blockers: confidences are softmax over SVM margins (uncalibrated); HTML entity `nbsp` can leak into `top_features`.
- 2026-09-11 R: `processed/data_card.md` confirmed current for v2 — every number (5,505 rows, 1,665/1,444/2,396 split, 251 control, 1.44:1 ratio) re-verified against the live CSVs. Not stale anymore.
- 2026-09-11 R: FastAPI backend built per build spec §06/§07/§11 — `POST /api/predict`, `GET /api/predictions[/{id}]`, `/api/health`, `/api/stats`; serves `model_v2.pkl`; SQLAlchemy one-table log (SQLite locally, Postgres/Neon via `DATABASE_URL`); explanation = top SVM terms + URL count; CORS allow-list. `backend/schema.sql` for Neon. Lives in the RESEARCH PAPER folder (`backend/`), not yet in this repo.
- 2026-09-11 R: live sprint checklist deployed — https://sairishitha-2787.github.io/phishlens/ — shared, real-time checklist (Firebase Firestore), synced across everyone who has it open. Replaces the old localStorage-only checklist file. Source lives in this repo as `index.html`.
- 2026-09-11 R: dataset v2 built and validated — folded AI-generated-legitimate rows into the `legitimate` class, fixing the v1 confound (model was detecting corpus/AI style, not phishing content). New control-set false-alarm rate: 0.0% (was 93.5%). See `backend/ml/artifacts/rebalanced_analysis.md`.
- 2026-09-11 R: confound diagnosed — v1 baselines (macro-F1 0.9878) were a dataset-construction artifact, not real phishing detection. Root cause + two diagnostic experiments in `backend/ml/artifacts/confound_analysis.md`.
- 2026-09-11 R: real dataset pipeline built — mbox parser, exact + SimHash near-dup dedup, stratified split. Verified real sources: `legit.csv` = CEAS-challenge + MIT mail (not Enron), `phishing.csv`/mbox = Nazario Phishing Corpus. `phishing3.mbox` still missing (likely Defender-quarantined).
- 2026-09-11 R: baseline models trained on v2 — Linear SVM wins outright, macro-F1 0.9913, saved as `backend/ml/artifacts/model_v2.pkl`. This is the model to serve.
- (earlier) R/P/V: topic locked, roles assigned, build spec written ("Phishlens Build Spec"), sprint plan set (Sept 7–20), faculty scope call done (topic approved, told to move fast — venue/disclosure specifics not yet discussed).

---

## Blocked / needs a decision

- **Deploy needs someone logged into Render + Neon (R or V):** (1) Neon → New project (free) → copy the connection string. (2) Render → New + → Blueprint → pick this repo → it reads `render.yaml`; paste the Neon string as `DATABASE_URL` when prompted (either `postgres://` or `postgresql://` form is fine — config handles both). (3) Once live, `curl <url>/api/health` should return `model_loaded: true`, then one `POST /api/predict` to prove Postgres logging works. (4) Put the live URL here and in the Reference docs. No CLI/API access to either account from R's machine, so this can't be automated by an AI tool — it's a few clicks in two dashboards.
- Faculty hasn't weighed in on the confound finding specifically (call happened before it was found) — worth a short follow-up, not blocking.
- `phishing3.mbox` missing — R needs to check Windows Defender's quarantine and re-download. Pipeline runs without it for now.

---

## Next up — per track

- **R (data/model/backend):** deploy via the Blueprint (see Blocked — joint with V) → paste live URL into STATUS.md → connect frontend to live endpoint (joint with P) → try a few real-world pastes before the demo (model has only been tested on corpus samples)
- **P (frontend/security):** run adversarial prompt set against `model_v2.pkl` (doesn't need to wait for the live backend) → connect frontend to live endpoint (joint with R)
- **V (infra/backend):** deploy is ready to go — `render.yaml` has every setting; only `DATABASE_URL` (from Neon) needs pasting. `schema.sql` is optional (the app runs `create_all` on startup). Set `ALLOWED_ORIGINS` once P's Netlify URL exists.

---

## Reference docs (for context, not status)

- **Live sprint checklist (check things off here, it's shared):** https://sairishitha-2787.github.io/phishlens/
- Build spec: "Phishlens Build Spec" (Claude Artifact, Rishitha's account)
- Full project narrative + all technical detail: Claude Project "RESEARCH PAPER" (Rishitha's account — ask her if you need something from it, since it's not shared to the repo)
