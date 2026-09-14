# Phishlens

A 3-class phishing detector — **legitimate / human-written phishing / LLM-generated phishing** — built as the demo for the paper *Detecting AI-Generated Phishing Content* (Rishitha, Poojitha & Vedha Sri, Woxsen University, Sept 2026).

Paste an email, get a verdict, a confidence, and a short "why this was flagged". Every check is logged.

| | |
|---|---|
| **Live API** | https://phishlens-api-tlx8.onrender.com — [`/api/health`](https://phishlens-api-tlx8.onrender.com/api/health) · [`/docs`](https://phishlens-api-tlx8.onrender.com/docs) (interactive) |
| **Frontend** | https://sairishitha-2787.github.io/phishlens/frontend/ |
| **Team status** | [`STATUS.md`](STATUS.md) — read first, update last |

> The API runs on Render's free tier and sleeps after 15 min idle. First request after that takes 30–60 s. Hit `/api/health` a minute before a demo.

## Results

Held-out test split, n = 825, seed 42. Full detail in [`backend/ml/artifacts/rebalanced_analysis.md`](backend/ml/artifacts/rebalanced_analysis.md).

| Model | Accuracy | Macro-F1 | ai_phishing F1 | legitimate F1 |
|---|---|---|---|---|
| **Linear SVM** (served) | 0.9927 | **0.9913** | 1.0000 | 0.9880 |
| Random Forest | 0.9915 | 0.9899 | 1.0000 | 0.9860 |
| Multinomial Naive Bayes | 0.9745 | 0.9718 | 0.9903 | 0.9581 |

Features: TF-IDF (word 1–2-grams + char 3–5-grams) plus a 6-feature stylometric block. All models `class_weight='balanced'`. Winner chosen by macro-F1, ties broken toward lower `ai_phishing` false-negative rate.

**Read these numbers with the confound analysis.** The first dataset build scored 0.9878 macro-F1 while detecting *which corpus a row came from*, not phishing — formatting alone reproduced 96.8% of the score, and the model flagged 93.5% of benign AI-written mail as phishing. [`confound_analysis.md`](backend/ml/artifacts/confound_analysis.md) documents that; [`rebalanced_analysis.md`](backend/ml/artifacts/rebalanced_analysis.md) shows how dataset v2 fixed it (benign-AI false-alarm rate → 0.0%). The paper's honest headline is that second story, not the F1.

Adversarial probing of the live model (5 hand-written cases, 4/5 pass): [`adversarial_test_results.md`](backend/ml/artifacts/adversarial_test_results.md).

## Repo layout

```
phishlens/
├── backend/                  FastAPI service + all ML work
│   ├── app/                  API: main.py, routers/, models/schemas.py (the API contract), models/inference.py
│   ├── ml/
│   │   ├── train.py          trains NB / SVM / RF, picks the winner
│   │   ├── features.py       stylometric features — single source of truth for train AND serve
│   │   ├── confound_analysis.py, rebuild_v2.py, within_corpus_check.py
│   │   ├── adversarial_probe.py
│   │   ├── data/data_card.md provenance, cleaning, split, confound history
│   │   └── artifacts/        model_v2.pkl (served), *_analysis.md, results.md, model_card.json
│   ├── tests/test_api.py
│   ├── schema.sql            Postgres schema (one table)
│   └── requirements.txt
├── frontend/                 static UI — index.html + app.js + styles.css, no build step
├── render.yaml               Render Blueprint for the API
└── STATUS.md                 team log
```

## API

Contract lives in [`backend/app/models/schemas.py`](backend/app/models/schemas.py) — read that before touching the frontend.

| Method | Path | |
|---|---|---|
| `POST` | `/api/predict` | body `{"text": "..."}` → label, confidence, probabilities, `explanation.top_features`, `model_version`, `truncated`, `created_at`. 400 on empty input; input over 3,000 chars is truncated (`truncated: true`). |
| `GET` | `/api/predictions?page=&page_size=` | history, newest first |
| `GET` | `/api/predictions/{id}` | one result (404 if unknown) |
| `GET` | `/api/health` | wake-up ping |
| `GET` | `/api/stats` | counts by class |

`confidence`/`probabilities` are a softmax over the SVM's decision scores — a sensible ranking, **not calibrated probabilities**.

## Run it locally

**Backend** (Python 3.11):

```bash
cd backend
python -m venv .venv && .venv\Scripts\activate      # or source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload                       # → http://localhost:8000/docs
python -m pytest tests/ -v
```

Uses a local SQLite file by default. Set `DATABASE_URL` to a Postgres URL (Neon's `postgres://…` form is fine) to use Postgres.

**Frontend** — it's static, but the API's CORS allow-list means you must serve it on port **5173**:

```bash
cd frontend
python -m http.server 5173                          # → http://localhost:5173
```

`app.js` points at the live API, so this hits real Postgres.

**Retrain** (needs `processed/dataset.csv`, which is not in the repo — see the data card):

```bash
cd backend/ml && python train.py
```

## Deploy

`render.yaml` is a Render Blueprint: *New + → Blueprint → this repo*. Set `DATABASE_URL` (Neon) in the dashboard. Env vars:

| Var | Value |
|---|---|
| `DATABASE_URL` | Neon connection string |
| `ALLOWED_ORIGINS` | comma-separated origins allowed to call the API; defaults to the GitHub Pages origin + `localhost:5173` |
| `MODEL_VERSION` | `svm-tfidf-v2-20260911` |

## Data

Not committed (size + licensing). Sources: CEAS-challenge/MIT list mail (legitimate), the Nazario Phishing Corpus (human phishing), and a GPT-generated phishing/legitimate set (Kaggle). 5,505 rows after exact + near-duplicate dedup, split 70/15/15 stratified. Everything — counts, licences, cleaning steps, what went wrong in v1 and how v2 fixed it — is in [`backend/ml/data/data_card.md`](backend/ml/data/data_card.md).

## Known limitations

- `human_phishing` comes from a single corpus (Nazario), so that class still carries a provenance fingerprint.
- One AI generator. "AI-written" and "this model's style" are not separable with this data.
- English only. Training data is English; other languages are untested.
- Confidence values are uncalibrated (see API).
- Perfect `ai_phishing` scores reflect a templated synthetic corpus, not real-world difficulty.

## Team

- **Rishitha** — dataset, models, backend
- **Poojitha** — frontend, adversarial testing
- **Vedha Sri** — infrastructure, deployment
