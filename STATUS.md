# Phishlens — Team Status

**Read this first. Update this last.** Whatever AI tool you're using this session — Claude Code, ChatGPT, Gemini, whatever — paste this instruction at the start: "Read STATUS.md before doing anything." And at the end: "Update STATUS.md with what you just did and what's next."

One line per entry: `- [date] [R/P/V] did X`. Newest entries go on top of each list. Don't delete old "Done" entries — that log is the project's memory across all of you and all your different tools.

---

## Now — in progress

- R: regenerating `processed/data_card.md` against the v2 dataset, then building the FastAPI backend (`/api/predict` etc.)
- P: (update this)
- V: (update this)

---

## Done — log, newest first

- 2026-09-11 R: live sprint checklist deployed — https://sairishitha-2787.github.io/phishlens/ — shared, real-time checklist (Firebase Firestore), synced across everyone who has it open. Replaces the old localStorage-only checklist file. Source lives in this repo as `index.html`.
- 2026-09-11 R: dataset v2 built and validated — folded AI-generated-legitimate rows into the `legitimate` class, fixing the v1 confound (model was detecting corpus/AI style, not phishing content). New control-set false-alarm rate: 0.0% (was 93.5%). See `backend/ml/artifacts/rebalanced_analysis.md`.
- 2026-09-11 R: confound diagnosed — v1 baselines (macro-F1 0.9878) were a dataset-construction artifact, not real phishing detection. Root cause + two diagnostic experiments in `backend/ml/artifacts/confound_analysis.md`.
- 2026-09-11 R: real dataset pipeline built — mbox parser, exact + SimHash near-dup dedup, stratified split. Verified real sources: `legit.csv` = CEAS-challenge + MIT mail (not Enron), `phishing.csv`/mbox = Nazario Phishing Corpus. `phishing3.mbox` still missing (likely Defender-quarantined).
- 2026-09-11 R: baseline models trained on v2 — Linear SVM wins outright, macro-F1 0.9913, saved as `backend/ml/artifacts/model_v2.pkl`. This is the model to serve.
- (earlier) R/P/V: topic locked, roles assigned, build spec written ("Phishlens Build Spec"), sprint plan set (Sept 7–20), faculty scope call done (topic approved, told to move fast — venue/disclosure specifics not yet discussed).

---

## Blocked / needs a decision

- Faculty hasn't weighed in on the confound finding specifically (call happened before it was found) — worth a short follow-up, not blocking.
- `phishing3.mbox` missing — R needs to check Windows Defender's quarantine and re-download. Pipeline runs without it for now.

---

## Next up — per track

- **R (data/model/backend):** regenerate `data_card.md` for v2 → build FastAPI `/api/predict` endpoint serving `model_v2.pkl` → connect frontend to live endpoint (joint with P)
- **P (frontend/security):** run adversarial prompt set against `model_v2.pkl` (doesn't need to wait for the live backend) → connect frontend to live endpoint (joint with R)
- **V (infra/backend):** blocked on R's backend for wiring Postgres logging in — can do prep (finalize schema, scaffold deployment target) in the meantime

---

## Reference docs (for context, not status)

- **Live sprint checklist (check things off here, it's shared):** https://sairishitha-2787.github.io/phishlens/
- Build spec: "Phishlens Build Spec" (Claude Artifact, Rishitha's account)
- Full project narrative + all technical detail: Claude Project "RESEARCH PAPER" (Rishitha's account — ask her if you need something from it, since it's not shared to the repo)
