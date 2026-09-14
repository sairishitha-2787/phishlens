# Data Card — Phishlens training set (v2)

**Files:** `processed/dataset.csv` (5,505 rows) · `processed/control_ai_legitimate.csv` (251 rows)
**Random seed:** 42 throughout. **Generated:** 2026-09-11.
**Supersedes:** v1 (4,500 rows), preserved as `dataset_v1.csv` / `control_ai_legitimate_v1.csv`.

Built by `merge_and_clean.py` → `stage2_neardup_split.py` → `stage3_split_and_card.py` (v1), then `backend/ml/rebuild_v2.py` (v2 merge). Analyses referenced below live in `backend/ml/artifacts/`.

## Schema

| Column | Values |
|---|---|
| `text` | Cleaned message body, subject prepended where available |
| `label` | `ai_phishing` · `human_phishing` · `legitimate` |
| `source` | Which raw file the row came from (see below) |
| `split` | `train` · `val` · `test` — 70/15/15, stratified by label |

## Sources and verified provenance

| Raw file | Rows kept | → label | Provenance (verified, not assumed) |
|---|---|---|---|
| `legit.csv` | 660 | `legitimate` | **CEAS-challenge + MIT mailing-list mail.** Confirmed by recipient domains (`*.ceas-challenge.cc`, `handyboard@media.mit.edu`). **Not Enron** — the original plan named Enron; the downloaded file is a different corpus. Cite as CEAS/MIT. |
| `phishing.csv` | 463 | `human_phishing` | **Nazario Phishing Corpus.** Recipient addresses include `jose@monkey.org` (Jose Nazario's own address). |
| `phishing0.mbox` | 217 | `human_phishing` | Nazario, raw mbox. Filenames match Nazario's sequential archive naming. |
| `phishing1.mbox` | 208 | `human_phishing` | Nazario, raw mbox. |
| `phishing2.mbox` | 556 | `human_phishing` | Nazario, raw mbox. |
| `phishing3.mbox` | **0 — missing** | — | Present at first listing (20 MB), gone minutes later. Never deleted by our tooling; most likely Windows Defender quarantine (mbox files with live phishing URLs are a routine AV trigger). Check Defender → Protection history, add a folder exclusion, re-download, then re-run the pipeline. |
| `GPT_Phishing_Email_dataset.csv`, `label=1` | 2,396 | `ai_phishing` | GPT-generated synthetic phishing. Single generator. |
| `GPT_Phishing_Email_dataset.csv`, `label=0` | 1,005 + 251 | `legitimate` / control | GPT-generated synthetic **legitimate** mail. Its `label` column is phishing-vs-not, **not** AI-vs-human. Split 80/20: 1,005 rows merged into `legitimate` (v2), 251 held out as the control set. |

Licences: check each source's terms before publication — CEAS and Nazario are research corpora; the GPT set is a Kaggle upload and its licence should be confirmed against the upload page.

## Class balance (v2)

| Label | Count | Share | Composition |
|---|---|---|---|
| `ai_phishing` | 2,396 | 43.5% | 100% GPT-generated |
| `human_phishing` | 1,444 | 26.2% | 100% Nazario (csv + 3 mbox) |
| `legitimate` | 1,665 | 30.2% | 660 human-written (CEAS/MIT) + 1,005 AI-written (GPT) |

Imbalance `ai_phishing:legitimate` = **1.44:1** (v1 was 3.63:1). All baselines still use `class_weight='balanced'` (`MultinomialNB` via balanced `sample_weight`, as it has no `class_weight` parameter).

## Split

| Split | Rows | ai_phishing | human_phishing | legitimate |
|---|---|---|---|---|
| train | 3,854 | 1,677 | 1,011 | 1,166 |
| val | 826 | 359 | 217 | 250 |
| test | 825 | 360 | 216 | 249 |

v1 split assignments were **preserved** for all pre-existing rows so results stay comparable across v1/v2 analyses; only the 1,005 newly merged AI-legitimate rows were freshly assigned (704/151/150). Disjointness of train/val/test, and of the control set from all three, is asserted in `rebuild_v2.py`.

Text length: min 30, median 1,503, max 70,007 chars. **255 rows (4.6%) exceed 3,000 chars** — relevant to the API's truncation limit (build spec §11/§12): truncation will touch a small but non-zero slice of real samples.

## Control set — `control_ai_legitimate.csv`

251 AI-written legitimate emails (`source = gpt_dataset_label0`), never seen in training, val, or test. Purpose: a standing check on whether the model flags benign AI-written text as `ai_phishing`. A model detecting phishing *content* should rarely do so; a model detecting AI *style* will flag nearly all of it. In v2 the best model's false-alarm rate on this set is **0.0%** (v1: 93.5% — see below).

## Cleaning and deduplication (all before the split)

1. mbox parsing via Python `mailbox`/`email`: 2,293 messages, 0 parse errors. Dropped 1 Eudora folder-marker pseudo-message and 29 messages with <30 chars of real text after HTML stripping.
2. Whitespace normalisation, HTML-tag stripping from raw email exports.
3. **Within-source exact dedup** (lowercased, whitespace-collapsed hash): `legit.csv` 1000→702, `phishing.csv` 1000→500, mbox 2263→1617. The Nazario corpora are heavily duplicated internally.
4. **Cross-source exact dedup** (`phishing.csv` vs mbox): 0 overlaps despite the shared origin.
5. **Near-duplicate dedup** (SimHash 64-bit, Hamming ≤3): removed 42 `legitimate`, 673 `human_phishing`, 81 `ai_phishing`, 13 from the AI-legitimate pool. Conservative threshold — catches near-identical text, not "same template, different bank name".

## Confound history — read before citing any metric

**v1 (disjoint corpora) was measuring provenance, not phishing.** In v1 every class came from a different corpus, so corpus identity and label were perfectly collinear. `backend/ml/artifacts/confound_analysis.md` documents that:

- A model trained on **21 formatting-only features** (line shape, header markers, encoding artifacts — no word content at all) reached **0.9559 macro-F1**, i.e. 96.8% of the full model's 0.9878.
- Corpus identity was recoverable from formatting alone at 0.9052 accuracy.
- Stripping headers, domains, boilerplate and line structure barely moved the score (0.9878 → 0.9853) — the confound was at the content level: the top `ai_phishing` terms were generator stylistic tics (*"hope this message finds you well"*, *"valued"*, *"ensure"*), not phishing indicators.
- The v1 model flagged **93.5%** of benign AI-written mail as `ai_phishing`. It was an AI-style detector.

**v2 breaks the collinearity** by folding 1,005 AI-written legitimate rows into `legitimate`, so both sides of the `ai_phishing`/`legitimate` boundary contain LLM text and style alone can no longer separate them. `backend/ml/artifacts/rebalanced_analysis.md` documents the result:

- Best model (Linear SVM): **0.9913 macro-F1**, `ai_phishing` F1 1.0000, `legitimate` F1 0.9880.
- AI-written legitimate test rows correctly classified: **150/150**. `ai_phishing`↔`legitimate` pairwise accuracy 1.0000.
- Control-set false-alarm rate: **93.5% → 0.0%**.
- Within the GPT corpus alone (generator held constant), pure-artifact formatting features score 0.7130 vs a 0.7045 majority baseline — no signal. Phishing-relevant features (led by URL count, importance 0.32) score 0.8833. The separation is content-driven.

## Caveats that still stand

- **`human_phishing` is single-sourced (Nazario).** v2 fixed the `ai_phishing`/`legitimate` axis only; the human-phishing class still carries its own provenance fingerprint. The source-only baseline still reaches 0.9163 macro-F1 on v2 for this reason. State it in Limitations.
- **One generator.** "AI-written" and "this specific model's style" are indistinguishable here. Transfer to a second generator is untested.
- **Perfect scores are a warning, not a boast.** The synthetic corpus is templated; real-world phishing will not separate this cleanly.
- **Vintage mismatch.** Nazario and CEAS mail is 2000s–2010s; the GPT corpus references 2023–2025. Some residual signal is inevitably era, not intent.
- `phishing3.mbox` is still missing; every count above shifts if it is recovered.
