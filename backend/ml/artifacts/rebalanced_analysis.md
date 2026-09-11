# Dataset v2 — does the model survive when both classes contain AI text?

Seed 42 · generated 2026-09-11 · companion to `results.md` and `confound_analysis.md`.

## What changed

In v1 each class came from a disjoint corpus, so `ai_phishing` was perfectly separable by generator style alone. v2 breaks that collinearity on purpose: **80% of the old AI-legitimate control set (1665 `legitimate` rows total) is merged into the `legitimate` class**, which now holds both human-written (CEAS/MIT) and AI-written (GPT) mail. The remaining 20% (251 rows) is kept back, still disjoint, as a fresh control.

v1 split assignments are preserved for pre-existing rows so the numbers stay comparable; only newly merged rows were freshly assigned (70/15/15, seed 42). Train/val/test and control disjointness are asserted in code.

| | v1 | v2 |
|---|---|---|
| rows | 4,500 | 5,505 |
| `legitimate` | 660 | 1,665 |
| ai_phishing:legitimate | 3.63:1 | 1.44:1 |
| control set | 1,256 | 251 |

## Baselines on v2

| Model | Accuracy | Macro-F1 | ai_phishing F1 | legitimate F1 |
|---|---|---|---|---|
| Linear SVM | 0.9927 | 0.9913 | 1.0000 | 0.9880 |
| Random Forest | 0.9915 | 0.9899 | 1.0000 | 0.9860 |
| Multinomial Naive Bayes | 0.9745 | 0.9718 | 0.9903 | 0.9581 |

## The key test — legitimate accuracy by who wrote it

Best model (Linear SVM) on `legitimate` test rows, split by origin:

| Origin | n | Correctly called `legitimate` | Misread as |
|---|---|---|---|
| human-written | 99 | 0.9899 | `human_phishing` ×1 |
| AI-written | 150 | 1.0000 | — |

`ai_phishing` ↔ `legitimate` pairwise accuracy: **1.0000** (0 ai→legit, 0 legit→ai).

### Fresh control set (251 held-out AI-written legitimate emails)

| | v1 | v2 |
|---|---|---|
| `ai_phishing` false-alarm rate | 93.5% | **0.0%** |

## Experiment A on v2 — source-only

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest (source-only) | 0.9152 | 0.9163 |
| Linear SVM (source-only) | 0.7745 | 0.7546 |

Source recoverable from formatting alone: **0.8667** (v1: 0.9052).

## Verdict

With AI-written text on both sides of the `ai_phishing`/`legitimate` boundary, the best model scores **0.9913** macro-F1, `ai_phishing` F1 **1.0000**, and classifies AI-written legitimate mail correctly **100.0%** of the time versus **99.0%** for human-written legitimate mail. The benign-AI false-alarm rate moved from 93.5% (v1) to **0.0%**.

Read this against `confound_analysis.md`: in v1 the same false-alarm rate was 93.5%, which is what a pure AI-style detector produces. The v2 number is the honest measure of whether phishing *content* is being detected, because style alone can no longer separate these two classes.

### Caveats that still stand

- `human_phishing` remains single-corpus (Nazario), so that class keeps its own provenance confound — v2 only fixes the `ai_phishing`/`legitimate` axis.
- Still one generator. 'AI-written' and 'this model's style' remain indistinguishable; a second generator is needed to test transfer.
- The merged AI-legitimate rows come from the same GPT corpus as `ai_phishing`, which is what makes this test meaningful — but it also means both AI classes share topic and vintage. Human-written mail is still older and differently distributed.
- `phishing3.mbox` is still missing; counts shift if recovered.
- v1 files are preserved as `processed/dataset_v1.csv` and `processed/control_ai_legitimate_v1.csv`. `processed/data_card.md` describes v1 and is now stale for the merged set.

---

## Within-corpus control — is the v2 separation real?

The v2 result could still be an artifact if the GPT corpus's phishing and legitimate halves differ structurally rather than in content. Restricting to the GPT corpus alone (n=3401) holds generator, vintage and formatting pipeline constant, then re-runs the formatting-only feature set — decomposed into genuinely phishing-relevant signals versus pure corpus artifacts.

| Feature set | 5-fold CV accuracy |
|---|---|
| Majority-class baseline | 0.7045 |
| Pure-artifact subset (line shape, headers, encoding) | **0.7130** |
| Phishing-relevant subset (URLs, caps, digits, punctuation, length) | **0.8833** |
| All formatting features | 0.9244 |

**The pure-artifact features carry almost nothing**: 0.7130 against a 0.7045 majority baseline, a lift of just +0.0085. The separation instead comes from phishing-relevant properties, led by URL count:

| Feature | Importance |
|---|---|
| `n_url_scheme` | 0.3213 |
| `header_markers` | 0.1441 |
| `upper_ratio` | 0.1201 |
| `punct_ratio` | 0.1015 |
| `digit_ratio` | 0.0764 |
| `max_line_len` | 0.0493 |
| `avg_line_len` | 0.0485 |
| `log_len` | 0.0477 |

This is the opposite of the v1 picture, where the dominant features were `frac_ws_newline`, `header_markers` and `n_lines` — pure storage artifacts. Holding the generator constant, what separates phishing from legitimate is link density and emphasis, which is genuine phishing signal.

**Caveat on the feature split.** The 'phishing-relevant' grouping is a judgement call, and `log_len` in particular is arguably both. The robust claim is the negative one: the pure-artifact subset lands within 0.0085 of the majority baseline, so corpus artifacts cannot account for the v2 separation. The positive attribution to URL density is supported by the importance ranking but rests on that grouping.
