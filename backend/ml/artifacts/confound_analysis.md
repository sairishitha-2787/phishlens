# Confound analysis — how much of the baseline score is corpus fingerprint?

Seed 42 · generated 2026-09-11 · held-out test set (n=675) · all models `class_weight='balanced'`

Companion to `results.md`. `results.md` reports what the baselines score; this file reports how much of that score is real.

## Summary

| Quantity | Macro-F1 |
|---|---|
| Baseline, raw text (best: Linear SVM) | **0.9878** |
| Experiment A — source-only, no word content (best: Random Forest (source-only)) | **0.9559** |
| Experiment B — sanitized retrain (best: Linear SVM) | **0.9853** |

- Formatting artifacts **alone** reproduce 96.8% of the baseline macro-F1 (0.9559 of 0.9878), using zero information about what the message actually says.
- After stripping source fingerprints, macro-F1 barely moves: 0.9878 → 0.9853 (**-0.0025**). Sanitization **did not remove the confound**.
- Benign-AI control false-alarm rate: 93.5% → 91.5% — still near-total.

> **Headline:** a model with no access to message content reaches 0.9559 macro-F1, and removing every formatting fingerprint we can detect costs the full model only 0.0025. The task as currently constructed is close to a corpus-identification task.

## Experiment A — source-only baseline

21 purely structural features — line-shape, whitespace style, encoding artifacts, header markers, character-class ratios. **No vocabulary, no n-grams, no semantic content of any kind.** A model here cannot possibly be detecting phishing; it can only be recognising which corpus a row came from.

| Model | Accuracy | Macro-F1 | ai_phishing F1 | ai FN rate |
|---|---|---|---|---|
| Random Forest (source-only) | 0.9719 | 0.9559 | 0.9986 | 0.0000 |
| Linear SVM (source-only) | 0.9170 | 0.8879 | 0.9741 | 0.0056 |

**Corpus identity is directly recoverable:** a Random Forest predicting the 6-way `source` field from these same formatting features alone reaches **0.9052 accuracy** on the test set. Since every label maps to a disjoint set of sources, recovering source *is* recovering the label.

Most informative formatting features:

| Feature | Importance |
|---|---|
| `frac_ws_newline` | 0.1738 |
| `header_markers` | 0.1713 |
| `n_lines` | 0.1197 |
| `punct_ratio` | 0.0918 |
| `avg_line_len` | 0.0799 |
| `n_at` | 0.0774 |
| `log_len` | 0.0631 |
| `max_line_len` | 0.0616 |
| `upper_ratio` | 0.0517 |
| `nonascii_ratio` | 0.0276 |

## Experiment B — sanitized retrain

Stripped before re-vectorising: email headers (`From:`/`To:`/`Date:`/`X-*` etc.), sender domains and addresses (→ `EMAILTOKEN`), URLs (→ `URLTOKEN`, preserving *that* a link exists but not its domain), greeting and sign-off boilerplate, quoted-reply blocks, HTML remnants, Unicode/encoding artifacts (`\xa0`, `•`, `\ufffd`, smart quotes), and **all line-break structure** — whitespace style was itself one of the strongest corpus tells.

Rows surviving with ≥20 characters of content: 3149/3150 train.

| Model | Accuracy | Macro-F1 | ai_phishing F1 | ai FN rate |
|---|---|---|---|---|
| Linear SVM | 0.9911 | 0.9853 | 1.0000 | 0.0000 |
| Random Forest | 0.9896 | 0.9830 | 1.0000 | 0.0000 |
| Multinomial Naive Bayes | 0.9778 | 0.9642 | 1.0000 | 0.0000 |

### Before / after

| Model | Raw macro-F1 | Sanitized macro-F1 | Δ |
|---|---|---|---|
| Multinomial Naive Bayes | 0.9667 | 0.9642 | -0.0025 |
| Linear SVM | 0.9878 | 0.9853 | -0.0025 |
| Random Forest | 0.9878 | 0.9830 | -0.0048 |

Source recoverability from formatting after sanitization: **0.7985** (was 0.9052).

### Benign-AI control set

| | Raw | Sanitized |
|---|---|---|
| `ai_phishing` false-alarm rate | 93.5% | 91.5% |

The control set is 1,256 AI-written but **legitimate** emails. A model detecting phishing intent should rarely flag them; a model detecting AI *style* will flag nearly all of them.

### What the model keys on — lexical evidence

Highest-weighted `ai_phishing` word features from the Linear SVM, **after** full sanitization:

| Term | Weight |
|---|---|
| `ensure` | 0.358 |
| `exclusive` | 0.356 |
| `finds you` | 0.339 |
| `finds` | 0.339 |
| `you well` | 0.339 |
| `hope this` | 0.338 |
| `secure` | 0.338 |
| `hope` | 0.334 |
| `our` | 0.334 |
| `valued` | 0.324 |
| `message finds` | 0.323 |
| `2023` | 0.313 |

These are not phishing indicators. `hope this message finds you well`, `valued`, `ensure`, `exclusive` are **stylistic tics of the generating model**. The classifier reaches a perfect 1.0000 F1 on `ai_phishing` by recognising how the generator writes, which is exactly why it also flags 91.5% of the benign AI-written control set. The same terms dominate before sanitization, so this is a content-level confound that no amount of header and boilerplate stripping can reach.

## Interpretation for the paper

The headline 0.9878 macro-F1 in `results.md` cannot be reported as phishing-detection performance. In this dataset each class is drawn from a disjoint corpus (`ai_phishing` ← GPT set, `human_phishing` ← Nazario, `legitimate` ← CEAS/MIT), so corpus identity and class label are perfectly collinear. Experiment A shows formatting alone — with no access to content — recovers 96.8% of that score, and source identity is recoverable at 0.9052 accuracy.

This is a dataset-construction limitation, not a modelling bug, and it is a known hazard in AI-generated-text detection work where the synthetic and human classes come from different collection pipelines. It should be stated plainly in Limitations, with Experiment A as the supporting evidence.

### What would actually fix it

1. **Source-balanced classes** — every class must draw from every corpus. In practice: generate AI phishing *and* AI legitimate mail, and obtain human phishing *and* human legitimate mail, from matched pipelines.
2. **A second AI generator** — with one generator, 'AI-written' and 'this specific model's style' are indistinguishable. A held-out generator tests whether the finding transfers.
3. **Do not report the sanitized number as a 'corrected' result.** Experiment B shows it is only 0.0025 below the raw figure and still flags 91.5% of benign AI text — it is the same confounded measurement with the headers removed, not a debiased one.
4. **Keep the control set as a standing check** — the benign-AI false-alarm rate is a more honest headline metric than macro-F1 on a confounded split.

### Residual limitations of this analysis

- Sanitization is regex-based and cannot remove *topic* differences between corpora (a 2000s Nazario phish and a 2025 GPT phish discuss different brands and services). Some of the surviving signal is still provenance, not phishing intent.
- With a single AI generator there is no way to separate 'AI-generated' from 'this generator'. The sanitized number is an upper bound on genuine skill, not a clean estimate.
- `phishing3.mbox` is still missing from the corpus; all counts shift if recovered.
