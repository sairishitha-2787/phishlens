# DistilBERT baseline — stretch goal (time-boxed)

Seed 42 · generated 2026-09-13 · `distilbert-base-uncased` · **REDUCED-SCALE CPU RUN** · device `cpu`

Same test split (n=825) as the SVM / RF / NB rows in `rebalanced_analysis.md`, so the test numbers are directly comparable. Training scale is **not** — see Config.

## Test-set results

| Metric | Value |
|---|---|
| Accuracy | **0.9758** |
| Macro-F1 | **0.9723** |
| `ai_phishing` F1 | 0.9945 |
| `human_phishing` F1 | 0.9633 |
| `legitimate` F1 | 0.9592 |
| `ai_phishing` FN rate | 0.0000 |

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| `ai_phishing` | 0.9890 | 1.0000 | 0.9945 | 360 |
| `human_phishing` | 0.9545 | 0.9722 | 0.9633 | 216 |
| `legitimate` | 0.9751 | 0.9438 | 0.9592 | 249 |

Confusion matrix (rows = true, cols = predicted; order ai_phishing / human_phishing / legitimate):

```
  360     0     0
    0   210     6
    4    10   235
```

## Config used

| | |
|---|---|
| Run type | **REDUCED-SCALE CPU RUN** |
| Train rows | 1200 of 3854 (stratified subsample, class proportions preserved) |
| Train class counts | {'ai_phishing': 522, 'human_phishing': 315, 'legitimate': 363} |
| Epochs | 1 |
| Batch size | 8 |
| Max length | 128 tokens |
| Learning rate | 5e-05 (AdamW, linear decay, 6% warmup) |
| Steps | 150 / 150 |
| Hardware | `cpu`, 10 torch threads |

## Wall-clock

Training 18.4 min · eval 1.0 min · total 20.6 min (excludes dependency install and model download).

Validation after each epoch: epoch 1 acc 0.9625 / macro-F1 0.9593

## How to read this against the classical baselines

**This is a reduced-scale CPU run and must be caveated as such in the paper.** It trained on 1200 of 3854 rows for 1 epoch at 128 tokens, because no CUDA GPU was available and the attempt was time-boxed. The SVM / RF / NB rows trained on the full 3,854-row split. Put it in the comparison table only with a footnote saying exactly that; do not present it as a like-for-like transformer result.

What it *does* show: how far a pretrained transformer gets on this task with a fraction of the data and compute, on the identical held-out test set.

## Caveats that carry over

- Same dataset, same confound history as `rebalanced_analysis.md` — `human_phishing` is still single-corpus (Nazario), still one AI generator.
- Model weights are **not** saved to the repo (~260 MB). Re-run the script to reproduce; seed 42 is fixed but CPU/GPU nondeterminism in attention kernels can shift the 4th decimal.
- Not deployed. The backend keeps serving `model_v2.pkl` (Linear SVM).
