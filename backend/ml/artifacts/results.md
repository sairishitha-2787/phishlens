# Baseline results — Phishlens

Seed 42 · trained 20260911 · held-out test set (n=675) · all models `class_weight='balanced'`

## Overall

| Model | Accuracy | Macro-F1 | ai_phishing FN | ai_phishing FN rate |
|---|---|---|---|---|
| Linear SVM **←winner** | 0.9926 | 0.9878 | 0 | 0.0000 |
| Random Forest | 0.9926 | 0.9878 | 0 | 0.0000 |
| Multinomial Naive Bayes | 0.9793 | 0.9667 | 0 | 0.0000 |

## Per class

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|---|
| Linear SVM | `ai_phishing` | 1.0000 | 1.0000 | 1.0000 | 360 |
| Linear SVM | `human_phishing` | 0.9953 | 0.9815 | 0.9883 | 216 |
| Linear SVM | `legitimate` | 0.9608 | 0.9899 | 0.9751 | 99 |
| Random Forest | `ai_phishing` | 1.0000 | 1.0000 | 1.0000 | 360 |
| Random Forest | `human_phishing` | 0.9907 | 0.9861 | 0.9884 | 216 |
| Random Forest | `legitimate` | 0.9700 | 0.9798 | 0.9749 | 99 |
| Multinomial Naive Bayes | `ai_phishing` | 1.0000 | 1.0000 | 1.0000 | 360 |
| Multinomial Naive Bayes | `human_phishing` | 0.9951 | 0.9398 | 0.9667 | 216 |
| Multinomial Naive Bayes | `legitimate` | 0.8829 | 0.9899 | 0.9333 | 99 |

## Selection

Rule: highest macro-F1 on the held-out test set; ties broken toward the lower false-negative rate on `ai_phishing`.

- **Winner: Linear SVM** (macro-F1 0.9878)
- Near-tie at 4 dp between Linear SVM, Random Forest; separated only at full precision (gap 6.53e-05) — the two are **statistically indistinguishable**, same total error count on test.
- Saved to `backend/ml/artifacts/model_svm_20260911.pkl`
- `MODEL_VERSION=svm-tfidf-20260911`

### Note on class weighting
`MultinomialNB` exposes no `class_weight` parameter, so it was balanced via
`compute_sample_weight('balanced')` passed to `fit()` — the equivalent treatment.
The other two use `class_weight='balanced'` directly.
---

## ⚠ Validity warning — read before citing the table above

Run on `control_ai_legitimate.csv` (1256 AI-written **legitimate** emails), the winning model predicts:

| Predicted | Count | Share |
|---|---|---|
| `ai_phishing` | 1174 | 93.5% |
| `human_phishing` | 0 | 0.0% |
| `legitimate` | 82 | 6.5% |

**93.5% of benign AI-written email is flagged `ai_phishing`.**

Combined with the fact that every `ai_phishing` row comes from a single source (`gpt_dataset_label1`) and all three models score a perfect 1.0000 F1 on that class, this is strong evidence the classifier separates **generator source**, not phishing intent. The macro-F1 above measures dataset provenance, not detection skill, and should not be reported as detection performance without addressing the confound.
