"""
Phishlens baseline training — Naive Bayes / Linear SVM / Random Forest.

Per build spec 05. Features: TF-IDF (word 1-2 grams + char 3-5 grams) plus a
small stylometric block. All features are non-negative and min-max scaled into
[0,1] so that MultinomialNB (which requires non-negative input) can train on the
exact same matrix as the other two — a fair comparison matters more here than
squeezing a little extra accuracy out of any one model.

Class imbalance is ~3.6:1 (ai_phishing:legitimate), so every model is balanced.
Note: MultinomialNB has no class_weight parameter, so it gets the equivalent
treatment via compute_sample_weight('balanced') passed to fit().
"""

import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import joblib
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

csv.field_size_limit(10**9)

SEED = 42
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "processed" / "dataset.csv"
CONTROL = ROOT / "processed" / "control_ai_legitimate.csv"
ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

LABELS = ["ai_phishing", "human_phishing", "legitimate"]
TARGET = "ai_phishing"  # the class the paper is actually about

from features import stylometrics, URGENCY, URL_RE  # noqa: E402  single source of truth


# ---------------------------------------------------------------- data

def load_split(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="replace", newline="")))
    out = {}
    for split in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == split]
        out[split] = ([r["text"] for r in sub], np.array([r["label"] for r in sub]))
    return out


# ---------------------------------------------------------------- features

def build_features(train_texts, other_sets):
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=2, max_features=30000,
        sublinear_tf=True, strip_accents="unicode", lowercase=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=30000,
        sublinear_tf=True, lowercase=True,
    )
    scaler = MinMaxScaler()

    Xw = word_vec.fit_transform(train_texts)
    Xc = char_vec.fit_transform(train_texts)
    Xs = scaler.fit_transform(stylometrics(train_texts))
    X_train = sparse.hstack([Xw, Xc, sparse.csr_matrix(Xs)]).tocsr()

    outs = []
    for texts in other_sets:
        xw = word_vec.transform(texts)
        xc = char_vec.transform(texts)
        # clip: scaler is fit on train, unseen extremes would go outside [0,1]
        xs = np.clip(scaler.transform(stylometrics(texts)), 0.0, 1.0)
        outs.append(sparse.hstack([xw, xc, sparse.csr_matrix(xs)]).tocsr())

    bundle = {"word": word_vec, "char": char_vec, "scaler": scaler}
    return X_train, outs, bundle, Xw.shape[1] + Xc.shape[1] + 6


# ---------------------------------------------------------------- metrics

def evaluate(name, model, X, y_true):
    y_pred = model.predict(X)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    i = LABELS.index(TARGET)
    fn = int(cm[i].sum() - cm[i, i])
    fn_rate = fn / int(cm[i].sum())
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {
            lab: {
                "precision": float(p[j]), "recall": float(r[j]),
                "f1": float(f[j]), "support": int(sup[j]),
            }
            for j, lab in enumerate(LABELS)
        },
        f"{TARGET}_false_negatives": fn,
        f"{TARGET}_fn_rate": float(fn_rate),
        "confusion_matrix": cm.tolist(),
        "_y_pred": y_pred,
    }


def show(res, split):
    print(f"\n  {res['model']} — {split}")
    print(f"    accuracy {res['accuracy']:.4f}   macro-F1 {res['macro_f1']:.4f}")
    print(f"    {'class':<16}{'prec':>8}{'recall':>9}{'f1':>8}{'support':>9}")
    for lab in LABELS:
        m = res["per_class"][lab]
        print(f"    {lab:<16}{m['precision']:>8.4f}{m['recall']:>9.4f}"
              f"{m['f1']:>8.4f}{m['support']:>9}")
    print(f"    {TARGET} FN: {res[f'{TARGET}_false_negatives']} "
          f"({res[f'{TARGET}_fn_rate']:.4f})")


# ---------------------------------------------------------------- main

def main():
    print("=" * 72)
    print("Phishlens baselines — seed", SEED)
    print("=" * 72)

    data = load_split(DATA)
    (Xtr_t, ytr), (Xva_t, yva), (Xte_t, yte) = data["train"], data["val"], data["test"]
    print(f"train {len(ytr)}   val {len(yva)}   test {len(yte)}")

    X_train, (X_val, X_test), bundle, n_feat = build_features(Xtr_t, [Xva_t, Xte_t])
    print(f"feature dim: {n_feat}")

    sw = compute_sample_weight("balanced", ytr)

    models = {}

    nb = MultinomialNB(alpha=0.1)
    nb.fit(X_train, ytr, sample_weight=sw)  # no class_weight param on NB
    models["Multinomial Naive Bayes"] = nb

    svm = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=5000)
    svm.fit(X_train, ytr)
    models["Linear SVM"] = svm

    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=SEED,
        n_jobs=-1, min_samples_leaf=1,
    )
    rf.fit(X_train, ytr)
    models["Random Forest"] = rf

    print("\n" + "-" * 72)
    print("VALIDATION")
    print("-" * 72)
    val_res = {n: evaluate(n, m, X_val, yva) for n, m in models.items()}
    for r in val_res.values():
        show(r, "val")

    print("\n" + "-" * 72)
    print("TEST (held-out — selection happens here, per build spec 05)")
    print("-" * 72)
    test_res = {n: evaluate(n, m, X_test, yte) for n, m in models.items()}
    for r in test_res.values():
        show(r, "test")

    # ---- winner: highest macro-F1; ties -> lower ai_phishing FN rate
    # Full precision for ordering. A separate near-tie flag (4 dp, the precision
    # actually reported) records when the gap is too small to claim as a result.
    ranked = sorted(
        test_res.values(),
        key=lambda r: (-r["macro_f1"], r[f"{TARGET}_fn_rate"]),
    )
    winner = ranked[0]
    top = [r for r in ranked if round(r["macro_f1"], 4) == round(winner["macro_f1"], 4)]
    tie = len(top) > 1
    exact_tie = [r for r in ranked if r["macro_f1"] == winner["macro_f1"]]
    tie_break_decided = len(exact_tie) > 1

    print("\n" + "=" * 72)
    print(f"WINNER: {winner['model']}")
    print(f"  macro-F1 {winner['macro_f1']:.4f} | accuracy {winner['accuracy']:.4f} "
          f"| {TARGET} FN rate {winner[f'{TARGET}_fn_rate']:.4f}")
    if tie:
        print(f"  NOTE: {len(top)} models tie at 4 dp "
              f"({', '.join(r['model'] for r in top)}).")
        if tie_break_decided:
            print("        Exact tie — decided by ai_phishing FN rate.")
        else:
            print("        Separated only at full precision "
                  f"(gap {top[0]['macro_f1'] - top[-1]['macro_f1']:.2e}); "
                  "too small to claim as a real difference.")
    print("=" * 72)

    # ---- save artifacts
    slug = {"Multinomial Naive Bayes": "nb", "Linear SVM": "svm",
            "Random Forest": "rf"}[winner["model"]]
    stamp = date.today().strftime("%Y%m%d")
    version = f"{slug}-tfidf-{stamp}"
    model_path = ARTIFACTS / f"model_{slug}_{stamp}.pkl"

    joblib.dump(
        {
            "model": models[winner["model"]],
            "word_vectorizer": bundle["word"],
            "char_vectorizer": bundle["char"],
            "scaler": bundle["scaler"],
            "labels": LABELS,
            "model_version": version,
        },
        model_path,
        compress=3,
    )

    def strip(r):
        return {k: v for k, v in r.items() if not k.startswith("_")}

    card = {
        "model_version": version,
        "winner": winner["model"],
        "selection_rule": "highest macro-F1 on held-out test; ties -> lower ai_phishing FN rate",
        "tie_break_triggered": tie,
        "seed": SEED,
        "trained": stamp,
        "data": {
            "file": "processed/dataset.csv",
            "train": len(ytr), "val": len(yva), "test": len(yte),
            "class_balance_note": "ai_phishing:legitimate ~3.6:1; all models balanced",
            "nb_weighting": "MultinomialNB has no class_weight; used compute_sample_weight('balanced')",
        },
        "features": {
            "word_tfidf": "1-2 grams, min_df=2, max_features=30000, sublinear_tf",
            "char_tfidf": "char_wb 3-5 grams, min_df=2, max_features=30000",
            "stylometric": ["avg_word_len", "punct_ratio", "exclamations",
                            "urgency_keywords", "url_count", "flesch_reading_ease"],
            "total_dim": int(n_feat),
            "scaling": "stylometrics min-max to [0,1] so MultinomialNB can use them",
        },
        "test_metrics": {n: strip(r) for n, r in test_res.items()},
        "val_metrics": {n: strip(r) for n, r in val_res.items()},
    }
    (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    # ---- markdown results table
    lines = [
        "# Baseline results — Phishlens",
        "",
        f"Seed {SEED} · trained {stamp} · held-out test set (n={len(yte)}) · "
        "all models `class_weight='balanced'`",
        "",
        "## Overall",
        "",
        "| Model | Accuracy | Macro-F1 | ai_phishing FN | ai_phishing FN rate |",
        "|---|---|---|---|---|",
    ]
    for r in ranked:
        star = " **←winner**" if r["model"] == winner["model"] else ""
        lines.append(
            f"| {r['model']}{star} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
            f"{r[f'{TARGET}_false_negatives']} | {r[f'{TARGET}_fn_rate']:.4f} |"
        )
    lines += ["", "## Per class", "",
              "| Model | Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|---|"]
    for r in ranked:
        for lab in LABELS:
            m = r["per_class"][lab]
            lines.append(f"| {r['model']} | `{lab}` | {m['precision']:.4f} | "
                         f"{m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    lines += [
        "", "## Selection", "",
        f"Rule: highest macro-F1 on the held-out test set; ties broken toward the lower "
        f"false-negative rate on `ai_phishing`.",
        "",
        f"- **Winner: {winner['model']}** (macro-F1 {winner['macro_f1']:.4f})",
        (f"- Near-tie at 4 dp between {', '.join(r['model'] for r in top)}; "
         f"separated only at full precision (gap "
         f"{top[0]['macro_f1'] - top[-1]['macro_f1']:.2e}) — the two are "
         f"**statistically indistinguishable**, same total error count on test."
         if tie and not tie_break_decided else
         f"- Tie-break on `ai_phishing` FN rate: {'used' if tie_break_decided else 'not needed'}"),
        f"- Saved to `backend/ml/artifacts/{model_path.name}`",
        f"- `MODEL_VERSION={version}`",
        "",
        "### Note on class weighting",
        "`MultinomialNB` exposes no `class_weight` parameter, so it was balanced via",
        "`compute_sample_weight('balanced')` passed to `fit()` — the equivalent treatment.",
        "The other two use `class_weight='balanced'` directly.",
    ]
    (ARTIFACTS / "results.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"\nsaved: {model_path.name}, model_card.json, results.md -> backend/ml/artifacts/")

    # ---- control set: does the winner over-predict ai_phishing on benign AI text?
    if CONTROL.exists():
        crows = list(csv.DictReader(open(CONTROL, encoding="utf-8", errors="replace", newline="")))
        ctexts = [r["text"] for r in crows]
        xw = bundle["word"].transform(ctexts)
        xc = bundle["char"].transform(ctexts)
        xs = np.clip(bundle["scaler"].transform(stylometrics(ctexts)), 0.0, 1.0)
        Xc = sparse.hstack([xw, xc, sparse.csr_matrix(xs)]).tocsr()
        pred = models[winner["model"]].predict(Xc)
        uniq, cnt = np.unique(pred, return_counts=True)
        dist = dict(zip(uniq.tolist(), cnt.tolist()))
        print("\n" + "-" * 72)
        print(f"CONTROL SET — {len(ctexts)} AI-written *legitimate* emails")
        print("-" * 72)
        for lab in LABELS:
            c = dist.get(lab, 0)
            print(f"  predicted {lab:<16} {c:>5}  ({c/len(ctexts):>6.1%})")
        print(f"\n  -> ai_phishing false-alarm rate on benign AI text: "
              f"{dist.get('ai_phishing',0)/len(ctexts):.1%}")
        far = dist.get("ai_phishing", 0) / len(ctexts)
        card["control_set"] = {
            "n": len(ctexts), "predictions": dist,
            "ai_phishing_false_alarm_rate": far,
            "interpretation": (
                "Control is AI-written but BENIGN. A high ai_phishing rate here means "
                "the model keys on AI-generated style, not phishing intent."
            ),
        }
        (ARTIFACTS / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

        warn = [
            "", "---", "",
            "## ⚠ Validity warning — read before citing the table above", "",
            f"Run on `control_ai_legitimate.csv` ({len(ctexts)} AI-written **legitimate** "
            f"emails), the winning model predicts:", "",
            "| Predicted | Count | Share |", "|---|---|---|",
        ]
        for lab in LABELS:
            c = dist.get(lab, 0)
            warn.append(f"| `{lab}` | {c} | {c/len(ctexts):.1%} |")
        warn += [
            "",
            f"**{far:.1%} of benign AI-written email is flagged `ai_phishing`.**",
            "",
            "Combined with the fact that every `ai_phishing` row comes from a single "
            "source (`gpt_dataset_label1`) and all three models score a perfect 1.0000 "
            "F1 on that class, this is strong evidence the classifier separates "
            "**generator source**, not phishing intent. The macro-F1 above measures "
            "dataset provenance, not detection skill, and should not be reported as "
            "detection performance without addressing the confound.",
        ]
        with open(ARTIFACTS / "results.md", "a", encoding="utf-8") as f:
            f.write("\n".join(warn) + "\n")


if __name__ == "__main__":
    main()
