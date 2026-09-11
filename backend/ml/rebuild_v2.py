"""
Dataset v2 — merge AI-written legitimate mail INTO the legitimate class.

In v1 every class came from a disjoint corpus, so `ai_phishing` was perfectly
separable by generator style alone (see confound_analysis.md). v2 breaks that
collinearity deliberately: the `legitimate` class now contains BOTH human-written
mail (CEAS/MIT) and AI-written mail (the GPT corpus's label=0 rows).

If the classifier is detecting phishing content, performance should largely hold.
If it was only ever detecting "was this written by an LLM", `ai_phishing` vs
`legitimate` should collapse, because both classes now contain LLM text.

80% of the old control set is merged in; 20% is kept back, still disjoint, as a
fresh control. Existing v1 split assignments are preserved so the numbers stay
comparable to the earlier runs; only the new rows are freshly assigned.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import joblib
from scipy import sparse

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from train import LABELS, TARGET, SEED, ARTIFACTS, stylometrics, build_features
from confound_analysis import source_only_features, evaluate, make_models, control_rate

csv.field_size_limit(10**9)

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "processed"
V1 = PROC / "dataset_v1.csv"
CTRL_V1 = PROC / "control_ai_legitimate_v1.csv"
OUT_DATA = PROC / "dataset.csv"
OUT_CTRL = PROC / "control_ai_legitimate.csv"
AI_LEGIT_SRC = "gpt_dataset_label0"
HOLDOUT = 0.20


def read(p):
    return list(csv.DictReader(open(p, encoding="utf-8", errors="replace", newline="")))


# ------------------------------------------------------------ rebuild

def rebuild():
    rng = np.random.RandomState(SEED)
    base = read(V1)
    ctrl = read(CTRL_V1)

    idx = rng.permutation(len(ctrl))
    n_hold = int(round(len(ctrl) * HOLDOUT))
    hold_i, merge_i = idx[:n_hold], idx[n_hold:]
    held = [ctrl[i] for i in hold_i]
    merged = [ctrl[i] for i in merge_i]

    # stratified 70/15/15 for the newly merged rows (they're all one class/source)
    m_idx = rng.permutation(len(merged))
    n_tr = int(round(0.70 * len(merged)))
    n_va = int(round(0.15 * len(merged)))
    split_of = {}
    for rank, j in enumerate(m_idx):
        split_of[j] = "train" if rank < n_tr else ("val" if rank < n_tr + n_va else "test")

    rows = [{"text": r["text"], "label": r["label"], "source": r["source"],
             "split": r["split"]} for r in base]
    for j, r in enumerate(merged):
        rows.append({"text": r["text"], "label": "legitimate",
                     "source": AI_LEGIT_SRC, "split": split_of[j]})

    with open(OUT_DATA, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "label", "source", "split"])
        w.writeheader()
        w.writerows(rows)
    with open(OUT_CTRL, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["text", "source"])
        w.writeheader()
        w.writerows([{"text": r["text"], "source": r["source"]} for r in held])

    # integrity
    tex = defaultdict(set)
    for r in rows:
        tex[r["split"]].add(r["text"].strip())
    held_t = set(r["text"].strip() for r in held)
    assert not (tex["train"] & tex["test"]), "train/test leak"
    assert not (tex["train"] & tex["val"]), "train/val leak"
    assert not (tex["val"] & tex["test"]), "val/test leak"
    all_t = tex["train"] | tex["val"] | tex["test"]
    assert not (all_t & held_t), "control leaked into dataset"

    print(f"v2 dataset: {len(rows)} rows  (v1 {len(base)} + {len(merged)} merged AI-legit)")
    print(f"fresh control: {len(held)} rows, disjoint")
    print("  labels:", dict(Counter(r['label'] for r in rows)))
    print("  splits:", dict(Counter(r['split'] for r in rows)))
    lc = Counter(r["label"] for r in rows)
    print(f"  imbalance ai_phishing:legitimate = {lc['ai_phishing']/lc['legitimate']:.2f}:1")
    return rows, held


# ------------------------------------------------------------ analysis

def load_splits(rows):
    out = {}
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        out[s] = ([r["text"] for r in sub], np.array([r["label"] for r in sub]),
                  np.array([r["source"] for r in sub]))
    return out


def main():
    print("=" * 74)
    print("REBUILD v2 — AI-legitimate merged into the legitimate class")
    print("=" * 74)
    rows, held = rebuild()
    d = load_splits(rows)
    (tr_t, ytr, str_s), (va_t, yva, _), (te_t, yte, ste_s) = d["train"], d["val"], d["test"]
    held_t = [r["text"] for r in held]

    rep = {"seed": SEED, "generated": date.today().strftime("%Y-%m-%d"),
           "n_rows": len(rows), "holdout_frac": HOLDOUT,
           "labels": dict(Counter(r["label"] for r in rows)),
           "sources": dict(Counter(r["source"] for r in rows)),
           "control_n": len(held)}

    # ---------------- baselines on v2
    print("\n" + "=" * 74)
    print("BASELINES on v2 (raw text)")
    print("=" * 74)
    Xtr, (Xva, Xte), bundle, ndim = build_features(tr_t, [va_t, te_t])
    models = make_models(Xtr, ytr)
    res = {n: evaluate(n, m, Xte, yte) for n, m in models.items()}
    for r in sorted(res.values(), key=lambda x: -x["macro_f1"]):
        print(f"  {r['model']:<26} macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")
        for lab in LABELS:
            m = r["per_class"][lab]
            print(f"      {lab:<16} P={m['precision']:.4f} R={m['recall']:.4f} "
                  f"F1={m['f1']:.4f} n={m['support']}")
    best = max(res.values(), key=lambda r: r["macro_f1"])
    rep["baselines_v2"] = res
    rep["winner_v2"] = best["model"]

    # ---------------- THE key test: legitimate split by origin
    print("\n" + "-" * 74)
    print("KEY TEST — accuracy on legitimate test rows, by who wrote them")
    print("-" * 74)
    pred = models[best["model"]].predict(Xte)
    origin = np.where(ste_s == AI_LEGIT_SRC, "AI-written", "human-written")
    key = {}
    for grp in ("human-written", "AI-written"):
        mask = (yte == "legitimate") & (origin == grp)
        if mask.sum() == 0:
            continue
        acc = float((pred[mask] == "legitimate").mean())
        mis = Counter(pred[mask][pred[mask] != "legitimate"].tolist())
        key[grp] = {"n": int(mask.sum()), "recall": acc, "misclassified_as": dict(mis)}
        print(f"  legitimate / {grp:<14} n={mask.sum():>4}  "
              f"correct {acc:.4f}   misread as {dict(mis) or '—'}")
    rep["legitimate_by_origin"] = key

    # ai_phishing <-> legitimate confusion only
    cm = confusion_matrix(yte, pred, labels=LABELS)
    ia, il = LABELS.index("ai_phishing"), LABELS.index("legitimate")
    pair = {"ai_as_ai": int(cm[ia, ia]), "ai_as_legit": int(cm[ia, il]),
            "legit_as_ai": int(cm[il, ia]), "legit_as_legit": int(cm[il, il])}
    denom = pair["ai_as_ai"] + pair["ai_as_legit"] + pair["legit_as_ai"] + pair["legit_as_legit"]
    pair_acc = (pair["ai_as_ai"] + pair["legit_as_legit"]) / denom
    print(f"\n  ai_phishing<->legitimate pairwise accuracy: {pair_acc:.4f}  {pair}")
    rep["ai_vs_legit_pair"] = {**pair, "pairwise_accuracy": float(pair_acc)}

    # ---------------- fresh control set
    print("\n" + "-" * 74)
    print(f"FRESH CONTROL — {len(held_t)} held-out AI-written legitimate emails")
    print("-" * 74)
    dist, far = control_rate(models[best["model"]], bundle, held_t)
    for lab in LABELS:
        c = dist.get(lab, 0)
        print(f"  predicted {lab:<16}{c:>5}  ({c/len(held_t):>6.1%})")
    print(f"  -> ai_phishing false-alarm rate: {far:.1%}   (v1 was 93.5%)")
    rep["control_v2"] = {"dist": dist, "far": float(far), "v1_far": 0.935}

    # ---------------- Experiment A on v2
    print("\n" + "=" * 74)
    print("EXPERIMENT A on v2 — source-only (formatting, NO word content)")
    print("=" * 74)
    Ftr, fnames = source_only_features(tr_t)
    Fte, _ = source_only_features(te_t)
    sc = StandardScaler().fit(Ftr)
    svm_a = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=10000)
    svm_a.fit(sc.transform(Ftr), ytr)
    rf_a = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1).fit(Ftr, ytr)
    a_res = {
        "Linear SVM (source-only)": evaluate("Linear SVM (source-only)", svm_a,
                                             sc.transform(Fte), yte),
        "Random Forest (source-only)": evaluate("Random Forest (source-only)", rf_a, Fte, yte),
    }
    for r in a_res.values():
        print(f"  {r['model']:<30} macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")

    rf_src = RandomForestClassifier(n_estimators=300, random_state=SEED,
                                    n_jobs=-1).fit(Ftr, str_s)
    src_acc = accuracy_score(ste_s, rf_src.predict(Fte))
    print(f"  source recoverable from formatting alone: {src_acc:.4f} (v1 was 0.9052)")
    rep["experiment_a_v2"] = {"metrics": a_res, "source_recovery": float(src_acc),
                              "v1_source_recovery": 0.9052}

    joblib.dump({"model": models[best["model"]], "word_vectorizer": bundle["word"],
                 "char_vectorizer": bundle["char"], "scaler": bundle["scaler"],
                 "labels": LABELS, "model_version": f"v2-{best['model']}"},
                ARTIFACTS / "model_v2.pkl", compress=3)
    (ARTIFACTS / "rebalanced_analysis.json").write_text(
        json.dumps(rep, indent=2), encoding="utf-8")
    write_md(rep, res, a_res, key, pair, pair_acc, far, src_acc, len(held_t))
    print("\nwrote rebalanced_analysis.md + .json + model_v2.pkl")


def write_md(rep, res, a_res, key, pair, pair_acc, far, src_acc, n_ctrl):
    best = max(res.values(), key=lambda r: r["macro_f1"])
    ab = max(a_res.values(), key=lambda r: r["macro_f1"])
    ai_f1 = best["per_class"]["ai_phishing"]["f1"]
    lg_f1 = best["per_class"]["legitimate"]["f1"]
    ai_h = key.get("AI-written", {}).get("recall", float("nan"))
    hu_h = key.get("human-written", {}).get("recall", float("nan"))

    L = [
        "# Dataset v2 — does the model survive when both classes contain AI text?",
        "",
        f"Seed {SEED} · generated {rep['generated']} · companion to `results.md` and "
        "`confound_analysis.md`.",
        "",
        "## What changed",
        "",
        "In v1 each class came from a disjoint corpus, so `ai_phishing` was perfectly "
        "separable by generator style alone. v2 breaks that collinearity on purpose: "
        f"**80% of the old AI-legitimate control set ({rep['labels']['legitimate']} "
        "`legitimate` rows total) is merged into the `legitimate` class**, which now holds "
        "both human-written (CEAS/MIT) and AI-written (GPT) mail. The remaining 20% "
        f"({n_ctrl} rows) is kept back, still disjoint, as a fresh control.",
        "",
        "v1 split assignments are preserved for pre-existing rows so the numbers stay "
        "comparable; only newly merged rows were freshly assigned (70/15/15, seed 42). "
        "Train/val/test and control disjointness are asserted in code.",
        "",
        "| | v1 | v2 |", "|---|---|---|",
        f"| rows | 4,500 | {rep['n_rows']:,} |",
        f"| `legitimate` | 660 | {rep['labels']['legitimate']:,} |",
        f"| ai_phishing:legitimate | 3.63:1 | "
        f"{rep['labels']['ai_phishing']/rep['labels']['legitimate']:.2f}:1 |",
        f"| control set | 1,256 | {n_ctrl} |",
        "",
        "## Baselines on v2",
        "",
        "| Model | Accuracy | Macro-F1 | ai_phishing F1 | legitimate F1 |",
        "|---|---|---|---|---|",
    ]
    for r in sorted(res.values(), key=lambda x: -x["macro_f1"]):
        L.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                 f"{r['per_class']['ai_phishing']['f1']:.4f} | "
                 f"{r['per_class']['legitimate']['f1']:.4f} |")

    L += [
        "",
        "## The key test — legitimate accuracy by who wrote it",
        "",
        f"Best model ({best['model']}) on `legitimate` test rows, split by origin:",
        "",
        "| Origin | n | Correctly called `legitimate` | Misread as |",
        "|---|---|---|---|",
    ]
    for grp, v in key.items():
        mis = ", ".join(f"`{k}` ×{c}" for k, c in v["misclassified_as"].items()) or "—"
        L.append(f"| {grp} | {v['n']} | {v['recall']:.4f} | {mis} |")

    L += [
        "",
        f"`ai_phishing` ↔ `legitimate` pairwise accuracy: **{pair_acc:.4f}** "
        f"({pair['ai_as_legit']} ai→legit, {pair['legit_as_ai']} legit→ai).",
        "",
        f"### Fresh control set ({n_ctrl} held-out AI-written legitimate emails)",
        "",
        "| | v1 | v2 |", "|---|---|---|",
        f"| `ai_phishing` false-alarm rate | 93.5% | **{far:.1%}** |",
        "",
        "## Experiment A on v2 — source-only",
        "",
        "| Model | Accuracy | Macro-F1 |", "|---|---|---|",
    ] + [f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} |"
         for r in sorted(a_res.values(), key=lambda x: -x["macro_f1"])] + [
        "",
        f"Source recoverable from formatting alone: **{src_acc:.4f}** (v1: 0.9052).",
        "",
        "## Verdict",
        "",
        f"With AI-written text on both sides of the `ai_phishing`/`legitimate` boundary, "
        f"the best model scores **{best['macro_f1']:.4f}** macro-F1, `ai_phishing` F1 "
        f"**{ai_f1:.4f}**, and classifies AI-written legitimate mail correctly "
        f"**{ai_h:.1%}** of the time versus **{hu_h:.1%}** for human-written legitimate "
        f"mail. The benign-AI false-alarm rate moved from 93.5% (v1) to **{far:.1%}**.",
        "",
        "Read this against `confound_analysis.md`: in v1 the same false-alarm rate was "
        "93.5%, which is what a pure AI-style detector produces. The v2 number is the "
        "honest measure of whether phishing *content* is being detected, because style "
        "alone can no longer separate these two classes.",
        "",
        "### Caveats that still stand",
        "",
        "- `human_phishing` remains single-corpus (Nazario), so that class keeps its own "
        "provenance confound — v2 only fixes the `ai_phishing`/`legitimate` axis.",
        "- Still one generator. 'AI-written' and 'this model's style' remain "
        "indistinguishable; a second generator is needed to test transfer.",
        "- The merged AI-legitimate rows come from the same GPT corpus as `ai_phishing`, "
        "which is what makes this test meaningful — but it also means both AI classes "
        "share topic and vintage. Human-written mail is still older and differently "
        "distributed.",
        "- `phishing3.mbox` is still missing; counts shift if recovered.",
        "- v1 files are preserved as `processed/dataset_v1.csv` and "
        "`processed/control_ai_legitimate_v1.csv`. `processed/data_card.md` describes v1 "
        "and is now stale for the merged set.",
    ]
    (ARTIFACTS / "rebalanced_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
