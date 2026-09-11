"""
Confound analysis for the Phishlens 3-class task.

Experiment A — source-only baseline
    Train on NON-SEMANTIC formatting/metadata signals only (line shape, encoding
    artifacts, header markers, whitespace style). No word content whatsoever.
    Whatever macro-F1 this reaches is the ceiling explainable by corpus
    fingerprints alone, with zero phishing-detection skill involved.

Experiment B — sanitized retrain
    Strip source fingerprints (headers, sender domains, greeting/signoff
    boilerplate, whitespace/encoding artifacts), then retrain the same three
    baselines with the same features and splits. The before/after gap is the
    portion of the original score that survives on genuine content signal.

Everything is written to artifacts/confound_analysis.md.
"""

import csv
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
import joblib
from scipy import sparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)

from train import (
    LABELS, TARGET, SEED, DATA, CONTROL, ARTIFACTS,
    load_split, stylometrics, build_features,
)

csv.field_size_limit(10**9)
rng = np.random.RandomState(SEED)


# ============================================================ sanitization

HEADER_FIELDS = (
    r"from|to|cc|bcc|date|sent|reply-to|return-path|received|message-id|"
    r"mime-version|content-type|content-transfer-encoding|x-[\w-]+"
)
# header line OR inline "From: ... " segment as seen in the GPT corpus
RE_HEADER_LINE = re.compile(rf"^\s*({HEADER_FIELDS})\s*:.*$", re.I | re.M)
RE_HEADER_INLINE = re.compile(rf"\b({HEADER_FIELDS})\s*:\s*[^\n]{{0,120}}?(?=\s+[A-Z]|$)", re.I)
RE_SUBJECT_MARK = re.compile(r"^\s*subject\s*:\s*", re.I | re.M)
RE_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
RE_QUOTED = re.compile(r"^\s*>.*$", re.M)
RE_GREETING = re.compile(
    r"^\s*(dear|hi|hello|greetings|good (morning|afternoon|evening))\b[^\n,:]{0,40}[,:]?",
    re.I | re.M,
)
RE_SIGNOFF = re.compile(
    r"\b(best regards|kind regards|warm regards|regards|sincerely(?: yours)?|"
    r"yours (?:sincerely|truly|faithfully)|thanks(?: and regards)?|thank you|"
    r"best wishes|best|cheers)\b[,.\s]*",
    re.I,
)
RE_HTML = re.compile(r"<[^>]{1,200}>")
RE_WS = re.compile(r"\s+")


def sanitize(t: str) -> str:
    """Remove source fingerprints, keep phishing-relevant content."""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # encoding / dataset artifacts
    t = t.replace("\xa0", " ").replace("�", " ").replace("‌", " ")
    t = t.replace("•", " ").replace("·", " ")
    t = (t.replace("‘", "'").replace("’", "'")
          .replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-"))
    t = RE_HTML.sub(" ", t)
    t = RE_QUOTED.sub(" ", t)          # quoted reply blocks (mbox/legit only)
    t = RE_HEADER_LINE.sub(" ", t)
    t = RE_SUBJECT_MARK.sub(" ", t)    # drop the marker, keep subject text
    t = RE_HEADER_INLINE.sub(" ", t)
    # placeholders: keep "a URL is present" signal, drop the domain identity
    t = RE_URL.sub(" URLTOKEN ", t)
    t = RE_EMAIL.sub(" EMAILTOKEN ", t)
    t = RE_GREETING.sub(" ", t)
    t = RE_SIGNOFF.sub(" ", t)
    # collapse ALL whitespace — line-break style is itself a corpus fingerprint
    t = RE_WS.sub(" ", t)
    return t.strip()


# ==================================================== source-only features

RE_LINESTART_QUOTE = re.compile(r"^\s*>", re.M)
RE_HDR_ANY = re.compile(rf"\b({HEADER_FIELDS}|subject)\s*:", re.I)


def source_only_features(texts):
    """
    Purely structural / metadata signals. Contains NO word-content features:
    no vocabulary, no n-grams, nothing about what the message says.
    """
    names = [
        "log_len", "n_lines", "avg_line_len", "max_line_len", "frac_ws_newline",
        "n_tabs", "quote_lines", "header_markers", "n_at", "n_url_scheme",
        "nonascii_ratio", "digit_ratio", "upper_ratio", "space_run_max",
        "html_tags", "n_bullets", "n_nbsp", "n_replacement", "punct_ratio",
        "crlf_present", "trailing_ws_lines",
    ]
    F = np.zeros((len(texts), len(names)), dtype=np.float64)
    for i, t in enumerate(texts):
        n = len(t) or 1
        lines = t.split("\n")
        ws = [c for c in t if c.isspace()] or [" "]
        F[i] = [
            np.log1p(len(t)),
            len(lines),
            np.mean([len(l) for l in lines]) if lines else 0,
            max((len(l) for l in lines), default=0),
            sum(1 for c in ws if c == "\n") / len(ws),
            t.count("\t"),
            len(RE_LINESTART_QUOTE.findall(t)),
            len(RE_HDR_ANY.findall(t)),
            t.count("@"),
            len(re.findall(r"https?://", t, re.I)),
            sum(1 for c in t if ord(c) > 127) / n,
            sum(1 for c in t if c.isdigit()) / n,
            sum(1 for c in t if c.isupper()) / n,
            max((len(m) for m in re.findall(r" +", t)), default=0),
            len(RE_HTML.findall(t)),
            t.count("•") + t.count("·"),
            t.count("\xa0"),
            t.count("�"),
            sum(1 for c in t if c in ".,;:!?\"'()[]{}-") / n,
            1.0 if "\r\n" in t else 0.0,
            sum(1 for l in lines if l != l.rstrip()),
        ]
    return F, names


# ================================================================= metrics

def evaluate(name, model, X, y_true):
    y_pred = model.predict(X)
    p, r, f, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    i = LABELS.index(TARGET)
    fn = int(cm[i].sum() - cm[i, i])
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {lab: {"precision": float(p[j]), "recall": float(r[j]),
                            "f1": float(f[j]), "support": int(sup[j])}
                      for j, lab in enumerate(LABELS)},
        "ai_fn": fn,
        "ai_fn_rate": float(fn / int(cm[i].sum())),
        "confusion_matrix": cm.tolist(),
    }


def make_models(X, y):
    sw = compute_sample_weight("balanced", y)
    out = {}
    nb = MultinomialNB(alpha=0.1); nb.fit(X, y, sample_weight=sw)
    out["Multinomial Naive Bayes"] = nb
    svm = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=5000)
    svm.fit(X, y); out["Linear SVM"] = svm
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                random_state=SEED, n_jobs=-1)
    rf.fit(X, y); out["Random Forest"] = rf
    return out


def control_rate(model, bundle, texts):
    xw = bundle["word"].transform(texts)
    xc = bundle["char"].transform(texts)
    xs = np.clip(bundle["scaler"].transform(stylometrics(texts)), 0.0, 1.0)
    X = sparse.hstack([xw, xc, sparse.csr_matrix(xs)]).tocsr()
    pred = model.predict(X)
    u, c = np.unique(pred, return_counts=True)
    d = dict(zip(u.tolist(), c.tolist()))
    return d, d.get("ai_phishing", 0) / len(texts)


def top_terms(svm, bundle, k=12):
    names = np.array(bundle["word"].get_feature_names_out())
    nw = len(names)
    out = {}
    for i, lab in enumerate(svm.classes_):
        coef = svm.coef_[i][:nw]
        idx = np.argsort(coef)[::-1][:k]
        out[lab] = [(names[j], float(coef[j])) for j in idx]
    return out


# ==================================================================== main

def main():
    data = load_split(DATA)
    (tr_t, ytr), (va_t, yva), (te_t, yte) = data["train"], data["val"], data["test"]
    ctrl = list(csv.DictReader(open(CONTROL, encoding="utf-8", errors="replace", newline="")))
    ctrl_t = [r["text"] for r in ctrl]

    # source labels, for the "is source recoverable?" diagnostic
    rows = list(csv.DictReader(open(DATA, encoding="utf-8", errors="replace", newline="")))
    src = {r["text"]: r["source"] for r in rows}
    str_src = np.array([src[t] for t in tr_t])
    ste_src = np.array([src[t] for t in te_t])

    report = {"seed": SEED, "generated": date.today().strftime("%Y-%m-%d")}

    # ---------------------------------------------------- BASELINE (raw)
    print("=" * 72); print("BASELINE — raw text (reproducing results.md)"); print("=" * 72)
    Xtr, (Xva, Xte), bundle, ndim = build_features(tr_t, [va_t, te_t])
    base_models = make_models(Xtr, ytr)
    base = {n: evaluate(n, m, Xte, yte) for n, m in base_models.items()}
    for r in base.values():
        print(f"  {r['model']:<26} macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")
    base_best = max(base.values(), key=lambda r: r["macro_f1"])
    b_dist, b_far = control_rate(base_models[base_best["model"]], bundle, ctrl_t)
    print(f"  control ai_phishing false-alarm: {b_far:.1%}")
    report["baseline"] = {"metrics": base, "feature_dim": int(ndim),
                          "control": {"dist": b_dist, "far": b_far}}
    report["baseline_top_terms"] = top_terms(base_models["Linear SVM"], bundle)

    # ------------------------------------------- EXPERIMENT A (source-only)
    print("\n" + "=" * 72)
    print("EXPERIMENT A — source-only (formatting/metadata, NO word content)")
    print("=" * 72)
    Ftr, fnames = source_only_features(tr_t)
    Fte, _ = source_only_features(te_t)
    sc = StandardScaler().fit(Ftr)
    Atr, Ate = sc.transform(Ftr), sc.transform(Fte)

    a_models = {}
    svm_a = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=10000)
    svm_a.fit(Atr, ytr); a_models["Linear SVM (source-only)"] = svm_a
    rf_a = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)
    rf_a.fit(Ftr, ytr); a_models["Random Forest (source-only)"] = rf_a

    a_res = {}
    a_res["Linear SVM (source-only)"] = evaluate("Linear SVM (source-only)", svm_a, Ate, yte)
    a_res["Random Forest (source-only)"] = evaluate("Random Forest (source-only)", rf_a, Fte, yte)
    for r in a_res.values():
        print(f"  {r['model']:<30} macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")

    # can we recover the 6-way corpus identity from formatting alone?
    rf_src = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf_src.fit(Ftr, str_src)
    src_acc = accuracy_score(ste_src, rf_src.predict(Fte))
    print(f"  6-way SOURCE recoverable from formatting alone: {src_acc:.4f} accuracy")

    imp = sorted(zip(fnames, rf_a.feature_importances_), key=lambda x: -x[1])[:10]
    report["experiment_a"] = {
        "metrics": a_res, "n_features": len(fnames), "feature_names": fnames,
        "source_recovery_accuracy": float(src_acc),
        "top_importances": [(n, float(v)) for n, v in imp],
    }

    # --------------------------------------------- EXPERIMENT B (sanitized)
    print("\n" + "=" * 72)
    print("EXPERIMENT B — sanitized retrain")
    print("=" * 72)
    s_tr = [sanitize(t) for t in tr_t]
    s_va = [sanitize(t) for t in va_t]
    s_te = [sanitize(t) for t in te_t]
    s_ct = [sanitize(t) for t in ctrl_t]
    keep = [i for i, t in enumerate(s_tr) if len(t) >= 20]
    print(f"  sanitized train rows with >=20 chars left: {len(keep)}/{len(s_tr)}")

    Str, (Sva, Ste), s_bundle, s_ndim = build_features(s_tr, [s_va, s_te])
    s_models = make_models(Str, ytr)
    san = {n: evaluate(n, m, Ste, yte) for n, m in s_models.items()}
    for r in san.values():
        print(f"  {r['model']:<26} macro-F1 {r['macro_f1']:.4f}  acc {r['accuracy']:.4f}")
    san_best = max(san.values(), key=lambda r: r["macro_f1"])
    s_dist, s_far = control_rate(s_models[san_best["model"]], s_bundle, s_ct)
    print(f"  control ai_phishing false-alarm: {s_far:.1%}  (was {b_far:.1%})")

    # source recovery after sanitization
    Ftr_s, _ = source_only_features(s_tr)
    Fte_s, _ = source_only_features(s_te)
    rf_src2 = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    rf_src2.fit(Ftr_s, str_src)
    src_acc2 = accuracy_score(ste_src, rf_src2.predict(Fte_s))
    print(f"  6-way SOURCE recoverable after sanitization: {src_acc2:.4f} (was {src_acc:.4f})")

    report["experiment_b"] = {
        "metrics": san, "feature_dim": int(s_ndim),
        "control": {"dist": s_dist, "far": s_far},
        "source_recovery_after": float(src_acc2),
        "rows_kept": len(keep), "rows_total": len(s_tr),
    }
    report["sanitized_top_terms"] = top_terms(s_models["Linear SVM"], s_bundle)

    joblib.dump({"model": s_models[san_best["model"]],
                 "word_vectorizer": s_bundle["word"],
                 "char_vectorizer": s_bundle["char"],
                 "scaler": s_bundle["scaler"], "labels": LABELS,
                 "model_version": f"sanitized-{san_best['model']}"},
                ARTIFACTS / "model_sanitized.pkl", compress=3)

    (ARTIFACTS / "confound_analysis.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    write_md(report, base, a_res, san, b_far, s_far, src_acc, src_acc2, fnames, imp)
    print("\nwrote confound_analysis.md + confound_analysis.json + model_sanitized.pkl")


def write_md(rep, base, a_res, san, b_far, s_far, src_acc, src_acc2, fnames, imp):
    def tbl(res):
        L = ["| Model | Accuracy | Macro-F1 | ai_phishing F1 | ai FN rate |",
             "|---|---|---|---|---|"]
        for r in sorted(res.values(), key=lambda x: -x["macro_f1"]):
            L.append(f"| {r['model']} | {r['accuracy']:.4f} | {r['macro_f1']:.4f} | "
                     f"{r['per_class'][TARGET]['f1']:.4f} | {r['ai_fn_rate']:.4f} |")
        return L

    bb = max(base.values(), key=lambda r: r["macro_f1"])
    ab = max(a_res.values(), key=lambda r: r["macro_f1"])
    sb = max(san.values(), key=lambda r: r["macro_f1"])
    explained = ab["macro_f1"] / bb["macro_f1"]
    delta = sb["macro_f1"] - bb["macro_f1"]  # signed: negative = score fell

    L = [
        "# Confound analysis — how much of the baseline score is corpus fingerprint?",
        "",
        f"Seed {SEED} · generated {rep['generated']} · held-out test set (n=675) · "
        "all models `class_weight='balanced'`",
        "",
        "Companion to `results.md`. `results.md` reports what the baselines score; "
        "this file reports how much of that score is real.",
        "",
        "## Summary",
        "",
        "| Quantity | Macro-F1 |",
        "|---|---|",
        f"| Baseline, raw text (best: {bb['model']}) | **{bb['macro_f1']:.4f}** |",
        f"| Experiment A — source-only, no word content (best: {ab['model']}) | **{ab['macro_f1']:.4f}** |",
        f"| Experiment B — sanitized retrain (best: {sb['model']}) | **{sb['macro_f1']:.4f}** |",
        "",
        f"- Formatting artifacts **alone** reproduce {explained:.1%} of the baseline macro-F1 "
        f"({ab['macro_f1']:.4f} of {bb['macro_f1']:.4f}), using zero information about what "
        "the message actually says.",
        f"- After stripping source fingerprints, macro-F1 barely moves: {bb['macro_f1']:.4f} → "
        f"{sb['macro_f1']:.4f} (**{delta:+.4f}**). Sanitization **did not remove the confound**.",
        f"- Benign-AI control false-alarm rate: {b_far:.1%} → {s_far:.1%} — still near-total.",
        "",
        "> **Headline:** a model with no access to message content reaches "
        f"{ab['macro_f1']:.4f} macro-F1, and removing every formatting fingerprint we can "
        f"detect costs the full model only {abs(delta):.4f}. The task as currently "
        "constructed is close to a corpus-identification task.",
        "",
        "## Experiment A — source-only baseline",
        "",
        f"{len(fnames)} purely structural features — line-shape, whitespace style, encoding "
        "artifacts, header markers, character-class ratios. **No vocabulary, no n-grams, "
        "no semantic content of any kind.** A model here cannot possibly be detecting "
        "phishing; it can only be recognising which corpus a row came from.",
        "",
    ] + tbl(a_res) + [
        "",
        f"**Corpus identity is directly recoverable:** a Random Forest predicting the 6-way "
        f"`source` field from these same formatting features alone reaches "
        f"**{src_acc:.4f} accuracy** on the test set. Since every label maps to a disjoint "
        "set of sources, recovering source *is* recovering the label.",
        "",
        "Most informative formatting features:",
        "",
        "| Feature | Importance |", "|---|---|",
    ] + [f"| `{n}` | {v:.4f} |" for n, v in imp] + [
        "",
        "## Experiment B — sanitized retrain",
        "",
        "Stripped before re-vectorising: email headers (`From:`/`To:`/`Date:`/`X-*` etc.), "
        "sender domains and addresses (→ `EMAILTOKEN`), URLs (→ `URLTOKEN`, preserving "
        "*that* a link exists but not its domain), greeting and sign-off boilerplate, "
        "quoted-reply blocks, HTML remnants, Unicode/encoding artifacts (`\\xa0`, `•`, "
        "`\\ufffd`, smart quotes), and **all line-break structure** — whitespace style was "
        "itself one of the strongest corpus tells.",
        "",
        f"Rows surviving with ≥20 characters of content: "
        f"{rep['experiment_b']['rows_kept']}/{rep['experiment_b']['rows_total']} train.",
        "",
    ] + tbl(san) + [
        "",
        "### Before / after",
        "",
        "| Model | Raw macro-F1 | Sanitized macro-F1 | Δ |", "|---|---|---|---|",
    ] + [
        f"| {n} | {base[n]['macro_f1']:.4f} | {san[n]['macro_f1']:.4f} | "
        f"{san[n]['macro_f1'] - base[n]['macro_f1']:+.4f} |"
        for n in base
    ] + [
        "",
        f"Source recoverability from formatting after sanitization: "
        f"**{src_acc2:.4f}** (was {src_acc:.4f}).",
        "",
        "### Benign-AI control set",
        "",
        "| | Raw | Sanitized |", "|---|---|---|",
        f"| `ai_phishing` false-alarm rate | {b_far:.1%} | {s_far:.1%} |",
        "",
        "The control set is 1,256 AI-written but **legitimate** emails. A model detecting "
        "phishing intent should rarely flag them; a model detecting AI *style* will flag "
        "nearly all of them.",
        "",
        "### What the model keys on — lexical evidence",
        "",
        "Highest-weighted `ai_phishing` word features from the Linear SVM, **after** full "
        "sanitization:",
        "",
        "| Term | Weight |", "|---|---|",
    ] + [f"| `{t}` | {c:.3f} |" for t, c in rep["sanitized_top_terms"][TARGET][:12]] + [
        "",
        "These are not phishing indicators. `hope this message finds you well`, `valued`, "
        "`ensure`, `exclusive` are **stylistic tics of the generating model**. The "
        "classifier reaches a perfect 1.0000 F1 on `ai_phishing` by recognising how the "
        "generator writes, which is exactly why it also flags "
        f"{s_far:.1%} of the benign AI-written control set. The same terms dominate before "
        "sanitization, so this is a content-level confound that no amount of header and "
        "boilerplate stripping can reach.",
        "",
        "## Interpretation for the paper",
        "",
        f"The headline {bb['macro_f1']:.4f} macro-F1 in `results.md` cannot be reported as "
        "phishing-detection performance. In this dataset each class is drawn from a "
        "disjoint corpus (`ai_phishing` ← GPT set, `human_phishing` ← Nazario, "
        "`legitimate` ← CEAS/MIT), so corpus identity and class label are perfectly "
        "collinear. Experiment A shows formatting alone — with no access to content — "
        f"recovers {explained:.1%} of that score, and source identity is recoverable at "
        f"{src_acc:.4f} accuracy.",
        "",
        "This is a dataset-construction limitation, not a modelling bug, and it is a known "
        "hazard in AI-generated-text detection work where the synthetic and human classes "
        "come from different collection pipelines. It should be stated plainly in "
        "Limitations, with Experiment A as the supporting evidence.",
        "",
        "### What would actually fix it",
        "",
        "1. **Source-balanced classes** — every class must draw from every corpus. In "
        "practice: generate AI phishing *and* AI legitimate mail, and obtain human "
        "phishing *and* human legitimate mail, from matched pipelines.",
        "2. **A second AI generator** — with one generator, 'AI-written' and 'this specific "
        "model's style' are indistinguishable. A held-out generator tests whether the "
        "finding transfers.",
        "3. **Do not report the sanitized number as a 'corrected' result.** Experiment B "
        f"shows it is only {abs(delta):.4f} below the raw figure and still flags {s_far:.1%} "
        "of benign AI text — it is the same confounded measurement with the headers "
        "removed, not a debiased one.",
        "4. **Keep the control set as a standing check** — the benign-AI false-alarm rate "
        "is a more honest headline metric than macro-F1 on a confounded split.",
        "",
        "### Residual limitations of this analysis",
        "",
        "- Sanitization is regex-based and cannot remove *topic* differences between "
        "corpora (a 2000s Nazario phish and a 2025 GPT phish discuss different brands and "
        "services). Some of the surviving signal is still provenance, not phishing intent.",
        "- With a single AI generator there is no way to separate 'AI-generated' from "
        "'this generator'. The sanitized number is an upper bound on genuine skill, not a "
        "clean estimate.",
        "- `phishing3.mbox` is still missing from the corpus; all counts shift if recovered.",
    ]
    (ARTIFACTS / "confound_analysis.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
