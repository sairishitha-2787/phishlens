"""
DistilBERT baseline for the Phishlens 3-class task — stretch goal, time-boxed.

Uses the SAME train/val/test split (the `split` column in processed/dataset.csv)
as the SVM / RF / NB baselines in rebalanced_analysis.md, so the numbers are
directly comparable.

Two configs, auto-selected:
  GPU (CUDA available):  full train split, 2 epochs, batch 16, max_length 256
  CPU (no CUDA):         stratified subsample of train (~1200 rows), 1 epoch,
                         batch 8, max_length 128  -> "reduced-scale CPU run"

The CPU config exists purely to fit a laptop time box; it is NOT equivalent to
the GPU config and results.md labels it as such.

Usage (from the repo's backend/ dir, in the transformer venv):
    python ml/artifacts/transformer_baseline/train_distilbert.py \
        --data "path/to/processed/dataset.csv" --max-minutes 45
"""

import argparse
import csv
import json
import random
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

csv.field_size_limit(10**9)

SEED = 42
LABELS = ["ai_phishing", "human_phishing", "legitimate"]  # same order as train.py
MODEL_NAME = "distilbert-base-uncased"
HERE = Path(__file__).resolve().parent

CONFIGS = {
    "gpu": dict(epochs=2, batch_size=16, max_length=128 * 2, train_rows=None, lr=5e-5),
    "cpu": dict(epochs=1, batch_size=8, max_length=128, train_rows=1200, lr=5e-5),
}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8", errors="replace", newline="")))
    out = {}
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["split"] == s]
        out[s] = ([r["text"] for r in sub], [LABELS.index(r["label"]) for r in sub])
    return out


def stratified_subsample(texts, labels, n, seed):
    """Keep class proportions while shrinking to ~n rows."""
    rng = random.Random(seed)
    by = {}
    for i, y in enumerate(labels):
        by.setdefault(y, []).append(i)
    total = len(labels)
    keep = []
    for y, idx in by.items():
        k = max(1, round(n * len(idx) / total))
        rng.shuffle(idx)
        keep.extend(idx[:k])
    rng.shuffle(keep)
    return [texts[i] for i in keep], [labels[i] for i in keep]


def batches(texts, labels, bs, tok, max_len, device, shuffle, seed):
    idx = list(range(len(texts)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for s in range(0, len(idx), bs):
        b = idx[s:s + bs]
        enc = tok([texts[i] for i in b], truncation=True, max_length=max_len,
                  padding=True, return_tensors="pt")
        yield ({k: v.to(device) for k, v in enc.items()},
               torch.tensor([labels[i] for i in b], device=device))


@torch.no_grad()
def predict(model, texts, labels, tok, cfg, device):
    model.eval()
    preds = []
    for enc, _ in batches(texts, labels, cfg["batch_size"] * 4, tok, cfg["max_length"], device, False, SEED):
        preds.extend(model(**enc).logits.argmax(-1).tolist())
    return preds


def metrics(y_true, y_pred):
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(3), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(3))
    ia = LABELS.index("ai_phishing")
    fn = int(cm[ia].sum() - cm[ia, ia])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(sup[i])} for i in range(3)},
        "ai_phishing_fn": fn,
        "ai_phishing_fn_rate": float(fn / max(int(cm[ia].sum()), 1)),
        "confusion_matrix": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--max-minutes", type=float, default=45.0,
                    help="training-loop guard: stop stepping past this and evaluate what exists")
    ap.add_argument("--force-cpu-config", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "cpu" if (device.type == "cpu" or args.force_cpu_config) else "gpu"
    cfg = CONFIGS[mode]
    print(f"device={device}  mode={mode}  cfg={cfg}  torch_threads={torch.get_num_threads()}")

    data = load(args.data)
    tr_t, tr_y = data["train"]
    full_train_n = len(tr_t)
    if cfg["train_rows"]:
        tr_t, tr_y = stratified_subsample(tr_t, tr_y, cfg["train_rows"], SEED)
    va_t, va_y = data["val"]
    te_t, te_y = data["test"]
    print(f"train {len(tr_t)} (of {full_train_n})  val {len(va_t)}  test {len(te_t)}")
    print("train class counts:", {LABELS[k]: v for k, v in sorted(Counter(tr_y).items())})

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    steps_per_epoch = (len(tr_t) + cfg["batch_size"] - 1) // cfg["batch_size"]
    total_steps = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)

    t_train0 = time.time()
    step = 0
    stopped_early = False
    val_history = []
    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for enc, y in batches(tr_t, tr_y, cfg["batch_size"], tok, cfg["max_length"], device, True, SEED + epoch):
            out = model(**enc, labels=y)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            running += out.loss.item(); step += 1
            if step % 10 == 0 or step == total_steps:
                el = (time.time() - t_train0) / 60
                eta = el / step * (total_steps - step)
                print(f"  epoch {epoch+1} step {step}/{total_steps}  loss {running/10:.4f}  "
                      f"elapsed {el:.1f}m  eta {eta:.1f}m", flush=True)
                running = 0.0
            if (time.time() - t_train0) / 60 > args.max_minutes:
                print(f"  !! time guard hit at step {step}/{total_steps} — stopping training early")
                stopped_early = True
                break
        vm = metrics(va_y, predict(model, va_t, va_y, tok, cfg, device))
        val_history.append({"epoch": epoch + 1, "steps": step, **{k: vm[k] for k in ("accuracy", "macro_f1")}})
        print(f"  [val] epoch {epoch+1}: acc {vm['accuracy']:.4f}  macro-F1 {vm['macro_f1']:.4f}")
        if stopped_early:
            break
    train_min = (time.time() - t_train0) / 60

    t_eval0 = time.time()
    tm = metrics(te_y, predict(model, te_t, te_y, tok, cfg, device))
    eval_min = (time.time() - t_eval0) / 60
    wall_min = (time.time() - t0) / 60

    label = ("FULL GPU RUN" if mode == "gpu" else "REDUCED-SCALE CPU RUN") + (" (TRAINING CUT SHORT BY TIME GUARD)" if stopped_early else "")
    print("\n" + "=" * 70); print(label); print("=" * 70)
    print(f"test accuracy  {tm['accuracy']:.4f}")
    print(f"test macro-F1  {tm['macro_f1']:.4f}")
    for lab in LABELS:
        m = tm["per_class"][lab]
        print(f"  {lab:<16} P {m['precision']:.4f}  R {m['recall']:.4f}  F1 {m['f1']:.4f}  n={m['support']}")
    print(f"ai_phishing FN rate {tm['ai_phishing_fn_rate']:.4f}")
    print(f"train {train_min:.1f} min | eval {eval_min:.1f} min | wall {wall_min:.1f} min")

    rep = {
        "generated": date.today().isoformat(), "seed": SEED, "model": MODEL_NAME,
        "mode": mode, "run_label": label, "config": cfg, "device": str(device),
        "cpu_threads": torch.get_num_threads(),
        "train_rows_used": len(tr_t), "train_rows_full": full_train_n,
        "train_class_counts": {LABELS[k]: v for k, v in sorted(Counter(tr_y).items())},
        "steps_completed": step, "steps_planned": total_steps, "stopped_early": stopped_early,
        "val_history": val_history, "test": tm,
        "timing_min": {"train": train_min, "eval": eval_min, "wall": wall_min},
    }
    (HERE / "results.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    write_md(rep)
    print(f"\nwrote {HERE / 'results.md'} and results.json")


def write_md(rep):
    tm, cfg = rep["test"], rep["config"]
    reduced = rep["mode"] == "cpu"
    L = [
        "# DistilBERT baseline — stretch goal (time-boxed)",
        "",
        f"Seed {rep['seed']} · generated {rep['generated']} · `{rep['model']}` · "
        f"**{rep['run_label']}** · device `{rep['device']}`",
        "",
        "Same test split (n=825) as the SVM / RF / NB rows in `rebalanced_analysis.md`, "
        "so the test numbers are directly comparable. Training scale is **not** — see Config.",
        "",
        "## Test-set results",
        "",
        "| Metric | Value |", "|---|---|",
        f"| Accuracy | **{tm['accuracy']:.4f}** |",
        f"| Macro-F1 | **{tm['macro_f1']:.4f}** |",
        f"| `ai_phishing` F1 | {tm['per_class']['ai_phishing']['f1']:.4f} |",
        f"| `human_phishing` F1 | {tm['per_class']['human_phishing']['f1']:.4f} |",
        f"| `legitimate` F1 | {tm['per_class']['legitimate']['f1']:.4f} |",
        f"| `ai_phishing` FN rate | {tm['ai_phishing_fn_rate']:.4f} |",
        "",
        "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|",
    ] + [f"| `{lab}` | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |"
         for lab, m in tm["per_class"].items()] + [
        "",
        "Confusion matrix (rows = true, cols = predicted; order ai_phishing / human_phishing / legitimate):",
        "", "```", *[" ".join(f"{v:>5}" for v in row) for row in tm["confusion_matrix"]], "```",
        "",
        "## Config used",
        "",
        "| | |", "|---|---|",
        f"| Run type | **{rep['run_label']}** |",
        f"| Train rows | {rep['train_rows_used']} of {rep['train_rows_full']} "
        + ("(stratified subsample, class proportions preserved)" if reduced else "(full split)") + " |",
        f"| Train class counts | {rep['train_class_counts']} |",
        f"| Epochs | {cfg['epochs']} |",
        f"| Batch size | {cfg['batch_size']} |",
        f"| Max length | {cfg['max_length']} tokens |",
        f"| Learning rate | {cfg['lr']} (AdamW, linear decay, 6% warmup) |",
        f"| Steps | {rep['steps_completed']} / {rep['steps_planned']}"
        + (" — **cut short by time guard**" if rep["stopped_early"] else "") + " |",
        f"| Hardware | `{rep['device']}`, {rep['cpu_threads']} torch threads |",
        "",
        "## Wall-clock",
        "",
        f"Training {rep['timing_min']['train']:.1f} min · eval {rep['timing_min']['eval']:.1f} min · "
        f"total {rep['timing_min']['wall']:.1f} min (excludes dependency install and model download).",
        "",
        "Validation after each epoch: " + ", ".join(
            f"epoch {h['epoch']} acc {h['accuracy']:.4f} / macro-F1 {h['macro_f1']:.4f}" for h in rep["val_history"]),
        "",
        "## How to read this against the classical baselines",
        "",
    ]
    if reduced:
        L += [
            "**This is a reduced-scale CPU run and must be caveated as such in the paper.** "
            f"It trained on {rep['train_rows_used']} of {rep['train_rows_full']} rows for "
            f"{cfg['epochs']} epoch at {cfg['max_length']} tokens, because no CUDA GPU was "
            "available and the attempt was time-boxed. The SVM / RF / NB rows trained on the "
            "full 3,854-row split. Put it in the comparison table only with a footnote saying "
            "exactly that; do not present it as a like-for-like transformer result.",
            "",
            "What it *does* show: how far a pretrained transformer gets on this task with a "
            "fraction of the data and compute, on the identical held-out test set.",
        ]
    else:
        L += [
            "Full-scale run on the same 3,854-row train split as the classical baselines; "
            "directly comparable.",
        ]
    L += [
        "",
        "## Caveats that carry over",
        "",
        "- Same dataset, same confound history as `rebalanced_analysis.md` — `human_phishing` "
        "is still single-corpus (Nazario), still one AI generator.",
        "- Model weights are **not** saved to the repo (~260 MB). Re-run the script to reproduce; "
        "seed 42 is fixed but CPU/GPU nondeterminism in attention kernels can shift the 4th decimal.",
        "- Not deployed. The backend keeps serving `model_v2.pkl` (Linear SVM).",
    ]
    (HERE / "results.md").write_text("\n".join(L) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
