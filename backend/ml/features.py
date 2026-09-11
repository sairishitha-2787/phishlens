"""
Feature builders shared by training and serving.

This is the single source of truth for the stylometric block. train.py and
app/models/inference.py both import from here — if they ever diverged, the
served model would silently see features it was never trained on.
"""

import re

import numpy as np

STYLO_NAMES = [
    "avg_word_len", "punct_ratio", "exclamations",
    "urgency_keywords", "url_count", "flesch_reading_ease",
]

URGENCY = [
    "verify", "immediately", "suspended", "urgent", "action required",
    "confirm", "expire", "click here", "account", "password", "security alert",
    "limited time", "act now", "final notice", "unauthorized", "locked",
]
URL_RE = re.compile(r"https?://|www\.", re.I)
WORD_RE = re.compile(r"[A-Za-z']+")
VOWEL_RUN_RE = re.compile(r"[aeiouy]+")


def _syllables(word):
    runs = VOWEL_RUN_RE.findall(word.lower())
    n = len(runs)
    if word.lower().endswith("e") and n > 1:
        n -= 1
    return max(n, 1)


def stylometrics(texts):
    """6 hand-built features, per build spec 05. Column order = STYLO_NAMES."""
    feats = np.zeros((len(texts), 6), dtype=np.float64)
    for i, t in enumerate(texts):
        words = WORD_RE.findall(t)
        n_words = len(words) or 1
        n_chars = len(t) or 1
        sentences = max(t.count(".") + t.count("!") + t.count("?"), 1)
        low = t.lower()

        avg_word_len = sum(len(w) for w in words) / n_words
        punct_ratio = sum(1 for c in t if c in ".,;:!?\"'()[]{}-") / n_chars
        excl = t.count("!")
        urgency = sum(low.count(k) for k in URGENCY)
        urls = len(URL_RE.findall(t))
        syl = sum(_syllables(w) for w in words)
        # Flesch Reading Ease
        flesch = 206.835 - 1.015 * (n_words / sentences) - 84.6 * (syl / n_words)

        feats[i] = [avg_word_len, punct_ratio, excl, urgency, urls, flesch]
    return feats
