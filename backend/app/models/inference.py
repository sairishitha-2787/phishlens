"""
Loads the served artifact once at startup and runs prediction + explanation.

Feature layout (must match ml/train.py::build_features exactly):
    [ word TF-IDF (nw) | char TF-IDF (nc) | 6 stylometric, min-max scaled ]

Explanation (build spec §05): for a linear model the reason is simply the
input's top positively-weighted terms for the predicted class — no SHAP/LIME
dependency, which matters on Render's free-tier RAM. Contribution of feature j
to class c is coef[c, j] * x[j]; we surface the top word n-grams, and append
the URL count when links are present, since the within-corpus control
(rebalanced_analysis.md) confirmed link density as a genuinely phishing-relevant
signal rather than a corpus artifact.

On probabilities: the winner is a LinearSVC, which has no predict_proba. The
`probabilities` field is a softmax over its one-vs-rest decision scores. That
is a monotone, well-behaved confidence proxy but it is NOT calibrated — a 0.94
does not mean "right 94% of the time". Good enough for a demo's confidence
bar; if calibrated numbers are ever needed, wrap the model in
CalibratedClassifierCV on the val split and re-save.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
from scipy import sparse

from ..config import MODEL_PATH, MODEL_VERSION, TOP_FEATURES

# ml/features.py is the single source of truth for the stylometric block.
_ML_DIR = Path(__file__).resolve().parents[2] / "ml"
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))
from features import stylometrics, STYLO_NAMES, URL_RE  # noqa: E402
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # noqa: E402

_STYLO_URL_IDX = STYLO_NAMES.index("url_count")
_STYLO_URGENCY_IDX = STYLO_NAMES.index("urgency_keywords")


def _readable(term: str) -> bool:
    """
    Explanation-quality filter only — the model still uses every feature.
    Drops n-grams made entirely of stopwords / contraction fragments ("ll",
    "to you", "your"), which are high-weight but meaningless to a reader.
    """
    toks = term.split()
    return any(len(t) >= 3 and t not in ENGLISH_STOP_WORDS for t in toks)


class Predictor:
    def __init__(self, path: Path = MODEL_PATH):
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.word_vec = bundle["word_vectorizer"]
        self.char_vec = bundle["char_vectorizer"]
        self.scaler = bundle["scaler"]
        self.labels = list(self.model.classes_)
        self.version = MODEL_VERSION
        self.n_word = len(self.word_vec.vocabulary_)
        self.n_char = len(self.char_vec.vocabulary_)
        self._word_names = self.word_vec.get_feature_names_out()

    # ---------------------------------------------------------------- features

    def _featurize(self, text: str):
        xw = self.word_vec.transform([text])
        xc = self.char_vec.transform([text])
        raw_stylo = stylometrics([text])
        xs = np.clip(self.scaler.transform(raw_stylo), 0.0, 1.0)
        X = sparse.hstack([xw, xc, sparse.csr_matrix(xs)]).tocsr()
        return X, raw_stylo[0]

    # ------------------------------------------------------------- probability

    @staticmethod
    def _softmax(scores: np.ndarray) -> np.ndarray:
        z = scores - scores.max()
        e = np.exp(z)
        return e / e.sum()

    def _probabilities(self, X) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[0]
        scores = self.model.decision_function(X)
        scores = np.atleast_1d(scores[0] if scores.ndim == 2 else scores)
        return self._softmax(scores)

    # ------------------------------------------------------------- explanation

    def _explain(self, X, class_idx: int, raw_stylo: np.ndarray) -> list[str]:
        if not hasattr(self.model, "coef_"):
            return []
        coef = self.model.coef_[class_idx]
        row = X.tocoo()
        # contribution of each *present* feature to the predicted class
        contrib = {j: float(coef[j] * v) for j, v in zip(row.col, row.data)}

        # top positively-contributing word n-grams
        words = sorted(
            ((j, c) for j, c in contrib.items() if j < self.n_word and c > 0),
            key=lambda t: -t[1],
        )
        reasons = []
        for j, _ in words:
            term = str(self._word_names[j])
            if _readable(term):
                reasons.append(term)
            if len(reasons) == TOP_FEATURES:
                break

        # surface the URL count when links are present — a confirmed content signal
        n_urls = int(raw_stylo[_STYLO_URL_IDX])
        if n_urls > 0:
            reasons.append(f"contains {n_urls} link{'s' if n_urls != 1 else ''}")
        n_urg = int(raw_stylo[_STYLO_URGENCY_IDX])
        if n_urg >= 3 and self.labels[class_idx] != "legitimate":
            reasons.append(f"{n_urg} urgency cues")
        return reasons

    # ------------------------------------------------------------------ public

    def predict(self, text: str) -> dict:
        X, raw_stylo = self._featurize(text)
        probs = self._probabilities(X)
        idx = int(np.argmax(probs))
        return {
            "label": self.labels[idx],
            "confidence": float(probs[idx]),
            "probabilities": {lab: float(p) for lab, p in zip(self.labels, probs)},
            "top_features": self._explain(X, idx, raw_stylo),
        }


_predictor: Predictor | None = None


def load() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def get_predictor() -> Predictor:
    if _predictor is None:
        raise RuntimeError("model not loaded — call inference.load() at startup")
    return _predictor
