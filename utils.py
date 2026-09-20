"""Shared helpers: text cleaning, tier assignment, and word-level explanations.

Used by both train.py (training) and app.py (the Streamlit demo), so the text is
cleaned in exactly the same way in both places.
"""
import re
from functools import lru_cache

import nltk


# ---------------------------------------------------------------- NLTK setup
def _have(resource: str) -> bool:
    for path in (f"corpora/{resource}", f"corpora/{resource}.zip"):
        try:
            nltk.data.find(path)
            return True
        except LookupError:
            continue
    return False


for _pkg in ("stopwords", "wordnet", "omw-1.4"):
    if not _have(_pkg):
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords  # noqa: E402  (after the downloads on purpose)
from nltk.stem import WordNetLemmatizer  # noqa: E402

# Keep negations: "not happy" must not become "happy".
NEGATIONS = {"not", "no", "nor", "never", "cannot"}
STOPWORDS = set(stopwords.words("english")) - NEGATIONS
# Leftovers from how this dataset was collected (HTML "&quot;", "&amp;", "URL" placeholders).
# They say nothing about a person's state, so keeping them would let the model "cheat".
STOPWORDS |= {"url", "quot", "amp", "nbsp"}
_lemmatizer = WordNetLemmatizer()

URL_RE = re.compile(r"http\S+|www\.\S+")
NON_LETTER_RE = re.compile(r"[^a-z\s]")


@lru_cache(maxsize=200_000)
def _lemma(word: str) -> str:
    # verb first ("feeling" -> "feel"), then noun ("friends" -> "friend")
    return _lemmatizer.lemmatize(_lemmatizer.lemmatize(word, pos="v"), pos="n")


def clean_text(text: str) -> str:
    """lowercase -> expand can't/won't/n't -> drop URLs & punctuation
    -> drop stopwords (keeping negations) -> lemmatize."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"n't\b", " not", text)
    text = NON_LETTER_RE.sub(" ", text)
    tokens = [w for w in text.split() if len(w) > 1 and w not in STOPWORDS]
    return " ".join(_lemma(w) for w in tokens)


# --------------------------------------------------------------------- tiers
def tier_for(prob: float, low_thr: float, high_thr: float) -> str:
    if prob < low_thr:
        return "Low"
    if prob < high_thr:
        return "Moderate"
    return "High"


# -------------------------------------------------------------- explanations
def explain(base_pipeline, cleaned_text: str, top_k: int = 8):
    """Exact word contributions for a linear model.

    contribution of a word = (its TF-IDF value in this text) x (its learned weight)
    Returns (words_pushing_up, words_pushing_down), each a list of (word, value).
    """
    vec = base_pipeline.named_steps["tfidf"]
    clf = base_pipeline.named_steps["clf"]
    x = vec.transform([cleaned_text]).tocoo()
    names = vec.get_feature_names_out()
    weights = clf.coef_[0]
    contribs = [(names[j], float(v * weights[j])) for j, v in zip(x.col, x.data)]
    up = sorted((c for c in contribs if c[1] > 0), key=lambda t: -t[1])[:top_k]
    down = sorted((c for c in contribs if c[1] < 0), key=lambda t: t[1])[:top_k]
    return up, down
