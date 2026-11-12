"""
TF-IDF vectorization for Myanmar syllables (built from scratch).

    1. Vocabulary dictionary : {token: column index}, learned from the corpus
    2. TF   (term frequency)  : count(t, d) / total tokens in d
    3. IDF  (inverse doc freq): ln((1 + N) / (1 + df(t))) + 1   (smoothed)
    4. TF-IDF = TF x IDF, then L2-normalize every row (length = 1)

Golden rule: fit() ONCE on the corpus, then transform() claims with the SAME
vocabulary and IDF, so documents and claims share the same columns.
"""

import math
from collections import Counter

import numpy as np

from .tokenizer import tokenize


def add_ngrams(tokens, ngram_range=(1, 1)):
    """Add syllable n-grams: ['ပိ','ဋ','က'] + (1,2) -> [..., 'ပိ_ဋ', 'ဋ_က']."""
    lo, hi = ngram_range
    out = []
    for n in range(lo, hi + 1):
        if n == 1:
            out.extend(tokens)
        else:
            out.extend("_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return out


class MyanmarTfidfVectorizer:
    def __init__(self, tokenizer=tokenize, ngram_range=(1, 2), sublinear_tf=False):
        self.tokenizer = tokenizer
        self.ngram_range = ngram_range
        self.sublinear_tf = sublinear_tf
        self.vocabulary_ = {}     # {token: index}
        self.idf_ = None          # np.array of IDF values, one per vocabulary column

    # -- helpers ------------------------------------------------------------
    def _analyze(self, text):
        return add_ngrams(self.tokenizer(text), self.ngram_range)

    # -- 1 + 3: vocabulary and IDF (learned from the corpus) -----------------
    def fit(self, documents):
        docs = [self._analyze(d) for d in documents]
        vocab = sorted({t for d in docs for t in d})
        self.vocabulary_ = {t: i for i, t in enumerate(vocab)}

        n_docs = len(docs)
        df = Counter(t for d in docs for t in set(d))          # document frequency
        self.idf_ = np.array(
            [math.log((1 + n_docs) / (1 + df[t])) + 1 for t in vocab])
        return self

    # -- 2 + 4: TF x IDF, then L2-normalize ------------------------------------
    def transform(self, texts):
        if self.idf_ is None:
            raise ValueError("Call fit() on the corpus before transform().")
        X = np.zeros((len(texts), len(self.vocabulary_)))
        for row, text in enumerate(texts):
            tokens = self._analyze(text)
            if not tokens:
                continue
            for t, c in Counter(tokens).items():
                col = self.vocabulary_.get(t)       # unknown tokens are ignored
                if col is None:
                    continue
                tf = (1 + math.log(c)) if self.sublinear_tf else c / len(tokens)
                X[row, col] = tf
        X *= self.idf_                              # TF x IDF
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1.0                     # avoid division by zero
        return X / norms                            # L2-normalized rows

    def fit_transform(self, documents):
        return self.fit(documents).transform(documents)

    def get_feature_names(self):
        return sorted(self.vocabulary_, key=self.vocabulary_.get)


def safe_dot(A, B):
    """A @ B.T without spurious warnings.

    On macOS, NumPy 2.x + Apple Accelerate can print false
    'divide by zero / overflow / invalid value encountered in matmul' warnings
    for matrices that contain many zeros (sparse TF-IDF). The result is correct,
    so we silence those warnings and replace any non-finite value with 0.
    """
    A = np.ascontiguousarray(A, dtype=np.float64)
    B = np.ascontiguousarray(B, dtype=np.float64)
    with np.errstate(all="ignore"):
        out = A @ B.T
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_similarity(A, B):
    """Rows are already L2-normalized, so cosine similarity = dot product."""
    return safe_dot(A, B)


def show_idf_examples(vec, top_n=10):
    """Print the highest- and lowest-IDF tokens (for the report).

    High IDF = rare token (appears in few documents) -> very informative.
    Low IDF  = common token (appears in many documents) -> weak signal.
    """
    if vec.idf_ is None:
        raise ValueError("Call fit() on the corpus first.")
    names = vec.get_feature_names()
    order = np.argsort(vec.idf_)
    print(f"\nLowest IDF (most common) - top {top_n}:")
    for i in order[:top_n]:
        print(f"  {names[i]:<20} idf = {vec.idf_[i]:.3f}")
    print(f"\nHighest IDF (rarest) - top {top_n}:")
    for i in order[::-1][:top_n]:
        print(f"  {names[i]:<20} idf = {vec.idf_[i]:.3f}")
    return [(names[i], float(vec.idf_[i])) for i in order]


if __name__ == "__main__":
    import os
    import pandas as pd

    data = os.path.join(os.path.dirname(__file__), "..", "data")
    corpus = pd.read_csv(os.path.join(data, "corpus.csv"), encoding="utf-8")
    claims = pd.read_csv(os.path.join(data, "train_claims.csv"), encoding="utf-8")

    vec = MyanmarTfidfVectorizer(ngram_range=(1, 2))
    X_docs = vec.fit_transform(corpus["text"])
    X_claims = vec.transform(claims["claim_text"])

    print("vocabulary size :", len(vec.vocabulary_))
    print("X_docs shape    :", X_docs.shape)
    print("X_claims shape  :", X_claims.shape)
    print("row L2 norms    :", np.round(np.linalg.norm(X_docs, axis=1)[:3], 3))

    sims = cosine_similarity(X_claims, X_docs)
    pred = corpus["doc_id"].values[sims.argmax(axis=1)]
    has_ev = claims["label"] != "NOT ENOUGH INFO"
    acc = (pred[has_ev] == claims["evidence_id"].values[has_ev]).mean()
    print(f"top-1 retrieval accuracy (train, SUPPORTS/REFUTES): {acc:.3f}")

    show_idf_examples(vec, top_n=10)
