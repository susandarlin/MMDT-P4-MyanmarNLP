"""
Task 3: Traditional NLP Classifier - Multinomial Naive Bayes on TF-IDF features.

    Class prior (smoothed):
        P(C_k) = (N_Ck + alpha) / (N + alpha * |C|)

    Feature likelihood (Laplace smoothing, alpha = 1), using summed TF-IDF weights:
        P(w_i | C_k) = ( sum_{d in C_k} TFIDF(w_i, d) + alpha )
                       / ( sum_w sum_{d in C_k} TFIDF(w, d) + alpha * |V| )

    Log-space inference (prevents numerical underflow):
        y_hat = argmax_k [ log P(C_k) + sum_i x_i * log P(w_i | C_k) ]
        x = the claim's TF-IDF vector
"""

import numpy as np

from .vectorizer import safe_dot

LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


class MultinomialNaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes_ = None            # array of class names
        self.log_prior_ = None          # shape (K,)
        self.log_likelihood_ = None     # shape (K, |V|)

    def fit(self, X, y):
        """X: TF-IDF matrix (n_claims x |V|), y: list of labels."""
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        present = set(y)
        self.classes_ = np.array([c for c in LABELS if c in present]
                                 + sorted(present - set(LABELS)))
        n, V = X.shape
        K = len(self.classes_)

        # Class prior P(C_k) = (N_Ck + a) / (N + a*|C|)
        n_ck = np.array([(y == c).sum() for c in self.classes_])
        self.log_prior_ = np.log((n_ck + self.alpha) / (n + self.alpha * K))

        # Likelihood P(w_i | C_k) from summed TF-IDF weights
        tfidf_sum = np.vstack([X[y == c].sum(axis=0) for c in self.classes_])   # (K, V)
        numer = tfidf_sum + self.alpha
        denom = tfidf_sum.sum(axis=1, keepdims=True) + self.alpha * V
        self.log_likelihood_ = np.log(numer / denom)
        return self

    def joint_log_likelihood(self, X):
        """log P(C_k) + sum_i x_i * log P(w_i | C_k)   -> shape (n, K)."""
        return safe_dot(X, self.log_likelihood_) + self.log_prior_

    def predict_proba(self, X):
        """Softmax of the log scores -> probabilities (used as `confidence`)."""
        jll = self.joint_log_likelihood(X)
        jll -= jll.max(axis=1, keepdims=True)
        p = np.exp(jll)
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.joint_log_likelihood(X).argmax(axis=1)]


class ComplementNaiveBayesClassifier(MultinomialNaiveBayesClassifier):
    """Complement Naive Bayes (Rennie et al., 2003) with Laplace smoothing.

    Instead of "how well does class k explain the words", CNB learns from the
    COMPLEMENT of each class (all claims NOT in k), which is more stable with
    small / imbalanced data and short texts:

        theta_{~k,i} = ( sum_{d not in C_k} TFIDF(w_i, d) + alpha )
                       / ( sum_w sum_{d not in C_k} TFIDF(w, d) + alpha * |V| )
        w_{k,i} = log theta_{~k,i},  then normalized:  w_{k,i} / sum_i |w_{k,i}|

        y_hat = argmin_k  sum_i x_i * w_{k,i}
              (the class whose complement fits the claim WORST)

    Stored as log_likelihood_ = -w, so predict()/predict_proba() are inherited.
    The class prior is not used (standard CNB, sklearn default).
    """

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        present = set(y)
        self.classes_ = np.array([c for c in LABELS if c in present]
                                 + sorted(present - set(LABELS)))
        V = X.shape[1]
        comp_sum = np.vstack([X[y != c].sum(axis=0) for c in self.classes_])   # (K, V)
        numer = comp_sum + self.alpha
        denom = comp_sum.sum(axis=1, keepdims=True) + self.alpha * V
        w = np.log(numer / denom)
        w /= np.abs(w).sum(axis=1, keepdims=True)          # weight normalization
        self.log_likelihood_ = -w
        self.log_prior_ = np.zeros(len(self.classes_))
        return self


if __name__ == "__main__":
    import os
    import pandas as pd
    from .vectorizer import MyanmarTfidfVectorizer
    from .evaluation import classification_report, print_report

    data = os.path.join(os.path.dirname(__file__), "..", "data")
    corpus = pd.read_csv(os.path.join(data, "corpus.csv"), encoding="utf-8")
    train = pd.read_csv(os.path.join(data, "train_claims.csv"), encoding="utf-8")

    vec = MyanmarTfidfVectorizer(ngram_range=(1, 2))
    vec.fit(corpus["text"])                       # vocabulary + IDF from the corpus
    X = vec.transform(train["claim_text"])        # claim TF-IDF vectors
    y = train["label"].values

    # Leave-one-out cross-validation: train on 48 claims, predict the 1 left out
    def loo(model_cls):
        return np.array([
            model_cls(alpha=1.0).fit(np.delete(X, j, axis=0), np.delete(y, j))
            .predict(X[j:j + 1])[0] for j in range(len(y))])

    rows = []
    for name, cls in [("Multinomial NB", MultinomialNaiveBayesClassifier),
                      ("Complement NB", ComplementNaiveBayesClassifier)]:
        pred = loo(cls)
        rep = classification_report(y, pred)
        print(f"\n=== {name} - leave-one-out on {len(y)} train claims ===")
        print_report(rep)
        rows.append((name, (pred == y).mean(), rep))

    print("\n=== Model comparison (leave-one-out) ===")
    print(f"{'model':<16}{'accuracy':>10}{'macro-F1':>10}")
    for name, acc, rep in rows:
        print(f"{name:<16}{acc:>10.3f}{rep['macro_f1']:>10.3f}")
