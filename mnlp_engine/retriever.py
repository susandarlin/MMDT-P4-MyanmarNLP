"""
Task 4: Semantic Retrieval & Evidence Extraction (TF-IDF sparse vectors).

    1. Represent every corpus document d_j and every claim q as an
       L2-normalized TF-IDF vector (from vectorizer.py).
    2. CosineSim(q, d_j) = (q . d_j) / (||q|| * ||d_j||)
       Rank documents by score and keep the Top-K candidates.
       If the best score < tau  ->  flag the claim as NOT ENOUGH INFO.
    3. Evaluate with Precision@K, Recall@K and Mean Reciprocal Rank (MRR)
       against the ground-truth evidence_id.
"""

import numpy as np

from .vectorizer import MyanmarTfidfVectorizer, safe_dot

NEI = "NOT ENOUGH INFO"


def cosine_similarity(Q, D):
    """CosineSim(q, d) = q.d / (||q|| ||d||) for every claim row q and document row d."""
    q_norm = np.linalg.norm(Q, axis=1, keepdims=True)
    d_norm = np.linalg.norm(D, axis=1, keepdims=True)
    q_norm[q_norm == 0] = 1.0
    d_norm[d_norm == 0] = 1.0
    return safe_dot(Q / q_norm, D / d_norm)


class MyanmarRetriever:
    def __init__(self, vectorizer=None, tau=0.30):
        self.vectorizer = vectorizer or MyanmarTfidfVectorizer(ngram_range=(1, 2))
        self.tau = tau                 # empirical "no evidence" threshold
        self.doc_ids = None
        self.doc_texts = None
        self.D = None                  # document matrix (n_docs x |V|)

    def fit(self, doc_ids, doc_texts):
        """Index the corpus: learn vocabulary + IDF and vectorize every document."""
        self.doc_ids = np.asarray(doc_ids)
        self.doc_texts = list(doc_texts)
        self.D = self.vectorizer.fit_transform(self.doc_texts)
        return self

    def scores(self, claims):
        """Cosine similarity of every claim against every document -> (n_claims x n_docs)."""
        return cosine_similarity(self.vectorizer.transform(list(claims)), self.D)

    def search(self, claims, k=3):
        """Top-K retrieval.

        Returns one dict per claim:
            top_ids     : K best doc_ids (highest score first)
            top_scores  : their cosine scores
            best_id     : top-1 doc_id
            best_score  : top-1 score
            no_evidence : True if best_score < tau  (-> NOT ENOUGH INFO)
        """
        S = self.scores(claims)
        k = max(1, min(k, S.shape[1]))
        order = np.argsort(-S, axis=1)[:, :k]
        results = []
        for i, idx in enumerate(order):
            results.append({
                "top_ids": list(self.doc_ids[idx]),
                "top_scores": [float(S[i, j]) for j in idx],
                "best_id": self.doc_ids[idx[0]],
                "best_score": float(S[i, idx[0]]),
                "no_evidence": bool(S[i, idx[0]] < self.tau),
            })
        return results


# ---------------------------------------------------------------------------
# Threshold tau: chosen empirically on labelled training claims
# ---------------------------------------------------------------------------
def tune_tau(best_scores, labels, candidates=None):
    """Pick tau that best separates NOT ENOUGH INFO from claims that have evidence.

    A claim is flagged NEI when best_score < tau. We choose the tau with the
    highest accuracy of this NEI / has-evidence decision (ties -> middle value).
    """
    best_scores, is_nei = np.asarray(best_scores), np.asarray(labels) == NEI
    if candidates is None:
        candidates = np.round(np.arange(0.05, 0.80, 0.01), 2)
    accs = np.array([np.mean((best_scores < t) == is_nei) for t in candidates])
    best = np.flatnonzero(accs == accs.max())
    return float(candidates[best[len(best) // 2]]), float(accs.max())


# ---------------------------------------------------------------------------
# Retrieval evaluation (only claims that HAVE a ground-truth evidence_id)
# ---------------------------------------------------------------------------
def retrieval_metrics(ranked_ids, gold_ids, ks=(1, 3, 5)):
    """Precision@K, Recall@K and MRR.

    NOT ENOUGH INFO claims (evidence_id = "NONE") are skipped: they have no
    relevant document. Each remaining claim has exactly ONE relevant document (its evidence_id), so
        Precision@K = (# relevant in top K) / K
        Recall@K    = (# relevant in top K) / 1   (= hit rate)
        MRR         = mean of 1 / rank of the relevant document (0 if not found)
    """
    pairs = [(list(r), g) for r, g in zip(ranked_ids, gold_ids)
             if isinstance(g, str) and g.strip() and g.strip().upper() != "NONE"]
    out = {"n": len(pairs)}
    for k in ks:
        hits = [1.0 if g in r[:k] else 0.0 for r, g in pairs]
        out[f"P@{k}"] = float(np.mean(hits) / k) if hits else 0.0
        out[f"R@{k}"] = float(np.mean(hits)) if hits else 0.0
    rr = [1.0 / (r.index(g) + 1) if g in r else 0.0 for r, g in pairs]
    out["MRR"] = float(np.mean(rr)) if rr else 0.0
    return out


if __name__ == "__main__":
    import os
    import pandas as pd

    data = os.path.join(os.path.dirname(__file__), "..", "data")
    corpus = pd.read_csv(os.path.join(data, "corpus.csv"), encoding="utf-8")
    train = pd.read_csv(os.path.join(data, "train_claims.csv"), encoding="utf-8")
    allc = pd.read_csv(os.path.join(data, "claims.csv"), encoding="utf-8")

    retriever = MyanmarRetriever().fit(corpus["doc_id"], corpus["text"])

    # 1) tune tau on TRAIN claims
    tr = retriever.search(train["claim_text"], k=len(corpus))
    tau, acc = tune_tau([r["best_score"] for r in tr], train["label"])
    retriever.tau = tau
    print(f"Empirical threshold tau = {tau:.2f}  (NEI-vs-evidence accuracy on train: {acc:.3f})")
    print(pd.DataFrame({"label": train["label"], "best_score": [r["best_score"] for r in tr]})
          .groupby("label")["best_score"].agg(["mean", "min", "max"]).round(3))

    # 2) retrieval quality
    for name, df in [("train_claims", train), ("claims (all 62)", allc)]:
        res = retriever.search(df["claim_text"], k=len(corpus))
        m = retrieval_metrics([r["top_ids"] for r in res], df["evidence_id"])
        print(f"\n{name}: evaluated on {m['n']} claims with evidence")
        for k in (1, 3, 5):
            print(f"  P@{k} = {m[f'P@{k}']:.3f}   R@{k} = {m[f'R@{k}']:.3f}")
        print(f"  MRR = {m['MRR']:.3f}")
        flagged = [r["no_evidence"] for r in res]
        is_nei = (df["label"] == NEI).values
        print(f"  NEI flag (score < tau): correct on {np.mean(np.array(flagged) == is_nei):.3f} of claims")

    # 3) example Top-3
    print("\nExample Top-3 for the first train claim:")
    r = retriever.search(train["claim_text"][:1], k=3)[0]
    print(" ", train["claim_text"][0])
    for d, s in zip(r["top_ids"], r["top_scores"]):
        print(f"   {d}  {s:.3f}")
