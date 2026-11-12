"""
MMDT Myanmar NLP (MNLP) Project: Myanmar Text Claim Verification & Evidence Retrieval - Main Entry Point

Instructions for Students:
1. You have complete flexibility in how you design your project folder structure, classes, and helper modules.
2. Build your custom Myanmar reference corpus (corpus.csv) and curate at least 50 Myanmar claims (claims.csv / train_claims.csv / test_claims.csv).
3. Implement your Myanmar NLP verification and retrieval pipeline inside or called from `studentverifier()`.
4. Your verifier should read data dynamically from `data_dir` (default: 'data') using UTF-8 encoding.
   - If datasets in `data/` are updated or replaced, your solution should still work dynamically.
5. Your system must generate the final prediction file and save it as `<studentid>_predictions.csv` inside `output_dir` (default: 'data').
6. The function `studentverifier()` must return the path to the created CSV file.
7. In your report and video presentation, explain how your project directory, Myanmar syllable/word tokenization, TF-IDF vectorizer, Naïve Bayes classifier, and semantic retrieval/RAG pipelines are built.
"""

import os
import re
import csv

import sys

import pandas as pd

from mnlp_engine.rag_verifier import MyanmarRAGClaimVerifier
from mnlp_engine.tokenizer import tokenize
# Pipeline modules used inside the verifier:
#   mnlp_engine.preprocessor  - Unicode normalization (Task 2.1)
#   mnlp_engine.tokenizer     - sylbreak syllables + stopwords (Task 2.2)
#   mnlp_engine.vectorizer    - TF-IDF vectors (Task 2.3)
#   mnlp_engine.retriever     - cosine Top-K retrieval + tau (Task 4)
#   mnlp_engine.rag_verifier  - stance verification + explanations (Task 5)
#   mnlp_engine.naive_bayes / evaluation - Naive Bayes baseline + metrics (Task 3)

VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
# Print every pipeline step to the terminal (python app.py --quiet turns it off)
VERBOSE = True

OUTPUT_COLUMNS = ["claim_id", "predicted_label", "retrieved_evidence_id",
                  "confidence", "explanation"]


def _read_csv(data_dir, name):
    """Read a CSV from data_dir with UTF-8 (utf-8-sig also strips a BOM if present)."""
    path = os.path.join(data_dir, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")


def get_student_id():
    """
    Helper function to extract your MMDT ID from report.md.
    Alternatively, you may specify your student ID directly.
    """
    report_path = os.path.join(os.path.dirname(__file__), 'report.md')
    if os.path.exists(report_path):
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
                match = re.search(r'-\s*\*\*MMDT ID:\*\*\s*(.*)', content)
                if match:
                    val = match.group(1).strip()
                    if val and not (val.startswith('[') and val.endswith(']')):
                        return re.sub(r'[^a-zA-Z0-9_-]', '_', val)
        except Exception:
            pass
    return "student"


def studentverifier(data_dir="data", output_dir="data", student_id=None):
    """
    Main entry point for Myanmar text claim verification and evidence retrieval.

    Pipeline Steps:
    1. Myanmar Text Preprocessing & Syllable/Word Tokenization (UTF-8)
    2. Feature Extraction (TF-IDF / Syllable N-grams)
    3. Evidence Retrieval (Vector / Lexical similarity search from corpus.csv)
    4. Stance / Claim Classification (SUPPORTS / REFUTES / NOT ENOUGH INFO)
    5. Evidence-Grounded Explanation / RAG Synthesis in Myanmar context

    Args:
        data_dir (str): Directory containing input CSV files (corpus.csv, claims.csv, train_claims.csv, test_claims.csv).
        output_dir (str): Directory where '<studentid>_predictions.csv' will be saved.
        student_id (str, optional): The student ID to use in the filename. Defaults to ID in report.md.

    Returns:
        str: Path to the generated '<studentid>_predictions.csv' file.
    """
    if student_id is None:
        student_id = get_student_id()

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{student_id}_predictions.csv"
    output_path = os.path.join(output_dir, output_filename)

    # =========================================================================
    # Myanmar NLP Claim Verification & Retrieval Pipeline
    #
    # Expected CSV columns:
    # claim_id,predicted_label,retrieved_evidence_id,confidence,explanation
    #
    # Labels must be one of: 'SUPPORTS', 'REFUTES', 'NOT ENOUGH INFO'
    # =========================================================================

    # 1. Load corpus.csv, train_claims.csv, and test_claims.csv dynamically with utf-8 encoding.
    corpus = _read_csv(data_dir, "corpus.csv")
    train = _read_csv(data_dir, "train_claims.csv")
    test = _read_csv(data_dir, "test_claims.csv")
    if corpus is None or corpus.empty:
        raise FileNotFoundError(f"corpus.csv not found or empty in '{data_dir}'")
    if test is None or test.empty:
        raise FileNotFoundError(f"test_claims.csv not found or empty in '{data_dir}'")
    claims = test["claim_text"].tolist()
    if VERBOSE:
        _show_loading(corpus, train, test)

    # 2. Preprocess Myanmar text (Unicode normalization, syllable segmentation / regex tokenization, stopword removal).
    #    tokenize() = normalize_myanmar -> strip_punctuation -> syllable_break -> remove_stopwords
    corpus_tokens = [tokenize(t) for t in corpus["text"]]
    claim_tokens = [tokenize(t) for t in claims]
    if VERBOSE:
        _show_preprocessing(claims[0])
        print(f"  tokenized         : {len(corpus_tokens)} documents, {len(claim_tokens)} claims "
              f"(avg {sum(map(len, claim_tokens)) / len(claim_tokens):.1f} syllables per claim)")

    # 3. Build TF-IDF / Vector space representations for Myanmar documents and claims.
    #    fit() learns vocabulary + IDF from the corpus only; claims reuse the same vocabulary.
    verifier = MyanmarRAGClaimVerifier().fit(corpus)
    vectorizer = verifier.retriever.vectorizer
    doc_vectors = verifier.retriever.D                  # (n_docs   x |V|)
    claim_vectors = vectorizer.transform(claims)        # (n_claims x |V|)
    if VERBOSE:
        _show_tfidf(verifier)
        print(f"  claim matrix      : {claim_vectors.shape}  (claims x vocabulary)")

    # 4. Perform Top-K evidence retrieval for each claim against corpus passages.
    #    cosine(claim, doc) on L2-normalized TF-IDF vectors -> keep the K best documents.
    top_k = verifier.retriever.search(claims, k=3)

    # 5. Classify claims using Naive Bayes baseline and/or RAG verification.
    #    5a. Naive Bayes baseline (claim text only) + tune tau / coverage on train claims
    if train is not None and "label" in train:
        labelled = train[train["label"].isin(VALID_LABELS)]
        if len(labelled) >= 2:
            acc, tau, cov = verifier.tune(labelled["claim_text"], labelled["label"])
            if VERBOSE:
                _show_naive_bayes(verifier, labelled, test)
                _show_tuning(verifier, labelled, acc, tau, cov)
    #    5b. RAG verification: Top-K evidence -> stance rules -> grounded Myanmar explanation
    results = verifier.verify(claims, k=3)
    if VERBOSE:
        _show_results(test, results)
        _show_evaluation(test, results, _read_csv(data_dir, "claims.csv"))

    # 6. Save predictions to `output_path`.
    predictions = pd.DataFrame({
        "claim_id": test["claim_id"].values,
        "predicted_label": [r["predicted_label"] for r in results],
        "retrieved_evidence_id": [r["retrieved_evidence_id"] for r in results],
        "confidence": [round(float(r["confidence"]), 4) for r in results],
        "explanation": [r["explanation"] for r in results],
    }, columns=OUTPUT_COLUMNS)
    assert set(predictions["predicted_label"]) <= VALID_LABELS
    predictions.to_csv(output_path, index=False, encoding="utf-8", quoting=csv.QUOTE_NONNUMERIC)

    if VERBOSE:
        _step(9, "Save predictions")
        print(f"  wrote {len(predictions)} rows -> {output_path}")
        print(f"  label counts: {predictions['predicted_label'].value_counts().to_dict()}")
    return output_path


# ---------------------------------------------------------------------------
# Step-by-step printing helpers (off with: python app.py --quiet)
# ---------------------------------------------------------------------------
def _step(n, title):
    print(f"\n{'=' * 70}\nSTEP {n}: {title}\n{'=' * 70}")


def _show_loading(corpus, train, test):
    _step(1, "Load datasets (UTF-8)")
    print(f"  corpus.csv       : {len(corpus)} documents")
    print(f"  train_claims.csv : {0 if train is None else len(train)} labelled claims")
    if train is not None and "label" in train:
        print(f"                     {train['label'].value_counts().to_dict()}")
    print(f"  test_claims.csv  : {len(test)} claims")


def _show_preprocessing(example):
    from mnlp_engine.preprocessor import normalize_myanmar
    from mnlp_engine.tokenizer import strip_punctuation, syllable_break, tokenize
    _step(2, "Preprocessing (Task 2.1 + 2.2) - first test claim")
    norm = normalize_myanmar(example)
    syl = syllable_break(strip_punctuation(norm))
    print(f"  raw text          : {example}")
    print(f"  normalized (NFC)  : {norm}")
    print(f"  sylbreak          : {syl}")
    print(f"  after stopwords   : {tokenize(example)}")


def _show_tfidf(verifier):
    vec = verifier.retriever.vectorizer
    _step(3, "TF-IDF index of the corpus (Task 2.3)")
    print(f"  vocabulary size   : {len(vec.vocabulary_)} (syllables + syllable bigrams)")
    print(f"  document matrix   : {verifier.retriever.D.shape}  (documents x vocabulary)")
    order = vec.idf_.argsort()
    names = sorted(vec.vocabulary_, key=vec.vocabulary_.get)
    print(f"  lowest IDF (common): {[names[i] for i in order[:5]]}")
    print(f"  highest IDF (rare) : {[names[i] for i in order[-5:]]}")


def _show_tuning(verifier, labelled, acc, tau, cov):
    _step(5, "Retrieval threshold + stance thresholds tuned on train (Task 4 + 5)")
    print(f"  tau (NOT ENOUGH INFO if best cosine < tau) : {tau:.2f}")
    print(f"  coverage threshold for SUPPORTS            : {cov:.2f}")
    print(f"  train accuracy with these thresholds       : {acc:.3f}")


def _show_naive_bayes(verifier, labelled, test):
    """Task 3 baseline: Multinomial vs Complement NB on claim TF-IDF only."""
    import numpy as np
    from mnlp_engine.naive_bayes import (MultinomialNaiveBayesClassifier,
                                         ComplementNaiveBayesClassifier)
    from mnlp_engine.evaluation import classification_report
    _step(4, "Naive Bayes baseline on claim TF-IDF (Task 3)")
    vec = verifier.retriever.vectorizer
    X = vec.transform(labelled["claim_text"].tolist())
    y = labelled["label"].values
    has_gold = "label" in test and test["label"].isin(VALID_LABELS).all()
    Xt = vec.transform(test["claim_text"].tolist())
    print(f"  {'model':<16}{'LOO train acc':>14}{'macro-F1':>10}" + (f"{'test acc':>10}" if has_gold else ""))
    for name, cls in [("Multinomial NB", MultinomialNaiveBayesClassifier),
                      ("Complement NB", ComplementNaiveBayesClassifier)]:
        pred = np.array([cls(alpha=1.0).fit(np.delete(X, i, axis=0), np.delete(y, i))
                         .predict(X[i:i + 1])[0] for i in range(len(y))])
        rep = classification_report(y, pred)
        line = f"  {name:<16}{rep['accuracy']:>14.3f}{rep['macro_f1']:>10.3f}"
        if has_gold:
            tp = cls(alpha=1.0).fit(X, y).predict(Xt)
            line += f"{np.mean(tp == test['label'].values):>10.3f}"
        print(line)
    print("  -> claim words alone cannot tell true from false; RAG compares claim vs evidence.")


def _show_evaluation(test, results, all_claims=None):
    """Task 3/4 metrics on the test set. Gold labels come from test_claims.csv,
    or are looked up by claim_text in claims.csv when the test file is unlabelled."""
    from mnlp_engine.evaluation import classification_report, print_report
    from mnlp_engine.retriever import retrieval_metrics
    _step(8, "Evaluation on test claims (Accuracy, P, R, Macro-F1, Confusion Matrix)")
    if "label" not in test and all_claims is not None and "label" in all_claims:
        gold = all_claims.drop_duplicates("claim_text").set_index("claim_text")
        test = test.copy()
        test["label"] = test["claim_text"].map(gold["label"]).fillna("")
        if "evidence_id" in gold:
            test["evidence_id"] = test["claim_text"].map(gold["evidence_id"]).fillna("")
        print("  (gold labels looked up from claims.csv by matching claim_text)")
    if "label" not in test:
        print("  no gold labels available -> skipped")
        return
    keep = test["label"].isin(VALID_LABELS).values
    if not keep.any():
        print("  no gold labels available -> skipped")
        return
    if not keep.all():
        print(f"  evaluating {keep.sum()} of {len(test)} claims (others have no gold label)")
    test = test[keep]
    results = [r for r, k in zip(results, keep) if k]
    print_report(classification_report(test["label"].values,
                                       [r["predicted_label"] for r in results]))
    if "evidence_id" in test:
        m = retrieval_metrics([r["top_k_ids"] for r in results], list(test["evidence_id"]), ks=(1, 3))
        print(f"\n  Retrieval on {m['n']} claims with evidence: "
              f"P@1={m['P@1']:.3f}  R@3={m['R@3']:.3f}  MRR={m['MRR']:.3f}")


def _show_results(test, results):
    _step(6, "Retrieval (Top-3 cosine) + stance verification per test claim")
    for cid, text, r in zip(test["claim_id"], test["claim_text"], results):
        print(f"\n  [{cid}] {text}")
        print(f"    top-3 docs  : {r['top_k_ids']}   best cosine = {r['similarity']:.3f}")
        print(f"    evidence    : {' ။ '.join(r['evidence_sentences'])}")
        print(f"    rule fired  : {r['reason']}")
        print(f"    prediction  : {r['predicted_label']}   (confidence {r['confidence']})")
    _step(7, "Grounded explanations (RAG synthesis)")
    for cid, r in zip(test["claim_id"], results):
        print(f"\n  [{cid}] {r['explanation']}")


if __name__ == "__main__":
    # Trace every module by default; use `python app.py --quiet` to hide it
    VERBOSE = "--quiet" not in sys.argv
    generated_file = studentverifier()
    print(f"Myanmar claim verification completed. Output file: {generated_file}")
