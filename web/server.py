"""
MNLP - web UI for the Myanmar claim verification project.

Run from the project root:
    python web/server.py            # http://127.0.0.1:5000
    python web/server.py --port 8080

Endpoints
    GET  /                       single-page UI
    GET  /api/files              list CSV files in data/
    GET  /api/files/<name>       CSV contents as JSON (columns + rows)
    GET  /api/files/<name>/raw   download the CSV
    GET  /api/stats              dataset overview numbers
    POST /api/verify             {"claim": "...", "pipeline": "A"|"B"} -> stance, evidence, explanation
                                 A = traditional NLP (TF-IDF + Naive Bayes), B = RAG verifier (default)
    POST /api/run                run studentverifier() and regenerate predictions
"""

import argparse
import io
import os
import sys
import threading
from contextlib import redirect_stdout
from datetime import datetime

import pandas as pd
from flask import Flask, abort, jsonify, request, send_from_directory

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app as mnlp_app  # noqa: E402  (project entry point)
from mnlp_engine.naive_bayes import MultinomialNaiveBayesClassifier  # noqa: E402
from mnlp_engine.rag_verifier import MyanmarRAGClaimVerifier  # noqa: E402

VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

web = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
web.json.ensure_ascii = False

_lock = threading.Lock()
_state = {"verifier": None, "nb": None, "signature": None, "titles": {}}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _csv_files():
    if not os.path.isdir(DATA_DIR):
        return []
    return sorted(f for f in os.listdir(DATA_DIR) if f.lower().endswith(".csv"))


def _safe_csv_path(name):
    if name not in _csv_files():
        abort(404, description=f"{name} not found in data/")
    return os.path.join(DATA_DIR, name)


def _read(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")


def _data_signature():
    """Changes whenever corpus/train files change, so the model is refit."""
    sig = []
    for name in ("corpus.csv", "train_claims.csv"):
        path = os.path.join(DATA_DIR, name)
        sig.append(os.path.getmtime(path) if os.path.exists(path) else None)
    return tuple(sig)


def _get_verifier():
    with _lock:
        sig = _data_signature()
        if _state["verifier"] is not None and _state["signature"] == sig:
            return _state["verifier"], _state["titles"]
        corpus = _read("corpus.csv")
        if corpus is None or corpus.empty:
            raise FileNotFoundError("data/corpus.csv is missing or empty")
        verifier = MyanmarRAGClaimVerifier().fit(corpus)
        nb = None
        train = _read("train_claims.csv")
        if train is not None and "label" in train:
            labelled = train[train["label"].isin(VALID_LABELS)]
            if len(labelled) >= 2:
                verifier.tune(labelled["claim_text"], labelled["label"])
                # Pipeline A: Naive Bayes on claim TF-IDF (vocabulary + IDF from the corpus)
                X = verifier.retriever.vectorizer.transform(labelled["claim_text"].tolist())
                nb = MultinomialNaiveBayesClassifier(alpha=1.0).fit(X, labelled["label"].values)
        _state.update(verifier=verifier, nb=nb, signature=sig,
                      titles=dict(zip(corpus["doc_id"], corpus.get("title", corpus["doc_id"]))))
        return verifier, _state["titles"]


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------
@web.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@web.get("/api/files")
def list_files():
    out = []
    for name in _csv_files():
        path = os.path.join(DATA_DIR, name)
        try:
            df = _read(name)
            rows, cols = len(df), list(df.columns)
        except Exception:  # unreadable CSV -> still list it
            rows, cols = None, []
        out.append({
            "name": name,
            "rows": rows,
            "columns": cols,
            "size": os.path.getsize(path),
            "modified": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds"),
        })
    return jsonify(out)


@web.get("/api/files/<path:name>")
def get_file(name):
    path = _safe_csv_path(name)
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str).fillna("")
    except Exception as e:
        return jsonify({"error": f"Could not parse {name}: {e}"}), 400
    return jsonify({"name": name, "columns": list(df.columns), "rows": df.values.tolist()})


@web.get("/api/files/<path:name>/raw")
def raw_file(name):
    _safe_csv_path(name)
    return send_from_directory(DATA_DIR, name, as_attachment=True)


@web.get("/api/stats")
def stats():
    corpus, claims = _read("corpus.csv"), _read("claims.csv")
    train, test = _read("train_claims.csv"), _read("test_claims.csv")

    def label_counts(df):
        if df is None or "label" not in df:
            return {}
        return df["label"].value_counts().to_dict()

    pred_file = next((f for f in _csv_files() if f.endswith("_predictions.csv")), None)
    preds = _read(pred_file) if pred_file else None
    pred_info = None
    if preds is not None and not preds.empty:
        conf = pd.to_numeric(preds.get("confidence"), errors="coerce")
        pred_info = {
            "file": pred_file,
            "rows": len(preds),
            "labels": preds["predicted_label"].value_counts().to_dict(),
            "mean_confidence": round(float(conf.mean()), 3) if conf.notna().any() else None,
        }
        # accuracy against claims.csv gold labels (matched by claim text)
        if test is not None and claims is not None and "label" in claims:
            gold = claims.drop_duplicates("claim_text").set_index("claim_text")["label"]
            merged = preds.merge(test[["claim_id", "claim_text"]], on="claim_id", how="left")
            merged["gold"] = merged["claim_text"].map(gold)
            m = merged[merged["gold"].isin(VALID_LABELS)]
            if len(m):
                pred_info["accuracy"] = round(float((m["gold"] == m["predicted_label"]).mean()), 3)
                pred_info["evaluated"] = len(m)

    sources = {}
    if corpus is not None and "source" in corpus:
        sources = corpus["source"].replace("", "unknown").value_counts().head(8).to_dict()

    return jsonify({
        "student_id": mnlp_app.get_student_id(),
        "corpus_docs": 0 if corpus is None else len(corpus),
        "claims": 0 if claims is None else len(claims),
        "train": 0 if train is None else len(train),
        "test": 0 if test is None else len(test),
        "claim_labels": label_counts(claims),
        "train_labels": label_counts(train),
        "sources": sources,
        "predictions": pred_info,
    })


@web.post("/api/verify")
def verify():
    body = request.get_json(silent=True) or {}
    claim = str(body.get("claim", "")).strip()
    pipeline = str(body.get("pipeline", "B")).upper()
    if not claim:
        return jsonify({"error": "Please enter a Myanmar claim."}), 400
    if pipeline not in ("A", "B"):
        return jsonify({"error": "pipeline must be 'A' or 'B'."}), 400
    try:
        verifier, titles = _get_verifier()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    k = 3
    hit = verifier.retriever.search([claim], k=k)[0]
    if pipeline == "A":
        result = _verify_traditional(claim, verifier, hit)
        if result is None:
            return jsonify({"error": "Pipeline A needs labelled claims in data/train_claims.csv."}), 400
    else:
        result = verifier.verify([claim], k=k)[0]
        result["pipeline"] = "B"
    result["top_k"] = [{"doc_id": d, "title": titles.get(d, ""), "score": round(s, 4)}
                       for d, s in zip(hit["top_ids"], hit["top_scores"])]
    result["tau"] = round(float(verifier.retriever.tau), 3)
    result["confidence"] = float(result["confidence"])
    return jsonify(result)


def _verify_traditional(claim, verifier, hit):
    """Pipeline A: syllable tokens -> TF-IDF -> Multinomial Naive Bayes (claim text only)."""
    nb = _state.get("nb")
    if nb is None:
        return None
    x = verifier.retriever.vectorizer.transform([claim])
    proba = nb.predict_proba(x)[0]
    i = int(proba.argmax())
    label = str(nb.classes_[i])
    probs = {str(c): round(float(p), 4) for c, p in zip(nb.classes_, proba)}
    ranked = "၊ ".join(f"{c} {p:.0%}" for c, p in sorted(probs.items(), key=lambda kv: -kv[1]))
    return {
        "pipeline": "A",
        "predicted_label": label,
        "retrieved_evidence_id": hit["best_id"],
        "top_k_ids": hit["top_ids"],
        "similarity": round(hit["best_score"], 4),
        "confidence": round(float(proba[i]), 4),
        "reason": "naive_bayes",
        "probabilities": probs,
        "evidence_sentences": [],
        "explanation": (f"Naive Bayes သည် claim ၏ TF-IDF syllable features ကိုသာ အသုံးပြု၍ "
                        f"label တစ်ခုစီ၏ ဖြစ်နိုင်ခြေကို တွက်ချက်သည် ({ranked})။ "
                        f"အထောက်အထား စာတမ်းကို မဖတ်ဘဲ ဖြစ်နိုင်ခြေ အမြင့်ဆုံး {label} ကို ရွေးချယ်သည်။"),
    }


@web.post("/api/run")
def run_pipeline():
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            path = mnlp_app.studentverifier(data_dir=DATA_DIR, output_dir=DATA_DIR)
    except Exception as e:
        return jsonify({"error": str(e), "log": buf.getvalue()}), 500
    return jsonify({"file": os.path.basename(path), "log": buf.getvalue()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNLP web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"MNLP running at http://{args.host}:{args.port}")
    web.run(host=args.host, port=args.port, debug=args.debug)
