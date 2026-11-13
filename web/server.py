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
    POST /api/verify             {"claim": "..."} -> stance, evidence, explanation
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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
# DATA_DIR = os.path.join(ROOT, "data")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import app as mnlp_app  # noqa: E402  (project entry point)
from mnlp_engine.rag_verifier import MyanmarRAGClaimVerifier  # noqa: E402

VALID_LABELS = {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}

web = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")
web.json.ensure_ascii = False

_lock = threading.Lock()
_state = {"verifier": None, "signature": None, "titles": {}}


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
        train = _read("train_claims.csv")
        if train is not None and "label" in train:
            labelled = train[train["label"].isin(VALID_LABELS)]
            if len(labelled) >= 2:
                verifier.tune(labelled["claim_text"], labelled["label"])
        _state.update(verifier=verifier, signature=sig,
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
    claim = (request.get_json(silent=True) or {}).get("claim", "").strip()
    if not claim:
        return jsonify({"error": "Please enter a Myanmar claim."}), 400
    try:
        verifier, titles = _get_verifier()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 400
    k = 3
    result = verifier.verify([claim], k=k)[0]
    hit = verifier.retriever.search([claim], k=k)[0]
    result["top_k"] = [{"doc_id": d, "title": titles.get(d, ""), "score": round(s, 4)}
                       for d, s in zip(hit["top_ids"], hit["top_scores"])]
    result["tau"] = round(float(verifier.retriever.tau), 3)
    result["confidence"] = float(result["confidence"])
    return jsonify(result)


@web.post("/api/run")
def run_pipeline():
    buf = io.StringIO()
    old_verbose = mnlp_app.VERBOSE
    try:
        mnlp_app.VERBOSE = True
        with redirect_stdout(buf):
            path = mnlp_app.studentverifier(data_dir=DATA_DIR, output_dir=DATA_DIR)
    except Exception as e:
        return jsonify({"error": str(e), "log": buf.getvalue()}), 500
    finally:
        mnlp_app.VERBOSE = old_verbose
    return jsonify({"file": os.path.basename(path), "log": buf.getvalue()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNLP web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    print(f"MNLP running at http://{args.host}:{args.port}")
    web.run(host=args.host, port=args.port, debug=args.debug)
