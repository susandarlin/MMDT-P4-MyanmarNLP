
import os
import re
import csv
import logging

import pandas as pd
# from flask import Flask, jsonify, send_file
from flask import Flask, jsonify, send_file, render_template, request, send_from_directory

from mnlp_engine.rag_verifier import MyanmarRAGClaimVerifier
from mnlp_engine.tokenizer import tokenize


# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")

OUTPUT_DIR = "/tmp"

VALID_LABELS = {
    "SUPPORTS",
    "REFUTES",
    "NOT ENOUGH INFO"
}

OUTPUT_COLUMNS = [
    "claim_id",
    "predicted_label",
    "retrieved_evidence_id",
    "confidence",
    "explanation"
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Flask Application (Required by Vercel)
# ============================================================

# app = Flask(__name__)
# app = Flask(__name__, template_folder="web")
# app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
app = Flask(__name__, static_folder="web/static")


@app.route("/", methods=["GET"])
def home():
    return send_from_directory(app.static_folder, "index.html")
    # return render_template("index.html")
    # return jsonify({
    #     "status": "success",
    #     "message": "Myanmar NLP Claim Verification API is running!"
    # })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy"
    })


# ============================================================
# Helper Functions
# ============================================================

def _read_csv(data_dir, name):
    """Read CSV using UTF-8 encoding."""

    path = os.path.join(data_dir, name)

    if not os.path.exists(path):
        return None

    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str
    ).fillna("")


def get_student_id():

    # Allow configuration through Vercel environment variables
    student_id = os.environ.get("STUDENT_ID")

    if student_id:
        return re.sub(r"[^a-zA-Z0-9_-]", "_", student_id)

    report_path = os.path.join(BASE_DIR, "report.md")

    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                content = f.read()

            match = re.search(
                r"MMDT ID:\s*(.+)",
                content,
                re.IGNORECASE
            )

            if match:
                value = match.group(1).strip()

                if value and not (
                    value.startswith("[") and value.endswith("]")
                ):
                    return re.sub(
                        r"[^a-zA-Z0-9_-]",
                        "_",
                        value
                    )

        except Exception:
            logger.exception("Could not read student ID from report.md")

    return "student"


# ============================================================
# Myanmar NLP Verification Pipeline
# ============================================================

def studentverifier(
    data_dir=None,
    output_dir=None,
    student_id=None
):
    """
    Main entry point for Myanmar text claim verification.

    Returns:
        str: Path to the generated prediction CSV.
    """

    if data_dir is None:
        data_dir = DATA_DIR

    if output_dir is None:
        output_dir = DATA_DIR

    if student_id is None:
        student_id = get_student_id()

    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"{student_id}_predictions.csv"

    output_path = os.path.join(
        output_dir,
        output_filename
    )

    # --------------------------------------------------------
    # Step 1: Load datasets dynamically
    # --------------------------------------------------------

    logger.info("STEP 1: Loading datasets")

    corpus = _read_csv(data_dir, "corpus.csv")

    train = _read_csv(data_dir, "train_claims.csv")

    test = _read_csv(data_dir, "test_claims.csv")

    if corpus is None or corpus.empty:
        raise FileNotFoundError(
            f"corpus.csv not found or empty in '{data_dir}'"
        )

    if test is None or test.empty:
        raise FileNotFoundError(
            f"test_claims.csv not found or empty in '{data_dir}'"
        )

    required_corpus_columns = {"text"}

    required_test_columns = {"claim_id", "claim_text"}

    if not required_corpus_columns.issubset(corpus.columns):
        raise ValueError(
            "corpus.csv must contain a 'text' column"
        )

    if not required_test_columns.issubset(test.columns):
        raise ValueError(
            "test_claims.csv must contain 'claim_id' "
            "and 'claim_text' columns"
        )

    claims = test["claim_text"].tolist()

    logger.info(
        "Loaded %s documents and %s test claims",
        len(corpus),
        len(test)
    )

    # --------------------------------------------------------
    # Step 2: Myanmar preprocessing and tokenization
    # --------------------------------------------------------

    logger.info("STEP 2: Myanmar text preprocessing")

    corpus_tokens = [
        tokenize(text)
        for text in corpus["text"]
    ]

    claim_tokens = [
        tokenize(text)
        for text in claims
    ]

    logger.info(
        "Tokenized %s documents and %s claims",
        len(corpus_tokens),
        len(claim_tokens)
    )

    # --------------------------------------------------------
    # Step 3: Build TF-IDF vectors
    # --------------------------------------------------------

    logger.info("STEP 3: Building TF-IDF vectors")

    verifier = MyanmarRAGClaimVerifier().fit(corpus)

    vectorizer = verifier.retriever.vectorizer

    doc_vectors = verifier.retriever.D

    claim_vectors = vectorizer.transform(claims)

    logger.info(
        "Document matrix: %s",
        doc_vectors.shape
    )

    logger.info(
        "Claim matrix: %s",
        claim_vectors.shape
    )

    # --------------------------------------------------------
    # Step 4: Evidence retrieval
    # --------------------------------------------------------

    logger.info("STEP 4: Retrieving top-3 evidence")

    top_k = verifier.retriever.search(
        claims,
        k=3
    )

    logger.info(
        "Retrieved evidence for %s claims",
        len(top_k)
    )

    # --------------------------------------------------------
    # Step 5: Naive Bayes baseline and threshold tuning
    # --------------------------------------------------------

    if train is not None and not train.empty and "label" in train:

        labelled = train[
            train["label"].isin(VALID_LABELS)
        ]

        if len(labelled) >= 2:

            logger.info(
                "STEP 5: Tuning thresholds using %s labelled claims",
                len(labelled)
            )

            acc, tau, cov = verifier.tune(
                labelled["claim_text"].tolist(),
                labelled["label"].tolist()
            )

            logger.info(
                "Training accuracy: %.3f, tau: %.2f, coverage: %.2f",
                acc,
                tau,
                cov
            )

    # --------------------------------------------------------
    # Step 6: RAG claim verification
    # --------------------------------------------------------

    logger.info("STEP 6: Running RAG verification")

    results = verifier.verify(
        claims,
        k=3
    )

    if len(results) != len(test):
        raise ValueError(
            "Number of verification results does not match test claims"
        )

    # --------------------------------------------------------
    # Step 7: Generate prediction CSV
    # --------------------------------------------------------

    logger.info("STEP 7: Generating predictions")

    predictions = pd.DataFrame({

        "claim_id": test["claim_id"].values,

        "predicted_label": [
            result["predicted_label"]
            for result in results
        ],

        "retrieved_evidence_id": [
            result["retrieved_evidence_id"]
            for result in results
        ],

        "confidence": [
            round(float(result["confidence"]), 4)
            for result in results
        ],

        "explanation": [
            result["explanation"]
            for result in results
        ]

    }, columns=OUTPUT_COLUMNS)

    # Validate prediction labels
    if not set(predictions["predicted_label"]).issubset(
        VALID_LABELS
    ):
        raise ValueError(
            "Invalid prediction labels generated"
        )

    # Save CSV using UTF-8
    predictions.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_NONNUMERIC
    )

    logger.info(
        "Successfully generated %s predictions",
        len(predictions)
    )

    logger.info(
        "Output file: %s",
        output_path
    )

    return output_path


# ============================================================
# Flask API: Run Myanmar Claim Verification
# ============================================================

@app.route("/verify", methods=["POST"])
def verify():

    try:

        logger.info("Starting Myanmar NLP verification")

        output_file = studentverifier(
            data_dir=DATA_DIR,
            output_dir=OUTPUT_DIR
        )

        return send_file(
            output_file,
            as_attachment=True,
            download_name=os.path.basename(output_file),
            mimetype="text/csv"
        )

    except Exception:

        logger.exception("Claim verification failed")

        return jsonify({
            "status": "error",
            "message": "Claim verification failed. Check deployment logs."
        }), 500


# ============================================================
# Local Development
# ============================================================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )
