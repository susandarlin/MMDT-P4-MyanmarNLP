## Project Overview: Myanmar NLP (MNLP) Claim Verification & Evidence Retrieval

In this assignment, you will model and implement an automated Natural Language Processing (NLP) system for **Myanmar Text Claim Verification and Evidence Retrieval** (Fact-Checking in Myanmar script). Given a natural-language Myanmar claim and a collection of reference documents (corpus), your system will retrieve relevant evidence passages and classify the claim stance into one of three categories:
- **`SUPPORTS`**: The retrieved evidence contains facts confirming the claim is true.
- **`REFUTES`**: The retrieved evidence contains facts contradicting the claim.
- **`NOT ENOUGH INFO`**: The corpus lacks sufficient factual evidence to verify or refute the claim.

### 🌟 Two Major Project Requirements

1. **Build Your Own Myanmar Reference Corpus (`data/corpus.csv`):**
   - You must construct a domain knowledge base containing at least **20 reference documents** (`doc_id`, `title`, `text`, `source`) written in standard Myanmar Unicode (UTF-8).
2. **Curate a Dataset with at least 50 Myanmar Claims (`data/claims.csv`):**
   - You must collect or curate at least **50 Myanmar claims** annotated with stance labels (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) and evidence document IDs.
   - Partition into **`train_claims.csv`** (at least 40 claims) and **`test_claims.csv`** (at least 10 claims).
   - **Note:** `.gitignore` is configured to ignore `data/*.csv` so your datasets remain local.

You will implement and evaluate two complementary NLP paradigms:
1. **Traditional Statistical NLP Classifier for Myanmar Text:** Myanmar Text Preprocessing $\rightarrow$ Syllable/Word Segmentation $\rightarrow$ TF-IDF Feature Extraction $\rightarrow$ Naïve Bayes Probabilistic Classifier $\rightarrow$ Multi-class Evaluation Metrics.
2. **Semantic Retrieval & RAG Extension:** Document Vector Space Representation $\rightarrow$ Cosine Similarity / Dense Vector Retrieval $\rightarrow$ Top-K Evidence Extraction $\rightarrow$ Evidence-Grounded Reasoning & Explanation Synthesis.

You will implement the main entry point `studentverifier()` inside `app.py`. You have full freedom in how you organize your internal classes, modules, and folder structure.

### Learning Objectives

1. **Myanmar Text Preprocessing & Tokenization:** Build modular text cleaning pipelines (Unicode normalization, regex syllable segmentation / tokenization, stopword removal) and compute TF-IDF sparse feature matrices.
2. **Naïve Bayes Probabilistic Classification:** Implement a Multinomial / Complement Naïve Bayes classifier with Laplace smoothing ($\alpha=1$) and log-likelihood calculations to predict claim stances.
3. **Comprehensive Model Evaluation:** Compute and analyze classification metrics including Accuracy, Precision, Recall, Macro/Micro F1-Score, and Confusion Matrices.
4. **Vector Space & Semantic Retrieval:** Implement cosine similarity search and Top-K retrieval over Myanmar document passages to isolate pertinent evidence, evaluating retrieval quality via Precision@K, Recall@K, and MRR.
5. **Evidence-Grounded RAG Reasoning:** Design a RAG verification pipeline that combines retrieved evidence with claim context to output a verifiable stance and generate a concise, grounded explanation in Myanmar context.
6. **Dynamic Batch Export:** Dynamically ingest dataset updates and export structured prediction records to `<studentid>_predictions.csv`.

---

## Submission Guidelines

To submit your project, please ensure you complete all required tasks and follow these instructions carefully.

### 1. Preparation Checklist
Before pushing your final submission, verify that:
*   [ ] You have constructed your custom Myanmar corpus (`corpus.csv` with $\ge 20$ documents) and curated $\ge 50$ Myanmar claims in `data/`.
*   [ ] You have implemented your verification pipeline in or called from `studentverifier()` in `app.py`.
*   [ ] Running `studentverifier()` dynamically reads from `data/` and generates `<studentid>_predictions.csv` in `data/`.
*   [ ] Your code works dynamically when datasets in `data/` are modified or expanded.
*   [ ] You have completed the AI Student Declaration on [PETRA AI](https://www.petraai.org/student) and saved it as `AI_declaration.png` in the repository root.
*   [ ] You have completed all sections in `report.md` (with your Name, MMDT ID, video presentation link, directory structure explanation, and technical discussion). Do not alter the structure of `report.md`.
*   [ ] You have recorded a 5–7 minute video presentation (.mp4, face visible) explaining your project directory structure, algorithmic formulation, and demonstration of `studentverifier()`, and uploaded it to YouTube, Google Drive, or Dropbox (with public view access).

### 2. How to Submit
Once everything is ready, commit and push your code to your GitHub repository's `main` or `master` branch:
```bash
git add .
git commit -m "Submit Project 4: Myanmar NLP Claim Verification and Evidence Retrieval"
git push origin main
```
The automated autograding system will automatically run tests upon pushing to the `main` or `master` branch.

---

## Checking Grades

You can check your project grades both locally and online via GitHub Actions.

### 1. Locally (Before Submitting)
You can run the autograding tests on your local machine:
1. Make sure you have installed the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. Run `pytest` inside the project root:
   ```bash
   pytest tests/
   ```
3. A scorecard summarizing your earned marks will be outputted directly in the terminal under the section **"AUTOGRADING SCORECARD"**.

### 2. On GitHub (After Submitting)
Every push to the repository triggers the online autograder.
1. Go to the **Actions** tab of your GitHub repository.
2. Select the latest run (e.g., "Autograder Check" or your commit message).
3. Under the run details, scroll to the bottom or open the job logs to view the **Autograding Scorecard** step summary, which details your score out of 70 marks.

### 3. Grading Details & Passing Requirements
*   **Total Score:** 70 marks (40 marks for implementation and predictions CSV generation, 30 marks for report/video presentation).
*   **Passing Score:** **70 marks**.
*   **Deadline:** **October 4, 2026, at 23:59:59**.
*   **Late Submission Policy:** A penalty of **-10 marks per day** will be applied for late submissions.
*   **Commitment Fee:** Any late, missing, or failed project (score < 70%) will carry a **30,000 MMK commitment fee**, calculated and collected at the end of the 10-week challenge.
# MMDT-P4-MyanmarNLP
