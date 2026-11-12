## AI Policy: Use & Declaration
- **Limited AI Use:** AI coding assistants (e.g., GitHub Copilot) may be used for syntax suggestions, debugging, coding support, and learning assistance.
- **Prohibited:** Do not paste project prompts, guidelines, problem statements, or substantial portions of project materials into AI systems for code generation, completion, or debugging.
- **Responsibility:** Submitted work must reflect your own understanding, implementation, and problem-solving effort.

### AI Use Declaration

1. Complete the [PETRA AI Student Declaration](https://www.petraai.org/student) based on your actual AI use for this project.
2. Generate and save the declaration image as **`AI_declaration.png`**.
3. Include **`AI_declaration.png`** in your project repository root.

> **Required:** Your declaration must accurately reflect how you used AI tools for this project.

---

## Project Overview: Myanmar NLP (MNLP) Claim Verification & Evidence Retrieval

Automated Fact Verification in low-resource and non-segmented languages like Myanmar (Burmese) is an essential task at the intersection of Information Retrieval (IR) and Natural Language Processing (NLP). Given a natural-language Myanmar text claim and a collection of reference documents (corpus), the objective is to:
1. **Retrieve relevant evidence passages** from your custom Myanmar corpus that contain factual context pertaining to the claim.
2. **Determine the stance / verdict** of the claim based on the retrieved evidence:
   - **`SUPPORTS`**: The retrieved evidence verifies that the claim is factually true.
   - **`REFUTES`**: The retrieved evidence contradicts the claim.
   - **`NOT ENOUGH INFO`**: The corpus lacks sufficient factual information to verify or refute the claim.
3. **Generate an evidence-grounded explanation** in Myanmar text clarifying *why* the evidence leads to the verdict.

### 🌟 Two Major Project Requirements

1. **Build Your Own Myanmar Reference Corpus (`data/corpus.csv`):**
   - You must construct a domain knowledge base containing at least **20 reference documents** (`doc_id`, `title`, `text`, `source`) written in standard Myanmar Unicode (UTF-8).
   - Your corpus should cover diverse knowledge domains (e.g., Myanmar history, culture, geography, science, global milestones, technology, famous personalities).
2. **Curate a Dataset with at least 50 Myanmar Claims (`data/claims.csv`):**
   - You must find, collect, or construct at least **50 Myanmar text claims** annotated with ground truth stance labels (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`) and target `evidence_id`s.
   - Partition your dataset into **`train_claims.csv`** (at least 40 claims) and **`test_claims.csv`** (at least 10 claims).
   - **Note on Git:** `.gitignore` is configured to ignore `data/*.csv` so your datasets remain local to your development environment.

In this project, you will build and evaluate two distinct architectures:
- **Pipeline A: Traditional Statistical NLP Classifier for Myanmar Text:**
  $$\text{Myanmar Text} \longrightarrow \text{Syllable/Word Segmentation} \longrightarrow \text{TF-IDF Matrix} \longrightarrow \text{Naïve Bayes} \longrightarrow \text{Predictions} \longrightarrow \text{Evaluation}$$
- **Pipeline B: Semantic Retrieval & RAG Verification:**
  $$\text{Claim} \longrightarrow \text{Vector Space / Embeddings} \longrightarrow \text{Cosine Similarity Search} \longrightarrow \text{Top-}K\text{ Evidence} \longrightarrow \text{RAG Reasoning / Stance Verdict}$$

---

### Learning Objectives

1. **Myanmar Text Preprocessing & Segmentation:** Address Myanmar script characteristics (no whitespace between words, consonant conjuncts, tone marks, Unicode normalization) using regex syllable breaking, pyidaungsu, or n-gram tokenization.
2. **TF-IDF Feature Representation:** Build vocabulary dictionaries for Myanmar syllables/words and compute Term Frequency ($TF$) and Inverse Document Frequency ($IDF$) feature matrices.
3. **Naïve Bayes Probabilistic Classification:** Implement a Multinomial / Complement Naïve Bayes classifier with Laplace smoothing ($\alpha=1$) and log-likelihood calculations.
4. **Classification Performance Metrics:** Compute multi-class Accuracy, Precision, Recall, Macro-F1 score, and Confusion Matrices across `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO`.
5. **Vector Space & Semantic Evidence Retrieval:** Implement cosine similarity search and Top-$K$ retrieval over Myanmar document passages, evaluating retrieval quality via Precision@K, Recall@K, and MRR.
6. **Evidence-Grounded RAG Reasoning:** Formulate a RAG verification pipeline that evaluates claim veracity against retrieved passages and outputs concise explanations.
7. **Dynamic Batch Export:** Dynamically ingest dataset updates and export structured prediction records to `<studentid>_predictions.csv`.

---

## Your Job

1. Build your custom reference corpus in `data/corpus.csv` (minimum 20 documents in Myanmar Unicode).
2. Curate and construct your dataset in `data/claims.csv` with at least 50 claims (split into `data/train_claims.csv` with $\ge 40$ claims and `data/test_claims.csv` with $\ge 10$ claims).
3. Design and implement your text preprocessing, syllable/word tokenizer, TF-IDF vectorizer, Naïve Bayes classifier, and semantic evidence retrieval algorithms.
4. Provide the main entry point in **`app.py`** via the function **`studentverifier(data_dir="data", output_dir="data", student_id=None)`**.
5. When `studentverifier()` is executed, it must dynamically process the datasets from `data_dir`, execute the verification and retrieval pipeline, and generate the prediction file named **`<studentid>_predictions.csv`** in the specified `output_dir` (default: `data/`).
6. Ensure that your verifier is dynamic: if datasets in `data/` are updated or changed, your system should still successfully load the new data and generate valid predictions.
7. **Project Structure Flexibility:** You are free to design and structure your own internal helper modules, classes, and folder structure however you like. In testing, evaluation focuses on executing `studentverifier()` from `app.py` and verifying that the predictions CSV is created.
8. Explain how your project directory, algorithms, and retrieval pipelines are built in your **`report.md`** and your **video presentation**.

---

### Project Structure Overview

```text
├── app.py                      # REQUIRED: Contains studentverifier() entry point
├── data/                       # Local datasets & output folder (Ignored by Git)
│   ├── .gitkeep                # Keeps data/ tracked in Git
│   ├── corpus.csv              # Custom Myanmar reference knowledge base (>= 20 docs)
│   ├── claims.csv              # Full collection of curated Myanmar claims (>= 50 claims)
│   ├── train_claims.csv        # Training claims with ground truth labels (>= 40 claims)
│   ├── test_claims.csv         # Evaluation claims for prediction (>= 10 claims)
│   └── <studentid>_predictions.csv # Output predictions generated by studentverifier()
├── report.md                   # REQUIRED: Complete the provided report template
├── AI_declaration.png          # REQUIRED: AI use declaration generated using PETRA
├── requirements.txt            # Python dependencies
└── [Your Custom Modules]       # Preprocessors, tokenizers, vectorizers, classifiers
```

---

### 📋 Task 1: Build Custom Corpus & Curate Myanmar Claims (`data/*.csv`)
- **Corpus (`corpus.csv`):** Columns: `doc_id`, `title`, `text`, `source`. Must contain **at least 20 reference documents** in Myanmar Unicode covering diverse factual topics.
- **Claims (`claims.csv` / `train_claims.csv`):** Columns: `claim_id`, `claim_text`, `evidence_id`, `label`. Annotated with target evidence passage (`doc_id` or `NONE`) and ground-truth verdict (`SUPPORTS`, `REFUTES`, `NOT ENOUGH INFO`). Must total **at least 50 claims** across train and test sets.
- **Test Claims (`test_claims.csv`):** Columns: `claim_id`, `claim_text`. Unlabeled claims for pipeline evaluation and predictions export (at least 10 claims).

---

### 📋 Task 2: Myanmar Text Preprocessing & Syllable/Word Segmentation
Implement robust text processing for Myanmar script:
1. **Myanmar Text Normalization & Cleaning:**
   - Unicode standard normalization (convert any Zawgyi artifacts if necessary, strip non-text characters).
   - Regex-based punctuation and special symbol stripping (`။`, `၊`, etc.).
2. **Syllable Breaking & Tokenization:**
   - Because standard Myanmar writing does not separate words with spaces, implement syllable segmentation (e.g. using regex syllable patterns such as `(?<![်္])(?=[က-အ|ဥ-ဪ])` or syllable tokenization) or word segmentation libraries (e.g. `pyidaungsu`).
   - Stopword filtering (removing high-frequency grammatical particles like `သည်`, `၏`, `ကို`, `တွင်`, `နှင့်`, `ဖြစ်သည်`).
3. **TF-IDF Vectorization:**
   - Build a vocabulary dictionary mapping unique Myanmar tokens/syllables to indices.
   - Compute Term Frequency ($TF$) and Inverse Document Frequency ($IDF$).
   - Compute L2-normalized TF-IDF feature vectors for both claims and documents.

---

### 📋 Task 3: Traditional NLP Classifier (Naïve Bayes Baseline)
Implement a probabilistic Naïve Bayes classification pipeline:
1. **Multinomial Naïve Bayes Model:**
   - Calculate class prior probabilities:
     $$P(C_k) = \frac{N_{C_k} + \alpha}{N + \alpha \cdot |C|}$$
   - Calculate word/syllable feature likelihoods with Laplace smoothing ($\alpha=1$):
     $$P(w_i | C_k) = \frac{\sum_{d \in C_k} \text{TF-IDF}(w_i, d) + \alpha}{\sum_{w} \left( \sum_{d \in C_k} \text{TF-IDF}(w, d) \right) + \alpha \cdot |V|}$$
   - Apply log-space inference to prevent numerical underflow:
     $$\hat{y} = \arg\max_{C_k} \left[ \log P(C_k) + \sum_{i=1}^{M} x_i \cdot \log P(w_i | C_k) \right]$$
2. **Evaluation Metrics:**
   - Compute multi-class Accuracy, Precision, Recall, and Macro-F1 score across `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO`.
   - Generate confusion matrix to diagnose classification errors.

---

### 📋 Task 4: Semantic Retrieval & Evidence Extraction
Implement vector search to retrieve relevant evidence from `corpus.csv`:
1. **Vector Space / Embedding Representation:**
   - Represent each document in `corpus.csv` and each query claim as normalized dense or TF-IDF sparse vectors.
2. **Similarity Scoring & Top-K Ranking:**
   - Compute Cosine Similarity between claim vector $\mathbf{q}$ and passage vector $\mathbf{d}_j$:
     $$\text{CosineSim}(\mathbf{q}, \mathbf{d}_j) = \frac{\mathbf{q} \cdot \mathbf{d}_j}{\|\mathbf{q}\| \|\mathbf{d}_j\|}$$
   - Rank documents by similarity score and extract Top-$K$ candidates (e.g., $K=1, 3, 5$).
   - If maximum similarity is below an empirical threshold $\tau$, flag the claim as lacking evidence (`NOT ENOUGH INFO`).
3. **Retrieval Evaluation:**
   - Compute Precision@K, Recall@K, and Mean Reciprocal Rank (MRR) using ground truth `evidence_id`.

---

### 📋 Task 5: Evidence-Grounded RAG Reasoning & Explanations
Combine evidence retrieval with stance verification:
1. **Claim–Evidence Joint Stance Verification:**
   - Evaluate whether the retrieved evidence passage semantically entails (`SUPPORTS`), contradicts (`REFUTES`), or fails to cover (`NOT ENOUGH INFO`) the Myanmar claim.
2. **Grounded Explanation Generation (RAG Synthesis):**
   - Synthesize a clear justification referencing specific facts from the retrieved Myanmar evidence passage that justify the verdict.

---

### 📋 Task 6: Main Entry Point & Predictions Export (`app.py`)
Implement the `studentverifier()` function in `app.py`:
- Accepts parameters: `data_dir="data"`, `output_dir="data"`, `student_id=None`.
- Loads the datasets dynamically from `data_dir` using UTF-8 encoding.
- Executes your NLP classification and retrieval pipeline.
- Saves the predictions to `<studentid>_predictions.csv` in `output_dir`.
- Returns the path to the generated CSV file.

Required CSV format columns:
```csv
claim_id,predicted_label,retrieved_evidence_id,confidence,explanation
```

Example output row:
```csv
CLM_TEST_001,SUPPORTS,DOC_001,0.95,"ရယူထားသော အထောက်အထားစာပိုဒ်အရ ပုဂံဒေသကို ၂၀၁၉ ခုနှစ်တွင် UNESCO ကမ္ဘာ့အမွေအနှစ်နေရာအဖြစ် သတ်မှတ်ခဲ့ကြောင်း အတည်ပြုပါသည်။"
```

---

### 📋 Task 7: Complete the Report (`report.md`)
Complete all required sections in [report.md](report.md):
- **Student Information:** Full Name and MMDT ID.
- **Dataset Configuration:** Counts of configured corpus documents ($\ge 20$), labeled claims ($\ge 40$), and test claims ($\ge 10$), totaling $\ge 50$ claims.
- **Local Verification Checklist:** Mark verified algorithms and features with `[x]`.
- **Project Structure & Video Presentation:** Provide your video link and explain your directory structure.
- **Discussion & Analysis:** Detailed answers covering Myanmar text preprocessing formulation, Traditional NLP vs RAG on Myanmar text, and hallucination mitigation.

> [!WARNING]
> Do not modify the section headers, bold titles, or overall formatting of [report.md](report.md). Changes to the structure may cause the autograder tests to fail.
> Replace all bracketed placeholder text with your answers. Do not leave any placeholders in the final report.

---

### 📋 Task 8: Record a Video Presentation

Record a **5–7 minute video presentation** explaining your project implementation:
- **Project Directory Structure:** Explain how your codebase, modules, and files are structured and how they interact.
- **Myanmar Text Preprocessing & Tokenization:** Explain your syllable breaking, stopword filtering, and TF-IDF matrix construction for Burmese script.
- **Naïve Bayes & Probability Estimation:** How do prior probabilities and Laplace smoothing compute class posteriors?
- **Evidence Retrieval & RAG Synthesis:** How does vector similarity search retrieve relevant Myanmar evidence, and how is the final grounded explanation generated?
- **Demonstration:** Show `studentverifier()` executing and generating `<studentid>_predictions.csv`.

#### Video Requirements
- You may use tools such as Zoom, Loom, or OBS Studio to record your presentation.
- Your face must be visible during the presentation. If you are not comfortable with this requirement, please discuss it with the instructor in advance.
- The recorded video must be in `.mp4` format and must not exceed **7 minutes**.
- Upload the video to a sharing platform such as YouTube, Google Drive, or Dropbox.
- Ensure the video link is publicly accessible to the instructor.
- Add the video link to Section 3 of `report.md`.

---

## Marking Scheme (70 marks)

### 1. Implementation & Predictions Generation (40 marks)
- **Required Files Check (10 marks):** `app.py`, `report.md`, `AI_declaration.png`, and `data` directory present.
- **Verifier & Predictions CSV Generation (30 marks):** `studentverifier()` in `app.py` runs successfully on the dataset, generates `<studentid>_predictions.csv` in `data/`, and outputs a valid non-empty predictions file.

### 2. Completion of report.md (30 marks)
- **Student Information (5 marks):** Full Name and MMDT ID filled in.
- **Dataset Configuration (5 marks):** Document, labeled claim, test claim, and total claim counts documented ($\ge 50$ total claims, $\ge 20$ corpus documents).
- **Verification Checklist (5 marks):** Checklist showing verified preprocessing, Naïve Bayes, retrieval, RAG, and CSV export.
- **Presentation & Project Structure (5 marks):** Provide valid video presentation link and project directory structure overview.
- **Discussion & Technical Formulation (10 marks):** Comprehensive analysis answering the 3 Myanmar NLP discussion questions.
