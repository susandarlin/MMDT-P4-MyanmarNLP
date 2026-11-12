# Project 3 Report: Myanmar NLP (MNLP) Claim Verification & Evidence Retrieval

> [!IMPORTANT]
> **AUTOGRADER COMPLIANCE INSTRUCTIONS:**
> This report is parsed automatically by the autograder. To ensure you receive full credit for your work:
> 1. **Do not modify** the section headers (`## ...`) or bold field keys (e.g., `**Name:**`, `**Total Corpus Documents Configured:**`, `**Video Presentation Link:**`, etc.).
> 2. **Write your answers directly after** the colon `:` of each field, replacing the placeholder text completely (including the outer brackets `[` and `]`).
> 3. **Maintain the file structure**. Changing headers, bold titles, or deleting lines can cause the autograder to miss your responses and award 0 marks.

---

## Student Information 
- **Name:** Ohnmar Nyunt
- **MMDT ID:** MMDT034

---

## Section 1: Dataset & Corpus Configuration
- **Total Corpus Documents Configured:** 35
- **Total Labeled Claims Configured:** 49
- **Total Test Claims Configured:** 13
- **Total Claims in Dataset:** 62

---

## Section 2: Local Verification & Implemented Pipelines
*Check the algorithms, features, and modules you successfully ran and verified by placing an `x` in the brackets (e.g., `[x]`):*
- [x] Myanmar Text Preprocessor & Syllable/Word Tokenizer (Unicode normalization, syllable segmentation, Myanmar stopword filtering)
- [x] TF-IDF / Feature Vectorizer (vocabulary indexing, term frequencies, inverse document frequencies for Myanmar tokens)
- [x] Naïve Bayes Classifier (Multinomial / Complement NB with Laplace smoothing for Myanmar text)
- [x] Multi-Class Evaluation Metrics (Accuracy, Precision, Recall, Macro-F1, Confusion Matrix)
- [x] Vector Space / Dense Semantic Evidence Retriever (Cosine Similarity & Top-K Ranking on Myanmar corpus)
- [x] Retrieval-Augmented Generation (RAG) Evidence-Grounded Verifier & Explainer
- [x] Predictions CSV Export (<studentid>_predictions.csv)

---

## Section 3: Video Presentation & Project Structure
- **Video Presentation Link:** https://youtu.be/9ZuZZ6tRk_g
- **Project Directory Structure:** The data/ folder holds the inputs: corpus.csv contains the Myanmar reference documents used as evidence, claims.csv contains all labelled claims, train_claims.csv is used for training and tuning, test_claims.csv contains the unseen claims for prediction, and MMDT034_predictions.csv includes the output. The mnlp_engine/ package holds the NLP logic, one module per step. preprocessor.py cleans the text (Zawgyi-to-Unicode conversion and normalization), and tokenizer.py performs syllable segmentation and stopword removal. vectorizer.py builds the TF-IDF features (vocabulary, TF and IDF), and naive_bayes.py provides the Multinomial and Complement Naive Bayes baselines. evaluation.py computes Accuracy, Precision, Recall, Macro-F1 and the Confusion Matrix, and retriever.py finds Top-K evidence using cosine similarity. Finally, rag_verifier.py assigns the final label (SUPPORTS, REFUTES or NOT ENOUGH INFO) and writes a Myanmar explanation.

---

## Section 4: Discussion & Technical Analysis
*Provide your written analysis for each point by replacing the bracket placeholders below:*
- **How did you formulate Myanmar text preprocessing (syllable/word segmentation, Unicode normalization, stopword filtering), TF-IDF feature extraction, and the Naïve Bayes probabilistic classifier?**
    I learned that Myanmar text needs cleaning before a computer can use it. So I first made all text standard Unicode, removed hidden characters and punctuation like ။, and used Sayar Ye Kyaw Thu's sylbreak rule to cut the text into syllables. Then I removed common grammar words like သည်, ၏ and ကို, but kept မ because it means "not". Next, I turned each text into TF-IDF numbers: TF counts how often a syllable appears, and IDF gives more weight to rare syllables. I built the vocabulary from the corpus only and reused it for every claim, so documents and claims can be compared fairly. Finally, I used Naïve Bayes to pick SUPPORTS, REFUTES or NOT ENOUGH INFO. It chooses the label with the highest probability based on how common each label is and how likely each syllable is under it. It only reached 18% accuracy, because SUPPORTS and REFUTES claims use almost the same words, which taught me that word counting shows what a claim is about, but not whether it is true.
- **Compare Traditional NLP (TF-IDF + Naïve Bayes) vs Dense Semantic Retrieval & RAG on Myanmar text in terms of vocabulary coverage, morphology handling, and stance classification (SUPPORTS, REFUTES, NOT ENOUGH INFO).**
    TF-IDF + Naive Bayes only matches exact Myanmar syllables, so it misses synonyms. It also depends on my sylbreak preprocessing to handle unspaced words. It reads only the claim, so it reached just 18% accuracy. My RAG verifier retrieves evidence with TF-IDF and checks it against the claim for numbers, names and negation (မဟုတ်). That raised test accuracy to 92%, though it still sometimes confuses NOT ENOUGH INFO with REFUTES. A dense retriever would understand meaning better, but it can't decide stance on its own, because a claim and its negation look almost the same as embeddings.
- **How does RAG help mitigate LLM hallucinations during claim verification on low-resource/Myanmar text, and how does evidence retrieval quality affect the grounded explanations?**
    I learned that an LLM can make up facts on Myanmar text because it has seen very little Myanmar data. RAG helps because it first finds real evidence from my corpus and then answers only from that evidence. My verifier uses no LLM at all. It quotes the evidence and checks numbers, names and negation, and it says NOT ENOUGH INFO when the similarity score is too low. So my explanations are only as good as my retrieval. It found the right document about 95% of the time, but a document on the same topic can still mislead it. For example, it said REFUTES for the claim that Aung San was a badminton (ကြက်တောင်) champion, but the right answer was NOT ENOUGH INFO because my corpus never mentions badminton.
