"""
Task 5: Evidence-Grounded RAG Reasoning & Explanations (offline, no LLM / API key).

R  - Retrieve : Top-K documents by TF-IDF cosine (retriever.py), then the best
                matching SENTENCES inside the top document (sentence-level cosine).
A  - Augment  : put the claim next to those evidence sentences.
G  - Generate : decide the stance and write a Myanmar justification that quotes
                the evidence and names the exact facts that agree / disagree.

Joint stance verification (claim vs evidence):
  1. best document score < tau                       -> NOT ENOUGH INFO
  2. a number in the claim is not in the evidence
     (and the evidence gives a different number)      -> REFUTES  (numeric contradiction)
  2b. a Latin-script name in the claim is missing from the evidence while
      the evidence names a different one (Microsoft vs Google) -> REFUTES
  3. negation differs (claim negative, evidence not,
     or vice versa: မဟုတ် / မရှိ / မ...ပါ / မ...ဘူး) -> REFUTES  (polarity contradiction)
  4. share of claim syllables covered by evidence
     >= coverage threshold                            -> SUPPORTS (entailment)
     otherwise                                        -> REFUTES  (facts not matching)
tau and the coverage threshold are tuned empirically on train_claims.
"""

import re

import numpy as np

from .retriever import MyanmarRetriever
from .tokenizer import tokenize
from .vectorizer import safe_dot
from .preprocessor import normalize_myanmar

SUPPORTS, REFUTES, NEI = "SUPPORTS", "REFUTES", "NOT ENOUGH INFO"

NUMBER = re.compile(r"[၀-၉0-9]+")


NUMBER_WORDS = ["တစ်", "နှစ်", "သုံး", "လေး", "ငါး", "ခြောက်", "ခုနစ်", "ခုနှစ်",
                "ရှစ်", "ကိုး", "ဆယ့်", "ဆယ်", "ရာ့", "ရာ", "ထောင့်", "ထောင်", "သောင်း", "သိန်း"]
CLASSIFIERS = ["ပါး", "ခု", "ကျမ်း", "ရပ်", "သွယ်", "ယောက်", "ဦး", "ပုဒ်", "ဝဂ်",
               "မျိုး", "ကြိမ်", "ချက်", "ပိုင်း", "နိကာယ်", "ပုံ", "ဆင့်", "ကောင်", "ဆူ", "စု"]
_NW = "|".join(sorted(NUMBER_WORDS, key=len, reverse=True))
_CL = "|".join(sorted(CLASSIFIERS, key=len, reverse=True))
WORD_NUMBER = re.compile(rf"((?:{_NW})+)\s?({_CL})")


def word_numbers(text):
    """{classifier: {number words}}  e.g. 'ပါရာဇိကအာပတ် လေးပါး' -> {'ပါး': {'လေး'}}"""
    out = {}
    for num, cls in WORD_NUMBER.findall(text):
        out.setdefault(cls, set()).add(num)
    return out
NEGATION = re.compile(r"မဟုတ်|မရှိ|မ[^\s။]{1,15}?(?:ပါ|ဘူး)(?=[\s။]|$)")
NOT_NEGATION = re.compile(r"သို့မဟုတ်|မဟုတ်လျှင်")


def is_negated(text):
    return bool(NEGATION.search(NOT_NEGATION.sub(" ", text)))


# Named entities written in Latin script (company / product / person names):
# Google, Microsoft, Android, HTC Dream, Samsung Galaxy, Linux ...
LATIN_NAME = re.compile(r"[A-Za-z][A-Za-z0-9+#.\-]*[A-Za-z0-9+#]|[A-Za-z]")


def latin_names(text):
    return {w.lower() for w in LATIN_NAME.findall(text)}


def split_sentences(text):
    """Myanmar sentences end with ။"""
    return [s.strip() for s in re.split(r"[။\n]", normalize_myanmar(text)) if s.strip()]


class MyanmarRAGClaimVerifier:
    def __init__(self, tau=0.30, coverage_threshold=0.85, n_sentences=2):
        self.retriever = MyanmarRetriever(tau=tau)
        self.coverage_threshold = coverage_threshold
        self.n_sentences = n_sentences
        self.doc_text, self.doc_title = {}, {}

    # ------------------------------------------------------------------ index
    def fit(self, corpus):
        """corpus: DataFrame with doc_id, text (and optional title)."""
        self.retriever.fit(corpus["doc_id"], corpus["text"])
        self.doc_text = dict(zip(corpus["doc_id"], corpus["text"]))
        titles = corpus["title"] if "title" in corpus else corpus["doc_id"]
        self.doc_title = dict(zip(corpus["doc_id"], titles))
        return self

    # --------------------------------------------------- R: evidence sentences
    def _evidence_sentences(self, claim, doc_id):
        sents = split_sentences(self.doc_text[doc_id]) or [self.doc_text[doc_id]]
        vec = self.retriever.vectorizer
        scores = safe_dot(vec.transform([claim]), vec.transform(sents))[0]
        order = np.argsort(-scores)[: self.n_sentences]
        return [sents[i] for i in sorted(order)]          # keep original order

    # ------------------------------------------------ joint stance features
    @staticmethod
    def _compare(claim, evidence):
        c_tok = tokenize(claim)
        e_tok = set(tokenize(evidence))
        coverage = float(np.mean([t in e_tok for t in c_tok])) if c_tok else 0.0
        # (a) digits: ၂၈ vs ၃၈
        c_num, e_num = NUMBER.findall(claim), NUMBER.findall(evidence)
        wrong_numbers = [n for n in c_num if n not in e_num]
        numeric_conflict = bool(wrong_numbers) and bool(e_num)
        # (b) number words with the same classifier: ဆယ်ပါး vs လေးပါး
        c_wn, e_wn = word_numbers(claim), word_numbers(evidence)
        for cls, nums in c_wn.items():
            if cls in e_wn and not (nums & e_wn[cls]):
                numeric_conflict = True
                wrong_numbers += [n + cls for n in nums]
                e_num = e_num + [n + cls for n in e_wn[cls]]
        # (c) entity swap: claim names an entity that is NOT in the evidence,
        #     while the evidence names a different one (Microsoft vs Google)
        c_ent, e_ent = latin_names(claim), latin_names(evidence)
        wrong_entities = sorted(c_ent - e_ent)
        other_entities = sorted(e_ent - c_ent)
        entity_conflict = bool(wrong_entities) and bool(other_entities)
        c_neg, e_neg = is_negated(claim), is_negated(evidence)
        missing_words = [w for w in re.split(r"\s+", re.sub(r"[။၊]", " ", claim))
                         if w and not all(t in e_tok for t in tokenize(w))]
        return {"coverage": coverage, "numeric_conflict": numeric_conflict,
                "entity_conflict": entity_conflict,
                "claim_entities": wrong_entities, "evidence_entities": other_entities,
                "claim_numbers": wrong_numbers, "evidence_numbers": e_num,
                "negation_conflict": c_neg != e_neg, "claim_negated": c_neg,
                "missing_words": missing_words}

    def _decide(self, score, f):
        if score < self.retriever.tau:
            return NEI, "low_similarity"
        if f["numeric_conflict"]:
            return REFUTES, "number"
        if f["entity_conflict"]:
            return REFUTES, "entity"
        if f["negation_conflict"]:
            return REFUTES, "negation"
        if f["coverage"] >= self.coverage_threshold:
            return SUPPORTS, "entailment"
        return REFUTES, "mismatch"

    @staticmethod
    def _confidence(label, reason, score, f, tau):
        if reason == "low_similarity":
            return round(0.5 + 0.5 * (1 - score / tau), 3)
        if reason in ("number", "negation", "entity"):
            return 0.9
        if label == SUPPORTS:
            return round(min(0.99, 0.5 + 0.5 * f["coverage"] * min(1.0, score / 0.5)), 3)
        return round(0.5 + 0.4 * (1 - f["coverage"]), 3)

    # ---------------------------------------------------- G: grounded explanation
    def _explain(self, label, reason, doc_id, score, evidence, f):
        src = f"{doc_id} ({self.doc_title.get(doc_id, doc_id)})"
        quote = "။ ".join(evidence) + "။"
        if reason == "low_similarity":
            return (f"အနီးစပ်ဆုံး စာတမ်း {src} ၏ ဆင်တူမှု {score:.2f} သည် "
                    f"သတ်မှတ်ချက် τ={self.retriever.tau:.2f} ထက် နိမ့်သဖြင့် "
                    f"corpus တွင် ဤ claim အတွက် အထောက်အထား မတွေ့ရပါ။ (NOT ENOUGH INFO)")
        head = f"အထောက်အထား {src} တွင် “{quote}” ဟု ဖော်ပြထားသည်။ "
        if reason == "number":
            return (head + f"claim ပါ ‘{', '.join(f['claim_numbers'])}’ သည် အထောက်အထားပါ "
                    f"‘{', '.join(f['evidence_numbers'])}’ နှင့် မကိုက်ညီသဖြင့် claim ကို ဆန့်ကျင်သည်။ (REFUTES)")
        if reason == "entity":
            ev_names = [w for w in re.findall(LATIN_NAME, quote)
                        if w.lower() in f["evidence_entities"]]
            c_names = [w for w in re.findall(LATIN_NAME, self._last_claim)
                       if w.lower() in f["claim_entities"]]
            return (head + f"claim ပါ ‘{', '.join(dict.fromkeys(c_names))}’ ကို အထောက်အထားတွင် မတွေ့ရဘဲ "
                    f"‘{', '.join(dict.fromkeys(ev_names))}’ ဟု ဖော်ပြထားသဖြင့် claim ကို ဆန့်ကျင်သည်။ (REFUTES)")
        if reason == "negation":
            side = "claim က ငြင်းဆိုထားသော်လည်း အထောက်အထားက အတည်ပြုထားသဖြင့်" if f["claim_negated"] \
                else "claim က အတည်ပြုထားသော်လည်း အထောက်အထားက ငြင်းဆိုထားသဖြင့်"
            return head + side + " claim ကို ဆန့်ကျင်သည်။ (REFUTES)"
        if reason == "entailment":
            return (head + f"claim ၏ အချက်အလက် {f['coverage']:.0%} သည် ဤအထောက်အထားနှင့် "
                    f"ကိုက်ညီသဖြင့် claim ကို ထောက်ခံသည်။ (SUPPORTS)")
        missing = "၊ ".join(f["missing_words"][:4]) or "အချို့အချက်များ"
        return (head + f"claim ပါ ‘{missing}’ ကို အထောက်အထားတွင် မတွေ့ရဘဲ အချက်အလက် မကိုက်ညီသဖြင့် "
                f"claim ကို ဆန့်ကျင်သည်။ (REFUTES)")

    # ------------------------------------------------------------- public API
    def verify(self, claims, k=3):
        """Return one result dict per claim."""
        claims = [str(c) for c in claims]
        results = []
        for claim, hit in zip(claims, self.retriever.search(claims, k=k)):
            doc_id, score = hit["best_id"], hit["best_score"]
            evidence = self._evidence_sentences(claim, doc_id)
            f = self._compare(claim, " ".join(evidence))
            self._last_claim = claim
            label, reason = self._decide(score, f)
            results.append({
                "predicted_label": label,
                "retrieved_evidence_id": doc_id,
                "top_k_ids": hit["top_ids"],
                "similarity": round(score, 4),
                "confidence": self._confidence(label, reason, score, f, self.retriever.tau),
                "reason": reason,
                "evidence_sentences": evidence,
                "explanation": self._explain(label, reason, doc_id, score, evidence, f),
            })
        return results

    def tune(self, claims, labels,
             taus=np.round(np.arange(0.20, 0.36, 0.01), 2),
             coverages=np.round(np.arange(0.60, 1.00, 0.05), 2)):
        """Grid-search tau and coverage threshold on labelled claims (accuracy)."""
        claims, labels = [str(c) for c in claims], np.asarray(labels)
        hits = self.retriever.search(claims, k=1)
        feats = []
        for claim, hit in zip(claims, hits):
            ev = " ".join(self._evidence_sentences(claim, hit["best_id"]))
            feats.append((hit["best_score"], self._compare(claim, ev)))
        best = (-1, self.retriever.tau, self.coverage_threshold)
        for tau in taus:
            for cov in coverages:
                self.retriever.tau, self.coverage_threshold = float(tau), float(cov)
                acc = np.mean([self._decide(s, f)[0] == y for (s, f), y in zip(feats, labels)])
                if acc > best[0]:
                    best = (acc, float(tau), float(cov))
        _, self.retriever.tau, self.coverage_threshold = best
        return best


if __name__ == "__main__":
    import os
    import pandas as pd
    from .evaluation import classification_report, print_report

    data = os.path.join(os.path.dirname(__file__), "..", "data")
    corpus = pd.read_csv(os.path.join(data, "corpus.csv"), encoding="utf-8")
    train = pd.read_csv(os.path.join(data, "train_claims.csv"), encoding="utf-8")
    allc = pd.read_csv(os.path.join(data, "claims.csv"), encoding="utf-8")
    test = allc[~allc["claim_id"].isin(train["claim_id"])]

    rag = MyanmarRAGClaimVerifier().fit(corpus)
    acc, tau, cov = rag.tune(train["claim_text"], train["label"])
    print(f"Tuned on train: tau={tau:.2f}, coverage>={cov:.2f}  (train accuracy {acc:.3f})")

    for name, df in [("TRAIN (49)", train), ("TEST (13, unseen)", test)]:
        res = rag.verify(df["claim_text"])
        print(f"\n===== {name} =====")
        print_report(classification_report(df["label"].values, [r["predicted_label"] for r in res]))

    print("\n===== Example explanations =====")
    for _, row in train.groupby("label").head(1).iterrows():
        r = rag.verify([row["claim_text"]])[0]
        print(f"\nCLAIM : {row['claim_text']}\nTRUE  : {row['label']}   PRED: {r['predicted_label']} "
              f"(conf {r['confidence']})\nWHY   : {r['explanation']}")
