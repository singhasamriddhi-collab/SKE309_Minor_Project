"""NLP engine: TF-IDF vectorisation + cosine similarity for FAQ matching."""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

FALLBACK = ("Sorry, I could not find a confident answer to that. "
            "Please contact support at support@shopease.example or try rephrasing.")


def preprocess(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


class FAQEngine:
    def __init__(self, faq_path, threshold: float = 0.2):
        self.faqs = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.threshold = threshold
        docs = [preprocess(f["question"] + " " + " ".join(f.get("keywords", [])))
                for f in self.faqs]
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform(docs)

    def ask(self, query: str, top_k: int = 3) -> dict:
        query = (query or "").strip()
        if not query:
            return {"matched": False, "answer": "Please type a question.",
                    "confidence": 0.0, "suggestions": []}
        vec = self.vectorizer.transform([preprocess(query)])
        scores = cosine_similarity(vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        best = int(ranked[0])
        suggestions = [self.faqs[int(i)]["question"] for i in ranked if scores[i] > 0][:top_k]
        if scores[best] < self.threshold:
            return {"matched": False, "answer": FALLBACK,
                    "confidence": round(float(scores[best]), 3),
                    "suggestions": suggestions}
        faq = self.faqs[best]
        return {"matched": True, "answer": faq["answer"],
                "matched_question": faq["question"], "category": faq["category"],
                "confidence": round(float(scores[best]), 3),
                "suggestions": [s for s in suggestions if s != faq["question"]]}
