"""NLP engine: TF-IDF + cosine similarity FAQ matching, with synonym handling,
small-talk replies and an optional AI (Claude) mode for questions outside the FAQ."""
import json
import re
from pathlib import Path

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from llm import ask_claude

FALLBACK = ("Sorry, I could not find a confident answer to that. "
            "Please contact support at support@shopease.example or try rephrasing.")

# Customers use many words for the same thing; map them to the words used in the FAQs.
SYNONYMS = {
    "parcel": "order", "package": "order", "packet": "order", "consignment": "order",
    "purchase": "order", "purchased": "order", "bought": "order",
    "item": "product", "goods": "product", "article": "product",
    "reimburse": "refund", "reimbursement": "refund", "refunded": "refund",
    "cancelled": "cancel", "canceled": "cancel", "cancelling": "cancel",
    "delivered": "delivery", "deliver": "delivery", "arrive": "delivery", "arrived": "delivery",
    "courier": "delivery", "dispatch": "delivery", "dispatched": "delivery",
    "paid": "payment", "pay": "payment", "paying": "payment",
    "gpay": "upi", "phonepe": "upi", "paytm": "upi",
    "debit": "card", "credit": "card",
    "mail": "email", "passcode": "password", "pwd": "password", "signup": "sign",
    "talk": "contact", "speak": "contact", "representative": "contact",
    "agent": "contact", "helpline": "contact", "mobile": "phone",
}

GREETINGS = {"hi", "hii", "hello", "hey", "hola", "namaste", "greetings"}
THANKS = {"thanks", "thank", "thx", "ty"}
BYES = {"bye", "goodbye"}
PHRASES = {
    "good morning": "greet", "good afternoon": "greet", "good evening": "greet",
    "how are you": "how", "who are you": "who", "what can you do": "who",
    "what are you": "who", "help me": "who", "help": "who",
}
SMALLTALK_REPLIES = {
    "greet": "Hello! I'm the ShopEase assistant. Ask me about orders, shipping, returns, payments or your account.",
    "thanks": "You're welcome! Let me know if there is anything else I can help with.",
    "bye": "Goodbye! Have a great day.",
    "how": "I'm doing well, thank you! How can I help you today?",
    "who": "I'm the ShopEase customer support assistant. I can answer questions about orders, shipping, returns, payments, your account and more.",
}


def preprocess(text: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def stem(word: str) -> str:
    """Very light plural stemmer (orders -> order, refunds -> refund)."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def analyze(text: str) -> list:
    """Tokenise: remove stop-words, map synonyms, stem, add bigrams."""
    tokens = [t for t in preprocess(text).split() if t not in ENGLISH_STOP_WORDS]
    tokens = [stem(SYNONYMS.get(t, t)) for t in tokens]
    return tokens + [a + " " + b for a, b in zip(tokens, tokens[1:])]


class FAQEngine:
    def __init__(self, faq_path, threshold: float = 0.2, llm=ask_claude):
        self.faqs = json.loads(Path(faq_path).read_text(encoding="utf-8"))
        self.threshold = threshold
        self.llm = llm  # callable(question, faqs) -> str | None ; pass None to disable
        docs = [f["question"] + " " + " ".join(f.get("keywords", [])) for f in self.faqs]
        self.vectorizer = TfidfVectorizer(analyzer=analyze)
        self.matrix = self.vectorizer.fit_transform(docs)

    def _smalltalk(self, query: str):
        text = preprocess(query)
        tokens = text.split()
        if not tokens or len(tokens) > 4:
            return None
        kind = None
        if tokens[0] in GREETINGS:
            kind = "greet"
        elif tokens[0] in THANKS:
            kind = "thanks"
        elif tokens[0] in BYES:
            kind = "bye"
        else:
            kind = PHRASES.get(text)
        if kind is None:
            return None
        picks = [f["question"] for f in self.faqs if f["id"] in (1, 7, 10)]
        return {"matched": True, "source": "smalltalk", "category": "Small talk",
                "answer": SMALLTALK_REPLIES[kind], "confidence": 1.0,
                "suggestions": picks if kind in ("greet", "who") else []}

    def ask(self, query: str, top_k: int = 3) -> dict:
        query = (query or "").strip()
        if not query:
            return {"matched": False, "source": "none", "answer": "Please type a question.",
                    "confidence": 0.0, "suggestions": []}
        vec = self.vectorizer.transform([query])
        scores = cosine_similarity(vec, self.matrix).ravel()
        ranked = scores.argsort()[::-1][:top_k]
        best = int(ranked[0])
        suggestions = [self.faqs[int(i)]["question"] for i in ranked if scores[i] > 0][:top_k]
        if scores[best] >= self.threshold:
            faq = self.faqs[best]
            return {"matched": True, "source": "faq", "answer": faq["answer"],
                    "matched_question": faq["question"], "category": faq["category"],
                    "confidence": round(float(scores[best]), 3),
                    "suggestions": [s for s in suggestions if s != faq["question"]]}
        small = self._smalltalk(query)
        if small:
            return small
        if self.llm:  # AI mode: answer anything the FAQ cannot
            text = self.llm(query, self.faqs)
            if text:
                return {"matched": True, "source": "ai", "category": "AI answer",
                        "answer": text, "confidence": round(float(scores[best]), 3),
                        "suggestions": suggestions}
        return {"matched": False, "source": "fallback", "answer": FALLBACK,
                "confidence": round(float(scores[best]), 3), "suggestions": suggestions}
