import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from faq_engine import FAQEngine, preprocess

FAQ = Path(__file__).resolve().parents[1] / "data" / "faqs.json"


@pytest.fixture(scope="module")
def engine():
    return FAQEngine(FAQ, llm=None)  # offline: AI mode disabled


@pytest.mark.parametrize("query,category", [
    ("where is my order", "Orders"),
    ("I want my money back", "Returns"),
    ("forgot my password", "Account"),
    ("do you accept UPI", "Payments"),
    ("how much is delivery charge", "Shipping"),
    ("talk to a human agent", "Support"),
])
def test_known_queries(engine, query, category):
    r = engine.ask(query)
    assert r["matched"] and r["category"] == category and r["source"] == "faq"


def test_synonym_parcel_maps_to_order(engine):
    assert engine.ask("where is my parcel")["matched"]


def test_unrelated_query_falls_back(engine):
    r = engine.ask("what is the capital of france")
    assert not r["matched"] and r["source"] == "fallback"


def test_empty_query(engine):
    assert not engine.ask("   ")["matched"]


@pytest.mark.parametrize("query", ["hello", "Hi!", "thanks", "thank you", "good morning", "bye", "who are you"])
def test_small_talk_gets_a_reply(engine, query):
    r = engine.ask(query)
    assert r["matched"] and r["source"] == "smalltalk"


def test_greeting_word_inside_a_real_question_is_not_small_talk(engine):
    assert engine.ask("translate hello to french")["source"] != "smalltalk"


def test_ai_mode_answers_questions_outside_the_faq():
    e = FAQEngine(FAQ, llm=lambda q, faqs: "Paris is the capital of France.")
    r = e.ask("what is the capital of france")
    assert r["matched"] and r["source"] == "ai" and "Paris" in r["answer"]


def test_faq_answer_takes_priority_over_ai_mode():
    def boom(q, faqs):
        raise AssertionError("AI must not be called when the FAQ matches")
    e = FAQEngine(FAQ, llm=boom)
    assert e.ask("how do I request a refund")["source"] == "faq"


def test_ai_failure_falls_back_safely():
    e = FAQEngine(FAQ, llm=lambda q, faqs: None)
    assert e.ask("what is the capital of france")["source"] == "fallback"


def test_preprocess():
    assert preprocess("  Hello,   WORLD!! ") == "hello world"
