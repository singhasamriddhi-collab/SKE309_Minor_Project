import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from faq_engine import FAQEngine, preprocess

@pytest.fixture(scope="module")
def engine():
    return FAQEngine(Path(__file__).resolve().parents[1] / "data" / "faqs.json")

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
    assert r["matched"] and r["category"] == category

def test_unrelated_query_falls_back(engine):
    assert not engine.ask("what is the capital of france")["matched"]

def test_empty_query(engine):
    assert not engine.ask("   ")["matched"]

def test_preprocess():
    assert preprocess("  Hello,   WORLD!! ") == "hello world"
