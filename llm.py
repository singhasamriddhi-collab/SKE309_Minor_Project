"""Optional AI mode: answers questions the FAQ cannot, using the Claude API.

Enabled only when the ANTHROPIC_API_KEY environment variable is set.
Uses only the Python standard library (no extra install). Any error -> returns None,
so the assistant falls back to the normal FAQ behaviour instead of crashing.
"""
import json
import os
import urllib.request

API_URL = "https://api.anthropic.com/v1/messages"


def is_enabled() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def ask_claude(question: str, faqs: list, timeout: int = 20):
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    knowledge = "\n".join(f"Q: {f['question']}\nA: {f['answer']}" for f in faqs)
    system = (
        "You are the customer support assistant for ShopEase, an online store. "
        "Treat the FAQ below as the source of truth for store policies. "
        "If the customer's question is covered, answer from it. "
        "If it is a general question, answer helpfully and briefly. "
        "Never invent store policies, prices, deadlines or contact details that are not in the FAQ; "
        "if unsure, say so and suggest contacting support. Keep answers under 80 words.\n\nFAQ:\n" + knowledge
    )
    body = {
        "model": os.environ.get("CLAUDE_MODEL", "claude-sonnet-5-5"),
        "max_tokens": 300,
        "system": system,
        "messages": [{"role": "user", "content": question}],
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", "x-api-key": key,
                 "anthropic-version": "2023-06-01"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
        text = "".join(b.get("text", "") for b in data.get("content", [])
                       if b.get("type") == "text").strip()
        return text or None
    except Exception:
        return None
