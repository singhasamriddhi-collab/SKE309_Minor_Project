"""Flask web app exposing the FAQ assistant."""
import csv
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from faq_engine import FAQEngine

BASE = Path(__file__).parent
app = Flask(__name__)
engine = FAQEngine(BASE / "data" / "faqs.json")


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/ask")
def ask():
    data = request.get_json(silent=True) or {}
    return jsonify(engine.ask(data.get("question", "")))


@app.get("/api/faqs")
def faqs():
    return jsonify(engine.faqs)


@app.post("/api/feedback")
def feedback():
    data = request.get_json(silent=True) or {}
    path = BASE / "logs" / "feedback.csv"
    path.parent.mkdir(exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "question", "answer_helpful"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),
                    data.get("question", ""), data.get("helpful", "")])
    return jsonify({"status": "saved"})


if __name__ == "__main__":
    app.run(debug=True)
