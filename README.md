# AI-Powered Customer Support FAQ Assistant

SKE309 Minor Project. A chatbot-style web app that answers customer support questions
by matching them to a FAQ knowledge base using NLP (TF-IDF + cosine similarity).

## Features
- Understands free-text questions (not just exact keywords)
- Confidence score for every answer, with a fallback when nothing matches well
- "Related questions" suggestions as clickable chips
- Thumbs up / down feedback saved to `logs/feedback.csv`
- REST API (`/api/ask`, `/api/faqs`, `/api/feedback`) and a simple chat UI
- Automated tests with pytest

## Tech Stack
Python 3, Flask, scikit-learn, HTML/CSS/JavaScript

## How It Works
1. FAQ questions + keywords are cleaned (lower-case, punctuation removed).
2. TF-IDF (unigrams + bigrams, stop-words removed) converts them to vectors.
3. A user query is converted the same way; cosine similarity finds the closest FAQ.
4. If similarity >= threshold (0.2) the answer is returned, otherwise a fallback message.

## Project Structure
```
app.py            Flask server and API routes
faq_engine.py     NLP matching engine
data/faqs.json    FAQ knowledge base (edit to add your own)
templates/        Chat UI
tests/            Unit tests
logs/             Feedback log (generated)
```

## Run Locally
```bash
git clone <your-repo-url>
cd faq-assistant
python -m venv venv
venv\Scripts\activate        # Windows  (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000

## Run Tests
```bash
pytest -q
```

## API Example
```bash
curl -X POST http://127.0.0.1:5000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"how do I get a refund\"}"
```

## Future Scope
Multi-language support, LLM-based answer generation, admin dashboard to edit FAQs,
database storage, live-agent handoff, analytics on unanswered questions.

## Author
Your Name | Enrollment No. | Course / Programme
