# AI-Powered Customer Support FAQ Assistant

SKE309 Minor Project. A chatbot-style web app that answers customer support questions
by matching them to a FAQ knowledge base using NLP (TF-IDF + cosine similarity).

## Features
- Understands free-text questions (not just exact keywords), including common synonyms (parcel/package -> order)
- Replies to small talk (hello, thanks, bye, who are you)
- **Optional AI mode**: answers questions outside the FAQ using the Claude API (see below)
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
4. If similarity >= threshold (0.2) the FAQ answer is returned.
5. Otherwise: small talk is answered; then, if AI mode is on, Claude answers (grounded on the FAQ list);
   otherwise a safe fallback message with related questions is shown.

## AI Mode (answers any question) - optional
By default the assistant is fully offline and only answers topics in `data/faqs.json`.
To let it answer **any** question, set an Anthropic API key before starting the app:
```bash
set ANTHROPIC_API_KEY=your-key-here        # Windows (Linux/Mac: export ANTHROPIC_API_KEY=your-key-here)
python app.py
```
FAQ answers are still used first; AI answers are labelled "AI-generated answer" in the chat.
Optional: `CLAUDE_MODEL` environment variable overrides the model name. If the key is missing or the
API fails, the assistant silently falls back to normal FAQ behaviour. Never commit your key to GitHub.

## Project Structure
```
app.py            Flask server and API routes
faq_engine.py     NLP matching engine (TF-IDF, synonyms, small talk)
llm.py            Optional Claude API answers (AI mode)
data/faqs.json    FAQ knowledge base (edit to add your own)
templates/        Chat UI
tests/            Unit tests
evaluate.py       Retrieval accuracy / threshold evaluation
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

## Run Tests and Evaluation
```bash
pytest -q
python evaluate.py   # accuracy on paraphrased + out-of-scope questions
```

## API Example
```bash
curl -X POST http://127.0.0.1:5000/api/ask -H "Content-Type: application/json" -d "{\"question\": \"how do I get a refund\"}"
```

## Future Scope
Multi-language support, semantic embeddings, admin dashboard to edit FAQs,
database storage, live-agent handoff, analytics on unanswered questions.

## Author
Khushi Purohit – A914145025018
A. Manasa – A914145025026
Samriddhi Singha – A914145025013
Pooja – A914145025001
|MCA 3rd Sem
