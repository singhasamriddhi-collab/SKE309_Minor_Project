"""Evaluation script: measures retrieval accuracy on paraphrased and out-of-scope questions."""
import json
from pathlib import Path
from faq_engine import FAQEngine

FAQ_PATH = Path(__file__).parent / "data" / "faqs.json"
engine = FAQEngine(FAQ_PATH, llm=None)  # offline: AI mode disabled so results are reproducible

# (query, expected FAQ id)  -- paraphrased questions written by the author
IN_SCOPE = [
 ("where is my parcel", 1), ("how do I track my shipment", 1), ("I want to cancel what I ordered", 2),
 ("stop my order", 2), ("I entered the wrong address", 3), ("how many days will delivery take", 4),
 ("is shipping free", 5), ("do you ship abroad", 6), ("can I send the product back", 7),
 ("I want my money back", 8), ("how long for refund", 8), ("the item I got is broken", 9),
 ("do you accept UPI", 10), ("can I pay cash on delivery", 10), ("money deducted but order failed", 11),
 ("where do I enter a discount code", 12), ("forgot my password", 13), ("how to sign up", 14),
 ("I want to close my account", 15), ("change my email address", 16), ("is this item available", 17),
 ("is there a warranty", 18), ("how do I talk to a human agent", 19), ("what time does support open", 20),
]
OUT_OF_SCOPE = ["what is the capital of france", "tell me a joke", "who won the cricket match",
                "asdf qwerty zxcv", "best pizza recipe", "how is the weather today"]

ok, rows = 0, []
for q, eid in IN_SCOPE:
    r = engine.ask(q)
    got = next((f["id"] for f in engine.faqs if r.get("matched_question") == f["question"]), None)
    hit = got == eid
    ok += hit
    rows.append((q, eid, got, r["confidence"], r["matched"], hit))
fb = 0
orows = []
for q in OUT_OF_SCOPE:
    r = engine.ask(q)
    fb += (not r["matched"])
    orows.append((q, r["matched"], r["confidence"]))

print("IN-SCOPE")
for x in rows: print(x)
print(f"Top-1 accuracy: {ok}/{len(IN_SCOPE)} = {ok/len(IN_SCOPE):.1%}")
print("OUT-OF-SCOPE")
for x in orows: print(x)
print(f"Correct fallback: {fb}/{len(OUT_OF_SCOPE)} = {fb/len(OUT_OF_SCOPE):.1%}")
matched_rows=[x for x in rows if x[4]]
print("answered:",len(matched_rows),"of",len(rows))
for th in (0.1,0.15,0.2,0.25,0.3,0.4):
    e=FAQEngine(Path("data/faqs.json"),threshold=th,llm=None)
    c=w=m=0
    for q,eid in IN_SCOPE:
        r=e.ask(q)
        if not r["matched"]: m+=1; continue
        got=next((f["id"] for f in e.faqs if r["matched_question"]==f["question"]),None)
        c+= got==eid; w+= got!=eid
    f=sum(not e.ask(q)["matched"] for q in OUT_OF_SCOPE)
    print(f"th={th}: correct={c} wrong={w} missed={m} oos_fallback={f}/{len(OUT_OF_SCOPE)}")


# ---- Held-out set: new paraphrases written separately, NOT used to tune synonyms ----
HELD_IN = [
 ("can you tell me the status of my order", 1), ("I need to cancel my purchase", 2),
 ("how do I edit the shipping address on my order", 3), ("when will my order arrive", 4),
 ("do I have to pay for delivery", 5), ("can I get my order shipped to the US", 6),
 ("what is the time limit for returning something", 7), ("how will I get my refund", 8),
 ("my parcel arrived broken", 9), ("can I pay using a credit card", 10),
 ("I was charged but the order did not go through", 11), ("where can I use a promo code", 12),
 ("I cannot log in to my account", 13), ("how do I register on your website", 14),
 ("please delete my profile", 15), ("how can I change my mobile number", 16),
 ("will this product be back in stock", 17), ("how long is the manufacturer warranty", 18),
 ("what is your helpline number", 19), ("until what time is your support team online", 20),
]
HELD_OUT = ["who is the prime minister of india", "write a python program to sort a list",
            "what is 25 times 4", "recommend a good movie", "how tall is mount everest",
            "translate hello to french"]
h_ok = 0
print("HELD-OUT IN-SCOPE")
for q, eid in HELD_IN:
    r = engine.ask(q)
    got = next((f["id"] for f in engine.faqs if r.get("matched_question") == f["question"]), None)
    h_ok += got == eid
    print((q, eid, got, r["confidence"], got == eid))
h_fb = sum(not engine.ask(q)["matched"] for q in HELD_OUT)
print(f"Held-out top-1 accuracy: {h_ok}/{len(HELD_IN)} = {h_ok/len(HELD_IN):.1%}")
print(f"Held-out correct fallback: {h_fb}/{len(HELD_OUT)}")


# ---- Final test set: written after tuning, never used to tune the engine ----
TEST_IN = [
 ("how do I follow my shipment online", 1), ("is it possible to stop an order I just placed", 2),
 ("I typed the wrong pincode, can I fix it", 3), ("how many days does shipping usually take", 4),
 ("what do you charge for shipping", 5), ("do you send products to other countries", 6),
 ("how many days do I have to return an item", 7), ("how long will the money take to reach my account", 8),
 ("I got a defective product", 9), ("is net banking supported", 10),
 ("amount debited but transaction failed", 11), ("how to redeem a voucher", 12),
 ("I forgot my login password", 13), ("how can I open a new account", 14),
 ("I want to deactivate my account", 15), ("how do I edit my profile details", 16),
 ("how do I know whether the item is available", 17), ("does the laptop have a guarantee", 18),
 ("how do I reach customer care", 19), ("what are your office hours", 20),
]
TEST_OUT = ["what is the boiling point of water", "who invented the telephone",
            "give me a recipe for pasta", "how do I learn guitar",
            "what is the population of japan", "tell me about the cricket world cup"]
t_ok = 0
print("FINAL TEST SET")
for q, eid in TEST_IN:
    r = engine.ask(q)
    got = next((f["id"] for f in engine.faqs if r.get("matched_question") == f["question"]), None)
    t_ok += got == eid
    if got != eid: print("  miss:", q, "expected", eid, "got", got, r["confidence"])
t_fb = sum(not engine.ask(q)["matched"] for q in TEST_OUT)
print(f"Test top-1 accuracy: {t_ok}/{len(TEST_IN)} = {t_ok/len(TEST_IN):.1%}")
print(f"Test correct fallback: {t_fb}/{len(TEST_OUT)}")
