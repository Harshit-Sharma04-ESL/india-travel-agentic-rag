"""Generate synthetic user questions (with reference answers) about India travel from the KB, using Ollama."""
import json, os, random, sys
from rag import chat, HERE

PERSONAS = ["first-time foreign backpacker", "Indian family with kids", "honeymoon couple", "solo woman traveller",
            "senior citizens", "budget student", "luxury traveller", "adventure/trekking enthusiast", "foodie"]
TYPES = ["factual", "planning/itinerary", "budget", "best time/season", "logistics/transport", "safety/visa", "comparison"]

PROMPT = """You create realistic questions travellers ask a travel assistant about India.
Context:
{ctx}

Write {n} diverse questions a "{persona}" might ask, answerable from the context. Mix question types: {types}.
Return JSON: {{"items":[{{"question":"...","type":"...","answer":"short answer grounded in the context"}}]}}"""


def main(per_doc=4, out=os.path.join(HERE, "data", "synthetic_questions.jsonl")):
    docs = json.load(open(os.path.join(HERE, "data", "india_kb.json")))
    rows = []
    for d in docs:
        persona = random.choice(PERSONAS)
        msg = chat([{"role": "user", "content": PROMPT.format(ctx=d["text"], n=per_doc, persona=persona,
                                                            types=", ".join(random.sample(TYPES, 3)))}], fmt="json")
        try:
            items = json.loads(msg["content"]).get("items", [])
        except json.JSONDecodeError:
            items = []
        for it in items:
            if isinstance(it, dict) and it.get("question"):
                rows.append({"question": it["question"], "type": it.get("type"), "reference_answer": it.get("answer"),
                             "persona": persona, "source_doc": d["id"]})
        print(f"{d['id']}: {len(items)} questions", flush=True)
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
