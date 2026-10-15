"""Agentic RAG core for India travel: retrieval + tool-using agent loop over Ollama."""
import json, os, re, datetime
import numpy as np, requests

OLLAMA = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
HERE = os.path.dirname(__file__)


def embed(texts):
    r = requests.post(f"{OLLAMA}/api/embed", json={"model": EMBED_MODEL, "input": texts}, timeout=300)
    r.raise_for_status()
    v = np.array(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def chat(messages, tools=None, fmt=None):
    body = {"model": CHAT_MODEL, "messages": messages, "stream": False, "options": {"temperature": 0.2}}
    if tools: body["tools"] = tools
    if fmt: body["format"] = fmt
    r = requests.post(f"{OLLAMA}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]


class KB:
    def __init__(self, path=os.path.join(HERE, "data", "india_kb.json")):
        self.docs = json.load(open(path))
        cache = path + ".npy"
        if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
            self.vecs = np.load(cache)
        else:
            self.vecs = embed([f"{d['title']}: {d['text']}" for d in self.docs])
            np.save(cache, self.vecs)

    def search(self, query, k=3):
        q = embed([query])[0]
        idx = np.argsort(-(self.vecs @ q))[:k]
        return [self.docs[i] for i in idx]


# ---------------- tools ----------------
def budget_estimate(days, style="mid"):
    per_day = {"budget": 1800, "mid": 4500, "luxury": 15000}.get(style, 4500)
    return {"days": days, "style": style, "per_day_inr": per_day, "total_inr": per_day * int(days)}


def season_check(month):
    m = str(month).lower()[:3]
    good = {"oct": "Rajasthan, Golden Triangle, Kerala, Goa (season start), Northeast",
            "nov": "Goa, Rajasthan (Pushkar fair), Kerala, Golden Triangle", "dec": "Goa, Kerala, Rajasthan, Andaman",
            "jan": "Goa, Kerala, Rajasthan, Golden Triangle, Andaman", "feb": "Goa, Rajasthan, Hampi, Kerala",
            "mar": "Himachal, Holi in Mathura/Vrindavan, Andaman", "apr": "Himachal, Sikkim, Andaman",
            "may": "Himachal, Sikkim, Darjeeling", "jun": "Ladakh, Spiti, Kerala Ayurveda",
            "jul": "Ladakh, Spiti, Kerala monsoon", "aug": "Ladakh, Spiti, Onam in Kerala",
            "sep": "Ladakh, Himachal, Durga Puja Kolkata"}
    return {"month": month, "recommended": good.get(m, "unknown month")}


TOOLS_IMPL = {"budget_estimate": budget_estimate, "season_check": season_check}
TOOLS = [
    {"type": "function", "function": {"name": "search_knowledge_base",
        "description": "Search the India travel knowledge base for destinations, visas, transport, safety, food.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "budget_estimate",
        "description": "Estimate trip cost in INR for a number of days and travel style.",
        "parameters": {"type": "object", "properties": {"days": {"type": "integer"},
            "style": {"type": "string", "enum": ["budget", "mid", "luxury"]}}, "required": ["days"]}}},
    {"type": "function", "function": {"name": "season_check",
        "description": "Recommend Indian destinations that are in season for a given month.",
        "parameters": {"type": "object", "properties": {"month": {"type": "string"}}, "required": ["month"]}}},
]

SYSTEM = (f"You are an expert India travel assistant. Today is {datetime.date.today()}. "
          "Use the tools to ground your answers: ALWAYS call search_knowledge_base for factual questions, "
          "season_check for 'when/which month' questions and budget_estimate for cost questions. "
          "You may call tools several times. Answer concisely, cite sources as [title], and say if the knowledge base lacks the answer.")


class Agent:
    def __init__(self, kb=None, max_steps=4):
        self.kb, self.max_steps = kb or KB(), max_steps

    def run(self, question, history=None):
        msgs = [{"role": "system", "content": SYSTEM}] + (history or []) + [{"role": "user", "content": question}]
        trace = []
        for _ in range(self.max_steps):
            msg = chat(msgs, tools=TOOLS)
            msgs.append(msg)
            calls = msg.get("tool_calls") or []
            if not calls:
                return msg.get("content", ""), trace
            for c in calls:
                name, args = c["function"]["name"], c["function"].get("arguments", {}) or {}
                try:
                    if name == "search_knowledge_base":
                        docs = self.kb.search(args.get("query", question))
                        out = "\n\n".join(f"[{d['title']}] {d['text']}" for d in docs)
                    else:
                        out = json.dumps(TOOLS_IMPL[name](**args))
                except Exception as e:
                    out = f"tool error: {e}"
                trace.append({"tool": name, "args": args, "result": out[:300]})
                msgs.append({"role": "tool", "content": out, "tool_name": name})
        msgs.append({"role": "user", "content": "Give your final answer now using the information gathered."})
        return chat(msgs).get("content", ""), trace


if __name__ == "__main__":
    import sys
    a = Agent()
    q = " ".join(sys.argv[1:]) or "When is the best time to visit Rajasthan and what's a 7 day budget?"
    ans, tr = a.run(q)
    for t in tr: print("TOOL:", t["tool"], t["args"])
    print("\nANSWER:\n", ans)
