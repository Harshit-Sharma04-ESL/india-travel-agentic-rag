# India Travel Agentic RAG

A local, tool-using RAG assistant for travel in India, running on [Ollama](https://ollama.com).

The agent decides which tools to call (up to 4 steps) before answering:

- `search_knowledge_base`: embedding search over `data/india_kb.json` (top 3 entries)
- `budget_estimate`: trip cost in INR for budget / mid / luxury travel
- `season_check`: destinations in season for a given month

## Setup

```bash
# Ollama in Docker
docker run -d --name ollama -p 11434:11434 -v ollama:/root/.ollama ollama/ollama
docker exec ollama ollama pull llama3.2:3b
docker exec ollama ollama pull nomic-embed-text

pip install -r requirements.txt
```

## Usage

```bash
python3 rag.py "Do I need a permit for Ladakh?"   # CLI
streamlit run app.py                               # web UI on :8501
python3 synth_data.py 4                            # regenerate synthetic questions (N per KB entry)
```

Configure with `OLLAMA_URL`, `CHAT_MODEL` and `EMBED_MODEL` environment variables.

## Files

| File | Purpose |
|---|---|
| `rag.py` | Retrieval, tools and agent loop |
| `app.py` | Streamlit chat UI with agent trace |
| `synth_data.py` | Generates synthetic traveller questions with reference answers |
| `data/india_kb.json` | Knowledge base (12 entries; hand-written, so prices and rules are approximate) |
| `data/synthetic_questions.jsonl` | 48 generated questions across traveller personas |
