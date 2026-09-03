# LangGraph Tutorial (with Google Gemini & LangSmith)

This project is adapted from Tech With Tim's YouTube tutorial: **["LangGraph Tutorial - How to Build Advanced AI Agent Systems"](https://youtu.be/1w5cCXlh7JQ)**, fully customized to use **Google Gemini** (via `langchain-google-genai`) and full **LangSmith observability/tracing**.

---

## Project Structure

```text
.
├── .env                  # API keys and LangSmith configuration
├── pyproject.toml        # Dependencies managed with `uv`
├── simple.py             # Part 1: Basic single-node LangGraph chatbot
├── main.py               # Part 2: Advanced multi-agent router system
├── output.png            # Visual architecture of simple.py graph
├── output1.png           # Visual architecture of main.py multi-agent graph
└── src/
    └── langraph_tutorial_tim/
        └── __init__.py   # CLI entry point
```

---

## Architecture Overview

### 1. `simple.py` (Introductory Graph)
- **State Schema:** `TypedDict` with `messages: Annotated[list, add_messages]`.
- **Flow:** `START` ➔ `chatbot` node (Gemini) ➔ `END`.
- **Graph Diagram:** Auto-exported to `output.png`.

### 2. `main.py` (Advanced Multi-Agent System)
- **State Schema:** `messages`, `message_type`, and `next` destination.
- **Node 1 (`classifier`):** Evaluates user intent into `"emotional"` or `"logical"` using Pydantic structured output (`with_structured_output`).
- **Node 2 (`router`):** Determines the target agent persona.
- **Node 3 (`therapist`):** Compassionate, empathetic response tailored for emotional support.
- **Node 4 (`logical`):** Objective, fact-focused response tailored for technical/logical queries.
- **Flow:** `START` ➔ `classifier` ➔ `router` ➔ (`therapist` OR `logical`) ➔ `END`.
- **Graph Diagram:** Auto-exported to `output1.png`.

---

## How to Run

Run with `uv`:

### Run the Simple Chatbot
```bash
uv run python simple.py
```

### Run the Advanced Multi-Agent Router System
```bash
uv run python main.py
```

---

## LangSmith Observability

LangSmith tracing is fully configured in `.env`:
- `LANGCHAIN_TRACING_V2="true"`
- `LANGCHAIN_ENDPOINT="https://eu.api.smith.langchain.com"` (Configured for European accounts)
- `LANGCHAIN_PROJECT="LangGraph-Tutorial"`
- `LANGCHAIN_API_KEY="..."`

Every prompt, LLM call, token usage, latency, and node transition is visible live in your [LangSmith Dashboard](https://eu.smith.langchain.com).
