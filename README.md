# Deep Research Agent (Pydantic AI + Gradio)

A multi-step research agent built with [Pydantic AI](https://ai.pydantic.dev/) and a [Gradio](https://www.gradio.app/) frontend. It accepts a **stock ticker** (e.g. `NVDA`) or a **free-text question**, runs layered **DuckDuckGo** searches, and produces a structured markdown report with citations.

## How it works

1. **Intent & entity detection** — Classifies ticker vs general query; resolves company name and context (e.g. NVDA → NVIDIA).
2. **Initial discovery** — One DuckDuckGo search on the optimized query.
3. **Research angles** — LLM proposes 3–4 non-overlapping keywords from top snippets (e.g. SWOT, competition, quarterly results).
4. **Parallel deep dives** — Separate DuckDuckGo search per angle; fetches page content; extracts facts with source tracking.
5. **Synthesis** — Single markdown report: executive summary, sections, evidence bullets with URLs, risks, and “what to watch next”.

## Prerequisites

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/api-keys)

## Setup

### 1. Open a terminal in this folder

```powershell
cd C:\Users\pavan\OneDrive\Desktop\pydantic
```

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

### 4. Add your API key

Edit `.env`:

```env
OPENAI_API_KEY=sk-your-real-key-here
```

## Run

```powershell
python app.py
```

Open the URL Gradio prints (usually `http://127.0.0.1:7860`).

**Try:** `NVDA`, `AAPL`, or a question like *What are the main risks of investing in renewable energy?*

## Project layout

| File | Purpose |
|------|---------|
| `research.py` | Pipeline orchestration and Pydantic AI sub-agents |
| `search.py` | DuckDuckGo search and page fetching |
| `models.py` | Structured output schemas |
| `agent.py` | Public entry point (`run_research`) |
| `app.py` | Gradio chat UI |
| `.env` | OpenAI API key |
| `docs.md` | Pydantic AI reference |

## Model

Uses **`openai:gpt-5-mini`** by default (see `MODEL` in `research.py`). Change it there if you want a different OpenAI model.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY is not set` | Set the key in `.env` and restart |
| No search results | DuckDuckGo may rate-limit; wait and retry |
| Slow runs | Normal — several LLM calls + parallel searches (1–3 min) |
| `ModuleNotFoundError: duckduckgo_search` | Run `pip install -r requirements.txt` |

## Customize

- **Research angles / report format:** edit agent `instructions` in `research.py`
- **Search depth:** adjust `max_results` and `fetch_pages` limits in `search.py` / `research.py`
- **UI:** edit `app.py`
