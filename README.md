# Market Research Assistant

> An AI-powered market research tool that turns a single industry name into a structured executive report — validated, grounded in real sources, and ready to download.

**Live pipeline:** Industry validation → Wikipedia retrieval & relevance scoring → Executive report generation

Built with **Streamlit**, **LangChain**, and **OpenAI GPT**.

---

## What it does

Enter any industry name (e.g. *Automotive*, *Cloud Computing*, *Fast Fashion*) and the app runs a three-step pipeline:

| Step | What happens | Guardrail |
|------|-------------|-----------|
| **1 — Validate** | LLM checks the input is a real, researchable industry — not a typo, abstract concept, or inappropriate term | Blocks bad inputs before any expensive API calls |
| **2 — Retrieve** | Pulls up to 10 Wikipedia pages, then scores each one for industry-level relevance (0–5) using a second LLM call | Adaptive threshold selection ensures only quality sources proceed |
| **3 — Generate** | Synthesises a ≤ 500-word structured executive report from the filtered sources | Source-only constraint prevents hallucination; post-generation word count check enforces the limit |

The final report is downloadable as a `.md` file.

---

## Architecture & design decisions

```
User input
    │
    ▼
┌─────────────────────────────────┐
│  Step 1: Industry Validation    │  LLM (temp=0.0) — deterministic gate
│  VALID / INVALID / INAPPROPRIATE│
└────────────┬────────────────────┘
             │ pass
             ▼
┌─────────────────────────────────┐
│  Step 2: Wikipedia Retrieval    │  WikipediaRetriever (top_k=8, max=10 pages)
│  + LLM Relevance Scoring (0–5) │  LLM (temp=0.0) per page — consistent scoring
│  + Adaptive Source Selection   │  Priority: score ≥4 → ≥3 → ≥2 → best available
└────────────┬────────────────────┘
             │ ≥3 sources
             ▼
┌─────────────────────────────────┐
│  Step 3: Report Generation      │  LLM (temp=0.2) — factual but readable prose
│  ≤ 500 words, 4 sections        │  Source-only constraint + word count guard
│  Markdown → HTML rendering     │
└─────────────────────────────────┘
```

**Why these choices:**
- **temp=0.0 for validation and scoring** — deterministic classification; the same input must always produce the same verdict
- **temp=0.2 for report generation** — enough variance for natural prose without drifting from the facts
- **LLM-based relevance scoring instead of keyword matching** — Wikipedia returns noisy results (country economy pages, company bios, tangential topics); an LLM scorer with explicit scoring rubrics filters these out reliably
- **`st.cache_data` on retrieval and validation** — repeated searches for the same industry skip the API calls entirely; cache version key allows prompt changes to invalidate stale results
- **Timeouts as guardrails** — validation capped at 15s, retrieval at 60s, generation at 60s; slow responses indicate vague inputs and surface an actionable error
- **Session state architecture** — Streamlit reruns the full script on every interaction; all pipeline state is persisted explicitly so downstream steps never show stale data

---

## Repository structure

```
├── market_research_assistant.py   # Full application — pipeline logic, UI, prompts
├── requirements.txt               # Python dependencies (pinned to compatible ranges)
└── .devcontainer/                 # Dev container config for reproducible environment
```

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| UI & app framework | Streamlit |
| LLM orchestration | LangChain Core, LangChain OpenAI |
| Language model | OpenAI GPT (gpt-5-mini) |
| Knowledge retrieval | LangChain Wikipedia Retriever |
| Secrets management | Streamlit Secrets (deployment) / password input (local) |

---

## Running locally

**Prerequisites:** Python 3.10+, an OpenAI API key

```bash
# 1. Clone the repo
git clone <repo-url>
cd <repo-folder>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run market_research_assistant.py
```

Enter your OpenAI API key in the sidebar when prompted. For deployment on Streamlit Cloud, set `OPENAI_API_KEY` in the app secrets instead.

---

## Key engineering choices worth noting

**Hallucination prevention** — the report prompt instructs the model to use *only* the retrieved Wikipedia sources and explicitly prohibits external knowledge. This is reinforced by limiting the context window to 12,000 characters of source material.

**Adaptive source selection** — rather than hard-failing when fewer than 5 high-quality sources are found, the pipeline degrades gracefully through three quality tiers and communicates source quality to the user transparently.

**State persistence across reruns** — Streamlit's execution model re-runs the entire script on every UI interaction. All pipeline state is stored in `st.session_state` with a defined schema (`DEFAULT_STATE`) so the workflow remains coherent without browser refreshes.

**Input sanitisation** — the validation step checks for inappropriate content (profanity, sensitive material) before any downstream API calls are made, making the tool safe for deployment.

---

*Built for MSIN0231 Machine Learning for Business · UCL School of Management*
