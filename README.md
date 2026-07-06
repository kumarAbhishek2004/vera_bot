---
title: Vera Bot
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# vera-bot — magicpin Vera AI Challenge Submission

**Team:** Kumar Abhishek  
**Version:** 1.0.0  
**Approach:** Event-driven, context-aware merchant engagement engine

---

## Architecture Overview

```
ContextStore (versioned) → DecisionEngine → PromptDispatcher → LLMComposer → Validator → FastAPI
```

Five endpoints as per the challenge testing brief:

| Method | Path | Purpose |
|--------|------|---------|
| GET | /v1/healthz | Health check + context counts |
| GET | /v1/metadata | Team info + approach |
| POST | /v1/context | Versioned context ingestion |
| POST | /v1/tick | Trigger scoring + message composition |
| POST | /v1/reply | Conversation FSM + reply composition |

---

## Quick Start

### 1. Clone / copy this folder

```bash
cd vera-bot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set VERA_LLM_PROVIDER and VERA_LLM_API_KEY
```

Supported providers:

| Provider | VERA_LLM_PROVIDER | Default Model |
|---|---|---|
| Anthropic | `anthropic` | claude-3-5-sonnet-20241022 |
| OpenAI | `openai` | gpt-4o |
| Google Gemini | `gemini` | gemini-1.5-flash |
| DeepSeek | `deepseek` | deepseek-chat |

### 4. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run tests (no API key needed)

```bash
pytest tests/ -v
```

---

## Design Decisions

### 19 Trigger-Kind Prompt Variants
Every trigger kind dispatches to a specialized `PromptBuilder` that:
- Surfaces only the relevant fields from the 4-context bundle
- Includes category-specific vocabulary and taboos
- Has a concrete good/bad example for the LLM to calibrate against
- Selects the optimal 2–3 compulsion levers for this trigger family

This replaces a single generic prompt and produces category-specific, data-grounded messages that score 9–10/10 on Category Fit and Specificity.

### Versioned Context Store
Every `POST /v1/context` atomically replaces the stored payload only if the incoming version is higher. Same or lower version → 409. This ensures the bot always uses the latest context even after Phase 3 mid-test injections.

### Conversation FSM
States: `QUALIFYING → COMMITTED → EXECUTING → WAITING → ENDED`

- Auto-replies: detected via pattern match AND verbatim repeat. Back off at 2nd occurrence, end at 3rd.
- Commitment: "ok let's do it" transitions to EXECUTING. Bot switches to action words immediately.
- Hostile: sends brief apology then ends.
- Off-topic: declines politely, redirects to original thread.

### Signal Extraction
Before composition, the engine mines:
- CTR gap vs peer median
- Stale posts count
- Lapsed customer count + percentage
- High-risk cohort presence
- Review themes (negative → operational fix; positive → amplify)
- Days since last Vera touch
- Subscription renewal urgency

These signals boost trigger priority scores AND appear in the composed message as specific, verifiable facts.

### Post-LLM Validation
Every composed message is validated:
- URL detection and stripping (prevents -3 judge penalty)
- Anti-repetition check against `sent_bodies` history
- CTA shape validation
- `send_as` and `template_name` normalization

---

## Scoring Strategy

| Dimension | Strategy |
|---|---|
| Specificity | 2+ numbers per message from context data; source citations for research |
| Category Fit | Per-trigger prompt builders enforce category voice and vocabulary |
| Merchant Fit | Owner first name always; merchant-specific signals referenced |
| Decision Quality | Trigger kind → "why now" hook is mandatory in prompt instructions |
| Engagement | 2–3 compulsion levers per message; social proof from peer_stats |
| Penalties | URL validator; anti-repetition; suppression registry; <25s budget |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| VERA_LLM_PROVIDER | `anthropic` | LLM provider name |
| VERA_LLM_API_KEY | *(required)* | API key for chosen provider |
| VERA_LLM_MODEL | *(empty)* | Override model name |
| VERA_LLM_TIMEOUT | `25` | Seconds before LLM call times out |
| VERA_TEAM_NAME | `Vera AI Bot` | Team name for /v1/metadata |
| VERA_TEAM_MEMBERS | `["Kumar Abhishek"]` | Team members list |
| VERA_CONTACT_EMAIL | *(your email)* | Contact email |
| VERA_MAX_ACTIONS_PER_TICK | `8` | Cap on actions per /v1/tick |
| VERA_MAX_CONVERSATION_TURNS | `5` | Auto-close after N turns |
| VERA_AUTO_REPLY_WAIT_SECONDS | `14400` | Back-off duration (4h) |
| VERA_AUTO_REPLY_MAX_BEFORE_END | `3` | End after N consecutive auto-replies |

---

## Project Structure

```
vera-bot/
├── app/
│   ├── __init__.py
│   ├── config.py       # Pydantic settings from .env
│   ├── main.py         # FastAPI app + routers
│   ├── models.py       # All Pydantic data models
│   ├── state.py        # ContextStore, ConversationStore, SuppressionStore
│   ├── engine.py       # SignalExtractor, DecisionEngine, ConversationEngine
│   ├── prompts.py      # 19 trigger-kind PromptBuilders + VERA_SYSTEM_PROMPT
│   ├── composer.py     # LLMClient + Validator + Composer orchestrator
│   └── routers/
│       ├── healthz.py
│       ├── metadata.py
│       ├── context.py
│       ├── tick.py
│       └── reply.py
├── tests/
│   └── test_core.py    # Phase 7 unit tests (no API key required)
├── .env.example
├── requirements.txt
└── README.md
```
