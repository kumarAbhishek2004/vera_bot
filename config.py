"""
app/config.py — Centralised settings loaded from .env
All runtime configuration lives here. Import `settings` anywhere.
"""
from __future__ import annotations

from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── LLM ───────────────────────────────────────────────────────
    llm_provider: str = "anthropic"          # anthropic | openai | gemini | deepseek
    llm_api_key: str = ""
    llm_model: str = ""                      # empty → use provider default
    llm_timeout: int = 25                    # seconds; hard limit is 30s

    # ── Team ──────────────────────────────────────────────────────
    team_name: str = "Vera AI Bot"
    team_members: List[str] = ["Kumar Abhishek"]
    contact_email: str = "kumar@example.com"
    bot_version: str = "1.0.0"

    # ── Bot Tuning ────────────────────────────────────────────────
    max_actions_per_tick: int = 8            # conservative under 20-action cap
    max_conversation_turns: int = 5
    auto_reply_wait_seconds: int = 14400     # 4h back-off after auto-reply
    auto_reply_max_before_end: int = 3       # end after N consecutive auto-replies

    model_config = {"env_file": ".env", "env_prefix": "VERA_"}


settings = Settings()
