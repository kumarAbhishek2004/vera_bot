"""
GET /v1/metadata — Team and bot identity information.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter
from ..config import settings
from ..models import MetadataResponse

router = APIRouter()


@router.get("/v1/metadata", response_model=MetadataResponse)
async def metadata() -> MetadataResponse:
    return MetadataResponse(
        team_name=settings.team_name,
        team_members=settings.team_members,
        model=settings.llm_model or f"{settings.llm_provider}-default",
        approach=(
            "Event-driven composition engine. "
            "4-context framework: Category → Merchant → Trigger → Customer. "
            "19 trigger-kind-specific prompt variants. "
            "Priority scoring with signal alignment. "
            "Conversation FSM with auto-reply detection, commitment transitions, and hostile handling. "
            "Post-LLM validation: URL stripping, anti-repetition, CTA shape."
        ),
        contact_email=settings.contact_email,
        version=settings.bot_version,
        submitted_at=datetime.now(timezone.utc).isoformat(),
    )
