"""
POST /v1/context — Versioned context ingestion.

Idempotency rules:
  - Same (scope, context_id, version) → 409 (already stored)
  - Higher version → atomic replace → 200
  - Lower version → 409 (stale, reject)

Supported scopes: category, merchant, customer, trigger
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from ..models import ContextPushRequest, ContextPushResponse
from ..state import context_store

router = APIRouter()

_VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


@router.post("/v1/context", response_model=ContextPushResponse, status_code=200)
async def push_context(req: ContextPushRequest) -> ContextPushResponse:
    # Validate scope
    if req.scope not in _VALID_SCOPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid scope '{req.scope}'. Must be one of: {sorted(_VALID_SCOPES)}",
        )

    accepted, current_version = await context_store.push(
        scope=req.scope,
        context_id=req.context_id,
        version=req.version,
        payload=req.payload,
    )

    if accepted:
        return ContextPushResponse(
            accepted=True,
            ack_id=str(uuid.uuid4()),
            stored_at=datetime.now(timezone.utc).isoformat(),
        )
    else:
        # Return 409 for already-stored or stale version
        return ContextPushResponse(
            accepted=False,
            reason="stale_version",
            current_version=current_version,
        )
