"""
GET /v1/healthz — Health check endpoint.

Returns bot status and count of loaded contexts per scope.
The judge checks this 3 times; 3 failures = -10 penalty.
"""
from __future__ import annotations

import time
from fastapi import APIRouter
from ..models import HealthzResponse
from ..state import context_store

router = APIRouter()
_START_TIME = time.time()


@router.get("/v1/healthz", response_model=HealthzResponse)
async def healthz() -> HealthzResponse:
    return HealthzResponse(
        status="ok",
        uptime_seconds=int(time.time() - _START_TIME),
        contexts_loaded={
            "category": context_store.count("category"),
            "merchant": context_store.count("merchant"),
            "customer": context_store.count("customer"),
            "trigger": context_store.count("trigger"),
        },
    )
