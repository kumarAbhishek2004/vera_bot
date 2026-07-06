"""
app/main.py — FastAPI application assembly.

Mounts all 5 required routers:
  GET  /v1/healthz
  GET  /v1/metadata
  POST /v1/context
  POST /v1/tick
  POST /v1/reply

Also includes a /v1/teardown endpoint (used by the judge between test runs).
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .routers import context, healthz, metadata, reply, tick
from .state import context_store, conversation_store, suppression_store

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("vera-bot")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"vera-bot {settings.bot_version} starting up")
    logger.info(f"LLM provider: {settings.llm_provider}")
    if not settings.llm_api_key:
        logger.warning("VERA_LLM_API_KEY is not set! LLM calls will fail.")
    yield
    logger.info("vera-bot shutting down")


app = FastAPI(
    title="vera-bot",
    description="magicpin Vera AI Challenge submission — event-driven merchant engagement engine",
    version=settings.bot_version,
    lifespan=lifespan,
)

# CORS (judge may call from different origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routers
app.include_router(healthz.router, tags=["health"])
app.include_router(metadata.router, tags=["metadata"])
app.include_router(context.router, tags=["context"])
app.include_router(tick.router, tags=["tick"])
app.include_router(reply.router, tags=["reply"])


# ── Teardown endpoint (judge uses between test runs) ──────────

from fastapi import Response


@app.post("/v1/teardown", tags=["admin"], status_code=204)
async def teardown() -> Response:
    """Reset all state between test runs."""
    context_store._data.clear()
    conversation_store._data.clear()
    await suppression_store.reset()
    logger.info("State reset via /v1/teardown")
    return Response(status_code=204)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "vera-bot",
        "version": settings.bot_version,
        "team": settings.team_name,
        "docs": "/docs",
    }
