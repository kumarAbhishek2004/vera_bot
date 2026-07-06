"""
POST /v1/tick — Main composition endpoint.

The judge calls this with a list of available trigger IDs.
We score, rank, compose, and return a list of TickActions.

Budget: must return in <25s (5s safety buffer under the 30s hard limit).
If LLM calls would exceed budget, we truncate the action list.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter

from ..composer import Composer, get_composer
from ..config import settings
from ..engine import rank_triggers
from ..models import (
    CategoryContext,
    CustomerContext,
    MerchantContext,
    TickAction,
    TickRequest,
    TickResponse,
    TriggerContext,
)
from ..state import context_resolver, context_store, conversation_store, suppression_store

router = APIRouter()

# Time budget: leave 5s safety margin
_TICK_BUDGET_SECONDS = min(settings.llm_timeout, 24)


@router.post("/v1/tick", response_model=TickResponse)
async def tick(req: TickRequest) -> TickResponse:
    start = time.time()
    actions: List[TickAction] = []
    composer = get_composer()

    # Get currently fired suppression keys
    fired = suppression_store._fired  # read-only access

    # Rank triggers by priority score
    ranked = rank_triggers(
        available_trigger_ids=req.available_triggers,
        context_resolver=context_resolver,
        fired_suppression_keys=fired,
        max_actions=settings.max_actions_per_tick,
    )

    for trigger, merchant, category, customer in ranked:
        # Check budget
        elapsed = time.time() - start
        if elapsed > _TICK_BUDGET_SECONDS:
            break

        # Skip if already in an active conversation for this trigger
        existing_conv_id = _find_active_conv(trigger, merchant)
        if existing_conv_id:
            continue

        # Compose message
        try:
            sent_bodies: List[str] = []  # proactive first message, no history
            composed = composer.compose_message(
                trigger=trigger,
                merchant=merchant,
                category=category,
                customer=customer,
                sent_bodies=sent_bodies,
            )
        except Exception as e:
            # Log and skip — don't let one failure block the rest
            print(f"[tick] compose failed for trigger {trigger.id}: {e}")
            continue

        # Build conversation ID
        conv_id = _build_conv_id(trigger, merchant, customer)

        # Create conversation state
        await conversation_store.get_or_create(
            conversation_id=conv_id,
            merchant_id=merchant.merchant_id,
            customer_id=customer.customer_id if customer else None,
            trigger_id=trigger.id,
            category_slug=category.slug,
        )

        # Update conv state with sent body
        state = conversation_store.get(conv_id)
        if state:
            from ..models import Turn, FSMState
            state.turn_count += 1
            state.sent_bodies.append(composed.body)
            state.turns.append(Turn(
                turn_number=state.turn_count,
                from_role="vera",
                body=composed.body,
                action="send",
            ))
            await conversation_store.update(state)

        # Register suppression key
        if composed.suppression_key:
            await suppression_store.fire(composed.suppression_key)

        actions.append(TickAction(
            conversation_id=conv_id,
            merchant_id=merchant.merchant_id,
            customer_id=customer.customer_id if customer else None,
            send_as=composed.send_as,
            trigger_id=trigger.id,
            template_name=composed.template_name,
            template_params=composed.template_params,
            body=composed.body,
            cta=composed.cta,
            suppression_key=composed.suppression_key,
            rationale=composed.rationale,
        ))

    return TickResponse(actions=actions)


def _find_active_conv(trigger: TriggerContext, merchant: MerchantContext) -> Optional[str]:
    """Check if there's already an active conversation for this trigger+merchant."""
    # Check a few candidate conv IDs
    candidates = [
        _build_conv_id(trigger, merchant, None),
    ]
    for cid in candidates:
        if conversation_store.is_active(cid):
            return cid
    return None


def _build_conv_id(
    trigger: TriggerContext,
    merchant: MerchantContext,
    customer: Optional[CustomerContext],
) -> str:
    """
    Build a deterministic, meaningful conversation ID.
    Format: conv_{merchant_id}_{trigger_kind}_{trigger_id[:6]}
    """
    base = f"conv_{merchant.merchant_id}_{trigger.kind}_{trigger.id[:6]}"
    if customer:
        base += f"_{customer.customer_id[:8]}"
    return base
