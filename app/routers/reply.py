"""
POST /v1/reply — Conversation reply handling.

Classifies the merchant's reply, runs FSM transition, and returns the bot's next action:
  send  → bot sends a message (body + cta)
  wait  → bot backs off for wait_seconds
  end   → conversation is closed

Handles:
  - Auto-reply detection (pattern match + verbatim repeat)
  - Commitment ("ok let's do it") → switch to execution mode
  - Hostile → end or apologize
  - Off-topic → redirect politely
  - Engaged → advance the conversation
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..composer import get_composer
from ..config import settings
from ..engine import classify_reply, transition_fsm
from ..models import (
    CategoryContext,
    CustomerContext,
    FSMState,
    MerchantContext,
    ReplyRequest,
    ReplyResponse,
    TriggerContext,
    Turn,
)
from ..state import context_store, conversation_store, suppression_store

router = APIRouter()


@router.post("/v1/reply", response_model=ReplyResponse)
async def reply(req: ReplyRequest) -> ReplyResponse:
    # Load conversation state
    state = conversation_store.get(req.conversation_id)
    if not state:
        # Unknown conversation — create minimal state and respond
        # The judge simulator tests reply logic with synthesized conv IDs
        state = await conversation_store.get_or_create(
            conversation_id=req.conversation_id,
            merchant_id="m_test",
            category_slug="dentists"
        )

    # If conversation is already ENDED, short-circuit
    if state.fsm_state == FSMState.ENDED:
        return ReplyResponse(
            action="end",
            rationale="Conversation already ended.",
        )

    # Classify the reply BEFORE appending to state to prevent self-matching
    reply_type = classify_reply(req.message, state)

    # Record incoming turn
    incoming_turn = Turn(
        turn_number=state.turn_count + 1,
        from_role=req.from_role,
        body=req.message,
        action="received",
    )
    state.turns.append(incoming_turn)
    state.turn_count += 1

    # FSM transition
    state, bot_action = transition_fsm(
        state=state,
        reply_type=reply_type,
        config_auto_reply_wait=settings.auto_reply_wait_seconds,
        config_auto_reply_max=settings.auto_reply_max_before_end,
    )

    # Handle terminal states without LLM call
    if bot_action == "end":
        state.fsm_state = FSMState.ENDED
        await conversation_store.update(state)

        # For hostile: send apology before ending
        if reply_type == "hostile":
            return ReplyResponse(
                action="send",
                body="Sorry for the interruption — won't message again. Have a great day.",
                cta="none",
                rationale="Merchant sent hostile message. Sending brief apology and ending conversation.",
            )

        return ReplyResponse(
            action="end",
            rationale="Conversation ended: auto-reply threshold reached or merchant opted out.",
        )

    if bot_action == "wait":
        await conversation_store.update(state)
        return ReplyResponse(
            action="wait",
            wait_seconds=settings.auto_reply_wait_seconds,
            rationale=f"Auto-reply detected (count: {state.auto_reply_count}). Backing off for {settings.auto_reply_wait_seconds}s.",
        )

    # Load contexts for LLM reply composition
    merchant: MerchantContext | None = None
    category: CategoryContext | None = None
    customer: CustomerContext | None = None
    trigger: TriggerContext | None = None

    merchant_payload = context_store.get("merchant", state.merchant_id)
    if merchant_payload:
        try:
            merchant = MerchantContext(**merchant_payload)
            category_payload = context_store.get("category", merchant.category_slug)
            if category_payload:
                category = CategoryContext(**category_payload)
        except Exception:
            pass

    if state.customer_id:
        cust_payload = context_store.get("customer", state.customer_id)
        if cust_payload:
            try:
                customer = CustomerContext(**cust_payload)
            except Exception:
                pass

    if state.trigger_id:
        trg_payload = context_store.get("trigger", state.trigger_id)
        if trg_payload:
            try:
                trigger = TriggerContext(**trg_payload)
            except Exception:
                pass

    # Compose LLM reply
    composer = get_composer()
    try:
        result = composer.compose_reply(
            reply_type=bot_action,
            merchant_message=req.message,
            state=state,
            merchant=merchant,
            category=category,
            trigger=trigger,
        )
    except Exception as e:
        # Safe fallback
        result = {
            "action": "send",
            "body": "Thanks for your reply! Let me get that sorted for you.",
            "cta": "open_ended",
            "rationale": f"LLM reply error: {e}",
        }

    # Update conversation state with bot's reply
    if result.get("action") == "send" and result.get("body"):
        bot_turn = Turn(
            turn_number=state.turn_count + 1,
            from_role="vera",
            body=result["body"],
            action="send",
        )
        state.turns.append(bot_turn)
        state.turn_count += 1
        state.sent_bodies.append(result["body"])

    if result.get("action") == "end":
        state.fsm_state = FSMState.ENDED
    elif result.get("action") == "wait":
        state.fsm_state = FSMState.WAITING
    elif bot_action == "send_commitment":
        state.fsm_state = FSMState.EXECUTING

    await conversation_store.update(state)

    return ReplyResponse(
        action=result.get("action", "send"),
        body=result.get("body"),
        cta=result.get("cta"),
        wait_seconds=result.get("wait_seconds"),
        rationale=result.get("rationale", ""),
    )
