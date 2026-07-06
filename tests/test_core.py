"""
tests/test_core.py — Phase 7 unit tests for vera-bot.

Tests:
  - ContextStore: versioning, idempotency, scoped counts
  - ConversationEngine: FSM transitions, auto-reply, commitment, hostile
  - DecisionEngine: trigger scoring, suppression
  - Validator: URL stripping, repetition detection
  - PromptDispatcher: all trigger kinds return a PromptBuilder
"""
from __future__ import annotations

import asyncio
import pytest
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.state import ContextStore, ConversationStore, SuppressionStore
from app.models import (
    CategoryContext, ConversationState, FSMState, MerchantContext, TriggerContext,
    PeerStats, PerformanceSnapshot, Identity, CustomerAggregate, Subscription, PerformanceDelta
)
from app.engine import classify_reply, transition_fsm, score_trigger, extract_signals, ExtractedSignals
from app.composer import validate_and_clean, _parse_llm_json, ValidationError
from app.prompts import get_prompt_builder, DefaultBuilder



# ══════════════════════════════════════════════════════════════
# CONTEXT STORE TESTS
# ══════════════════════════════════════════════════════════════

def test_context_store_accepts_new():
    store = ContextStore()
    accepted, cv = asyncio.run(store.push("merchant", "m001", 1, {"name": "Test"}))
    assert accepted is True
    assert cv is None


def test_context_store_rejects_same_version():
    store = ContextStore()
    asyncio.run(store.push("merchant", "m001", 1, {"name": "Test"}))
    accepted, cv = asyncio.run(store.push("merchant", "m001", 1, {"name": "Test Updated"}))
    assert accepted is False
    assert cv == 1


def test_context_store_accepts_higher_version():
    store = ContextStore()
    asyncio.run(store.push("merchant", "m001", 1, {"name": "Test"}))
    accepted, cv = asyncio.run(store.push("merchant", "m001", 2, {"name": "Test Updated"}))
    assert accepted is True
    assert cv is None
    payload = store.get("merchant", "m001")
    assert payload["name"] == "Test Updated"


def test_context_store_count_by_scope():
    store = ContextStore()
    asyncio.run(store.push("merchant", "m001", 1, {}))
    asyncio.run(store.push("merchant", "m002", 1, {}))
    asyncio.run(store.push("category", "dentists", 1, {}))
    assert store.count("merchant") == 2
    assert store.count("category") == 1
    assert store.count("trigger") == 0


def test_context_store_different_scopes_same_id():
    store = ContextStore()
    asyncio.run(store.push("merchant", "001", 1, {"type": "merchant"}))
    asyncio.run(store.push("customer", "001", 1, {"type": "customer"}))
    assert store.get("merchant", "001")["type"] == "merchant"
    assert store.get("customer", "001")["type"] == "customer"


# ══════════════════════════════════════════════════════════════
# CONVERSATION ENGINE TESTS
# ══════════════════════════════════════════════════════════════

def _make_state(conv_id="conv_test", merchant_id="m001"):
    return ConversationState(
        conversation_id=conv_id,
        merchant_id=merchant_id,
        fsm_state=FSMState.QUALIFYING,
    )


def test_classify_auto_reply():
    state = _make_state()
    msg = "Thank you for contacting us! Our team will respond shortly."
    result = classify_reply(msg, state)
    assert result == "auto_reply"


def test_classify_commitment():
    state = _make_state()
    msg = "ok let's do it! go ahead"
    result = classify_reply(msg, state)
    assert result == "commitment"


def test_classify_hostile():
    state = _make_state()
    msg = "Stop messaging me. This is useless spam."
    result = classify_reply(msg, state)
    assert result == "hostile"


def test_classify_engaged():
    state = _make_state()
    msg = "Interesting! Can you tell me more about the fluoride varnish finding?"
    result = classify_reply(msg, state)
    assert result == "engaged"


def test_hostile_transitions_to_ended():
    state = _make_state()
    updated, action = transition_fsm(state, "hostile", 14400, 3)
    assert updated.fsm_state == FSMState.ENDED
    assert action == "end"


def test_auto_reply_count_increments():
    state = _make_state()
    state.auto_reply_count = 0
    updated, action = transition_fsm(state, "auto_reply", 14400, 3)
    assert updated.auto_reply_count == 1
    assert action == "send_auto_reply_notice"


def test_auto_reply_3rd_ends():
    state = _make_state()
    state.auto_reply_count = 2
    updated, action = transition_fsm(state, "auto_reply", 14400, 3)
    assert updated.fsm_state == FSMState.ENDED
    assert action == "end"


def test_auto_reply_2nd_waits():
    state = _make_state()
    state.auto_reply_count = 1
    updated, action = transition_fsm(state, "auto_reply", 14400, 3)
    assert updated.fsm_state == FSMState.WAITING
    assert action == "wait"


def test_commitment_transitions_to_committed():
    state = _make_state()
    updated, action = transition_fsm(state, "commitment", 14400, 3)
    assert updated.fsm_state == FSMState.COMMITTED
    assert action == "send_commitment"


# ══════════════════════════════════════════════════════════════
# VALIDATOR TESTS
# ══════════════════════════════════════════════════════════════

def test_validator_strips_urls():
    raw = {
        "body": "Check this out: https://example.com for more info",
        "cta": "open_ended",
    }
    cleaned = validate_and_clean(raw)
    assert "http" not in cleaned["body"]
    assert "example.com" not in cleaned["body"]
    assert len(cleaned["body"]) > 0


def test_validator_rejects_empty_body():
    raw = {"body": "", "cta": "open_ended"}
    with pytest.raises(ValidationError):
        validate_and_clean(raw)


def test_validator_anti_repetition():
    raw = {"body": "Dr. Meera, the JIDA Oct issue landed with a key finding.", "cta": "open_ended"}
    sent_bodies = ["Dr. Meera, the JIDA Oct issue landed with a key finding."]
    with pytest.raises(ValidationError):
        validate_and_clean(raw, sent_bodies)


def test_validator_normalizes_invalid_cta():
    raw = {"body": "Hello Dr. Meera", "cta": "invalid_cta_value"}
    cleaned = validate_and_clean(raw)
    assert cleaned["cta"] == "open_ended"


def test_validator_fills_template_name():
    raw = {"body": "Hello", "cta": "none", "template_name": ""}
    cleaned = validate_and_clean(raw)
    assert cleaned["template_name"] == "vera_generic_v1"


# ══════════════════════════════════════════════════════════════
# JSON PARSER TESTS
# ══════════════════════════════════════════════════════════════

def test_parse_clean_json():
    text = '{"body": "Hello Dr. Meera", "cta": "open_ended"}'
    result = _parse_llm_json(text)
    assert result["body"] == "Hello Dr. Meera"


def test_parse_markdown_wrapped_json():
    text = '```json\n{"body": "Hello", "cta": "binary_yes_no"}\n```'
    result = _parse_llm_json(text)
    assert result["cta"] == "binary_yes_no"


def test_parse_fails_gracefully():
    with pytest.raises(ValidationError):
        _parse_llm_json("This is not JSON at all.")


# ══════════════════════════════════════════════════════════════
# PROMPT DISPATCHER TESTS
# ══════════════════════════════════════════════════════════════

def test_all_trigger_kinds_have_builder():
    trigger_kinds = [
        "research_digest", "regulation_change", "festival_upcoming",
        "perf_dip", "perf_spike", "seasonal_perf_dip", "recall_due",
        "customer_lapsed_soft", "customer_lapsed_hard", "curious_ask_due",
        "renewal_due", "review_theme_emerged", "milestone_reached",
        "dormant_with_vera", "competitor_opened", "wedding_package_followup",
        "chronic_refill_due", "active_planning_intent", "weather_heatwave",
    ]
    for kind in trigger_kinds:
        builder = get_prompt_builder(kind)
        assert builder is not None, f"No builder for trigger kind: {kind}"


def test_unknown_trigger_kind_returns_default():
    builder = get_prompt_builder("some_future_trigger_kind_xyz")
    from app.prompts import DefaultBuilder
    assert isinstance(builder, DefaultBuilder)


# ══════════════════════════════════════════════════════════════
# SIGNAL EXTRACTOR TESTS
# ══════════════════════════════════════════════════════════════

def _make_merchant(signals=None, ctr=0.021, lapsed=5, total=100):
    return MerchantContext(
        merchant_id="m_test",
        category_slug="dentists",
        identity=Identity(name="Test Clinic", owner_first_name="Test"),
        performance=PerformanceSnapshot(ctr=ctr, views=1000, calls=15,
                                         delta_7d=PerformanceDelta()),
        customer_aggregate=CustomerAggregate(
            total_unique_ytd=total,
            lapsed_180d_plus=lapsed,
            retention_6mo_pct=0.38,
        ),
        signals=signals or [],
        subscription=Subscription(),
    )


def _make_category():
    return CategoryContext(
        slug="dentists",
        peer_stats=PeerStats(avg_ctr=0.030, retention_6mo_pct=0.42),
    )


def test_extract_ctr_below_peer():
    merchant = _make_merchant(ctr=0.021)
    category = _make_category()
    sigs = extract_signals(merchant, category)
    assert sigs.ctr_below_peer is True
    assert sigs.ctr_gap_pct > 0


def test_extract_stale_posts():
    merchant = _make_merchant(signals=["stale_posts:22d"])
    category = _make_category()
    sigs = extract_signals(merchant, category)
    assert sigs.stale_posts is True
    assert sigs.days_since_post == 22


def test_extract_lapse_count():
    merchant = _make_merchant(lapsed=45, total=200)
    category = _make_category()
    sigs = extract_signals(merchant, category)
    assert sigs.lapsed_count == 45
    assert abs(sigs.lapsed_pct - 22.5) < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
