"""
app/engine.py — Decision Engine, Signal Extractor, Conversation Engine.

SignalExtractor   : mines MerchantContext + CategoryContext → structured signals
DecisionEngine    : scores each trigger, returns ranked action list
ConversationEngine: FSM transitions for /v1/reply handling
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    CategoryContext,
    ConversationState,
    CustomerContext,
    FSMState,
    MerchantContext,
    TriggerContext,
    Turn,
)


# ══════════════════════════════════════════════════════════════
# SIGNAL EXTRACTOR
# ══════════════════════════════════════════════════════════════

class ExtractedSignals:
    """Structured signals derived from contexts for use in prompt building."""

    def __init__(self) -> None:
        # Performance signals
        self.ctr_below_peer: bool = False
        self.ctr_gap_pct: float = 0.0
        self.ctr_ratio: float = 1.0
        self.views_delta_7d: float = 0.0
        self.calls_delta_7d: float = 0.0

        # Content signals
        self.stale_posts: bool = False
        self.days_since_post: int = 0
        self.post_freq_below_peer: bool = False

        # Customer signals
        self.lapsed_count: int = 0
        self.lapsed_pct: float = 0.0
        self.high_risk_cohort: bool = False
        self.high_risk_count: int = 0
        self.retention_gap: float = 0.0
        self.total_customers: int = 0

        # Engagement signals
        self.recently_engaged: bool = False
        self.dormant_with_vera: bool = False
        self.days_since_vera_touch: int = 0
        self.last_merchant_intent: str = ""

        # Review signals
        self.negative_review_themes: List[str] = []
        self.positive_review_themes: List[str] = []

        # Subscription signals
        self.renewal_imminent: bool = False
        self.days_to_renewal: int = 365


def extract_signals(
    merchant: MerchantContext,
    category: CategoryContext,
) -> ExtractedSignals:
    """
    Derive structured signals from merchant + category contexts.
    Returns an ExtractedSignals object for use by PromptBuilders.
    """
    sig = ExtractedSignals()
    signals_raw = set(merchant.signals)

    # ── Performance ────────────────────────────────────────────
    peer_ctr = category.peer_stats.avg_ctr if category.peer_stats else 0.025
    merchant_ctr = merchant.performance.ctr
    sig.ctr_below_peer = merchant_ctr < peer_ctr
    sig.ctr_gap_pct = round((peer_ctr - merchant_ctr) / peer_ctr * 100, 1) if peer_ctr > 0 else 0.0
    sig.ctr_ratio = round(merchant_ctr / peer_ctr, 2) if peer_ctr > 0 else 1.0
    sig.views_delta_7d = merchant.performance.delta_7d.views_pct
    sig.calls_delta_7d = merchant.performance.delta_7d.calls_pct

    # ── Posts ──────────────────────────────────────────────────
    for s in signals_raw:
        if s.startswith("stale_posts:"):
            sig.stale_posts = True
            try:
                sig.days_since_post = int(s.split(":")[1].replace("d", ""))
            except (IndexError, ValueError):
                sig.days_since_post = 14

    peer_post_freq = category.peer_stats.avg_post_freq_days if category.peer_stats else 14
    sig.post_freq_below_peer = sig.stale_posts and sig.days_since_post > peer_post_freq

    # ── Customers ─────────────────────────────────────────────
    agg = merchant.customer_aggregate
    sig.total_customers = agg.total_unique_ytd
    sig.lapsed_count = agg.lapsed_180d_plus
    sig.lapsed_pct = (
        round(agg.lapsed_180d_plus / agg.total_unique_ytd * 100, 1)
        if agg.total_unique_ytd > 0
        else 0.0
    )
    sig.high_risk_cohort = "high_risk_adult_cohort" in signals_raw
    sig.high_risk_count = agg.high_risk_adult_count or 0
    peer_retention = category.peer_stats.retention_6mo_pct if category.peer_stats else 0.42
    sig.retention_gap = round((peer_retention - agg.retention_6mo_pct) * 100, 1)

    # ── Engagement ────────────────────────────────────────────
    sig.recently_engaged = "engaged_in_last_48h" in signals_raw
    sig.dormant_with_vera = "dormant_with_vera" in signals_raw

    if merchant.conversation_history:
        last_vera_msg = None
        for turn in reversed(merchant.conversation_history):
            from_field = turn.get("from", "")
            if from_field == "vera":
                last_vera_msg = turn
                break
        if last_vera_msg:
            try:
                ts_str = last_vera_msg.get("ts", "")
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                now = datetime.now(timezone.utc)
                sig.days_since_vera_touch = (now - ts).days
            except Exception:
                sig.days_since_vera_touch = 30

        # Extract last merchant intent
        for turn in reversed(merchant.conversation_history):
            if turn.get("from", "") == "merchant":
                sig.last_merchant_intent = turn.get("body", "")
                break

    # ── Reviews ───────────────────────────────────────────────
    for theme in merchant.review_themes:
        if theme.sentiment == "neg":
            sig.negative_review_themes.append(theme.theme)
        else:
            sig.positive_review_themes.append(theme.theme)

    # ── Subscription ──────────────────────────────────────────
    sig.days_to_renewal = merchant.subscription.days_remaining
    sig.renewal_imminent = merchant.subscription.days_remaining <= 14

    return sig


# ══════════════════════════════════════════════════════════════
# DECISION ENGINE
# ══════════════════════════════════════════════════════════════

def _is_expired(trigger: TriggerContext) -> bool:
    """Check if trigger has passed its expires_at."""
    if not trigger.expires_at:
        return False
    try:
        exp = datetime.fromisoformat(trigger.expires_at.replace("Z", "+00:00"))
        return datetime.now(timezone.utc) > exp
    except Exception:
        return False


def score_trigger(
    trigger: TriggerContext,
    merchant: MerchantContext,
    category: CategoryContext,
    signals: ExtractedSignals,
    fired_suppression_keys: set,
) -> float:
    """
    Compute a priority score for sending this trigger to this merchant.
    Higher = more important to send now.
    Returns 0 or negative if should be skipped.
    """
    # Hard skips
    if _is_expired(trigger):
        return -100.0
    if trigger.suppression_key and trigger.suppression_key in fired_suppression_keys:
        return -100.0

    print(f"\n------ SCORE DEBUG for {trigger.id} ------")
    print(f"Urgency: {trigger.urgency}")

    score = float(trigger.urgency * 2)  # base: 2–10
    print(f"Base Score: {score}")

    kind = trigger.kind

    # Signal-alignment boosts
    if kind == "research_digest" and signals.high_risk_cohort:
        score += 3.0
        print("+3.0 research digest + high risk cohort")
    if kind in ("perf_dip", "seasonal_perf_dip") and signals.calls_delta_7d < -0.25:
        score += 3.0
        print("+3.0 perf dip")
    if kind == "perf_spike" and signals.views_delta_7d > 0.20:
        score += 2.0
        print("+2.0 perf spike")
    if kind == "renewal_due" and signals.renewal_imminent:
        score += 4.0
        print("+4.0 renewal imminent")
    if kind == "recall_due":
        score += 2.0  # customer-facing, high intent
        print("+2.0 recall due")
    if kind in ("customer_lapsed_soft", "customer_lapsed_hard") and signals.lapsed_count > 30:
        score += 2.0
        print("+2.0 high lapsed count")
    if kind == "review_theme_emerged" and signals.negative_review_themes:
        score += 2.0
        print("+2.0 negative review theme")
    if kind == "dormant_with_vera" and signals.dormant_with_vera:
        score += 3.0
        print("+3.0 dormant with vera")

    # Engagement warmth boost
    if signals.recently_engaged:
        score += 1.5
        print("+1.5 recently engaged")

    # Staleness discount (trigger not yet expired but getting old)
    if signals.days_since_vera_touch < 1:
        score -= 1.0  # don't spam if touched today
        print("-1.0 touched today")

    print(f"Final Score: {score}")
    return score


def rank_triggers(
    available_trigger_ids: List[str],
    context_resolver,
    fired_suppression_keys: set,
    max_actions: int,
) -> List[Tuple[TriggerContext, MerchantContext, CategoryContext, Optional[CustomerContext]]]:
    """
    Load contexts, score each trigger, return top-N ranked tuples ready for composition.
    """
    scored: List[Tuple[float, TriggerContext, MerchantContext, CategoryContext, Optional[CustomerContext]]] = []

    for tid in available_trigger_ids:
        print("\n" + "=" * 60)
        print("Processing Trigger:", tid)

        trigger = context_resolver.get_trigger(tid)
        print("Trigger:", trigger)
        if not trigger:
            print("❌ Trigger NOT FOUND")
            continue

        merchant = context_resolver.get_merchant(trigger.merchant_id)
        print("Merchant:", merchant)
        if not merchant:
            print("❌ Merchant NOT FOUND")
            continue

        category = context_resolver.get_category(merchant.category_slug)
        print("Category:", category)

        customer = None
        if trigger.customer_id:
            customer = context_resolver.get_customer(trigger.customer_id)

        signals = extract_signals(merchant, category)
        print("Signals:", signals)

        s = score_trigger(trigger, merchant, category, signals, fired_suppression_keys)

        if s > 0:
            print("✅ Trigger Accepted")
            scored.append((s, trigger, merchant, category, customer))
        else:
            print("❌ Trigger Rejected (Score <= 0)")

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)

    # Return top-N tuples
    return [(t, m, c, cu) for (_, t, m, c, cu) in scored[:max_actions]]


# ══════════════════════════════════════════════════════════════
# CONVERSATION ENGINE
# ══════════════════════════════════════════════════════════════

# ── Pattern Banks ──────────────────────────────────────────────

_AUTO_REPLY_PATTERNS = [
    r"thank you for contacting",
    r"our team will respond",
    r"automated (response|reply|message)",
    r"we will get back to you",
    r"this is an auto",
    r"main ek automated",
    r"yeh ek automated",
    r"aapki jaankari ke liye bahut-bahut shukriya.*team",
    r"hamari team.*pahuncha deti",
]
_AUTO_REPLY_RE = re.compile("|".join(_AUTO_REPLY_PATTERNS), re.IGNORECASE)

_COMMITMENT_PHRASES = [
    "ok lets do it", "ok let's do it", "let's do it", "let's start",
    "yes please", "yes, please", "haan kar do", "haan chalega",
    "go ahead", "proceed", "confirm", "sounds good", "done",
    "yes go", "chalega", "karega", "theek hai", "bilkul",
    "i'm in", "let's go", "sure, go", "yes, go ahead",
]

_HOSTILE_PHRASES = [
    "stop messaging", "stop sending", "don't message", "don't contact",
    "not interested", "spam", "useless", "annoying", "leave me alone",
    "block", "unsubscribe", "opt out", "band karo", "mat bhejo",
    "disturb mat karo", "bakwas", "bekar",
]

_OFF_TOPIC_PHRASES = [
    "gst", "tax", "income tax", "filing", "tally", "legal",
    "loan", "insurance", "property", "bank", "salary",
    "other business", "different topic",
]


def classify_reply(message: str, state: ConversationState) -> str:
    """
    Classify a merchant/customer reply into one of:
    auto_reply | commitment | hostile | off_topic | engaged
    """
    msg_lower = message.lower().strip()

    # Auto-reply: pattern match OR verbatim repeat
    if _AUTO_REPLY_RE.search(msg_lower):
        return "auto_reply"
    if state.turns and any(t.body.lower().strip() == msg_lower for t in state.turns if t.from_role != "vera"):
        return "auto_reply"

    # Hostile
    if any(p in msg_lower for p in _HOSTILE_PHRASES):
        return "hostile"

    # Commitment
    if any(p in msg_lower for p in _COMMITMENT_PHRASES):
        return "commitment"

    # Off-topic
    if any(p in msg_lower for p in _OFF_TOPIC_PHRASES):
        return "off_topic"

    return "engaged"


def transition_fsm(
    state: ConversationState,
    reply_type: str,
    config_auto_reply_wait: int,
    config_auto_reply_max: int,
) -> Tuple[ConversationState, str]:
    """
    Apply FSM transition based on reply_type.
    Returns (updated_state, next_bot_action) where next_bot_action is:
      "send" | "wait" | "end"
    """
    if state.fsm_state == FSMState.ENDED:
        return state, "end"

    if reply_type == "hostile":
        state.fsm_state = FSMState.ENDED
        return state, "end"

    if reply_type == "auto_reply":
        state.auto_reply_count += 1
        if state.auto_reply_count >= config_auto_reply_max:
            state.fsm_state = FSMState.ENDED
            return state, "end"
        elif state.auto_reply_count >= 2:
            state.fsm_state = FSMState.WAITING
            return state, "wait"
        else:
            # First auto-reply: try once more with "for the owner" message
            return state, "send_auto_reply_notice"

    if reply_type == "commitment":
        state.fsm_state = FSMState.COMMITTED
        return state, "send_commitment"

    if reply_type == "off_topic":
        # Stay in current state, redirect
        return state, "send_off_topic"

    # Engaged reply: advance conversation
    if state.turn_count >= 5:
        state.fsm_state = FSMState.ENDED
        return state, "end"

    return state, "send_engaged"


def build_reply_context_summary(state: ConversationState) -> str:
    """Build a short conversation history string for the reply prompt."""
    lines = []
    for t in state.turns[-4:]:  # last 4 turns
        role = "Vera" if t.from_role == "vera" else "Merchant"
        lines.append(f"[{role}]: {t.body}")
    return "\n".join(lines)
