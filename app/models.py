"""
app/models.py — All Pydantic models for the challenge.

Covers:
  • Context models (CategoryContext, MerchantContext, TriggerContext, CustomerContext)
  • Conversation state (ConversationState, FSMState, Turn)
  • API request / response models (5 endpoints)
  • Composed message output
"""
from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ══════════════════════════════════════════════════════════════
# CONTEXT TYPES
# ══════════════════════════════════════════════════════════════

class VoiceProfile(BaseModel):
    tone: str = "peer"
    register: str = "friendly"
    code_mix: str = "english"
    vocab_allowed: List[str] = []
    vocab_taboo: List[str] = []
    salutation_examples: List[str] = []
    tone_examples: List[str] = []


class OfferTemplate(BaseModel):
    id: str = ""
    title: str
    value: str = ""
    audience: str = "all"
    type: str = "service_at_price"
    status: str = "active"


class PeerStats(BaseModel):
    scope: str = ""
    avg_rating: float = 4.0
    avg_review_count: int = 0
    avg_views_30d: int = 0
    avg_calls_30d: int = 0
    avg_directions_30d: int = 0
    avg_ctr: float = 0.025
    avg_photos: int = 5
    avg_post_freq_days: int = 14
    retention_6mo_pct: float = 0.40


class DigestItem(BaseModel):
    id: str = ""
    kind: str = "research"
    title: str
    source: str = ""
    trial_n: Optional[int] = None
    patient_segment: Optional[str] = None
    summary: str = ""
    actionable: str = ""
    date: Optional[str] = None
    credits: Optional[int] = None


class ContentItem(BaseModel):
    id: str = ""
    title: str
    channel: str = "whatsapp"
    body: str = ""
    length_seconds: Optional[int] = None


class SeasonalBeat(BaseModel):
    month_range: str
    note: str


class TrendSignal(BaseModel):
    query: str
    delta_yoy: float
    segment_age: str = ""
    skew: str = ""


class CategoryContext(BaseModel):
    slug: str
    display_name: str = ""
    voice: VoiceProfile = Field(default_factory=VoiceProfile)
    offer_catalog: List[OfferTemplate] = []
    peer_stats: PeerStats = Field(default_factory=PeerStats)
    digest: List[DigestItem] = []
    patient_content_library: List[ContentItem] = []
    seasonal_beats: List[SeasonalBeat] = []
    trend_signals: List[TrendSignal] = []
    regulatory_authorities: List[str] = []
    professional_journals: List[str] = []


class Identity(BaseModel):
    name: str
    city: str = ""
    locality: str = ""
    place_id: str = ""
    verified: bool = False
    languages: List[str] = ["en"]
    owner_first_name: str = ""
    established_year: Optional[int] = None


class Subscription(BaseModel):
    status: str = "active"
    plan: str = "Basic"
    days_remaining: int = 365
    renewed_at: str = ""


class PerformanceDelta(BaseModel):
    views_pct: float = 0.0
    calls_pct: float = 0.0
    ctr_pct: float = 0.0


class PerformanceSnapshot(BaseModel):
    window_days: int = 30
    views: int = 0
    calls: int = 0
    directions: int = 0
    ctr: float = 0.0
    leads: int = 0
    delta_7d: PerformanceDelta = Field(default_factory=PerformanceDelta)


class MerchantOffer(BaseModel):
    id: str = ""
    title: str
    status: str = "active"
    started: str = ""
    ended: str = ""


class ConversationTurn(BaseModel):
    ts: str
    from_: str = Field(alias="from")
    body: str
    engagement: str = ""

    model_config = {"populate_by_name": True}


class CustomerAggregate(BaseModel):
    total_unique_ytd: int = 0
    lapsed_180d_plus: int = 0
    retention_6mo_pct: float = 0.0
    high_risk_adult_count: Optional[int] = None


class ReviewTheme(BaseModel):
    theme: str
    sentiment: str = "pos"
    occurrences_30d: int = 0
    common_quote: str = ""


class MerchantContext(BaseModel):
    merchant_id: str
    category_slug: str
    identity: Identity = Field(default_factory=lambda: Identity(name="Unknown"))
    subscription: Subscription = Field(default_factory=Subscription)
    performance: PerformanceSnapshot = Field(default_factory=PerformanceSnapshot)
    offers: List[MerchantOffer] = []
    conversation_history: List[Dict[str, Any]] = []
    customer_aggregate: CustomerAggregate = Field(default_factory=CustomerAggregate)
    signals: List[str] = []
    review_themes: List[ReviewTheme] = []


class TriggerContext(BaseModel):
    id: str
    scope: str  # "merchant" | "customer"
    kind: str
    source: str  # "external" | "internal"
    merchant_id: str
    customer_id: Optional[str] = None
    payload: Dict[str, Any] = {}
    urgency: int = 2
    suppression_key: str = ""
    expires_at: Optional[str] = None


class CustomerRelationship(BaseModel):
    first_visit: str = ""
    last_visit: str = ""
    visits_total: int = 0
    services_received: List[str] = []
    lifetime_value: float = 0.0
    favourite_dish: str = ""
    chronic_conditions: List[str] = []


class CustomerPreferences(BaseModel):
    preferred_slots: str = ""
    channel: str = "whatsapp"
    reminder_opt_in: bool = True
    wedding_date: str = ""
    training_focus: str = ""
    health_focus: str = ""
    preferred_stylist: str = ""
    office_nearby: bool = False
    family_size: int = 0
    household_size: int = 0
    delivery_address: str = ""


class CustomerConsent(BaseModel):
    opted_in_at: Optional[str] = None
    scope: List[str] = []


class CustomerIdentity(BaseModel):
    name: str
    phone_redacted: str = ""
    language_pref: str = "english"
    age_band: str = ""
    senior_citizen: bool = False


class CustomerContext(BaseModel):
    customer_id: str
    merchant_id: str
    identity: CustomerIdentity = Field(default_factory=lambda: CustomerIdentity(name="Customer"))
    relationship: CustomerRelationship = Field(default_factory=CustomerRelationship)
    state: str = "active"  # new | active | lapsed_soft | lapsed_hard | churned
    preferences: CustomerPreferences = Field(default_factory=CustomerPreferences)
    consent: CustomerConsent = Field(default_factory=CustomerConsent)


# ══════════════════════════════════════════════════════════════
# CONVERSATION STATE
# ══════════════════════════════════════════════════════════════

class FSMState(str, Enum):
    QUALIFYING = "QUALIFYING"
    COMMITTED = "COMMITTED"
    EXECUTING = "EXECUTING"
    WAITING = "WAITING"
    ENDED = "ENDED"


class Turn(BaseModel):
    turn_number: int
    from_role: str  # "vera" | "merchant" | "customer"
    body: str
    action: str = "send"
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ConversationState(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    trigger_id: str = ""
    category_slug: str = ""
    fsm_state: FSMState = FSMState.QUALIFYING
    turn_count: int = 0
    auto_reply_count: int = 0
    sent_bodies: List[str] = []
    turns: List[Turn] = []
    last_action_at: datetime = Field(default_factory=datetime.utcnow)


# ══════════════════════════════════════════════════════════════
# COMPOSED MESSAGE OUTPUT
# ══════════════════════════════════════════════════════════════

class ComposedMessage(BaseModel):
    body: str
    cta: str = "open_ended"
    send_as: str = "vera"
    template_name: str = "vera_generic_v1"
    template_params: List[str] = []
    suppression_key: str = ""
    rationale: str = ""


# ══════════════════════════════════════════════════════════════
# API REQUEST / RESPONSE MODELS
# ══════════════════════════════════════════════════════════════

# POST /v1/context
class ContextPushRequest(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: Dict[str, Any]
    delivered_at: str = ""


class ContextPushResponse(BaseModel):
    accepted: bool
    ack_id: str = ""
    stored_at: str = ""
    reason: str = ""
    current_version: Optional[int] = None


# POST /v1/tick
class TickRequest(BaseModel):
    now: str
    available_triggers: List[str] = []


class TickAction(BaseModel):
    conversation_id: str
    merchant_id: str
    customer_id: Optional[str] = None
    send_as: str
    trigger_id: str
    template_name: str
    template_params: List[str]
    body: str
    cta: str
    suppression_key: str
    rationale: str


class TickResponse(BaseModel):
    actions: List[TickAction] = []


# POST /v1/reply
class ReplyRequest(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str
    received_at: str = ""
    turn_number: int = 1


class ReplyResponse(BaseModel):
    action: str  # "send" | "wait" | "end"
    body: Optional[str] = None
    cta: Optional[str] = None
    wait_seconds: Optional[int] = None
    rationale: str = ""


# GET /v1/healthz
class HealthzResponse(BaseModel):
    status: str = "ok"
    uptime_seconds: int = 0
    contexts_loaded: Dict[str, int] = Field(
        default_factory=lambda: {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
    )


# GET /v1/metadata
class MetadataResponse(BaseModel):
    team_name: str
    team_members: List[str]
    model: str
    approach: str
    contact_email: str
    version: str
    submitted_at: str
