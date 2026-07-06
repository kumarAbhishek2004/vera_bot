"""
app/prompts.py — Trigger-kind-specific prompt builders.

Architecture:
  VERA_SYSTEM_PROMPT   — Vera's identity, voice rules, compulsion levers
  PromptBuilder (ABC)  — base class with helper utilities
  19 concrete builders — one per trigger kind
  PromptDispatcher     — maps trigger.kind → PromptBuilder instance
  ReplyPromptBuilder   — for /v1/reply contexts (all reply types)
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .engine import ExtractedSignals
from .models import (
    CategoryContext,
    ConversationState,
    CustomerContext,
    MerchantContext,
    TriggerContext,
)


# ══════════════════════════════════════════════════════════════
# VERA SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

VERA_SYSTEM_PROMPT = """You are Vera, magicpin's AI WhatsApp assistant for Indian merchants.
You help merchants grow their Google Business Profile, run campaigns, and engage their customers.

═══ YOUR IDENTITY ═══
• Peer/colleague tone — NOT salesperson, NOT corporate bot
• You do the work FOR merchants first, then ask for approval
• ONLY use data from the provided context. Never fabricate numbers, names, or citations.
• Match the merchant's language preference exactly (see LANGUAGE section below)

═══ LANGUAGE RULES ═══
• languages includes "hi" OR language_pref = "hi-en mix" → Hindi-English code-mix naturally
  Examples: "Apke liye 2 slots ready hain", "Main draft kar deta hoon", "Chalega?"
• language_pref = "english" or languages = ["en"] only → Pure English
• language_pref = "te-en mix" → Telugu-English mix welcome
• language_pref = "ta-en mix" → Tamil-English mix welcome
• language_pref = "kn-en mix" → Kannada-English mix welcome
• language_pref = "hi" → Primarily Hindi
• Always use "Dr." prefix for dentists when addressing by full name or title

═══ CATEGORY VOICE MATRIX ═══
dentists    → peer_clinical: technical vocabulary OK (fluoride, caries, JIDA, DCI, scaling,
               endodontic, aligner, OPG, IOPA, zirconia). NEVER "guaranteed", "cure", "100% safe"
salons      → warm_practical: friendly, professional, uses service names (balayage, keratin, bridal)
restaurants → operator_peer: industry terms (covers, AOV, Swiggy, thali, delivery radius, footfall)
gyms        → coach_motivational: encouraging, no guilt-tripping; uses (ad spend, conversion, PT, HIIT)
pharmacies  → trustworthy_precise: molecule names, batch numbers, chronic Rx. NO unverified claims

═══ MESSAGE COMPOSITION RULES ═══
1. Open with owner's first name or "Dr. [name]" for dentists — NEVER generic "Hi there" or "Hello"
2. State WHY you're messaging NOW in the first sentence (the trigger reason)
3. Include 2+ specific, verifiable numbers from the provided context data
4. End with EXACTLY ONE call-to-action — never multiple
5. For action triggers: binary CTA (Reply YES / Reply CONFIRM / Reply 1 for X)
6. For information triggers: open-ended question (Want me to...? / Shall I...?)
7. NEVER include URLs of any kind in the message body
8. NEVER repeat body text already sent in this conversation
9. Keep under 250 words total
10. No long preambles ("I hope you're doing well, I'm reaching out today to...")
11. No re-introducing yourself after the first message

═══ COMPULSION LEVERS (use 2–3 per message) ═══
PRIORITY 1 — Specificity (ALWAYS): numbers, dates, source citations, verifiable facts
PRIORITY 2 — Social proof: "N other {category} in your area did Y" (when peer_stats available)
PRIORITY 3 — Loss aversion: "X customers haven't returned in 180+ days"
PRIORITY 4 — Effort externalization: "I've already drafted this — just say go"
PRIORITY 5 — Asking the merchant: "What's your most-asked {service} this week?"
PRIORITY 6 — Curiosity: "Want to see the full breakdown?" / "Want me to show who?"
PRIORITY 7 — Reciprocity: "I noticed Y about your account, thought you'd want to know"
PRIORITY 8 — Single binary commitment: "Reply YES" or "Reply STOP"

═══ ANTI-PATTERNS (will lose points) ═══
✗ "Grow your business" / "Increase your sales" / "Amazing opportunity"
✗ URLs of any kind
✗ Generic percentage discounts ("Flat 30% off") — use service+price instead
✗ Multiple CTAs ("Reply YES for X, NO for Y, MAYBE for Z")
✗ Long preambles or pleasantries
✗ Fabricating data not in the provided context
✗ Exposing internal jargon to the merchant

═══ OUTPUT FORMAT ═══
Output ONLY valid JSON. No explanation. No markdown. No text before or after the JSON.

{
  "body": "<the WhatsApp message — plain text, no markdown, no URLs>",
  "cta": "<open_ended | binary_yes_no | binary_confirm_cancel | multi_choice_slot | none>",
  "send_as": "<vera | merchant_on_behalf>",
  "template_name": "<vera_{trigger_kind}_v1 or merchant_{trigger_kind}_v1>",
  "template_params": ["<owner_name_or_merchant_name>", "<key_fact>", "<cta_phrase>"],
  "rationale": "<1-2 sentences: what trigger prompted this, what compulsion levers used, why this specific merchant>"
}"""


# ══════════════════════════════════════════════════════════════
# HELPER UTILITIES
# ══════════════════════════════════════════════════════════════

def _language_instruction(merchant: MerchantContext) -> str:
    langs = [l.lower() for l in merchant.identity.languages]
    pref = ""
    for h in merchant.conversation_history:
        if h.get("from", "") == "merchant" and h.get("body"):
            # If prior merchant message was in Hindi, use hi-en mix
            body = h.get("body", "")
            if any(c > "\u0900" for c in body):
                pref = "hi-en mix"
                break

    if "hi" in langs or pref == "hi-en mix":
        return "LANGUAGE: Use Hindi-English code-mix naturally (like real WhatsApp messages between Indian professionals)."
    if "te" in langs:
        return "LANGUAGE: Use Telugu-English mix where natural."
    if "ta" in langs:
        return "LANGUAGE: Use Tamil-English mix where natural."
    if "kn" in langs:
        return "LANGUAGE: Use Kannada-English mix where natural."
    return "LANGUAGE: Use clear English."


def _active_offers(merchant: MerchantContext) -> str:
    active = [o.title for o in merchant.offers if o.status == "active"]
    return ", ".join(active) if active else "None currently active"


def _peer_ctr_comparison(merchant: MerchantContext, category: CategoryContext) -> str:
    m_ctr = merchant.performance.ctr
    p_ctr = category.peer_stats.avg_ctr
    if p_ctr > 0:
        gap = round((p_ctr - m_ctr) / p_ctr * 100, 0)
        direction = "below" if m_ctr < p_ctr else "above"
        return (
            f"CTR: {m_ctr:.1%} (peer median {p_ctr:.1%}, "
            f"{abs(gap):.0f}% {direction} peer median)"
        )
    return f"CTR: {m_ctr:.1%}"


def _top_digest_item(category: CategoryContext, top_item_id: Optional[str] = None) -> Optional[Dict]:
    if not category.digest:
        return None
    if top_item_id:
        for d in category.digest:
            if d.id == top_item_id:
                return d.model_dump()
    return category.digest[0].model_dump()


def _owner_salutation(merchant: MerchantContext, category_slug: str) -> str:
    name = merchant.identity.owner_first_name or merchant.identity.name.split()[0]
    if category_slug == "dentists":
        return f"Dr. {name}"
    return name


def _suppress_key_fallback(trigger: TriggerContext) -> str:
    if trigger.suppression_key:
        return trigger.suppression_key
    return f"{trigger.kind}:{trigger.merchant_id}:{trigger.id[:8]}"


# ══════════════════════════════════════════════════════════════
# BASE PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

class PromptBuilder(ABC):
    """Abstract base for all trigger-kind prompt builders."""

    @abstractmethod
    def build_user_prompt(
        self,
        trigger: TriggerContext,
        merchant: MerchantContext,
        category: CategoryContext,
        signals: ExtractedSignals,
        customer: Optional[CustomerContext] = None,
    ) -> str:
        """Return the user-role prompt for LLM composition."""

    def template_name(self, trigger: TriggerContext) -> str:
        return f"vera_{trigger.kind}_v1"

    def send_as(self, trigger: TriggerContext) -> str:
        return "vera"


# ══════════════════════════════════════════════════════════════
# TRIGGER-KIND BUILDERS
# ══════════════════════════════════════════════════════════════

class ResearchDigestBuilder(PromptBuilder):
    """
    For: research_digest, regulation_change
    Hook: journal/regulatory finding relevant to merchant's patient cohort
    Levers: specificity (trial_n, %), source citation, effort externalization, curiosity
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        top_item_id = payload.get("top_item_id")
        digest = _top_digest_item(category, top_item_id)
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        digest_block = ""
        if digest:
            digest_block = f"""
DIGEST ITEM TO COMMUNICATE:
- Title: {digest.get("title", "")}
- Source: {digest.get("source", "")}
- Trial size: {digest.get("trial_n", "N/A")} participants
- Patient segment: {digest.get("patient_segment", "")}
- Summary: {digest.get("summary", "")}
- Actionable insight: {digest.get("actionable", "")}
"""

        cohort_note = ""
        if signals.high_risk_cohort and signals.high_risk_count:
            cohort_note = f"\nIMPORTANT: This merchant has {signals.high_risk_count} high-risk adult patients in their roster — directly relevant to this digest item."
        elif signals.total_customers:
            cohort_note = f"\nMerchant has {signals.total_customers} unique patients YTD."

        return f"""TASK: Compose a WhatsApp message from Vera to {salutation} about a new research/compliance digest item.

{lang}

MERCHANT PROFILE:
- Salutation: {salutation}
- Business: {merchant.identity.name}, {merchant.identity.locality}, {merchant.identity.city}
- {_peer_ctr_comparison(merchant, category)}
- Signals: {", ".join(merchant.signals) or "none"}
- Active offers: {_active_offers(merchant)}
- Languages: {merchant.identity.languages}
{cohort_note}

{digest_block}
CATEGORY VOICE: {category.voice.tone}
Taboo words to avoid: {", ".join(category.voice.vocab_taboo[:5])}
Professional journals: {", ".join(category.professional_journals)}

INSTRUCTIONS:
1. Open with the journal/source name as the hook — this IS the reason you're messaging
2. Connect the finding to their specific patient cohort or merchant signal
3. Use the exact trial_n and percentage from the digest
4. Cite the source at the very end (after a dash: "— JIDA Oct 2026 p.14")
5. Offer to pull the abstract AND draft patient-education content for them
6. CTA must be open_ended ("Want me to pull it + draft a patient-ed WhatsApp?")

GOOD EXAMPLE (50/50 score):
"Dr. Meera, JIDA's Oct issue landed. One item relevant to your high-risk adult patients — 2,100-patient trial showed 3-month fluoride recall cuts caries recurrence 38% better than 6-month. Worth a look (2-min abstract). Want me to pull it + draft a patient-ed WhatsApp you can share? — JIDA Oct 2026 p.14"

BAD (do NOT do this):
"Hi! I wanted to share some exciting research with you today about dental health opportunities..."

Now compose the message for {salutation}:"""


class RegulationBuilder(PromptBuilder):
    """
    For: regulation_change
    Hook: urgency of compliance deadline
    Levers: specificity (deadline, batch/rule numbers), loss aversion, effort externalization
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        top_item_id = payload.get("top_item_id")
        digest = _top_digest_item(category, top_item_id)
        salutation = _owner_salutation(merchant, category.slug)
        deadline = payload.get("deadline_iso", "")
        lang = _language_instruction(merchant)

        digest_block = ""
        if digest:
            digest_block = f"""
REGULATION DETAILS:
- Title: {digest.get("title", "")}
- Source: {digest.get("source", "")}
- Summary: {digest.get("summary", "")}
- Required action: {digest.get("actionable", "")}
- Deadline: {deadline or "check source for deadline"}
"""

        return f"""TASK: Compose an URGENT compliance alert from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.city}
CATEGORY: {category.slug} | Voice: {category.voice.tone}

{digest_block}

INSTRUCTIONS:
1. Open with "Urgent:" or "Quick compliance heads-up:" — this is urgency-4 trigger
2. State the regulatory body and what changed
3. Give the deadline if available
4. Tell them exactly what they need to check/do
5. Offer to draft the compliance checklist for them
6. CTA: binary_yes_no ("Want me to draft a checklist? Reply YES")

GOOD EXAMPLE:
"Ramesh, urgent: voluntary recall on 2 atorvastatin batches (AT2024-1102, AT2024-1108) by Mfr Z — sub-potency, no safety risk, but customers should be informed. Pulled your repeat-Rx list: 22 of your chronic-Rx customers were dispensed these batches in last 90 days. Want me to draft their WhatsApp note + the replacement-pickup workflow?"

Compose for {salutation}:"""


class FestivalBuilder(PromptBuilder):
    """
    For: festival_upcoming, local_news_event, weather_heatwave
    Hook: time-sensitive external event = opportunity or risk
    Levers: specificity (days until, event details), effort externalization
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        festival = payload.get("festival", payload.get("event", "upcoming event"))
        days_until = payload.get("days_until", "")
        date_str = payload.get("date", "")
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        days_note = f"{days_until} days away" if days_until else f"on {date_str}" if date_str else "coming up"

        return f"""TASK: Compose a festival/event awareness message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}, {merchant.identity.city}
EVENT: {festival} ({days_note})
CATEGORY: {category.slug} | Voice: {category.voice.tone}
Active offers: {_active_offers(merchant)}
Signals: {", ".join(merchant.signals) or "none"}
{_peer_ctr_comparison(merchant, category)}

INSTRUCTIONS:
1. Open with the festival/event as the hook
2. Give a SPECIFIC recommendation tied to the category (salons → bridal/party season, restaurants → group bookings, gyms → new-year resolution surge)
3. Reference their existing active offer if relevant, or recommend creating one
4. Offer to draft the post/campaign for them
5. CTA: binary_yes_no ("Want me to draft it? Reply YES")
6. If festival is far away (>60 days), frame it as "perfect planning window"
7. If close (<7 days), frame as urgency ("before the weekend rush")

Compose for {salutation}:"""


class PerfDipBuilder(PromptBuilder):
    """
    For: perf_dip, seasonal_perf_dip
    Hook: metric dropped — but add context (seasonal? or genuine problem?)
    Levers: specificity (exact numbers), loss aversion OR anxiety pre-emption, action proposal
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        metric = payload.get("metric", "calls")
        delta_pct = payload.get("delta_pct", signals.calls_delta_7d)
        delta_display = f"{abs(delta_pct * 100):.0f}%" if delta_pct else "significantly"
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)
        is_seasonal = trigger.kind == "seasonal_perf_dip"

        seasonal_note = ""
        if is_seasonal:
            # Find matching seasonal beat
            from datetime import datetime
            month = datetime.now().strftime("%b")
            for beat in category.seasonal_beats:
                if month.lower() in beat.month_range.lower():
                    seasonal_note = f"\nSEASONAL CONTEXT: {beat.note} — this dip is EXPECTED, not alarming."
                    break

        return f"""TASK: Compose a performance dip message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}
PERFORMANCE DROP: {metric} down {delta_display} in last 7 days
Current {metric}: {getattr(merchant.performance, metric, "N/A")}
{_peer_ctr_comparison(merchant, category)}
Active members/customers: {signals.total_customers}
Signals: {", ".join(merchant.signals) or "none"}
{seasonal_note}

INSTRUCTIONS:
{"1. REFRAME this as normal/expected before suggesting action — don't alarm them needlessly" if is_seasonal else "1. Acknowledge the dip with the specific number"}
2. Give peer benchmark context (what's normal for {category.slug} in {merchant.identity.city})
3. Suggest a SPECIFIC action — not generic "improve your profile"
4. If seasonal: recommend saving ad spend for high-conversion months + retention play now
5. If genuine dip: identify the most likely cause from their signals and suggest fix
6. Offer to draft something concrete (post, campaign, reminder)
7. CTA: open_ended

GOOD EXAMPLE (seasonal, gym):
"Karthik, your views are down 30% this week — but I want to flag this is the normal April-June acquisition lull (every metro gym sees -25 to -35% in this window). Action: skip ad spend now, save it for Sept-Oct when conversion is 2x. For now, focus retention on your 245 members. Want me to draft a 'summer attendance challenge' to keep them through the dip?"

Compose for {salutation}:"""


class PerfSpikeBuilder(PromptBuilder):
    """
    For: perf_spike
    Hook: views/calls jumped — Vera spotted it first
    Levers: reciprocity (I noticed), curiosity, social proof, effort externalization
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        metric = payload.get("metric", "views")
        delta_pct = payload.get("delta_pct", signals.views_delta_7d)
        delta_display = f"+{abs(delta_pct * 100):.0f}%" if delta_pct else "+significantly"
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        return f"""TASK: Compose a performance spike celebration + action message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}
METRIC SPIKE: {metric} up {delta_display} yesterday vs 7-day average
Current performance: views={merchant.performance.views}, calls={merchant.performance.calls}
{_peer_ctr_comparison(merchant, category)}
Signals: {", ".join(merchant.signals) or "none"}

INSTRUCTIONS:
1. Open with the specific spike number — "I spotted X" not "I noticed an improvement"
2. Offer a hypothesis for WHY it spiked (connect to their recent activity if any)
3. Suggest how to capitalize on this momentum (specific action)
4. Offer to draft the follow-on content/post
5. CTA: open_ended

Compose for {salutation}:"""


class RecallBuilder(PromptBuilder):
    """
    For: recall_due
    Hook: customer's recall window opened — send from merchant's number
    Levers: personalization, specific slots, price, language match, effort externalization
    """

    def send_as(self, trigger):
        return "merchant_on_behalf"

    def template_name(self, trigger):
        return "merchant_recall_reminder_v1"

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        service_due = payload.get("service_due", "checkup")
        last_service_date = payload.get("last_service_date", "")
        slots = payload.get("available_slots", [])
        salutation = _owner_salutation(merchant, category.slug)

        customer_name = customer.identity.name if customer else "the patient"
        cust_lang = customer.identity.language_pref if customer else "english"
        cust_pref = customer.preferences.preferred_slots if customer else ""
        cust_state = customer.state if customer else "active"

        # Calculate time since last visit
        months_since = ""
        if last_service_date:
            try:
                from datetime import datetime
                last = datetime.fromisoformat(last_service_date.replace("Z", "+00:00"))
                now = datetime.utcnow()
                months = round((now - last.replace(tzinfo=None)).days / 30)
                months_since = f"{months} months"
            except Exception:
                months_since = "some time"

        slots_str = ""
        if slots:
            slot_labels = [s.get("label", s.get("iso", "")) for s in slots[:2]]
            slots_str = " or ".join(slot_labels)

        active_offer = next((o.title for o in merchant.offers if o.status == "active"), "")

        return f"""TASK: Compose a customer recall reminder from {salutation}'s clinic/business to {customer_name}.
This is sent AS THE MERCHANT (send_as: merchant_on_behalf) — NOT as Vera.

CUSTOMER LANGUAGE PREFERENCE: {cust_lang}
If hi-en mix: use Hindi-English naturally. If english: use clean English.

CUSTOMER: {customer_name}
Service due: {service_due.replace("_", " ")}
Last visit: {months_since} ago
Customer state: {cust_state}
Preferred slots: {cust_pref}
Available slots: {slots_str or "please check with the clinic"}

MERCHANT SENDING THIS:
Business: {merchant.identity.name}
Owner: {salutation}
Active offer: {active_offer or "none"}
Category: {category.slug}

INSTRUCTIONS:
1. Address {customer_name} by name with a warm greeting
2. Identify the clinic/business ("Dr. Meera's clinic here" or "PowerHouse Fitness here")
3. State how long it has been since their last visit ({months_since})
4. Name the specific service that's due
5. Offer the specific time slots (if available) using customer's preferred time
6. Include the active offer price if applicable + any bonus ("complimentary fluoride")
7. CTA: multi_choice_slot (Reply 1 for first slot, Reply 2 for second, or tell us a time)
8. Match language preference EXACTLY

GOOD EXAMPLE (dentist, hi-en mix, lapsed_soft, 50/50 score):
"Hi Priya, Dr. Meera's clinic here 🦷 It's been 5 months since your last visit — your 6-month cleaning recall is due. Apke liye 2 slots ready hain: Wed 5 Nov, 6pm ya Thu 6 Nov, 5pm. ₹299 cleaning + complimentary fluoride. Reply 1 for Wed, 2 for Thu, or tell us a time that works."

Compose for {customer_name}:"""


class WinbackBuilder(PromptBuilder):
    """
    For: customer_lapsed_soft, customer_lapsed_hard
    Hook: customer hasn't come back — reach out warmly
    Levers: no-shame, goal-match, specific offer, no-commitment trial
    """

    def send_as(self, trigger):
        return "merchant_on_behalf"

    def template_name(self, trigger):
        return "merchant_winback_v1"

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        salutation = _owner_salutation(merchant, category.slug)
        customer_name = customer.identity.name if customer else "the customer"
        cust_lang = customer.identity.language_pref if customer else "english"
        cust_pref = customer.preferences.preferred_slots if customer else ""
        training_focus = customer.preferences.training_focus if customer else ""
        health_focus = customer.preferences.health_focus if customer else ""
        services = customer.relationship.services_received[-3:] if customer else []
        lapse_kind = "soft" if "soft" in trigger.kind else "hard"
        last_visit = customer.relationship.last_visit if customer else ""

        weeks_lapsed = ""
        if last_visit:
            try:
                from datetime import datetime
                last = datetime.fromisoformat(last_visit.replace("Z", "+00:00"))
                weeks = round((datetime.utcnow() - last.replace(tzinfo=None)).days / 7)
                weeks_lapsed = f"{weeks} weeks"
            except Exception:
                weeks_lapsed = "some weeks"

        active_offer = next((o.title for o in merchant.offers if o.status == "active"), "")
        goal_focus = training_focus or health_focus or (services[0].replace("_", " ") if services else "")

        return f"""TASK: Compose a warm winback message from {salutation} to {customer_name}.
Send AS THE MERCHANT (send_as: merchant_on_behalf).

CUSTOMER LANGUAGE: {cust_lang}
CUSTOMER: {customer_name}, lapsed {lapse_kind} ({weeks_lapsed} since last visit)
Previous services: {", ".join(services) if services else "N/A"}
Customer's focus/goal: {goal_focus or "N/A"}
Preferred time: {cust_pref}

MERCHANT: {merchant.identity.name} | Owner: {salutation}
Active offer: {active_offer or "none"}

INSTRUCTIONS:
1. NO shame, NO guilt-tripping — normalize the lapse ("happens to most members at some point")
2. Use owner first name to make it personal ("Karthik from PowerHouse here")
3. Reference a SPECIFIC new thing that matches their previous goal/focus
4. Mention a specific time slot for a no-commitment trial
5. CTA: binary_yes_no with explicit "no commitment, no auto-charge" if gym/recurring
6. Keep it warm and low-pressure

GOOD EXAMPLE (gym, lapsed_hard, english, 50/50 score):
"Hi Rashmi 👋 Karthik from PowerHouse here. It's been about 8 weeks — happens to most members at some point, no judgment. We've added a Tue/Thu evening HIIT class that fits weight-loss goals well (45 min, 6:30pm). Want me to hold a free trial spot for you next Tue, 30 Apr? Reply YES — no commitment, no auto-charge."

Compose for {customer_name}:"""


class CuriousAskBuilder(PromptBuilder):
    """
    For: curious_ask_due, active_planning_intent (if exploratory)
    Hook: ask the merchant a question → use answer to create something for them
    Levers: asking the merchant, reciprocity (promise upfront), effort externalization
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        ask_template = payload.get("ask_template", "what_service_in_demand_this_week")
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        # Map ask_template to a category-appropriate question
        category_questions = {
            "dentists": "which treatment has been most requested this week — whitening, aligners, or scaling?",
            "salons": "what service has been most asked-for this week at your salon?",
            "restaurants": "what dish is getting the most orders or calls this week?",
            "gyms": "which class or program are members asking about most this month?",
            "pharmacies": "which OTC product or category has been most asked about this week?",
        }
        question = category_questions.get(category.slug, "what's been most popular with your customers this week?")

        # What Vera will create with the answer
        deliverable_map = {
            "dentists": "a Google post + a patient-ed WhatsApp reply for that treatment",
            "salons": "a Google post + a 4-line WhatsApp reply for pricing questions",
            "restaurants": "a Google post + an Insta story",
            "gyms": "a Google post + a member WhatsApp announcement",
            "pharmacies": "a Google post + a seasonal health tip card",
        }
        deliverable = deliverable_map.get(category.slug, "a Google post + customer-facing content")

        return f"""TASK: Compose a curiosity-ask message from Vera to {salutation}.
This is the "asking the merchant" engagement family — low-stakes question, high-value offer.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}
Signals: {", ".join(merchant.signals) or "none"}
Active offers: {_active_offers(merchant)}

INSTRUCTIONS:
1. Open with {salutation}'s name warmly — NO preamble
2. Ask ONE specific question about their business this week: "{question}"
3. Immediately offer what YOU will do with their answer: "{deliverable}"
4. Time-box the effort: "Takes 5 min" or "I'll have it ready in 10 min"
5. CTA: open_ended (their answer IS the reply)
6. Keep it short — this should feel like a quick WhatsApp from a smart colleague

GOOD EXAMPLE (salon, 44/50 score):
"Hi Lakshmi! Quick check — what service has been most asked-for this week at Studio11? I'll turn the answer into a Google post + a 4-line WhatsApp reply you can use when customers ask about pricing. Takes 5 min."

Compose for {salutation}:"""


class RenewalBuilder(PromptBuilder):
    """
    For: renewal_due
    Hook: subscription expiring — frame as loss + value recap
    Levers: loss aversion, specificity (days remaining, plan value), effort externalization
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        days_remaining = payload.get("days_remaining", merchant.subscription.days_remaining)
        plan = payload.get("plan", merchant.subscription.plan)
        renewal_amount = payload.get("renewal_amount", "")
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        # Compute value Vera delivered
        value_stats = []
        if merchant.performance.views > 0:
            value_stats.append(f"{merchant.performance.views} profile views")
        if merchant.performance.calls > 0:
            value_stats.append(f"{merchant.performance.calls} calls generated")
        if signals.total_customers > 0:
            value_stats.append(f"{signals.total_customers} unique customers")

        value_summary = f"({', '.join(value_stats)} this month)" if value_stats else ""

        return f"""TASK: Compose a subscription renewal message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}
Plan: {plan} | Days remaining: {days_remaining}
Renewal amount: {f'₹{renewal_amount}' if renewal_amount else 'check your plan'}
Performance last 30d {value_summary}

INSTRUCTIONS:
1. Lead with the urgency: "{days_remaining} days left on your {plan} plan"
2. Remind them of the SPECIFIC value they've gotten (views, calls, customers from their data)
3. If days_remaining < 7, use strong loss framing ("your listing drops in X days")
4. If days_remaining 7-14, use moderate framing ("worth renewing before the window closes")
5. Offer to handle the renewal paperwork/process for them
6. CTA: binary_yes_no ("Reply YES to renew, I'll take care of the rest")

Compose for {salutation}:"""


class ReviewBuilder(PromptBuilder):
    """
    For: review_theme_emerged
    Hook: review pattern emerged — actionable insight for the merchant
    Levers: specificity (count, quote), social proof, effort externalization
    """

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        theme = payload.get("theme", signals.negative_review_themes[0] if signals.negative_review_themes else "service")
        count = payload.get("count", 3)
        sentiment = payload.get("sentiment", "neg")
        quote = payload.get("sample_quote", "")

        # Find matching review theme from merchant
        for rt in merchant.review_themes:
            if rt.theme == theme:
                count = rt.occurrences_30d
                quote = rt.common_quote
                sentiment = rt.sentiment
                break

        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)
        pos_themes = ", ".join(signals.positive_review_themes) if signals.positive_review_themes else ""

        return f"""TASK: Compose a review-theme awareness message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}
REVIEW PATTERN: {count} reviews this month mention "{theme}" ({sentiment} sentiment)
Sample quote: "{quote}"
Positive themes also found: {pos_themes or "none noted"}

INSTRUCTIONS:
1. Lead with the specific observation: "{count} reviews this month mention {theme}"
2. Quote or paraphrase the sample quote briefly
3. If sentiment is negative: suggest a specific operational fix + offer to draft a Google reply
4. If sentiment is positive: suggest amplifying it (post, patient content)
5. Offer to draft the response/content for them
6. CTA: open_ended

Compose for {salutation}:"""


class MilestoneBuilder(PromptBuilder):
    """For: milestone_reached — celebrate + momentum ride"""

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        milestone = payload.get("milestone", "100 reviews")
        metric = payload.get("metric", "reviews")
        value = payload.get("value", "100")
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        return f"""TASK: Compose a milestone celebration + momentum message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}
MILESTONE ACHIEVED: {milestone} ({metric} = {value})
Current performance: views={merchant.performance.views}, calls={merchant.performance.calls}, CTR={merchant.performance.ctr:.1%}

INSTRUCTIONS:
1. Celebrate specifically: "{value} {metric}!" — not generic "congratulations"
2. Put it in peer context ("top {round(100 - signals.ctr_ratio * 100):.0f}% of {category.slug} in {merchant.identity.city}")
3. Suggest the natural next action (next milestone, using the reviews to attract more)
4. Offer to create a "thank your customers" post
5. CTA: open_ended

Compose for {salutation}:"""


class DormancyBuilder(PromptBuilder):
    """For: dormant_with_vera — re-engage a merchant who went quiet"""

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        days_dormant = payload.get("days_dormant", signals.days_since_vera_touch or 14)
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        # What's notable about this merchant right now
        notable = []
        if signals.stale_posts:
            notable.append(f"no new Google posts in {signals.days_since_post} days")
        if signals.ctr_below_peer:
            notable.append(f"CTR {merchant.performance.ctr:.1%} vs peer median {category.peer_stats.avg_ctr:.1%}")
        if signals.lapsed_count > 20:
            notable.append(f"{signals.lapsed_count} lapsed customers not yet re-engaged")

        return f"""TASK: Compose a re-engagement message from Vera to {salutation} who has been quiet for {days_dormant} days.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}
Days without Vera engagement: {days_dormant}
Notable signals: {", ".join(notable) if notable else "profile needs attention"}
Active offers: {_active_offers(merchant)}
CTR vs peer: {_peer_ctr_comparison(merchant, category)}

INSTRUCTIONS:
1. Don't mention "you haven't replied in X days" — that sounds accusatory
2. Come in with a FRESH hook — something new and valuable
3. Lead with the most actionable signal (stale posts, CTR gap, lapsed customers)
4. Make it feel like Vera just spotted something worth sharing — not a reminder
5. CTA: open_ended
6. Short — re-engagement needs to feel light-touch

Compose for {salutation}:"""


class CompetitorBuilder(PromptBuilder):
    """For: competitor_opened — voyeur curiosity + defense framing"""

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        distance = payload.get("distance_km", payload.get("distance", ""))
        competitor_name = payload.get("competitor_name", "a new competitor")
        platform = payload.get("platform", "Google Business Profile")
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        distance_str = f"{distance}km away" if distance else "nearby"

        return f"""TASK: Compose a competitive awareness message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}
NEW COMPETITOR: {competitor_name} opened {distance_str} on {platform}
Merchant's current position: {_peer_ctr_comparison(merchant, category)}
Signals: {", ".join(merchant.signals) or "none"}

INSTRUCTIONS:
1. Use curiosity framing: "A new {category.slug.rstrip('s')} opened {distance_str} from you on GBP..."
2. Don't alarm — frame as "worth knowing" not "emergency"
3. Suggest a specific defensive action (update profile, add photos, respond to reviews)
4. Their {merchant.performance.views} views vs what a fresh competitor might attract
5. Offer to audit their GBP strength vs the new competitor
6. CTA: open_ended ("Want me to do a quick comparison? Reply YES")

Compose for {salutation}:"""


class BridalBuilder(PromptBuilder):
    """For: wedding_package_followup"""

    def send_as(self, trigger):
        return "merchant_on_behalf"

    def template_name(self, trigger):
        return "merchant_bridal_followup_v1"

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        wedding_date = payload.get("wedding_date", customer.preferences.wedding_date if customer else "")
        days_to_wedding = payload.get("days_to_wedding", "")
        trial_completed = payload.get("trial_completed", "")
        next_window = payload.get("next_step_window_open", "")
        salutation = _owner_salutation(merchant, category.slug)

        customer_name = customer.identity.name if customer else "the bride"
        cust_lang = customer.identity.language_pref if customer else "english"
        cust_pref = customer.preferences.preferred_slots if customer else ""

        active_offer = next((o.title for o in merchant.offers if o.status == "active"), "")

        return f"""TASK: Compose a bridal package follow-up from {salutation}'s salon to {customer_name}.
Send AS THE MERCHANT (send_as: merchant_on_behalf).

CUSTOMER LANGUAGE: {cust_lang}
CUSTOMER: {customer_name}
Wedding date: {wedding_date} ({days_to_wedding} days away)
Trial completed: {trial_completed}
Next recommended step: {next_window.replace("_", " ") if next_window else "booking pre-bridal treatments"}
Preferred slot: {cust_pref}

MERCHANT: {merchant.identity.name} | Owner: {salutation}
Active offer: {active_offer or "standard bridal packages"}

INSTRUCTIONS:
1. Open with {customer_name}'s name + wedding day count as excitement hook
2. Reference the trial they completed at the salon (continuity + relationship)
3. Frame the current window as critical for timely results
4. Name the specific program/treatment and price
5. Offer to block her preferred slot immediately
6. CTA: binary_yes_no

GOOD EXAMPLE (47/50 score):
"Hi Kavya 💍 Lakshmi from Studio11 Kapra here. 196 days to your wedding — perfect window to start the 30-day skin-prep program before serious bridal bookings roll in. ₹2,499 covers 4 sessions + a take-home kit. Want me to block your preferred Saturday 4pm slot for the first session next week?"

Compose for {customer_name}:"""


class RefillBuilder(PromptBuilder):
    """For: chronic_refill_due"""

    def send_as(self, trigger):
        return "merchant_on_behalf"

    def template_name(self, trigger):
        return "merchant_refill_reminder_v1"

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        medicines = payload.get("medicines", customer.relationship.chronic_conditions if customer else [])
        runout_date = payload.get("runout_date", "")
        salutation = _owner_salutation(merchant, category.slug)

        customer_name = customer.identity.name if customer else "the patient"
        cust_lang = customer.identity.language_pref if customer else "hi-en mix"
        senior = customer.identity.senior_citizen if customer else False
        channel = customer.preferences.channel if customer else "whatsapp"

        active_offer = next((o.title for o in merchant.offers if o.status == "active"), "")
        # Extract medicines from chronic_conditions or payload
        meds_list = medicines if isinstance(medicines, list) else [medicines]
        meds_str = ", ".join(meds_list[:3]) if meds_list else "regular medicines"

        # via_son flag
        via_son = "son" in (channel or "")

        return f"""TASK: Compose a chronic medicine refill reminder from {merchant.identity.name} to {customer_name}.
Send AS THE MERCHANT (send_as: merchant_on_behalf).
{"Address via son's WhatsApp" if via_son else ""}

CUSTOMER LANGUAGE: {cust_lang}
PATIENT: {customer_name} {"(senior citizen)" if senior else ""}
Medicines due for refill: {meds_str}
Runout date: {runout_date}
Channel: {channel}

PHARMACY: {merchant.identity.name}, {merchant.identity.locality}
Active offer: {active_offer or "none"}

INSTRUCTIONS:
1. {"Use Namaste greeting and respectful tone for senior" if senior else "Use warm, professional greeting"}
2. {"Address the son/caregiver, not the patient directly" if via_son else "Address the patient"}
3. Name the specific medicines (use molecule/brand names from the provided list)
4. State the exact runout date
5. If pharmacy has senior discount: show total + savings clearly
6. Offer home delivery with timeline
7. CTA: binary_confirm_cancel ("Reply CONFIRM to dispatch")
8. Include a call option for dose changes

GOOD EXAMPLE (pharmacy, senior, via son, 49/50):
"Namaste — Apollo Health Plus Malviya Nagar yahan. Sharma ji ki 3 monthly medicines (metformin, atorvastatin, telmisartan) 28 April ko khatam hongi. Same dose, same brand pack ready hai. Senior discount 15% applied — total ₹1,420 (₹240 saved). Free home delivery to saved address by 5pm tomorrow. Reply CONFIRM to dispatch, or call 9876543210 if any change in dosage."

Compose for {customer_name}:"""


class PlanningBuilder(PromptBuilder):
    """For: active_planning_intent — merchant said yes, switch to execution"""

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        intent = payload.get("intent", "campaign planning")
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        last_intent = signals.last_merchant_intent or "expressed interest in a campaign"

        return f"""TASK: Compose an execution/planning message from Vera to {salutation} who has shown active planning intent.
The merchant has signaled they want to proceed — switch to ACTION mode immediately.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}
Merchant's expressed intent: "{last_intent}"
Active offers: {_active_offers(merchant)}
Performance: views={merchant.performance.views}, calls={merchant.performance.calls}

INSTRUCTIONS:
1. Switch to action mode IMMEDIATELY — no more qualifying questions
2. Present a concrete starter plan or draft artifact (price tier, template, schedule)
3. Include 3-4 specific details the merchant can immediately react to (prices, dates, volumes)
4. Reference their locality-specific opportunity (corporate offices nearby, local competitors, etc.)
5. Offer to handle outreach for them
6. CTA: binary_confirm_cancel ("Reply CONFIRM to proceed, or tell me what to change")

GOOD EXAMPLE (restaurant, planning, 49/50):
"Suresh, here's a starter version — you can edit:
Mylari Corporate Thali — for offices in Indiranagar
- 10 thalis @ ₹125 each (₹25 off retail) + free delivery
- 25 thalis @ ₹115 each + 2 free filter coffees
- 50+: ₹105 each + 1 free dosa platter
WhatsApp the day-before by 5pm; we deliver 12:30-1pm
3 offices in Indiranagar are in your delivery radius. Want me to draft a 3-line WhatsApp for their facilities managers?"

Compose for {salutation}:"""


class WeatherBuilder(PromptBuilder):
    """For: weather_heatwave, local_news_event"""

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        payload = trigger.payload
        event_type = payload.get("event_type", trigger.kind)
        temperature = payload.get("temperature_c", "")
        city = payload.get("city", merchant.identity.city)
        impact = payload.get("impact", "")
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        temp_str = f"{temperature}°C" if temperature else "extreme heat"

        return f"""TASK: Compose a weather/event situational awareness message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.city}
EVENT: {event_type} — {temp_str} in {city}
Expected impact: {impact or "footfall disruption likely"}
Active offers: {_active_offers(merchant)}
Category: {category.slug}

INSTRUCTIONS:
1. Lead with the weather/event fact and specific temperature/detail
2. Give a category-specific recommendation:
   - gyms: people skip outdoor workouts → push indoor classes
   - restaurants: delivery spikes → push delivery offer/Swiggy optimization
   - pharmacies: heat-related health products (ORS, sunscreen) demand rises
   - salons: less foot traffic → push appointment booking
   - dentists: low footfall → push calls for existing patients
3. Offer to draft the specific content
4. CTA: binary_yes_no

Compose for {salutation}:"""


class DefaultBuilder(PromptBuilder):
    """Fallback for unknown trigger kinds."""

    def build_user_prompt(self, trigger, merchant, category, signals, customer=None):
        salutation = _owner_salutation(merchant, category.slug)
        lang = _language_instruction(merchant)

        return f"""TASK: Compose a relevant WhatsApp message from Vera to {salutation}.

{lang}

MERCHANT: {salutation}, {merchant.identity.name}, {merchant.identity.locality}, {merchant.identity.city}
Trigger kind: {trigger.kind}
Trigger payload: {json.dumps(trigger.payload)}
Performance: views={merchant.performance.views}, calls={merchant.performance.calls}, CTR={merchant.performance.ctr:.1%}
Signals: {", ".join(merchant.signals) or "none"}
Active offers: {_active_offers(merchant)}
Category: {category.slug} | Voice: {category.voice.tone}

INSTRUCTIONS:
1. Use the most actionable data point from the merchant's signals or performance
2. Connect it to something the merchant can act on today
3. Offer to do the work for them
4. Single clear CTA

Compose for {salutation}:"""


# ══════════════════════════════════════════════════════════════
# PROMPT DISPATCHER
# ══════════════════════════════════════════════════════════════

_TRIGGER_KIND_MAP: Dict[str, PromptBuilder] = {
    "research_digest": ResearchDigestBuilder(),
    "regulation_change": RegulationBuilder(),
    "festival_upcoming": FestivalBuilder(),
    "weather_heatwave": WeatherBuilder(),
    "local_news_event": WeatherBuilder(),
    "perf_dip": PerfDipBuilder(),
    "perf_spike": PerfSpikeBuilder(),
    "seasonal_perf_dip": PerfDipBuilder(),
    "recall_due": RecallBuilder(),
    "customer_lapsed_soft": WinbackBuilder(),
    "customer_lapsed_hard": WinbackBuilder(),
    "curious_ask_due": CuriousAskBuilder(),
    "renewal_due": RenewalBuilder(),
    "review_theme_emerged": ReviewBuilder(),
    "milestone_reached": MilestoneBuilder(),
    "dormant_with_vera": DormancyBuilder(),
    "competitor_opened": CompetitorBuilder(),
    "wedding_package_followup": BridalBuilder(),
    "chronic_refill_due": RefillBuilder(),
    "active_planning_intent": PlanningBuilder(),
    "appointment_tomorrow": RecallBuilder(),
    "supply_alert": RegulationBuilder(),
    "ipl_match_today": FestivalBuilder(),
}


def get_prompt_builder(trigger_kind: str) -> PromptBuilder:
    """Return the appropriate PromptBuilder for a given trigger kind."""
    return _TRIGGER_KIND_MAP.get(trigger_kind, DefaultBuilder())


# ══════════════════════════════════════════════════════════════
# REPLY PROMPT BUILDER
# ══════════════════════════════════════════════════════════════

REPLY_SYSTEM_PROMPT = """You are Vera, magicpin's AI merchant WhatsApp assistant. You are continuing an ongoing conversation.

CRITICAL RULES:
1. Output ONLY valid JSON — no explanation, no markdown, no preamble
2. If action is "send": include body and cta
3. If action is "wait": include wait_seconds
4. If action is "end": just the action and rationale
5. Body must NOT repeat anything already sent in the conversation
6. Body must NOT contain URLs
7. Match the merchant's language preference
8. For commitment responses: use ACTION words (sending, drafting, creating, publishing, confirming) — NEVER ask another qualifying question

OUTPUT FORMAT:
{
  "action": "send | wait | end",
  "body": "<message if action=send>",
  "cta": "<cta type if action=send>",
  "wait_seconds": <integer if action=wait>,
  "rationale": "<1 sentence: why this action>"
}"""


def build_reply_prompt(
    reply_type: str,
    merchant_message: str,
    conv_summary: str,
    merchant_name: str,
    category_slug: str,
    language_pref: str,
    trigger_summary: str,
    sent_bodies: List[str],
) -> str:
    """Build the user prompt for reply composition."""

    already_sent = "\n".join(f"- {b[:80]}..." for b in sent_bodies[-3:]) if sent_bodies else "none"

    if reply_type == "send_auto_reply_notice":
        return f"""CONVERSATION HISTORY:
{conv_summary}

MERCHANT REPLY (auto-reply detected): "{merchant_message}"
REPLY TYPE: auto_reply (first occurrence)

MERCHANT: {merchant_name} | CATEGORY: {category_slug} | LANGUAGE: {language_pref}

TASK: The merchant's WhatsApp auto-reply fired. Send ONE brief message for the OWNER to see when they check their phone.
- Acknowledge you noticed it's an automated reply
- Keep the hook from the original message alive for when the owner sees it
- Short: 1-2 sentences max
- CTA: binary_yes_no (simple "Reply YES when you get a chance")

Respond with JSON:"""

    if reply_type == "send_commitment":
        return f"""CONVERSATION HISTORY:
{conv_summary}

MERCHANT REPLY (commitment): "{merchant_message}"
REPLY TYPE: commitment — merchant said YES / LET'S DO IT

MERCHANT: {merchant_name} | CATEGORY: {category_slug} | LANGUAGE: {language_pref}
Original trigger context: {trigger_summary}

ALREADY SENT IN THIS CONVERSATION (do NOT repeat):
{already_sent}

TASK: Switch IMMEDIATELY to action/execution mode.
RULES:
- Use action words: "Drafting now", "Sending", "Creating", "Publishing", "Here it is"
- State the specific deliverable you are creating
- Include the scope (e.g. "for your 40 high-risk adult patients")
- CTA must be binary_confirm_cancel
- ZERO qualifying questions ("would you like", "do you want to") — they already committed
- Keep under 120 words

Respond with JSON:"""

    if reply_type == "send_off_topic":
        return f"""CONVERSATION HISTORY:
{conv_summary}

MERCHANT REPLY (off-topic): "{merchant_message}"
REPLY TYPE: off_topic

MERCHANT: {merchant_name} | CATEGORY: {category_slug}
Original trigger context: {trigger_summary}

TASK: Politely decline the off-topic request and redirect to the original conversation thread.
- One sentence acknowledgment + decline
- One sentence redirect back to the original CTA
- Keep it friendly, not dismissive
- CTA: open_ended

Respond with JSON:"""

    if reply_type == "send_engaged":
        return f"""CONVERSATION HISTORY:
{conv_summary}

MERCHANT REPLY: "{merchant_message}"
REPLY TYPE: engaged (real reply from merchant)

MERCHANT: {merchant_name} | CATEGORY: {category_slug} | LANGUAGE: {language_pref}
Original trigger context: {trigger_summary}

ALREADY SENT IN THIS CONVERSATION (do NOT repeat these):
{already_sent}

TASK: Advance the conversation toward the original goal.
- Answer the merchant's question/concern concisely
- Keep the original hook alive
- Move toward a concrete commitment
- CTA: open_ended or binary_yes_no
- Under 100 words

Respond with JSON:"""

    # Default engaged
    return f"""Continue this conversation with {merchant_name}.
History: {conv_summary}
Latest reply: "{merchant_message}"
Context: {trigger_summary}
Respond helpfully and advance toward a concrete next step.
Respond with JSON:"""


# ══════════════════════════════════════════════════════════════
# CONVENIENCE EXPORTS
# ══════════════════════════════════════════════════════════════

# Re-export from engine for single-import convenience
from .engine import build_reply_context_summary  # noqa: F401, E402

# Complete list of trigger kinds that have dedicated builders
TRIGGER_KINDS_COVERED: List[str] = list(_TRIGGER_KIND_MAP.keys())
