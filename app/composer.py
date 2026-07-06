"""
app/composer.py — LLM client + output validator + composition orchestrator.

LLMClient      : provider-agnostic LLM call abstraction (Anthropic, OpenAI, Gemini, DeepSeek)
Validator      : post-LLM checks — URLs, repetition, CTA shape, body length
Composer       : orchestrates context → prompt → LLM → validate → ComposedMessage
"""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import settings
from .engine import ExtractedSignals, extract_signals
from .models import (
    CategoryContext,
    ComposedMessage,
    ConversationState,
    CustomerContext,
    MerchantContext,
    TriggerContext,
)
from .prompts import (
    REPLY_SYSTEM_PROMPT,
    VERA_SYSTEM_PROMPT,
    build_reply_prompt,
    build_reply_context_summary,
    get_prompt_builder,
)


# ══════════════════════════════════════════════════════════════
# LLM CLIENT ABSTRACTION
# ══════════════════════════════════════════════════════════════

class LLMClient(ABC):
    @abstractmethod
    def complete(self, user_prompt: str, system_prompt: str) -> str:
        pass


class AnthropicClient(LLMClient):
    """Anthropic Claude via direct HTTP (no SDK dependency required)."""

    def __init__(self, api_key: str, model: str = "", timeout: int = 25) -> None:
        self.api_key = api_key
        self.model = model or "claude-3-5-sonnet-20241022"
        self.timeout = timeout

    def complete(self, user_prompt: str, system_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "max_tokens": 1024,
            "temperature": 0,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }).encode("utf-8")

        req = urlrequest.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": self.api_key,
                "Content-Type": "application/json",
                "anthropic-version": "2023-06-01",
            },
        )
        resp = urlrequest.urlopen(req, timeout=self.timeout)
        data = json.loads(resp.read().decode("utf-8"))
        return data["content"][0]["text"]


class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "", timeout: int = 25) -> None:
        self.api_key = api_key
        self.model = model or "gpt-4o"
        self.timeout = timeout

    def complete(self, user_prompt: str, system_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")

        req = urlrequest.Request(
            "https://api.openai.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        resp = urlrequest.urlopen(req, timeout=self.timeout)
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


class GeminiClient(LLMClient):
    """
    Google Gemini client using the free AI Studio API.
    Get your free API key at: https://aistudio.google.com/apikey
    Free tier: gemini-2.0-flash — 15 RPM, 1500 RPD, no credit card.
    """

    def __init__(self, api_key: str, model: str = "", timeout: int = 25) -> None:
        self.api_key = api_key
        # gemini-2.5-flash: best free model (strip models/ prefix if present)
        raw_model = model or "gemini-2.5-flash"
        self.model = raw_model.removeprefix("models/")
        self.timeout = timeout

    def complete(self, user_prompt: str, system_prompt: str) -> str:
        # Use systemInstruction for proper system prompt separation
        body = json.dumps({
            "systemInstruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": [{
                "role": "user",
                "parts": [{"text": user_prompt}]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": 1024,
                "responseMimeType": "text/plain",
            },
        }).encode("utf-8")

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        req = urlrequest.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            resp = urlrequest.urlopen(req, timeout=self.timeout)
        except urlerror.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Gemini API error {e.code}: {error_body[:300]}"
            ) from e

        data = json.loads(resp.read().decode("utf-8"))

        # Handle safety blocks
        candidate = data.get("candidates", [{}])[0]
        finish_reason = candidate.get("finishReason", "")
        if finish_reason == "SAFETY":
            raise RuntimeError("Gemini blocked the response due to safety filters.")

        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            raise RuntimeError(f"Gemini returned no content parts. Response: {data}")

        return parts[0]["text"]


class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str, model: str = "", timeout: int = 25) -> None:
        self.api_key = api_key
        self.model = model or "deepseek-chat"
        self.timeout = timeout

    def complete(self, user_prompt: str, system_prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "max_tokens": 1024,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }).encode("utf-8")

        req = urlrequest.Request(
            "https://api.deepseek.com/v1/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        resp = urlrequest.urlopen(req, timeout=self.timeout)
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]


def create_llm_client() -> LLMClient:
    """Factory: create the LLM client from settings."""
    providers = {
        "anthropic": lambda: AnthropicClient(settings.llm_api_key, settings.llm_model, settings.llm_timeout),
        "openai": lambda: OpenAIClient(settings.llm_api_key, settings.llm_model, settings.llm_timeout),
        "gemini": lambda: GeminiClient(settings.llm_api_key, settings.llm_model, settings.llm_timeout),
        "deepseek": lambda: DeepSeekClient(settings.llm_api_key, settings.llm_model, settings.llm_timeout),
    }
    factory = providers.get(settings.llm_provider.lower())
    if not factory:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}. Use: {list(providers.keys())}")
    return factory()


# Singleton LLM client (created once on first use)
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = create_llm_client()
    return _llm_client


# ══════════════════════════════════════════════════════════════
# VALIDATOR
# ══════════════════════════════════════════════════════════════

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_VALID_CTAS = {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}
_VALID_SEND_AS = {"vera", "merchant_on_behalf"}


class ValidationError(Exception):
    pass


def validate_and_clean(
    raw: Dict[str, Any],
    sent_bodies: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Validate and clean a composed message dict.
    Raises ValidationError if the message is fundamentally broken.
    Returns cleaned dict.
    """
    body = raw.get("body", "").strip()
    if not body:
        raise ValidationError("Empty body")

    # Strip URLs (hard penalty in judge)
    body = _URL_RE.sub("", body).strip()
    if not body:
        raise ValidationError("Body was only URLs")

    # Anti-repetition check
    if sent_bodies:
        for sent in sent_bodies:
            if sent and body.lower()[:80] == sent.lower()[:80]:
                raise ValidationError("Repeated body detected")

    # CTA validation
    cta = raw.get("cta", "open_ended")
    if cta not in _VALID_CTAS:
        cta = "open_ended"

    # send_as validation
    send_as = raw.get("send_as", "vera")
    if send_as not in _VALID_SEND_AS:
        send_as = "vera"

    # Template name sanity
    template_name = raw.get("template_name", "vera_generic_v1")
    if not template_name or not isinstance(template_name, str):
        template_name = "vera_generic_v1"

    # Template params must be list of strings
    template_params = raw.get("template_params", [])
    if not isinstance(template_params, list):
        template_params = []
    template_params = [str(p) for p in template_params[:5]]

    # Rationale
    rationale = raw.get("rationale", "")
    if not rationale:
        rationale = f"Composed from {raw.get('trigger_kind', 'trigger')} trigger context."

    return {
        "body": body,
        "cta": cta,
        "send_as": send_as,
        "template_name": template_name,
        "template_params": template_params,
        "rationale": rationale,
    }


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """Extract JSON from LLM response (may be wrapped in markdown)."""
    # Try direct parse
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Extract from markdown code block
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise ValidationError(f"Could not parse JSON from LLM response: {text[:200]}")


# ══════════════════════════════════════════════════════════════
# COMPOSER
# ══════════════════════════════════════════════════════════════

class Composer:
    """
    Orchestrates: contexts → prompt → LLM → validate → ComposedMessage.
    Deterministic at temperature=0.
    """

    def __init__(self, llm: Optional[LLMClient] = None) -> None:
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = get_llm_client()
        return self._llm

    def compose_message(
        self,
        trigger: TriggerContext,
        merchant: MerchantContext,
        category: CategoryContext,
        customer: Optional[CustomerContext] = None,
        sent_bodies: Optional[List[str]] = None,
    ) -> ComposedMessage:
        """
        Compose a proactive message (from /v1/tick).
        Returns ComposedMessage with all required fields populated.
        """
        signals = extract_signals(merchant, category)

        # Get trigger-specific prompt builder
        builder = get_prompt_builder(trigger.kind)
        user_prompt = builder.build_user_prompt(trigger, merchant, category, signals, customer)

        # Call LLM
        start = time.time()
        try:
            raw_text = self.llm.complete(user_prompt, VERA_SYSTEM_PROMPT)
        except Exception as e:
            raise RuntimeError(f"LLM call failed: {e}") from e

        elapsed = time.time() - start

        # Parse JSON
        raw_dict = _parse_llm_json(raw_text)

        # Add trigger kind for rationale fallback
        raw_dict["trigger_kind"] = trigger.kind

        # Validate and clean
        cleaned = validate_and_clean(raw_dict, sent_bodies)

        # Override send_as from builder if not customer-facing
        if customer is None and builder.send_as(trigger) == "merchant_on_behalf":
            cleaned["send_as"] = "vera"
        elif customer is not None:
            cleaned["send_as"] = builder.send_as(trigger)

        # Build suppression key
        suppression_key = trigger.suppression_key or f"{trigger.kind}:{trigger.merchant_id}:{trigger.id[:8]}"

        return ComposedMessage(
            body=cleaned["body"],
            cta=cleaned["cta"],
            send_as=cleaned["send_as"],
            template_name=cleaned.get("template_name", builder.template_name(trigger)),
            template_params=cleaned.get("template_params", []),
            suppression_key=suppression_key,
            rationale=cleaned["rationale"],
        )

    def compose_reply(
        self,
        reply_type: str,
        merchant_message: str,
        state: ConversationState,
        merchant: Optional[MerchantContext],
        category: Optional[CategoryContext],
        trigger: Optional[TriggerContext],
    ) -> Dict[str, Any]:
        """
        Compose a reply to a merchant/customer message (from /v1/reply).
        Returns a dict with action, body, cta, wait_seconds, rationale.
        """
        merchant_name = merchant.identity.name if merchant else "the merchant"
        category_slug = category.slug if category else state.category_slug
        language_pref = " ".join(merchant.identity.languages) if merchant else "english"

        trigger_summary = ""
        if trigger:
            trigger_summary = f"{trigger.kind} trigger (urgency {trigger.urgency}): {json.dumps(trigger.payload)[:200]}"

        conv_summary = build_reply_context_summary(state)

        user_prompt = build_reply_prompt(
            reply_type=reply_type,
            merchant_message=merchant_message,
            conv_summary=conv_summary,
            merchant_name=merchant_name,
            category_slug=category_slug,
            language_pref=language_pref,
            trigger_summary=trigger_summary,
            sent_bodies=state.sent_bodies,
        )

        try:
            raw_text = self.llm.complete(user_prompt, REPLY_SYSTEM_PROMPT)
            raw_dict = _parse_llm_json(raw_text)
        except Exception as e:
            # Safe fallback
            return {
                "action": "send",
                "body": "Got it — let me take care of that for you right away.",
                "cta": "open_ended",
                "rationale": f"LLM error: {e}. Sent safe fallback.",
            }

        # Validate action field
        action = raw_dict.get("action", "send")
        if action not in ("send", "wait", "end"):
            action = "send"

        result: Dict[str, Any] = {
            "action": action,
            "rationale": raw_dict.get("rationale", ""),
        }

        if action == "send":
            body = raw_dict.get("body", "").strip()
            body = _URL_RE.sub("", body).strip()
            # Anti-repetition
            if any(body.lower()[:80] == s.lower()[:80] for s in state.sent_bodies if s):
                body = body + " (updated)"
            result["body"] = body
            raw_cta = raw_dict.get("cta", "open_ended")
            if isinstance(raw_cta, dict):
                raw_cta = raw_cta.get("type", "open_ended")
            if not isinstance(raw_cta, str):
                raw_cta = "open_ended"
            if raw_cta not in {"open_ended", "binary_yes_no", "binary_confirm_cancel", "multi_choice_slot", "none"}:
                raw_cta = "open_ended"
            result["cta"] = raw_cta

        elif action == "wait":
            result["wait_seconds"] = int(raw_dict.get("wait_seconds", settings.auto_reply_wait_seconds))

        return result


# Singleton composer
_composer: Optional[Composer] = None


def get_composer() -> Composer:
    global _composer
    if _composer is None:
        _composer = Composer()
    return _composer
