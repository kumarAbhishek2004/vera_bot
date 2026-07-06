"""
app/state.py — Thread-safe in-memory stores.

Three stores:
  ContextStore      — versioned (scope, context_id) → payload
  ConversationStore — conversation_id → ConversationState
  SuppressionStore  — set of fired suppression keys for this test window
"""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional, Set, Tuple

from .models import ConversationState


# ══════════════════════════════════════════════════════════════
# CONTEXT STORE
# ══════════════════════════════════════════════════════════════

class ContextStore:
    """
    Versioned context store.
    - Idempotent: same (scope, context_id, version) → rejected as stale
    - Higher version atomically replaces lower version
    - Thread-safe via asyncio Lock
    """

    def __init__(self) -> None:
        self._data: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def push(
        self, scope: str, context_id: str, version: int, payload: Dict[str, Any]
    ) -> Tuple[bool, Optional[int]]:
        """
        Returns (accepted, current_version_if_rejected).
        accepted=True  → stored successfully.
        accepted=False → caller already has same or higher version.
        """
        async with self._lock:
            key = (scope, context_id)
            current = self._data.get(key)
            if current and current["version"] >= version:
                return False, current["version"]
            self._data[key] = {"version": version, "payload": payload}
            return True, None

    def get(self, scope: str, context_id: str) -> Optional[Dict[str, Any]]:
        """Return the stored payload (not version wrapper)."""
        entry = self._data.get((scope, context_id))
        return entry["payload"] if entry else None

    def get_version(self, scope: str, context_id: str) -> Optional[int]:
        entry = self._data.get((scope, context_id))
        return entry["version"] if entry else None

    def count(self, scope: str) -> int:
        return sum(1 for (s, _) in self._data if s == scope)

    def all_of_scope(self, scope: str) -> Dict[str, Dict[str, Any]]:
        """Return {context_id: payload} for all items of a given scope."""
        return {
            cid: entry["payload"]
            for (s, cid), entry in self._data.items()
            if s == scope
        }


# ══════════════════════════════════════════════════════════════
# CONVERSATION STORE
# ══════════════════════════════════════════════════════════════

class ConversationStore:
    """
    Tracks per-conversation FSM state, turn history, sent body history.
    Used by /v1/reply and /v1/tick (to avoid duplicating ongoing convos).
    """

    def __init__(self) -> None:
        self._data: Dict[str, ConversationState] = {}
        self._lock = asyncio.Lock()

    async def get_or_create(
        self,
        conversation_id: str,
        merchant_id: str,
        customer_id: Optional[str] = None,
        trigger_id: str = "",
        category_slug: str = "",
    ) -> ConversationState:
        async with self._lock:
            if conversation_id not in self._data:
                self._data[conversation_id] = ConversationState(
                    conversation_id=conversation_id,
                    merchant_id=merchant_id,
                    customer_id=customer_id,
                    trigger_id=trigger_id,
                    category_slug=category_slug,
                )
            return self._data[conversation_id]

    async def update(self, state: ConversationState) -> None:
        async with self._lock:
            state.last_action_at = datetime.utcnow()
            self._data[state.conversation_id] = state

    def get(self, conversation_id: str) -> Optional[ConversationState]:
        return self._data.get(conversation_id)

    def exists(self, conversation_id: str) -> bool:
        return conversation_id in self._data

    def is_active(self, conversation_id: str) -> bool:
        state = self._data.get(conversation_id)
        if not state:
            return False
        from .models import FSMState
        return state.fsm_state not in (FSMState.ENDED,)


# ══════════════════════════════════════════════════════════════
# SUPPRESSION STORE
# ══════════════════════════════════════════════════════════════

class SuppressionStore:
    """
    Tracks fired suppression keys within this test window.
    Prevents re-sending the same message family to the same entity.
    """

    def __init__(self) -> None:
        self._fired: Set[str] = set()
        self._lock = asyncio.Lock()

    async def fire(self, suppression_key: str) -> None:
        async with self._lock:
            self._fired.add(suppression_key)

    def is_fired(self, suppression_key: str) -> bool:
        return suppression_key in self._fired

    async def reset(self) -> None:
        """Called on /v1/teardown — wipes all state."""
        async with self._lock:
            self._fired.clear()


# ══════════════════════════════════════════════════════════════
# GLOBAL SINGLETON STORES
# ══════════════════════════════════════════════════════════════

context_store = ContextStore()
conversation_store = ConversationStore()
suppression_store = SuppressionStore()


class ContextResolver:
    """Helper to retrieve contexts and handle graceful fallbacks (e.g. missing Category)."""
    def __init__(self, store: ContextStore):
        self.store = store

    def get_trigger(self, trigger_id: str):
        from .models import TriggerContext
        payload = self.store.get("trigger", trigger_id)
        print("Lookup Trigger ID:", trigger_id)
        print("Raw Payload:", payload)

        if not payload: return None
        try:
            return TriggerContext(**payload)
        except Exception as e:
            print("Trigger Parse Error:", e)
            print("Payload causing error:", payload)
            return None

    def get_merchant(self, merchant_id: str):
        from .models import MerchantContext
        payload = self.store.get("merchant", merchant_id)
        if not payload: return None
        try: return MerchantContext(**payload)
        except Exception: return None

    def get_category(self, slug: str):
        from .models import CategoryContext
        payload = self.store.get("category", slug)
        if payload:
            try: return CategoryContext(**payload)
            except Exception: pass
        return CategoryContext(slug=slug)

    def get_customer(self, customer_id: str):
        from .models import CustomerContext
        payload = self.store.get("customer", customer_id)
        if not payload: return None
        try: return CustomerContext(**payload)
        except Exception: return None

context_resolver = ContextResolver(context_store)

