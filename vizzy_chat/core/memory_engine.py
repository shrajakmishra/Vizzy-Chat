"""
Vizzy Chat — Memory Engine (Taste Memory System)
──────────────────────────────────────────────────
Persistent storage of user preferences that evolve over time.
Preferences are automatically injected into generation prompts
to produce increasingly personalised outputs.

Storage backend: JSON file (upgradeable to SQLite / Redis).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import config
from services.openai_service import chat_completion
from utils.logger import get_logger

log = get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Preference Dimensions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREFERENCE_DIMENSIONS: list[str] = [
    "tone_warmth",          # warm ↔ cool
    "detail_level",         # minimal ↔ detailed
    "premium_level",        # premium ↔ playful
    "emotional_density",    # subtle ↔ intense
    "colour_preference",    # e.g. earth tones, pastels, monochrome
    "style_preference",     # e.g. photographic, illustrated, abstract
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# JSON Persistence
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _load_store() -> dict:
    """Load the full memory store from disk."""
    path: Path = config.MEMORY_FILE
    if not path.exists():
        return {"users": {"default": {"preferences": {}, "history": []}}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_store(store: dict) -> None:
    """Persist the memory store to disk."""
    path: Path = config.MEMORY_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    log.debug("Memory store saved.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_preferences(user_id: str = "default") -> dict:
    """Return the current preference dict for *user_id*."""
    store = _load_store()
    user = store.get("users", {}).get(user_id, {})
    return user.get("preferences", {})


def update_preferences(new_prefs: dict, user_id: str = "default") -> dict:
    """
    Merge *new_prefs* into the stored preferences for *user_id*.
    Returns the updated preference dict.
    """
    store = _load_store()
    users = store.setdefault("users", {})
    user = users.setdefault(user_id, {"preferences": {}, "history": []})
    user["preferences"].update(new_prefs)
    _save_store(store)
    log.info(f"Preferences updated for '{user_id}': {new_prefs}")
    return user["preferences"]


def add_history_entry(
    entry: dict,
    user_id: str = "default",
    max_history: int = 200,
) -> None:
    """
    Append a session entry to the user's history.
    Keeps the last *max_history* entries.
    """
    store = _load_store()
    users = store.setdefault("users", {})
    user = users.setdefault(user_id, {"preferences": {}, "history": []})

    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
    user["history"].append(entry)
    user["history"] = user["history"][-max_history:]

    _save_store(store)
    log.debug(f"History entry added for '{user_id}'.")


def clear_preferences(user_id: str = "default") -> None:
    """Reset preferences for *user_id*."""
    store = _load_store()
    if user_id in store.get("users", {}):
        store["users"][user_id]["preferences"] = {}
        _save_store(store)
        log.info(f"Preferences cleared for '{user_id}'.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Taste Inference (LLM-powered)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def infer_preferences_from_interaction(
    user_message: str,
    assistant_output: str,
    current_prefs: Optional[dict] = None,
) -> dict:
    """
    After each creative interaction, use the LLM to infer whether
    the user revealed any taste preferences. Returns a dict of
    updated preference keys (may be empty).
    """
    current = json.dumps(current_prefs or {})
    dims = ", ".join(PREFERENCE_DIMENSIONS)

    messages = [
        {
            "role": "system",
            "content": (
                "You analyse creative interactions to detect user taste. "
                "Given a user message and assistant output, determine if the "
                "user has revealed preferences along these dimensions:\n"
                f"  {dims}\n\n"
                f"Current stored preferences: {current}\n\n"
                "Output a JSON object with ONLY the dimensions that changed "
                "or were newly revealed. Values should be short descriptive "
                "strings. If nothing changed, output an empty object {{}}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"User said: \"{user_message}\"\n"
                f"Assistant produced: \"{assistant_output[:500]}…\""
            ),
        },
    ]

    try:
        raw = chat_completion(
            messages,
            temperature=0.3,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
        inferred = json.loads(raw)
        # Filter to known dimensions
        valid = {k: v for k, v in inferred.items() if k in PREFERENCE_DIMENSIONS}
        if valid:
            log.info(f"Inferred preferences: {valid}")
        return valid
    except (json.JSONDecodeError, Exception) as exc:
        log.warning(f"Preference inference failed: {exc}")
        return {}


def learn_from_interaction(
    user_message: str,
    assistant_output: str,
    user_id: str = "default",
) -> dict:
    """
    High-level method: infer preferences and persist them.
    Returns the full updated preference dict.
    """
    current = get_preferences(user_id)
    new = infer_preferences_from_interaction(user_message, assistant_output, current)
    if new:
        return update_preferences(new, user_id)
    return current
