"""
Vizzy Chat — Text Service
───────────────────────────
High-level text/narrative generation built on openai_service.
Handles stories, poems, marketing copy, and scene breakdowns.
"""

from __future__ import annotations

from typing import Optional

import config
from services.openai_service import chat_completion
from utils.prompt_builder import build_system_prompt, inject_memory_context, build_story_prompt
from utils.logger import get_logger

log = get_logger(__name__)


def generate_narrative(
    user_request: str,
    mode: str,
    preferences: Optional[dict] = None,
) -> str:
    """
    Produce a rich narrative / story / poem based on the user request.
    Automatically incorporates mode context and taste memory.
    """
    system = build_system_prompt(mode)
    if preferences:
        system = inject_memory_context(system, preferences)

    story_prompt = build_story_prompt(user_request, mode, preferences)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": story_prompt},
    ]
    text = chat_completion(messages, temperature=0.85, max_tokens=3000)
    log.info(f"Narrative generated ({len(text)} chars)")
    return text


def generate_marketing_copy(
    user_request: str,
    mode: str,
    preferences: Optional[dict] = None,
) -> str:
    """
    Produce strategic marketing copy — taglines, descriptions,
    brand messaging.
    """
    system = build_system_prompt(mode)
    if preferences:
        system = inject_memory_context(system, preferences)

    system += (
        "\nYou are also a senior brand strategist. "
        "Write copy that is premium, concise, and psychologically sharp. "
        "Avoid clichés. Avoid exclamation marks. "
        "Think Apple keynote, not clearance flyer."
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_request},
    ]
    return chat_completion(messages, temperature=0.75, max_tokens=1500)


def conversational_reply(
    messages_history: list[dict[str, str]],
    mode: str,
    preferences: Optional[dict] = None,
) -> str:
    """
    Handle general conversation — answer questions, guide the user,
    suggest creative directions.
    """
    system = build_system_prompt(mode)
    if preferences:
        system = inject_memory_context(system, preferences)

    system += (
        "\nIf the user seems unsure, gently suggest creative directions. "
        "Always be warm and helpful. Never sound robotic."
    )

    full_messages = [{"role": "system", "content": system}] + messages_history
    return chat_completion(full_messages, temperature=0.7, max_tokens=1500)


def extract_scenes_from_story(story_text: str) -> list[dict[str, str]]:
    """
    Given a generated story, extract numbered scenes with visual notes.

    Returns a list of dicts:
        [{"scene_number": "1", "description": "...", "visual_note": "..."}, ...]
    """
    prompt = (
        "Extract each numbered scene from the following story. "
        "For each scene output a JSON array of objects with keys: "
        '"scene_number", "description" (1-2 sentences), '
        '"visual_note" (a one-line art direction note).\n\n'
        f"Story:\n{story_text}\n\n"
        "Output JSON only."
    )
    messages = [
        {
            "role": "system",
            "content": "You extract structured scene data. Output valid JSON only.",
        },
        {"role": "user", "content": prompt},
    ]
    import json
    raw = chat_completion(
        messages,
        temperature=0.3,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(raw)
        # Handle both {"scenes": [...]} and direct [...]
        if isinstance(data, dict) and "scenes" in data:
            return data["scenes"]
        if isinstance(data, list):
            return data
        return []
    except json.JSONDecodeError:
        log.warning("Failed to parse scene extraction JSON.")
        return []
