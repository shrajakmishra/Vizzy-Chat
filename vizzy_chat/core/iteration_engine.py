"""
Vizzy Chat — Iteration Engine
───────────────────────────────
Handles refinement requests: understands the delta change,
modifies the previous prompt, and re-generates.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.openai_service import chat_completion
from services.image_service import generate_variations
from utils.prompt_builder import build_iteration_prompt
from utils.logger import get_logger

if TYPE_CHECKING:
    from core.pathway_selector import PipelineState

log = get_logger(__name__)


def extract_refinement_delta(
    user_message: str,
    previous_context: str = "",
) -> str:
    """
    Use the LLM to interpret what change the user is asking for.
    Returns a concise refinement instruction.

    Examples:
        "Make it warmer"      → "Shift colour palette to warm tones (amber, terracotta). Softer lighting."
        "Less dramatic"       → "Reduce contrast, calmer composition, muted palette."
        "More premium but not flashy" → "Elevate material quality, restrained luxury, no metallic excess."
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You interpret creative feedback. "
                "Given a user's refinement request and optional context, "
                "output a concise, actionable aesthetic instruction (1-3 sentences). "
                "This will be injected into an image prompt. "
                "Do NOT output anything else."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Previous context: {previous_context}\n\n"
                f"User feedback: {user_message}"
            ),
        },
    ]
    delta = chat_completion(messages, temperature=0.5, max_tokens=200)
    log.info(f"Refinement delta: {delta}")
    return delta


def revise_prompt(original_prompt: str, user_feedback: str) -> str:
    """
    Take an original DALL-E prompt and user feedback,
    produce a revised prompt that incorporates the change.
    """
    revision_prompt = build_iteration_prompt(original_prompt, user_feedback)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a creative prompt editor. "
                "Rewrite the original prompt incorporating the user's "
                "feedback. Keep the core concept. Output ONLY the revised prompt."
            ),
        },
        {"role": "user", "content": revision_prompt},
    ]
    revised = chat_completion(messages, temperature=0.6, max_tokens=500)
    log.info(f"Prompt revised ({len(revised)} chars)")
    return revised


def handle_iteration(state: "PipelineState") -> "PipelineState":
    """
    Pipeline node: handle an iteration / refinement request.

    Looks at conversation history to find the previous prompt,
    applies the user's delta, and regenerates.
    """
    user_msg = state["user_message"]
    history = state.get("conversation_history", [])

    # Try to find the last image prompt used
    previous_prompt = ""
    for msg in reversed(history):
        content = msg.get("content", "")
        # If a previous assistant message has prompt info, use it
        if msg.get("role") == "assistant" and "prompt_used" in str(msg.get("metadata", {})):
            previous_prompt = msg["metadata"].get("prompt_used", "")
            break

    # If we can't find a stored prompt, use the last user message as context
    if not previous_prompt:
        for msg in reversed(history):
            if msg.get("role") == "user" and msg.get("content") != user_msg:
                previous_prompt = msg["content"]
                break

    # Extract the refinement delta
    delta = extract_refinement_delta(user_msg, previous_prompt)
    state["refinement_delta"] = delta

    if previous_prompt:
        # Revise and regenerate
        revised = revise_prompt(previous_prompt, delta)
        from services.openai_service import generate_image
        urls = generate_image(revised)
        state["images"] = [
            {"url": url, "prompt_used": revised, "variation": i + 1}
            for i, url in enumerate(urls)
        ]
        state["text_output"] = (
            "I've refined the visual based on your feedback. "
            "Here's the updated version — let me know if it's closer to what you had in mind."
        )
    else:
        # No previous context — generate fresh with the delta as guidance
        images = generate_variations(
            user_request=user_msg,
            mode=state["mode"],
            num_variations=1,
            preferences=state.get("preferences", {}),
            refinement_delta=delta,
        )
        state["images"] = images
        state["text_output"] = (
            "I interpreted your refinement as a new creative direction. "
            "Here's what I came up with."
        )

    return state
