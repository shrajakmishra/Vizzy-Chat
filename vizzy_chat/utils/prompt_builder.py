"""
Vizzy Chat — Prompt Builder
─────────────────────────────
Central module for constructing structured prompts.
Handles context injection, memory injection, mode-based
instructions, and emotional tone calibration.
"""

from __future__ import annotations

from typing import Optional

from config import MODE_HOME, MODE_BUSINESS


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# System Prompt Templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM_BASE = """\
You are Vizzy — a calm, intelligent, premium creative assistant.
You speak like a thoughtful creative director: warm, clear, never robotic.
Never say "Generating image…" or "Processing request…".
Instead, use human language: "Exploring visual directions…", \
"Interpreting your creative intent…".
Always be concise; avoid filler.
"""

_HOME_CONTEXT = """\
The user is in Home mode — they are creating for personal expression.
Prioritise emotional depth, aesthetic beauty, and personal resonance.
Styles lean toward: artistic, dreamlike, emotional, poetic, warm.
Avoid anything that feels corporate or clinical.
"""

_BUSINESS_CONTEXT = """\
The user is in Business mode — they are creating for marketing & branding.
Prioritise brand alignment, premium perception, strategic positioning.
Outputs must never look cheap, generic, or template-driven.
Think: Apple-level restraint, Aesop-level warmth, Diptyque-level taste.
Psychological framing matters: premium ≠ expensive-looking.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image Prompt Templates
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_IMAGE_VARIATION_INSTRUCTION = """\
Generate a DALL-E prompt for a high-quality, visually striking image.
This is variation {variation_number} of {total_variations}.
Each variation must differ in:
  • Composition / framing
  • Colour palette / mood
  • Artistic style or texture

Variation guidance for #{variation_number}:
{variation_guidance}

Avoid: cliché AI art looks, over-saturated neon, generic stock-photo feel.
Produce something that could hang in a gallery or appear in a brand campaign.
"""

_VARIATION_GUIDES = [
    "Focus on a bold, high-contrast composition with dramatic lighting.",
    "Take a softer, more intimate approach — gentle tones, close detail.",
    "Go abstract or conceptual — evoke the essence rather than literalness.",
]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Builder Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_system_prompt(mode: str) -> str:
    """Compose the full system prompt given the active mode."""
    mode_block = _HOME_CONTEXT if mode == MODE_HOME else _BUSINESS_CONTEXT
    return f"{_SYSTEM_BASE}\n{mode_block}"


def inject_memory_context(
    system_prompt: str,
    preferences: dict,
) -> str:
    """Append user-preference memory to the system prompt."""
    if not preferences:
        return system_prompt

    lines = ["", "User taste profile (apply subtly, do not mention explicitly):"]
    for key, value in preferences.items():
        lines.append(f"  • {key}: {value}")
    return system_prompt + "\n".join(lines) + "\n"


def build_image_prompt(
    user_request: str,
    mode: str,
    variation_index: int = 0,
    total_variations: int = 3,
    preferences: Optional[dict] = None,
    refinement_delta: Optional[str] = None,
) -> str:
    """
    Build a structured prompt for image generation.

    Parameters
    ----------
    user_request : str
        The raw user message.
    mode : str
        'home' or 'business'.
    variation_index : int
        0-based index of the current variation.
    total_variations : int
        How many variations are being produced.
    preferences : dict, optional
        User taste memory.
    refinement_delta : str, optional
        Natural-language adjustment from the iteration engine.
    """
    guide = _VARIATION_GUIDES[variation_index % len(_VARIATION_GUIDES)]
    variation_block = _IMAGE_VARIATION_INSTRUCTION.format(
        variation_number=variation_index + 1,
        total_variations=total_variations,
        variation_guidance=guide,
    )

    mode_note = (
        "Style: personal, artistic, emotionally resonant."
        if mode == MODE_HOME
        else "Style: brand-worthy, premium, strategically positioned."
    )

    parts = [
        f"User request: {user_request}",
        mode_note,
        variation_block,
    ]

    if preferences:
        pref_str = ", ".join(f"{k}={v}" for k, v in preferences.items())
        parts.append(f"User preferences: {pref_str}")

    if refinement_delta:
        parts.append(f"Refinement instruction: {refinement_delta}")

    return "\n\n".join(parts)


def build_story_prompt(
    user_request: str,
    mode: str,
    preferences: Optional[dict] = None,
) -> str:
    """Build a prompt for narrative / story generation."""
    mode_note = (
        "Write with emotional richness and poetic detail."
        if mode == MODE_HOME
        else "Write with brand-level clarity and strategic storytelling."
    )
    parts = [
        f"User request: {user_request}",
        mode_note,
        (
            "Break the narrative into clear scenes that can each become "
            "a visual. Number each scene and give it a one-line visual note."
        ),
    ]
    if preferences:
        pref_str = ", ".join(f"{k}={v}" for k, v in preferences.items())
        parts.append(f"User taste: {pref_str}")
    return "\n\n".join(parts)


def build_iteration_prompt(
    original_prompt: str,
    user_feedback: str,
) -> str:
    """Build a prompt that adjusts a previous output based on feedback."""
    return (
        f"Original creative prompt:\n{original_prompt}\n\n"
        f"User feedback / adjustment:\n{user_feedback}\n\n"
        "Rewrite the creative prompt incorporating the feedback. "
        "Keep the core concept intact but shift the aesthetic direction "
        "as instructed. Output ONLY the revised prompt, nothing else."
    )


def build_intent_classification_prompt(user_message: str, mode: str) -> str:
    """Build the prompt used by the intent engine to classify user input."""
    return (
        "You are an intent classifier for a creative AI platform.\n"
        f"Current mode: {mode}\n\n"
        "Classify the following user message into EXACTLY ONE intent:\n"
        "  visual_creation — user wants a new image / artwork / poster\n"
        "  image_transformation — user wants to modify / transform an existing image\n"
        "  story_generation — user wants a narrative, story, poem, or text piece\n"
        "  marketing_asset — user wants a branded visual, signage, or ad creative\n"
        "  emotional_interpretation — user wants an abstract emotional rendering\n"
        "  multi_step_creative — user wants a complex flow (e.g. story→scenes→images)\n"
        "  iteration_refinement — user is giving feedback on a previous output\n"
        "  general_conversation — user is chatting, asking questions, or unclear\n\n"
        f"User message: \"{user_message}\"\n\n"
        "Respond with a JSON object:\n"
        '{"intent": "<label>", "confidence": <0.0-1.0>, "reasoning": "<brief>"}'
    )
