"""
Vizzy Chat — Application Configuration
───────────────────────────────────────
Central configuration sourced from environment variables with
sensible defaults. All modules import settings from here.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
MEMORY_FILE = STORAGE_DIR / "memory.json"
LOG_DIR = BASE_DIR / "logs"

# Ensure directories exist
STORAGE_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# ── OpenAI ─────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL: str = os.getenv("VIZZY_CHAT_MODEL", "gpt-4o")
IMAGE_MODEL: str = os.getenv("VIZZY_IMAGE_MODEL", "dall-e-3")
EMBEDDING_MODEL: str = os.getenv("VIZZY_EMBEDDING_MODEL", "text-embedding-3-small")

# ── Application ────────────────────────────────────────────────
DEFAULT_VARIATIONS: int = int(os.getenv("VIZZY_DEFAULT_VARIATIONS", "1"))
LOG_LEVEL: str = os.getenv("VIZZY_LOG_LEVEL", "INFO")
MEMORY_BACKEND: str = os.getenv("VIZZY_MEMORY_BACKEND", "json")

# ── Intent Labels ──────────────────────────────────────────────
INTENT_LABELS: list[str] = [
    "visual_creation",
    "image_transformation",
    "story_generation",
    "marketing_asset",
    "emotional_interpretation",
    "multi_step_creative",
    "iteration_refinement",
    "general_conversation",
]

# ── Mode Labels ────────────────────────────────────────────────
MODE_HOME = "home"
MODE_BUSINESS = "business"

# ── Human-Centred Status Messages ──────────────────────────────
STATUS_MESSAGES: dict[str, list[str]] = {
    "visual_creation": [
        "Exploring visual directions based on your idea…",
        "Crafting aesthetic interpretations for you…",
        "Translating your vision into imagery…",
    ],
    "image_transformation": [
        "Reimagining your image through a new lens…",
        "Applying a creative transformation…",
        "Reinterpreting your visual with a fresh perspective…",
    ],
    "story_generation": [
        "Weaving a narrative from your prompt…",
        "Building a story world around your idea…",
        "Crafting scenes and characters for you…",
    ],
    "marketing_asset": [
        "Designing brand-aligned creative assets…",
        "Producing marketing-ready visuals…",
        "Generating strategic creative outputs…",
    ],
    "emotional_interpretation": [
        "Interpreting the emotional landscape you described…",
        "Translating feelings into visual language…",
        "Sensing the mood and rendering it visually…",
    ],
    "multi_step_creative": [
        "Orchestrating a multi-layered creative process…",
        "Breaking your idea into creative stages…",
        "Building your vision step by step…",
    ],
    "iteration_refinement": [
        "Refining based on your feedback…",
        "Adjusting the creative direction…",
        "Fine-tuning to get closer to your vision…",
    ],
    "general_conversation": [
        "Thinking about that…",
        "Let me help with that…",
    ],
}

# ── Validation ─────────────────────────────────────────────────
def validate_config() -> list[str]:
    """Return a list of configuration warnings (empty = all good)."""
    warnings: list[str] = []
    if not OPENAI_API_KEY or OPENAI_API_KEY.startswith("sk-your"):
        warnings.append("OPENAI_API_KEY is not set. Add it to your .env file.")
    return warnings
