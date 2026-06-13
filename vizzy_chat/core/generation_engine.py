"""
Vizzy Chat — Generation Engine
─────────────────────────────────
Executes the actual creative generation for each pipeline pathway.
Called by pathway_selector graph nodes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from services.image_service import generate_variations
from services.text_service import (
    generate_narrative,
    generate_marketing_copy,
    conversational_reply,
    extract_scenes_from_story,
)
from utils.logger import get_logger

if TYPE_CHECKING:
    from core.pathway_selector import PipelineState

log = get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_image_pipeline(state: "PipelineState") -> "PipelineState":
    """
    Generate image variations for visual_creation, marketing_asset,
    emotional_interpretation, or image_transformation intents.
    """
    log.info("Running image pipeline…")
    images = generate_variations(
        user_request=state["user_message"],
        mode=state["mode"],
        preferences=state.get("preferences", {}),
        refinement_delta=state.get("refinement_delta"),
    )
    state["images"] = images

    count = len(images)
    if count > 0:
        state["text_output"] = (
            "Here's what I created based on your idea. "
            "You can refine it — say things like *'make it warmer'*, "
            "*'more minimal'*, or *'try a different style'*. "
            "Or ask me to *'generate 3 more variations'*."
        )
    else:
        state["text_output"] = (
            "I wasn't able to generate an image this time. "
            "Could you try rephrasing your idea?"
        )
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Text Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_text_pipeline(state: "PipelineState") -> "PipelineState":
    """Generate narrative, story, or poem content."""
    log.info("Running text pipeline…")

    intent = state["intent"]
    mode = state["mode"]
    prefs = state.get("preferences", {})

    if intent == "marketing_asset":
        text = generate_marketing_copy(state["user_message"], mode, prefs)
    else:
        text = generate_narrative(state["user_message"], mode, prefs)

    state["text_output"] = text
    state["images"] = []
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Multi-Step Pipeline (Story → Scenes → Images)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_multi_step_pipeline(state: "PipelineState") -> "PipelineState":
    """
    1. Generate a narrative
    2. Extract scenes
    3. Generate an image for each scene (up to 3)
    """
    log.info("Running multi-step creative pipeline…")

    mode = state["mode"]
    prefs = state.get("preferences", {})

    # Step 1: Generate the story
    story = generate_narrative(state["user_message"], mode, prefs)
    state["text_output"] = story

    # Step 2: Extract scenes
    scenes = extract_scenes_from_story(story)

    # Step 3: Generate image for up to 3 scenes
    images: list[dict] = []
    for scene in scenes[:3]:
        visual_note = scene.get("visual_note", scene.get("description", ""))
        scene_images = generate_variations(
            user_request=visual_note,
            mode=mode,
            num_variations=1,
            preferences=prefs,
        )
        if scene_images:
            img = scene_images[0]
            img["scene_number"] = scene.get("scene_number", "")
            img["scene_description"] = scene.get("description", "")
            images.append(img)

    state["images"] = images
    log.info(f"Multi-step complete: {len(scenes)} scenes, {len(images)} images.")
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Conversation Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_conversation_pipeline(state: "PipelineState") -> "PipelineState":
    """Handle general chat and guidance."""
    log.info("Running conversation pipeline…")
    history = state.get("conversation_history", [])
    # Ensure the latest message is in history
    if not history or history[-1].get("content") != state["user_message"]:
        history = history + [{"role": "user", "content": state["user_message"]}]

    reply = conversational_reply(history, state["mode"], state.get("preferences", {}))
    state["text_output"] = reply
    state["images"] = []
    return state
