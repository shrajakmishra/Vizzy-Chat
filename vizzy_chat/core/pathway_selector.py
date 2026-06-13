"""
Vizzy Chat — Pathway Selector
───────────────────────────────
Routes classified intents to the appropriate creative pipeline.
Implemented as a LangGraph state machine.
"""

from __future__ import annotations

from typing import TypedDict, Literal, Optional
from langgraph.graph import StateGraph, END

from core.intent_engine import IntentResult
from utils.logger import get_logger

log = get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph State
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class PipelineState(TypedDict):
    """State that flows through the creative pipeline graph."""
    user_message: str
    mode: str
    intent: str
    confidence: float
    preferences: dict
    conversation_history: list
    # Outputs (populated by pipeline nodes)
    pathway: str                       # e.g. "image", "text", "multi_step", "transform", "conversation"
    images: list                       # list of image result dicts
    text_output: str
    status_message: str
    needs_iteration: bool
    refinement_delta: Optional[str]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mapping: Intent → Pathway
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

INTENT_TO_PATHWAY: dict[str, str] = {
    "visual_creation":          "image",
    "image_transformation":     "transform",
    "story_generation":         "text",
    "marketing_asset":          "image",
    "emotional_interpretation": "image",
    "multi_step_creative":      "multi_step",
    "iteration_refinement":     "iteration",
    "general_conversation":     "conversation",
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Graph Node Functions
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def route_intent(state: PipelineState) -> PipelineState:
    """Map intent → pathway and set an appropriate status message."""
    intent = state["intent"]
    pathway = INTENT_TO_PATHWAY.get(intent, "conversation")

    import random
    from config import STATUS_MESSAGES
    msgs = STATUS_MESSAGES.get(intent, STATUS_MESSAGES["general_conversation"])
    status = random.choice(msgs)

    state["pathway"] = pathway
    state["status_message"] = status
    log.info(f"Pathway selected: {intent} → {pathway}")
    return state


def should_generate_images(state: PipelineState) -> Literal["generate_images", "generate_text", "multi_step", "iterate", "converse"]:
    """Conditional edge: decide which generation node to enter."""
    pathway = state.get("pathway", "conversation")
    mapping = {
        "image": "generate_images",
        "transform": "generate_images",
        "text": "generate_text",
        "multi_step": "multi_step",
        "iteration": "iterate",
        "conversation": "converse",
    }
    return mapping.get(pathway, "converse")


# ── Generation nodes (thin wrappers — real work in generation_engine) ──

def generate_images_node(state: PipelineState) -> PipelineState:
    """Generate image variations."""
    from core.generation_engine import run_image_pipeline
    state = run_image_pipeline(state)
    return state


def generate_text_node(state: PipelineState) -> PipelineState:
    """Generate narrative / text content."""
    from core.generation_engine import run_text_pipeline
    state = run_text_pipeline(state)
    return state


def multi_step_node(state: PipelineState) -> PipelineState:
    """Run the multi-step creative flow (story → scenes → images)."""
    from core.generation_engine import run_multi_step_pipeline
    state = run_multi_step_pipeline(state)
    return state


def iterate_node(state: PipelineState) -> PipelineState:
    """Handle iteration / refinement of previous output."""
    from core.iteration_engine import handle_iteration
    state = handle_iteration(state)
    return state


def converse_node(state: PipelineState) -> PipelineState:
    """Handle general conversation."""
    from core.generation_engine import run_conversation_pipeline
    state = run_conversation_pipeline(state)
    return state


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Build the LangGraph
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_creative_graph() -> StateGraph:
    """
    Construct and return the compiled LangGraph creative pipeline.

    Flow:
        route_intent → conditional → {generate_images | generate_text |
                                        multi_step | iterate | converse} → END
    """
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("route_intent", route_intent)
    graph.add_node("generate_images", generate_images_node)
    graph.add_node("generate_text", generate_text_node)
    graph.add_node("multi_step", multi_step_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("converse", converse_node)

    # Entry point
    graph.set_entry_point("route_intent")

    # Conditional edges from router
    graph.add_conditional_edges(
        "route_intent",
        should_generate_images,
        {
            "generate_images": "generate_images",
            "generate_text": "generate_text",
            "multi_step": "multi_step",
            "iterate": "iterate",
            "converse": "converse",
        },
    )

    # All generation nodes → END
    graph.add_edge("generate_images", END)
    graph.add_edge("generate_text", END)
    graph.add_edge("multi_step", END)
    graph.add_edge("iterate", END)
    graph.add_edge("converse", END)

    compiled = graph.compile()
    log.info("Creative pipeline graph compiled.")
    return compiled
