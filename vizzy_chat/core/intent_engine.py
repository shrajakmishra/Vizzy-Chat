"""
Vizzy Chat — Intent Engine
────────────────────────────
Uses LangChain + LangGraph to classify user intent.
This is the first node in the creative orchestration graph.

Intents:
    visual_creation | image_transformation | story_generation |
    marketing_asset | emotional_interpretation | multi_step_creative |
    iteration_refinement | general_conversation
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

import config
from utils.prompt_builder import build_intent_classification_prompt
from utils.logger import get_logger

log = get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class IntentResult(BaseModel):
    """Structured output from intent classification."""
    intent: str = Field(default="general_conversation", description="Classified intent label")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence 0-1")
    reasoning: str = Field(default="", description="Brief reasoning for classification")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LLM Instance (LangChain)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _get_llm() -> ChatOpenAI:
    """Return a LangChain ChatOpenAI instance for classification.
    Uses gpt-4o-mini for fast, cheap intent classification."""
    return ChatOpenAI(
        model="gpt-4o-mini",
        api_key=config.OPENAI_API_KEY,
        temperature=0.2,           # Low temp for deterministic classification
        max_tokens=200,
        model_kwargs={"response_format": {"type": "json_object"}},
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Classification
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def classify_intent(
    user_message: str,
    mode: str = config.MODE_HOME,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> IntentResult:
    """
    Classify the user's message into a creative intent.

    Parameters
    ----------
    user_message : str
        The latest message from the user.
    mode : str
        Current mode — 'home' or 'business'.
    conversation_history : list[dict], optional
        Recent messages for context (helps detect iteration_refinement).

    Returns
    -------
    IntentResult
        Structured intent classification.
    """
    llm = _get_llm()

    # Build the classification prompt
    classification_prompt = build_intent_classification_prompt(user_message, mode)

    # If there is history, include the last few messages for context
    messages = [
        SystemMessage(content=(
            "You are an intent classifier. Respond ONLY with valid JSON. "
            "Keys: intent (string), confidence (float 0-1), reasoning (string)."
        )),
    ]

    if conversation_history:
        # Include last 4 messages for context
        for msg in conversation_history[-4:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=classification_prompt))

    try:
        response = llm.invoke(messages)
        raw = response.content.strip()
        data = json.loads(raw)

        intent_label = data.get("intent", "general_conversation")
        if intent_label not in config.INTENT_LABELS:
            log.warning(f"Unknown intent '{intent_label}', falling back.")
            intent_label = "general_conversation"

        result = IntentResult(
            intent=intent_label,
            confidence=float(data.get("confidence", 0.5)),
            reasoning=data.get("reasoning", ""),
        )
        log.info(f"Intent classified: {result.intent} ({result.confidence:.0%})")
        return result

    except (json.JSONDecodeError, KeyError, ValueError) as exc:
        log.error(f"Intent classification failed: {exc}")
        return IntentResult(
            intent="general_conversation",
            confidence=0.3,
            reasoning="Classification parsing failed; defaulting to conversation.",
        )


def detect_mode_from_message(user_message: str) -> Optional[str]:
    """
    Heuristic mode detection from user message content.
    Returns 'home', 'business', or None (no signal).
    """
    msg_lower = user_message.lower()

    business_signals = [
        "brand", "marketing", "product", "signage", "store",
        "sale", "menu", "poster", "campaign", "premium",
        "customer", "business", "commercial", "advertisement",
        "seasonal", "promotion", "retail",
    ]
    home_signals = [
        "memory", "dream", "poem", "emotional", "kids",
        "story", "feeling", "vision board", "personal",
        "living room", "bedroom", "home", "family",
    ]

    biz_score = sum(1 for w in business_signals if w in msg_lower)
    home_score = sum(1 for w in home_signals if w in msg_lower)

    if biz_score > home_score and biz_score >= 2:
        return config.MODE_BUSINESS
    if home_score > biz_score and home_score >= 2:
        return config.MODE_HOME
    return None
