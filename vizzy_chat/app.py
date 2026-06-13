"""
Vizzy Chat - Main Streamlit Application
----------------------------------------
A premium conversational creative interface.
Supports Home and Business modes, image generation,
narrative creation, iteration, and persistent taste memory.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import sys
import os
import io
import time
import requests
from pathlib import Path

# -- Ensure the project root is on sys.path --------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

import config
from config import MODE_HOME, MODE_BUSINESS, validate_config
from core.intent_engine import classify_intent, detect_mode_from_message
from core.pathway_selector import build_creative_graph, PipelineState
from core.memory_engine import (
    get_preferences,
    learn_from_interaction,
    add_history_entry,
    clear_preferences,
)
from utils.logger import get_logger

log = get_logger(__name__)


# =============================================================
# Page Config & Custom CSS
# =============================================================

st.set_page_config(
    page_title="Vizzy Chat",
    page_icon="\U0001f7e3",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* -- Global -------------------------------------------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* -- Hide Streamlit defaults --------------------------- */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* -- Sidebar ------------------------------------------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff;
    }

    /* -- Chat messages ------------------------------------- */
    .stChatMessage {
        border-radius: 16px;
        margin-bottom: 0.5rem;
    }

    /* -- Image gallery ------------------------------------- */
    .image-card {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        transition: transform 0.2s ease;
    }
    .image-card:hover {
        transform: translateY(-2px);
    }

    /* -- Status pill --------------------------------------- */
    .status-pill {
        display: inline-block;
        padding: 6px 16px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 50px;
        font-size: 0.85rem;
        font-weight: 500;
        margin: 8px 0;
    }

    /* -- Mode toggle --------------------------------------- */
    .mode-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .mode-home {
        background: #e8f5e9;
        color: #2e7d32;
    }
    .mode-business {
        background: #e3f2fd;
        color: #1565c0;
    }

    /* -- Buttons ------------------------------------------- */
    .stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* -- Divider ------------------------------------------- */
    .premium-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================
# Session State Initialisation
# =============================================================

def _init_session():
    """Initialise all session-state keys on first run."""
    defaults = {
        "messages": [],
        "mode": MODE_HOME,
        "graph": None,
        "processing": False,
        "last_prompts": [],
        "pending_input": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session()


# =============================================================
# Compile Graph (once per session)
# =============================================================

@st.cache_resource
def get_graph():
    """Build and cache the creative pipeline graph."""
    return build_creative_graph()


# =============================================================
# Sidebar
# =============================================================

PURPLE_CIRCLE = "\U0001f7e3"

with st.sidebar:
    st.markdown("# " + PURPLE_CIRCLE + " Vizzy Chat")
    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # -- Mode Toggle --
    st.markdown("### Mode")
    mode_option = st.radio(
        "Choose your creative mode",
        options=["Home", "Business"],
        index=0 if st.session_state.mode == MODE_HOME else 1,
        label_visibility="collapsed",
    )
    st.session_state.mode = MODE_HOME if "Home" in mode_option else MODE_BUSINESS

    mode_class = "mode-home" if st.session_state.mode == MODE_HOME else "mode-business"
    mode_label = "Home" if st.session_state.mode == MODE_HOME else "Business"
    st.markdown(
        f'<span class="mode-badge {mode_class}">{mode_label} Mode Active</span>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # -- Taste Profile --
    st.markdown("### Your Taste Profile")
    prefs = get_preferences()
    if prefs:
        for dim, val in prefs.items():
            pretty_dim = dim.replace("_", " ").title()
            st.markdown(f"**{pretty_dim}:** {val}")
    else:
        st.markdown("_No preferences learned yet. Use Vizzy and it will adapt to your taste._")

    if prefs and st.button("Reset Taste Profile", use_container_width=True):
        clear_preferences()
        st.rerun()

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

    # -- Session Controls --
    st.markdown("### Session")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_prompts = []
        st.rerun()

    # -- Config Warnings --
    warnings = validate_config()
    if warnings:
        st.markdown("---")
        for w in warnings:
            st.warning(w)

    st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
    st.caption("Built by me - v1.0")


# =============================================================
# Chat History Display
# =============================================================

# Welcome message
if not st.session_state.messages:
    welcome_mode = "personal creativity" if st.session_state.mode == MODE_HOME else "business creativity"
    st.markdown(f"""
    <div style="text-align: center; padding: 3rem 1rem;">
        <h1 style="font-weight: 300; font-size: 2.2rem; color: #333;">
            Welcome to <span style="background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 600;">Vizzy Chat</span>
        </h1>
        <p style="color: #666; font-size: 1.1rem; max-width: 600px; margin: 0.5rem auto;">
            Your intelligent creative companion for {welcome_mode}.
            <br/>Describe what you'd like to create - I'll explore visual and narrative directions for you.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Suggestion chips
    if st.session_state.mode == MODE_HOME:
        suggestions = [
            "Paint something that feels like how my last year felt",
            "Create a dreamlike version of a childhood memory",
            "Generate a story for my kids, then visualise each scene",
            "Design a quote poster for my living room",
        ]
    else:
        suggestions = [
            "Create premium-looking visuals for our new product",
            "Design a seasonal ambiance visual for evenings",
            "Create a sale poster that doesn't feel cheap",
            "Generate brand-themed artwork using warm, earthy tones",
        ]

    cols = st.columns(2)
    for idx, suggestion in enumerate(suggestions):
        with cols[idx % 2]:
            if st.button(suggestion, key=f"sug_{idx}", use_container_width=True):
                st.session_state.pending_input = suggestion
                st.rerun()


# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=PURPLE_CIRCLE if msg["role"] == "assistant" else None):
        if msg.get("content"):
            st.markdown(msg["content"])

        # Render images if present
        if msg.get("images"):
            img_cols = st.columns(min(len(msg["images"]), 3))
            for i, img_data in enumerate(msg["images"]):
                with img_cols[i % 3]:
                    url = img_data.get("url", "")
                    if url:
                        st.image(url, use_container_width=True)
                        # Scene label for multi-step
                        if img_data.get("scene_description"):
                            st.caption(f"Scene {img_data.get('scene_number', '')}: {img_data['scene_description']}")
                        st.caption(f"Variation {img_data.get('variation', i+1)}")

                        # Download button
                        try:
                            img_bytes = requests.get(url, timeout=15).content
                            st.download_button(
                                label="Download",
                                data=img_bytes,
                                file_name=f"vizzy_v{img_data.get('variation', i+1)}.png",
                                mime="image/png",
                                key=f"dl_{id(msg)}_{i}",
                            )
                        except Exception:
                            pass


# =============================================================
# Chat Input & Processing
# =============================================================

user_input = st.chat_input(
    "Describe what you'd like to create...",
    disabled=st.session_state.processing,
)

# Also accept input from suggestion buttons / quick-action buttons
if not user_input and st.session_state.get("pending_input"):
    user_input = st.session_state.pop("pending_input")

if user_input:
    st.session_state.processing = True

    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        # -- Auto-detect mode if strong signals --
        detected_mode = detect_mode_from_message(user_input)
        if detected_mode and detected_mode != st.session_state.mode:
            st.session_state.mode = detected_mode
            st.toast(f"Switched to {detected_mode.title()} mode based on your request.")

        # -- Classify intent --
        mode = st.session_state.mode
        history_for_intent = [
            {"role": m["role"], "content": m.get("content", "")}
            for m in st.session_state.messages[-8:]
        ]

        with st.chat_message("assistant", avatar=PURPLE_CIRCLE):
            import random

            # -- Step 1: Classify intent (with live status) --
            with st.status("Interpreting your creative intent...", expanded=True) as status:
                try:
                    status.update(label="Understanding what you'd like to create...")
                    intent_result = classify_intent(user_input, mode, history_for_intent)
                    status.update(
                        label=f"Intent: {intent_result.intent.replace('_', ' ').title()} "
                              f"({intent_result.confidence:.0%} confidence)"
                    )
                except Exception as e:
                    log.error(f"Intent classification error: {e}")
                    from core.intent_engine import IntentResult
                    intent_result = IntentResult(
                        intent="general_conversation",
                        confidence=0.3,
                        reasoning=f"Classification failed: {e}",
                    )
                    status.update(label="Proceeding with creative interpretation...")

                status_msgs = config.STATUS_MESSAGES.get(
                    intent_result.intent,
                    config.STATUS_MESSAGES["general_conversation"],
                )
                status_text = random.choice(status_msgs)

                # -- Step 2: Build pipeline state --
                try:
                    preferences = get_preferences()
                except Exception:
                    preferences = {}

                initial_state: PipelineState = {
                    "user_message": user_input,
                    "mode": mode,
                    "intent": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "preferences": preferences,
                    "conversation_history": history_for_intent,
                    "pathway": "",
                    "images": [],
                    "text_output": "",
                    "status_message": "",
                    "needs_iteration": False,
                    "refinement_delta": None,
                }

                # -- Step 3: Execute the LangGraph pipeline --
                status.update(label=status_text)
                graph = get_graph()
                try:
                    result = graph.invoke(initial_state)
                except Exception as e:
                    log.error(f"Pipeline error: {e}", exc_info=True)
                    result = dict(initial_state)
                    result["text_output"] = (
                        "Something went wrong while creating for you. "
                        "Could you try rephrasing your idea? "
                        f"\n\n_Error detail: {str(e)[:300]}_"
                    )

                status.update(label="Done!", state="complete", expanded=False)

            # -- Display results --
            text_output = result.get("text_output", "")
            images = result.get("images", [])

            if not text_output and not images:
                st.markdown(
                    "_I processed your request but didn't generate any output. "
                    "Could you try rephrasing?_"
                )
                text_output = "No output generated."

            if text_output:
                st.markdown(text_output)

            if images:
                img_cols = st.columns(min(len(images), 3))
                for i, img_data in enumerate(images):
                    with img_cols[i % 3]:
                        url = img_data.get("url", "")
                        if url:
                            st.image(url, use_container_width=True)
                            if img_data.get("scene_description"):
                                st.caption(
                                    f"Scene {img_data.get('scene_number', '')}: "
                                    f"{img_data['scene_description']}"
                                )
                            st.caption(f"Variation {img_data.get('variation', i+1)}")

                            try:
                                img_bytes = requests.get(url, timeout=15).content
                                st.download_button(
                                    label="Download",
                                    data=img_bytes,
                                    file_name=f"vizzy_v{img_data.get('variation', i+1)}.png",
                                    mime="image/png",
                                    key=f"new_dl_{i}_{time.time_ns()}",
                                )
                            except Exception:
                                pass

                # Track prompts for iteration
                st.session_state.last_prompts = [
                    img.get("prompt_used", "") for img in images
                ]

            # -- Store assistant message in history --
            assistant_msg = {
                "role": "assistant",
                "content": text_output,
                "images": images,
                "metadata": {
                    "intent": intent_result.intent,
                    "confidence": intent_result.confidence,
                    "pathway": result.get("pathway", ""),
                    "prompt_used": images[0].get("prompt_used", "") if images else "",
                },
            }
            st.session_state.messages.append(assistant_msg)

            # -- Save to persistent history (lightweight, no LLM) --
            try:
                add_history_entry({
                    "user_message": user_input,
                    "intent": intent_result.intent,
                    "mode": mode,
                    "had_images": len(images) > 0,
                })
            except Exception as e:
                log.warning(f"History save failed: {e}")

            # -- Quick-action buttons --
            if images or text_output:
                st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
                action_cols = st.columns(5)
                with action_cols[0]:
                    if st.button("Regenerate", key=f"regen_{time.time_ns()}"):
                        st.session_state.pending_input = f"Regenerate: {user_input}"
                        st.session_state.processing = False
                        st.rerun()
                with action_cols[1]:
                    if st.button("Make warmer", key=f"warm_{time.time_ns()}"):
                        st.session_state.pending_input = "Make it warmer and more inviting"
                        st.session_state.processing = False
                        st.rerun()
                with action_cols[2]:
                    if st.button("More minimal", key=f"min_{time.time_ns()}"):
                        st.session_state.pending_input = "Make it more minimal and clean"
                        st.session_state.processing = False
                        st.rerun()
                with action_cols[3]:
                    if st.button("More premium", key=f"prem_{time.time_ns()}"):
                        st.session_state.pending_input = "Make it feel more premium but not flashy"
                        st.session_state.processing = False
                        st.rerun()
                with action_cols[4]:
                    if st.button("3 Variations", key=f"var_{time.time_ns()}"):
                        st.session_state.pending_input = f"Generate 3 different variations of: {user_input}"
                        st.session_state.processing = False
                        st.rerun()

    except Exception as e:
        log.error(f"Unhandled error in processing: {e}", exc_info=True)
        with st.chat_message("assistant", avatar=PURPLE_CIRCLE):
            st.error(f"Something went wrong: {str(e)[:500]}")
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Error: {str(e)[:300]}",
        })
    finally:
        st.session_state.processing = False
