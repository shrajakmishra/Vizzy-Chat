"""
Vizzy Chat — Image Service
────────────────────────────
High-level image operations built on top of openai_service.
Handles variation generation, prompt refinement for visuals,
and image download / caching.
"""

from __future__ import annotations

import io
import base64
from typing import Optional

import requests
from PIL import Image

import config
from services.openai_service import chat_completion, generate_image
from utils.prompt_builder import build_image_prompt
from utils.logger import get_logger

log = get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Prompt Refinement via LLM
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _refine_dalle_prompt(raw_prompt: str) -> str:
    """
    Use GPT-4o-mini to refine a structured prompt into a concise,
    high-quality DALL-E prompt (≤ 950 chars). Uses a fast model
    to keep latency low.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a DALL-E prompt engineer. Given a structured creative "
                "brief, output a single, vivid, concise image prompt under 950 "
                "characters that will produce a stunning, non-generic image. "
                "Do NOT include any explanation — only the prompt text."
            ),
        },
        {"role": "user", "content": raw_prompt},
    ]
    refined = chat_completion(
        messages, model="gpt-4o-mini", temperature=0.8, max_tokens=300
    )
    log.debug(f"Refined DALL-E prompt ({len(refined)} chars)")
    return refined


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_variations(
    user_request: str,
    mode: str,
    num_variations: int = config.DEFAULT_VARIATIONS,
    preferences: Optional[dict] = None,
    refinement_delta: Optional[str] = None,
    progress_callback: Optional[object] = None,
) -> list[dict]:
    """
    Generate *num_variations* distinct image interpretations.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with (step_label: str) to report progress.

    Returns a list of dicts:
        [{"url": str, "prompt_used": str, "variation": int}, ...]
    """
    results: list[dict] = []

    for i in range(num_variations):
        if progress_callback:
            progress_callback(
                f"Crafting visual direction {i + 1} of {num_variations}…"
            )

        # 1. Build structured prompt
        raw = build_image_prompt(
            user_request=user_request,
            mode=mode,
            variation_index=i,
            total_variations=num_variations,
            preferences=preferences,
            refinement_delta=refinement_delta,
        )

        # 2. Refine into DALL-E-ready prompt
        if progress_callback:
            progress_callback(
                f"Refining creative prompt for variation {i + 1}…"
            )
        dalle_prompt = _refine_dalle_prompt(raw)

        # 3. Generate
        if progress_callback:
            progress_callback(
                f"Generating image {i + 1} of {num_variations}… (this takes ~15s)"
            )
        try:
            urls = generate_image(dalle_prompt)
            if urls:
                results.append(
                    {
                        "url": urls[0],
                        "prompt_used": dalle_prompt,
                        "variation": i + 1,
                    }
                )
                log.info(f"Variation {i + 1}/{num_variations} generated.")
        except Exception as e:
            log.error(f"Image generation failed for variation {i+1}: {e}")
            continue

    return results


def download_image_as_bytes(url: str) -> bytes:
    """Download an image URL and return raw bytes."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.content


def image_to_base64(url: str) -> str:
    """Download an image and return a base64-encoded string."""
    raw = download_image_as_bytes(url)
    return base64.b64encode(raw).decode("utf-8")


def url_to_pil(url: str) -> Image.Image:
    """Download an image URL and return a PIL Image object."""
    raw = download_image_as_bytes(url)
    return Image.open(io.BytesIO(raw))
