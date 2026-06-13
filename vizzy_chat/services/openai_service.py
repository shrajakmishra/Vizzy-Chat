"""
Vizzy Chat — OpenAI Service
─────────────────────────────
Thin wrapper around the OpenAI Python SDK.
All other modules call this — never the SDK directly.
Provides chat completion and image generation primitives.
"""

from __future__ import annotations

from typing import Optional

from openai import OpenAI

import config
from utils.logger import get_logger

log = get_logger(__name__)

# ── Lazy-initialised client ────────────────────────────────────
_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    """Return (and cache) an OpenAI client instance."""
    global _client
    if _client is None:
        if not config.OPENAI_API_KEY:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. "
                "Please add it to vizzy_chat/.env"
            )
        _client = OpenAI(api_key=config.OPENAI_API_KEY)
        log.info("OpenAI client initialised.")
    return _client


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chat Completion
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chat_completion(
    messages: list[dict[str, str]],
    model: str = config.CHAT_MODEL,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    response_format: Optional[dict] = None,
) -> str:
    """
    Run a chat completion and return the assistant's text.

    Parameters
    ----------
    messages : list[dict]
        OpenAI-format messages (role + content).
    model : str
        Model identifier.
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens in the response.
    response_format : dict, optional
        E.g. {"type": "json_object"} for JSON mode.
    """
    client = _get_client()
    log.debug(f"Chat completion → model={model}, msgs={len(messages)}")

    kwargs: dict = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if response_format:
        kwargs["response_format"] = response_format

    response = client.chat.completions.create(**kwargs)
    text = response.choices[0].message.content or ""
    log.debug(f"Chat completion ← {len(text)} chars")
    return text.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Image Generation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_image(
    prompt: str,
    model: str = config.IMAGE_MODEL,
    size: str = "1024x1024",
    quality: str = "standard",
    style: str = "vivid",
    n: int = 1,
) -> list[str]:
    """
    Generate image(s) via DALL-E and return a list of URLs.

    Parameters
    ----------
    prompt : str
        The DALL-E prompt.
    model : str
        Image model identifier.
    size : str
        Image dimensions.
    quality : str
        'hd' or 'standard'.
    style : str
        'vivid' or 'natural'.
    n : int
        Number of images (DALL-E 3 supports 1 per call).
    """
    client = _get_client()
    log.debug(f"Image generation → model={model}, size={size}")

    urls: list[str] = []
    # DALL-E 3 only supports n=1, so loop if needed
    for _ in range(n):
        response = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            style=style,
            n=1,
        )
        url = response.data[0].url
        if url:
            urls.append(url)
            log.debug(f"Image generated: {url[:80]}…")

    return urls


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Embeddings (for future taste-memory similarity)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_embedding(
    text: str,
    model: str = config.EMBEDDING_MODEL,
) -> list[float]:
    """Return the embedding vector for *text*."""
    client = _get_client()
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding
