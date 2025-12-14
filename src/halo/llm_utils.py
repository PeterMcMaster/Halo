"""LLM helpers for query expansion and explanations."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Optional

import os

import google.generativeai as genai
from openai import OpenAI

from .config import get_config

EXPAND_PROMPT = (
    "You expand short photo search prompts into detailed descriptions mentioning colors,"
    " lighting, mood, and scene elements. Return a concise enriched sentence."
)

EXPLAIN_PROMPT = (
    "Given the user's description and photo metadata/caption, explain in no more than two"
    " sentences why the photo matches. Mention concrete visual cues (colors, lighting,"
    " scene elements) plus metadata like time or GPS if available."
)


def expand_query(prompt: str) -> str:
    response = _generate_text(EXPAND_PROMPT, prompt, max_tokens=128)
    return response or prompt


def explain_match(query: str, metadata: dict) -> str:
    caption = metadata.get("caption", "Unknown scene")
    dt_value = metadata.get("datetime_iso") or metadata.get("datetime_raw") or "Unknown time"
    lat = metadata.get("latitude")
    lon = metadata.get("longitude")
    location = "Unknown location"
    if lat is not None and lon is not None:
        location = f"Lat {lat:.3f}, Lon {lon:.3f}"

    context = (
        f"Query: {query}\n"
        f"Caption: {caption}\n"
        f"Timestamp: {dt_value}\n"
        f"Location: {location}"
    )

    response = _generate_text(EXPLAIN_PROMPT, context, max_tokens=160)
    if not response:
        return "Explanation unavailable (set GEMINI_API_KEY or OPENAI_API_KEY)."
    return response


# ---------------------------------------------------------------------------
# Provider routing
# ---------------------------------------------------------------------------


def _generate_text(instruction: str, user_content: str, max_tokens: Optional[int]) -> Optional[str]:
    cfg = get_config()
    provider = cfg.llm_provider.lower()

    if provider == "gemini" and cfg.gemini_api_key:
        result = _call_gemini(instruction, user_content, cfg.gemini_model, max_tokens)
        if result:
            return result
        # Fallback to OpenAI if Gemini fails but OpenAI is available.
        if cfg.openai_api_key:
            return _call_openai(instruction, user_content, cfg.openai_model, max_tokens)
    if provider == "openai" and cfg.openai_api_key:
        result = _call_openai(instruction, user_content, cfg.openai_model, max_tokens)
        if result:
            return result
        if cfg.gemini_api_key:
            return _call_gemini(instruction, user_content, cfg.gemini_model, max_tokens)
    if provider == "none":
        return None

    # Fallback order: OpenAI first (if key), then Gemini.
    if cfg.openai_api_key:
        return _call_openai(instruction, user_content, cfg.openai_model, max_tokens)
    if cfg.gemini_api_key:
        return _call_gemini(instruction, user_content, cfg.gemini_model, max_tokens)
    return None


@lru_cache(maxsize=1)
def _openai_client() -> Optional[OpenAI]:
    cfg = get_config()
    if not cfg.openai_api_key:
        return None
    return OpenAI(api_key=cfg.openai_api_key)


def _call_openai(instruction: str, user_content: str, model: str, max_tokens: Optional[int]) -> Optional[str]:
    client = _openai_client()
    if not client:
        return None
    try:
        kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": user_content},
            ],
        }
        if max_tokens is not None:
            kwargs["max_output_tokens"] = max_tokens
        completion = client.responses.create(**kwargs)
        return completion.output[0].content[0].text.strip()
    except Exception as exc:  # noqa: BLE001 - defensive: log and allow graceful fallback
        logging.warning("OpenAI call failed: %s", exc)
        return None


@lru_cache(maxsize=None)
def _normalize_gemini_model_name(name: str) -> str:
    """Normalize model aliases to current Gemini naming."""
    trimmed = (name or "").strip()
    # Map legacy / shorthand names to currently available identifiers.
    aliases = {
        # Legacy names -> current
        "gemini-1.5-flash-latest": "models/gemini-2.0-flash",
        "gemini-1.5-pro-latest": "models/gemini-2.0-pro",
        "gemini-1.5-flash": "models/gemini-2.0-flash",
        "gemini-1.5-pro": "models/gemini-2.0-pro",
        # Common short-hands without models/ prefix
        "gemini-2.5-flash": "models/gemini-2.5-flash",
        "gemini-2.5-pro": "models/gemini-2.5-pro",
        "gemini-2.0-flash": "models/gemini-2.0-flash",
        "gemini-2.0-flash-001": "models/gemini-2.0-flash-001",
        "gemini-flash-latest": "models/gemini-flash-latest",
        "gemini-flash-lite-latest": "models/gemini-flash-lite-latest",
        "gemini-pro-latest": "models/gemini-pro-latest",
        "gemini-2.5-flash-image": "models/gemini-2.5-flash-image",
        "gemini-2.5-flash-lite": "models/gemini-2.5-flash-lite",
    }
    normalized = aliases.get(trimmed, trimmed)
    if normalized.startswith("models/"):
        return normalized
    if normalized:
        # Add prefix if user omitted it for a new-style model name.
        if normalized.startswith("gemini-") or normalized.startswith("gemma-"):
            return f"models/{normalized}"
        return normalized
    return "models/gemini-2.5-flash"


@lru_cache(maxsize=None)
def _gemini_model(model_name: str):
    cfg = get_config()
    if not cfg.gemini_api_key:
        return None
    genai.configure(api_key=cfg.gemini_api_key)
    normalized = _normalize_gemini_model_name(model_name)
    return genai.GenerativeModel(normalized)


def _call_gemini(instruction: str, user_content: str, model_name: str, max_tokens: Optional[int]) -> Optional[str]:
    model = _gemini_model(model_name)
    if not model:
        return None
    prompt = f"{instruction}\n\nUser input:\n{user_content}\n\nResponse:"
    try:
        generation_config = {"max_output_tokens": max_tokens} if max_tokens is not None else None
        response = model.generate_content(
            prompt,
            generation_config=generation_config,
        )
        return response.text.strip()
    except Exception as exc:  # noqa: BLE001 - defensive: log and allow graceful fallback
        logging.warning("Gemini call failed: %s", exc)
        return None
