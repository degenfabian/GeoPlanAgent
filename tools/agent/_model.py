"""Model-name resolution + OpenRouterModel construction.

Single source of truth for the alias table and the model object so
reader / worker / locate sub-agent / critic all go through the same
path.
"""

from __future__ import annotations

from pydantic_ai.models.openrouter import OpenRouterModel


MODEL_ALIASES = {
    "claude-opus": "anthropic/claude-opus-4.6",
    "claude-sonnet": "anthropic/claude-sonnet-4-6",
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.4-mini": "openai/gpt-5.4-mini",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
    "gemini-pro": "google/gemini-3.1-pro-preview",
    "gemini-flash": "google/gemini-3-flash-preview",
    "gemini-flash-lite": "google/gemini-3.1-flash-lite-preview",
}


def resolve_model_name(name: str) -> str:
    """Map a short alias (gemini-flash, claude-sonnet, …) to a full
    OpenRouter model identifier. Already-qualified IDs (containing "/")
    or unknown aliases pass through unchanged."""
    return MODEL_ALIASES.get(name, name)


def resolve_model(name: str) -> OpenRouterModel:
    """Convenience: resolve alias + construct OpenRouterModel."""
    return OpenRouterModel(resolve_model_name(name))
