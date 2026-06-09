"""Model-name resolution + OpenRouterModel construction.

The alias table and model construction live only here, so reader,
worker, locate sub-agent and critic all go through the same path.
"""

from __future__ import annotations

from pydantic_ai.models.openrouter import OpenRouterModel


MODEL_ALIASES = {
    "claude-opus": "anthropic/claude-opus-4.7",
    "gpt-5.5-pro": "openai/gpt-5.5-pro",
    "gemini-pro": "google/gemini-3.1-pro-preview",
    "gemini-flash": "google/gemini-3-flash-preview",
}


def resolve_model_name(name: str) -> str:
    """Map a short alias (gemini-flash, claude-sonnet, …) to a full
    OpenRouter model identifier. Already-qualified IDs (containing "/")
    or unknown aliases pass through unchanged."""
    return MODEL_ALIASES.get(name, name)


def resolve_model(name: str) -> OpenRouterModel:
    """Convenience: resolve alias + construct OpenRouterModel."""
    return OpenRouterModel(resolve_model_name(name))
