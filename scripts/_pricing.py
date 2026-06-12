"""OpenRouter $/MTok (input, output) at paper-submission time (2026-05-24).

Single source of truth for every cost computation in scripts/. Keyed by
both the full OpenRouter ID and the short alias from geoplanagent/agent/_model.py.
"""

PRICES: dict[str, tuple[float, float]] = {
    "google/gemini-3-flash-preview":    (0.55,  2.20),
    "gemini-flash":                     (0.55,  2.20),
    "google/gemini-3.1-pro-preview":    (1.25, 12.50),
    "gemini-pro":                       (1.25, 12.50),
    "anthropic/claude-opus-4.7":        (5.00, 25.00),
    "claude-opus":                      (5.00, 25.00),
    "openai/gpt-5.5-pro":               (30.0, 180.0),
    "gpt-5.5-pro":                      (30.0, 180.0),
}

DEFAULT_MODEL = "gemini-flash"
