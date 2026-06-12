"""OpenRouter $/MTok (input, output) at paper-submission time (2026-05-24).

Single source of truth for every cost computation in scripts/. Keyed by
the short alias from geoplanagent/utils.py.
"""

PRICES: dict[str, tuple[float, float]] = {
    "gemini-flash":                     (0.55,  2.20),
    "gemini-pro":                       (1.25, 12.50),
    "claude-opus":                      (5.00, 25.00),
    "gpt-5.5-pro":                      (30.0, 180.0),
}

DEFAULT_MODEL = "gemini-flash"
