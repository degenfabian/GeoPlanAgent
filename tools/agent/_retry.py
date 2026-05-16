"""Transient-HTTP-error retry helper for OpenRouter calls.

Gemini-Flash (and other OpenRouter-routed models) hits transient 400s
("Provider returned error") that recover on retry. Without this wrapper
a single mid-conversation hiccup kills the whole case and we lose the
agent's accumulated state.
"""

from __future__ import annotations

import re
import time as _time


# Status codes that are typically transient and worth retrying. 400 is
# included because OpenRouter routinely surfaces upstream Gemini hiccups
# (rate limit, model overload, transient safety-check backend failures)
# as a generic 400 with body "Provider returned error".
_RETRYABLE_STATUS = {400, 408, 425, 429, 500, 502, 503, 504}


def _is_retryable_http_error(exc: Exception) -> bool:
    """True if this exception looks like a transient OpenRouter/provider hiccup."""
    try:
        from pydantic_ai.exceptions import ModelHTTPError
    except Exception:
        return False
    if not isinstance(exc, ModelHTTPError):
        return False
    s = str(exc)
    m = re.search(r"status_code:\s*(\d+)", s)
    if not m:
        return False
    return int(m.group(1)) in _RETRYABLE_STATUS


def _run_sync_with_retry(agent_obj, *args, max_retries: int = 2,
                          backoff_s: float = 5.0, label: str = "agent",
                          **kwargs):
    """Wrap Agent.run_sync with retries on transient HTTP errors.

    Non-retryable errors (auth, bad input, ModelRetry / UnexpectedModelBehavior)
    are re-raised immediately so we don't waste cycles.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return agent_obj.run_sync(*args, **kwargs)
        except Exception as e:
            if not _is_retryable_http_error(e) or attempt == max_retries:
                raise
            wait = backoff_s * (2 ** attempt)
            print(f"  {label}: transient HTTP error (attempt "
                  f"{attempt + 1}/{max_retries + 1}): {str(e)[:140]}"
                  f" — retrying in {wait:.0f}s")
            _time.sleep(wait)
            last_exc = e
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label}: retry loop fell through without error")
