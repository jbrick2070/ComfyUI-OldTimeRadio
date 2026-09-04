"""Cross-module settings for first-call story generation guidance.

These levers shape prompts before prose is authored. They never inspect,
accept, reject, trim, replace, or re-author a completed story.
"""
from __future__ import annotations


try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

OBJECTIVE_DEFLECTION_TENSION_MIN: int = 4
STYLE_GRAMMAR_DEFAULT: bool = True


def style_grammar_enabled() -> bool:
    """Return whether the style-selected ending grammar guides generation."""
    raw = (otr_env.get("OTR_ENABLE_STYLE_GRAMMAR") or "").strip().lower()
    if raw == "":
        return STYLE_GRAMMAR_DEFAULT
    return raw in ("1", "true", "yes", "on")


__all__ = [
    "OBJECTIVE_DEFLECTION_TENSION_MIN",
    "STYLE_GRAMMAR_DEFAULT",
    "style_grammar_enabled",
]
