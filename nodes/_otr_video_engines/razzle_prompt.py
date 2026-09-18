"""Shared raised-razzle motion policy -- ONE owner for cloud and local.

The July 3 living-poster default (gentle / mist / subtle parallax) is retired.
A video lane moves. A hold belongs on still_word / still_flat. Artifact
guards stay: whip pans, melting geometry, warped faces, drifting text.
"""
from __future__ import annotations

try:
    from .._otr_shared import env as otr_env
    from .._otr_story_brief_helpers import (
        append_visual_safety_clause, visual_safety_negative)
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore
    from _otr_story_brief_helpers import (  # type: ignore
        append_visual_safety_clause, visual_safety_negative)

MOTION_ENV = "OTR_CLOUD_RAZZLE_MOTION_PROMPT"
NEG_ENV = "OTR_CLOUD_RAZZLE_NEG"

MOTION_DEFAULT = (
    "Generate one continuous shot from the first frame. Preserve the "
    "first-frame subject, composition, lighting, and visual style. The "
    "subject carries out a full, decisive action that develops across the "
    "shot -- turning, reaching, rising, or crossing the space -- and finishes "
    "on a clear final position, with a purposeful camera move. Motion "
    "begins immediately in the first frame and is sustained throughout. No "
    "whip pans, handheld shake, sudden reframing, jump cuts, rapid zooms, "
    "melting geometry, warped faces, drifting text, or unreadable lettering")

NEG_EXTRA = (
    "warped text, melting letters, distorted typography, "
    "illegible words, garbled text, flickering letters, "
    "static hold, frozen frame, no motion")

BANNED_DAMPING = ("subtle", "gentle", "drifting mist", "soft neon")


def motion_clause(override: str | None = None) -> str:
    """Env override, then the raised default. Empty override is ignored."""
    pinned = str(override or "").strip()
    if pinned:
        return pinned
    env = otr_env.get(MOTION_ENV, "").strip()
    return env or MOTION_DEFAULT


def compose_positive(beat, *, override: str | None = None) -> str:
    """Raised motion clause LEADS; the beat text is appended when present."""
    motion = motion_clause(override)
    text = str(beat or "").strip()
    prompt = f"{motion}. {text}".strip().rstrip(".") if text else motion
    return append_visual_safety_clause(prompt)


def compose_negative(*parts: str) -> str:
    """Merge engine extras with the razzle artifact / no-hold terms.

    ``OTR_CLOUD_RAZZLE_NEG`` replaces the razzle extras entirely -- same
    as the old cloud adapter -- and does not silently re-append NEG_EXTRA.
    Caller-supplied ``parts`` (the LTX recipe negative) still merge.
    """
    env = otr_env.get(NEG_ENV, "").strip()
    extra = env or NEG_EXTRA
    head = ", ".join(str(p).strip() for p in parts if str(p or "").strip())
    return visual_safety_negative(
        ", ".join(p for p in (head, extra) if p))


def damping_hits(text: str) -> tuple[str, ...]:
    lowered = str(text or "").lower()
    return tuple(token for token in BANNED_DAMPING if token in lowered)
