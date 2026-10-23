"""The two environment settings that change a still's seed outside its graph.

``OTR_RADIO_BOOKEND_SEED`` is the fixed seed of the radio bookend still and the
radio host's face. ``OTR_PORTRAIT_IDENTITY_SEED=0`` switches off the identity
seed a character's scene stills share with their portrait. One reader for
both, so the image dispatcher that paints with them and the Scene Producer's
seed lock that pins them for a project can never read them differently (Codex
QA, 2026-10-22: changed between two stills runs, either one repainted stills
nobody had edited).

UTF-8, no BOM. Standard library only.
"""
from __future__ import annotations

try:
    from . import env as otr_env
except ImportError:  # pragma: no cover -- flat (sys.path) load
    import env as otr_env  # type: ignore

#: The bookend seed when the environment names none, or names something unreadable.
DEFAULT_BOOKEND_SEED = 4242


def bookend_seed() -> int:
    """The seed the radio bookend still and the radio host's face paint with."""
    try:
        return int(otr_env.get("OTR_RADIO_BOOKEND_SEED", DEFAULT_BOOKEND_SEED))
    except (TypeError, ValueError):
        return DEFAULT_BOOKEND_SEED


def identity_seed_enabled() -> bool:
    """Whether a character's scene stills share their portrait's identity seed."""
    return otr_env.get("OTR_PORTRAIT_IDENTITY_SEED", "1") != "0"
