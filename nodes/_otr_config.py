"""nodes/_otr_config.py -- OTR cross-module configuration constants.

A tiny, dependency-free leaf module (stdlib only) so any node can import a
shared tunable without forming an import cycle. Constants here are the single
source of truth for cross-cutting feature flags + thresholds.

STORY-QUALITY V2 (2026-06-22, R3)
---------------------------------
The dialogue-craft spine (L1 objective-literal gate, L2 authoring contract,
L7 dialogue|action split, + telemetry) is gated behind a SINGLE per-episode
feature flag carried on the ledger meta:

    meta["story_quality_v2_enabled"]  (bool, default False)

Flag OFF => byte-identical to the pre-R3 pipeline: no new code path runs, no
new compose_flags are minted, and no ``meta.story_quality`` summary object is
written. The flag is read at every new seam via ``story_quality_v2_enabled``.

The default is FALSE: the spine ships dark and is enabled deliberately
(operator config / a future workflow toggle), never by accident.
"""
from __future__ import annotations

from typing import Any, Mapping

# Default state of the story-quality-v2 spine.
# 2026-06-23 (operator): flipped to TRUE -- the R3 soak proved the spine is
# stable + harmless (15+ clean episodes, continuity issues ~0, no test/golden
# impact) even though it is not a measurable quality LIFT on weak local writers.
# Left ON so L1/L7 stay armed as a defect safety-net + telemetry keeps
# accruing; OTR_STORY_QUALITY_V2=0 (or false/no/off) is the kill-switch.
STORY_QUALITY_V2_DEFAULT: bool = True

# L2 authoring contract: the minimum beat_tension (1..5) at which a character
# line withholds its literal Objective and is asked to play the deflection.
# Below this, the line states its objective normally (unchanged behaviour).
OBJECTIVE_DEFLECTION_TENSION_MIN: int = 4


def story_quality_l12_enabled() -> bool:
    """Return True iff the L1/L2 deterministic beat-shaping lever is on.

    Story-Quality LIFT (2026-06-23). SEPARATE from the ``story_quality_v2``
    meta spine: L12 reshapes the BEAT PLAN upstream (premise-anchored
    conflict objects/types + dramatic beat_role sequence + crisis-noun
    grounding), so the weak local writer cannot collapse every premise into
    the same "console standoff". Env-gated via ``OTR_STORY_QUALITY_L12``;
    DEFAULT OFF (1/true/yes/on => on). When OFF, no SQ field is populated,
    no ``meta.story_quality`` key is written, and the prompt is byte-
    identical to the pre-LIFT pipeline.
    """
    import os
    raw = (os.environ.get("OTR_STORY_QUALITY_L12") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def story_quality_v2_enabled(meta: Any) -> bool:
    """Return True iff the story-quality-v2 spine is enabled for this episode.

    Reads ``meta["story_quality_v2_enabled"]`` (a plain bool on the ledger
    meta dict). Defensive: a missing key, a non-mapping ``meta``, or any
    error falls back to ``STORY_QUALITY_V2_DEFAULT`` (False). The flag is a
    real bool -- it is NEVER inferred from generated text.
    """
    try:
        if isinstance(meta, Mapping):
            return bool(meta.get("story_quality_v2_enabled", STORY_QUALITY_V2_DEFAULT))
    except Exception:  # noqa: BLE001 -- a malformed meta is never fatal
        pass
    return STORY_QUALITY_V2_DEFAULT
