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

# Default state of the story-quality-v2 spine. FALSE => the new craft levers
# are dormant and the pipeline is byte-identical to pre-R3.
STORY_QUALITY_V2_DEFAULT: bool = False

# L2 authoring contract: the minimum beat_tension (1..5) at which a character
# line withholds its literal Objective and is asked to play the deflection.
# Below this, the line states its objective normally (unchanged behaviour).
OBJECTIVE_DEFLECTION_TENSION_MIN: int = 4


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
