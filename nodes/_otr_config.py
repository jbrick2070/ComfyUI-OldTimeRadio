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


# Default state of the style-grammar lever.
# 2026-06-24 (operator): flipped to TRUE -- the live A/B proved it improves the
# endings (the announcer stops narrating the news outcome; the climax leaves the
# console standoff and lands the injected ending in dialogue). The operator's
# directive: if it makes a better story, it should be the DEFAULT, not a lever
# users have to find and flip. ``OTR_ENABLE_STYLE_GRAMMAR=0`` (or false/no/off)
# is the kill-switch. See docs/2026-06-24-ending-mode/LIVE_AB_RESULTS.md.
STYLE_GRAMMAR_DEFAULT: bool = True


def style_grammar_enabled() -> bool:
    """Return True iff the STYLE-GRAMMAR lever (climax SHAPE selection) is on.

    Story-grammar build (2026-06-24). SEPARATE axis from L1/L2 (which shape the
    trope VOCABULARY): the style grammar deterministically picks a radio-drama
    style per episode (``_otr_style_catalog.select_style``) and feeds that
    style's ending taxonomy class as the climax ROLE -- demoting the forced
    ``irreversible_choice`` console standoff to one class among a closed set --
    plus injects the matching final-beat ending template at the climax beat and
    steers the announcer close off the news-outcome. Bundled with
    ``OTR_STORY_QUALITY_L12`` (the grammar runs the L1/L2 build path so the
    climax role can flow through ``build_sq_data``).

    DEFAULT ON (``STYLE_GRAMMAR_DEFAULT``) as of 2026-06-24: an UNSET (or empty)
    ``OTR_ENABLE_STYLE_GRAMMAR`` enables it; ``OTR_ENABLE_STYLE_GRAMMAR=0`` (or
    false/no/off) is the kill-switch that restores the exact pre-grammar render.
    """
    import os
    raw = (os.environ.get("OTR_ENABLE_STYLE_GRAMMAR") or "").strip().lower()
    if raw == "":
        return STYLE_GRAMMAR_DEFAULT
    return raw in ("1", "true", "yes", "on")


def composer_action_strip_enabled() -> bool:
    """Return True iff L3 (composer ACTION-marker strip) is on.

    Story-Quality LIFT (2026-06-23). Env-gated via ``OTR_COMPOSER_ACTION_STRIP``;
    DEFAULT OFF. AUDIO-AFFECTING (it can change shipped line text), so it ships
    dark and a golden re-baseline is required before it is enabled in a render.
    """
    import os
    raw = (os.environ.get("OTR_COMPOSER_ACTION_STRIP") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def transcript_sanitizer_enabled() -> bool:
    """Return True iff L4 (minimal transcript sanitizer) is on.

    Story-Quality LIFT (2026-06-23). Env-gated via ``OTR_TRANSCRIPT_SANITIZER``;
    DEFAULT OFF. AUDIO-AFFECTING (final-text hygiene), so it ships dark and a
    golden re-baseline is required before it is enabled in a render.
    """
    import os
    raw = (os.environ.get("OTR_TRANSCRIPT_SANITIZER") or "").strip().lower()
    return raw in ("1", "true", "yes", "on")


def leak_floor_v2_enabled() -> bool:
    """Return True iff the leak-floor-v2 deterministic line verifier is on.

    Leaking-words build (2026-06-25, ``docs/2026-06-25-leaking-words/``). One
    mandatory deterministic offline verifier over four named leak classes
    (capitalised-participle-before-a-quote extract; ALL-CAPS roster-vocative
    drop; double-quote-only malformed-internal check; news-bleed real-person /
    political-figure filtering at ``build_allowed_roster``).

    DEFAULT ON (operator directive 2026-06-26: ship the quality fixes in the
    canonical end-to-end workflow, no waiting for a per-lane validation gate --
    the leaks degrade the story, e.g. the announcer SPEAKING a leaked stage
    participle / vocative / scaffold). The rules are deterministic + NARROW and
    negative-tested (first-name vocative, legit orgs, non-stage ``-ing`` openings
    are NOT touched), so the false-positive risk that held it dark is low.
    AUDIO-AFFECTING by design -- it scrubs the leaked text out of the spoken line,
    so a render with a leak is intentionally NO LONGER byte-identical to the
    pre-fix (leaky) render; that is the point. Opt OUT for an A/B with
    ``OTR_ENABLE_LEAK_FLOOR_V2=0``.
    """
    import os
    raw = (os.environ.get("OTR_ENABLE_LEAK_FLOOR_V2") or "on").strip().lower()
    return raw not in ("0", "false", "no", "off")


def strict_local_clean_enabled() -> bool:
    """Return True iff leak-floor-v2 runs in FAIL-CLOSED (strict) mode.

    When on, the verifier's correctness layer is mandatory: an unrepairable
    leak (e.g. a malformed internal quote with the per-line recompose budget
    exhausted) marks the ``VerificationResult`` as ``failed`` so CI / release
    promotion can assert 0 leaks. DEFAULT OFF => best-effort: ship the
    best-effort cleaned text + telemetry (the pre-promotion dark window).
    Env-gated via ``OTR_STRICT_LOCAL_CLEAN``. Independent of
    ``leak_floor_v2_enabled`` (strict has no effect while the verifier is off).
    """
    import os
    raw = (os.environ.get("OTR_STRICT_LOCAL_CLEAN") or "").strip().lower()
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
