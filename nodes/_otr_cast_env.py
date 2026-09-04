"""nodes/_otr_cast_env.py -- frozen env-var contract for the cast name system.

All cast knobs are environment variables; there are NO new ComfyUI widgets
(per the cast-system sprint plan, "Frozen interface contracts"). Every default
reproduces the pre-fix behavior exactly, so a run that sets none of these is
byte-identical to the baseline.

  OTR_NAME_MODE              pool | llm_slot_fill          (default: pool)
  OTR_CAST_GENRE             auto | scifi_1950s | noir | space_opera (default: auto)
  OTR_NAME_CROSS_GENDER_RATE float 0.0..1.0               (default: 0.0 = strict)
  OTR_OTHER_NAME_POLICY      unisex | all                 (default: unisex)

Centralizing the reads here (rather than scattering os.environ.get calls) keeps
the contract auditable and gives every consumer the same parsing + clamping.
"""
from __future__ import annotations


try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

_VALID_MODES = ("pool", "llm_slot_fill")
def name_mode() -> str:
    """OTR_NAME_MODE: 'pool' (deterministic pool fill, default) or
    'llm_slot_fill' (schema-locked LLM naming with pool fallback)."""
    v = (otr_env.get("OTR_NAME_MODE", "pool") or "pool").strip().lower()
    return v if v in _VALID_MODES else "pool"


def cross_gender_rate() -> float:
    """OTR_NAME_CROSS_GENDER_RATE in [0.0, 1.0].

    0.0 (default, strict): every binary name/gender mismatch is repaired.
    >0:  with this probability (drawn from the per-character ISOLATED rng, so
         still fully deterministic) a mismatch is LEFT as a deliberate
         non-stereotypical name. Out-of-range / unparseable -> 0.0."""
    raw = (otr_env.get("OTR_NAME_CROSS_GENDER_RATE", "0.0") or "0.0").strip()
    try:
        v = float(raw)
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, v))

