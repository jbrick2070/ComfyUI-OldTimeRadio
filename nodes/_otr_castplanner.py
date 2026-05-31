"""nodes/_otr_castplanner.py -- immutable CastPlanner slot schema (S4).

Python owns cast STRUCTURE; the LLM only names + textures. The CastPlanner
freezes the structural decision -- char_id / gender / age_band / voice_preset /
dramatic_role -- into an immutable slot the LLM Pass-1 call (S6) writes a name
and texture against, and the validator (S7) referees. The LLM may not alter any
field on this slot.

This is a pure helper module, NOT a registered ComfyUI node: it introduces no
widget / INPUT_TYPES / socket, so it needs no workflow-JSON wiring (PD3 N/A).
It is inert unless OTR_NAME_MODE=llm_slot_fill (pool mode never builds a plan),
so pool-mode behavior is byte-identical (C7).

Frozen S0 contract (docs/2026-05-30-cast-system__go-forward-sprint-plan.md):

    { "char_id": "c02", "gender": "female", "age_band": "middle_adult",
      "voice_preset": "v2/en_speaker_4", "dramatic_role": "skeptical mission doctor" }

age_band enum: young_adult | middle_adult | older_adult | ageless
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

AGE_BANDS: tuple[str, ...] = (
    "young_adult", "middle_adult", "older_adult", "ageless",
)

# The deterministic ensemble rotates the three concrete bands across slots so
# an episode spans ages instead of clustering; "ageless" is reserved (never
# auto-assigned) for callers/LLM that want an age-neutral character.
_ROTATING_BANDS: tuple[str, ...] = (
    "young_adult", "middle_adult", "older_adult",
)

# age_band -> the VOICE_PROFILES age tags it prefers (S5 voice x age). Used as a
# ranking hint only; an empty preference (ageless) means "no age preference".
AGE_BAND_VOICE_TAGS: Dict[str, frozenset] = {
    "young_adult": frozenset({"20s", "30s"}),
    "middle_adult": frozenset({"30s", "40s", "50s"}),
    "older_adult": frozenset({"50s", "60s"}),
    "ageless": frozenset(),
}


@dataclass(frozen=True)
class CastPlanSlot:
    """One immutable planned slot. The LLM writes name + texture against this;
    it may not change any field here (the validator enforces that)."""

    char_id: str
    gender: str        # male | female | other
    age_band: str      # one of AGE_BANDS
    voice_preset: str  # the Python-decided Bark/Kokoro preset
    dramatic_role: str  # a dramatic function, not a job title


def age_band_for_index(index: int) -> str:
    """Deterministic age_band for an open-slot index -- rotates the three
    concrete bands so the ensemble spans ages. Pure, no rng."""
    return _ROTATING_BANDS[index % len(_ROTATING_BANDS)]


def build_cast_plan(
    ensemble_slots,
    voice_by_char_id: Dict[str, str],
    *,
    age_band_by_char_id: Optional[Dict[str, str]] = None,
) -> List[CastPlanSlot]:
    """Build the immutable CastPlanner slots from the deterministic ensemble.

    `ensemble_slots`: the EnsembleSlot list from precompute_ensemble_slots
        (carries char_id / gender / timbre / role).
    `voice_by_char_id`: the Python-assigned voice_preset per char_id (from
        python_assign_voice_preset in the lock_cast loop).
    `age_band_by_char_id`: optional explicit age bands; otherwise derived
        deterministically by slot index.

    dramatic_role is the ensemble's role (lead/foil/support/wildcard) textured
    with the slot timbre so the LLM has a concrete hook ("a sharp foil") without
    the planner prescribing a job title.
    """
    plan: List[CastPlanSlot] = []
    for i, ens in enumerate(ensemble_slots):
        if age_band_by_char_id and ens.char_id in age_band_by_char_id:
            age_band = age_band_by_char_id[ens.char_id]
        else:
            age_band = age_band_for_index(i)
        role = (getattr(ens, "role", "") or "").strip()
        timbre = (getattr(ens, "timbre", "") or "").strip()
        dramatic_role = (f"{timbre} {role}".strip() or role) if (role or timbre) else "character"
        plan.append(CastPlanSlot(
            char_id=ens.char_id,
            gender=ens.gender,
            age_band=age_band if age_band in AGE_BANDS else "ageless",
            voice_preset=voice_by_char_id.get(ens.char_id, ""),
            dramatic_role=dramatic_role,
        ))
    return plan
