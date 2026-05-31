r"""
nodes/_otr_casting.py -- cast contract caller (PROSE-PLANE).

Plane assignment: PROSE (Content LLM). The cast row carries an
audience-facing field (character_description) written by the same
model that writes dialogue, polishes title, and generates visual
prompts. Voice-pool validation and ensemble balance are Python
concerns; they are NOT a reason to route casting to the structural
LLM.
See `spaces/.../memory/project_cast_contract_architecture_target.md`
and `spaces/.../memory/project_llm_agnostic_design_constraint.md`
for the full architectural spec.

Sprint 3D -- three-stage split. Casting used to be ONE LLM call per
open character that produced character_description + gender +
voice_preset together: the LLM picked the voice and there was no
Python-side global gender/timbre/role balance, only a static prompt
line "~40% male / ~40% female / ~20% other". Sprint 3D moves balance
and voice selection out of the LLM:

  1. precompute_ensemble_slots -- PURE PYTHON. Decides the whole
     ensemble's gender / timbre / role distribution up front. Python
     owns balance now, not the LLM.
  2. llm_write_description -- the LLM writes ONLY the prose character
     description for one slot. It no longer picks gender or voice.
  3. python_assign_voice_preset -- PURE PYTHON. Picks the voice preset
     from the pre-filtered pool by gender + timbre, per slot.

Net effect: voice selection and gender/timbre/role balance leave the
LLM; the LLM's per-character job shrinks to description-only. The
total LLM call count is unchanged-or-lower -- still at most one call
per open character, and no extra call site is added.

Voice collisions remain impossible by construction: Python pre-filters
`available_voices = full_pool - taken_voices` before each slot, and
python_assign_voice_preset draws only from that pre-filtered set. The
post-cast `_assert_unique_bark_voices` invariant is the existing
uniqueness guard and is NOT duplicated here -- python_assign_voice_preset
owns DISTRIBUTION, not a second uniqueness check.

Cast assembly order (Python-only, no LLM):
  1. ANNOUNCER pre-baked at char_id="c01"  (always present, bonus)
  2. LEMMY pre-baked at char_id="c02"      (11% roll, consumes a slot)
  3. Pool-fill open characters             (precompute -> describe ->
                                            assign-voice, c02..cNN
                                            shifted +1 if LEMMY hit)

Era-agnostic: every prompt this module emits passes news_seed AND
style as the only flavor inputs; no hardcoded period literals
appear in any prompt string.

LLM-agnostic: lean prompts, no model-specific instructions, no
chat-template assumptions. Validator + 3-attempt reroll is the
cross-model safety net. See
`spaces/.../memory/feedback_keep_local_llm_prompts_short.md`.
"""
from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field, field_validator

# Shared tolerant JSON extractor (BUG-LOCAL-261 consolidation). Package
# import in production; flat import when loaded standalone / under test.
try:
    from . import _otr_json
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_json  # type: ignore

# Sprint 2A/2D: the shared structured-JSON retry ladder. cast_one_character
# routes its per-character LLM call through it. Package import in
# production; flat import when loaded standalone / under test.
try:
    from ._otr_structured_call import (
        structured_call,
        StructuredCallFailedError,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_structured_call import (  # type: ignore
        structured_call,
        StructuredCallFailedError,
    )

# Sprint 2C: typed repair-prompt factories. cast_one_character passes a
# dispatching factory so structured_call's Attempt 3 routes the repair
# turn by failure class. Package import in production; flat import when
# loaded standalone / under test.
try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore

# Import the cast pools. Try relative first (production: ComfyUI loads
# this as part of the ComfyUI-OldTimeRadio package); fall back to
# absolute (tests: tests/ adds nodes/'s parent to sys.path).
try:
    from ..config import cast_pools as _POOLS  # type: ignore[no-redef]
except (ImportError, ValueError):
    import sys
    from pathlib import Path
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from config import cast_pools as _POOLS  # type: ignore[no-redef]

# Frozen cast env-var contract (S0). Package import in production; flat import
# when loaded standalone / under test.
try:
    from . import _otr_cast_env
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_cast_env  # type: ignore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_VALID_GENDERS = {"male", "female", "other"}

# Sprint 3D: the "other" share of the ensemble gender split renders
# through the Bark voice pool (which today is binary male/female), so
# an "other"-gender slot is voiced from whichever gender column has the
# most headroom. Gender on the cast row is the audience-facing label;
# the voice pool is a TTS implementation detail.
_DEFAULT_GENDER_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("male", 0.40),
    ("female", 0.40),
    ("other", 0.20),
)


class DescriptionResponse(BaseModel):
    """LLM response shape for one open-character DESCRIPTION call.

    Sprint 3D: the LLM's per-character job is description-only. It no
    longer picks gender or voice_preset -- Python owns the ensemble
    gender/timbre/role distribution (precompute_ensemble_slots) and
    voice selection (python_assign_voice_preset). This schema is the
    full surface the LLM is now asked to produce.
    """

    # BUG-LOCAL-263 (2026-05-24): max_length 200 -> 750. The casting
    # prompt only ever showed the model the placeholder "<short>" --
    # it never stated a character count -- so a normal 1-2 sentence
    # description routinely overran the old 200 cap and aborted the
    # run. 750 is a generous runaway guard, not a content target;
    # _format_prior_entry already trims the echoed description to
    # 60 chars, so the stored length never touches prompt budget.
    character_description: str = Field(..., min_length=10, max_length=750)

    @field_validator("character_description")
    @classmethod
    def _strip_desc(cls, v: str) -> str:
        return v.strip()


class CastingResponse(BaseModel):
    """Assembled casting result for one open character.

    Sprint 3D: this is no longer a raw LLM-response shape. The LLM
    now produces only `character_description` (see DescriptionResponse);
    `gender` is decided by precompute_ensemble_slots and `voice_preset`
    by python_assign_voice_preset, both pure Python. cast_one_character
    composes the three stages and returns this combined object so the
    writer-facing contract (one CastingResponse per open slot) is
    preserved.

    Voice-in-pool validation is NOT done here -- pydantic cannot see
    the runtime available_voices set. python_assign_voice_preset draws
    from the pre-filtered pool, so a pool miss is impossible by
    construction.
    """

    character_description: str = Field(..., min_length=10, max_length=750)
    gender: str = Field(..., min_length=3, max_length=12)
    voice_preset: str = Field(..., min_length=3, max_length=80)

    @field_validator("gender")
    @classmethod
    def _gender_in_set(cls, v: str) -> str:
        v_norm = v.strip().lower()
        if v_norm not in _VALID_GENDERS:
            raise ValueError(
                f"gender must be one of {sorted(_VALID_GENDERS)}, got {v!r}"
            )
        return v_norm

    @field_validator("character_description")
    @classmethod
    def _strip_desc(cls, v: str) -> str:
        return v.strip()

    @field_validator("voice_preset")
    @classmethod
    def _strip_voice(cls, v: str) -> str:
        return v.strip()


# ---------------------------------------------------------------------------
# Error class
# ---------------------------------------------------------------------------


class CastingFailedError(RuntimeError):
    """Raised after cast_one_character exhausts all retry attempts.

    Attributes:
        attempts: list of (raw_response, error_message) tuples per attempt
        name: the character name being cast
    """

    def __init__(
        self,
        attempts: List[tuple[str, str]],
        name: str,
    ) -> None:
        self.attempts = attempts
        self.name = name
        last_err = attempts[-1][1] if attempts else "no attempts"
        super().__init__(
            f"Casting failed for {name!r} after {len(attempts)} "
            f"attempts. Last error: {last_err}"
        )


class CastValidationLLMError(CastingFailedError):
    """S32 B3 (D2) -- schema-validation (repair) pass failed.

    The repair attempt routes to the technical slot (single attempt,
    fail-fast, no internal retry). When the technical-slot output
    fails validation, this exception fires instead of the generic
    `CastingFailedError`. Subclass so existing handlers catching the
    base class still match; new handlers can branch on this specific
    type to trigger writer-side creative regen rather than a hard
    failure.
    """
    pass


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
# The naive first-'{'-to-last-'}' extractor was removed in the
# BUG-LOCAL-261 consolidation. Cast JSON is now parsed via the shared
# _otr_json.parse_first_json_object, which takes the first complete
# object and tolerates a trailing second object / prose.


# ---------------------------------------------------------------------------
# Prompt builder -- LEAN, no model-specific tweaks
# ---------------------------------------------------------------------------

_NEWS_SEED_CAP = 500


def _build_user_prompt(
    name: str,
    news_seed: str,
    style: str,
    prior_cast: List[dict],
    *,
    gender: str = "",
    timbre: str = "",
    role: str = "",
    casting_brief: str = "",
) -> str:
    """Build the description prompt for one open character.

    Sprint 3D: the LLM writes ONLY the prose character description.
    Python has already decided this slot's gender / timbre / role
    (precompute_ensemble_slots); those are handed to the LLM as fixed
    facts to write into, NOT choices to make. The 'Voices:' block and
    the 'Aim ~40/40/20' balance line are gone -- voice selection and
    ensemble balance are now pure-Python concerns.

    Layout (every line except 'Cast so far' is mandatory; the cast-
    so-far block is omitted entirely when prior_cast is empty):

      Write a character for a radio drama.
      Story: <casting_brief if non-empty else news_seed[:500]>
      Style: <style>

      Name: <NAME>
      Gender: <male|female|other>
      Voice: <timbre>
      Role: <role>

      [optional]
      Cast so far:
      - LEMMY (M, gravelly engineer, 50s, gruff mechanic)
      - BOB   (M, weary doctor, 40s, dry humor)

      JSON only:
      {"character_description":"<vivid, 1-2 sentences>"}

    The Gender / Voice / Role lines are emitted only when the caller
    supplies them; legacy callers and tests that pass none still get a
    well-formed description prompt.

    casting_brief (added in commit 3 of the news_interpreter sprint,
    ADR docs/news_interpreter_adr.md) is the purpose-specific
    distillation of the article for casting -- "what kinds of people
    belong in this story". When provided (non-empty), it replaces the
    mechanical 500-char slice of news_seed on the Story: line. When
    absent, the legacy slice still runs so older callers and tests
    keep their behavior.
    """
    brief = (casting_brief or "").strip()
    if brief:
        story_text = brief
    else:
        story_text = (news_seed or "").strip()[:_NEWS_SEED_CAP]
    style_str = (style or "").strip() or "open"

    parts: list[str] = [
        "Write a character for a radio drama.",
        f"Story: {story_text}",
        f"Style: {style_str}",
        "",
        f"Name: {name}",
    ]
    # Gender / timbre / role are Python-decided facts the LLM writes
    # into -- emitted only when the caller supplies them so legacy
    # callers keep a lean prompt.
    if (gender or "").strip():
        parts.append(f"Gender: {gender.strip()}")
    if (timbre or "").strip():
        parts.append(f"Voice: {timbre.strip()}")
    if (role or "").strip():
        role_clean = role.strip()
        parts.append(f"Role: {role_clean}")
        # Sprint 9 (Narrative Face): Python-pinned face-pressure anchor.
        # Same pattern as the timbre/role rotation above -- Python
        # decides the fact (which dramatic pressure the role implies),
        # the LLM writes facial geometry that earns it. The lookup
        # falls through silently when the caller passes a role outside
        # _FACE_PRESSURE_BY_ROLE so future role-vocab extensions don't
        # break this surface.
        pressure = _FACE_PRESSURE_BY_ROLE.get(role_clean.lower(), "")
        if pressure:
            parts.append(f"Face pressure: {pressure}")

    if prior_cast:
        parts.append("")
        parts.append("Cast so far:")
        for c in prior_cast:
            parts.append(f"- {_format_prior_entry(c)}")

    # Sprint 9 (Narrative Face): CHARACTER VISUAL CONTRACT block.
    # The casting LLM's output is the single source of truth for BOTH
    # voice-cast prose AND FLUX portrait composition (the writer K.5
    # copies character_description verbatim into
    # meta.visual_plan.characters[NAME].portrait_prompt). For FLUX to
    # paint visually distinct faces, the description has to carry
    # concrete facial geometry instead of mood adjectives. The
    # CONTRACT below tells the LLM *how* to compose; the JSON template
    # shows the format it must fit. Explicit rules survive model
    # swaps better than examples alone.
    parts.append("")
    parts.append("CHARACTER VISUAL CONTRACT:")
    parts.append(
        "Write one compact character_description that serves both "
        "audio and portrait generation."
    )
    parts.append("")
    parts.append(
        'Format: "<age decade>, <story-linked role>. Face: <face '
        'shape>, <eyes/brow>, <nose/mouth/jaw>, <hair/hairline>, '
        '<one distinctive story-linked detail>. Presence: <how the '
        'character carries the episode pressure>. Voice: <radio-'
        'performance cue>."'
    )
    parts.append("")
    parts.append("Rules:")
    parts.append(
        "- The face must match the character's role and emotional "
        "function in this story."
    )
    parts.append(
        "- The distinctive detail must feel earned by the premise, "
        "not random."
    )
    parts.append(
        "- Use concrete facial geometry, not vague mood words."
    )
    parts.append(
        "- Make this character visually distinct from the rest of "
        "the cast."
    )
    parts.append(
        "- Avoid glamour, fashion-model, influencer, symmetrical "
        "stock-photo language."
    )
    parts.append("")
    parts.append("JSON only:")
    parts.append('{"character_description":"<as above>"}')
    return "\n".join(parts)


def _format_prior_entry(row: dict) -> str:
    """Compact one-line summary of a prior cast row for the
    'Cast so far' block. Format: 'NAME (G, description)'.

    Sprint 9 (Narrative Face): smart-trim at the first sentence
    boundary instead of a hard char cut. The CHARACTER VISUAL
    CONTRACT produces descriptions that lead with
    "<age decade>, <story-linked role>." -- a full sentence sized
    well to anchor a prior-cast echo. A hard 60-char cut would
    chop mid-Face-block, leaving the next character's prompt
    staring at "Late-30s mission technician, the person who
    no..." with no useful signal beyond age + role. The smart
    trim preserves the lead sentence whole; falls back to the
    legacy char-trim when no period lands inside the cap (e.g.
    a single-sentence ~150-char description with the period at
    the very end).
    """
    name = (row.get("name") or "?").upper()
    g = (row.get("gender") or "?").lower()
    g_short = "M" if g == "male" else "F" if g == "female" else "X"
    desc = (row.get("character_description") or "").strip()
    # Sprint 9: smart trim. Within the same 120-char cap (lean prompt
    # discipline -- the lead sentence of the CONTRACT format target is
    # ~30-60 chars), prefer trimming at the first period so the echo
    # carries one whole sentence. If no period lands inside the cap,
    # fall back to the char-trim path so degenerate single-sentence
    # descriptions still get trimmed instead of overflowing.
    _PRIOR_CAST_CAP: int = 120
    if len(desc) > _PRIOR_CAST_CAP:
        period_idx = desc.find(".", 0, _PRIOR_CAST_CAP)
        if period_idx >= 20:
            desc = desc[: period_idx + 1].rstrip()
        else:
            desc = desc[: _PRIOR_CAST_CAP - 3].rstrip(",.;:!? ") + "..."
    return f"{name} ({g_short}, {desc})"


# ---------------------------------------------------------------------------
# Sprint 3D Stage 1 -- precompute_ensemble_slots (PURE PYTHON)
#
# Python owns the ensemble gender / timbre / role distribution. The LLM
# no longer makes any of these choices; it only writes prose into the
# facts Python has fixed.
# ---------------------------------------------------------------------------


# Coarse timbre vocabulary. Each entry is a one-word descriptor the LLM
# can write prose around AND a tunable knob python_assign_voice_preset
# uses to rank voices within a gender column. The set is intentionally
# small -- it must read naturally in a prompt and map onto the quality
# tags already present in cast_pools.VOICE_PROFILES.
_TIMBRE_VOCAB: tuple[str, ...] = (
    "warm",
    "sharp",
    "deep",
    "bright",
    "dry",
    "gravelly",
)

# Role vocabulary. A role is a dramatic function, not a job title --
# it gives the description LLM a hook without prescribing the scene.
# Python rotates through these so the ensemble is not all "leads".
_ROLE_VOCAB: tuple[str, ...] = (
    "lead",
    "foil",
    "support",
    "wildcard",
)


# Sprint 9 (Narrative Face): story-causal face-pressure anchors keyed
# off the same _ROLE_VOCAB Python already rotates across the ensemble.
# Each entry is a phrase the LLM weaves into the FACE block of the
# character_description -- a structural anchor that does not depend on
# the LLM faithfully following the CHARACTER VISUAL CONTRACT rules
# alone. The pattern matches the existing Python-decides /
# LLM-writes-prose split (gender / timbre / role rotation): Python
# pins the dramatic pressure the role implies, the LLM writes facial
# geometry that earns it. Survives LLM-prompt drift better than a
# rule embedded only in the CONTRACT text body.
_FACE_PRESSURE_BY_ROLE: dict[str, str] = {
    "lead":     "face shows responsibility, fatigue, and moral pressure",
    "foil":     "face shows skepticism, alertness, and controlled impatience",
    "support":  "face shows practical competence and lived-in worry",
    "wildcard": "face shows watchfulness, unpredictability, and survival instincts",
}


@dataclass(frozen=True)
class EnsembleSlot:
    """One open slot after Stage 1. Python has fixed gender / timbre /
    role; the LLM writes the description, Python assigns the voice.
    """

    char_id: str
    name: str
    gender: str   # one of _VALID_GENDERS
    timbre: str   # one of _TIMBRE_VOCAB
    role: str     # one of _ROLE_VOCAB


def _plan_gender_distribution(
    count: int,
    prior_genders: List[str],
    rng: random.Random,
) -> List[str]:
    """Largest-remainder allocation of the ~40/40/20 male/female/other
    split across `count` open slots, accounting for genders already
    locked in the prior cast (LEMMY etc.) so the WHOLE ensemble lands
    near target -- not just the open slots in isolation.

    Pure Python, deterministic for a given (count, prior_genders, rng).
    This is the balance the old static prompt line only *asked* the LLM
    to honour; Python now enforces it.
    """
    if count <= 0:
        return []

    total = count + len(prior_genders)
    prior_counts = {g: 0 for g, _ in _DEFAULT_GENDER_WEIGHTS}
    for g in prior_genders:
        g_norm = (g or "").strip().lower()
        if g_norm in prior_counts:
            prior_counts[g_norm] += 1

    # Ideal whole-ensemble count per gender, minus what the prior cast
    # already supplies; never negative.
    raw: list[tuple[str, float]] = []
    for gender, weight in _DEFAULT_GENDER_WEIGHTS:
        want_total = weight * total
        want_open = want_total - prior_counts[gender]
        raw.append((gender, max(0.0, want_open)))

    # Largest-remainder rounding so the parts sum to exactly `count`.
    floors = {g: int(v) for g, v in raw}
    assigned = sum(floors.values())
    remainder = count - assigned
    # Distribute the leftover to the largest fractional parts.
    frac_order = sorted(
        raw,
        key=lambda gv: (gv[1] - int(gv[1])),
        reverse=True,
    )
    out_counts = dict(floors)
    idx = 0
    while remainder > 0 and frac_order:
        gender = frac_order[idx % len(frac_order)][0]
        out_counts[gender] += 1
        remainder -= 1
        idx += 1
    # If rounding overshot (rare with non-negative clamps), trim from
    # whichever gender carries the most.
    while sum(out_counts.values()) > count:
        gender = max(out_counts, key=lambda g: out_counts[g])
        out_counts[gender] -= 1

    genders: list[str] = []
    for gender, _ in _DEFAULT_GENDER_WEIGHTS:
        genders.extend([gender] * out_counts[gender])
    # Shuffle so gender does not correlate with slot order (the cast-so-
    # far context the LLM sees would otherwise always run M, M, F, ...).
    rng.shuffle(genders)
    return genders


def _pick_same_gender_first_name(
    current_name: str,
    gender: str,
    iso: random.Random,
    taken_names: set,
) -> Optional[str]:
    """Swap the FIRST token of a 'FIRST LAST' cast name for a same-gender first
    name (keeping the last name), avoiding collisions with names already in the
    ensemble. Draws ONLY from the isolated rng `iso`, never the cast rng.
    Returns the new UPPER name, or None if the gender bucket is empty.
    """
    parts = current_name.split(" ", 1)
    last = parts[1] if len(parts) > 1 else ""
    pool = list(_POOLS.FIRST_NAMES_BY_GENDER.get(gender, ()))
    if not pool:
        return None
    iso.shuffle(pool)
    for first in pool:
        cand = (first + " " + last).strip().upper()
        if cand not in taken_names:
            return cand
    # Saturated (every same-gender first collides on this last name) -- accept
    # the first shuffled candidate anyway; a same-gender near-duplicate still
    # beats leaving a cross-gender mismatch.
    return (pool[0] + " " + last).strip().upper()


def _repair_ensemble_names(
    ensemble: List[EnsembleSlot],
    *,
    cast_seed: Optional[int],
) -> List[EnsembleSlot]:
    """C7-safe name<->gender repair (the core of the cast coherence fix).

    For each binary-gender slot whose rolled first name is tagged the OTHER
    binary gender, swap the first name for a same-gender one. The swap draws
    from a per-character ISOLATED rng -- random.Random(f"{cast_seed}:{char_id}")
    -- so the main cast rng sequence is NEVER perturbed: a no-op (byte-identical)
    for an already-coherent seed, full coherence otherwise. 'unisex'/'unknown'
    names and 'other'-gender slots are left untouched (coherent with either /
    any gender). OTR_NAME_CROSS_GENDER_RATE > 0 lets a deterministic fraction of
    mismatches stand as deliberate cross-gender names.
    """
    rate = _otr_cast_env.cross_gender_rate()
    taken_names = {e.name for e in ensemble}
    out: List[EnsembleSlot] = []
    for ens in ensemble:
        repaired = ens
        if ens.gender in ("male", "female"):
            tag = _POOLS.gender_of_first_name(ens.name)
            if tag in ("male", "female") and tag != ens.gender:
                iso = random.Random(f"{cast_seed}:{ens.char_id}")
                keep_cross = rate > 0.0 and iso.random() < rate
                if not keep_cross:
                    new_name = _pick_same_gender_first_name(
                        ens.name, ens.gender, iso, taken_names)
                    if new_name is not None and new_name != ens.name:
                        taken_names.discard(ens.name)
                        taken_names.add(new_name)
                        repaired = replace(ens, name=new_name)
        out.append(repaired)
    return out


def precompute_ensemble_slots(
    open_slots: List["CastSlot"],
    *,
    prior_cast: Optional[List[dict]] = None,
    rng: Optional[random.Random] = None,
    cast_seed: Optional[int] = None,
    repair_names: bool = True,
) -> List[EnsembleSlot]:
    """Stage 1: decide the whole ensemble's gender / timbre / role
    distribution up front. PURE PYTHON -- no LLM.

    Sprint 3D: this is where ensemble balance now lives. Previously the
    LLM was merely *asked* (a static "~40% male / ~40% female / ~20%
    other" prompt line) to honour a split it had no global view of.
    Python now decides it deterministically:

      * gender -- largest-remainder allocation of the 40/40/20 split
        across the open slots, offset by the genders the prior cast
        (LEMMY, etc.) already contributes.
      * timbre -- round-robin through `_TIMBRE_VOCAB` so the ensemble
        spans the vocal range instead of clustering.
      * role  -- round-robin through `_ROLE_VOCAB` so the ensemble is
        not all leads.

    The rng makes the gender shuffle deterministic for a fixed seed
    (C7 byte-identity). timbre/role rotation is index-based and needs
    no rng.
    """
    prior_cast = list(prior_cast or [])
    rng = rng or random.Random()
    prior_genders = [
        (row.get("gender") or "") for row in prior_cast
    ]
    genders = _plan_gender_distribution(
        len(open_slots), prior_genders, rng,
    )

    ensemble: list[EnsembleSlot] = []
    for i, slot in enumerate(open_slots):
        ensemble.append(EnsembleSlot(
            char_id=slot.char_id,
            name=slot.name,
            gender=genders[i],
            timbre=_TIMBRE_VOCAB[i % len(_TIMBRE_VOCAB)],
            role=_ROLE_VOCAB[i % len(_ROLE_VOCAB)],
        ))
    # Cast name<->gender coherence repair (isolated rng; byte-identical for an
    # already-coherent seed). See _repair_ensemble_names.
    if repair_names:
        ensemble = _repair_ensemble_names(ensemble, cast_seed=cast_seed)
    return ensemble


# ---------------------------------------------------------------------------
# Sprint 3D Stage 2 -- llm_write_description (the ONLY LLM call)
#
# The LLM's per-character job: write the prose description for one
# slot. It does NOT pick gender or voice -- Python owns both.
# ---------------------------------------------------------------------------


def llm_write_description(
    generate_fn: Callable[..., str],
    *,
    slot: EnsembleSlot,
    news_seed: str,
    style: str,
    prior_cast: List[dict],
    max_attempts: int = 3,
    base_temperature: float = 0.7,
    max_new_tokens: int = 250,
    casting_brief: str = "",
) -> DescriptionResponse:
    """Stage 2: the LLM writes ONLY the prose description for one slot.

    Sprint 3D: this is the lone LLM call in the casting pipeline. It
    used to also pick gender and voice_preset; those have moved to
    pure-Python stages (precompute_ensemble_slots and
    python_assign_voice_preset). The call still routes through the
    shared `structured_call` retry ladder (base -> structural retry ->
    typed repair); the schema is now `DescriptionResponse` (one field),
    so the voice-pool post_validator is gone -- there is no voice for
    the LLM to get wrong.

    `max_attempts=1` is allowed (single-shot, no retry). `0` is not.

    Raises CastingFailedError if the ladder is exhausted or the slot fn
    raises (structured_call does not catch slot-fn failures).
    """
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1, got {max_attempts}"
        )

    user_prompt = _build_user_prompt(
        name=slot.name,
        news_seed=news_seed,
        style=style,
        prior_cast=prior_cast,
        gender=slot.gender,
        timbre=slot.timbre,
        role=slot.role,
        casting_brief=casting_brief,
    )
    messages = [{"role": "user", "content": user_prompt}]

    # LLM slot: creative -- writing the audience-facing prose character
    # description is a creative pass; it rides the content/creative
    # plane. (Sprint 3D shrank this call to description-only; gender and
    # voice are now pure-Python, so the slot stays creative for the same
    # reason -- prose -- and there is no second call to retag.)
    # The structural retry runs at half the base temperature: strictly
    # below base, never above (the Sprint 2B principle).
    try:
        response = structured_call(
            prompt=messages,
            schema=DescriptionResponse,
            slot_fn=generate_fn,
            base_temperature=float(base_temperature),
            structural_retry_temperature=float(base_temperature) / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=int(max_new_tokens),
            max_attempts=int(max_attempts),
            helper_name=f"llm_write_description:{slot.name}",
        )
    except StructuredCallFailedError as exc:
        # Ladder exhausted. Rebuild an `attempts` list of the length the
        # ladder actually ran so lock_cast's CastValidationLLMError
        # promotion -- which keys on len(attempts) == max_attempts --
        # still fires on a full exhaustion.
        last_error_text = (
            f"{type(exc.last_error).__name__}: {exc.last_error}"
            if exc.last_error is not None
            else "no error captured"
        )
        raise CastingFailedError(
            attempts=[("", last_error_text)] * max(exc.attempts, 1),
            name=slot.name,
        ) from exc
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM loader) varies
        # structured_call does not catch slot-fn exceptions: a loader /
        # VRAM / framework failure inside generate_fn lands here.
        raise CastingFailedError(
            attempts=[("", f"slot fn raised: {type(exc).__name__}: {exc}")],
            name=slot.name,
        ) from exc

    return response


# ---------------------------------------------------------------------------
# Sprint 3D Stage 3 -- python_assign_voice_preset (PURE PYTHON)
#
# Python picks the voice from the pre-filtered pool by gender + timbre.
# This stage owns voice DISTRIBUTION. It does NOT re-implement voice
# uniqueness -- the pool pre-filter (open_voice_pool drops taken
# voices) plus the post-cast _assert_unique_bark_voices invariant
# already guarantee uniqueness. Drawing from a pre-filtered pool means
# a collision is impossible by construction here.
# ---------------------------------------------------------------------------


def python_assign_voice_preset(
    slot: EnsembleSlot,
    *,
    available_voices: List[tuple[str, str]],
    rng: Optional[random.Random] = None,
) -> str:
    """Stage 3: pick the voice preset for one slot. PURE PYTHON.

    Sprint 3D: voice selection has left the LLM. The voice is chosen
    from `available_voices` -- the pool the caller has ALREADY
    pre-filtered to exclude every voice taken by an earlier slot, so
    every candidate here is collision-free by construction.

    Selection ranks candidates by fit to the slot's Python-decided
    gender + timbre:

      1. Prefer voices whose short-description names the slot gender.
      2. Among those, prefer voices whose short-description carries the
         slot timbre word.
      3. Break ties deterministically with the rng (C7 byte-identity).

    `available_voices` is a list of (preset, short_description) tuples;
    short_description starts with the voice gender (see
    cast_pools.open_voice_pool). An empty pool raises CastingFailedError
    -- the caller's preflight should have caught it first.
    """
    rng = rng or random.Random()
    if not available_voices:
        raise CastingFailedError(
            attempts=[("", "available_voices is empty -- nothing to pick")],
            name=slot.name,
        )

    gender = (slot.gender or "").strip().lower()
    timbre = (slot.timbre or "").strip().lower()

    # 1. Gender-matched candidates. The Bark voice pool is binary
    #    male/female; an "other"-gender slot has no voice column of its
    #    own, so it draws from the full pool (the cast-row gender label
    #    still reads "other" -- the voice is a TTS detail).
    if gender in ("male", "female"):
        gender_pool = [
            (p, s) for p, s in available_voices
            if (s or "").strip().lower().startswith(gender)
        ]
    else:
        gender_pool = list(available_voices)
    # Defensive fallback: if a gender column is exhausted (more open
    # slots of one gender than the pool can supply), fall back to the
    # whole pre-filtered pool rather than failing -- a voiced character
    # of a slightly-off timbre beats a hard cast failure.
    candidates = gender_pool or list(available_voices)

    # 2. Timbre-matched subset within the gender pool.
    timbre_matched = [
        (p, s) for p, s in candidates
        if timbre and timbre in (s or "").strip().lower()
    ]
    pick_from = timbre_matched or candidates

    # 3. Deterministic tie-break. Sort by preset id first so the rng
    #    draws from a stable ordering (C7: dict / set iteration order
    #    is hash-randomized; a sorted list is byte-stable).
    pick_from = sorted(pick_from, key=lambda ps: ps[0])
    preset, _short = rng.choice(pick_from)
    return preset


# ---------------------------------------------------------------------------
# Per-character caller -- composes the three Sprint 3D stages
# ---------------------------------------------------------------------------


def cast_one_character(
    generate_fn: Callable[..., str],
    *,
    name: str,
    news_seed: str,
    style: str,
    prior_cast: List[dict],
    available_voices: List[tuple[str, str]],
    max_attempts: int = 3,
    base_temperature: float = 0.7,
    max_new_tokens: int = 250,
    casting_brief: str = "",
    ensemble_slot: Optional[EnsembleSlot] = None,
    rng: Optional[random.Random] = None,
) -> CastingResponse:
    """Cast one open character. Returns an assembled CastingResponse.

    Sprint 3D: this is now a thin composer over the three stages --

      1. precompute_ensemble_slots -- Python decides gender/timbre/role
         (skipped here when `ensemble_slot` is supplied: lock_cast
         precomputes the WHOLE ensemble once and passes each slot down).
      2. llm_write_description -- the LLM writes the prose description.
      3. python_assign_voice_preset -- Python picks the voice.

    The writer-facing contract (one validated CastingResponse per open
    slot, carrying character_description + gender + voice_preset) is
    preserved so lock_cast and its callers are unchanged in shape.

    `ensemble_slot`: when None (a standalone single-character call),
    Stage 1 runs for this one slot so the function still works on its
    own. lock_cast always passes a precomputed slot.

    `max_attempts=1` is allowed (single-shot, no retry). `0` is not.

    Raises CastingFailedError if the ladder is exhausted, the slot fn
    raises, or the voice pool is empty.
    """
    if max_attempts < 1:
        raise ValueError(
            f"max_attempts must be >= 1, got {max_attempts}"
        )
    if not available_voices:
        raise CastingFailedError(
            attempts=[("", "available_voices is empty -- nothing to pick")],
            name=name,
        )

    rng = rng or random.Random()

    # Stage 1 -- ensemble plan. lock_cast precomputes the whole ensemble
    # once and hands each slot down; a standalone call plans just this
    # one slot so the function keeps working on its own.
    if ensemble_slot is None:
        # Standalone single-character call: the caller passed an EXPLICIT name,
        # so honour it verbatim -- the name<->gender repair is an ensemble
        # (lock_cast) concern for the gender-blind POOL roll, not for a name a
        # caller chose on purpose. repair_names=False keeps this path
        # byte-identical to its pre-repair behavior.
        slot = precompute_ensemble_slots(
            [CastSlot(char_id="", name=name)],
            prior_cast=prior_cast,
            rng=rng,
            repair_names=False,
        )[0]
    else:
        slot = ensemble_slot

    # Stage 2 -- the LLM writes the description (and only that).
    description = llm_write_description(
        generate_fn,
        slot=slot,
        news_seed=news_seed,
        style=style,
        prior_cast=prior_cast,
        max_attempts=max_attempts,
        base_temperature=base_temperature,
        max_new_tokens=max_new_tokens,
        casting_brief=casting_brief,
    )

    # Stage 3 -- Python picks the voice from the pre-filtered pool.
    voice_preset = python_assign_voice_preset(
        slot,
        available_voices=available_voices,
        rng=rng,
    )

    response = CastingResponse(
        character_description=description.character_description,
        gender=slot.gender,
        voice_preset=voice_preset,
    )
    log.info(
        "[OTR_Casting] cast %s -> voice=%s gender=%s (timbre=%s role=%s)",
        name, response.voice_preset, response.gender,
        slot.timbre, slot.role,
    )
    return response


# ---------------------------------------------------------------------------
# Cast assembly -- Python only (no LLM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CastSlot:
    """A pool-fill open slot. Python rolls the name; Sprint 3D then
    runs the three-stage fill (precompute_ensemble_slots decides
    gender/timbre/role, llm_write_description writes the prose,
    python_assign_voice_preset picks the voice).
    """

    char_id: str
    name: str


def assemble_pre_locked_rows(
    *,
    num_characters: int,
    rng: Optional[random.Random] = None,
    force_lemmy: Optional[bool] = None,
    taken_names: Optional[set[str]] = None,
) -> tuple[List[dict], List[CastSlot], bool]:
    """Roll ANNOUNCER + (maybe LEMMY) + open-slot names. Pure Python,
    no LLM.

    Args:
        num_characters: 1-6, the number of NAMED characters
            (excluding ANNOUNCER). LEMMY consumes one of these slots
            when he hits.
        rng: seeded random.Random for the announcer pick + name rolls.
            If None, uses a fresh random.Random() (non-deterministic).
            The LEMMY 11% roll uses SystemRandom regardless and is
            never affected by this RNG.
        force_lemmy: testing knob. None = roll the 11% naturally.
            True = force LEMMY in. False = force LEMMY out.
        taken_names: optional set of names to exclude from the pool
            roll (useful when re-running cast assembly across a
            soak / fixture matrix).

    Returns:
        (pre_locked_rows, open_slots, lemmy_hit)
        - pre_locked_rows: list of fully-populated cast row dicts
            (ANNOUNCER plus LEMMY if hit). Each row has char_id,
            name, gender, voice_preset, character_description.
        - open_slots: list of CastSlot for the remaining open
            characters (Python rolled names, LLM fills the rest).
        - lemmy_hit: True if LEMMY was rolled in, False otherwise.
    """
    if not (1 <= num_characters <= 6):
        raise ValueError(
            f"num_characters must be 1-6, got {num_characters}"
        )

    rng = rng or random.Random()
    taken_names = set(taken_names or set())
    pre_locked: list[dict] = []

    # 1. ANNOUNCER first, char_id="c01", always.
    announcer = _POOLS.pick_announcer(rng)
    announcer["char_id"] = "c01"
    pre_locked.append(announcer)
    taken_names.add("ANNOUNCER")

    # 2. LEMMY 11% roll (or forced via the testing knob).
    # roll_lemmy() always uses OS entropy, never the seeded `rng` --
    # the cameo is decoupled from the C7 seed so it stays a genuine
    # ~11% surprise (BUG-LOCAL-260: a fixed seed otherwise pinned
    # LEMMY to 100% or 0%). force_lemmy still forces the cameo in
    # (True) or out (False) deterministically for tests and the
    # writer's operator-facing cameo control.
    if force_lemmy is None:
        lemmy_hit = _POOLS.roll_lemmy()
    else:
        lemmy_hit = bool(force_lemmy)

    next_cid_int = 2
    if lemmy_hit and num_characters >= 1:
        lemmy = _POOLS.lemmy_row()
        lemmy["char_id"] = f"c{next_cid_int:02d}"
        pre_locked.append(lemmy)
        taken_names.add("LEMMY")
        next_cid_int += 1
        remaining_open = num_characters - 1
    else:
        remaining_open = num_characters

    # 3. Roll the open-slot NAMES from the pool; LLM fills the rest.
    open_slots: list[CastSlot] = []
    for _ in range(remaining_open):
        name = _POOLS.pick_first_last(rng, taken_names)
        taken_names.add(name)
        open_slots.append(CastSlot(
            char_id=f"c{next_cid_int:02d}",
            name=name,
        ))
        next_cid_int += 1

    return pre_locked, open_slots, lemmy_hit


# ---------------------------------------------------------------------------
# Top-level: lock_cast -- runs the LLM call per open slot, returns
# the full locked cast.
# ---------------------------------------------------------------------------


def lock_cast(
    *,
    creative_fn: Callable[..., str],
    num_characters: int,
    news_seed: str,
    style: str,
    rng: Optional[random.Random] = None,
    cast_seed: Optional[int] = None,
    force_lemmy: Optional[bool] = None,
    max_attempts_per_call: int = 3,
    casting_brief: str = "",
) -> tuple[List[dict], dict]:
    """Build the full locked cast for an episode. Returns
    (cast_rows, meta).

    Cast row shape (uniform across announcer / LEMMY / pool-fill):
      {
        "char_id":               "cNN",
        "name":                  "ALICE",
        "gender":                "female",
        "voice_preset":          "v2/en_speaker_4",
        "character_description": "...",
      }

    meta dict has: lemmy_hit, casting_attempts (list of attempt
    counts per open slot, for telemetry).

    Sprint 3D: lock_cast runs the three-stage fill -- it precomputes the
    WHOLE ensemble's gender/timbre/role distribution ONCE
    (precompute_ensemble_slots) so Python balance has a global view,
    then per slot the LLM writes the description and Python assigns the
    voice. Each open slot still costs at most one LLM call; no extra
    call site is introduced.
    """
    # The per-slot fill runs every description attempt via the shared
    # structured_call ladder. lock_cast feeds it `creative_fn` -- the
    # cast row carries audience-facing prose, so casting rides the
    # creative plane. (S32 B3's technical-slot repair routing was
    # retired in the Sprint 2A/2D structured_call conversion.)
    generate_fn = creative_fn

    # Voice assignment (Stage 3) needs a seeded rng for its deterministic
    # tie-break; reuse the cast rng so a fixed seed stays byte-identical
    # (C7). A fresh Random() when the caller passed none keeps the
    # non-deterministic path working too.
    cast_rng = rng or random.Random()

    pre_locked, open_slots, lemmy_hit = assemble_pre_locked_rows(
        num_characters=num_characters,
        rng=cast_rng,
        force_lemmy=force_lemmy,
    )

    cast: list[dict] = list(pre_locked)
    # Open-character voice exclusion set tracks BARK voices only.
    # ANNOUNCER renders through Kokoro TTS (separate namespace --
    # voice IDs like "bm_george" / "bf_emma" can never collide with
    # Bark's "v2/en_speaker_X" pool), so the announcer's voice is
    # NOT added here. Per Jeffrey 2026-05-10: "announcer is in
    # Kokoro so there can be no cast overlaps." LEMMY's voice
    # (v2/en_speaker_8, Bark) IS added when LEMMY is rolled in.
    taken_voices: set[str] = {
        row["voice_preset"]
        for row in pre_locked
        if row["name"] != "ANNOUNCER"
    }

    # Preflight capacity check: if the open-slot count exceeds the
    # voices still in the pool, every later iteration will fail. Catch
    # this BEFORE any LLM call so we don't burn time + tokens on a
    # doomed cast. Per round-robin synthesis 2026-05-10 (both reviewers
    # flagged).
    initial_pool_size = len(_POOLS.open_voice_pool(taken_voices))
    if initial_pool_size < len(open_slots):
        raise CastingFailedError(
            attempts=[(
                "",
                f"voice pool too small at lock_cast entry: "
                f"{initial_pool_size} voices available for "
                f"{len(open_slots)} open slots. Pre-locked rows "
                f"already claim {sorted(taken_voices)!r}.",
            )],
            name=(open_slots[0].name if open_slots else "<no-slots>"),
        )

    # "Cast so far" context for the LLM excludes ANNOUNCER (narrator,
    # not ensemble) but includes LEMMY when rolled.
    prior_cast_for_llm: list[dict] = [
        row for row in pre_locked if row["name"] != "ANNOUNCER"
    ]

    # Sprint 3D Stage 1 -- precompute the WHOLE open ensemble's
    # gender/timbre/role distribution ONCE, up front, before any LLM
    # call. Python owns balance here, with a global view of the prior
    # cast (LEMMY's gender feeds the 40/40/20 allocation).
    ensemble_slots = precompute_ensemble_slots(
        open_slots,
        prior_cast=prior_cast_for_llm,
        rng=cast_rng,
        cast_seed=cast_seed,
    )

    casting_attempts: list[int] = []
    for slot, ens in zip(open_slots, ensemble_slots):
        available_voices = _POOLS.open_voice_pool(taken_voices)
        if not available_voices:
            # Belt-and-braces: should never fire because of the
            # preflight check above, but kept as a defensive assert.
            raise CastingFailedError(
                attempts=[("", "voice pool exhausted mid-loop "
                              "(preflight should have caught this)")],
                name=slot.name,
            )

        try:
            response = cast_one_character(
                generate_fn,
                name=ens.name,
                news_seed=news_seed,
                style=style,
                prior_cast=prior_cast_for_llm,
                available_voices=available_voices,
                max_attempts=max_attempts_per_call,
                casting_brief=casting_brief,
                ensemble_slot=ens,
                rng=cast_rng,
            )
        except CastingFailedError as exc:
            # S32 B3 (D2): if the failure came from the repair-attempt
            # (technical-slot validation pass), surface it as a
            # CastValidationLLMError so the writer-side caller can
            # branch on the more-specific subclass and trigger creative
            # regen rather than a hard fail. The signal is structural:
            # the last attempt in `exc.attempts` corresponds to the
            # repair call when max_attempts_per_call >= 2. Subclass
            # remains catchable as CastingFailedError for legacy
            # handlers.
            attempts_count = len(getattr(exc, "attempts", []) or [])
            if (
                max_attempts_per_call >= 2
                and attempts_count == max_attempts_per_call
            ):
                raise CastValidationLLMError(
                    attempts=exc.attempts,
                    name=exc.name,
                ) from exc
            raise
        new_row = {
            "char_id":               slot.char_id,
            "name":                  ens.name,
            "gender":                response.gender,
            # Open-character voices are always drawn from the Bark
            # pool (VOICE_PROFILES in config/cast_pools.py), so the
            # tts_model is Bark by construction. Downstream consumers
            # route on this field rather than pattern-matching the
            # voice_preset prefix.
            "tts_model":             "bark",
            "voice_preset":          response.voice_preset,
            # voice_params: None today (consumers fall back to their
            # defaults). Phase 2 expands the casting LLM call to ask
            # for per-character knobs (Bark temperature, Kokoro speed)
            # bounded by VOICE_REGISTRY[tts_model]["params_spec"].
            "voice_params":          None,
            "character_description": response.character_description,
        }
        cast.append(new_row)
        taken_voices.add(response.voice_preset)
        prior_cast_for_llm.append(new_row)
        # Telemetry: how many attempts did this slot need? We can't
        # see it from the response object; the caller can wrap
        # cast_one_character if granular telemetry is needed. For now
        # just stamp 1 -- a successful call returned without raising.
        casting_attempts.append(1)

    # Post-cast voice-uniqueness invariant. Belt-and-braces guard
    # against a future refactor breaking the pre-filter / validator /
    # reroll chain that today already guarantees uniqueness by
    # construction. Cheap, deterministic, fast-fails at the right
    # spot. Per Jeffrey 2026-05-10: "ensure no two characters have
    # the same voice model including LEMMY a hard decision."
    #
    # ANNOUNCER is intentionally excluded -- it's Kokoro-namespaced
    # (bm_/bf_) and cannot collide with the Bark pool by construction.
    # The check covers LEMMY (Bark) + all open-character Bark voices.
    _assert_unique_bark_voices(cast)
    # Gate 1 (voice-path-cleanbreak): every non-ANNOUNCER row carries a
    # non-empty v2/* voice_preset. Earliest of three gates; Gate 2 lives
    # in FreezeCascade Phase 0 G6, Gate 3 in BatchBarkGenerator.
    _assert_voice_preset_invariant(cast)
    # S13.1: structural-token guard. Reject cast rows whose name is a
    # SFX cue / screenplay meta-direction / parser artefact / one of
    # TITLE / NOTE / TARGET / STYLE.
    _assert_no_structural_tokens_in_cast(cast)

    meta = {
        "lemmy_hit":              lemmy_hit,
        "casting_attempts":       casting_attempts,
        "num_characters_request": num_characters,
        "num_characters_locked":  len(cast) - 1,  # minus ANNOUNCER
    }
    return cast, meta


def _assert_voice_preset_invariant(cast: List[dict]) -> None:
    """Gate 1 (writer cast-lock exit) -- the earliest of three gates
    enforcing the cast.voice_preset contract for the voice-path-cleanbreak.

    Every non-ANNOUNCER cast row must carry a non-empty ``voice_preset``
    starting with ``v2/`` (the Bark preset namespace). ANNOUNCER is
    intentionally excluded because it lives in the Kokoro namespace
    (``bm_*`` / ``bf_*``) by construction.

    Empty / None / non-v2 preset on a Bark row indicates a writer
    contract violation. Today the pre-filter + cast LLM + reroll chain
    already guarantees well-formed v2 presets on every open slot, and
    pre-locked rows (LEMMY) carry hardcoded v2 ids. This assertion
    catches a future refactor that breaks any of those guarantees and
    surfaces the failure at the writer rather than letting an empty
    preset propagate to the voice nodes (Gate 3) or to the freeze
    cascade G6 interlock (Gate 2).
    """
    missing: list[str] = []
    bad: list[str] = []
    for row in cast or []:
        if not isinstance(row, dict):
            continue
        if row.get("name") == "ANNOUNCER":
            continue
        char_id = row.get("char_id") or "<no char_id>"
        preset = row.get("voice_preset")
        if not preset:
            missing.append(char_id)
        elif not str(preset).startswith("v2/"):
            bad.append(f"{char_id}={preset}")
    if not missing and not bad:
        return
    msg_parts: list[str] = []
    if missing:
        msg_parts.append(
            f"empty voice_preset on {len(missing)} row(s): {', '.join(missing)}"
        )
    if bad:
        msg_parts.append(
            f"non-v2/* voice_preset on {len(bad)} row(s): {', '.join(bad)}"
        )
    raise CastingFailedError(
        attempts=[(
            "",
            f"GATE 1 (writer cast-lock exit) FAILED: {'; '.join(msg_parts)}. "
            "Bark requires v2/* presets on every non-ANNOUNCER cast row.",
        )],
        name="<lock_cast voice_preset invariant>",
    )


# ---------------------------------------------------------------------------
# Structural-token guard (S13.1, ports + extends the deleted
# story_orchestrator._looks_like_non_character_cast_name heuristic)
# ---------------------------------------------------------------------------


# Patterns that indicate the cast name is a parser artefact / SFX cue /
# screenplay meta-direction tag, NOT a real character. Ported verbatim
# from the pre-S7.1 story_orchestrator._SFX_CAST_BLOCKLIST_PATTERNS
# (deleted in commit b6fb314) plus FIVE additional standalone tokens
# (TITLE / NOTE / TARGET / STYLE / NARRATOR-as-name) that appeared in
# the pre-S7.1 story_orchestrator._BRACKET_STRUCTURAL_TOKENS but were
# not in the cast-blocklist patterns. The S13.1 cast-contract
# verification confirmed all five slipped through pre-port. After
# port: each one raises CastingFailedError with a structural-token
# diagnostic.
_NON_CHARACTER_CAST_PATTERNS = (
    # SFX cue artefacts (BUG-LOCAL-090 root cause)
    r"^SFX\b", r"^MUSIC\b", r"^THEME\b",
    r"\bBLARING\b", r"\bBLARE\b", r"\bWHOOSH\b", r"\bWHOOSHING\b",
    r"\bFLICKERS?\b", r"\bFLICKER\b",
    r"\bCHAMBER\b", r"\bPORTAL\b", r"\bALARM\b",
    r"\bEQUIPMENT\b", r"\bCUE\b",
    r"\bAT THE\b",
    r"\bSOUND\b", r"\bMUSIC QUEUE\b",
    r"\bINTENSE\b", r"\bMYSTERIOUS VOICE\b",
    # Screenplay meta-direction (BUG-LOCAL-097)
    r"\bVOICEOVER\b", r"\bVOICE\s?OVER\b", r"\bVOICEOBER\b",
    r"\bNARRATOR\b",
    # NOTE: Original pre-S7.1 patterns had trailing ``\b`` after the
    # final ``\.`` -- a no-op because ``.`` is non-word and the post-
    # period regex \b never fires. Faithful port + bugfix here drops
    # the trailing \b so ``JOHN V.O.`` actually matches.
    r"\bV\.O\.", r"\bO\.S\.",
    r"\bSCREEN\b", r"\bOFF.SCREEN\b",
    # S13.1 additions: structural tokens that the LLM occasionally
    # emits as standalone "character" names. The risk asymmetry
    # (real character named "Style" gets rejected) is far lower than
    # the false-negative cost (an LLM hallucination renders as a
    # voice line in production).
    r"^TITLE$", r"^NOTE$", r"^TARGET$", r"^STYLE$",
)


def _looks_like_non_character_cast_name(name: str) -> bool:
    """Return True when ``name`` is almost certainly an SFX cue,
    music stinger, scene-direction fragment, structural token, or
    other parser artefact -- not a real character.

    Ported from story_orchestrator (deleted in S7.1 / commit b6fb314)
    and extended with TITLE / NOTE / TARGET / STYLE per S13.1
    cast-contract verification.
    """
    if not name:
        return True
    n = name.upper().strip()
    for pat in _NON_CHARACTER_CAST_PATTERNS:
        if re.search(pat, n):
            return True
    return False


def _assert_no_structural_tokens_in_cast(cast: List[dict]) -> None:
    """Cast contract S13.1: reject any cast row whose ``name`` is
    a structural token (SFX cue, screenplay meta-direction, parser
    artefact, or one of TITLE / NOTE / TARGET / STYLE). ANNOUNCER
    is allowed because it's the canonical narrator slot, not an
    artefact.

    The risk asymmetry (false-positive: a real character named
    "Style" gets rejected; false-negative: an LLM hallucination
    renders as a voice line in production) heavily favors
    rejection. If a future story legitimately needs a character
    named one of these tokens, the right move is to add a
    case-sensitive whitelist check, not to widen the patterns.
    """
    bad: list[str] = []
    for row in cast or []:
        if not isinstance(row, dict):
            continue
        name = row.get("name") or ""
        if name == "ANNOUNCER":
            continue
        if _looks_like_non_character_cast_name(name):
            bad.append(name)
    if not bad:
        return
    raise CastingFailedError(
        attempts=[(
            "",
            f"S13.1 STRUCTURAL TOKEN GUARD: {len(bad)} cast row(s) "
            f"have names that look like SFX cues / screenplay meta-"
            f"direction / structural tokens, not real characters: "
            f"{', '.join(repr(n) for n in bad)}. The pre-filter + cast "
            f"LLM should have rejected these; a refactor likely broke "
            f"the upstream guarantee.",
        )],
        name="<lock_cast structural-token invariant>",
    )


def _assert_unique_bark_voices(cast: List[dict]) -> None:
    """Raise CastingFailedError if any two Bark cast rows share a
    voice_preset. ANNOUNCER (Kokoro namespace) is excluded.

    Called at the end of lock_cast() as a final invariant check.
    Today this is guaranteed-true by the pre-filter + validator +
    reroll path; this assertion catches any future regression.
    """
    bark_voices: list[tuple[str, str]] = []  # (char_id, voice_preset)
    for row in cast:
        if row["name"] == "ANNOUNCER":
            continue
        bark_voices.append((row["char_id"], row["voice_preset"]))
    voices_only = [v for _, v in bark_voices]
    if len(set(voices_only)) != len(voices_only):
        # Build a precise duplicate report for the error message
        seen: dict[str, str] = {}
        duplicates: list[str] = []
        for cid, v in bark_voices:
            if v in seen:
                duplicates.append(
                    f"{cid} and {seen[v]} both have {v!r}"
                )
            seen[v] = cid
        raise CastingFailedError(
            attempts=[(
                "",
                "POST-CAST INVARIANT FAILED: duplicate Bark "
                f"voice_preset across cast rows: {duplicates!r}. "
                "Pre-filter + validator + reroll should have "
                "prevented this; a refactor likely broke the "
                "collision guarantee.",
            )],
            name="<lock_cast invariant>",
        )


__all__ = [
    "CastingResponse",
    "DescriptionResponse",
    "CastingFailedError",
    "CastValidationLLMError",
    "_assert_unique_bark_voices",
    "_assert_voice_preset_invariant",
    "_assert_no_structural_tokens_in_cast",
    "_looks_like_non_character_cast_name",
    "CastSlot",
    "EnsembleSlot",
    "assemble_pre_locked_rows",
    "precompute_ensemble_slots",
    "llm_write_description",
    "python_assign_voice_preset",
    "cast_one_character",
    "lock_cast",
]
