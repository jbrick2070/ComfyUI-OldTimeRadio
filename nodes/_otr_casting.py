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

import hashlib
import logging
import random
from dataclasses import dataclass, replace
from typing import Callable, List, Mapping, Optional

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

# ONE naming authority, enforced at the boundary (Bug Bible 11.61). Shared with
# the offline archive sweep so the sweep can never certify a different rule than
# the one runtime enforces.
try:
    from . import _otr_name_authority as _NAMES
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_name_authority as _NAMES  # type: ignore

# CastPlanner (S4) + Pass-1 validator (S7). Only consulted on the llm_slot_fill
# path; pool mode never imports-uses them at runtime.
try:
    from . import _otr_castplanner as _CASTPLAN
    from . import _otr_cast_validator as _CASTVAL
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_castplanner as _CASTPLAN  # type: ignore
    import _otr_cast_validator as _CASTVAL  # type: ignore

# Bake-off variant -> base bank id. Source-fidelity casting is FAMILY
# behaviour, so `shakespeare_v2` must cast exactly like `shakespeare`.
# Dependency-free leaf module; safe to import from the cast contract.
try:
    from ._otr_bank_variants import base_source_bank_id
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_bank_variants import base_source_bank_id  # type: ignore

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_VALID_GENDERS = {"male", "female", "other"}
# What a SOURCE roster may pin. 'other' is deliberately absent: the roster only
# ever records male/female/unknown, and an unknown is left to the roll rather
# than pinned, so CastingResponse._gender_in_set is never handed a novel value.
_PINNABLE_GENDERS = frozenset({"male", "female"})

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

    # BUG-LOCAL-263 (2026-05-24): max_length 200 -> 750. 2026-05-31: 750 ->
    # 1500 -- a verbose frontier remote model (claude-opus-4.8) writes
    # richer descriptions that overran 750 and burned repair-ladder calls
    # recovering. The cap is a runaway guard, not a content target;
    # _format_prior_entry already trims the echoed description to 60 chars,
    # so the stored length never touches prompt budget. Local Mistral's
    # short descriptions are unaffected.
    character_description: str = Field(..., min_length=1)
    # F5 (story-engine v1): a <=5-word speech register/signature ("clipped,
    # formal", "warm, rambling") the line composer threads into the cast card
    # so each character reads as a distinct voice. Optional (default "") so a
    # model that omits it never fails the schema; backfilled downstream.
    speech_signature: str = ""

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

    character_description: str = Field(..., min_length=1)
    gender: str = Field(..., min_length=1)
    # F5 (story-engine v1): speech register/signature, assembled from the
    # description call. Optional (default "") -> backfilled to "plain spoken".
    speech_signature: str = ""
    # Sprint 2 (a): voice_preset is no longer assigned by the writer -- OTR_CastLock
    # replays the picker and stamps it after the freeze. cast_one_character leaves
    # it EMPTY, so the field allows "" (was min_length=3).
    # VC chunk 2 (2026-06-22): cap 80 -> 255. The two-lane identity contract lets
    # this field carry a verbose voice_ref_id (cloner id) in addition to a short
    # bark v2/* preset; a deeply-named clone reference can exceed 80 chars. The cap
    # is a runaway guard, not a content target.
    voice_preset: str = ""

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
      - LEMMY (M, genial communications officer, 50s, warm Cockney)
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
    parts.append(
        "- Also give a brief speech_signature naming how this character "
        "talks. This is generation guidance, not an acceptance score."
    )
    parts.append("")
    parts.append("JSON only:")
    parts.append(
        '{"character_description":"<as above>",'
        '"speech_signature":"<spoken delivery note>"}'
    )
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
    role / age_band; the LLM writes the description, Python assigns the voice.
    """

    char_id: str
    name: str
    gender: str   # one of _VALID_GENDERS
    timbre: str   # one of _TIMBRE_VOCAB
    role: str     # one of _ROLE_VOCAB
    # VC chunk 3 (2026-06-22): age_band carried on the slot so the cast_voice_slots
    # stamp (and CastLock's bank caster) can match on age, not just gender. Pool
    # mode has no finer age signal than "adult" (radio leads are adults; the bank
    # is adult-dominant), so the default keeps the ensemble honest + deterministic.
    # This does NOT touch python_assign_voice_preset's pool-mode call (that still
    # receives age_band=None), so the bark replay stays byte-identical.
    age_band: str = "adult"   # child / young_adult / adult / elder
    # True when this slot's NAME came off the source's own cast list rather than
    # the invented-name pool. Such a slot is a real person in the source, so its
    # gender is pinned from the roster and its name is exempt from the
    # coherence rename. APPENDED, never inserted: tests/test_cast_llm_naming.py
    # builds EnsembleSlot with five positional arguments.
    source_owned: bool = False


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

    SOURCE-OWNED slots are exempt. TOBY is Sir Toby Belch whatever the pool
    thinks that first name's gender tag is, and renaming him to ERIN or MARGOT to
    satisfy a coherence rule loses the character the adaptation exists to perform
    -- measured, that rename fired on 133 of 400 seeds.
    """
    rate = _otr_cast_env.cross_gender_rate()
    taken_names = {e.name for e in ensemble}
    out: List[EnsembleSlot] = []
    for ens in ensemble:
        repaired = ens
        if getattr(ens, "source_owned", False):
            out.append(repaired)
            continue
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
    gender_by_name: Optional[Mapping[str, str]] = None,
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

    ``gender_by_name`` pins the genders the SOURCE records, for source-owned
    slots only. The allocation itself is deliberately NOT re-run: the call to
    ``_plan_gender_distribution`` below takes the same count, the same priors and
    the same rng it always did, and the pin is applied afterwards by overwriting
    the drawn value at pinned indices.

    That is the whole design, and the alternatives are both wrong. Feeding the
    pins in as ``prior_genders`` makes the largest-remainder allocator push the
    remaining slot the other way -- measured, `_plan_gender_distribution(1,
    ['male'])` returns female on 400 of 400 seeds -- turning a coin flip into a
    guaranteed error. Re-calling it with a reduced count changes how far the
    shuffle advances the stream (measured getrandbits: 0, 0, 3, 3, 9, 11 for
    counts 0..5), which desynchronizes any replay that reconstructs the ensemble.
    Overriding in place keeps the rng call count and the post-call stream
    identical, keeps the unpinned slots' distribution exactly as it is today, and
    is byte-identical on every lane when ``gender_by_name`` is None.
    """
    prior_cast = list(prior_cast or [])
    rng = rng or random.Random()
    prior_genders = [
        (row.get("gender") or "") for row in prior_cast
    ]
    genders = _plan_gender_distribution(
        len(open_slots), prior_genders, rng,
    )

    pins = {
        str(k).strip().upper(): str(v).strip().lower()
        for k, v in (gender_by_name or {}).items()
        if str(v).strip().lower() in _PINNABLE_GENDERS
    }
    ensemble: list[EnsembleSlot] = []
    for i, slot in enumerate(open_slots):
        source_owned = bool(getattr(slot, "source_owned", False))
        gender = genders[i]
        if source_owned and pins:
            # assemble_pre_locked_rows upper-cases slot names; the writer's
            # _adapt_names are title-case. Both sides are normalized to UPPER.
            gender = pins.get(str(slot.name or "").strip().upper(), gender)
        ensemble.append(EnsembleSlot(
            char_id=slot.char_id,
            name=slot.name,
            gender=gender,
            timbre=_TIMBRE_VOCAB[i % len(_TIMBRE_VOCAB)],
            role=_ROLE_VOCAB[i % len(_ROLE_VOCAB)],
            source_owned=source_owned,
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
    age_band: Optional[str] = None,
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

    # 2b. Age-matched subset (S5, voice x age). ONLY when an age_band is
    #     supplied -- the CastPlanner / llm_slot_fill path. Pool mode passes
    #     age_band=None, so this is a no-op and pool-mode voice picks stay
    #     byte-identical (C7). Still exactly ONE rng.choice below (R6): age is
    #     a filter, never a draw.
    if age_band:
        try:
            from ._otr_castplanner import AGE_BAND_VOICE_TAGS
        except ImportError:  # pragma: no cover - standalone / test load
            from _otr_castplanner import AGE_BAND_VOICE_TAGS  # type: ignore
        age_tags = AGE_BAND_VOICE_TAGS.get(age_band, frozenset())
        if age_tags:
            age_matched = [
                (p, s) for p, s in pick_from
                if any(t in (s or "").strip().lower() for t in age_tags)
            ]
            pick_from = age_matched or pick_from

    # 3. Deterministic tie-break. Sort by preset id first so the rng
    #    draws from a stable ordering (C7: dict / set iteration order
    #    is hash-randomized; a sorted list is byte-stable).
    pick_from = sorted(pick_from, key=lambda ps: ps[0])
    preset, _short = rng.choice(pick_from)
    return preset


# ---------------------------------------------------------------------------
# Sprint 2 (a) -- pure voice-assignment REPLAY for OTR_CastLock
# ---------------------------------------------------------------------------
def replay_voice_assignment(
    *,
    cast_seed: int,
    num_characters: int,
    lemmy_hit: bool,
) -> dict:
    """Reproduce lock_cast's deterministic bark voice_preset assignment WITHOUT
    the LLM. Returns ``{char_id: voice_preset}`` for every non-ANNOUNCER row
    (LEMMY, when present, plus the open slots).

    Sprint 2 (a): this is the pure replay OTR_CastLock runs to OWN bark casting,
    byte-identical to what the writer's ``lock_cast`` assigned for the same
    (cast_seed, num_characters, lemmy_hit). It replays the EXACT seeded-rng
    sequence the writer used -- ``random.Random(cast_seed)`` drives the announcer
    pick + the open-slot name rolls (assemble_pre_locked_rows) + the gender
    shuffle (precompute_ensemble_slots) + the per-slot voice pick
    (python_assign_voice_preset). The LLM description step in lock_cast draws NO
    cast rng, so skipping it does not perturb the sequence. The LEMMY cameo roll
    uses OS entropy (never the seeded rng), so passing ``force_lemmy=lemmy_hit``
    (the writer's persisted outcome) reproduces the cast structure + draw count
    exactly. Keys on the writer's ``cast_seed`` -- NOT
    ``_otr_voice_bank.stable_cast_seed`` (a different per-character clip seed).
    The parity test pins ``replay == lock_cast`` char-for-char.
    """
    rng = random.Random(cast_seed)
    pre_locked, open_slots, _hit = assemble_pre_locked_rows(
        num_characters=num_characters, rng=rng, force_lemmy=bool(lemmy_hit),
    )
    prior_cast = [r for r in pre_locked if r["name"] != "ANNOUNCER"]
    taken_voices = {r["voice_preset"] for r in prior_cast}
    ensemble_slots = precompute_ensemble_slots(
        open_slots, prior_cast=prior_cast, rng=rng, cast_seed=cast_seed,
    )
    name_mode = _otr_cast_env.name_mode()
    out: dict = {r["char_id"]: r["voice_preset"] for r in prior_cast}
    for i, (slot, ens) in enumerate(zip(open_slots, ensemble_slots)):
        age_band = (_CASTPLAN.age_band_for_index(i)
                    if name_mode == "llm_slot_fill" else None)
        available_voices = _POOLS.open_voice_pool(taken_voices)
        voice = python_assign_voice_preset(
            ens, available_voices=available_voices, rng=rng, age_band=age_band,
        )
        taken_voices.add(voice)
        out[slot.char_id] = voice
    return out


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
    age_band: Optional[str] = None,
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

    # Sprint 2 (a): bark voice_preset is assigned by OTR_CastLock AFTER the
    # freeze (replay_voice_assignment -- byte-identical to this picker), NOT
    # here. The writer no longer stamps it; it stays empty through the writer +
    # the freeze and is filled at cast-lock.
    voice_preset = ""

    response = CastingResponse(
        character_description=description.character_description,
        gender=slot.gender,
        voice_preset=voice_preset,
        speech_signature=str(
            getattr(description, "speech_signature", "") or ""
        ).strip(),
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
    # See EnsembleSlot.source_owned -- set when the name was popped off the
    # source's cast list instead of rolled from the pool. Trailing default keeps
    # every existing keyword construction valid.
    source_owned: bool = False


# Source-faithful adaptations must use their own cast, never the recurring
# Lemmy cameo. This overrides both the entropy roll and the operator cameo
# setting; invention and archive banks retain the existing cameo behavior.
#: The most named characters THIS assembler can seat, set by the Bark voice
#: stock it draws from -- two characters never share a voice, so the pool is
#: the real ceiling. NAMED, not a bare 6 in two places: the bound and the
#: clamp in `lock_cast` have to move together or an over-request starts
#: raising again. Deliberately NOT `_otr_scifi_news_pro.MAX_SPEAKING_CAST`
#: (10): that lane seats its own cast from a different stock, and the
#: `num_characters` widget spans both, which is exactly why an over-request
#: has to degrade here rather than refuse.
_LEGACY_MAX_SPEAKING_CAST = 6

_LEMMY_EXCLUDED_SOURCE_BANK_IDS = frozenset({"public_domain", "shakespeare"})


def _source_bank_excludes_lemmy(source_bank_id: str | None) -> bool:
    """True when the bank's family is a source-faithful adaptation.

    Normalized through ``base_source_bank_id`` so bake-off variants
    (``shakespeare_v2``, ``public_domain_v3``) inherit the exclusion --
    fidelity is a family behaviour, not a per-row opt-in.
    """
    normalized = base_source_bank_id(str(source_bank_id or "").strip().lower())
    return normalized in _LEMMY_EXCLUDED_SOURCE_BANK_IDS


#: ``lemmy_policy`` for a lane that builds its OWN cast and never runs the
#: cameo roll at all. Deliberately NOT ``operator_cameo``: that value asserts
#: the 11% roll ran and came up short, so stamping it beside ``lemmy_hit:
#: False`` would record a decline nobody ever made -- the same silence in a
#: better disguise. A reader can now tell "asked and declined" from "never
#: asked", which is the whole point of the stamp (PBUG-20260811-03).
CONTENT_OWNED_NO_CAMEO_ROLL = "content_owned_cast_no_cameo_roll"

#: The two keys a content-owned contract MUST NOT carry, and why. Named here
#: rather than left implicit because both are load-bearing absences:
#:
#: - ``cast_seed`` is not a generic episode seed. It is a CLAIM that the
#:   writer's seeded picker produced this cast and can be replayed from it.
#:   OTR_CastLock replays the picker whenever it sees the key, and a lane-owned
#:   cast has no ``num_characters_request`` to replay with, so claiming it
#:   detonates the replay ("num_characters must be 1-6, got 0"). The durable
#:   seed receipt a content-owned lane owes downstream is ``meta.episode_seed``,
#:   stamped by the shared writer tail.
#: - ``cast_seed_source`` describes a seed that is not here.
#:
#: otr_credits_roll reads ``cast_contract.get("cast_seed", meta.episode_seed)``,
#: so the key must be ABSENT rather than present-and-None -- a None default
#: would satisfy ``.get`` and hand credits a null seed.
CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS = frozenset({
    "cast_seed", "cast_seed_source",
})


#: The two ``lemmy_policy`` values that describe a cameo actually DECIDED, and
#: the three knob states that describe what the operator ASKED FOR. They are
#: constants rather than literals because a policy string is a contract between
#: a producer and a reader who never meet, and one of them retyping it is a
#: whole Bible-covered defect class. Every spelling in this module resolves
#: here.
LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION = "source_fidelity_exclusion"
LEMMY_POLICY_OPERATOR_CAMEO = "operator_cameo"

LEMMY_KNOB_NATURAL_ROLL = "natural_roll"
LEMMY_KNOB_FORCED_INCLUDE = "forced_include"
LEMMY_KNOB_FORCED_EXCLUDE = "forced_exclude"

#: Versioned because this dict lands in durable ledger meta and outlives the
#: code that wrote it.
LEMMY_CAMEO_DECISION_SCHEMA = "lemmy-cameo-decision.v1"


@dataclass(frozen=True)
class LemmyCameoDecision:
    """One episode's cameo answer, decided ONCE at runner entry.

    Immutable on purpose: every consumer -- the cast contract, the roll
    receipt, the prompt contract, the voice deal -- must be reading the same
    answer. A mutable decision is how a lane ends up stamping ``lemmy_hit:
    False`` beside a Lemmy who is standing in the script.

    ``knob_state`` records what the OPERATOR asked for and ``lemmy_policy``
    records what the BANK allowed. Keeping them separate is what lets a reader
    tell "he was excluded from an adaptation" from "the roll came up short",
    even when both land on ``lemmy_hit: False``.
    """

    lemmy_hit: bool
    lemmy_policy: str
    knob_state: str
    source_bank_id: str
    roll_executed: bool

    def to_meta(self) -> dict:
        """A PRIMITIVE-ONLY dict, because ``Ledger.save()`` never raises.

        A non-serializable object in meta logs a warning, returns None, and a
        dozen call sites never check -- so the episode loses its receipt in
        silence. Everything below is a bool, an int or a str by construction.
        """
        return {
            "schema_version": LEMMY_CAMEO_DECISION_SCHEMA,
            "lemmy_hit": bool(self.lemmy_hit),
            "lemmy_policy": str(self.lemmy_policy),
            "knob_state": str(self.knob_state),
            "source_bank_id": str(self.source_bank_id),
            "roll_executed": bool(self.roll_executed),
        }


def resolve_lemmy_cameo(source_bank_id, force_lemmy) -> LemmyCameoDecision:
    """Decide the cameo once, before any authoring.

    Called EXACTLY ONCE per episode at runner entry. It must be early because
    the content-owned lanes derive their cast FROM the finished script and then
    gate on it -- so a cameo injected after the script cannot pass those gates,
    and one decided twice could disagree with itself between the prompt and the
    cast.

    ``roll_lemmy()`` is called in exactly one branch and nowhere else, so the
    ~11% OS-entropy roll happens once or not at all. Exclusion OUTRANKS the
    operator knob: a fidelity lane refuses the cameo even when the operator
    forced it, because the source's cast is the point of that lane.
    """
    if force_lemmy is None:
        knob_state = LEMMY_KNOB_NATURAL_ROLL
    elif force_lemmy:
        knob_state = LEMMY_KNOB_FORCED_INCLUDE
    else:
        knob_state = LEMMY_KNOB_FORCED_EXCLUDE

    # The RAW id is recorded; the exclusion test normalizes internally. The
    # receipt should say what it was handed, not what the check made of it.
    bank_id = str(source_bank_id or "").strip()

    if _source_bank_excludes_lemmy(bank_id):
        return LemmyCameoDecision(
            lemmy_hit=False,
            lemmy_policy=LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION,
            knob_state=knob_state,
            source_bank_id=bank_id,
            roll_executed=False,
        )

    if force_lemmy is None:
        return LemmyCameoDecision(
            lemmy_hit=bool(_POOLS.roll_lemmy()),
            lemmy_policy=LEMMY_POLICY_OPERATOR_CAMEO,
            knob_state=knob_state,
            source_bank_id=bank_id,
            roll_executed=True,
        )

    return LemmyCameoDecision(
        lemmy_hit=bool(force_lemmy),
        lemmy_policy=LEMMY_POLICY_OPERATOR_CAMEO,
        knob_state=knob_state,
        source_bank_id=bank_id,
        roll_executed=False,
    )


def content_owned_cast_contract(
    *,
    source_bank_id: str | None,
    num_characters_request: int,
    num_characters_locked: int,
    decision: "LemmyCameoDecision | None" = None,
) -> dict:
    """Build ``meta.cast_contract`` for a lane that owns its own cast.

    The content-owned lane (``_otr_scifi_news_pro``)
    derive their cast from the script the model already wrote, so they never
    reach :func:`lock_cast` and never stamped this key at all -- both shipped
    ``cast_contract: {}`` on every episode (PBUG-20260811-03, re-measured on
    two 2026-08-15 legs). That is the SILENT half of the defect: a downstream
    reader could not distinguish a cameo that was declined from a cameo that
    was never considered, and nothing failed and nothing logged.

    This is the same rule the invention lanes already follow twelve hundred
    lines below -- *one stable shape on every lane, stamp the empty contract
    rather than omitting the key* -- applied to the lanes that were missing it.

    Routing these lanes back through :func:`lock_cast` is explicitly the WRONG
    fix: it would claim a ``cast_seed`` for a cast its picker never rolled.
    The contract is built here and stamped by each runner instead.

    Returns exactly the legacy contract's key set minus
    :data:`CONTENT_OWNED_CONTRACT_FORBIDDEN_KEYS`; the shapes are held equal by
    ``tests/test_content_owned_cast_contract.py``.
    """
    if decision is None:
        # NO DECISION SUPPLIED -- the pre-chunk-B answer, byte-for-byte. A lane
        # that has not yet migrated keeps stamping exactly what it stamped
        # before, so the API can land green ahead of its callers.
        #
        # No cameo is cast on such a lane: it builds its cast FROM the finished
        # script and never runs the roll. This records the true state rather
        # than a hopeful one.
        #
        # Fidelity still outranks the lane's own silence: if a content-owned
        # pipeline is pointed at an adaptation bank, the reason the cameo is
        # absent is the exclusion, not the missing roll.
        lemmy_hit = False
        lemmy_policy = (
            LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION
            if _source_bank_excludes_lemmy(source_bank_id)
            else CONTENT_OWNED_NO_CAMEO_ROLL
        )
    else:
        # A MIGRATED LANE. The contract and the roll receipt now read from one
        # decision, so they cannot disagree about whether Lemmy is in the show.
        lemmy_hit = bool(decision.lemmy_hit)
        lemmy_policy = str(decision.lemmy_policy)

    return {
        "lemmy_hit":              lemmy_hit,
        "lemmy_policy":           lemmy_policy,
        # The writer's cast LLM made zero attempts here, which is the honest
        # count -- this lane's casting attempts live in its own pass receipts.
        "casting_attempts":       [],
        # Kept as a PAIR on purpose. These lanes take num_characters as a
        # request and routinely land somewhere else (a leg asked for 2 and
        # produced 3), and the divergence is only visible if both numbers are
        # recorded. Neither is a gate -- THE LAW.
        "num_characters_request": int(num_characters_request),
        "num_characters_locked":  int(num_characters_locked),
    }


def count_locked_characters(cast_rows) -> int:
    """Non-ANNOUNCER cast rows, for ``num_characters_locked``.

    Counted by identity rather than ``len(cast) - 1`` so a lane that ships no
    announcer row, or more than one, reports what it actually has instead of
    an off-by-one dressed as a fact.

    A TWO-SHAPE ALLOWLIST, not a general rule (QA note 2026-08-16): codex
    ships ``char_id == "announcer"``, scifi_news_pro ships ``name == "ANNOUNCER"`` at
    ``c01``. A third content-owned lane with a differently keyed announcer row
    would be miscounted -- add its shape here when it exists.
    """
    total = 0
    for row in cast_rows or []:
        if not isinstance(row, dict):
            continue
        if (str(row.get("char_id") or "").strip().lower() == "announcer"
                or row.get("name") == "ANNOUNCER"):
            continue
        total += 1
    return total


def assemble_pre_locked_rows(
    *,
    num_characters: int,
    rng: Optional[random.Random] = None,
    force_lemmy: Optional[bool] = None,
    taken_names: Optional[set[str]] = None,
    source_character_names: Optional[List[str]] = None,
    source_bank_id: str | None = None,
    decision: "LemmyCameoDecision | None" = None,
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
        source_character_names: ADAPTATION lanes only (shakespeare,
            public_domain_story). The SOURCE's real character names --
            MACBETH, BANQUO, the WITCHES. When supplied, each open slot
            takes the next unused source name (upper-cased, collision-
            checked) BEFORE falling back to the random pool roll, so the
            adaptation preserves the play's people instead of inventing
            "QUASIMODO VAUGHN". None/empty -> byte-identical to the pool
            path (C7: the invention lanes are untouched). When the source
            names run out (fewer than num_characters), the remaining
            slots fall through to pick_first_last as before.
        source_bank_id: canonical source-bank identifier. Shakespeare and
            public-domain adaptations exclude Lemmy even when the global cameo
            control requests him, preserving source fidelity.

    Returns:
        (pre_locked_rows, open_slots, lemmy_hit)
        - pre_locked_rows: list of fully-populated cast row dicts
            (ANNOUNCER plus LEMMY if hit). Each row has char_id,
            name, gender, voice_preset, character_description.
        - open_slots: list of CastSlot for the remaining open
            characters (Python rolled names, LLM fills the rest).
        - lemmy_hit: True if LEMMY was rolled in, False otherwise.
    """
    if not (1 <= num_characters <= _LEGACY_MAX_SPEAKING_CAST):
        raise ValueError(
            f"num_characters must be 1-{_LEGACY_MAX_SPEAKING_CAST}, "
            f"got {num_characters}"
        )

    rng = rng or random.Random()
    taken_names = set(taken_names or set())
    pre_locked: list[dict] = []

    # 1. ANNOUNCER first, char_id="c01", always.
    announcer = _POOLS.pick_announcer(rng)
    announcer["char_id"] = "c01"
    pre_locked.append(announcer)
    taken_names.add("ANNOUNCER")

    # 2. LEMMY: ONE DECISION FUNCTION, NOT A SECOND COPY OF THE RULE.
    #
    # This block used to re-implement `resolve_lemmy_cameo` inline -- the same
    # three branches in the same order (fidelity exclusion, then the natural
    # roll, then the forced knob), deciding the same thing from the same
    # inputs. The two agreed by luck and nothing made them keep agreeing: no
    # test compared them, and a change to the exclusion set or to the
    # precedence had to be remembered in two places. That is Bug Bible 12.132's
    # class exactly ("one matcher, never two"), and it cost the LEGACY lanes
    # their receipt -- a bare bool cannot say WHY, so `media_archive`,
    # `original` and `science_news` shipped 0 cameo receipts across 413
    # episodes while `scifi_news_pro`, which calls the real function, stamped
    # one every time.
    #
    # `roll_lemmy()` inside it still uses OS entropy, never the seeded `rng` --
    # the cameo stays decoupled from the C7 seed (BUG-LOCAL-260: a fixed seed
    # otherwise pinned LEMMY to 100% or 0%) -- and the fidelity exclusion still
    # outranks the operator knob.
    #
    # `decision` is passed in by `lock_cast` so the roll happens EXACTLY ONCE
    # per episode and the receipt cannot disagree with the cast. Left None (the
    # standalone/test path) it is resolved here, which is still the one
    # implementation.
    if decision is None:
        decision = resolve_lemmy_cameo(source_bank_id, force_lemmy)
    lemmy_hit = bool(decision.lemmy_hit)
    # READ OFF THE DECISION, never re-derived. The open-slot name filter below
    # needs this too: on a fidelity-bound adaptation a SOURCE character
    # literally named "Lemmy" is skipped, so the play's own people cannot
    # smuggle the cameo in through the back door. Taking it from the carried
    # policy keeps one authority -- calling `_source_bank_excludes_lemmy` again
    # here would restore the duplication this change exists to delete.
    source_fidelity_excludes_lemmy = (
        decision.lemmy_policy == LEMMY_POLICY_SOURCE_FIDELITY_EXCLUSION)

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

    # 3. Open-slot NAMES. ADAPTATION lanes: take the SOURCE's real character
    # names first (MACBETH, BANQUO), preserving the play's people; only fall back
    # to the random pool when the source has fewer named characters than slots.
    # INVENTION lanes pass source_character_names=None and this whole block is
    # byte-identical to the pool-only path (C7).
    source_queue: list[str] = []
    for raw in (source_character_names or []):
        nm = str(raw or "").strip().upper()
        # Skip blanks, ANNOUNCER (always pre-locked), Lemmy on fidelity-bound
        # adaptations, and anything already taken.
        if (
            nm
            and nm != "ANNOUNCER"
            and (not source_fidelity_excludes_lemmy or nm != "LEMMY")
            and nm not in taken_names
            and nm not in source_queue
        ):
            source_queue.append(nm)

    open_slots: list[CastSlot] = []
    for _ in range(remaining_open):
        if source_queue:
            name = source_queue.pop(0)   # the source's real character name
            source_owned = True
        else:
            name = _POOLS.pick_first_last(rng, taken_names)  # pool fallback
            source_owned = False
        taken_names.add(name)
        open_slots.append(CastSlot(
            char_id=f"c{next_cid_int:02d}",
            name=name,
            source_owned=source_owned,
        ))
        next_cid_int += 1

    return pre_locked, open_slots, lemmy_hit


# ---------------------------------------------------------------------------
# llm_slot_fill Pass-1 (S6) -- optional LLM naming overlay on the finished
# deterministic cast. Gated by OTR_NAME_MODE=llm_slot_fill; pool mode skips it
# entirely (byte-identical, C7).
# ---------------------------------------------------------------------------


def _extract_json_list(raw):
    """Tolerantly pull a JSON array out of an LLM response (handles code fences
    + surrounding prose). Returns a list, or None if none parses."""
    import json
    s = (raw or "").strip()
    if s.startswith("```"):
        parts = s.split("```")
        if len(parts) >= 2:
            s = parts[1]
        if s.lstrip().lower().startswith("json"):
            s = s.lstrip()[4:]
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return v
    except Exception:
        pass
    a, b = s.find("["), s.rfind("]")
    if a != -1 and b > a:
        try:
            v = json.loads(s[a:b + 1])
            if isinstance(v, list):
                return v
        except Exception:
            return None
    return None


def _build_pass1_prompt(plan, news_seed, style):
    """Compact, schema-locked Pass-1 naming prompt: a name + two texture notes
    per slot. Gender / voice / age / role are Python-fixed facts the LLM writes
    into, never chooses."""
    story = (news_seed or "").strip()[:_NEWS_SEED_CAP]
    style_str = (style or "").strip() or "open"
    lines = [
        "Name the cast of a radio drama. For EACH slot, return a name that fits "
        "the stated gender plus two short texture notes. Do NOT change gender, "
        "voice, age, or role.",
        f"Story: {story}",
        f"Style: {style_str}",
        "",
        "Slots:",
    ]
    for s in plan:
        lines.append(f"- {s.char_id}: {s.gender}, {s.age_band}, {s.dramatic_role}")
    lines += [
        "",
        "Return ONLY a JSON array, one object per slot, with EXACTLY these keys:",
        '[{"char_id":"c02","name":"First Last",'
        '"one_line_presence":"<6-10 words>","dialogue_style":"<6-10 words>"}]',
        "Names must fit the stated gender. No duplicate names. No extra keys.",
    ]
    return "\n".join(lines)


def _apply_llm_slot_fill(
    cast, ensemble_slots, voice_by_char_id, age_by_char_id,
    *, generate_fn, news_seed, style, cast_seed, meta,
):
    """Overlay LLM names + texture onto the finished deterministic cast. ONE
    creative-slot call, NO retry (per the sprint plan). This lane is OPT-IN
    (name_mode == "llm_slot_fill"); NO-FALLBACK rip (2026-07-03): when the naming
    LLM is selected and it fails (raises) or returns un-validatable output, this
    FAILS LOUD (CastValidationLLMError) rather than silently keeping the
    deterministic RNG-pool names -- a failed naming LLM stops the episode. A
    successful LLM name still passes the same gender-coherence repair as S2
    (isolated rng), so the result is always coherent in strict mode.
    """
    meta["name_mode"] = "llm_slot_fill"
    plan = _CASTPLAN.build_cast_plan(
        ensemble_slots, voice_by_char_id, age_band_by_char_id=age_by_char_id)
    prompt = _build_pass1_prompt(plan, news_seed, style)
    # LLM slot: creative -- cast naming + texture is a creative-writing pass; it
    # reuses the writer's creative_fn (no new model_id widget, PD6).
    try:
        raw = generate_fn(
            [{"role": "user", "content": prompt}],
            temperature=0.7, max_new_tokens=400,
        )
    except Exception as exc:  # noqa: BLE001 -- loader/LLM varies
        # NO-FALLBACK rip (2026-07-03): opt-in naming LLM failure = LOUD stop,
        # not a silent keep of the deterministic RNG-pool names.
        raise CastValidationLLMError(
            [("", f"llm_slot_fill naming generate_fn raised "
                  f"({type(exc).__name__}: {exc}) -- NO deterministic-name "
                  "fallback (no-fallback rip 2026-07-03); a failed naming LLM "
                  "stops the episode.")],
            "llm_slot_fill",
        ) from exc
    items = _extract_json_list(raw)
    result = _CASTVAL.validate_pass1(items if items is not None else raw, plan)
    if not result.ok:
        raise CastValidationLLMError(
            [(str(raw)[:200], f"llm_slot_fill naming output failed validation "
                              f"({result.reason}) -- NO deterministic-name "
                              "fallback (no-fallback rip 2026-07-03); a failed "
                              "naming LLM stops the episode.")],
            "llm_slot_fill",
        )
    # Gender-coherence repair on the LLM names (the LLM may not honour gender).
    rate = _otr_cast_env.cross_gender_rate()
    gender_by_id = {s.char_id: s.gender for s in plan}
    plan_ids = {s.char_id for s in plan}
    taken = {row.get("name") for row in cast if row.get("char_id") not in plan_ids}
    final_names: dict = {}
    for cid, name in result.names_by_char_id.items():
        final = name
        g = gender_by_id.get(cid, "")
        if g in ("male", "female"):
            tag = _POOLS.gender_of_first_name(name)
            if tag in ("male", "female") and tag != g:
                iso = random.Random(f"{cast_seed}:{cid}:llm")
                keep_cross = rate > 0.0 and iso.random() < rate
                if not keep_cross:
                    swapped = _pick_same_gender_first_name(name, g, iso, taken)
                    if swapped:
                        final = swapped
        final_names[cid] = final
        taken.add(final)
    texture = result.texture_by_char_id
    for row in cast:
        cid = row.get("char_id")
        if cid in final_names:
            row["name"] = final_names[cid]
            if cid in texture:
                row["cast_texture"] = texture[cid]
    meta["llm_naming_applied"] = True
    meta["cast_texture"] = texture
    return cast


# ---------------------------------------------------------------------------
# Top-level: lock_cast -- runs the LLM call per open slot, returns
# the full locked cast.
# ---------------------------------------------------------------------------


def _deterministic_identity_floor(slot: "EnsembleSlot") -> str:
    """A valid, on-format description built from Python-owned slot facts only.

    THE LAST RUNG, and it exists because no mechanism in this item may fail an
    episode. It is deliberately NOT a fixed string: BUG-098 was one generic
    fallback producing ONE portrait for a whole cast, so the age band, role and
    the role's own face pressure are woven in to keep rows distinguishable.
    It names an occupation, never a person -- occupations are demonstrably what
    the model writes when it gets this right.
    """
    role = (getattr(slot, "role", "") or "featured player").strip()
    timbre = (getattr(slot, "timbre", "") or "clear").strip()
    pressure = _FACE_PRESSURE_BY_ROLE.get(role.lower(), "steady, unhurried features")
    return (
        f"Adult, the story's {role}. Face: {pressure}. "
        f"Presence: carries the episode's pressure without flourish. "
        f"Voice: {timbre}, plainly spoken."
    )


def _enforce_name_authority(
    response: "CastingResponse",
    *,
    generate_fn: Callable[..., str],
    slot: "EnsembleSlot",
    news_seed: str,
    style: str,
    casting_brief: str,
    superseded_names: List[str],
    roster_names: List[str],
) -> "tuple[CastingResponse, Optional[dict]]":
    """Bug Bible 11.61 layers 2+3. TOTAL: never raises, never fails an episode.

    Returns ``(response, event_or_None)``. A clean row returns the response
    untouched and ``None``, so every lane without superseded identities is
    byte-identical.

    The contaminated response is DISCARDED WHOLE and regenerated from trusted
    slot facts -- never name-swapped in place. 11.61 is explicit that the reflex
    repair is actively harmful: it rewrites the foreign name to the record's own
    and leaves that other person's face, bearing and delivery prose sitting
    there, so the record still describes the wrong human and nothing points at
    it any more.

    The regeneration is a CLEAN ROOM: no story, no brief, no prior cast, no
    rejected prose, and ``max_attempts=1``. Re-running the same reconciled
    prompt would be mere resampling and can repeat verbatim, so it would not be
    an independent second chance.
    """
    if not superseded_names:
        return response, None

    fields = {
        "character_description": response.character_description,
        "speech_signature": getattr(response, "speech_signature", "") or "",
    }
    hits = _NAMES.find_foreign_identities(fields, superseded_names, roster_names)
    # The model can also copy the neutral LABEL out of the reconciled brief --
    # "30s, CHARACTER A." is not a wrong-person description but it is broken
    # text that would be spoken, printed in the credits and painted into a
    # portrait. The identity detector cannot see it, because the label is the
    # thing that replaced the identity.
    labels = [_NAMES.default_label(i) for i in range(len(superseded_names))]
    leaked = _NAMES.find_leaked_labels(fields, labels)
    if not hits and not leaked:
        return response, None

    event = {
        "rung": "detected",
        "matched": sorted({h.matched for h in hits} | set(leaked)),
        "surfaces": sorted({h.field for h in hits}) or sorted(
            k for k, v in fields.items() if v),
        "identities": sorted({h.identity for h in hits}),
        "leaked_labels": sorted(leaked),
    }
    log.warning(
        "[OTR_Casting] name-authority guard fired on %s: %s in %s "
        "-- discarding the response and regenerating clean-room",
        slot.name, event["matched"], event["surfaces"],
    )

    try:
        clean = llm_write_description(
            generate_fn,
            slot=slot,
            # THE RETRY KEEPS THE RECONCILED STORY CONTEXT. An earlier version
            # stripped everything, and that was the wrong trade: the row was
            # regenerated with no premise, no world and no plot, so the model
            # could only invent generic filler -- which is then copied verbatim
            # into the FLUX portrait prompt. It swapped a row that was wrong
            # about WHO for a row that is about NOTHING, and permanently
            # detached that character from the episode.
            #
            # The context is safe to keep because it is the reconciled text:
            # the superseded names are already gone from it. What is dropped is
            # `prior_cast`, and that IS the genuinely different lever -- rows
            # written earlier may themselves be contaminated, and
            # `_format_prior_entry` echoes their lead sentence back as
            # authoritative "Cast so far" context, which is a live path for
            # reintroducing the very person being removed.
            news_seed=news_seed,
            style=style,
            prior_cast=[],
            max_attempts=1,
            casting_brief=casting_brief,
        )
        retry_fields = {
            "character_description": clean.character_description,
            "speech_signature": getattr(clean, "speech_signature", "") or "",
        }
        if not _NAMES.find_foreign_identities(
                retry_fields, superseded_names, roster_names)                 and not _NAMES.find_leaked_labels(retry_fields, labels):
            event["rung"] = "regenerated"
            return response.model_copy(update={
                "character_description": clean.character_description,
                "speech_signature": (
                    getattr(clean, "speech_signature", "") or ""
                ).strip(),
            }), event
        event["retry_also_contaminated"] = True
    except Exception as exc:  # noqa: BLE001 -- the floor exists for exactly this
        event["retry_error"] = f"{type(exc).__name__}: {exc}"

    event["rung"] = "floor"
    log.warning(
        "[OTR_Casting] name-authority guard: %s fell to the deterministic "
        "floor (%s)", slot.name,
        event.get("retry_error") or "regeneration was also contaminated",
    )
    return response.model_copy(update={
        "character_description": _deterministic_identity_floor(slot),
        "speech_signature": "plain spoken",
    }), event


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
    source_character_names: Optional[List[str]] = None,
    source_bank_id: str | None = None,
    source_character_genders: Optional[Mapping[str, Mapping[str, str]]] = None,
    upstream_identity_names: Optional[List[str]] = None,
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
    counts per open slot, for telemetry), and name_authority (the
    Bug Bible 11.61 guard's structured events -- see below).

    `upstream_identity_names` is LANE-NEUTRAL by design (Bug Bible 11.61).
    It is "names an earlier pass invented for the people in this story,
    which this cast is about to override" -- pitch cast, treatment
    characters, article subjects. Nothing here knows which lane produced
    them. Every name the assembled roster OWNS is dropped from the set,
    so an adaptation lane whose roster IS the source's cast supplies its
    names harmlessly and its fidelity is never attacked. Omit it (None)
    and this whole mechanism is inert: prompts stay byte-identical.

    The guard runs in three layers, and NONE of them can fail an episode
    (operator rule 2026-08-20: no mechanism in this item may block,
    reject, retire or fail an episode):
      1. the brief is RECONCILED before it reaches any prompt, so the
         conflicting name is simply not in the context;
      2. every generated row is CHECKED (both prose fields) before it is
         appended or exposed as prior-cast context, and a contaminated
         response is discarded whole -- never rewritten in place, which
         11.61 calls actively harmful;
      3. a contaminated row is regenerated ONCE from trusted slot facts
         only, then falls to a deterministic floor.

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

    # llm_slot_fill (S6): name_mode decides whether an LLM Pass-1 renames the
    # cast AFTER the deterministic build. Pool mode (default) never enters that
    # path, so pool behavior is byte-identical (C7). The voice/age maps feed the
    # CastPlanner (S4) when the llm path runs.
    name_mode = _otr_cast_env.name_mode()
    voice_by_char_id: dict = {}
    age_by_char_id: dict = {}

    # Voice assignment (Stage 3) needs a seeded rng for its deterministic
    # tie-break; reuse the cast rng so a fixed seed stays byte-identical
    # (C7). A fresh Random() when the caller passed none keeps the
    # non-deterministic path working too.
    cast_rng = rng or random.Random()

    # A CAST SIZE IS A REQUEST, NOT A GATE -- so deliver the closest
    # performable cast instead of killing the episode.
    #
    # THE DEFECT (found 2026-08-24 by the character-selection trace). The
    # `num_characters` widget advertises 1-10 and `_resolve_inputs` clamps to
    # that same band, because the DISPATCHED lane really does seat up to
    # `MAX_SPEAKING_CAST` (10). This legacy assembler's ceiling is 6 -- the
    # voice stock it draws from -- and it enforced that by RAISING. So an
    # operator who moved the slider to 7 on `original` or `media_archive` got
    # an uncaught `ValueError: num_characters must be 1-6, got 7` out of
    # `assemble_pre_locked_rows`, with no try/except between here and the
    # node body: the run died after the RSS fetch, the bank roll and the
    # story-contract build had already spent minutes, and produced no episode.
    #
    # That is exactly the shape the standing directive forbids: "The target
    # value is a REQUEST, not a gate: no refusals, no hard caps, no shunts ...
    # a request beyond it simply delivers the closest performable episode."
    # `_resolve_inputs` already clamps rather than raises for the same reason.
    #
    # LOUDLY, never silently -- the operator asked for 8 and is getting 6, and
    # a quiet clamp is how that becomes a mystery instead of a decision. The
    # assembler keeps its own 1-6 contract intact (a 0 or a negative is still
    # a programming error and still raises); this only converts an
    # OVER-request into the largest cast this lane can actually voice.
    if num_characters > _LEGACY_MAX_SPEAKING_CAST:
        log.warning(
            "[lock_cast] requested %d speaking characters; this lane's voice "
            "stock seats %d, so the cast is the closest performable size. "
            "The request is honoured as far as it can be, never refused.",
            num_characters, _LEGACY_MAX_SPEAKING_CAST,
        )
        num_characters = _LEGACY_MAX_SPEAKING_CAST

    # THE CAMEO IS DECIDED HERE, ONCE, AND CARRIED -- never re-derived.
    # Resolved before `assemble_pre_locked_rows` and handed to it, so the
    # single OS-entropy roll cannot happen twice and disagree with itself
    # between the cast rows and the receipt. This is the same shape the
    # dispatched `scifi_news_pro` lane already uses, and the chunk B build
    # contract asked for on BOTH lanes; only that one lane ever got it.
    lemmy_decision = resolve_lemmy_cameo(source_bank_id, force_lemmy)

    pre_locked, open_slots, lemmy_hit = assemble_pre_locked_rows(
        num_characters=num_characters,
        rng=cast_rng,
        force_lemmy=force_lemmy,
        source_character_names=source_character_names,
        source_bank_id=source_bank_id,
        decision=lemmy_decision,
    )

    cast: list[dict] = list(pre_locked)
    # Open-character voice exclusion set tracks BARK voices only.
    # At THIS writer-stage point the announcer row is ALWAYS Kokoro-shaped
    # (pick_announcer() stamps a "bm_george" / "bf_emma"-style id here,
    # never a Bark id -- separate namespace, so it could never collide with
    # Bark's "v2/en_speaker_X" pool anyway), so the announcer's voice is NOT
    # added here. Per Jeffrey 2026-05-10: "announcer is in Kokoro so there
    # can be no cast overlaps." OTR_CastLock may re-stamp the announcer to a
    # real Bark v2/* preset LATER (2026-08-24, announcer_voice_engine=
    # "bark") -- that happens downstream of this function and is exactly
    # why CastLock owns its own dynamic exclusion set
    # (`_assign_bark_announcer`) rather than trusting this one. LEMMY's
    # voice (v2/en_speaker_8, Bark) IS added when LEMMY is rolled in.
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
    # The source contract carries one entry per RESOLVED source character:
    # {NAME: {gender, evidence, tier, roster_name}}. The pin map is derived from
    # it here so there is exactly one input and one stamped receipt, and the
    # evidence that justified each pin travels into the ledger with it.
    source_contract = {
        str(name).strip().upper(): dict(spec)
        for name, spec in (source_character_genders or {}).items()
        if isinstance(spec, Mapping)
    }
    gender_by_name = {
        name: str(spec.get("gender") or "")
        for name, spec in source_contract.items()
    }
    ensemble_slots = precompute_ensemble_slots(
        open_slots,
        prior_cast=prior_cast_for_llm,
        rng=cast_rng,
        cast_seed=cast_seed,
        gender_by_name=gender_by_name or None,
    )

    # ---------------------------------------------------------------
    # Bug Bible 11.61 -- LAYER 1: reconcile the brief at the boundary.
    #
    # THIS IS THE ONLY POINT WHERE IT CAN BE DONE, and that is a fact
    # about the code rather than a preference. The redaction set is
    # "upstream identities the assembled roster does not own", so it
    # needs the roster; and the roster does not exist until BOTH
    # assemble_pre_locked_rows (ANNOUNCER/LEMMY + open-slot names) and
    # precompute_ensemble_slots (ens.name) have run, which is here. The
    # writer never sees open_slots or ensemble_slots, so it structurally
    # cannot compute this set -- and redacting every upstream name
    # unconditionally would strip MACBETH from a Macbeth adaptation.
    roster_names_for_authority = (
        [str(r.get("name") or "") for r in pre_locked]
        + [ens.name for ens in ensemble_slots]
    )
    superseded_names = _NAMES.superseded_identities(
        upstream_identity_names or [], roster_names_for_authority,
    )
    reconciled_brief, _identity_labels = _NAMES.reconcile_text(
        casting_brief, superseded_names, roster_names_for_authority,
    )
    # RECONCILE THE SEED TOO. `_build_user_prompt` falls back to a slice of
    # news_seed whenever the brief is empty, so leaving the seed raw would keep
    # a second door open into the same `Story:` line. Measured today: across all
    # 125 annotated-cohort ledgers, ZERO carry an upstream identity in the seed
    # (on this lane the seed is the spark digest, fixed before the concept pass
    # invents any name) -- so this is inert in practice and costs nothing. It is
    # here so the guarantee holds by construction rather than by that
    # coincidence continuing to be true on some future lane.
    reconciled_seed, _seed_labels = _NAMES.reconcile_text(
        news_seed, superseded_names, roster_names_for_authority,
    )
    # THE EMPTY-CONTEXT TRAP: _build_user_prompt falls back to the raw
    # news_seed slice when the brief is empty. If reconciliation ever
    # emptied a brief that HAD content, that fallback could reinstate the
    # very names just removed on a lane whose seed carries them. Keep the
    # reconciled brief non-empty by preferring the original's shape; an
    # already-empty brief is left empty and the legacy fallback stands.
    if casting_brief.strip() and not reconciled_brief.strip():
        # NEVER `or casting_brief` here. Reaching this branch means
        # reconciliation emptied a brief that HAD content, which can only
        # happen with superseded names present -- so falling back to the
        # original would restore exactly the names this guard exists to
        # remove, turning the safety net into the injection path. A bare
        # label is the safe floor: non-empty, and naming nobody.
        reconciled_brief = " ".join(
            _NAMES.default_label(i) for i in range(len(superseded_names))
        ) or _NAMES.default_label(0)
    name_authority_events: list[dict] = []

    casting_attempts: list[int] = []
    for i, (slot, ens) in enumerate(zip(open_slots, ensemble_slots)):
        # Age axis (S5) is active ONLY in llm_slot_fill mode; pool mode passes
        # None so voice picks stay byte-identical (C7).
        age_band = (_CASTPLAN.age_band_for_index(i)
                    if name_mode == "llm_slot_fill" else None)
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
                # RECONCILED seed, matching the reconciled brief: the prompt
                # builder falls back to this slice whenever the brief is empty.
                news_seed=reconciled_seed,
                style=style,
                prior_cast=prior_cast_for_llm,
                available_voices=available_voices,
                max_attempts=max_attempts_per_call,
                # RECONCILED, never the raw brief. Bug Bible 11.61 verify (6)
                # asserts this call site statically, so a later refactor cannot
                # quietly restore the two-authority prompt.
                casting_brief=reconciled_brief,
                ensemble_slot=ens,
                rng=cast_rng,
                age_band=age_band,
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

        # -----------------------------------------------------------
        # Bug Bible 11.61 -- LAYERS 2 and 3: check, then regenerate.
        #
        # This runs AFTER a successful response and BEFORE the row is
        # appended or exposed via prior_cast_for_llm, which is what cuts
        # the amplifier: _format_prior_entry echoes the lead sentence
        # into every later slot's prompt in the same episode, so a
        # contaminated row left here re-injects the wrong person as
        # "Cast so far" context for the rest of the cast.
        #
        # It is deliberately NOT a structured_call post_validator:
        # validator exhaustion raises StructuredCallFailedError, which
        # llm_write_description converts to CastingFailedError and
        # lock_cast re-raises -- i.e. it could FAIL AN EPISODE, which the
        # operator forbids for this mechanism.
        response, guard_event = _enforce_name_authority(
            response,
            generate_fn=generate_fn,
            slot=ens,
            news_seed=reconciled_seed,
            style=style,
            casting_brief=reconciled_brief,
            superseded_names=superseded_names,
            roster_names=roster_names_for_authority,
        )
        if guard_event:
            guard_event["char_id"] = slot.char_id
            guard_event["name"] = ens.name
            name_authority_events.append(guard_event)

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
            # F5 (story-engine v1): deterministic backfill so EVERY locked
            # cast row carries a non-empty speech_signature for the composer.
            "speech_signature":      (response.speech_signature or "plain spoken"),
        }
        cast.append(new_row)
        taken_voices.add(response.voice_preset)
        prior_cast_for_llm.append(new_row)
        voice_by_char_id[ens.char_id] = response.voice_preset
        age_by_char_id[ens.char_id] = age_band
        # Telemetry: how many attempts did this slot need? We can't
        # see it from the response object; the caller can wrap
        # cast_one_character if granular telemetry is needed. For now
        # just stamp 1 -- a successful call returned without raising.
        casting_attempts.append(1)

    meta: dict = {}
    # Bug Bible 11.61: the guard's structured events. lock_cast has no episode
    # identity, so it cannot satisfy "always log the episode and row" alone --
    # it reports rows and the WRITER attaches the episode and persists them.
    # Always present (empty list when clean) so a consumer can distinguish
    # "checked, nothing found" from "never ran".
    meta["name_authority"] = {
        "upstream_identities": list(superseded_names),
        "events": name_authority_events,
    }
    # llm_slot_fill Pass-1 (S6): overlay LLM names + texture onto the
    # finished, already-coherent deterministic cast. Names are accepted as
    # authored strings; Python does not classify them by vocabulary.
    #
    # 11.61 FENCE: this is a SECOND naming authority and it acts AFTER every
    # description is written, so the reconciliation above -- which was computed
    # against the pre-fill roster -- does not describe the names this cast ends
    # up with. The guarantee genuinely does not hold in this mode, so it is
    # recorded rather than quietly assumed. Pool mode is the production default
    # and is unaffected.
    if name_mode == "llm_slot_fill" and superseded_names:
        meta["name_authority"]["unfenced_mode"] = "llm_slot_fill"
        log.warning(
            "[OTR_Casting] OTR_NAME_MODE=llm_slot_fill renames rows AFTER "
            "their descriptions are written, so the 11.61 boundary guarantee "
            "does not hold for this cast (%d upstream identities reconciled "
            "against the pre-fill roster).", len(superseded_names),
        )
    if name_mode == "llm_slot_fill":
        cast = _apply_llm_slot_fill(
            cast, ensemble_slots, voice_by_char_id, age_by_char_id,
            generate_fn=generate_fn, news_seed=news_seed, style=style,
            cast_seed=cast_seed, meta=meta,
        )

    # Sprint 2 (a): the bark voice_preset + uniqueness invariants relocated to
    # OTR_CastLock's exit. The writer no longer assigns voice_preset (CastLock
    # replays it byte-identically after the freeze), so asserting v2/* here would
    # fail on the now-empty rows. _assert_unique_bark_voices +
    # _assert_voice_preset_invariant run in OTR_CastLock after it stamps voices.
    # THE HYBRID LLM VOICE-FIT IS GONE (ripped 2026-08-18). The deterministic
    # scorer in `_otr_voice_bank.assign_voice_for_slot` is the caster.
    #
    # It was removed rather than tuned because it had no information the scorer
    # lacks: its prompt carried the character's gender / timbre / role / age and
    # each card's age_band + timbre + style_tags -- exactly the four dimensions
    # `_score()` already weights -- and per I-9 no character name and no
    # description. Measured over 1711 ledgers it cast with 13 distinct voices at
    # 96% top-5 where the scorer used 43 at 25%, chose card #0 of an
    # alphabetically ordered list 62% of the time, and was the one unseeded step
    # in a pipeline whose contract is seed-determinism.
    #
    # `voice_cast_decision` is KEPT and stamped empty, deliberately. CastLock
    # still reads `meta.get("voice_cast_decision") or {}`, and every published
    # ledger keeps one stable shape rather than gaining and losing a key.
    voice_cast_decision: dict = {}
    # WHICH CASTER RAN, stamped positively so a published episode can say so
    # rather than leaving a reader to infer it from an absent field.
    voice_cast_mode = "scorer"

    # VC chunk 3 (2026-06-22): stamp meta.cast_voice_slots so OTR_CastLock can
    # match a bank voice on timbre / age_band (not just gender). The cast ROW
    # schema is frozen and carries no timbre/role/age, so these ride free-form
    # meta. The voice-fit facts come from the Python-decided ensemble slots
    # (timbre/role/age_band); gender + speech_signature come off the locked row;
    # description_digest is a short, PII-free sha1 of the prose (lets CastLock /
    # the hybrid caster key on description identity without storing the text).
    ens_by_id = {e.char_id: e for e in ensemble_slots}
    cast_voice_slots: dict = {}
    for row in cast:
        if not isinstance(row, dict):
            continue
        cid = str(row.get("char_id") or "")
        if not cid:
            continue
        ens = ens_by_id.get(cid)
        desc = str(row.get("character_description") or "")
        digest = (
            hashlib.sha1(desc.encode("utf-8")).hexdigest()[:12] if desc else ""
        )
        cast_voice_slots[cid] = {
            "gender":             str(row.get("gender") or ""),
            # timbre is a LIST so it feeds the bank caster's set-intersection
            # match (one Python-decided timbre word per open slot; empty for the
            # pre-locked announcer / LEMMY rows which have no ensemble slot).
            "timbre":             [ens.timbre] if ens else [],
            "role":              (ens.role if ens else ""),
            "age_band":          (ens.age_band if ens else ""),
            "speech_signature":   str(row.get("speech_signature") or ""),
            "description_digest": digest,
        }

    meta.update({
        "lemmy_hit":              lemmy_hit,
        # BOTH READ THE CARRIED DECISION -- this key used to re-derive the
        # policy from `_source_bank_excludes_lemmy` all over again, a THIRD
        # independent derivation of the same rule sitting a thousand lines from
        # the first two. It could only ever express two of the four outcomes,
        # so a forced include and a natural hit stamped identically.
        "lemmy_policy":           lemmy_decision.lemmy_policy,
        # THE RECEIPT THE LEGACY LANES NEVER HAD. `lemmy_hit` alone cannot say
        # whether the roll was SPENT: a harness forcing the cameo off and a
        # roll that came up short both stamp False, which is exactly why the
        # shipped-episode rate was unreadable. `to_meta()` carries knob_state
        # and roll_executed alongside, so "asked and declined" is finally
        # distinguishable from "never asked" and from "forced off".
        # Primitive-only by construction -- `Ledger.save()` never raises, so a
        # non-serializable value here would lose the receipt in silence.
        "lemmy_roll_receipt":     lemmy_decision.to_meta(),
        "casting_attempts":       casting_attempts,
        "num_characters_request": num_characters,
        "num_characters_locked":  len(cast) - 1,  # minus ANNOUNCER
        "cast_voice_slots":       cast_voice_slots,
        "voice_cast_decision":    voice_cast_decision,
        # Which caster actually ran: "scorer" | "hybrid" | "hybrid_unavailable".
        # MUST also be copied at OTR_LedgerScriptWriter's key-by-key meta copy or
        # it never reaches the ledger -- see the invariant stated there.
        "voice_cast_mode":        voice_cast_mode,
        # ONE stable shape on EVERY lane -- the invention lanes stamp an empty
        # contract rather than omitting the key, so a downstream reader never has
        # to distinguish "no source" from "field never written".
        "cast_source_contract":   {
            "source_bank_id":  str(source_bank_id or ""),
            "character_names": [str(n) for n in (source_character_names or [])],
            "gender_by_name":  {
                name: spec["gender"] for name, spec in source_contract.items()
                if spec.get("gender")
            },
            "evidence":        source_contract,
        },
    })
    return cast, meta


def _row_is_announcer(row: dict) -> bool:
    """The ONE canonical "is this row the announcer" predicate
    (``_otr_cast_voice_coverage.is_announcer_cast_row``), re-exposed under
    this module's own name. NOT re-implemented: this module used to carry a
    hand-rolled duplicate of ``cast_lock._is_announcer_entry`` (import from
    ``cast_lock`` is unsafe -- it already imports this module, a cycle), but
    ``_otr_cast_voice_coverage`` imports neither, so it is the safe shared
    home (kibitz r3, cursor: unify rather than grow a fourth near-duplicate).

    Relative-then-absolute fallback (matches ``_otr_voice_bank.bark_preset_
    gender``'s existing pattern): ``lock_cast``'s own dynamic-import path can
    leave THIS module loaded without its normal ``nodes`` package context
    (bare ``_otr_casting`` in ``sys.modules`` rather than
    ``nodes._otr_casting``), which makes the relative import below raise
    ``ImportError: attempted relative import with no known parent package``
    -- caught live by the suite, not hypothetical."""
    try:
        from ._otr_cast_voice_coverage import is_announcer_cast_row
    except ImportError:
        import os
        import sys

        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from _otr_cast_voice_coverage import is_announcer_cast_row  # type: ignore

    return is_announcer_cast_row(row)


def _assert_voice_preset_invariant(cast: List[dict]) -> None:
    """Gate 1 (writer cast-lock exit) -- the earliest of three gates
    enforcing the cast.voice_preset contract for the voice-path-cleanbreak.

    Every non-ANNOUNCER cast row must carry a non-empty ``voice_preset``
    starting with ``v2/`` (the Bark preset namespace). The ANNOUNCER row is
    excluded UNLESS it was itself delivered on Bark (``tts_model == "bark"``,
    2026-08-24) -- Bark is no longer exclusively a character engine, so an
    announcer actually rendered on Bark must satisfy the same v2/* contract
    a character row does. A non-Bark announcer (the common case, Kokoro's
    ``bm_*`` / ``bf_*`` namespace) stays exempt exactly as before.

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
        if _row_is_announcer(row) and row.get("tts_model") != "bark":
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


def _assert_unique_bark_voices(cast: List[dict]) -> None:
    """Raise CastingFailedError if any two Bark cast rows share a
    voice_preset. A non-Bark ANNOUNCER (Kokoro namespace) is excluded; an
    announcer actually delivered on Bark (``tts_model == "bark"``,
    2026-08-24) is included, same as any character row -- two shipping
    8gb-tier profiles mix ``char_voice_engine: bark`` with a Kokoro
    announcer, and both legitimately carry the SAME leftover Bark-namespace
    label with no live collision; only a bark-delivered announcer needs the
    uniqueness guarantee a bark-delivered character already gets.

    Called at the end of lock_cast() as a final invariant check.
    Today this is guaranteed-true by the pre-filter + validator +
    reroll path; this assertion catches any future regression.
    """
    bark_voices: list[tuple[str, str]] = []  # (char_id, voice_preset)
    for row in cast:
        if _row_is_announcer(row) and row.get("tts_model") != "bark":
            continue
        bark_voices.append((row.get("char_id"), row.get("voice_preset")))
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
    "CastSlot",
    "EnsembleSlot",
    "assemble_pre_locked_rows",
    "precompute_ensemble_slots",
    "llm_write_description",
    "python_assign_voice_preset",
    "replay_voice_assignment",
    "cast_one_character",
    "lock_cast",
]
