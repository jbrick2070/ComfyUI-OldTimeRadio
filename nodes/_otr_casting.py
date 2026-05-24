r"""
nodes/_otr_casting.py -- cast contract LLM caller (PROSE-PLANE).

Plane assignment: PROSE (Content LLM). The cast row carries
audience-facing fields (character_description, gender, voice_preset)
so the same model that writes dialogue, polishes title, and
generates visual prompts also makes the casting choices. Voice pool
validation is a Python concern; it is NOT a reason to route casting
to the structural LLM.
See `spaces/.../memory/project_cast_contract_architecture_target.md`
and `spaces/.../memory/project_llm_agnostic_design_constraint.md`
for the full architectural spec.

Call shape: ONE LLM CALL PER OPEN CHARACTER.
  - Smaller prompt per call (~120-150 tokens) than a single call
    covering all characters together.
  - Voice collisions impossible by construction: Python pre-filters
    `available_voices = full_pool - taken_voices` before each call.
  - Each call sees a "cast so far" context (LEMMY if rolled, plus
    all previously-cast open characters). ANNOUNCER excluded --
    narrator role, not part of the dramatic ensemble.

Cast assembly order (Python-only, no LLM):
  1. ANNOUNCER pre-baked at char_id="c01"  (always present, bonus)
  2. LEMMY pre-baked at char_id="c02"      (11% roll, consumes a slot)
  3. Pool-fill open characters             (LLM-cast, c02..cNN
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

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

from pydantic import BaseModel, Field, ValidationError, field_validator

# Shared tolerant JSON extractor (BUG-LOCAL-261 consolidation). Package
# import in production; flat import when loaded standalone / under test.
try:
    from . import _otr_json
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_json  # type: ignore

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

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_VALID_GENDERS = {"male", "female", "other"}


class CastingResponse(BaseModel):
    """LLM response shape for one open-character casting call.

    Voice-in-pool validation is NOT done here -- pydantic cannot see
    the runtime available_voices set. The caller checks separately
    and rerolls if the response picks a voice outside the pool.
    """

    character_description: str = Field(..., min_length=10, max_length=200)
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
    available_voices: List[tuple[str, str]],
    casting_brief: str = "",
) -> str:
    """Build the casting prompt for one open character.

    Layout (every line except 'Cast so far' is mandatory; the cast-
    so-far block is omitted entirely when prior_cast is empty):

      Cast this character in a radio drama.
      Story: <casting_brief if non-empty else news_seed[:500]>
      Style: <style>

      Name: <NAME>

      [optional]
      Cast so far:
      - LEMMY (M, gravelly engineer, 50s, gruff mechanic)
      - BOB   (M, weary doctor, 40s, dry humor)

      Voices:
      - v2/en_speaker_4 (female bright 30s)
      - v2/en_speaker_6 (female throaty 40s)

      Aim ~40% male, ~40% female, ~20% other.

      JSON only:
      {"character_description":"<short>","gender":"male|female|other","voice_preset":"<id>"}

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
        "Cast this character in a radio drama.",
        f"Story: {story_text}",
        f"Style: {style_str}",
        "",
        f"Name: {name}",
    ]

    if prior_cast:
        parts.append("")
        parts.append("Cast so far:")
        for c in prior_cast:
            parts.append(f"- {_format_prior_entry(c)}")

    parts.append("")
    parts.append("Voices:")
    for preset, short in available_voices:
        parts.append(f"- {preset} ({short})")

    parts.append("")
    parts.append("Aim ~40% male, ~40% female, ~20% other.")
    parts.append("")
    parts.append("JSON only:")
    parts.append(
        '{"character_description":"<short>",'
        '"gender":"male|female|other",'
        '"voice_preset":"<id>"}'
    )
    return "\n".join(parts)


def _format_prior_entry(row: dict) -> str:
    """Compact one-line summary of a prior cast row for the
    'Cast so far' block. Format: 'NAME (G, description)'.
    """
    name = (row.get("name") or "?").upper()
    g = (row.get("gender") or "?").lower()
    g_short = "M" if g == "male" else "F" if g == "female" else "X"
    desc = (row.get("character_description") or "").strip()
    # Trim long descriptions so the prompt stays lean. Strip a wider
    # set of trailing punctuation so the appended "..." doesn't read
    # as e.g. "weary broadcaster!..." -- per round-robin nit
    # 2026-05-10.
    if len(desc) > 60:
        desc = desc[:57].rstrip(",.;:!? ") + "..."
    return f"{name} ({g_short}, {desc})"


# Cap on how much of a prior raw response we embed into the repair
# prompt on attempt 3. Local LLMs occasionally babble 4000+ tokens of
# malformed garbage; passing that into the repair prompt blows the KV
# cache on the 14.5 GB VRAM ceiling. 1200 chars is plenty for the
# LLM to see "what it tried last time" without risking OOM. Per
# round-robin synthesis 2026-05-10 (both ChatGPT and Gemini flagged).
_REPAIR_RAW_CAP_CHARS = 1200


# ---------------------------------------------------------------------------
# Per-character LLM caller -- validate + 3-attempt reroll
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
    validation_fn: Optional[Callable[..., str]] = None,
) -> CastingResponse:
    """Cast one open character. Returns a validated CastingResponse.

    Retry strategy mirrors `_otr_outline.generate_outline`:
      Attempt 1: fresh, base_temperature (0.7).
      Attempt 2: fresh, base_temperature + 0.1 (0.8).
      Attempt 3: REPAIR call -- prior raw output (truncated) + the
                 validation error, temp 0.3.

    `max_attempts=1` is allowed (single-shot, no repair). `0` is not.

    Raises CastingFailedError if all attempts fail.
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

    available_presets = {p for p, _ in available_voices}
    user_prompt = _build_user_prompt(
        name=name,
        news_seed=news_seed,
        style=style,
        prior_cast=prior_cast,
        available_voices=available_voices,
        casting_brief=casting_brief,
    )
    attempts: list[tuple[str, str]] = []
    last_raw: str | None = None

    for attempt_idx in range(max_attempts):
        # Repair branch: only on the final attempt, only when there
        # IS a prior attempt (so attempt_idx > 0), only when the
        # prior attempt produced a string we can hand back. Use
        # `is not None` so an empty-string prior does not silently
        # change branch semantics. Per round-robin nit 2026-05-10.
        is_repair = (
            attempt_idx == max_attempts - 1
            and attempt_idx > 0
            and last_raw is not None
        )
        if is_repair:
            # Repair attempt: hand the prior raw (TRUNCATED) + last
            # error back to the LLM. Lower temperature for a stable
            # recovery. Truncation is a VRAM ceiling protection --
            # see _REPAIR_RAW_CAP_CHARS.
            last_err = attempts[-1][1] if attempts else "validation failed"
            truncated_raw = last_raw[:_REPAIR_RAW_CAP_CHARS]
            messages = [
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": truncated_raw},
                {
                    "role": "user",
                    "content": (
                        "That response did not validate. Error:\n"
                        f"{last_err}\n\n"
                        "Return ONLY corrected JSON matching the schema. "
                        "No prose. No code fences."
                    ),
                },
            ]
            temperature = 0.3
        else:
            # Fresh attempt -- single user message, no system role.
            # Model loader's chat-template handles role-folding for
            # models that require a system role.
            messages = [{"role": "user", "content": user_prompt}]
            temperature = base_temperature + (0.1 * attempt_idx)

        # S32 B3 (D2): repair attempt routes to `validation_fn`
        # (technical slot) when provided. Generation attempts
        # 1..N-1 stay on `generate_fn` (creative slot). When
        # validation_fn is unset (None), repair falls back to
        # generate_fn -- preserves the legacy single-fn contract
        # for any direct caller.
        active_fn = validation_fn if (is_repair and validation_fn is not None) else generate_fn
        try:
            raw = active_fn(
                messages,
                temperature=float(temperature),
                max_new_tokens=int(max_new_tokens),
            )
        except Exception as exc:  # noqa: BLE001
            attempts.append(("", f"generate_fn raised: {exc!r}"))
            continue

        last_raw = raw or ""
        try:
            parsed = _otr_json.parse_first_json_object(last_raw)
        except json.JSONDecodeError as exc:
            attempts.append((last_raw, f"json parse failed: {exc!r}"))
            continue

        try:
            response = CastingResponse(**parsed)
        except ValidationError as exc:
            attempts.append((last_raw, f"schema validation failed: {exc!r}"))
            continue

        # Voice pool check -- separate from pydantic since the pool
        # is runtime data, not schema.
        if response.voice_preset not in available_presets:
            err = (
                f"voice_preset {response.voice_preset!r} not in "
                f"available_voices ({sorted(available_presets)!r})"
            )
            attempts.append((last_raw, err))
            continue

        log.info(
            "[OTR_Casting] cast %s -> voice=%s gender=%s (attempt %d/%d)",
            name, response.voice_preset, response.gender,
            attempt_idx + 1, max_attempts,
        )
        return response

    raise CastingFailedError(attempts=attempts, name=name)


# ---------------------------------------------------------------------------
# Cast assembly -- Python only (no LLM)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CastSlot:
    """A pool-fill open slot. Python rolls the name; the LLM fills
    description / gender / voice_preset later via cast_one_character.
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
    technical_fn: Callable[..., str],
    num_characters: int,
    news_seed: str,
    style: str,
    rng: Optional[random.Random] = None,
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
    """
    # S32 B3 (D2): generation runs through creative_fn; schema-
    # validation repair attempt routes to technical_fn (single
    # attempt, fail-fast). The body alias `generate_fn` keeps the
    # call sites readable; `validation_fn` is threaded into
    # `cast_one_character` as the dedicated repair-slot callable.
    generate_fn = creative_fn

    pre_locked, open_slots, lemmy_hit = assemble_pre_locked_rows(
        num_characters=num_characters,
        rng=rng,
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

    casting_attempts: list[int] = []
    for slot in open_slots:
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
                name=slot.name,
                news_seed=news_seed,
                style=style,
                prior_cast=prior_cast_for_llm,
                available_voices=available_voices,
                max_attempts=max_attempts_per_call,
                casting_brief=casting_brief,
                validation_fn=technical_fn,
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
            "name":                  slot.name,
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
    "CastingFailedError",
    "_assert_unique_bark_voices",
    "_assert_voice_preset_invariant",
    "_assert_no_structural_tokens_in_cast",
    "_looks_like_non_character_cast_name",
    "CastSlot",
    "assemble_pre_locked_rows",
    "cast_one_character",
    "lock_cast",
]
