"""nodes/_otr_story_brief.py -- reflection pure module (Sprint C C5a1).

The reflection pass is a single LLM call that reads the JUST-COMPOSED
ledger (cast + lines + meta.style) and produces a compact visual brief
plus three sidecar term arrays (setting / lighting / atmosphere).
Downstream visual / audio consumers read the brief via the central
helpers in `_otr_story_brief_helpers.py` (shipped at C5b).

This module is PURE: no I/O, no GPU, no ComfyUI imports, no writer
imports. It is tagged `# LLM slot: technical` at the entrypoint and
runs on the writer's technical slot per L-2 (so creative-slot
narrative passes are untouched, and a two-model config can use a
cheap structured-output model here without spending the creative
budget on a JSON validation pass).

C5a1 ships the module ONLY -- writer wiring at K.5.5 lands at C5a2.

Design references:
- Sprint C: L-6 fail-loud pattern, L-8 8-key meta delta,
  E-17 scoped try/except, E-18 repair temp clamp, E-21 technical_fn-
  only signature, R-06 CRITICAL prefix, R-01 single-load cache profile.
- `docs/2026-05-12-story-brief-v2-design-refinements.md` (post-C0b):
  section 2 input builder, section 3.1 strict JSON, section 3.2 LLM
  settings, section 3.3 no-period rule, section 3.4 validation gate,
  section 3.5 repair pass, section 4 8-key storage schema, section 4.1
  empty-string failure mode.
- `nodes/_otr_ledger_reviewer.py` `run_script_doctor` -- L-6 reference
  implementation of the 3-arm scoped try/except pattern.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants -- LLM settings, length caps, prompt version
# ---------------------------------------------------------------------------

# Per refinement section 3.2: temperature 0.2-0.4 keeps the model
# compliance-honest for JSON output. 0.3 is the centerpoint.
_REFLECTION_TEMPERATURE: float = 0.3

# BUG-LOCAL-285 (2026-05-27): bumped 160 -> 512. The v2 schema added
# five new fields (music_mood_terms / visual_palette / key_objects /
# tempo_hint / atmosphere_line) on top of the original four; the
# combined JSON object easily wants 200-300 tokens once the prose
# brief (300 chars) + atmosphere_line (200 chars) + five 3-6 term
# arrays + field-name overhead are summed. The 160-token cap was
# starving Mistral-Nemo into emitting an EMPTY string (line 1 col
# 0) -- not a truncated JSON, an immediate stop_token before any
# generation -- so all 3 ladder attempts failed with
# "no decodable top-level JSON object found". 512 gives generous
# headroom; the schema's hard caps still bound the actual prose.
_REFLECTION_MAX_NEW_TOKENS: int = 512

# Must-Fix 2 (r4 AG M2): Token budget override for visual_storybased reflection
# pass when generating both story brief reflection fields and visual_card object.
_DYNAMIC_REFLECTION_MAX_NEW_TOKENS: int = 1024

# Sprint 2A/2B: Attempt 2 structural-retry temperature for the shared
# structured_call retry ladder. STRICTLY BELOW _REFLECTION_TEMPERATURE
# -- a JSON-schema re-roll LOWERS entropy, it never raises it.
# structured_call asserts this invariant at entry. This replaces the
# old _REPAIR_TEMPERATURE_BUMP (which RAISED repair temperature by
# +0.15 -- the Sprint 2B bug the ladder conversion corrects).
_REFLECTION_STRUCTURAL_RETRY_TEMPERATURE: float = 0.15

# Refinement section 4 prompt-version field stamped on every output.
# Bump only when the prompt body changes in a way consumers must
# observe (e.g. a new sidecar term family).
#
# Sprint 8.1 (decision A1, flat additive): bumped v1 -> v2 when the
# reflection prompt body grew the five v2 fields consumed by the
# downstream brief readers: music_mood_terms, visual_palette,
# key_objects, tempo_hint, atmosphere_line. The v2 fields all carry
# safe defaults on the schema side so a v1 producer-era LLM response
# (missing the new keys) still validates -- the field is empty and
# the consumer falls through to its v1 path. The eight v1 meta keys
# stay untouched; v2 fields land alongside them as top-level meta
# entries, so v1 consumers and the existing 8-key meta-shape tests
# continue to pass.
_PROMPT_VERSION: str = "v2"

# Refinement section 4 source-tag stamped on every output -- documents
# the lifecycle origin of the brief (post-composition reflection, not
# pre-script outline). Helps Sprint G's later brief-sweep audit
# distinguish where each meta.story_brief came from.
_BRIEF_SOURCE: str = "post_script_reflection"


# Refinement section 2: input-builder caps. The reflection prompt sees
# a fixed-shape string regardless of episode length; these caps keep
# total input under ~1500 tokens on a 30-minute episode.
_OPENING_LINE_CAP: int = 18      # first N lines from the script
_CLOSING_LINE_CAP: int = 8       # last N lines from the script
_PER_LINE_TEXT_CAP: int = 240    # truncate each line text at this many chars


# Structural failure classes retained for the observable fail-soft receipt.
REJECT_JSON_PARSE: str = "json_parse_failed"
REJECT_SCHEMA: str = "schema_validation_failed"

# ---------------------------------------------------------------------------
# Pydantic schema -- required-field, type, and provider-size integrity only.
# Authored vocabulary and style never participate in acceptance.
# ---------------------------------------------------------------------------



class StoryBriefModel(BaseModel):
    """Schema for the reflection-pass LLM output.

    Shape (v1): one required non-empty visual brief plus three short
    sidecar term arrays. The model enforces type and provider-size bounds
    only; authored names, actions, eras, punctuation, and style are preserved.

    Sprint 8.1 (v2, decision A1 flat additive). The v2 fields all
    carry safe defaults so:
      * A v1 producer-era LLM response (missing the new keys) still
        validates -- the field is empty and the consumer falls
        through to its v1 path.
      * The MusicGenTheme rewire can read `music_mood_terms` first
        and gracefully fall through to the existing atmosphere-as-
        mood path on empty.
      * The audit-recommended six A-class consumers (FLUX env / FLUX
        portrait / FLUX radio bookend / LTX motion / HuMo lip-sync /
        OTR_VideoPlan era tail) each have a v2 field shaped to their
        prompt budget (see `downstream_brief_consumer_followup.md`
        for the consumer-by-consumer mapping).
    """

    story_brief:      str
    setting_terms:    list[str] = Field(default_factory=list)
    lighting_terms:   list[str] = Field(default_factory=list)
    atmosphere_terms: list[str] = Field(default_factory=list)
    # Sprint 8.1 v2 additive fields.
    music_mood_terms: list[str] = Field(default_factory=list)
    visual_palette:   list[str] = Field(default_factory=list)
    key_objects:      list[str] = Field(default_factory=list)
    tempo_hint:       str       = ""
    atmosphere_line:  str       = ""

    @field_validator("story_brief")
    @classmethod
    def _story_brief_must_not_be_blank(cls, value: str) -> str:
        """The required brief may use any vocabulary, but not be whitespace-only."""
        if not value.strip():
            raise ValueError("story_brief must not be blank")
        return value


try:
    from ._otr_visual_styles import VisualStyleCardModel
except ImportError:
    from _otr_visual_styles import VisualStyleCardModel


class StoryBriefVisualStorybasedModel(StoryBriefModel):
    """Extended schema for the dynamic visual_storybased reflection pass."""

    visual_card: VisualStyleCardModel


# ---------------------------------------------------------------------------
# Prompt bodies
# ---------------------------------------------------------------------------

# Per refinement section 3 + 3.3 + 3.4. Hard-capped at <250 tokens of
# prompt body (acceptance row implicit in commit gate). Strict JSON
# output, period-neutral framing, sidecar terms for the helpers.
#
# The reflection model receives exact ledger context. Model-facing craft guidance
# can shape the optional brief, but Python does not classify or suppress names,
# nouns, places, eras, or other authored world terms.
_REFLECTION_PROMPT: str = """\
Write a visual + audio brief for this audio drama. Return ONE JSON
object, no Markdown:

{
  "story_brief":      "a useful visual and audio clause",
  "setting_terms":    ["setting terms"],
  "lighting_terms":   ["lighting terms"],
  "atmosphere_terms": ["atmosphere terms"],
  "music_mood_terms": ["music mood terms"],
  "visual_palette":   ["color or texture terms"],
  "key_objects":      ["objects present in the episode"],
  "tempo_hint":       "a tempo description",
  "atmosphere_line":  "an atmosphere sentence"
}

RULES:
- ONE sentence per string field. No quotes, no Markdown.
- Preserve exact context names and world terms when they help describe the
  places, light, sound, and mood.
- No dialogue verbs (speaking, arguing) or plot verbs (interrogates,
  discovers).
- No invented dates, decades, centuries, cities, countries, or eras.
  If a period is implied, use atmosphere terms (smoke-filtered,
  incandescent glow) instead of dated ones (1940s, Victorian).
- music_mood_terms are for instrumental scoring (tense, sombre,
  hopeful). visual_palette is colors / textures. key_objects are
  concrete nouns the scene contains.
- setting_terms are PLACES (control room, riverbank), not adjectives.
  Plain words, never underscores.

Script context:
"""

_DYNAMIC_REFLECTION_PROMPT: str = """\
Write a visual + audio brief and a visual style card for this audio drama.
Return ONE JSON object, no Markdown:

{
  "story_brief":      "a useful visual and audio clause",
  "setting_terms":    ["setting terms"],
  "lighting_terms":   ["lighting terms"],
  "atmosphere_terms": ["atmosphere terms"],
  "music_mood_terms": ["music mood terms"],
  "visual_palette":   ["color or texture terms"],
  "key_objects":      ["objects present in the episode"],
  "tempo_hint":       "a tempo description",
  "atmosphere_line":  "an atmosphere sentence",
  "visual_card": {
    "medium_short":        "1-3 word art medium (e.g. oil painting, stained glass diorama)",
    "medium":              "descriptive art medium phrase",
    "radio_material":      "material and craftsmanship of radio console",
    "character_art_style": "color-neutral portrait rendering style",
    "texture":             "surface texture and weathering",
    "linework":            "line quality and edge definition",
    "lighting_character":  "atmospheric lighting quality",
    "typography_voice":    "letterform character for title cards",
    "motion_temperament":  "surface motion descriptor (max 60 chars)"
  }
}

RULES:
- ONE sentence per string field. No quotes, no Markdown.
- Preserve exact context names and world terms when they help describe the
  places, light, sound, and mood.
- No dialogue verbs or plot verbs.
- No invented dates, decades, centuries, cities, countries, or eras.
- music_mood_terms are for scoring. visual_palette is colors / textures. key_objects are concrete nouns.
- setting_terms are PLACES (control room, riverbank), not adjectives. Plain words, never underscores.
- visual_card fields must be concise, expressive aesthetic descriptions. medium_short must be 1-3 words (max 40 chars).

Script context:
"""


# ---------------------------------------------------------------------------
# Input builder -- refinement section 2
# ---------------------------------------------------------------------------


def _ledger_data(led: Any) -> dict:
    """Accept either a Ledger object (with .data attribute) or a raw dict."""
    return getattr(led, "data", led)


def _meta_title(ledger: dict) -> str:
    meta = ledger.get("meta") or {}
    title = (meta.get("episode_title") or "").strip()
    if title:
        return title
    title = (meta.get("title") or "").strip()
    if title:
        return title
    return ledger.get("title") or ""


def _truncate(text: str, cap: int) -> str:
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "..."


def _format_line(line: dict) -> str:
    """One-line summary of a ledger line for the reflection input."""
    role = (line.get("speaker_role") or "?").lower()
    char = (line.get("char_id") or line.get("speaker") or "").strip()
    text = (line.get("text") or "").strip()
    text = _truncate(text, _PER_LINE_TEXT_CAP)
    if role == "character" and char:
        return f"[{role}/{char}] {text}"
    return f"[{role}] {text}"


# ---------------------------------------------------------------------------
# Reflection input transport
# ---------------------------------------------------------------------------


def _build_reflection_input(led: Any) -> str:
    """Build bounded reflection context without vocabulary transformation.

    Title, cast names/descriptions, scene headers, spoken excerpts, and audio
    rows retain their authored wording. Only existing transport-size caps are
    applied so the optional reflection call cannot grow without bound.
    """
    ledger = _ledger_data(led)
    lines: Sequence[dict] = ledger.get("lines") or []
    cast: Sequence[dict] = ledger.get("cast") or []
    meta = ledger.get("meta") or {}

    parts: list[str] = [
        f"TITLE: {_meta_title(ledger)}",
        f"STYLE: {meta.get('style') or ''}",
        "CAST:",
    ]
    if not cast:
        parts.append("  (no cast)")
    for row in cast:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or "").strip()
        if not name:
            continue
        desc = _truncate(
            str(row.get("character_description") or "").strip(), 200,
        )
        parts.append(f"  - {name}: {desc}" if desc else f"  - {name}")

    scene_rows = [
        row for row in lines
        if str(row.get("speaker_role") or "").lower() == "scene"
    ]
    if scene_rows:
        parts.append("SCENE HEADERS:")
        for row in scene_rows:
            parts.append(
                f"  {_truncate(str(row.get('text') or '').strip(), 120)}"
            )

    if lines:
        opening = lines[:_OPENING_LINE_CAP]
        parts.append(f"OPENING ({len(opening)} lines):")
        parts.extend(f"  {_format_line(row)}" for row in opening)
        if len(lines) > _OPENING_LINE_CAP:
            closing = lines[
                max(_OPENING_LINE_CAP, len(lines) - _CLOSING_LINE_CAP):
            ]
            parts.append(f"CLOSING ({len(closing)} lines):")
            parts.extend(f"  {_format_line(row)}" for row in closing)

    non_dialogue = [
        row for row in lines
        if str(row.get("speaker_role") or "").lower() in {"music", "env"}
    ]
    if non_dialogue:
        parts.append(f"NON-DIALOGUE ROWS ({len(non_dialogue)}):")
        parts.extend(
            f"  {_format_line(row)}" for row in non_dialogue[:30]
        )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------
# The naive first-'{'-to-last-'}' extractor was removed in the
# BUG-LOCAL-261 consolidation; story-brief JSON is now parsed via the
# shared _otr_json.parse_first_json_object. Package import in production;
# flat import when loaded standalone / under test.
try:
    from . import _otr_json
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_json  # type: ignore

# Sprint 2A/2D: the shared structured-JSON retry ladder. The reflection
# pass routes its single LLM call through it -- the ladder subsumes the
# former hand-rolled call -> parse -> validate -> repair sequence.
# Package import in production; flat import under standalone test.
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

# Sprint 2C: typed repair-prompt factories. run_story_brief_reflection
# uses the dispatching factory only for JSON/schema repair. Authored content
# never enters the repair ladder. Package import in production; flat import
# under standalone test.
try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore

# The safe-open viability predicate is defined ONCE, in the composer, and
# imported here -- not the other way round. _otr_line_composer keeps a
# stdlib-only module-load import surface by design (see its LineRequest note);
# this module imports pydantic above, so the dependency has to run heavy ->
# light. Sharing the predicate is the point: a brief this validator accepts
# must be one the composer can actually open on, and the gap between those two
# judgements is what shipped episodes that asked the operator for the setting.
try:
    from ._otr_line_composer import safe_open_viability
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_line_composer import safe_open_viability  # type: ignore


# ---------------------------------------------------------------------------
# Failure sentinel -- refinement section 4.1 + L-6
# ---------------------------------------------------------------------------


def _failure_sentinel(
    *,
    reason: str,
    technical_model_id: str,
    prompt_version: str,
) -> dict:
    """Return the meta delta for a failed reflection pass.

    `story_brief == ""` lets consumers fall through to legacy behavior.
    `story_brief_status == "failed"` makes the failure observable per
    refinement section 4.1; consumers (FLUX env, LTX, HuMo, MusicGen)
    log the status in their render output per E-07.

    Sprint 8.1 (decision A1, flat additive): the v2 fields are stamped
    with safe-empty defaults on failure so downstream readers can call
    `_otr_brief_reader._read_brief_field` unconditionally and fall
    through to their v1 paths on falsiness. v1 keys are unchanged.
    """
    return {
        "story_brief":               "",
        "story_brief_status":        "failed",
        "story_brief_error":         reason,
        "story_brief_model":         technical_model_id,
        "story_brief_prompt_version": prompt_version,
        "story_brief_source":        _BRIEF_SOURCE,
        "story_brief_char_count":    0,
        "story_brief_terms": {
            "setting":    [],
            "lighting":   [],
            "atmosphere": [],
        },
        # Sprint 8.1 v2 fields -- safe-empty so reader returns the
        # caller-supplied default on every consumer path.
        "music_mood_terms":          [],
        "visual_palette":            [],
        "key_objects":               [],
        "tempo_hint":                "",
        "atmosphere_line":           "",
    }


def _success_delta(
    *,
    brief_model: StoryBriefModel,
    technical_model_id: str,
    prompt_version: str,
) -> dict:
    """Return the meta delta for a successful reflection pass.

    Sprint 8.1 (decision A1, flat additive): the 8 v1 meta keys land
    unchanged. The 5 v2 fields are stamped as top-level meta entries
    alongside them. v1 consumers that read `story_brief` /
    `story_brief_terms` continue to work; v2 consumers (MusicGenTheme
    first; the audit-recommended six A-class consumers in sprints
    8.2-8.7) read through `_otr_brief_reader._read_brief_field`.
    """
    text = brief_model.story_brief.strip()
    tempo_hint = (
        brief_model.tempo_hint.strip()
        if isinstance(brief_model.tempo_hint, str)
        else ""
    )
    atmosphere_line = (
        brief_model.atmosphere_line.strip()
        if isinstance(brief_model.atmosphere_line, str)
        else ""
    )
    return {
        "story_brief":               text,
        "story_brief_status":        "ok",
        "story_brief_error":         None,
        "story_brief_model":         technical_model_id,
        "story_brief_prompt_version": prompt_version,
        "story_brief_source":        _BRIEF_SOURCE,
        "story_brief_char_count":    len(text),
        "story_brief_terms": {
            "setting":    list(brief_model.setting_terms),
            "lighting":   list(brief_model.lighting_terms),
            "atmosphere": list(brief_model.atmosphere_terms),
        },
        # Sprint 8.1 v2 fields -- top-level (A1 flat additive).
        "music_mood_terms":          list(brief_model.music_mood_terms),
        "visual_palette":            list(brief_model.visual_palette),
        "key_objects":               list(brief_model.key_objects),
        "tempo_hint":                tempo_hint,
        "atmosphere_line":           atmosphere_line,
    }


# ---------------------------------------------------------------------------
# Main entrypoint -- structured_call ladder + L-8 + E-21
# ---------------------------------------------------------------------------


def run_story_brief_reflection(
    led: Any,
    technical_fn: Callable[..., str],
    *,
    technical_model_id: str = "",
    prompt_version: str = _PROMPT_VERSION,
    is_visual_storybased: bool = False,
) -> dict:
    """Run the reflection pass and return the 8-key meta delta (plus visual_card if dynamic).

    LLM slot: technical -- structured JSON output, NOT narrative
    composition. Per L-2 the writer's technical slot is the right
    home for this call site; per E-21 the signature accepts ONLY
    `technical_fn` (no `creative_fn` parameter) so a future refactor
    cannot accidentally route the call through the creative slot
    and burn the creative budget on a JSON validation pass.

    Sprint 2A/2D: the call routes through the shared `structured_call`
    3-attempt retry ladder (base -> structural retry -> typed repair).
    The ladder handles JSON syntax and schema/required-field integrity only.
    Schema-valid authored content is accepted on the first call regardless
    of vocabulary, names, era, punctuation, or style. `structured_call` does
    not catch slot-fn (technical_fn) exceptions; the broad exception boundary
    below converts those infrastructure failures to the fail-soft sentinel.

    An exhausted ladder (`StructuredCallFailedError`) and a raising
    slot fn both map to `_failure_sentinel`. On every path the
    function returns a dict; it never raises.
    """
    if is_visual_storybased:
        prompt_body = _DYNAMIC_REFLECTION_PROMPT
        target_schema = StoryBriefVisualStorybasedModel
        max_tokens = _DYNAMIC_REFLECTION_MAX_NEW_TOKENS
    else:
        prompt_body = _REFLECTION_PROMPT
        target_schema = StoryBriefModel
        max_tokens = _REFLECTION_MAX_NEW_TOKENS

    user_message = prompt_body + _build_reflection_input(led)
    messages = [{"role": "user", "content": user_message}]

    # Attempt counter for the SUCCESS path. The failure path already reported
    # honestly via `exc.attempts` below, but a success reported nothing -- so
    # the writer's `_brief_delta.get("story_brief_attempts", 1)` fell through
    # to its default and `visual_style_receipt["attempts"]` read 1 forever,
    # even when the ladder had retried twice before succeeding. A receipt that
    # can only ever print one value is not a measurement.
    # `on_attempt_complete` fires once per ladder attempt (structured_call
    # :1010-1013); the same hook the structured lane passes already use.
    _attempts_seen = 0

    def _mark_attempt(attempts_run, _last_raw, _error):
        nonlocal _attempts_seen
        try:
            _attempts_seen = int(attempts_run)
        except (TypeError, ValueError):  # never let telemetry break the pass
            pass

    # LLM slot: technical -- structured JSON validation pass, not
    # narrative composition; routed through the shared ladder.
    try:
        brief_model = structured_call(
            prompt=messages,
            schema=target_schema,
            slot_fn=technical_fn,
            base_temperature=_REFLECTION_TEMPERATURE,
            structural_retry_temperature=_REFLECTION_STRUCTURAL_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=max_tokens,
            max_attempts=3,
            helper_name="run_story_brief_reflection",
            on_attempt_complete=_mark_attempt,
        )
    except StructuredCallFailedError as exc:
        # Ladder exhausted. Attribute the failure to its class via the
        # last error the ladder captured so the sentinel `reason` stays
        # as forensically specific as the pre-conversion arms did.
        last = exc.last_error
        if isinstance(last, json.JSONDecodeError):
            reason = REJECT_JSON_PARSE
        elif isinstance(last, ValidationError):
            reason = REJECT_SCHEMA
        else:
            reason = "structured_call_failed"
        log.warning(
            "[OTR_StoryBrief] structured_call exhausted the retry ladder "
            "after %d attempt(s) (last error: %s); returning failed-status "
            "sentinel (reason=%s)", exc.attempts, exc.last_error, reason,
        )
        delta = _failure_sentinel(
            reason=reason,
            technical_model_id=technical_model_id,
            prompt_version=prompt_version,
        )
        delta["story_brief_attempts"] = exc.attempts
        return delta
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        # structured_call does not catch slot-fn exceptions: a network /
        # VRAM / framework failure inside the technical_fn closure lands
        # here. A bare reraise would crash the whole script generation
        # over a flavor-text failure -- the L-6 reason for catching broad.
        log.warning(
            "[OTR_StoryBrief] technical_fn raised %s: %s; returning "
            "failed-status sentinel", type(exc).__name__, exc,
        )
        _slot_delta = _failure_sentinel(
            reason="technical_fn_exception",
            technical_model_id=technical_model_id,
            prompt_version=prompt_version,
        )
        # Report the count here too. structured_call fires the hook before
        # re-raising on every rung, so by the time a slot-fn exception reaches
        # this branch `_attempts_seen` already holds the true number. Dropping
        # it would leave the consumer's `.get("story_brief_attempts", 1)`
        # defaulting to 1 on exactly the sibling branch this change exists to
        # fix -- the same "receipt that can only print one value" defect, moved
        # one except-clause over.
        _slot_delta["story_brief_attempts"] = _attempts_seen or 1
        return _slot_delta

    delta = _success_delta(
        brief_model=brief_model,
        technical_model_id=technical_model_id,
        prompt_version=prompt_version,
    )
    # Report what the ladder ACTUALLY cost. Falls back to 1 only if the hook
    # never fired, which would mean a single clean attempt anyway -- so the
    # value is honest in every branch rather than merely plausible.
    delta["story_brief_attempts"] = _attempts_seen or 1
    if is_visual_storybased and hasattr(brief_model, "visual_card"):
        delta["visual_card"] = brief_model.visual_card
    return delta


# ---------------------------------------------------------------------------
# Produced-story summary (meta-split chunk 1, operator directive 2026-07-09)
#
# Operator ruling: `meta` was ALWAYS meant to carry a brief of the ACTUAL
# produced story for downstream consumer prompts; the pre-generation
# interpreter digest (`meta["news"]`) got conflated with it. The reflection
# pass above is deliberately a visual/audio MOOD board rather than a factual
# episode summary, so the FACTUAL consumers -- the
# video HUD story spine, the story-treatment text, the credits premise --
# kept printing the SOURCE digest as if it described the episode.
#
# This is the missing half: one more post-composition LLM pass that
# summarizes what the produced episode is actually about -- real names
# allowed, plot required, ending stated -- stamped under the quiet,
# distinct key `meta["produced_story"]` (never inside `meta["news"]`;
# per the operator, the two concepts never share a name again).
# ---------------------------------------------------------------------------

# The structured_call ladder requires the structural retry to be STRICTLY
# COOLER than base (raising entropy during JSON-schema repair encourages
# structural hallucination).
_PRODUCED_STORY_TEMPERATURE: float = 0.3
_PRODUCED_STORY_STRUCTURAL_RETRY_TEMPERATURE: float = 0.15
_PRODUCED_STORY_MAX_NEW_TOKENS: int = 400
_PRODUCED_STORY_SOURCE: str = "llm_post_composition"

# Excerpt caps for the summary input -- opening / middle / closing windows
# of the SPOKEN lines (character + announcer), real names intact.
_PRODUCED_STORY_OPENING_CAP: int = 8
_PRODUCED_STORY_MIDDLE_CAP: int = 6
_PRODUCED_STORY_CLOSING_CAP: int = 8

class ProducedStoryModel(BaseModel):
    """Schema for the produced-story summary LLM output.

    Any nonblank provider-bounded strings are accepted. The prompt can request
    a useful summary, but Python does not judge its names, vocabulary, style,
    punctuation, or narrative quality.
    """

    logline: str
    subject: str

    @field_validator("logline", "subject")
    @classmethod
    def _text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("produced-story text must not be blank")
        return value


_PRODUCED_STORY_PROMPT: str = """\
Summarize this finished radio-drama episode. Return ONE JSON object,
no Markdown:

{
  "logline": "who the story was about, what happened, and how it ended",
  "subject": "a noun phrase naming the story's subject"
}

RULES:
- Summarize the EPISODE text below, not any outside source or news item.
- Use the character names exactly as they appear.
- State what actually happened; do not tease or withhold the ending.
- No quotation marks, brackets, or markup in either field.

EPISODE:
"""


def _build_produced_story_input(led: Any) -> str:
    """Build the summary-prompt input: title + REAL cast names + spoken-line
    excerpts (opening / middle / closing windows).

    The inverse of `_build_reflection_input`: no name scrubbing -- the whole
    point of this pass is a summary grounded in the episode's actual names
    and events. Accepts a Ledger object OR a raw dict (duck-typed via
    `_ledger_data`), same as the reflection builder.
    """
    ledger = _ledger_data(led)
    lines: Sequence[dict] = ledger.get("lines") or []
    cast: Sequence[dict] = ledger.get("cast") or []

    parts: list[str] = [f"TITLE: {_meta_title(ledger)}"]
    names = [
        (row.get("name") or "").strip()
        for row in cast
        if isinstance(row, dict) and (row.get("name") or "").strip()
    ]
    parts.append("CAST: " + (", ".join(names) if names else "(no cast)"))

    spoken = [
        ln for ln in lines
        if (ln.get("speaker_role") or "").lower() in ("character", "announcer")
        and (ln.get("text") or "").strip()
    ]
    n = len(spoken)
    opening = spoken[:_PRODUCED_STORY_OPENING_CAP]
    closing: list = []
    middle: list = []
    # Local-fanout QA 2026-07-09: any spoken rows beyond the opening window
    # get a closing window (sliced to avoid overlap) -- the old strict
    # `n > OPEN + CLOSE` dropped the whole second half at exact equality.
    if n > _PRODUCED_STORY_OPENING_CAP:
        closing = spoken[
            max(_PRODUCED_STORY_OPENING_CAP, n - _PRODUCED_STORY_CLOSING_CAP):
        ]
        mid_start = max(
            _PRODUCED_STORY_OPENING_CAP,
            n // 2 - _PRODUCED_STORY_MIDDLE_CAP // 2,
        )
        mid_rows = spoken[mid_start:mid_start + _PRODUCED_STORY_MIDDLE_CAP]
        # Middle only earns a window when it adds rows beyond the other two.
        if n > (_PRODUCED_STORY_OPENING_CAP + _PRODUCED_STORY_CLOSING_CAP
                + _PRODUCED_STORY_MIDDLE_CAP):
            middle = mid_rows
    if opening:
        parts.append(f"OPENING ({len(opening)} lines):")
        parts.extend(f"  {_format_line(ln)}" for ln in opening)
    if middle:
        parts.append(f"MIDDLE ({len(middle)} lines):")
        parts.extend(f"  {_format_line(ln)}" for ln in middle)
    if closing:
        parts.append(f"CLOSING ({len(closing)} lines):")
        parts.extend(f"  {_format_line(ln)}" for ln in closing)
    return "\n".join(parts)


def _produced_story_failure(reason: str, technical_model_id: str) -> dict:
    """Sentinel delta for a failed summary pass -- status stamped LOUD,
    `produced_story` itself absent so no consumer prints a broken value."""
    return {
        "produced_story_status": f"failed:{reason}",
        "produced_story_model_id": technical_model_id,
    }


def run_produced_story_summary(
    led: Any,
    technical_fn: Callable[..., str],
    *,
    technical_model_id: str = "",
) -> dict:
    """Run the produced-story summary pass and return the meta delta.

    LLM slot: technical -- structured JSON output, not narrative
    composition (same E-21 posture as `run_story_brief_reflection`:
    the signature accepts ONLY `technical_fn`).

    Success delta:
      produced_story           {"logline": ..., "subject": ...}
      produced_story_status    "ok"
      produced_story_model_id  the technical slot's model id
      produced_story_source    "llm_post_composition"

    Failure (exhausted ladder / raising slot fn) returns the sentinel
    delta with `produced_story` ABSENT and a `failed:<reason>` status.
    On every path the function returns a dict; it never raises -- a
    post-hoc summary must never kill a finished episode.
    """
    user_message = _PRODUCED_STORY_PROMPT + _build_produced_story_input(led)
    messages = [{"role": "user", "content": user_message}]

    # LLM slot: technical -- structured JSON summary pass.
    try:
        model = structured_call(
            prompt=messages,
            schema=ProducedStoryModel,
            slot_fn=technical_fn,
            base_temperature=_PRODUCED_STORY_TEMPERATURE,
            structural_retry_temperature=(
                _PRODUCED_STORY_STRUCTURAL_RETRY_TEMPERATURE
            ),
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=_PRODUCED_STORY_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name="run_produced_story_summary",
        )
    except StructuredCallFailedError as exc:
        log.warning(
            "[OTR_ProducedStory] structured_call exhausted the retry ladder "
            "after %d attempt(s) (last error: %s); stamping failed status",
            exc.attempts, exc.last_error,
        )
        return _produced_story_failure(
            "structured_call_failed", technical_model_id
        )
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        log.warning(
            "[OTR_ProducedStory] technical_fn raised %s: %s; stamping "
            "failed status", type(exc).__name__, exc,
        )
        return _produced_story_failure(
            "technical_fn_exception", technical_model_id
        )

    return {
        "produced_story": {
            "logline": model.logline.strip(),
            "subject": model.subject.strip(),
        },
        "produced_story_status": "ok",
        "produced_story_model_id": technical_model_id,
        "produced_story_source": _PRODUCED_STORY_SOURCE,
    }


# ---------------------------------------------------------------------------
# Produced OPEN brief -- post-composition intro-rewrite derive pass
# (INTRO_REWRITE_SPEC 2026-07-09, kibitz r2-r4 shape A).
#
# Same meta-split family as the produced-story summary above: the in-loop
# intro was starved to a PRE-GEN SafeOpenBrief (outline-derived, writer
# ~3438); this pass derives the same field shape from the PRODUCED scene-1
# rows + cast roster so the writer can recompose the intro through the
# existing routed safe-open composer. Spoiler safety is STRUCTURAL: the
# input builder slices scene 1 only -- the pass cannot reveal what it
# never saw.
#
# Failure posture differs from run_produced_story_summary BY DESIGN
# (kibitz r4 P1): this function RAISES on ladder exhaustion / empty input;
# the WRITER's rewrite block catches and keeps the in-loop intro. The
# never-raise boundary lives at the caller, where the keep-intro decision
# is made.
# ---------------------------------------------------------------------------

# Mirrors the reflection/produced-story JSON-compliance settings (3.2).
_PRODUCED_OPEN_TEMPERATURE: float = 0.3
_PRODUCED_OPEN_STRUCTURAL_RETRY_TEMPERATURE: float = 0.2
_PRODUCED_OPEN_MAX_NEW_TOKENS: int = 320
# Scene-1 excerpt budget (chars) -- keep the technical call tight
# (kibitz r2 codex S2: derive runs on the technical slot between two
# creative-slot passes; a small prompt keeps the swap cheap).
_PRODUCED_OPEN_EXCERPT_CAP: int = 1200


class ProducedOpenBriefModel(BaseModel):
    """Derived no-spoiler OPEN brief -- SafeOpenBrief field parity.

    Field-for-field the shape `_otr_line_composer.SafeOpenBrief` accepts
    (minus `era`, which the WRITER threads from meta["period"] for parity
    with the in-loop construction). Schema stays loose; the content gate
    is `_validate_produced_open` (post_validator), which enforces
    non-emptiness and roster membership."""
    setting: str = Field(default="")
    time_of_day: str = Field(default="")
    opening_status_quo: str = Field(default="")
    cast: list[str] = Field(default_factory=list)


_PRODUCED_OPEN_PROMPT = """\
You are preparing the radio announcer's opening for a finished episode.
Below are the CAST ROSTER and the spoken lines of SCENE 1 ONLY. From them,
extract a no-spoiler opening brief. Return ONE JSON object, keys exactly:
  "setting"            -- where scene 1 takes place (concrete place words)
  "time_of_day"        -- when, if the lines say or imply it, else ""
  "opening_status_quo" -- one sentence: where things stand as scene 1 opens
  "cast"               -- the roster names that actually appear in scene 1
Rules: use ONLY information visible in the scene-1 lines; never invent
names not on the roster; never hint at how the story might end; plain
prose values, no markup. JSON only -- no commentary.

"""


def _build_produced_open_input(
    led: Any,
    first_announcer_line_id: str,
) -> "tuple[str, list[str]]":
    """Scene-1 input slice + cast roster for the derive pass.

    Scene-1 rule (matches the writer's scene convention: music-marker rows
    reset the conversational window, writer ~4921): skip the in-loop intro
    row itself, skip any leading non-spoken rows (e.g. the intro->scene-1
    music marker), collect spoken rows (character/announcer, non-empty
    text) until the NEXT non-spoken row, capped at
    `_PRODUCED_OPEN_EXCERPT_CAP` chars. Raises ValueError when no scene-1
    spoken row exists (caller keeps the in-loop intro)."""
    ledger = _ledger_data(led)
    lines: Sequence[dict] = ledger.get("lines") or []
    cast: Sequence[dict] = ledger.get("cast") or []

    roster = [
        (row.get("name") or "").strip()
        for row in cast
        if isinstance(row, dict) and (row.get("name") or "").strip()
    ]

    collected: list[dict] = []
    started = False
    used = 0
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ln.get("line_id") == first_announcer_line_id:
            # The in-loop intro row: excluded from its own rewrite input,
            # and it does NOT start the scene (a marker may follow it).
            continue
        role = str(ln.get("speaker_role") or "").lower()
        text = str(ln.get("text") or "").strip()
        spoken = role in ("character", "announcer") and bool(text)
        if not spoken:
            if started:
                break  # scene break: first non-spoken row after content
            continue   # leading marker/empty row before scene 1
        started = True
        collected.append(ln)
        used += len(text)
        if used >= _PRODUCED_OPEN_EXCERPT_CAP:
            break
    if not collected:
        raise ValueError(
            "produced-open derive: no scene-1 spoken rows in the ledger"
        )

    parts: list[str] = []
    parts.append("CAST ROSTER: " + (", ".join(roster) if roster else "(none)"))
    parts.append(f"SCENE 1 ({len(collected)} spoken lines):")
    parts.extend(f"  {_format_line(ln)}" for ln in collected)
    return "\n".join(parts), roster


def _validate_produced_open(
    model: ProducedOpenBriefModel,
    roster: "list[str]",
) -> "str | None":
    """Content gate (structured_call post_validator contract).

    Rejects: a brief the announcer cannot open on, and any cast name not on the
    roster (case-insensitive) -- the derive pass must never introduce a name the
    episode does not have.

    Viability is the COMPOSER's predicate, imported rather than restated. This
    used to check only that setting and opening_status_quo were not both empty,
    which let a brief through with NO CAST -- and every bank's safe-open seam
    promises the model "the cast list below", so those briefs produced an
    announcer asking the operator for the roster. Two definitions of "usable
    brief" is what the shared predicate exists to end.
    """
    starved = safe_open_viability(
        setting=model.setting,
        opening_status_quo=model.opening_status_quo,
        cast=model.cast,
    )
    if starved:
        return starved
    lowered = {n.lower(): n for n in roster}
    for name in model.cast:
        if str(name).strip().lower() not in lowered:
            return f"cast name {name!r} is not on the roster"
    return None


def derive_produced_open_brief(
    led: Any,
    *,
    first_announcer_line_id: str,
    technical_fn: Callable[..., str],
    technical_model_id: str = "",
) -> ProducedOpenBriefModel:
    """Run the produced-open derive pass; return the validated brief.

    LLM slot: technical -- structured JSON extraction, not narrative
    composition (E-21 posture; same slot family as the produced-story
    summary). RAISES on failure (ValueError for an empty scene slice,
    StructuredCallFailedError on ladder exhaustion): the writer's rewrite
    block owns the keep-intro decision (kibitz r4 P1). Cast names in the
    returned model are normalized to roster casing and de-duplicated,
    order preserved."""
    input_text, roster = _build_produced_open_input(
        led, first_announcer_line_id,
    )
    messages = [
        {"role": "user", "content": _PRODUCED_OPEN_PROMPT + input_text},
    ]

    def _content_validator(model: ProducedOpenBriefModel) -> "str | None":
        return _validate_produced_open(model, roster)

    # LLM slot: technical -- structured JSON derive pass.
    model = structured_call(
        prompt=messages,
        schema=ProducedOpenBriefModel,
        slot_fn=technical_fn,
        base_temperature=_PRODUCED_OPEN_TEMPERATURE,
        structural_retry_temperature=(
            _PRODUCED_OPEN_STRUCTURAL_RETRY_TEMPERATURE
        ),
        repair_prompt_factory=make_dispatching_repair_factory(),
        post_validator=_content_validator,
        max_new_tokens=_PRODUCED_OPEN_MAX_NEW_TOKENS,
        max_attempts=3,
        helper_name="derive_produced_open_brief",
    )
    log.info(
        "[OTR_ProducedOpen] derive ok (model=%s, cast=%d, setting=%r)",
        technical_model_id, len(model.cast), model.setting[:60],
    )
    lowered = {n.lower(): n for n in roster}
    normalized: list[str] = []
    for name in model.cast:
        cased = lowered.get(str(name).strip().lower())
        if cased and cased not in normalized:
            normalized.append(cased)
    return model.model_copy(update={"cast": normalized})
