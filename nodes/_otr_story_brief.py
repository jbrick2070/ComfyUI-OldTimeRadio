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
- `SPRINT.md` Sprint C: L-6 fail-loud pattern, L-8 8-key meta delta,
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

from pydantic import BaseModel, Field, ValidationError


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Constants -- LLM settings, length caps, prompt version
# ---------------------------------------------------------------------------

# Per refinement section 3.2: temperature 0.2-0.4 keeps the model
# compliance-honest for JSON output. 0.3 is the centerpoint.
_REFLECTION_TEMPERATURE: float = 0.3

# Per refinement section 3.2: 120 tokens covers the JSON object plus
# the 300-char prose brief without leaving room for chatty preamble.
_REFLECTION_MAX_NEW_TOKENS: int = 160

# Per E-18 / RR-B5: repair clamp upper bound. Even if a future operator
# sets _REFLECTION_TEMPERATURE above 0.4, the repair pass stays inside
# the 0.35-0.55 declared-safe range.
_REPAIR_TEMPERATURE_CEILING: float = 0.55

# Per refinement section 3.2: +0.15 jump on repair breaks the
# deterministic-retry failure loop characteristic of low-temperature
# local-model JSON output (R-06).
_REPAIR_TEMPERATURE_BUMP: float = 0.15

# Per refinement section 7 hard caps: max 300 chars on the prose brief.
_BRIEF_HARD_MAX_CHARS: int = 300

# Refinement section 4 prompt-version field stamped on every output.
# Bump only when the prompt body changes in a way consumers must
# observe (e.g. a new sidecar term family).
_PROMPT_VERSION: str = "v1"

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


# Refinement section 3.4 validation rejection-class strings. The
# central list is exposed for callers / tests that want to assert
# specific rejection classes without depending on log strings.
REJECT_NAMED_CHARACTER:    str = "named_character"
REJECT_DIALOGUE_VERB:      str = "dialogue_verb"
REJECT_PLOT_VERB:          str = "plot_verb"
REJECT_UNSUPPORTED_PERIOD: str = "unsupported_period"
REJECT_TOO_LONG:           str = "too_long"
REJECT_MULTI_SENTENCE:     str = "multi_sentence"
REJECT_QUOTES_OR_MARKUP:   str = "quotes_or_markup"
REJECT_JSON_PARSE:         str = "json_parse_failed"
REJECT_SCHEMA:             str = "schema_validation_failed"


# Refinement section 3.4 verb-class lists. Kept short / specific --
# the validator is allowed to over-reject; the repair pass and the
# empty-string fallback both handle a too-strict reject cleanly.
_DIALOGUE_VERBS: frozenset[str] = frozenset({
    "speaking", "speaks", "spoke", "speak",
    "saying", "says", "said", "say",
    "arguing", "argues", "argued", "argue",
    "watching", "watches", "watched", "watch",
    "whispering", "whispers", "whispered", "whisper",
    "shouting", "shouts", "shouted", "shout",
    "asking", "asks", "asked", "ask",
    "telling", "tells", "told", "tell",
    "yelling", "yells", "yelled", "yell",
    "muttering", "mutters", "muttered", "mutter",
})
_PLOT_VERBS: frozenset[str] = frozenset({
    "interrogates", "interrogate", "interrogated", "interrogating",
    "escapes", "escape", "escaped", "escaping",
    "discovers", "discover", "discovered", "discovering",
    "investigates", "investigate", "investigated", "investigating",
    "attacks", "attack", "attacked", "attacking",
    "rescues", "rescue", "rescued", "rescuing",
    "kills", "kill", "killed", "killing",
    "saves", "save", "saved", "saving",
    "betrays", "betray", "betrayed", "betraying",
    "confesses", "confess", "confessed", "confessing",
})

# Refinement section 3.3 period rejection: dated terms that imply a
# specific era. The validator catches these only when they are NOT
# already present in the source ledger (e.g. a script explicitly set
# in 1947 can carry "1947" through the brief). Detection is regex-
# based so the test fixture cap is upper-bounded.
_PERIOD_REGEX: re.Pattern[str] = re.compile(
    r"\b("
    r"\d{4}s?"                                       # 1947, 1940s
    r"|\d{2}th\s+century"                            # 20th century
    r"|victorian|edwardian|georgian|elizabethan"     # named eras
    r"|art\s*deco|art\s*nouveau"
    r")\b",
    re.IGNORECASE,
)

# Quote / markup characters disallowed in the prose brief per
# refinement section 3.4. Smart quotes included.
_QUOTE_OR_MARKUP_REGEX: re.Pattern[str] = re.compile(
    r'["“”‘’`#*_]',
)


# ---------------------------------------------------------------------------
# Pydantic schema -- shape validation only (content rules live in
# _validate_brief). Lengths are enforced strictly so a runaway LLM
# response cannot smuggle a 5000-char brief past the gate.
# ---------------------------------------------------------------------------


class StoryBriefModel(BaseModel):
    """Schema for the reflection-pass LLM output.

    Shape: a single visual brief sentence plus three short sidecar
    term arrays. The model enforces type + length only; content rules
    (named characters, dialogue verbs, period literals) live in the
    Python `_validate_brief` post-gate so the rejection-class
    messages stay readable rather than leaking pydantic internals
    into the repair-prompt body.
    """

    story_brief:      str       = Field(min_length=10, max_length=_BRIEF_HARD_MAX_CHARS)
    setting_terms:    list[str] = Field(default_factory=list, max_length=10)
    lighting_terms:   list[str] = Field(default_factory=list, max_length=10)
    atmosphere_terms: list[str] = Field(default_factory=list, max_length=10)


# ---------------------------------------------------------------------------
# Prompt body
# ---------------------------------------------------------------------------

# Per refinement section 3 + 3.3 + 3.4. Hard-capped at <250 tokens of
# prompt body (acceptance row implicit in commit gate). Strict JSON
# output, period-neutral framing, sidecar terms for the helpers.
_REFLECTION_PROMPT: str = """\
Write a one-sentence visual brief for this audio drama. Return ONE JSON
object, no Markdown:

{
  "story_brief":      "one clause under 300 chars",
  "setting_terms":    ["3-6 short setting nouns"],
  "lighting_terms":   ["3-6 short lighting nouns"],
  "atmosphere_terms": ["3-6 short atmosphere nouns"]
}

RULES:
- ONE sentence, under 300 chars. No quotes. No Markdown.
- No cast names. No proper nouns.
- No dialogue verbs (speaking, arguing) or plot verbs (interrogates,
  discovers).
- No invented dates / decades / centuries / cities / countries / eras.
  If a period is implied, use atmosphere terms (smoke-filtered,
  incandescent glow) instead of dated ones (1940s, Victorian).
- Each term under 24 chars.

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


def _build_reflection_input(led: Any) -> str:
    """Build the capped reflection-prompt input from the ledger.

    Per refinement section 2: title + style slug + cast roster + scene
    headers + opening / closing snippets + non-dialogue rows. Total
    output stays well under 1500 tokens on a typical episode.

    Accepts a Ledger object OR a raw dict (`led.data`) -- duck-typed
    via `_ledger_data` so tests can pass a plain dict without
    constructing a Ledger.
    """
    ledger = _ledger_data(led)
    lines: Sequence[dict] = ledger.get("lines") or []
    cast: Sequence[dict] = ledger.get("cast") or []
    meta = ledger.get("meta") or {}

    parts: list[str] = []

    title = _meta_title(ledger)
    parts.append(f"TITLE: {title}")
    parts.append(f"STYLE: {meta.get('style') or ''}")

    parts.append("CAST:")
    if not cast:
        parts.append("  (no cast)")
    for row in cast:
        if not isinstance(row, dict):
            continue
        name = (row.get("name") or "").strip()
        if not name:
            continue
        desc = _truncate((row.get("character_description") or "").strip(), 200)
        parts.append(f"  - {name}: {desc}" if desc else f"  - {name}")

    # Scene headers (refinement section 2 says any line with
    # speaker_role == "scene" or section markers).
    scene_rows = [ln for ln in lines if (ln.get("speaker_role") or "") == "scene"]
    if scene_rows:
        parts.append("SCENE HEADERS:")
        for row in scene_rows:
            parts.append(f"  {_truncate((row.get('text') or '').strip(), 120)}")

    # Opening + closing snippets.
    if lines:
        opening = lines[:_OPENING_LINE_CAP]
        parts.append(f"OPENING ({len(opening)} lines):")
        for ln in opening:
            parts.append(f"  {_format_line(ln)}")
        if len(lines) > _OPENING_LINE_CAP + _CLOSING_LINE_CAP:
            closing = lines[-_CLOSING_LINE_CAP:]
            parts.append(f"CLOSING ({len(closing)} lines):")
            for ln in closing:
                parts.append(f"  {_format_line(ln)}")

    # Non-dialogue rows (SFX / MUSIC / ENV / SCENE markers) -- these
    # carry the sonic-architecture signal a visual brief should reflect.
    non_dialogue_roles = {"sfx", "music", "env"}
    non_dialogue = [
        ln for ln in lines
        if (ln.get("speaker_role") or "").lower() in non_dialogue_roles
    ]
    if non_dialogue:
        parts.append(f"NON-DIALOGUE ROWS ({len(non_dialogue)}):")
        for ln in non_dialogue[:30]:  # cap at 30 to bound the prompt
            parts.append(f"  {_format_line(ln)}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Validation -- refinement section 3.4 + 3.3
# ---------------------------------------------------------------------------


def _cast_name_tokens(ledger: dict) -> set[str]:
    """Return lowercase tokens for every cast name (full + parts)."""
    tokens: set[str] = set()
    for row in ledger.get("cast") or []:
        if not isinstance(row, dict):
            continue
        name = (row.get("name") or "").strip()
        if not name:
            continue
        full = name.lower()
        tokens.add(full)
        for part in re.split(r"\s+|[-_]", full):
            if len(part) >= 3:
                tokens.add(part)
    return tokens


def _existing_period_tokens(ledger: dict) -> set[str]:
    """Period strings already present in the source ledger.

    A brief is allowed to mention "1947" if "1947" appears in the
    script text (the writer was already given an explicit period).
    Detection scans cast descriptions + line text.
    """
    found: set[str] = set()
    sources: list[str] = []
    for row in ledger.get("cast") or []:
        if isinstance(row, dict):
            sources.append((row.get("character_description") or ""))
    for ln in ledger.get("lines") or []:
        if isinstance(ln, dict):
            sources.append((ln.get("text") or ""))
    for src in sources:
        for m in _PERIOD_REGEX.finditer(src):
            found.add(m.group(0).lower())
    return found


def _validate_brief(brief: str, ledger: dict) -> list[str]:
    """Return a list of rejection reason codes; empty means accept.

    Per refinement section 3.4 + 3.3. Each rejection appends exactly
    one code from the REJECT_* constants. Multiple rejection classes
    can fire on the same brief -- the repair pass sees the full list
    in its CRITICAL prefix.
    """
    reasons: list[str] = []
    if not isinstance(brief, str):
        reasons.append(REJECT_SCHEMA)
        return reasons

    # Length cap.
    if len(brief) > _BRIEF_HARD_MAX_CHARS:
        reasons.append(REJECT_TOO_LONG)

    # Multi-sentence detection: count sentence-ending punctuation
    # outside the trailing-clause window. Allow one terminal period.
    body = brief.rstrip()
    if body.endswith("."):
        body = body[:-1]
    if re.search(r"[.!?]\s+\S", body):
        reasons.append(REJECT_MULTI_SENTENCE)

    # Quote / markup characters.
    if _QUOTE_OR_MARKUP_REGEX.search(brief):
        reasons.append(REJECT_QUOTES_OR_MARKUP)

    # Named characters.
    cast_tokens = _cast_name_tokens(ledger)
    if cast_tokens:
        brief_lower = brief.lower()
        for token in cast_tokens:
            # Word-boundary match to avoid "ed" appearing inside other words.
            if re.search(rf"\b{re.escape(token)}\b", brief_lower):
                reasons.append(REJECT_NAMED_CHARACTER)
                break

    # Dialogue verbs.
    brief_words = set(re.findall(r"[A-Za-z][A-Za-z'\-]*", brief.lower()))
    if brief_words & _DIALOGUE_VERBS:
        reasons.append(REJECT_DIALOGUE_VERB)
    if brief_words & _PLOT_VERBS:
        reasons.append(REJECT_PLOT_VERB)

    # Period literals not already present in the source ledger.
    existing = _existing_period_tokens(ledger)
    for m in _PERIOD_REGEX.finditer(brief):
        hit = m.group(0).lower()
        if hit not in existing:
            reasons.append(REJECT_UNSUPPORTED_PERIOD)
            break

    return reasons


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


# ---------------------------------------------------------------------------
# Repair pass -- refinement section 3.5 + E-18 + R-06
# ---------------------------------------------------------------------------


def _build_repair_messages(
    failed_output: str,
    rejection_reasons: list[str],
    base_user_message: str,
) -> list[dict]:
    """Build the repair-pass messages with CRITICAL prefix.

    Per R-06 / C0b refinement section 3.5: the repair prompt prepends
    an explicit `CRITICAL: You previously failed validation because:
    <reasons>` directive so the model re-orients toward the actual
    rejection class rather than re-generating the failed shape.
    """
    reasons_str = ", ".join(rejection_reasons) if rejection_reasons else "unknown"
    critical = (
        f"CRITICAL: You previously failed validation because: {reasons_str}.\n\n"
        "Rewrite this visual brief to obey the schema. Remove named "
        "characters, dialogue verbs, plot actions, unsupported dates "
        "or locations, extra sentences, quotation marks, and Markdown. "
        "Return only the JSON object.\n\n"
        f"Failed brief: {failed_output[:400]}\n"
        f"Rejection reasons: {reasons_str}\n\n"
    )
    return [
        {"role": "user", "content": critical + base_user_message},
    ]


def _repair_pass(
    failed_output: str,
    rejection_reasons: list[str],
    technical_fn: Callable[..., str],
    base_user_message: str,
    reflection_temperature: float,
) -> str:
    """Run ONE repair attempt at the clamped repair temperature.

    Per E-18 / RR-B5: `repair_temperature = min(reflection_temperature
    + 0.15, 0.55)`. The clamp keeps the repair pass inside the declared
    0.35-0.55 safe range even if a future operator sets the base
    temperature above 0.4.
    """
    repair_temperature = min(
        reflection_temperature + _REPAIR_TEMPERATURE_BUMP,
        _REPAIR_TEMPERATURE_CEILING,
    )
    # SA-101 (Sprint D D0c): surface the silent clamp. Logs the
    # temperature math plus the validator rejection reasons in scope
    # so Sprint A inspectors can tell a 0.55-ceilinged retry from a
    # 0.35 retry from a pre-clamp failure. Purely additive emission;
    # no existing log string modified.
    log.info(
        "[OTR_StoryBrief] repair pass clamped: base=%.3f bump=%.3f "
        "ceiling=%.3f -> repair_temperature=%.3f reasons=%s",
        reflection_temperature, _REPAIR_TEMPERATURE_BUMP,
        _REPAIR_TEMPERATURE_CEILING, repair_temperature, rejection_reasons,
    )
    messages = _build_repair_messages(
        failed_output, rejection_reasons, base_user_message,
    )
    return technical_fn(
        messages,
        temperature=repair_temperature,
        max_new_tokens=_REFLECTION_MAX_NEW_TOKENS,
    )


# ---------------------------------------------------------------------------
# Failure sentinel -- refinement section 4.1 + L-6
# ---------------------------------------------------------------------------


def _failure_sentinel(
    *,
    reason: str,
    technical_model_id: str,
    prompt_version: str,
) -> dict:
    """Return the 8-key meta delta for a failed reflection pass.

    `story_brief == ""` lets consumers fall through to legacy behavior.
    `story_brief_status == "failed"` makes the failure observable per
    refinement section 4.1; consumers (FLUX env, LTX, HuMo, MusicGen)
    log the status in their render output per E-07.
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
    }


def _success_delta(
    *,
    brief_model: StoryBriefModel,
    technical_model_id: str,
    prompt_version: str,
) -> dict:
    """Return the 8-key meta delta for a successful reflection pass."""
    text = brief_model.story_brief.strip()
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
    }


# ---------------------------------------------------------------------------
# Main entrypoint -- L-6 + L-8 + E-17 + E-21
# ---------------------------------------------------------------------------


def run_story_brief_reflection(
    led: Any,
    technical_fn: Callable[..., str],
    *,
    technical_model_id: str = "",
    prompt_version: str = _PROMPT_VERSION,
) -> dict:
    """Run the reflection pass and return the 8-key meta delta.

    LLM slot: technical -- structured JSON output, NOT narrative
    composition. Per L-2 the writer's technical slot is the right
    home for this call site; per E-21 the signature accepts ONLY
    `technical_fn` (no `creative_fn` parameter) so a future refactor
    cannot accidentally route the call through the creative slot
    and burn the creative budget on a JSON validation pass.

    Three SCOPED try/except blocks per E-17 / RR-B3 + L-6 /
    `run_script_doctor` precedent. Each except arm covers EXACTLY one
    operation:

      Block 1: technical_fn(messages, ...)   -> failure_sentinel
      Block 2: json.loads(extracted)         -> failure_sentinel
      Block 3: pydantic validate + content   -> repair_pass once,
                                                then failure_sentinel
                                                if still invalid

    On any path the function returns a dict; it never raises.
    """
    ledger = _ledger_data(led)

    user_message = _REFLECTION_PROMPT + _build_reflection_input(led)
    messages = [{"role": "user", "content": user_message}]

    # Block 1 -- LLM call only. Catches network / VRAM / framework
    # failures inside the generate_fn closure. The L-6 pattern (run
    # script doctor) catches broad Exception here for the same reason:
    # the LLM call is the most variable-failure-mode part of the
    # pipeline and a bare reraise would crash the whole script
    # generation over a 5-second flavor-text failure.
    try:
        raw = technical_fn(
            messages,
            temperature=_REFLECTION_TEMPERATURE,
            max_new_tokens=_REFLECTION_MAX_NEW_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001 -- narrow: only the LLM call line
        log.warning(
            "[OTR_StoryBrief] technical_fn raised: %s; "
            "returning failed-status sentinel", exc,
        )
        return _failure_sentinel(
            reason="technical_fn_exception",
            technical_model_id=technical_model_id,
            prompt_version=prompt_version,
        )

    # Block 2 -- JSON parse only. Catches malformed-JSON LLM output.
    try:
        data = _otr_json.parse_first_json_object(raw or "")
    except json.JSONDecodeError as exc:
        log.warning(
            "[OTR_StoryBrief] JSON parse failed (%s); raw=%r; "
            "returning failed-status sentinel",
            exc, (raw or "")[:200],
        )
        return _failure_sentinel(
            reason=REJECT_JSON_PARSE,
            technical_model_id=technical_model_id,
            prompt_version=prompt_version,
        )

    # Block 3 -- schema validation only. On failure run ONE repair
    # pass at the clamped repair temperature per E-18, then fall
    # through to the empty-string sentinel if still invalid.
    try:
        brief_model = StoryBriefModel.model_validate(data)
    except ValidationError as exc:
        log.warning(
            "[OTR_StoryBrief] schema validation failed (%s); attempting "
            "repair pass", exc,
        )
        rejection_reasons = [REJECT_SCHEMA]
        try:
            repaired = _repair_pass(
                failed_output=raw,
                rejection_reasons=rejection_reasons,
                technical_fn=technical_fn,
                base_user_message=user_message,
                reflection_temperature=_REFLECTION_TEMPERATURE,
            )
            brief_model = StoryBriefModel.model_validate(
                _otr_json.parse_first_json_object(repaired or ""),
            )
        except (Exception, ValidationError) as exc2:  # noqa: BLE001
            log.warning(
                "[OTR_StoryBrief] schema validation failed after repair "
                "(%s); returning failed-status sentinel", exc2,
            )
            return _failure_sentinel(
                reason=REJECT_SCHEMA,
                technical_model_id=technical_model_id,
                prompt_version=prompt_version,
            )

    # Content-level validation gate (refinement section 3.4). Pydantic
    # shape passed; now check named characters / dialogue verbs /
    # plot verbs / unsupported period literals / quote chars / etc.
    content_reasons = _validate_brief(brief_model.story_brief, ledger)
    if content_reasons:
        log.info(
            "[OTR_StoryBrief] content validation rejected: %s; "
            "attempting repair pass", content_reasons,
        )
        try:
            repaired = _repair_pass(
                failed_output=brief_model.story_brief,
                rejection_reasons=content_reasons,
                technical_fn=technical_fn,
                base_user_message=user_message,
                reflection_temperature=_REFLECTION_TEMPERATURE,
            )
            repaired_model = StoryBriefModel.model_validate(
                _otr_json.parse_first_json_object(repaired or ""),
            )
            repaired_reasons = _validate_brief(
                repaired_model.story_brief, ledger,
            )
            if repaired_reasons:
                log.warning(
                    "[OTR_StoryBrief] content validation still failed "
                    "after repair (%s); returning failed-status sentinel",
                    repaired_reasons,
                )
                return _failure_sentinel(
                    reason="content_validation_failed_after_repair",
                    technical_model_id=technical_model_id,
                    prompt_version=prompt_version,
                )
            brief_model = repaired_model
        except (Exception, ValidationError) as exc:  # noqa: BLE001
            log.warning(
                "[OTR_StoryBrief] repair pass failed (%s); "
                "returning failed-status sentinel", exc,
            )
            return _failure_sentinel(
                reason="repair_pass_exception",
                technical_model_id=technical_model_id,
                prompt_version=prompt_version,
            )

    return _success_delta(
        brief_model=brief_model,
        technical_model_id=technical_model_id,
        prompt_version=prompt_version,
    )
