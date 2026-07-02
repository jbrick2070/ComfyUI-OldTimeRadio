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

# Sprint 2A/2B: Attempt 2 structural-retry temperature for the shared
# structured_call retry ladder. STRICTLY BELOW _REFLECTION_TEMPERATURE
# -- a JSON-schema re-roll LOWERS entropy, it never raises it.
# structured_call asserts this invariant at entry. This replaces the
# old _REPAIR_TEMPERATURE_BUMP (which RAISED repair temperature by
# +0.15 -- the Sprint 2B bug the ladder conversion corrects).
_REFLECTION_STRUCTURAL_RETRY_TEMPERATURE: float = 0.15

# Per refinement section 7 hard caps: max 300 chars on the prose brief.
_BRIEF_HARD_MAX_CHARS: int = 300

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


# Sprint 8.1: cap for `atmosphere_line` (one prose sentence) sized to
# fit inside the same visual budget the prose `story_brief` uses --
# this field is what HuMo lip-sync, portrait, and OTR_VideoPlan
# consume when they want one sentence of mood without the full 300-
# char brief. 200 chars is the tight-but-usable upper bound.
_ATMOSPHERE_LINE_HARD_MAX_CHARS: int = 200

# Sprint 8.1: cap for `tempo_hint` (one or two words, e.g. "slow",
# "hurried"). Bounded small so an over-eager LLM cannot smuggle a
# paragraph into the LTX motion prompt through this field.
_TEMPO_HINT_HARD_MAX_CHARS: int = 80


class StoryBriefModel(BaseModel):
    """Schema for the reflection-pass LLM output.

    Shape (v1): a single visual brief sentence plus three short
    sidecar term arrays. The model enforces type + length only;
    content rules (named characters, dialogue verbs, period literals)
    live in the Python `_validate_brief` post-gate so the rejection-
    class messages stay readable rather than leaking pydantic
    internals into the repair-prompt body.

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

    story_brief:      str       = Field(min_length=10, max_length=_BRIEF_HARD_MAX_CHARS)
    setting_terms:    list[str] = Field(default_factory=list, max_length=10)
    lighting_terms:   list[str] = Field(default_factory=list, max_length=10)
    atmosphere_terms: list[str] = Field(default_factory=list, max_length=10)
    # Sprint 8.1 v2 additive fields.
    music_mood_terms: list[str] = Field(default_factory=list, max_length=10)
    visual_palette:   list[str] = Field(default_factory=list, max_length=10)
    key_objects:      list[str] = Field(default_factory=list, max_length=10)
    tempo_hint:       str       = Field(default="", max_length=_TEMPO_HINT_HARD_MAX_CHARS)
    atmosphere_line:  str       = Field(default="", max_length=_ATMOSPHERE_LINE_HARD_MAX_CHARS)


# ---------------------------------------------------------------------------
# Prompt body
# ---------------------------------------------------------------------------

# Per refinement section 3 + 3.3 + 3.4. Hard-capped at <250 tokens of
# prompt body (acceptance row implicit in commit gate). Strict JSON
# output, period-neutral framing, sidecar terms for the helpers.
#
# Sprint 3G: the script context handed to the model is pre-sanitized by
# `_build_reflection_input` -- every cast name and proper noun is
# replaced with a neutral token (character_a, source_entity) BEFORE the
# model sees it. The model is no longer staring at names it would then
# be told to suppress, so the long "no cast names / no proper nouns"
# suppression list collapses to a single positive instruction: write a
# visual atmosphere brief, use no names. The dialogue/plot-verb and
# invented-period guidance stays -- those constrain words the model
# could still volunteer on its own (they are not in the sanitized
# input), so they continue to do real work. The schema-side reject
# lists in `_validate_brief` remain the OUTPUT safety net unchanged.
_REFLECTION_PROMPT: str = """\
Write a visual + audio brief for this audio drama. Return ONE JSON
object, no Markdown:

{
  "story_brief":      "one clause under 300 chars",
  "setting_terms":    ["3-6 short setting nouns"],
  "lighting_terms":   ["3-6 short lighting nouns"],
  "atmosphere_terms": ["3-6 short atmosphere nouns"],
  "music_mood_terms": ["3-6 music mood adjectives"],
  "visual_palette":   ["3-6 short color or texture terms"],
  "key_objects":      ["3-6 short object nouns"],
  "tempo_hint":       "one or two words (e.g. slow, hurried)",
  "atmosphere_line":  "one sentence under 200 chars"
}

RULES:
- ONE sentence per string field. No quotes, no Markdown.
- Use no names. The context below is anonymized; describe places,
  light, and mood, never people by name.
- No dialogue verbs (speaking, arguing) or plot verbs (interrogates,
  discovers).
- No invented dates, decades, centuries, cities, countries, or eras.
  If a period is implied, use atmosphere terms (smoke-filtered,
  incandescent glow) instead of dated ones (1940s, Victorian).
- Each list term under 24 chars.
- music_mood_terms are for instrumental scoring (tense, sombre,
  hopeful). visual_palette is colors / textures. key_objects are
  concrete nouns the scene contains.

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
# Sprint 3G -- input sanitization
# ---------------------------------------------------------------------------
# Operating-Philosophy point 3: Python owns name sanitization. The
# reflection input used to hand the model a CAST: block of real
# character names plus title / scene / line text rich in proper nouns,
# then the prompt told the model to suppress exactly those names. The
# model was staring at names it was told not to use. Sprint 3G runs a
# deterministic strip BEFORE the LLM sees the text: every cast name and
# every other proper noun maps to a neutral token (character_a,
# source_entity). The mapping is stable within one call -- the same
# surface form always maps to the same token -- so cross-references in
# the input stay coherent. The OUTPUT safety net (`_validate_brief`
# reject lists) is untouched: it still catches a name the model
# volunteers on its own.

# Common-word stoplist for the proper-noun sweep. A single capitalized
# word that is one of these is NEVER treated as a proper noun even
# mid-sentence -- structural ledger markers (SCENE / INTERIOR / NIGHT),
# the section labels the input builder emits, and a handful of common
# words that show up Title-Cased in headers. Cast names are handled by
# the dedicated cast pass, so this list never needs to carry a name.
_PROPER_NOUN_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on",
    "at", "by", "for", "with", "from", "as", "is", "are", "was", "were",
    "be", "this", "that", "these", "those", "it", "its", "he", "she",
    "they", "we", "you", "i", "his", "her", "their", "our", "your",
    "scene", "interior", "exterior", "int", "ext", "night", "day",
    "morning", "evening", "afternoon", "dawn", "dusk", "later",
    "continuous", "sfx", "env", "music", "sound", "opening", "closing",
    "title", "style", "cast", "rooftop", "no", "yes",
    "ok", "okay",
})

# A multi-word Title Case run (2+ adjacent capitalized words). A run
# like "Halloran Tower" or "The Bulkhead Closes" is a proper-noun
# phrase with high confidence regardless of where it sits in a line.
_PROPER_NOUN_PHRASE_REGEX: re.Pattern[str] = re.compile(
    r"\b[A-Z][A-Za-z'\-]*(?:\s+[A-Z][A-Za-z'\-]*)+\b",
)

# A single capitalized word. Used only for the MID-SENTENCE pass -- a
# lone capitalized word at a line / sentence start is ordinary casing,
# not a proper noun, and is left alone (over-sanitizing common words
# like "Tall" or "Quiet" would strip real visual signal).
_CAPITALIZED_WORD_REGEX: re.Pattern[str] = re.compile(
    r"\b[A-Z][A-Za-z'\-]*\b",
)


def _neutral_token(prefix: str, index: int) -> str:
    """Return a stable neutral token: prefix_a, prefix_b, ... prefix_z,
    then prefix_27 onward. Deterministic for a given index."""
    letters = "abcdefghijklmnopqrstuvwxyz"
    if index < len(letters):
        return f"{prefix}_{letters[index]}"
    # Past 26 entries fall back to a numeric suffix -- still stable.
    return f"{prefix}_{index + 1}"


def _build_cast_substitution(cast: Sequence[dict]) -> dict[str, str]:
    """Map each cast name surface form (lowercase) to a neutral token.

    All surface forms of one character collapse onto that character's
    single `character_x` token, so "Jones" and "Detective Jones" both
    become `character_a` -- the input stays internally coherent.
    """
    mapping: dict[str, str] = {}
    for char_index, row in enumerate(
        [r for r in cast if isinstance(r, dict) and (r.get("name") or "").strip()]
    ):
        token = _neutral_token("character", char_index)
        name = (row.get("name") or "").strip()
        forms = [name]
        for part in re.split(r"\s+|[-_]", name):
            part = part.strip()
            if len(part) >= 3:
                forms.append(part)
        for form in forms:
            mapping.setdefault(form.lower(), token)
    return mapping


def _apply_cast_substitution(text: str, cast_map: dict[str, str]) -> str:
    """Replace every cast name surface form in `text` with its token.

    Word-boundary, case-insensitive. Longest forms first so a
    multi-word name is consumed before its parts.
    """
    if not text or not cast_map:
        return text
    for form in sorted(cast_map, key=len, reverse=True):
        token = cast_map[form]
        text = re.sub(
            rf"\b{re.escape(form)}\b", token, text, flags=re.IGNORECASE,
        )
    return text


def _proper_noun_token(
    surface: str, proper_noun_map: dict[str, str],
) -> str:
    """Return the stable `source_entity_*` token for `surface`.

    `proper_noun_map` is mutated in place: a proper noun seen for the
    first time gets a fresh token, every later occurrence within the
    same call reuses it.
    """
    key = surface.lower()
    token = proper_noun_map.get(key)
    if token is None:
        token = _neutral_token("source_entity", len(proper_noun_map))
        proper_noun_map[key] = token
    return token


def _sanitize_reflection_text(
    text: str,
    cast_map: dict[str, str],
    proper_noun_map: dict[str, str],
) -> str:
    """Apply cast-name and proper-noun substitution to one text span.

    Three deterministic passes, conservative by design so the visual
    signal the brief depends on is not stripped:

      1. Cast names -- the highest-confidence proper nouns; every
         surface form maps onto its character's `character_*` token.
      2. Multi-word Title Case runs (e.g. "Halloran Tower") -- a
         proper-noun phrase with high confidence regardless of
         position; each non-stopword word in the run maps to its own
         stable `source_entity_*` token, so a later standalone mention
         of one of those words reuses the same token.
      3. Single capitalized words that appear MID-SENTENCE -- a lone
         capitalized word at a line / sentence start is ordinary
         casing (e.g. "Tall detective", "Quiet suspect"), so only a
         capitalized word preceded by other text on the same line is
         treated as a proper noun. Structural ledger words
         (`_PROPER_NOUN_STOPWORDS`) are always left alone.

    `proper_noun_map` is mutated in place so a proper noun stays
    stable across every occurrence within one call.
    """
    if not text:
        return text
    text = _apply_cast_substitution(text, cast_map)

    # Pass 2: multi-word Title Case phrases. Each capitalized word in
    # the run is tokenized individually (stopwords left alone), so the
    # mapping for a single proper-noun word stays consistent whether it
    # appears inside a phrase here or standalone elsewhere.
    def _replace_phrase(match: re.Match[str]) -> str:
        surface = match.group(0)
        out: list[str] = []
        for word in re.split(r"(\s+)", surface):
            if not word or word.isspace():
                out.append(word)
                continue
            if word.lower() in _PROPER_NOUN_STOPWORDS:
                out.append(word)
                continue
            out.append(_proper_noun_token(word, proper_noun_map))
        return "".join(out)

    text = _PROPER_NOUN_PHRASE_REGEX.sub(_replace_phrase, text)

    # Pass 3: lone capitalized words, but only mid-sentence. A word is
    # "mid-sentence" when the character before it (skipping spaces) is
    # not a line / sentence start. Boundary characters: a true line
    # start, sentence-ending punctuation, and `]` -- the `_format_line`
    # role marker `[character/...] ` precedes the actual line text, so
    # the first word AFTER the bracket is a fresh sentence start, not a
    # mid-sentence proper noun.
    _BOUNDARY_CHARS = ".!?\n]"

    def _replace_lone(match: re.Match[str]) -> str:
        surface = match.group(0)
        if surface.lower() in _PROPER_NOUN_STOPWORDS:
            return surface
        # Look back from the match start for the first non-space char.
        i = match.start() - 1
        while i >= 0 and text[i] == " ":
            i -= 1
        if i < 0 or text[i] in _BOUNDARY_CHARS:
            # Line / sentence start -- ordinary casing, leave it.
            return surface
        return _proper_noun_token(surface, proper_noun_map)

    return _CAPITALIZED_WORD_REGEX.sub(_replace_lone, text)


def _build_reflection_input(led: Any) -> str:
    """Build the capped, sanitized reflection-prompt input from the ledger.

    Per refinement section 2: title + style slug + cast roster + scene
    headers + opening / closing snippets + non-dialogue rows. Total
    output stays well under 1500 tokens on a typical episode.

    Sprint 3G: before the text is returned, every cast name and proper
    noun is replaced with a neutral token (character_a, source_entity)
    so the LLM never sees a real name it is then told to suppress. The
    mapping is deterministic and stable within one call. This changes
    only WHAT THE LLM SEES -- the reflection still produces a meta-only
    visual atmosphere brief, and the output reject-list safety net in
    `_validate_brief` is unchanged.

    Accepts a Ledger object OR a raw dict (`led.data`) -- duck-typed
    via `_ledger_data` so tests can pass a plain dict without
    constructing a Ledger.
    """
    ledger = _ledger_data(led)
    lines: Sequence[dict] = ledger.get("lines") or []
    cast: Sequence[dict] = ledger.get("cast") or []
    meta = ledger.get("meta") or {}

    # Sprint 3G: deterministic substitution maps. The cast map is built
    # once up front so every cast member has a token before any text is
    # scanned; the proper-noun map grows as text spans are sanitized,
    # keeping each proper noun stable across its occurrences.
    cast_map = _build_cast_substitution(cast)
    proper_noun_map: dict[str, str] = {}

    def _clean(text: str) -> str:
        return _sanitize_reflection_text(text, cast_map, proper_noun_map)

    parts: list[str] = []

    # The episode title is known structured proper-noun data. Register
    # each of its non-stopword, non-cast words in the proper-noun map up
    # front so a single-word title -- which the mid-sentence sweep would
    # otherwise leave alone at a line start -- is still tokenized, and
    # so a later mention of a title word in line text reuses the same
    # `source_entity_*` token.
    title = _meta_title(ledger)
    for word in re.findall(r"[A-Za-z][A-Za-z'\-]*", title):
        if word.lower() in _PROPER_NOUN_STOPWORDS:
            continue
        if word.lower() in cast_map:
            continue
        _proper_noun_token(word, proper_noun_map)
    parts.append(f"TITLE: {_clean(title)}")
    # The style slug is a controlled vocabulary token (e.g.
    # noir_interrogation), not free prose, so it is passed through as-is
    # -- it carries genre signal the brief should reflect and contains
    # no character names.
    parts.append(f"STYLE: {meta.get('style') or ''}")

    # CAST roster: real names are replaced with their neutral tokens so
    # the roster still tells the model how many distinct people exist
    # and what each looks like, without naming any of them.
    parts.append("CAST:")
    if not cast:
        parts.append("  (no cast)")
    for row in cast:
        if not isinstance(row, dict):
            continue
        name = (row.get("name") or "").strip()
        if not name:
            continue
        token = cast_map.get(name.lower(), "character")
        desc = _clean(
            _truncate((row.get("character_description") or "").strip(), 200)
        )
        parts.append(f"  - {token}: {desc}" if desc else f"  - {token}")

    # Scene headers (refinement section 2 says any line with
    # speaker_role == "scene" or section markers).
    scene_rows = [ln for ln in lines if (ln.get("speaker_role") or "") == "scene"]
    if scene_rows:
        parts.append("SCENE HEADERS:")
        for row in scene_rows:
            parts.append(
                f"  {_clean(_truncate((row.get('text') or '').strip(), 120))}"
            )

    # Opening + closing snippets.
    if lines:
        opening = lines[:_OPENING_LINE_CAP]
        parts.append(f"OPENING ({len(opening)} lines):")
        for ln in opening:
            parts.append(f"  {_clean(_format_line(ln))}")
        if len(lines) > _OPENING_LINE_CAP + _CLOSING_LINE_CAP:
            closing = lines[-_CLOSING_LINE_CAP:]
            parts.append(f"CLOSING ({len(closing)} lines):")
            for ln in closing:
                parts.append(f"  {_clean(_format_line(ln))}")

    # Non-dialogue rows (MUSIC / ENV / SCENE markers) -- these
    # carry the sonic-architecture signal a visual brief should reflect.
    # ("sfx" removed 2026-07-01, rip-sfx-broll.)
    non_dialogue_roles = {"music", "env"}
    non_dialogue = [
        ln for ln in lines
        if (ln.get("speaker_role") or "").lower() in non_dialogue_roles
    ]
    if non_dialogue:
        parts.append(f"NON-DIALOGUE ROWS ({len(non_dialogue)}):")
        for ln in non_dialogue[:30]:  # cap at 30 to bound the prompt
            parts.append(f"  {_clean(_format_line(ln))}")

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

# Sprint 2A/2D: the shared structured-JSON retry ladder. The reflection
# pass routes its single LLM call through it -- the ladder subsumes the
# former hand-rolled call -> parse -> validate -> repair sequence.
# Package import in production; flat import under standalone test.
try:
    from ._otr_structured_call import (
        structured_call,
        StructuredCallFailedError,
        PostValidationError,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_structured_call import (  # type: ignore
        structured_call,
        StructuredCallFailedError,
        PostValidationError,
    )

# Sprint 2C: typed repair-prompt factories. run_story_brief_reflection
# passes a dispatching factory so structured_call's Attempt 3 routes the
# repair turn by failure class (forbidden name, narration leak, over-
# length, and so on). Package import in production; flat import under
# standalone test.
try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore


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
) -> dict:
    """Run the reflection pass and return the 8-key meta delta.

    LLM slot: technical -- structured JSON output, NOT narrative
    composition. Per L-2 the writer's technical slot is the right
    home for this call site; per E-21 the signature accepts ONLY
    `technical_fn` (no `creative_fn` parameter) so a future refactor
    cannot accidentally route the call through the creative slot
    and burn the creative budget on a JSON validation pass.

    Sprint 2A/2D: the call routes through the shared `structured_call`
    3-attempt retry ladder (base -> structural retry -> typed repair).
    The ladder subsumes the former three hand-rolled arms -- LLM call,
    JSON parse, schema validate -- plus the single repair pass. Two
    failure classes the ladder reports back here:

      * `post_validator` carries the section-3.4 content gate (named
        characters, dialogue / plot verbs, unsupported period
        literals, quote / markup chars). A content rejection re-rolls
        the ladder exactly like a schema failure.
      * `structured_call` does NOT catch slot-fn (technical_fn)
        exceptions -- a VRAM / loader / framework failure inside the
        LLM closure propagates out, and the broad `except Exception`
        below catches it.

    An exhausted ladder (`StructuredCallFailedError`) and a raising
    slot fn both map to `_failure_sentinel`. On every path the
    function returns a dict; it never raises.
    """
    ledger = _ledger_data(led)

    user_message = _REFLECTION_PROMPT + _build_reflection_input(led)
    messages = [{"role": "user", "content": user_message}]

    # Content gate (refinement section 3.4). structured_call runs this
    # on every schema-valid instance; a non-None return is a content
    # rejection (named characters / dialogue + plot verbs / unsupported
    # period literals / quote chars) and advances the ladder exactly
    # like a schema failure. Returns None to accept the brief.
    def _content_validator(model: StoryBriefModel) -> str | None:
        reasons = _validate_brief(model.story_brief, ledger)
        if reasons:
            return ", ".join(reasons)
        return None

    # LLM slot: technical -- structured JSON validation pass, not
    # narrative composition; routed through the shared ladder.
    try:
        brief_model = structured_call(
            prompt=messages,
            schema=StoryBriefModel,
            slot_fn=technical_fn,
            base_temperature=_REFLECTION_TEMPERATURE,
            structural_retry_temperature=_REFLECTION_STRUCTURAL_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_content_validator,
            max_new_tokens=_REFLECTION_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name="run_story_brief_reflection",
        )
    except StructuredCallFailedError as exc:
        # Ladder exhausted. Attribute the failure to its class via the
        # last error the ladder captured so the sentinel `reason` stays
        # as forensically specific as the pre-conversion arms did.
        last = exc.last_error
        if isinstance(last, json.JSONDecodeError):
            reason = REJECT_JSON_PARSE
        elif isinstance(last, PostValidationError):
            reason = "content_validation_failed_after_repair"
        elif isinstance(last, ValidationError):
            reason = REJECT_SCHEMA
        else:
            reason = "structured_call_failed"
        log.warning(
            "[OTR_StoryBrief] structured_call exhausted the retry ladder "
            "after %d attempt(s) (last error: %s); returning failed-status "
            "sentinel (reason=%s)", exc.attempts, exc.last_error, reason,
        )
        return _failure_sentinel(
            reason=reason,
            technical_model_id=technical_model_id,
            prompt_version=prompt_version,
        )
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        # structured_call does not catch slot-fn exceptions: a network /
        # VRAM / framework failure inside the technical_fn closure lands
        # here. A bare reraise would crash the whole script generation
        # over a flavor-text failure -- the L-6 reason for catching broad.
        log.warning(
            "[OTR_StoryBrief] technical_fn raised %s: %s; returning "
            "failed-status sentinel", type(exc).__name__, exc,
        )
        return _failure_sentinel(
            reason="technical_fn_exception",
            technical_model_id=technical_model_id,
            prompt_version=prompt_version,
        )

    return _success_delta(
        brief_model=brief_model,
        technical_model_id=technical_model_id,
        prompt_version=prompt_version,
    )
