"""nodes/_otr_radio_editor.py -- Story Spine Stream D: the Radio Editor.

Owns episode LENGTH and radio pacing as LEDGER-NATIVE, RENDER-SAFE
per-beat edits. The editor LLM never writes ledger keys: it returns a
``RadioEditPlan`` (an edit list against fixed beat indices) which pure
Python validates (Guards 1/2/3 + post_validator) and applies. The
ledger is read-only to the LLM -- a beat's segment count, scene,
bindings, and visual nouns are render contracts for the
FLUX -> HuMo -> LTX -> Bark cascade, not just text (story-spine Sec 2,
invariant 3).

Implements exactly the Stream D contract in
``docs/otr-story-spine.md`` Section 3 (BeatEdit / RadioEditPlan, the
two-tier action set, Guards 1/2/3, the apply step, post_validator, and
the gate), corrected by PART 0 drift note D7 (``meta.visual_plan.scenes``
is ALWAYS ``[]``; the Guard-3 noun baseline = original beat prose UNION
the cast ``portrait_prompt``s in ``meta.visual_plan.characters``).

Action set -- two tiers by render impact:

  Tier 1 (segment-set STABLE -- only one line's text/audio changes):
    KEEP, SHORTEN_LINE, CLEAN_PUNCTUATION, RECOMPOSE_BEAT_SAME_INTENT
  Tier 2 (segment-set CHANGING -- adds/removes a dialogue unit, forces
          cascade re-alignment):
    CUT_LINE, REMOVE_REDUNDANT_BEAT, SPLIT_LINE, MERGE_SHORT_LINES

Tier 1 leaves the beat/slot/scene set byte-identical -- downstream
re-renders only that one line's audio and re-times that single segment.
Tier 2 changes the COUNT of dialogue units, so Python re-indexes the
affected beats with clean ids and stamps the affected span
``needs_render_realign`` so the cascade knows to re-time it.

This module is DORMANT until the writer wires it in-process inside
``run()`` (mirroring ``run_story_brief_reflection``). It adds no node
surface, no widget, no output socket, and no workflow-JSON entry
(story-spine D4/D5). When wired, the editor reads its model id from the
in-process ``resolved["creative_writing_model"]`` dict; the recompose
sub-pass reuses ``_otr_line_composer.compose_line`` on the same creative
slot. To keep this module + its self-test runnable with no real model,
the recompose/generate callable is INJECTED as an argument
(``recompose_fn``) and the editor LLM call is INJECTED as a ``slot_fn``
-- the production wiring passes the real slot fns; the self-test passes
stubs.

Slot tag: see the ``# LLM slot: creative`` markers at the two call
sites (the editor plan call and the recompose) -- both narrative passes
route to the creative slot per project rule 6.

PD3 (workflow JSON): N/A; this module adds no node surface.
PD6 (LLM-slot tagging): both LLM call sites here are tagged creative.

REAL ledger keys this module guards against (verified against the live
tree, file:line):
  * ``lines[]`` row keys (the editable dialogue unit) come from
    ``production_ledger.Ledger.init_lines_from_outline``
    (nodes/production_ledger.py:763-788):
      line_id, shot_id, beat_id, char_id, text, traits, boundary,
      char_count, word_count, bark_wav_path, start_s, dur_s,
      speaker_role, arc_phase, compose_flags, beat_intent,
      target_words, dialogue_slot_id
  * ``speaker_role`` (nodes/production_ledger.py:716, :753-762, :1196)
    distinguishes VOICED beats ("character" / "announcer") from
    NON-VOICED beats ("music_open" / "music_close" / "music_inter").
    The editor only ever touches voiced beats; music beats
    are preserved by construction.
  * ``voice_preset`` lives on ``cast[]`` and on ``beats[]``
    (nodes/production_ledger.py:588, :646 via set_beats) -- NOT on a
    line row -- so Guard 1 watches it as a forbidden mutation key and
    the apply step never writes it.
  * ``scene_id`` lives on ``beats[]`` / ``shots[]``
    (nodes/production_ledger.py:600, :644) -- never on a line row.
  * ``dialogue_slot_id`` (nodes/production_ledger.py:752, :787) is the
    stable voiced-slot id (``d\\d{3}``); Guard 1 forbids reassigning it.
  * ``meta.visual_plan`` shape (nodes/OTR_LedgerScriptWriter.py:3711):
    ``{"characters": {name: {"portrait_prompt": str}}, "scenes": [],
    "style": str}`` -- scenes is ALWAYS empty (D7), so Guard 3 draws
    its allowed-noun baseline from the original beat prose UNION the
    cast portrait_prompts only.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
4-space indentation. Placeholders/stubs use descriptive names ("test"),
never the forbidden filler word (project rule 5).
"""
# LLM slot: creative
# Reason: the editor PLAN call and the RECOMPOSE_BEAT_SAME_INTENT pass
# are both narrative passes (they reshape spoken dialogue and own
# length / pacing), so per project rule 6 they route to the
# creative_writing_model slot. The model id is supplied by the wiring
# from resolved["creative_writing_model"]; this module never exposes a
# model_id widget.
from __future__ import annotations

import copy
import hashlib
import logging
import math
import re
from dataclasses import dataclass, replace
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, model_validator

try:  # pragma: no cover - package and standalone import styles
    from ._otr_text_metrics import (
        canonical_char_count,
        canonical_word_count,
        set_line_text_metrics,
    )
except ImportError:  # pragma: no cover
    from _otr_text_metrics import (  # type: ignore
        canonical_char_count,
        canonical_word_count,
        set_line_text_metrics,
    )

try:  # pragma: no cover - package and standalone import styles
    from . import _otr_word_delivery as _OTRWD
except ImportError:  # pragma: no cover
    import _otr_word_delivery as _OTRWD  # type: ignore


log = logging.getLogger("OTR")


# Shared helpers. Package import in production; flat import when the
# module is loaded standalone / under test. Mirrors the import guard in
# `_otr_structured_call.py` and `_otr_story_brief.py`. structured_call
# is the ONLY structured-JSON path (story-spine Sec 2: no new parser,
# no new retry). It is imported lazily inside the entrypoint so a
# pure-Python caller (guards / apply / self-test) never pulls it in.


# ---------------------------------------------------------------------------
# Tunable constants -- the deterministic guard ceilings
# ---------------------------------------------------------------------------

# Stream D Guard 2: one breath per spoken line. ~35 words is the hard
# per-line ceiling (Bark timing, HuMo mouth motion, LTX temporal
# consistency). SPLIT_LINE is the over-cap remedy.
MAX_WORDS_PER_LINE: int = 35

# Stream D Guard 2: a fixed character ceiling per line, paired with the
# word ceiling. A line can be under 35 words yet still be an over-long
# run-on; the char cap catches that. Sized for one breath of radio
# delivery.
MAX_CHARS_PER_LINE: int = 240

# Historical fallback for ledgers/fixtures that predate the production
# ``meta.word_budget`` receipt. A present production receipt always owns
# the target; these constants are not a live-route policy.
TARGET_WORD_TOTAL: int = 350
WORD_TOTAL_TOLERANCE: float = 0.20

# A SHORTEN/RECOMPOSE/SPLIT/MERGE line must not be empty -- an empty
# spoken slot is a render defect, not an edit.
MIN_WORDS_PER_LINE: int = 1


@dataclass(frozen=True)
class _WordBudgetSpec:
    """Validated character-dialogue budget owned by the ledger receipt."""

    target_words: int
    min_ratio: float
    max_ratio: float
    source: str

    @property
    def lower_words(self) -> float:
        return self.target_words * self.min_ratio

    @property
    def upper_words(self) -> float:
        return self.target_words * self.max_ratio


# ---------------------------------------------------------------------------
# Action taxonomy -- the two tiers (story-spine Sec 3, Stream D)
# ---------------------------------------------------------------------------

# Tier 1: segment-set STABLE. The beat/slot/scene set is byte-identical
# after the edit; only one line's text (and therefore its re-rendered
# audio) changes. SMOOTH_DIALOGUE / CLARIFY_TURN are micro-repair-only
# Tier-1 rewrites (fresh prose for one beat, like RECOMPOSE): they change
# a single line's text in place and so leave the segment set intact.
TIER1_ACTIONS: frozenset = frozenset(
    {
        "KEEP",
        "SHORTEN_LINE",
        "CLEAN_PUNCTUATION",
        "RECOMPOSE_BEAT_SAME_INTENT",
        "SMOOTH_DIALOGUE",
        "CLARIFY_TURN",
    }
)

# Tier 2: segment-set CHANGING. Adds or removes a dialogue unit, so the
# count of voiced beats changes and the affected span needs cascade
# re-alignment.
TIER2_ACTIONS: frozenset = frozenset(
    {"CUT_LINE", "REMOVE_REDUNDANT_BEAT", "SPLIT_LINE", "MERGE_SHORT_LINES"}
)

ALL_ACTIONS: Tuple[str, ...] = (
    "KEEP",
    "SHORTEN_LINE",
    "CLEAN_PUNCTUATION",
    "RECOMPOSE_BEAT_SAME_INTENT",
    "SMOOTH_DIALOGUE",
    "CLARIFY_TURN",
    "CUT_LINE",
    "REMOVE_REDUNDANT_BEAT",
    "SPLIT_LINE",
    "MERGE_SHORT_LINES",
)

# Actions that supply replacement prose in `new_line`. RECOMPOSE is in
# this set for length/visual-noun checking purposes, but its prose is
# OPTIONAL at plan-validation time -- the editor may name a beat to
# recompose and delegate the actual text to the injected recompose_fn
# (production: compose_line), so `new_line` may be None for RECOMPOSE
# and is resolved before apply. See `_PROSE_REQUIRED_ACTIONS`.
_NEW_LINE_ACTIONS: frozenset = frozenset(
    {
        "SHORTEN_LINE",
        "RECOMPOSE_BEAT_SAME_INTENT",
        "SPLIT_LINE",
        "MERGE_SHORT_LINES",
        "SMOOTH_DIALOGUE",
        "CLARIFY_TURN",
    }
)

# Actions whose `new_line` MUST be present at plan-validation time.
# RECOMPOSE_BEAT_SAME_INTENT is excluded: its prose may be supplied
# inline OR generated by recompose_fn after the plan validates.
# SMOOTH_DIALOGUE / CLARIFY_TURN are micro-repair edits whose prose the
# LLM emits inline in the plan, so their new_line is required up front.
_PROSE_REQUIRED_ACTIONS: frozenset = frozenset(
    {
        "SHORTEN_LINE",
        "SPLIT_LINE",
        "MERGE_SHORT_LINES",
        "SMOOTH_DIALOGUE",
        "CLARIFY_TURN",
    }
)

# Actions that emit FRESH prose (not a deterministic transform of the
# original text): RECOMPOSE regenerates a flat beat; SMOOTH_DIALOGUE /
# CLARIFY_TURN rewrite a chopped or muddy line. All three are fenced by
# Guard 3 (visual noun) because fresh prose is exactly where a new
# physical prop / place / creature could slip in (go-forward Sprint 2).
_FRESH_PROSE_ACTIONS: frozenset = frozenset(
    {"RECOMPOSE_BEAT_SAME_INTENT", "SMOOTH_DIALOGUE", "CLARIFY_TURN"}
)

# Voiced speaker_role values -- the only beats the editor may touch.
# (production_ledger.py: "character" / "announcer" are filled by the
# composer; the rest are non-voiced and complete at init.)
VOICED_ROLES: frozenset = frozenset({"character", "announcer"})

# Non-voiced (music) speaker_role values. Preserved by
# construction -- the editor never sees or writes them.
NON_VOICED_ROLES: frozenset = frozenset(
    {"music_open", "music_close", "music_inter"}
)

# Ledger line-row keys that are render-contract BINDINGS. Guard 1
# rejects any plan that would alter one of these on a surviving beat;
# the apply step never writes them. (`scene_id` / `voice_preset` are
# not line-row keys but a non-conforming edit dict could try to inject
# them, so they are watched explicitly per the Stream D Guard-1 list.)
STRUCTURAL_LOCK_KEYS: Tuple[str, ...] = (
    "beat_id",
    "line_id",
    "dialogue_slot_id",
    "char_id",
    "speaker_id",
    "character_id",
    "scene_id",
    "voice_preset",
    "speaker_role",
    "shot_id",
)


# ---------------------------------------------------------------------------
# Schemas (story-spine Sec 3, Stream D) -- via structured_call
# ---------------------------------------------------------------------------


class BeatEdit(BaseModel):
    """One edit instruction against a fixed beat index.

    The editor returns these; it never writes ledger keys. `new_line`
    carries replacement prose for the actions that rewrite a line's
    text (SHORTEN/RECOMPOSE/SPLIT/MERGE); `merge_with_index` names the
    adjacent same-speaker beat for MERGE_SHORT_LINES.
    """

    # Schema-adherence (2026-06-25): canonical_field -> synonym keys a writer may
    # emit. Read by the shared `apply_field_aliases` helper in the before-validator
    # below. v1 v-set proven against the real Opus failure (the action under
    # `lever`) + the shipped BUG-LOCAL-303 remaps (`index`, `merge_with`).
    __otr_field_aliases__: ClassVar[dict[str, tuple[str, ...]]] = {
        "beat_index": ("index",),
        "merge_with_index": ("merge_with",),
        "action": ("lever",),
    }

    beat_index: int = Field(..., ge=0, description="0-based index into the voiced-beat list")
    action: str = Field(
        ...,
        description="one of the two-tier action set",
    )
    new_line: Optional[str] = Field(
        default=None,
        description="replacement prose for SHORTEN/RECOMPOSE/SPLIT/MERGE",
    )
    merge_with_index: Optional[int] = Field(
        default=None,
        ge=0,
        description="MERGE_SHORT_LINES only: adjacent same-speaker beat to fold in",
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_field_aliases(cls, data):
        """BUG-LOCAL-303 + 2026-06-25 schema-adherence: frontier / remote writers
        (claude-opus included) emit shortened or renamed field names -- ``index``
        for ``beat_index``, ``merge_with`` for ``merge_with_index``, and (the
        proven 2026-06-25 Opus failure) ``lever`` for the required ``action`` on a
        NESTED BeatEdit. Normalize via the shared whitelist-exact alias helper
        keyed on ``__otr_field_aliases__`` (canonical-wins, exactly-one-synonym,
        dict-only, byte-identical on canonical input) so the length / micro-repair
        pass validates on attempt 1 instead of burning credit-billed
        structured-call retries on a pure field-name mismatch. pydantic runs this
        on every nested edit during ``RadioEditPlan`` validation. A genuinely
        missing ``action`` still fails LOUD, and Guard1 (``post_validate_plan``)
        rejects an out-of-``ALL_ACTIONS`` value, so a mis-aliased value is
        fail-closed, never silent-wrong."""
        try:
            from ._otr_structured_call import apply_field_aliases
        except ImportError:  # pragma: no cover - standalone / test load
            from _otr_structured_call import apply_field_aliases  # type: ignore
        return apply_field_aliases(cls.__otr_field_aliases__, data)

    def is_tier1(self) -> bool:
        return self.action in TIER1_ACTIONS

    def is_tier2(self) -> bool:
        return self.action in TIER2_ACTIONS


class RadioEditPlan(BaseModel):
    """The full edit list for one episode + the editor's projected word
    total. ``post_validate_plan`` is the deterministic gate over an
    instance (passed to ``structured_call`` as ``post_validator``)."""

    edits: List[BeatEdit] = Field(default_factory=list)
    projected_word_total: int = Field(..., ge=0)

    @model_validator(mode="before")
    @classmethod
    def _unwrap_schema_name_wrapper(cls, data):
        """S-D (gemma wrapper-key drift): some local writers (gemma) emit the
        plan nested under a top-level schema-name key --
        ``{"RadioEditPlan": {"edits": [...], "projected_word_total": N}}`` --
        which made ``projected_word_total`` read as 'missing', exhausted the
        retry ladder, and SILENTLY skipped length normalization (warn-only).

        Peel EXACTLY a single ``{"RadioEditPlan": {...}}`` wrapper (the class
        name, dict-only inner). An AMBIGUOUS / multi-key wrapper, a wrong key, or
        a non-dict inner is LEFT UNTOUCHED so it still fails LOUD -- never
        silently reinterpret a malformed payload. Narrow + local; NOT the shared
        tolerant alias core (BeatEdit._accept_field_aliases)."""
        if (isinstance(data, dict) and len(data) == 1
                and isinstance(data.get(cls.__name__), dict)):
            return data[cls.__name__]
        return data


# ---------------------------------------------------------------------------
# Word / char counting -- match the writer's stamp formula
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    """Word count matching the writer's ledger stamp formula."""
    return canonical_word_count(text)


def _char_count(text: str) -> int:
    return canonical_char_count(text)


# ---------------------------------------------------------------------------
# Voiced-beat view over the ledger
# ---------------------------------------------------------------------------
#
# The editor operates on the list of VOICED beats only (the dialogue
# units it is allowed to reshape). Non-voiced music beats are
# invisible to it and are preserved by construction. A "beat_index" in
# a BeatEdit indexes THIS voiced view, not the raw ledger lines[] list
# (which interleaves music rows). The mapping back to the raw rows
# is kept so the apply step mutates the real ledger in place.


def _is_voiced(row: Dict[str, Any]) -> bool:
    """A line row is voiced (editor-touchable) when its speaker_role is
    a voiced role. Rows with an unknown / missing role are treated as
    NON-voiced (fail-safe: the editor never touches what it cannot
    classify)."""
    if not isinstance(row, dict):
        return False
    role = str(row.get("speaker_role") or "")
    return role in VOICED_ROLES


def voiced_beats(ledger: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the editor's working view: the list of voiced line rows,
    in ledger order. References into ``ledger["lines"]`` (mutating-
    friendly), so an in-place text edit on a returned row updates the
    real ledger."""
    lines = (ledger.get("lines") if isinstance(ledger, dict) else None) or []
    return [row for row in lines if _is_voiced(row)]


def _voiced_index_map(ledger: Dict[str, Any]) -> List[int]:
    """Return the raw ``lines[]`` indices of the voiced rows, in order.
    ``map[voiced_i] == raw_lines_index``. Lets the apply step splice
    the raw list while the LLM only ever named voiced indices."""
    lines = (ledger.get("lines") if isinstance(ledger, dict) else None) or []
    return [i for i, row in enumerate(lines) if _is_voiced(row)]


# ---------------------------------------------------------------------------
# Concrete-noun extraction (Guard 3, synonym-tolerant)
# ---------------------------------------------------------------------------
#
# Guard 3 is deterministic and synonym-tolerant: it diffs the
# concrete-noun set of a recomposed line against an allowed set built
# from (the original beat prose) UNION (the cast portrait_prompts). It
# must REJECT a NEW physical prop / location / weather / costume /
# machine / creature / scenery, while ALLOWING morphological and
# common-synonym variants (so it does not fight the recompose it is
# meant to fence, per P4/P5). It runs with no NLP model -- a small
# closed lexicon + morphological normalization + a synonym map.

# Function words / pronouns / connectives that are never "concrete
# nouns" for the purpose of the visual diff. Kept deliberately small;
# the goal is to drop grammatical scaffolding, not to do real POS.
_STOPWORDS: frozenset = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "if", "then", "so", "as",
        "of", "to", "in", "on", "at", "by", "for", "with", "from", "into",
        "onto", "over", "under", "up", "down", "out", "off", "about",
        "this", "that", "these", "those", "it", "its", "he", "she", "they",
        "them", "him", "her", "his", "their", "we", "us", "our", "you",
        "your", "i", "me", "my", "mine", "who", "whom", "whose", "which",
        "what", "when", "where", "why", "how", "is", "are", "was", "were",
        "be", "been", "being", "am", "do", "does", "did", "have", "has",
        "had", "will", "would", "shall", "should", "can", "could", "may",
        "might", "must", "not", "no", "yes", "now", "here", "there",
        "all", "any", "some", "each", "every", "more", "most", "very",
        "just", "only", "too", "again", "still", "yet", "than", "such",
        "one", "two", "three", "first", "last", "next", "well", "okay",
        "ok", "oh", "ah", "hey", "please", "thanks", "thank", "let",
        "lets", "go", "come", "get", "got", "make", "made", "see", "saw",
        "say", "said", "tell", "told", "know", "knew", "think", "thought",
        "want", "wanted", "need", "needed", "feel", "felt", "look",
        "looked", "take", "took", "give", "gave", "keep", "kept", "put",
        "good", "bad", "great", "little", "big", "old", "new", "right",
        "wrong", "sure", "fine", "really", "maybe", "perhaps", "because",
        "while", "until", "before", "after", "though", "although",
    }
)

# Common irregular plural / verb normalizations the regular -s / -es /
# -ies suffix stripper would miss. Maps surface form -> singular stem so
# "men" and "man" collapse to one token (morphological tolerance).
_IRREGULAR_SINGULARS: Dict[str, str] = {
    "men": "man",
    "women": "woman",
    "children": "child",
    "people": "person",
    "feet": "foot",
    "teeth": "tooth",
    "mice": "mouse",
    "geese": "goose",
    "knives": "knife",
    "wolves": "wolf",
    "lives": "life",
    "leaves": "leaf",
    "shelves": "shelf",
    "thieves": "thief",
    "halves": "half",
}

# Synonym / variant groups. Every member maps to a shared canonical
# token so a recompose that swaps "gun" for "pistol" or "automobile"
# for "car" is treated as the SAME concrete noun (synonym tolerance).
# Only swaps that preserve the physical referent belong here -- this is
# not a thesaurus, it is an equivalence map for render-relevant props /
# places / weather / creatures.
_SYNONYM_GROUPS: Tuple[Tuple[str, ...], ...] = (
    ("car", "auto", "automobile", "motorcar", "vehicle"),
    ("gun", "pistol", "revolver", "firearm", "sidearm"),
    ("knife", "blade", "dagger"),
    ("ship", "vessel", "boat"),
    ("spaceship", "starship", "spacecraft", "rocket"),
    ("house", "home", "dwelling", "cottage"),
    ("road", "street", "lane", "avenue"),
    ("coat", "jacket", "overcoat"),
    ("hat", "cap"),
    ("storm", "tempest"),
    ("rain", "downpour", "drizzle"),
    ("fog", "mist", "haze"),
    ("robot", "android", "automaton"),
    ("creature", "beast", "monster"),
    ("light", "lamp", "lantern"),
    ("phone", "telephone", "receiver"),
    ("radio", "wireless", "transmitter"),
    ("forest", "woods", "woodland"),
    ("hill", "ridge", "slope"),
    ("door", "doorway", "hatch"),
    ("window", "porthole"),
)


def _build_synonym_canon(groups: Tuple[Tuple[str, ...], ...]) -> Dict[str, str]:
    canon: Dict[str, str] = {}
    for group in groups:
        head = group[0]
        for member in group:
            canon[member] = head
    return canon


_SYNONYM_CANON: Dict[str, str] = _build_synonym_canon(_SYNONYM_GROUPS)


def _singularize(token: str) -> str:
    """Best-effort morphological normalization to a singular stem.

    Handles irregulars, then regular -ies/-es/-s plurals. Deliberately
    conservative -- it must not over-collapse distinct nouns (so it does
    not touch short tokens or strip -ss). The goal is morphological
    tolerance ("guards" == "guard"), not stemming."""
    if token in _IRREGULAR_SINGULARS:
        return _IRREGULAR_SINGULARS[token]
    if len(token) <= 3:
        return token
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("sses"):
        return token[:-2]  # "glasses" -> "glass"
    if token.endswith("es") and token[-3:-2] in ("s", "x", "z", "h"):
        return token[:-2]  # "boxes" -> "box", "dishes" -> "dish"
    if token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _normalize_noun(token: str) -> str:
    """Lowercase -> singularize -> map through the synonym canon. Two
    surface words that share a normalized form are the SAME concrete
    noun for Guard 3."""
    low = token.lower()
    singular = _singularize(low)
    return _SYNONYM_CANON.get(singular, singular)


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def concrete_nouns(text: str) -> set:
    """Extract the normalized concrete-noun set of ``text``.

    Heuristic (no NLP model): tokenize on word characters, drop
    stopwords / pronouns / connectives and very short tokens, then
    normalize each survivor (lowercase, singularize, synonym-canon).
    The result is the set of render-relevant content words -- props,
    places, weather, costume, machines, creatures, scenery -- modulo
    morphology and synonymy. It deliberately over-includes plain
    content words (a few verbs/adjectives slip through), because Guard 3
    only ever asks "is this token NEW vs the allowed baseline" -- a word
    already present in the baseline is allowed regardless of its part of
    speech, and the cost of a false content word in BOTH sets is zero.
    """
    out: set = set()
    for raw in _TOKEN_RE.findall(text or ""):
        low = raw.lower()
        if low in _STOPWORDS:
            continue
        if len(low) <= 2:
            continue
        out.add(_normalize_noun(raw))
    return out


def _portrait_prompt_corpus(ledger: Dict[str, Any]) -> str:
    """Concatenate every cast portrait_prompt from
    ``meta.visual_plan.characters`` into one corpus string.

    Per story-spine D7 the writer stamps
    ``meta["visual_plan"]["characters"][name]["portrait_prompt"]`` and
    ``scenes`` is ALWAYS ``[]`` -- so the scene plan contributes nothing
    and is intentionally NOT read here. Tolerates both the live key
    ``portrait_prompt`` and a legacy plural ``portrait_prompts``
    (string or list) so the guard does not silently lose its baseline
    if the writer key ever drifts.
    """
    if not isinstance(ledger, dict):
        return ""
    meta = ledger.get("meta") or {}
    visual_plan = meta.get("visual_plan") or {}
    characters = visual_plan.get("characters") or {}
    parts: List[str] = []
    if isinstance(characters, dict):
        char_iter = characters.values()
    elif isinstance(characters, list):
        char_iter = characters
    else:
        char_iter = []
    for entry in char_iter:
        if not isinstance(entry, dict):
            continue
        val = entry.get("portrait_prompt")
        if val is None:
            val = entry.get("portrait_prompts")
        if isinstance(val, (list, tuple)):
            parts.extend(str(v) for v in val)
        elif val:
            parts.append(str(val))
    return " ".join(parts)


def allowed_noun_set(original_beat_text: str, ledger: Dict[str, Any]) -> set:
    """Guard 3 allowed baseline = the original beat's own prose UNION
    the cast portrait_prompts. ``scenes`` is empty (D7) so it does not
    contribute. Returned as a normalized concrete-noun set."""
    corpus = (original_beat_text or "") + " " + _portrait_prompt_corpus(ledger)
    return concrete_nouns(corpus)


def new_visual_nouns(
    new_line: str,
    original_beat_text: str,
    ledger: Dict[str, Any],
) -> List[str]:
    """Return the sorted list of concrete nouns present in ``new_line``
    but absent from the allowed baseline. Empty list == the recompose
    introduced no new physical prop/location/weather/costume/machine/
    creature/scenery (synonyms + morphology already collapsed)."""
    allowed = allowed_noun_set(original_beat_text, ledger)
    incoming = concrete_nouns(new_line)
    return sorted(incoming - allowed)


# ---------------------------------------------------------------------------
# Guard 1 -- Structural Lock (deterministic)
# ---------------------------------------------------------------------------


class GuardError(ValueError):
    """A deterministic guard rejected the plan. Carries a human-readable
    message; raised by the standalone guard helpers, surfaced as a
    return string by ``post_validate_plan`` (which never raises, per the
    structured_call post_validator contract)."""


def guard1_structural_lock(
    plan: RadioEditPlan,
    voiced: List[Dict[str, Any]],
) -> Optional[str]:
    """Guard 1 (story-spine Sec 3): reject any plan that would alter a
    render-contract binding or the ledger structure beyond the explicit
    Tier-2 add/remove-with-reindex. No line is ever reassigned to a
    different speaker.

    Deterministic Python over the PLAN (the LLM only returns indices +
    replacement prose, so the only structural changes it can request are
    the legal Tier-2 ones; this guard catches a malformed plan that
    smuggles a binding change in, an out-of-range index, an unknown
    action, a missing new_line, or a MERGE that would cross speakers /
    cross a non-voiced beat).

    Returns ``None`` when the plan is structurally clean, else an error
    string.
    """
    n = len(voiced)
    seen_indices: set = set()
    for edit in plan.edits:
        # Unknown action -> reject (pydantic allows any str on `action`
        # by design so the guard owns the closed-set check and can name
        # the offending value).
        if edit.action not in ALL_ACTIONS:
            return (
                f"Guard1: unknown action {edit.action!r} on beat_index "
                f"{edit.beat_index}; allowed actions are {ALL_ACTIONS}"
            )
        # Index must be a valid voiced-beat index.
        if not (0 <= edit.beat_index < n):
            return (
                f"Guard1: beat_index {edit.beat_index} out of range "
                f"[0, {n}) for action {edit.action}"
            )
        # A non-MERGE action targeting the same index twice is an
        # ambiguous plan; reject. (MERGE legitimately references a
        # second index via merge_with_index, handled below.)
        if edit.beat_index in seen_indices:
            return (
                f"Guard1: duplicate edit on beat_index {edit.beat_index} "
                f"(action {edit.action})"
            )
        seen_indices.add(edit.beat_index)

        # An edit may never carry a structural-lock binding override.
        # The schema has no such field, but a future field or a model
        # that smuggled extra keys through model_validate would be
        # caught here. We inspect the model's extra fields if any.
        extra = getattr(edit, "model_extra", None) or {}
        for key in STRUCTURAL_LOCK_KEYS:
            if key in extra:
                return (
                    f"Guard1: edit on beat_index {edit.beat_index} attempts "
                    f"to set locked binding {key!r}; bindings are read-only "
                    f"(no speaker/scene/voice reassignment)"
                )

        # new_line presence rules.
        if edit.action in _PROSE_REQUIRED_ACTIONS:
            if edit.new_line is None or not str(edit.new_line).strip():
                return (
                    f"Guard1: action {edit.action} on beat_index "
                    f"{edit.beat_index} requires non-empty new_line"
                )
        elif edit.action == "RECOMPOSE_BEAT_SAME_INTENT":
            # new_line OPTIONAL: inline prose is checked by Guards 2/3
            # when present; when absent, recompose_fn supplies it before
            # apply. Nothing to reject here on presence grounds.
            pass
        else:
            # KEEP / CLEAN_PUNCTUATION / CUT_LINE / REMOVE_REDUNDANT_BEAT
            # must NOT carry replacement prose (CLEAN_PUNCTUATION is a
            # deterministic normalize, not a rewrite). A stray new_line
            # signals an out-of-contract plan.
            if edit.new_line is not None and str(edit.new_line).strip():
                return (
                    f"Guard1: action {edit.action} on beat_index "
                    f"{edit.beat_index} must not carry new_line "
                    f"(it changes no text or is a deterministic op)"
                )

        # merge_with_index is MERGE_SHORT_LINES only, and must name an
        # adjacent SAME-SPEAKER voiced beat -- never a reassignment,
        # never a cross-speaker fold, never a non-voiced beat.
        if edit.action == "MERGE_SHORT_LINES":
            j = edit.merge_with_index
            if j is None:
                return (
                    f"Guard1: MERGE_SHORT_LINES on beat_index "
                    f"{edit.beat_index} requires merge_with_index"
                )
            if not (0 <= j < n):
                return (
                    f"Guard1: MERGE_SHORT_LINES merge_with_index {j} out of "
                    f"range [0, {n})"
                )
            if abs(j - edit.beat_index) != 1:
                return (
                    f"Guard1: MERGE_SHORT_LINES on {edit.beat_index} may only "
                    f"fold an ADJACENT beat; merge_with_index {j} is not "
                    f"adjacent"
                )
            spk_a = _beat_speaker_key(voiced[edit.beat_index])
            spk_b = _beat_speaker_key(voiced[j])
            if spk_a != spk_b:
                return (
                    f"Guard1: MERGE_SHORT_LINES would cross speakers "
                    f"({spk_a!r} vs {spk_b!r}); merging may never reassign a "
                    f"line's speaker"
                )
        else:
            if edit.merge_with_index is not None:
                return (
                    f"Guard1: merge_with_index is MERGE_SHORT_LINES only; "
                    f"action {edit.action} on beat_index {edit.beat_index} "
                    f"must leave it null"
                )
    return None


def _beat_speaker_key(row: Dict[str, Any]) -> str:
    """The identity a surviving beat's speaker is locked to. Prefer the
    char_id (the canonical binding); fall back to speaker_role so
    announcer beats compare equal to announcer beats. Never the display
    name (which can repeat)."""
    if not isinstance(row, dict):
        return ""
    cid = str(row.get("char_id") or "")
    if cid:
        return "char:" + cid
    return "role:" + str(row.get("speaker_role") or "")


# ---------------------------------------------------------------------------
# Guard 2 -- Text Length (deterministic)
# ---------------------------------------------------------------------------


def line_over_cap(text: str) -> Optional[str]:
    """Return a reason string if ``text`` exceeds either the word cap or
    the char cap, else None. Shared by Guard 2 and the apply step."""
    wc = _word_count(text)
    cc = _char_count(text)
    if wc > MAX_WORDS_PER_LINE:
        return f"{wc} words > {MAX_WORDS_PER_LINE} word cap"
    if cc > MAX_CHARS_PER_LINE:
        return f"{cc} chars > {MAX_CHARS_PER_LINE} char cap"
    return None


def guard2_text_length(
    plan: RadioEditPlan,
    voiced: List[Dict[str, Any]],
) -> Optional[str]:
    """Guard 2 (story-spine Sec 3): every edited/recomposed line obeys a
    hard ceiling -- ``MAX_WORDS_PER_LINE`` AND ``MAX_CHARS_PER_LINE``.
    ``SPLIT_LINE`` is the remedy when a line exceeds the cap, so a
    SPLIT_LINE's new_line is allowed to be longer ONLY in the sense that
    apply() will divide it; here we still require each emitted text to
    pass the cap (a SPLIT whose halves individually bust the cap is a
    bad split). Returns None when clean, else an error string.

    Only the actions that PRODUCE new spoken text are checked
    (SHORTEN/RECOMPOSE/SPLIT/MERGE). KEEP/CLEAN_PUNCTUATION leave the
    existing text, which was already within bounds when the writer
    stamped it; CUT/REMOVE delete text. (A KEEP on a pre-existing
    over-cap line is the writer's bug, not the editor's -- Guard 2
    governs what the EDITOR emits.)
    """
    for edit in plan.edits:
        if edit.action not in _NEW_LINE_ACTIONS:
            continue
        text = str(edit.new_line or "")
        if edit.action == "SPLIT_LINE":
            # apply() splits on sentence boundaries; validate each
            # resulting fragment against the cap.
            for frag in _split_into_units(text):
                reason = line_over_cap(frag)
                if reason is not None:
                    return (
                        f"Guard2: SPLIT_LINE on beat_index {edit.beat_index} "
                        f"produced a fragment still over cap ({reason}): "
                        f"{frag!r}"
                    )
            continue
        reason = line_over_cap(text)
        if reason is not None:
            return (
                f"Guard2: {edit.action} on beat_index {edit.beat_index} is "
                f"over cap ({reason}); use SPLIT_LINE to divide an over-long "
                f"line"
            )
    return None


# ---------------------------------------------------------------------------
# Guard 3 -- Visual Noun (deterministic, synonym-tolerant)
# ---------------------------------------------------------------------------


def guard3_visual_noun(
    plan: RadioEditPlan,
    voiced: List[Dict[str, Any]],
    ledger: Dict[str, Any],
) -> Optional[str]:
    """Guard 3 (story-spine Sec 3, corrected by D7): before accepting a
    RECOMPOSE_BEAT_SAME_INTENT, diff the concrete-noun set of new_line
    against the allowed baseline = (original beat prose) UNION (cast
    portrait_prompts in meta.visual_plan.characters). ``scenes`` is
    always empty (D7) so it does not contribute. Reject any NEW physical
    prop / location / weather / costume / machine / creature / scenery;
    allow morphological / synonym variants.

    Returns None when clean, else an error string naming the offending
    noun(s). NOTE: this is the deterministic detector. The Stream D
    contract says "on a new noun, REPAIR, do not hard-FAIL" -- that
    REPAIR is owned by ``post_validate_plan`` + ``structured_call``'s
    ladder (a Guard-3 rejection advances the ladder exactly like any
    post_validator rejection, and the repair prompt names the offending
    noun); this helper's job is only to surface the violation.
    """
    for edit in plan.edits:
        if edit.action not in _FRESH_PROSE_ACTIONS:
            continue
        original = str(voiced[edit.beat_index].get("text") or "")
        offenders = new_visual_nouns(str(edit.new_line or ""), original, ledger)
        if offenders:
            return (
                f"Guard3: {edit.action} on beat_index "
                f"{edit.beat_index} introduces NEW visual noun(s) "
                f"{offenders} not implied by the original beat prose or any "
                f"cast portrait_prompt; rewrite without naming them"
            )
    return None


# ---------------------------------------------------------------------------
# post_validator (story-spine Sec 3, Stream D)
# ---------------------------------------------------------------------------


def _fallback_word_budget() -> _WordBudgetSpec:
    """Return the pre-receipt contract used only by old fixtures/ledgers."""
    return _WordBudgetSpec(
        target_words=TARGET_WORD_TOTAL,
        min_ratio=1.0 - WORD_TOTAL_TOLERANCE,
        max_ratio=1.0 + WORD_TOTAL_TOLERANCE,
        source="historical_fallback",
    )


def _resolve_word_budget(
    ledger: Dict[str, Any],
) -> Tuple[Optional[_WordBudgetSpec], Optional[str]]:
    """Resolve the one authoritative character-dialogue budget.

    A ledger with no ``meta.word_budget`` predates the production receipt and
    receives the historical 350-word fallback. Once the key is present it is
    authoritative: malformed values return a typed error instead of silently
    rewriting a requested short episode toward 350 words.
    """
    meta = ledger.get("meta") if isinstance(ledger, dict) else None
    if not isinstance(meta, dict) or "word_budget" not in meta:
        return _fallback_word_budget(), None

    receipt = meta.get("word_budget")
    if not isinstance(receipt, dict):
        return None, "meta.word_budget must be an object"

    raw_target = receipt.get("target_words")
    try:
        if isinstance(raw_target, bool):
            raise ValueError("boolean is not a word target")
        target_number = float(raw_target)
        if not math.isfinite(target_number) or not target_number.is_integer():
            raise ValueError("target must be a finite integer")
        target_words = int(target_number)
        if target_words <= 0:
            raise ValueError("target must be positive")
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"meta.word_budget.target_words is invalid: {exc}"

    band = receipt.get("band")
    if not isinstance(band, (list, tuple)) or len(band) != 2:
        return None, "meta.word_budget.band must contain two ratio multipliers"
    try:
        min_ratio = float(band[0])
        max_ratio = float(band[1])
    except (TypeError, ValueError, OverflowError) as exc:
        return None, f"meta.word_budget.band is invalid: {exc}"
    if not (math.isfinite(min_ratio) and math.isfinite(max_ratio)):
        return None, "meta.word_budget.band ratios must be finite"
    if not (0.0 < min_ratio <= 1.0 <= max_ratio):
        return None, (
            "meta.word_budget.band must be ordered positive ratios that "
            "contain 1.0"
        )
    return _WordBudgetSpec(
        target_words=target_words,
        min_ratio=min_ratio,
        max_ratio=max_ratio,
        source="meta.word_budget",
    ), None


def _budgeted_word_total(ledger: Dict[str, Any]) -> int:
    """Count the receipt-owned surface: non-skipped character dialogue."""
    lines = ledger.get("lines") if isinstance(ledger, dict) else None
    total = 0
    for row in lines or []:
        if not isinstance(row, dict) or row.get("skip"):
            continue
        if str(row.get("speaker_role") or "") != "character":
            continue
        total += _word_count(str(row.get("text") or ""))
    return total


def word_total_in_band(
    total: int,
    budget: Optional[_WordBudgetSpec] = None,
) -> bool:
    """True when a character-word total is inside the supplied budget."""
    spec = budget or _fallback_word_budget()
    return spec.lower_words <= total <= spec.upper_words


def make_post_validator(
    voiced: List[Dict[str, Any]],
    ledger: Dict[str, Any],
    *,
    turn_beat_index: Optional[int] = None,
    button_beat_index: Optional[int] = None,
    enforce_word_band: bool = True,
) -> Callable[[RadioEditPlan], Optional[str]]:
    """Build the deterministic gate over a RadioEditPlan, closed over the
    voiced-beat view + the ledger + the arc anchor indices.

    Suitable as ``structured_call(post_validator=...)``: it RETURNS an
    error string to reject (advancing the ladder / triggering the single
    repair), or None to accept. It never raises (the structured_call
    contract).

    The arc anchors (``turn_beat_index`` / ``button_beat_index``) are the
    VOICED-view indices of the Stream A turning_point / button beats. A
    Tier-2 removal that would drop either is rejected so the editor can
    never delete the turn or the payoff.
    """

    budget, budget_error = _resolve_word_budget(ledger)

    def _validate(plan: RadioEditPlan) -> Optional[str]:
        # 1. Every beat_index valid + unique (Guard 1 owns the closed-set
        #    + range + uniqueness checks; surface its message).
        err = guard1_structural_lock(plan, voiced)
        if err is not None:
            return err
        # 2. Guard 2 (length) + Guard 3 (visual noun).
        err = guard2_text_length(plan, voiced)
        if err is not None:
            return err
        err = guard3_visual_noun(plan, voiced, ledger)
        if err is not None:
            return err
        # 3. Tier-2 removals never drop the turning_point or button beat.
        removed = {
            e.beat_index
            for e in plan.edits
            if e.action in ("CUT_LINE", "REMOVE_REDUNDANT_BEAT")
        }
        if turn_beat_index is not None and turn_beat_index in removed:
            return (
                f"post_validator: a Tier-2 removal targets the turning_point "
                f"beat (voiced index {turn_beat_index}); the turn may never "
                f"be cut"
            )
        if button_beat_index is not None and button_beat_index in removed:
            return (
                f"post_validator: a Tier-2 removal targets the button beat "
                f"(voiced index {button_beat_index}); the payoff may never "
                f"be cut"
            )
        # 4. Length normalization gates on the deterministic result of the
        #    proposed edit, never the model's arithmetic claim. Micro-repair
        #    disables only this global episode-band gate; Guards 1/2/3 and the
        #    anchor checks above remain active.
        if enforce_word_band:
            if not isinstance(budget, _WordBudgetSpec):
                return (
                    "post_validator: invalid production word-budget receipt; "
                    f"refusing length mutation ({budget_error})"
                )
            try:
                simulation_ledger = copy.deepcopy(ledger)
                simulation_plan = plan
                # A general radio-editor plan may defer RECOMPOSE prose to the
                # injected composer, which runs after structured validation.
                # Preserve the original row during this structural simulation;
                # normalize_length itself never permits RECOMPOSE.
                if any(
                    e.action == "RECOMPOSE_BEAT_SAME_INTENT"
                    and not str(e.new_line or "").strip()
                    for e in plan.edits
                ):
                    simulated_edits: List[BeatEdit] = []
                    for edit in plan.edits:
                        if (
                            edit.action == "RECOMPOSE_BEAT_SAME_INTENT"
                            and not str(edit.new_line or "").strip()
                        ):
                            simulated_edits.append(edit.model_copy(update={
                                "new_line": str(
                                    voiced[edit.beat_index].get("text") or ""
                                ),
                            }))
                        else:
                            simulated_edits.append(edit)
                    simulation_plan = plan.model_copy(
                        update={"edits": simulated_edits}
                    )
                apply_plan(simulation_ledger, simulation_plan)
                actual_total = _budgeted_word_total(simulation_ledger)
            except Exception as exc:  # noqa: BLE001 - validator returns, never raises
                return (
                    "post_validator: could not simulate the proposed edit "
                    f"safely ({type(exc).__name__}: {exc})"
                )

            for sim_index, row in enumerate(voiced_beats(simulation_ledger)):
                reason = line_over_cap(str(row.get("text") or ""))
                if reason is not None:
                    return (
                        "post_validator: simulated spoken row "
                        f"{sim_index} remains over the one-breath cap "
                        f"({reason}); split or shorten that row"
                    )

            if not word_total_in_band(actual_total, budget):
                lo = math.ceil(budget.lower_words)
                hi = math.floor(budget.upper_words)
                return (
                    "post_validator: simulated character-word total "
                    f"{actual_total} outside requested band [{lo}, {hi}] "
                    f"(target {budget.target_words}); the model claimed "
                    f"projected_word_total={plan.projected_word_total}. "
                    "Revise the edits toward the requested band"
                )
        return None

    return _validate


# Convenience: the bare deterministic gate without arc anchors, for
# callers / tests that only want Guards 1-3 + the word band.
def post_validate_plan(
    plan: RadioEditPlan,
    voiced: List[Dict[str, Any]],
    ledger: Dict[str, Any],
    *,
    turn_beat_index: Optional[int] = None,
    button_beat_index: Optional[int] = None,
    enforce_word_band: bool = True,
) -> Optional[str]:
    """One-shot wrapper over ``make_post_validator``. Returns the error
    string or None."""
    return make_post_validator(
        voiced,
        ledger,
        turn_beat_index=turn_beat_index,
        button_beat_index=button_beat_index,
        enforce_word_band=enforce_word_band,
    )(plan)


# ---------------------------------------------------------------------------
# Sentence splitting (SPLIT_LINE remedy)
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_into_units(text: str) -> List[str]:
    """Split a line into spoken units for SPLIT_LINE. Splits on sentence
    terminators first; if the text has none (one long run-on clause),
    falls back to a single mid-point split on a clause boundary
    (comma/semicolon/dash) so an over-cap line still divides. Always
    returns at least one non-empty fragment."""
    text = (text or "").strip()
    if not text:
        return [""]
    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) >= 2:
        return parts
    # No sentence boundary -- try a single split at the nearest clause
    # boundary to the midpoint so the two halves are roughly even.
    clause_marks = [m.start() for m in re.finditer(r"[,;:]|\s-\s", text)]
    if clause_marks:
        mid = len(text) // 2
        cut = min(clause_marks, key=lambda i: abs(i - mid))
        left = text[: cut + 1].strip().rstrip(",;:-").strip()
        right = text[cut + 1:].strip()
        out = [p for p in (left, right) if p]
        if len(out) >= 2:
            return out
    return [text]


# ---------------------------------------------------------------------------
# Apply step (pure Python) -- map edits back into the real ledger
# ---------------------------------------------------------------------------


def _stamp_line_text(row: Dict[str, Any], text: str) -> None:
    """Replace a line row's text in place + recompute the derived
    char_count / word_count in lockstep (mirrors
    production_ledger.update_line_text / _otr_ledger.patch_line_text so
    downstream budget consumers never read a stale count). Bindings are
    untouched."""
    set_line_text_metrics(row, text)


def _mark_realign(row: Dict[str, Any]) -> None:
    """Stamp the affected beat for cascade re-alignment. Tier-2 changed
    the dialogue-unit COUNT, so the FLUX -> HuMo -> LTX -> Bark timing
    for this span must be recomputed. Additive flag, default-absent
    everywhere else -- readers use ``row.get("needs_render_realign")``.
    """
    if isinstance(row, dict):
        row["needs_render_realign"] = True


def _clone_voiced_row(
    src: Dict[str, Any],
    *,
    suffix: str,
    text: str,
) -> Dict[str, Any]:
    """Build a NEW voiced line row that inherits every binding from
    ``src`` (so the split child stays bound to the same speaker / scene /
    slot family) with a fresh ``line_id`` derived from the source line id
    plus ``suffix`` while retaining the accepted parent ``beat_id``. The
    new row is stamped
    needs_render_realign (it is brand-new to the segment set).

    Binding inheritance is the render-safety guarantee: a SPLIT child
    speaks with the SAME char_id / voice (voice lives on cast, joined by
    char_id) / scene as its parent. We do NOT mint a new dialogue_slot_id
    -- the slot id is a stable voiced-slot key; a child carries the
    parent's slot id (downstream slot-join logic dedups on it) so we
    never invent a binding the cast contract has not seen.
    """
    new_row = dict(src)
    parent_beat_id = str(src.get("beat_id") or src.get("line_id") or "beat")
    parent_line_id = str(src.get("line_id") or parent_beat_id)
    new_id = f"{parent_line_id}{suffix}"
    new_row["line_id"] = new_id
    new_row["beat_id"] = parent_beat_id
    _stamp_line_text(new_row, text)
    _mark_realign(new_row)
    return new_row


def _sync_beat_line_ids(ledger: Dict[str, Any]) -> None:
    """Rebuild retained accepted beats' exact final line membership.

    Structural editing may cut, merge, or split line rows, but it never owns
    the accepted outline beat set. Existing beat rows are therefore retained
    verbatim and only their denormalized ``line_ids`` lists are refreshed.
    A fully cut beat truthfully remains with ``line_ids=[]``.
    """
    beats = ledger.get("beats") if isinstance(ledger, dict) else None
    if not isinstance(beats, list):
        return
    by_beat: Dict[str, List[str]] = {}
    for row in ledger.get("lines") or []:
        if not isinstance(row, dict):
            continue
        beat_id = str(row.get("beat_id") or "")
        line_id = str(row.get("line_id") or "")
        if not beat_id or not line_id:
            continue
        bucket = by_beat.setdefault(beat_id, [])
        if line_id not in bucket:
            bucket.append(line_id)
    for beat in beats:
        if not isinstance(beat, dict):
            continue
        beat_id = str(beat.get("beat_id") or "")
        beat["line_ids"] = list(by_beat.get(beat_id, []))


def apply_plan(
    ledger: Dict[str, Any],
    plan: RadioEditPlan,
) -> Dict[str, Any]:
    """Apply a validated ``RadioEditPlan`` to ``ledger`` IN PLACE and
    return a small report dict.

    Contract (story-spine Sec 3, Stream D apply step):
      * Tier 1 (KEEP / SHORTEN_LINE / CLEAN_PUNCTUATION /
        RECOMPOSE_BEAT_SAME_INTENT): replace the line's text in place;
        the segment set is byte-identical (same rows, same count, same
        bindings). KEEP is a no-op.
      * Tier 2 (CUT_LINE / REMOVE_REDUNDANT_BEAT / SPLIT_LINE /
        MERGE_SHORT_LINES): add/remove dialogue units, re-index the
        affected span cleanly, and stamp every touched row
        needs_render_realign.
      * Music beats + all bindings are preserved by construction --
        the editor only ever indexed the VOICED view, and this function
        only splices voiced rows; non-voiced rows are copied through
        untouched in their original positions.

    The caller is expected to have run ``post_validate_plan`` (or passed
    it as the ``structured_call`` post_validator) BEFORE applying. This
    function assumes a clean plan and does not re-validate; it does,
    however, skip a malformed edit defensively rather than corrupt the
    ledger.

    Returns:
      {
        "tier1_edits": int,
        "tier2_edits": int,
        "removed_line_ids": [..],
        "added_line_ids": [..],
        "realigned_line_ids": [..],
        "final_voiced_count": int,
      }

    Deprecated ``*_beat_ids`` aliases are retained for compatibility; they
    reference the same line-id lists and must not be interpreted as parent
    beat identities.
    """
    if not isinstance(ledger, dict):
        raise GuardError("apply_plan: ledger must be a dict")
    lines: List[Dict[str, Any]] = list(ledger.get("lines") or [])
    voiced_positions = _voiced_index_map({"lines": lines})
    voiced = [lines[i] for i in voiced_positions]
    n = len(voiced)
    used_line_ids = {
        str(row.get("line_id") or "")
        for row in lines
        if isinstance(row, dict) and str(row.get("line_id") or "")
    }

    removed_line_ids: List[str] = []
    added_line_ids: List[str] = []
    realigned_line_ids: List[str] = []
    report: Dict[str, Any] = {
        "tier1_edits": 0,
        "tier2_edits": 0,
        "removed_line_ids": removed_line_ids,
        "added_line_ids": added_line_ids,
        "realigned_line_ids": realigned_line_ids,
        # Backward-compatible aliases. These values have always identified
        # editable line rows, despite the historical field names.
        "removed_beat_ids": removed_line_ids,
        "added_beat_ids": added_line_ids,
        "realigned_beat_ids": realigned_line_ids,
        "final_voiced_count": 0,
    }

    # Index edits by voiced beat_index for a single deterministic pass
    # in ledger order. A MERGE is keyed on its primary index; the folded
    # partner is consumed.
    edits_by_index: Dict[int, BeatEdit] = {}
    for edit in plan.edits:
        if 0 <= edit.beat_index < n:
            edits_by_index[edit.beat_index] = edit

    # --- Pass 1: Tier-1 in-place text edits (segment set unchanged). ---
    # Done first so RECOMPOSE/SHORTEN text lands before any Tier-2
    # reindex moves rows around.
    for vi in range(n):
        edit = edits_by_index.get(vi)
        if edit is None:
            continue
        if edit.action == "KEEP":
            continue
        if edit.action == "CLEAN_PUNCTUATION":
            cleaned = _clean_punctuation(str(voiced[vi].get("text") or ""))
            _stamp_line_text(voiced[vi], cleaned)
            report["tier1_edits"] += 1
        elif edit.action in ("SHORTEN_LINE", "RECOMPOSE_BEAT_SAME_INTENT",
                             "SMOOTH_DIALOGUE", "CLARIFY_TURN"):
            _stamp_line_text(voiced[vi], str(edit.new_line or ""))
            report["tier1_edits"] += 1

    # --- Pass 2: Tier-2 structural edits (segment set changes). ---
    # Build the new VOICED list by walking the old one and applying
    # add/remove/split/merge, marking every product needs_render_realign.
    # Indices consumed by a MERGE partner are skipped.
    consumed: set = set()
    new_voiced: List[Dict[str, Any]] = []
    for vi in range(n):
        if vi in consumed:
            continue
        edit = edits_by_index.get(vi)
        row = voiced[vi]
        action = edit.action if edit is not None else "KEEP"

        if action in ("CUT_LINE", "REMOVE_REDUNDANT_BEAT"):
            report["tier2_edits"] += 1
            line_id = str(row.get("line_id") or "")
            report["removed_line_ids"].append(line_id)
            # Stamp the NEIGHBOR(s) that survive as needing realign --
            # removing a unit re-times whatever now abuts the gap.
            # (Handled below after the new list is built; here we just
            # drop the row.)
            continue

        if action == "SPLIT_LINE":
            report["tier2_edits"] += 1
            units = _split_into_units(str(edit.new_line or row.get("text") or ""))
            if len(units) < 2:
                # A split that did not actually divide degrades to an
                # in-place text replace (no count change), still marked
                # realign for safety.
                _stamp_line_text(row, units[0] if units else "")
                _mark_realign(row)
                new_voiced.append(row)
                report["realigned_line_ids"].append(
                    str(row.get("line_id") or "")
                )
                continue
            for k, unit in enumerate(units):
                if k == 0:
                    # First fragment reuses the original row id (stable
                    # head) so a downstream reader keyed on the original
                    # beat_id still finds the head of the split.
                    _stamp_line_text(row, unit)
                    _mark_realign(row)
                    new_voiced.append(row)
                    report["realigned_line_ids"].append(
                        str(row.get("line_id") or "")
                    )
                else:
                    # Repairs are deliberately multi-pass. If this parent was
                    # split in an earlier pass, never reuse its existing child
                    # id (for example b001_s1); advance deterministically to
                    # the first free suffix in the whole ledger namespace.
                    suffix_number = k
                    parent_line_id = str(row.get("line_id") or "beat")
                    candidate = f"{parent_line_id}_s{suffix_number}"
                    while candidate in used_line_ids:
                        suffix_number += 1
                        candidate = f"{parent_line_id}_s{suffix_number}"
                    child = _clone_voiced_row(
                        row,
                        suffix=f"_s{suffix_number}",
                        text=unit,
                    )
                    used_line_ids.add(str(child.get("line_id") or ""))
                    new_voiced.append(child)
                    report["added_line_ids"].append(str(child.get("line_id") or ""))
            continue

        if action == "MERGE_SHORT_LINES":
            report["tier2_edits"] += 1
            j = edit.merge_with_index
            # Guard 1 already proved j is adjacent + same speaker. Fold
            # the partner's text into this row; consume the partner.
            partner = voiced[j] if (j is not None and 0 <= j < n) else None
            if partner is None:
                # Defensive: treat as a plain text replace.
                _stamp_line_text(row, str(edit.new_line or row.get("text") or ""))
                _mark_realign(row)
                new_voiced.append(row)
                report["realigned_line_ids"].append(
                    str(row.get("line_id") or "")
                )
                continue
            merged_text = str(edit.new_line or "").strip()
            if not merged_text:
                a = str(row.get("text") or "").strip()
                b = str(partner.get("text") or "").strip()
                merged_text = (a + " " + b).strip()
            # The surviving row is the EARLIER of the two (lower index)
            # so ledger order is preserved; keep its id (a stable head).
            keep_row, drop_index = (row, j) if vi < j else (partner, vi)
            _stamp_line_text(keep_row, merged_text)
            _mark_realign(keep_row)
            new_voiced.append(keep_row)
            report["realigned_line_ids"].append(
                str(keep_row.get("line_id") or "")
            )
            report["removed_line_ids"].append(
                str(voiced[drop_index].get("line_id") or "")
            )
            consumed.add(j)
            continue

        # KEEP / Tier-1-already-applied rows pass through unchanged.
        new_voiced.append(row)

    # Mark the survivors adjacent to any removal as needing realign:
    # a deletion re-times whatever now abuts the gap. We detect this by
    # comparing the surviving voiced ids against the pre-edit neighbors
    # of removed indices.
    removed_ids = set(report["removed_line_ids"])
    if removed_ids:
        survivor_ids = [
            str(r.get("line_id") or "") for r in new_voiced
        ]
        # Any survivor whose original neighbor was removed gets stamped.
        for idx, r in enumerate(new_voiced):
            rid = survivor_ids[idx]
            if rid in report["realigned_line_ids"]:
                continue
            _mark_realign(r)
            report["realigned_line_ids"].append(rid)

    # --- Splice the new voiced list back into the full lines list,
    # preserving every non-voiced (music) row in its original slot.
    rebuilt = _splice_voiced(lines, voiced_positions, new_voiced)
    ledger["lines"] = rebuilt
    _sync_beat_line_ids(ledger)
    report["final_voiced_count"] = len(new_voiced)

    # Keep the ledger's denormalized totals coherent if present.
    _recompute_ledger_totals(ledger)
    return report


def _splice_voiced(
    original_lines: List[Dict[str, Any]],
    voiced_positions: List[int],
    new_voiced: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rebuild the full ``lines`` list from the (possibly resized) voiced
    list while keeping every non-voiced row in its original relative
    position.

    Strategy: walk the original list; emit non-voiced rows verbatim; at
    the position of the FIRST original voiced row, splice in the ENTIRE
    new voiced list; skip the remaining original voiced rows. This keeps
    music bookends around the dialogue block intact (open music ->
    dialogue -> inter -> dialogue -> close music) for the common case
    where voiced rows form contiguous runs, and is order-stable.
    """
    if not voiced_positions:
        return list(original_lines)
    voiced_pos_set = set(voiced_positions)
    first_voiced = voiced_positions[0]
    out: List[Dict[str, Any]] = []
    spliced = False
    for i, row in enumerate(original_lines):
        if i in voiced_pos_set:
            if not spliced:
                out.extend(new_voiced)
                spliced = True
            # else: skip subsequent original voiced rows (already
            # represented by new_voiced).
            continue
        out.append(row)
    if not spliced:
        out.extend(new_voiced)
    return out


def _recompute_ledger_totals(ledger: Dict[str, Any]) -> None:
    """Best-effort refresh of denormalized line totals if the ledger
    carries them. Never raises; a missing field is fine."""
    try:
        lines = ledger.get("lines") or []
        if "total_lines" in ledger:
            ledger["total_lines"] = len(lines)
        if "total_words" in ledger:
            ledger["total_words"] = sum(int(r.get("word_count") or 0) for r in lines)
    except Exception:  # noqa: BLE001 - totals are diagnostic, never fatal
        pass


_PUNCT_FIXES = (
    (re.compile(r"\s+"), " "),          # collapse runs of whitespace
    (re.compile(r"\s+([,.;:!?])"), r"\1"),  # no space before terminal punct
    (re.compile(r"([,;:])([^\s])"), r"\1 \2"),  # space after a clause mark
    (re.compile(r"\.{4,}"), "..."),     # cap ellipsis run
    (re.compile(r"([!?]){3,}"), r"\1\1"),  # cap bang/query runs
)


def _clean_punctuation(text: str) -> str:
    """Deterministic punctuation/whitespace normalization for
    CLEAN_PUNCTUATION. Does NOT change words -- only spacing and
    repeated punctuation. TTS-safety hygiene."""
    out = (text or "").strip()
    for pattern, repl in _PUNCT_FIXES:
        out = pattern.sub(repl, out)
    return out.strip()


# ---------------------------------------------------------------------------
# The editor entrypoint -- structured_call on the CREATIVE slot
# ---------------------------------------------------------------------------


def _default_repair_prompt(
    *,
    original_prompt: Any,
    failed_output: str,
    error: BaseException,
) -> Any:
    """Build the Attempt-3 typed-repair prompt for the editor plan. Names
    the deterministic guard violation (Guard 1/2/3 or the word-band) so
    the model fixes THAT, including the Guard-3 "you named a new visual
    noun X -- recompose without it" repair path (story-spine Sec 3:
    on a new noun, REPAIR, do not hard-FAIL)."""
    base = original_prompt if isinstance(original_prompt, str) else str(original_prompt)
    return (
        base
        + "\n\nYOUR PREVIOUS EDIT PLAN WAS REJECTED.\n"
        + f"Reason: {error}\n"
        + "Return a corrected RadioEditPlan that fixes exactly that "
        + "problem and changes nothing else. Do not introduce any new "
        + "visual noun (prop, place, weather, costume, machine, creature, "
        + "scenery) not already in the original beat or a cast portrait. "
        + "Keep every spoken line within the word and character caps, make "
        + "the edits' actual character-dialogue total land in the requested "
        + "target band, and never cut "
        + "the turning-point or button beat.\n"
        + "Output JSON only, matching the RadioEditPlan schema."
    )


def run_radio_editor(
    ledger: Dict[str, Any],
    *,
    editor_model: str,
    slot_fn: Callable[..., str],
    recompose_fn: Callable[..., str],
    editor_prompt: Optional[str] = None,
    base_temperature: float = 0.6,
    structural_retry_temperature: float = 0.3,
    max_new_tokens: int = 2048,
    turn_beat_index: Optional[int] = None,
    button_beat_index: Optional[int] = None,
    apply: bool = True,
    structured_call_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[RadioEditPlan, Dict[str, Any]]:
    """Run the Radio Editor over ``ledger`` and (optionally) apply the
    resulting plan.

    DORMANT-SAFE BY INJECTION. The two model-dependent callables are
    arguments, so this module + its self-test run with no GPU/network:
      * ``slot_fn`` -- the editor LLM generate callable, signature
        ``slot_fn(messages, *, temperature, max_new_tokens[, stop])
        -> str`` (the structured_call convention). In production the
        wiring binds this to the creative slot built from
        ``resolved["creative_writing_model"]``.
      * ``recompose_fn`` -- the per-beat recompose callable used to
        regenerate a flat beat under the RECOMPOSE fence. In production
        the wiring wraps ``_otr_line_composer.compose_line`` (which
        takes ``creative_fn=`` the creative slot and ``req=`` a
        LineRequest, returning a ``LineResult`` whose ``.text`` is the
        line). Signature here is intentionally light --
        ``recompose_fn(beat_index, original_text, hint) -> str`` -- so
        the wiring adapts compose_line to it and the self-test stubs it.

    ``editor_model`` is the model id string (creative slot); it is
    threaded into ``helper_name`` for slot attribution and is the proof
    that this module receives the id as a plain str argument, never via
    a widget (Prime Directive / D4).

    The structured-JSON plan is produced by ``structured_call`` (the one
    structured path -- no new parser, no new retry; story-spine Sec 2),
    validated by ``make_post_validator`` (Guards 1/2/3 + arc-anchor +
    word band), and applied by ``apply_plan`` when ``apply=True``.

    ``structured_call_fn`` is an injection seam for the self-test ONLY;
    production leaves it None and the real ``structured_call`` is lazily
    imported here so a pure-Python caller never pulls in pydantic-heavy
    siblings at import time.

    Returns ``(plan, apply_report)``. When ``apply=False`` the report is
    ``{}``.
    """
    if not isinstance(ledger, dict):
        raise GuardError("run_radio_editor: ledger must be a dict")

    voiced = voiced_beats(ledger)
    post_validator = make_post_validator(
        voiced,
        ledger,
        turn_beat_index=turn_beat_index,
        button_beat_index=button_beat_index,
    )

    prompt = editor_prompt if editor_prompt is not None else _build_editor_prompt(
        voiced, ledger
    )

    # Lazy import: the real structured_call lives in a sibling that
    # imports pydantic. Importing it inside the function keeps the
    # module's import surface light and lets the guards/apply run with no
    # ladder dependency. The self-test injects its own structured_call_fn
    # so it never reaches this import.
    if structured_call_fn is None:
        try:
            from ._otr_structured_call import structured_call as structured_call_fn  # type: ignore
        except ImportError:  # pragma: no cover - standalone / test load
            from _otr_structured_call import structured_call as structured_call_fn  # type: ignore

    # LLM slot: creative
    # Reason: the editor reshapes spoken dialogue and owns episode length
    # / radio pacing -- a narrative pass -- so it routes to the creative
    # slot per project rule 6. editor_model is the creative-slot id.
    plan: RadioEditPlan = structured_call_fn(
        prompt=prompt,
        schema=RadioEditPlan,
        slot_fn=slot_fn,
        base_temperature=base_temperature,
        structural_retry_temperature=structural_retry_temperature,
        repair_prompt_factory=_default_repair_prompt,
        post_validator=post_validator,
        max_new_tokens=max_new_tokens,
        helper_name=f"radio_editor[{editor_model}]",
    )

    # RECOMPOSE_BEAT_SAME_INTENT: the editor named the beats to rewrite;
    # the actual line text is regenerated by the injected recompose_fn
    # (production: compose_line on the creative slot). We re-run any
    # recompose whose new_line is missing OR whose new_line tripped a
    # repairable Guard-3 new-noun violation, naming the offending noun in
    # the hint so the recompose avoids it (story-spine: REPAIR, do not
    # hard-FAIL). The plan that reached here already passed the
    # post_validator, so its new_line values are guard-clean; this loop
    # exists so the wiring CAN delegate text generation to compose_line
    # rather than trusting the plan's inline prose.
    # LLM slot: creative
    # Reason: recompose regenerates a flat beat's spoken line -- narrative
    # generation -- on the creative slot per project rule 6.
    plan = _resolve_recompose_text(plan, voiced, ledger, recompose_fn)

    apply_report: Dict[str, Any] = {}
    if apply:
        apply_report = apply_plan(ledger, plan)
    return plan, apply_report


def _resolve_recompose_text(
    plan: RadioEditPlan,
    voiced: List[Dict[str, Any]],
    ledger: Dict[str, Any],
    recompose_fn: Callable[..., str],
) -> RadioEditPlan:
    """For each RECOMPOSE_BEAT_SAME_INTENT edit, obtain the final line
    text via ``recompose_fn`` when the plan did not supply usable prose,
    and re-check Guard 3, repairing once by re-running the recompose with
    the offending noun named in the hint. Returns a plan whose recompose
    new_line values are final + guard-clean. Pure orchestration around
    the injected callable -- no model dependency in this module."""
    new_edits: List[BeatEdit] = []
    for edit in plan.edits:
        if edit.action != "RECOMPOSE_BEAT_SAME_INTENT":
            new_edits.append(edit)
            continue
        original = str(voiced[edit.beat_index].get("text") or "")
        text = edit.new_line
        # Generate via recompose_fn if the plan left it empty.
        if text is None or not str(text).strip():
            text = recompose_fn(edit.beat_index, original, "")
        # Guard 3 repair-once: if the recomposed text names a new visual
        # noun, re-run naming it, then accept whatever comes back (the
        # single repair; downstream post-apply gate is the final word).
        offenders = new_visual_nouns(str(text or ""), original, ledger)
        if offenders:
            hint = (
                "Do not mention "
                + ", ".join(offenders)
                + " -- keep only props, places, and people already in the "
                + "beat."
            )
            text = recompose_fn(edit.beat_index, original, hint)
        new_edits.append(
            edit.model_copy(update={"new_line": str(text or "")})
        )
    return plan.model_copy(update={"edits": new_edits})


def _build_editor_prompt(
    voiced: List[Dict[str, Any]],
    ledger: Dict[str, Any],
) -> str:
    """Render the editor's instruction prompt: the dialogue units it may
    edit (by voiced index + speaker + text) and the rules. The prompt
    DESCRIBES the contract; the schema + guards ENFORCE it (story-spine
    Sec 2, invariant 1). Kept deterministic + model-agnostic (no model
    branch)."""
    rows: List[str] = []
    for i, beat in enumerate(voiced):
        spk = (
            beat.get("char_id")
            or beat.get("speaker_role")
            or "?"
        )
        text = str(beat.get("text") or "").replace("\n", " ").strip()
        rows.append(f"[{i}] ({spk}) {text}")
    listing = "\n".join(rows) if rows else "(no voiced beats)"
    budget, _budget_error = _resolve_word_budget(ledger)
    budget = budget or _fallback_word_budget()
    return (
        "You are a radio script editor. Tighten the episode character "
        f"dialogue to about {budget.target_words} words while keeping the story, the "
        "speakers, and the order intact.\n\n"
        "DIALOGUE UNITS (edit by index; never reassign a speaker):\n"
        + listing
        + "\n\nReturn a RadioEditPlan: one BeatEdit per unit you change. "
        "Tier 1 keeps the unit count (KEEP, SHORTEN_LINE, "
        "CLEAN_PUNCTUATION, RECOMPOSE_BEAT_SAME_INTENT). Tier 2 changes "
        "it (CUT_LINE, REMOVE_REDUNDANT_BEAT, SPLIT_LINE, "
        "MERGE_SHORT_LINES). Keep every line to one breath "
        f"(<= {MAX_WORDS_PER_LINE} words). Do not introduce any new prop, "
        "place, weather, costume, machine, creature, or scenery. Never cut "
        "the turning point or the final payoff. Announcer and music overhead "
        "do not count toward the requested character-dialogue budget. Set "
        "projected_word_total to your estimate of the new character-dialogue "
        "total. Output JSON only."
    )


# ---------------------------------------------------------------------------
# Two-entry contract (go-forward Sprint 2): normalize_length + micro_repair
# ---------------------------------------------------------------------------
#
# The editor is one module with TWO entrypoints, each a thin specialization
# of the same machinery (structured_call -> Guards 1/2/3 + arc + band ->
# apply_plan). They differ only in (a) when they run, (b) which actions are
# allowed, (c) which beats are in scope, and (d) the prompt. All edits are
# still proposed by the LLM as a RadioEditPlan and applied deterministically
# by Python -- the ledger stays read-only to the model (Sec 2 invariant 3).

# normalize_length owns episode LENGTH: Tier-1 in-place tightening + Tier-2
# count changes. It is NOT a quality pass -- it never SMOOTHs, CLARIFYs, or
# RECOMPOSEs (those regenerate prose; a length pass only tightens / cuts /
# splits / merges).
NORMALIZE_LENGTH_ACTIONS: frozenset = frozenset(
    {
        "KEEP",
        "SHORTEN_LINE",
        "CLEAN_PUNCTUATION",
        "CUT_LINE",
        "REMOVE_REDUNDANT_BEAT",
        "SPLIT_LINE",
        "MERGE_SHORT_LINES",
    }
)

# micro_repair owns beat-local quality fixes on the QA-flagged beats:
# tighten, split an over-cap line, clean punctuation, smooth a chopped line,
# clarify a muddy turn. It never deletes or merges a beat (no CUT / REMOVE /
# MERGE) -- a one-cycle, flagged-only fix, with SPLIT_LINE the single
# count-changing remedy for an over-long flagged line.
MICRO_REPAIR_ACTIONS: frozenset = frozenset(
    {
        "KEEP",
        "SHORTEN_LINE",
        "SPLIT_LINE",
        "CLEAN_PUNCTUATION",
        "SMOOTH_DIALOGUE",
        "CLARIFY_TURN",
    }
)


def _make_scoped_validator(
    base_validator: Callable[[RadioEditPlan], Optional[str]],
    *,
    allowed_actions: frozenset,
    allowed_beat_indices: Optional[set] = None,
) -> Callable[[RadioEditPlan], Optional[str]]:
    """Wrap a base post_validator with an action whitelist and an optional
    beat-scope fence, both checked BEFORE the base guards.

    ``allowed_actions`` -- only these actions may appear (the entrypoint's
    action set; an out-of-set action is rejected, advancing the ladder).
    ``allowed_beat_indices`` -- when not None, every edit's ``beat_index``
    (and any ``merge_with_index``) must be in this set. This is the
    micro-repair beat-local fence: the pass may touch ONLY the QA-flagged
    beats, never a clean one. Returns an error string to reject or None to
    accept; never raises (the structured_call post_validator contract)."""

    def _validate(plan: RadioEditPlan) -> Optional[str]:
        for edit in plan.edits:
            if edit.action not in allowed_actions:
                return (
                    f"scoped_validator: action {edit.action!r} on beat_index "
                    f"{edit.beat_index} is not permitted in this pass; "
                    f"allowed actions are {sorted(allowed_actions)}"
                )
            if allowed_beat_indices is not None:
                if edit.beat_index not in allowed_beat_indices:
                    return (
                        f"scoped_validator: beat_index {edit.beat_index} is "
                        f"not in the flagged-beat set "
                        f"{sorted(allowed_beat_indices)}; this pass may edit "
                        f"only the flagged beats"
                    )
                if (
                    edit.merge_with_index is not None
                    and edit.merge_with_index not in allowed_beat_indices
                ):
                    return (
                        f"scoped_validator: merge_with_index "
                        f"{edit.merge_with_index} is not in the flagged-beat "
                        f"set; a merge may only fold a flagged beat"
                    )
        return base_validator(plan)

    return _validate


def needs_length_normalization(ledger: Dict[str, Any]) -> bool:
    """True when the draft is OUT OF SPEC, so the length pass should run.

    Out of spec means the non-skipped character-dialogue total is outside the
    ledger's requested ``meta.word_budget`` ratio band, OR any single voiced
    character/announcer line exceeds the per-line cap (word OR char). On an
    already-clean draft this returns False and ``normalize_length`` is a no-op
    -- the editor LLM is never paid for on a draft already in spec.

    Never raises. A malformed present budget returns True only so the caller
    enters ``normalize_length`` and receives its explicit non-mutating
    ``SKIPPED_INVALID_BUDGET`` receipt."""
    try:
        safe_ledger = ledger if isinstance(ledger, dict) else {}
        budget, budget_error = _resolve_word_budget(safe_ledger)
        if budget_error is not None or not isinstance(budget, _WordBudgetSpec):
            return True
        voiced = voiced_beats(safe_ledger)
        if not word_total_in_band(_budgeted_word_total(safe_ledger), budget):
            return True
        for beat in voiced:
            if line_over_cap(str(beat.get("text") or "")) is not None:
                return True
        return False
    except Exception:  # noqa: BLE001 -- a precondition probe must never break a run
        return False


def _build_length_prompt(
    voiced: List[Dict[str, Any]],
    ledger: Dict[str, Any],
) -> str:
    """Episode-length normalizer prompt: tighten the whole episode toward
    the target band using length levers only (no smoothing / clarifying /
    recomposing -- those are micro-repair quality edits, fenced out of this
    pass by the scoped validator)."""
    rows: List[str] = []
    for i, beat in enumerate(voiced):
        spk = beat.get("char_id") or beat.get("speaker_role") or "?"
        text = str(beat.get("text") or "").replace("\n", " ").strip()
        rows.append(f"[{i}] ({spk}) {text}")
    listing = "\n".join(rows) if rows else "(no voiced beats)"
    budget, _budget_error = _resolve_word_budget(ledger)
    budget = budget or _fallback_word_budget()
    total = _budgeted_word_total(ledger)
    lo = math.ceil(budget.lower_words)
    hi = math.floor(budget.upper_words)
    return (
        "You are a radio script editor. The episode is out of spec on "
        f"length or pacing (current character-dialogue total {total} words; "
        f"requested target {budget.target_words}, accepted band [{lo}, {hi}]). "
        "Tighten it toward the target "
        "while keeping the story, the speakers, and the order intact.\n\n"
        "DIALOGUE UNITS (edit by index; never reassign a speaker):\n"
        + listing
        + "\n\nReturn a RadioEditPlan. Use length levers only: KEEP, "
        "SHORTEN_LINE, CLEAN_PUNCTUATION (Tier 1, same beat count); "
        "CUT_LINE, REMOVE_REDUNDANT_BEAT, SPLIT_LINE, MERGE_SHORT_LINES "
        "(Tier 2, changes the beat count). Keep every line to one breath "
        f"(<= {MAX_WORDS_PER_LINE} words). Do not introduce any new prop, "
        "place, weather, costume, machine, creature, or scenery. Never cut "
        "the turning point or the final payoff. Announcer and music overhead "
        "are excluded from the requested budget. Set projected_word_total to "
        f"your estimate of the new character-dialogue total (aim for "
        f"{budget.target_words}). "
        "Output JSON only."
    )


def _build_micro_repair_prompt(
    voiced: List[Dict[str, Any]],
    ledger: Dict[str, Any],
    flagged: List[int],
    repair_context: str = "",
) -> str:
    """Beat-local micro-repair prompt: show every voiced unit for context,
    MARK the flagged beats, and instruct an edit ONLY on the marked ones.
    The scoped validator enforces the flagged-beat fence regardless of what
    the model returns, so this prompt only has to guide it."""
    flagged_set = set(flagged)
    rows: List[str] = []
    for i, beat in enumerate(voiced):
        spk = beat.get("char_id") or beat.get("speaker_role") or "?"
        text = str(beat.get("text") or "").replace("\n", " ").strip()
        mark = "   <== FIX THIS BEAT" if i in flagged_set else ""
        rows.append(f"[{i}] ({spk}) {text}{mark}")
    listing = "\n".join(rows) if rows else "(no voiced beats)"
    flagged_str = ", ".join(str(i) for i in flagged) if flagged else "(none)"
    total = _budgeted_word_total(ledger)
    context = " ".join(str(repair_context or "").split())[:1200]
    feedback = (
        "\n\nQUALITY JUDGE FEEDBACK (resolve this exact concern):\n"
        + context
        if context else ""
    )
    return (
        "You are a radio script editor doing a SINGLE round of small, "
        "beat-local repairs. Only the marked beats have a problem; leave "
        "every other beat exactly as it is.\n\n"
        f"DIALOGUE UNITS (edit ONLY the marked indices {flagged_str}):\n"
        + listing
        + "\n\nReturn a RadioEditPlan with one BeatEdit per MARKED beat "
        "only. Allowed actions: SHORTEN_LINE (tighten), SPLIT_LINE (divide "
        "an over-long line), CLEAN_PUNCTUATION, SMOOTH_DIALOGUE (make a "
        "chopped or stilted line read naturally), CLARIFY_TURN (make a "
        "muddy decision or reversal land clearly). Keep every line to one "
        f"breath (<= {MAX_WORDS_PER_LINE} words). Do NOT introduce any new "
        "prop, place, weather, costume, machine, creature, or scenery, and "
        "never reassign a speaker. Do not edit any unmarked beat. Set "
        "projected_word_total to your estimate of the new character-dialogue "
        "total (it should "
        f"stay near {total}). Output JSON only."
        + feedback
    )


def normalize_length(
    ledger: Dict[str, Any],
    *,
    editor_model: str,
    slot_fn: Callable[..., str],
    recompose_fn: Callable[..., str],
    base_temperature: float = 0.6,
    structural_retry_temperature: float = 0.3,
    max_new_tokens: int = 2048,
    turn_beat_index: Optional[int] = None,
    button_beat_index: Optional[int] = None,
    apply: bool = True,
    structured_call_fn: Optional[Callable[..., Any]] = None,
) -> Tuple[Optional[RadioEditPlan], Dict[str, Any]]:
    """Conditional episode-length normalizer (go-forward Sprint 2, entry 1).

    Runs the editor ONLY when ``needs_length_normalization(ledger)`` is True
    (character dialogue outside the requested receipt band OR any voiced line
    over the per-line cap). On an in-spec draft it is a NO-OP: returns ``(None, {"status":
    "SKIPPED_IN_SPEC", ...})`` with no LLM call -- the editor is never paid
    for on a clean draft.

    Action set = ``NORMALIZE_LENGTH_ACTIONS`` (Tier-1 tighten + Tier-2 count
    change); the quality actions (SMOOTH_DIALOGUE / CLARIFY_TURN /
    RECOMPOSE) are fenced out by the scoped validator. All three guards +
    the arc-anchor protection + the word band apply; Python applies the plan
    deterministically (apply_plan). Lands the total in band and every line
    under the spoken cap.

    Injection seams (``slot_fn`` / ``recompose_fn`` / ``structured_call_fn``)
    match ``run_radio_editor`` so the module + its self-test run with no
    GPU / network. ``editor_model`` is the creative-slot id (threaded into
    ``helper_name`` for attribution; this module exposes no model_id widget).

    Returns ``(plan, apply_report)``. ``plan`` is None on the skip path; the
    report carries a ``status`` of SKIPPED_IN_SPEC / APPLIED / PLANNED.
    """
    if not isinstance(ledger, dict):
        raise GuardError("normalize_length: ledger must be a dict")

    budget, budget_error = _resolve_word_budget(ledger)
    if budget_error is not None or not isinstance(budget, _WordBudgetSpec):
        log.warning(
            "[OTR_RadioEditor] length mutation skipped: invalid word budget (%s)",
            budget_error or "unknown error",
        )
        return None, {
            "status": "SKIPPED_INVALID_BUDGET",
            "error": budget_error or "invalid word budget",
            "tier1_edits": 0,
            "tier2_edits": 0,
        }

    if not needs_length_normalization(ledger):
        return None, {
            "status": "SKIPPED_IN_SPEC",
            "tier1_edits": 0,
            "tier2_edits": 0,
        }

    voiced = voiced_beats(ledger)
    base_validator = make_post_validator(
        voiced,
        ledger,
        turn_beat_index=turn_beat_index,
        button_beat_index=button_beat_index,
        enforce_word_band=True,
    )
    post_validator = _make_scoped_validator(
        base_validator, allowed_actions=NORMALIZE_LENGTH_ACTIONS
    )
    prompt = _build_length_prompt(voiced, ledger)

    if structured_call_fn is None:
        try:
            from ._otr_structured_call import structured_call as structured_call_fn  # type: ignore
        except ImportError:  # pragma: no cover - standalone / test load
            from _otr_structured_call import structured_call as structured_call_fn  # type: ignore

    # LLM slot: creative
    # Reason: the length pass reshapes spoken dialogue and owns episode
    # length / radio pacing -- a narrative pass -- so it routes to the
    # creative slot per project rule 6. editor_model is the creative-slot id.
    plan: RadioEditPlan = structured_call_fn(
        prompt=prompt,
        schema=RadioEditPlan,
        slot_fn=slot_fn,
        base_temperature=base_temperature,
        structural_retry_temperature=structural_retry_temperature,
        repair_prompt_factory=_default_repair_prompt,
        post_validator=post_validator,
        max_new_tokens=max_new_tokens,
        helper_name=f"normalize_length[{editor_model}]",
    )
    plan = _resolve_recompose_text(plan, voiced, ledger, recompose_fn)

    report: Dict[str, Any] = {"status": "PLANNED"}
    if apply:
        report = apply_plan(ledger, plan)
        report["status"] = "APPLIED"
        report["projected_word_total"] = plan.projected_word_total
        report["actual_word_total"] = _budgeted_word_total(ledger)
        report["target_word_total"] = budget.target_words
        report["word_band"] = [budget.min_ratio, budget.max_ratio]
        report["word_budget_source"] = budget.source
    return plan, report


# ---------------------------------------------------------------------------
# Final inline-producer word delivery
# ---------------------------------------------------------------------------


class FinalWordDeliveryError(GuardError):
    """Inline story rows could not satisfy the explicit delivery contract."""


_FINAL_WORD_FIT_FAKE_AD_RE = re.compile(
    r"\b(?:commercial break|sponsored by|our sponsor|buy now|call now|"
    r"order now|limited[- ]time offer|available wherever|available now|"
    r"new product|try the new|brought to you by)\b",
    re.IGNORECASE,
)
_FINAL_WORD_FIT_NUMERAL_RE = re.compile(
    r"\b(?:\d[\d,.]*|zero|one|two|three|four|five|six|seven|eight|nine|ten|"
    r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
    r"nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    r"hundred|thousand|million|billion)\b",
    re.IGNORECASE,
)
_FINAL_WORD_FIT_TEMPERATURES = (0.38, 0.30, 0.24, 0.18, 0.14)
# The final spoken-hygiene authority uses the composer's one-breath law. Keep
# the fitter's target honest so it does not pay for candidates that the next
# deterministic scan must reject (the broader Radio Editor structural cap is
# intentionally not the spoken craft cap).
_FINAL_WORD_FIT_MAX_LINE_WORDS = 24


def _word_fit_sha256(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def _word_fit_number_tokens(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _FINAL_WORD_FIT_NUMERAL_RE.findall(str(text or ""))
    }


def _word_fit_speaker_labels(ledger: Dict[str, Any]) -> tuple[str, ...]:
    labels = {"ANNOUNCER"}
    for row in ledger.get("cast") or ():
        if not isinstance(row, dict):
            continue
        for key in ("name", "char_id"):
            value = str(row.get(key) or "").strip()
            if value:
                labels.add(value)
    return tuple(sorted(labels, key=lambda item: (-len(item), item)))


def _word_fit_candidate_error(
    *,
    raw: str,
    candidate: str,
    current_text: str,
    req: Any,
    ledger: Dict[str, Any],
    story_rules: Any,
) -> Optional[str]:
    """Validate one proposed line without granting it any ledger authority."""
    if not candidate or not re.search(r"[A-Za-z]", candidate):
        return "replacement must contain spoken words"
    if candidate == str(current_text or "").strip():
        return "replacement must change the target line"
    if re.search(r"[\[\]{}*]", raw) or "(" in raw or ")" in raw:
        return "replacement cannot contain markup or stage directions"
    for label in _word_fit_speaker_labels(ledger):
        if re.match(rf"^\s*{re.escape(label)}\s*:", raw, re.IGNORECASE):
            return "replacement cannot contain a speaker label"
    if _FINAL_WORD_FIT_FAKE_AD_RE.search(candidate):
        return "replacement cannot introduce a fake commercial or product"
    prior_numbers = _word_fit_number_tokens(current_text)
    new_numbers = _word_fit_number_tokens(candidate)
    if not new_numbers.issubset(prior_numbers):
        return "replacement cannot introduce a new numeric claim"
    locked_story_corpus = " ".join(
        str(row.get("text") or "") + " " + str(row.get("beat_intent") or "")
        for row in ledger.get("lines") or ()
        if isinstance(row, dict)
    ) + " " + str(getattr(req, "canon_header", "") or "")
    visual_nouns = new_visual_nouns(candidate, locked_story_corpus, ledger)
    if visual_nouns:
        return "replacement cannot introduce new visual nouns: " + ", ".join(
            visual_nouns[:8]
        )
    try:
        from ._otr_line_composer import spoken_surface_flags_for_line
    except ImportError:  # pragma: no cover - standalone / test load
        from _otr_line_composer import (  # type: ignore
            spoken_surface_flags_for_line,
        )
    flags = spoken_surface_flags_for_line(
        candidate, req, rules=story_rules, include_thesis=False,
    )
    if flags:
        return "replacement failed spoken hygiene: " + ", ".join(
            str(flag) for flag in flags[:8]
        )
    return None


def _word_fit_call(
    slot_fn: Callable[..., str],
    *,
    messages: list[dict[str, str]],
    temperature: float,
) -> str:
    try:
        return str(slot_fn(
            messages,
            temperature=temperature,
            max_new_tokens=256,
            stop=["\n\n", "```"],
        ) or "")
    except TypeError:
        return str(slot_fn(
            messages,
            temperature=temperature,
            max_new_tokens=256,
        ) or "")


def fit_final_word_delivery(
    ledger: Dict[str, Any],
    *,
    creative_fn: Callable[..., str],
    technical_fn: Callable[..., str],
    creative_model: str,
    technical_model: str,
    source_bank_id: str,
    canon_header: str = "",
) -> Dict[str, Any]:
    """Fit final inline character speech via finite one-row authored repairs.

    Python selects exactly one writable row per cycle. A creative pass and then
    a technical fallback may propose replacement prose, but only a candidate
    that preserves every binding, passes spoken/visual/fact hygiene, and makes
    strict progress toward the persisted common word band is adopted. No
    deterministic filler is ever added.
    """
    if not isinstance(ledger, dict):
        raise FinalWordDeliveryError("fit_final_word_delivery: ledger must be a dict")
    try:
        spec = _OTRWD.resolve_spec(ledger)
    except _OTRWD.WordDeliveryError as exc:
        raise FinalWordDeliveryError(str(exc)) from exc
    if not spec.owner.startswith("inline:"):
        raise FinalWordDeliveryError(
            f"inline word fitter cannot mutate producer-owned receipt {spec.owner!r}"
        )

    try:
        from . import _otr_reroll as _OTRRR_WORD_FIT
        from ._otr_line_composer import strip_line_formatting
        from ._otr_story_rules import get_story_rules
    except ImportError:  # pragma: no cover - standalone / test load
        import _otr_reroll as _OTRRR_WORD_FIT  # type: ignore
        from _otr_line_composer import strip_line_formatting  # type: ignore
        from _otr_story_rules import get_story_rules  # type: ignore

    meta = ledger.setdefault("meta", {})
    story_rules = get_story_rules(meta)
    actual = _OTRWD.character_word_count(ledger)
    initial_actual = actual
    repair_budget = _OTRWD.delivery_repair_cycle_budget(
        actual_words=actual,
        target_words=spec.target_words,
        lower_words=spec.lower_words,
        upper_words=spec.upper_words,
    )
    cycles: list[dict[str, Any]] = []
    terminal_status = "pass" if spec.contains(actual) else "repair_exhausted"

    for cycle in range(1, repair_budget + 1):
        if spec.contains(actual):
            terminal_status = "pass"
            break
        under = actual < spec.lower_words
        rows = [
            row for row in ledger.get("lines") or ()
            if isinstance(row, dict)
            and not row.get("skip")
            and str(row.get("speaker_role") or "") == "character"
            and (
                canonical_word_count(row.get("text"))
                < _FINAL_WORD_FIT_MAX_LINE_WORDS
                if under else canonical_word_count(row.get("text")) > 1
            )
        ]
        if not rows:
            terminal_status = "no_actionable_rows"
            break
        ranked = sorted(
            rows,
            key=(
                lambda row: (
                    canonical_word_count(row.get("text")),
                    str(row.get("line_id") or ""),
                )
                if under else (
                    -canonical_word_count(row.get("text")),
                    str(row.get("line_id") or ""),
                )
            ),
        )
        row = ranked[(cycle - 1) % len(ranked)]
        line_id = str(row.get("line_id") or row.get("beat_id") or "")
        current_text = str(row.get("text") or "").strip()
        current_line_words = canonical_word_count(current_text)
        distance = (
            spec.lower_words - actual if under else actual - spec.upper_words
        )
        correction = min(
            distance, _OTRWD.delivery_step_words(spec.target_words),
        )
        desired_line_words = (
            min(_FINAL_WORD_FIT_MAX_LINE_WORDS, current_line_words + correction)
            if under else max(1, current_line_words - correction)
        )
        req = _OTRRR_WORD_FIT.build_reroll_line_request(
            ledger, row, ledger.get("cast") or [],
        )
        req = replace(
            req,
            canon_header=str(canon_header or req.canon_header or ""),
            target_words=desired_line_words,
            beat_objective=(
                req.beat_objective
                or str(row.get("beat_intent") or "").strip()
            ),
        )
        all_character_rows = [
            item for item in ledger.get("lines") or ()
            if isinstance(item, dict)
            and not item.get("skip")
            and str(item.get("speaker_role") or "") == "character"
        ]
        row_pos = next(
            (i for i, item in enumerate(all_character_rows) if item is row), 0,
        )
        context_rows = all_character_rows[max(0, row_pos - 1):row_pos + 2]
        context = "\n".join(
            f"{str(item.get('char_id') or item.get('speaker_role') or '?')}: "
            f"{str(item.get('text') or '').strip()}"
            for item in context_rows
        )
        direction = "expand" if under else "tighten"
        messages = [
            {
                "role": "system",
                "content": (
                    "Repair exactly one SFW radio-play line. Return only the "
                    "replacement dialogue, with no label, cue, markup, or "
                    "explanation. Preserve the speaker, beat intent, locked "
                    "facts, conflict, and visual world. Use natural spoken "
                    "dialogue; never add filler, a named or numeric claim, a "
                    "fake commercial, product, sponsor message, or moral summary. "
                    "Reuse content vocabulary already present in the locked "
                    "canon and context."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Source bank: {source_bank_id}\n"
                    f"Writable line_id: {line_id}\n"
                    f"Speaker: {req.speaker}\n"
                    f"Beat intent: {req.intent}\n"
                    f"Direction: {direction}\n"
                    f"Current line words: {current_line_words}\n"
                    f"Desired line words: about {desired_line_words}, maximum "
                    f"{_FINAL_WORD_FIT_MAX_LINE_WORDS}\n"
                    f"Episode character words: {actual}; required band: "
                    f"{spec.lower_words}..{spec.upper_words}\n"
                    f"Locked canon:\n{str(req.canon_header or '')[-1200:]}\n"
                    f"Locked local context:\n{context}\n"
                    f"Current writable line:\n{current_text}"
                ),
            },
        ]
        cycle_receipt: Dict[str, Any] = {
            "cycle": cycle,
            "line_id": line_id,
            "direction": direction,
            "before_character_words": actual,
            "current_line_words": current_line_words,
            "desired_line_words": desired_line_words,
            "attempts": [],
            "adopted": False,
        }
        for slot_name, slot_fn, model_id, temperature in (
            (
                "creative", creative_fn, creative_model,
                _FINAL_WORD_FIT_TEMPERATURES[
                    (cycle - 1) % len(_FINAL_WORD_FIT_TEMPERATURES)
                ],
            ),
            ("technical", technical_fn, technical_model, 0.16),
        ):
            attempt: Dict[str, Any] = {
                "slot": slot_name,
                "model_id": str(model_id or ""),
                "status": "rejected",
            }
            try:
                # LLM slot: per-sub-pass -- creative author, technical fallback.
                raw = _word_fit_call(
                    slot_fn, messages=messages, temperature=temperature,
                )
                attempt["raw_sha256"] = _word_fit_sha256(raw)
                candidate = strip_line_formatting(raw or "").strip()
                attempt["candidate_words"] = canonical_word_count(candidate)
                error = _word_fit_candidate_error(
                    raw=raw,
                    candidate=candidate,
                    current_text=current_text,
                    req=req,
                    ledger=ledger,
                    story_rules=story_rules,
                )
                if error is not None:
                    attempt["error"] = error
                    cycle_receipt["attempts"].append(attempt)
                    continue
                candidate_actual = (
                    actual - current_line_words
                    + canonical_word_count(candidate)
                )
                prior_distance = _OTRWD.word_band_distance(
                    actual, spec.lower_words, spec.upper_words,
                )
                candidate_distance = _OTRWD.word_band_distance(
                    candidate_actual, spec.lower_words, spec.upper_words,
                )
                if candidate_distance >= prior_distance:
                    attempt["error"] = "candidate made no strict progress"
                    attempt["candidate_character_words"] = candidate_actual
                    cycle_receipt["attempts"].append(attempt)
                    continue
                flags = list(row.get("compose_flags") or [])
                if "final_word_fit" not in flags:
                    flags.append("final_word_fit")
                set_line_text_metrics(row, candidate)
                row["compose_flags"] = flags
                actual = candidate_actual
                attempt["status"] = "accepted"
                attempt["candidate_character_words"] = candidate_actual
                cycle_receipt["attempts"].append(attempt)
                cycle_receipt["adopted"] = True
                cycle_receipt["accepted_slot"] = slot_name
                break
            except Exception as exc:  # noqa: BLE001 -- alternate slot continues
                attempt["error"] = (
                    f"{type(exc).__name__}: {' '.join(str(exc).split())[:240]}"
                )
                cycle_receipt["attempts"].append(attempt)
        cycle_receipt["after_character_words"] = actual
        cycle_receipt["distance_to_band"] = _OTRWD.word_band_distance(
            actual, spec.lower_words, spec.upper_words,
        )
        cycles.append(cycle_receipt)
        if spec.contains(actual):
            terminal_status = "pass"
            break

    actual = _OTRWD.character_word_count(ledger)
    receipt = {
        "status": terminal_status if spec.contains(actual) else terminal_status,
        "owner": spec.owner,
        "source_bank": str(source_bank_id or ""),
        "target_words": spec.target_words,
        "acceptance_words": [spec.lower_words, spec.upper_words],
        "initial_character_words": initial_actual,
        "final_character_words": actual,
        "repair_budget": repair_budget,
        "cycles": cycles,
        "actual_drift": not spec.contains(actual),
    }
    meta["inline_word_fit"] = receipt
    try:
        _OTRWD.stamp_actual(
            ledger,
            stage=(
                "inline_word_fit" if spec.contains(actual)
                else "inline_word_fit_exhausted"
            ),
            require_in_band=spec.contains(actual),
        )
    except _OTRWD.WordDeliveryError as exc:
        raise FinalWordDeliveryError(str(exc)) from exc
    if not spec.contains(actual):
        raise FinalWordDeliveryError(
            f"{spec.owner} word delivery exhausted {repair_budget} row-local "
            f"cycles: required {spec.lower_words}..{spec.upper_words}, "
            f"actual {actual}"
        )
    return receipt


def micro_repair(
    ledger: Dict[str, Any],
    flagged_beats: List[int],
    *,
    editor_model: str,
    slot_fn: Callable[..., str],
    recompose_fn: Callable[..., str],
    base_temperature: float = 0.6,
    structural_retry_temperature: float = 0.3,
    max_new_tokens: int = 1024,
    turn_beat_index: Optional[int] = None,
    button_beat_index: Optional[int] = None,
    apply: bool = True,
    structured_call_fn: Optional[Callable[..., Any]] = None,
    repair_context: str = "",
) -> Tuple[Optional[RadioEditPlan], Dict[str, Any]]:
    """Beat-local micro-repair on the QA-flagged beats (go-forward Sprint 2,
    entry 2). ONE cycle, flagged beats ONLY -- a scoped validator rejects
    any edit whose ``beat_index`` is not in ``flagged_beats``, so a clean
    beat is never touched. Action set = ``MICRO_REPAIR_ACTIONS`` (SHORTEN /
    SPLIT / CLEAN_PUNCTUATION / SMOOTH_DIALOGUE / CLARIFY_TURN). The three
    guards + arc protection apply; Guard 3 fences the fresh prose that
    SMOOTH / CLARIFY emit, so no micro-repair can introduce a new visual
    noun.

    NO-OP (no LLM call) when ``flagged_beats`` is empty or names no valid
    voiced index. Returns ``(plan, apply_report)``; ``plan`` is None on the
    skip path and the report ``status`` is SKIPPED_NO_BEATS / APPLIED /
    PLANNED."""
    if not isinstance(ledger, dict):
        raise GuardError("micro_repair: ledger must be a dict")

    voiced = voiced_beats(ledger)
    n = len(voiced)
    flagged = {
        i for i in (flagged_beats or [])
        if isinstance(i, int) and 0 <= i < n
    }
    if not flagged:
        return None, {
            "status": "SKIPPED_NO_BEATS",
            "tier1_edits": 0,
            "tier2_edits": 0,
        }

    base_validator = make_post_validator(
        voiced,
        ledger,
        turn_beat_index=turn_beat_index,
        button_beat_index=button_beat_index,
        enforce_word_band=False,
    )
    post_validator = _make_scoped_validator(
        base_validator,
        allowed_actions=MICRO_REPAIR_ACTIONS,
        allowed_beat_indices=flagged,
    )
    prompt = _build_micro_repair_prompt(
        voiced,
        ledger,
        sorted(flagged),
        repair_context=repair_context,
    )

    if structured_call_fn is None:
        try:
            from ._otr_structured_call import structured_call as structured_call_fn  # type: ignore
        except ImportError:  # pragma: no cover - standalone / test load
            from _otr_structured_call import structured_call as structured_call_fn  # type: ignore

    # LLM slot: creative
    # Reason: micro-repair rewrites spoken dialogue on the flagged beats
    # (a narrative pass), so it routes to the creative slot per project
    # rule 6. editor_model is the creative-slot id.
    plan: RadioEditPlan = structured_call_fn(
        prompt=prompt,
        schema=RadioEditPlan,
        slot_fn=slot_fn,
        base_temperature=base_temperature,
        structural_retry_temperature=structural_retry_temperature,
        repair_prompt_factory=_default_repair_prompt,
        post_validator=post_validator,
        max_new_tokens=max_new_tokens,
        helper_name=f"micro_repair[{editor_model}]",
    )
    plan = _resolve_recompose_text(plan, voiced, ledger, recompose_fn)

    report: Dict[str, Any] = {"status": "PLANNED"}
    if apply:
        report = apply_plan(ledger, plan)
        report["status"] = "APPLIED"
    return plan, report


__all__ = [
    # schemas
    "BeatEdit",
    "RadioEditPlan",
    # taxonomy
    "TIER1_ACTIONS",
    "TIER2_ACTIONS",
    "ALL_ACTIONS",
    "NORMALIZE_LENGTH_ACTIONS",
    "MICRO_REPAIR_ACTIONS",
    "VOICED_ROLES",
    "NON_VOICED_ROLES",
    "STRUCTURAL_LOCK_KEYS",
    # caps
    "MAX_WORDS_PER_LINE",
    "MAX_CHARS_PER_LINE",
    "TARGET_WORD_TOTAL",
    "WORD_TOTAL_TOLERANCE",
    # views
    "voiced_beats",
    # noun analysis (Guard 3)
    "concrete_nouns",
    "allowed_noun_set",
    "new_visual_nouns",
    # guards
    "guard1_structural_lock",
    "guard2_text_length",
    "guard3_visual_noun",
    "line_over_cap",
    "word_total_in_band",
    # gate
    "make_post_validator",
    "post_validate_plan",
    # apply
    "apply_plan",
    # preconditions
    "needs_length_normalization",
    # entrypoints
    "run_radio_editor",
    "normalize_length",
    "fit_final_word_delivery",
    "micro_repair",
    # exception
    "FinalWordDeliveryError",
    "GuardError",
]


# ===========================================================================
# Self-test (no network, no GPU) -- mirrors the writer's SELF-TEST idiom.
# Stubs the editor slot_fn + recompose_fn so the full path runs offline.
# ===========================================================================

if __name__ == "__main__":
    import json
    import sys

    _passed = 0
    _total = 0

    def _check(name: str, cond: bool, detail: str = "") -> None:
        global _passed, _total
        _total += 1
        if cond:
            _passed += 1
            print(f"  ok   {name}")
        else:
            print(f"  FAIL {name}: {detail}")

    def _make_ledger() -> Dict[str, Any]:
        """A small, realistic ledger: open-music bookend (non-voiced),
        four voiced character/announcer beats, a close-music bookend.
        Keys mirror production_ledger.init_lines_from_outline."""
        def line(line_id, role, char_id, text, slot=None):
            return {
                "line_id": line_id,
                "shot_id": None,
                "beat_id": line_id,
                "char_id": char_id,
                "text": text,
                "traits": "neutral",
                "boundary": None,
                "char_count": _char_count(text),
                "word_count": _word_count(text),
                "bark_wav_path": None,
                "start_s": None,
                "dur_s": None,
                "speaker_role": role,
                "arc_phase": "rising",
                "compose_flags": [],
                "beat_intent": None,
                "target_words": 40,
                "dialogue_slot_id": slot,
            }
        return {
            "episode_id": "test_episode",
            "cast": [
                {"char_id": "c01", "name": "EDNA", "voice_preset": "bark:v2/en_speaker_6"},
                {"char_id": "c02", "name": "ROY", "voice_preset": "bark:v2/en_speaker_9"},
            ],
            "lines": [
                line("b000", "music_open", "music_open", "[MUSIC] organ swell"),
                line(
                    "b001", "character", "c01",
                    "Roy, the harbor lights went dark an hour ago and nobody "
                    "at the station will tell me why, which frankly is the "
                    "part that frightens me most of all tonight.",
                    slot="d001",
                ),
                line(
                    "b002", "character", "c02",
                    "Stay calm, Edna.",
                    slot="d002",
                ),
                line(
                    "b003", "character", "c02",
                    "I will walk down to the pier myself.",
                    slot="d003",
                ),
                line(
                    "b004", "announcer", "announcer",
                    "And so our two friends step into the waiting dark.",
                    slot="d004",
                ),
                line("b900", "music_close", "music_close", "[MUSIC] closing theme"),
            ],
            "meta": {
                "word_budget": {
                    "target_words": 35,
                    "band": [0.7, 1.3],
                },
                "visual_plan": {
                    "characters": {
                        "EDNA": {"portrait_prompt": "older woman, wool shawl, lantern"},
                        "ROY": {"portrait_prompt": "young man, oilskin coat, by the pier"},
                    },
                    "scenes": [],
                    "style": "noir_radio",
                },
            },
        }

    # --- Test 1: Tier-1 SHORTEN leaves the segment set byte-identical
    #     except the one line's text, and recomputes its counts. -------
    print("[1] Tier-1 SHORTEN (segment set stable)")
    led = _make_ledger()
    pre_ids = [r["line_id"] for r in led["lines"]]
    short_text = "Roy, the harbor lights died an hour ago and no one will say why."
    plan = RadioEditPlan(
        edits=[BeatEdit(beat_index=0, action="SHORTEN_LINE", new_line=short_text)],
        projected_word_total=300,
    )
    voiced0 = voiced_beats(led)
    err = post_validate_plan(plan, voiced0, led)
    _check("plan passes post_validator", err is None, str(err))
    rep = apply_plan(led, plan)
    post_ids = [r["line_id"] for r in led["lines"]]
    _check("segment ids unchanged", pre_ids == post_ids, f"{pre_ids} != {post_ids}")
    edna = next(r for r in led["lines"] if r["line_id"] == "b001")
    _check("line text replaced", edna["text"] == short_text)
    _check(
        "word_count recomputed",
        edna["word_count"] == _word_count(short_text),
        str(edna["word_count"]),
    )
    _check("no realign stamp on Tier-1", "needs_render_realign" not in edna)
    _check("tier1 counted", rep["tier1_edits"] == 1 and rep["tier2_edits"] == 0)

    # --- Test 2: Tier-2 CUT reindexes + stamps needs_render_realign,
    #     preserves the music bookends + bindings. -----------------
    print("[2] Tier-2 CUT_LINE (reindex + needs_render_realign)")
    led = _make_ledger()
    voiced0 = voiced_beats(led)
    pre_voiced = len(voiced0)
    # Cut the short filler beat (voiced index 1 == 'b002', 'Stay calm').
    plan = RadioEditPlan(
        edits=[BeatEdit(beat_index=1, action="CUT_LINE")],
        projected_word_total=300,
    )
    err = post_validate_plan(plan, voiced0, led)
    _check("cut plan passes post_validator", err is None, str(err))
    rep = apply_plan(led, plan)
    voiced1 = voiced_beats(led)
    _check(
        "voiced count dropped by one",
        len(voiced1) == pre_voiced - 1,
        f"{len(voiced1)} vs {pre_voiced}",
    )
    ids_now = [r["line_id"] for r in led["lines"]]
    _check("b002 removed", "b002" not in ids_now, str(ids_now))
    _check("music_open preserved", ids_now[0] == "b000")
    _check("music_close preserved", ids_now[-1] == "b900")
    _check("tier2 counted", rep["tier2_edits"] == 1)
    _check("a removal was recorded", "b002" in rep["removed_beat_ids"])
    realigned = [r for r in led["lines"] if r.get("needs_render_realign")]
    _check("at least one beat stamped needs_render_realign", len(realigned) >= 1)
    # The surviving neighbors of the cut must be flagged for re-timing.
    roy_pier = next(r for r in led["lines"] if r["line_id"] == "b003")
    _check("survivor neighbor stamped realign", roy_pier.get("needs_render_realign") is True)
    # Bindings preserved on survivors.
    _check("survivor char_id intact", roy_pier["char_id"] == "c02")
    _check("survivor slot id intact", roy_pier["dialogue_slot_id"] == "d003")

    # --- Test 3: Guard 1 rejects a speaker-crossing MERGE (never
    #     reassign a line's speaker). -----------------------------------
    print("[3] Guard 1 reject (speaker-crossing MERGE)")
    led = _make_ledger()
    voiced0 = voiced_beats(led)
    # voiced[0]=c01 (EDNA), voiced[1]=c02 (ROY): adjacent but DIFFERENT
    # speakers -> MERGE must be rejected by Guard 1.
    bad_plan = RadioEditPlan(
        edits=[
            BeatEdit(
                beat_index=0,
                action="MERGE_SHORT_LINES",
                merge_with_index=1,
                new_line="Roy, the lights died. Stay calm.",
            )
        ],
        projected_word_total=300,
    )
    err = guard1_structural_lock(bad_plan, voiced0)
    _check("guard1 returns an error", err is not None, "expected rejection")
    _check("error names speaker-crossing", err is not None and "cross speakers" in err, str(err))
    # And the full post_validator surfaces it too.
    err2 = post_validate_plan(bad_plan, voiced0, led)
    _check("post_validator rejects the bad merge", err2 is not None, str(err2))
    # Also reject an out-of-range index.
    oob = RadioEditPlan(
        edits=[BeatEdit(beat_index=99, action="KEEP")],
        projected_word_total=300,
    )
    _check(
        "guard1 rejects out-of-range index",
        guard1_structural_lock(oob, voiced0) is not None,
    )

    # --- Test 4: Guard 3 new-noun REPAIR path. The stub recompose_fn
    #     first emits a line with a NEW noun ('helicopter'), then on the
    #     repair hint emits a clean line. run_radio_editor must accept
    #     the repaired text. --------------------------------------------
    print("[4] Guard 3 new-noun repair (recompose retried with offender named)")
    led = _make_ledger()
    voiced0 = voiced_beats(led)

    _recompose_calls: List[Tuple[int, str, str]] = []

    def _stub_recompose(beat_index: int, original_text: str, hint: str) -> str:
        _recompose_calls.append((beat_index, original_text, hint))
        if not hint:
            # First pass: introduce a NEW visual noun not in the beat or
            # any portrait_prompt -> Guard 3 must flag 'helicopter'.
            return "Stay calm, Edna, the rescue helicopter is on its way."
        # Repair pass (hint names the offender): clean line, no new noun.
        return "Stay calm, Edna, help is already on its way."

    # The editor slot_fn returns a fixed plan: RECOMPOSE beat 1 with NO
    # inline prose (forces the recompose_fn path).
    recompose_plan_json = json.dumps(
        {
            "edits": [
                {"beat_index": 1, "action": "RECOMPOSE_BEAT_SAME_INTENT", "new_line": None},
            ],
            "projected_word_total": 300,
        }
    )

    def _stub_slot_fn(messages, *, temperature, max_new_tokens, stop=None):
        return recompose_plan_json

    # Inject a real-shaped structured_call substitute that validates +
    # runs the post_validator exactly like the production ladder, so the
    # test exercises the guard wiring without importing pydantic-heavy
    # siblings beyond this module.
    def _fake_structured_call(
        *, prompt, schema, slot_fn, base_temperature,
        structural_retry_temperature, repair_prompt_factory=None,
        post_validator=None, max_new_tokens=512, max_attempts=3,
        helper_name="",
    ):
        raw = slot_fn([{"role": "user", "content": prompt}],
                      temperature=base_temperature, max_new_tokens=max_new_tokens)
        inst = schema.model_validate(json.loads(raw))
        if post_validator is not None:
            content_error = post_validator(inst)
            if content_error is not None:
                raise AssertionError(f"post_validator rejected: {content_error}")
        return inst

    plan, rep = run_radio_editor(
        led,
        editor_model="test-creative-model",
        slot_fn=_stub_slot_fn,
        recompose_fn=_stub_recompose,
        base_temperature=0.6,
        structural_retry_temperature=0.3,
        apply=True,
        structured_call_fn=_fake_structured_call,
    )
    _check("recompose_fn called twice (initial + repair)", len(_recompose_calls) == 2,
           f"calls={len(_recompose_calls)}")
    _check("repair hint named the offending noun",
           len(_recompose_calls) == 2 and "helicopter" in _recompose_calls[1][2],
           str(_recompose_calls))
    final_b002 = next(r for r in led["lines"] if r["line_id"] == "b002")
    _check("final line is the repaired (clean) text",
           "helicopter" not in final_b002["text"],
           final_b002["text"])
    # Confirm the offending PHYSICAL noun is gone. (Guard 3 over-includes
    # abstract content words like 'help'/'way' into the noun set by
    # design -- it only matters whether a word is NEW vs the baseline,
    # and in real use the baseline carries those same words. Here we
    # assert the specific render-relevant offender 'helicopter' no longer
    # appears in the recomposed line's concrete-noun set.)
    _check("offending physical noun removed after repair",
           "helicopter" not in concrete_nouns(final_b002["text"]),
           str(sorted(concrete_nouns(final_b002["text"]))))

    # --- Test 4b: Guard 3 directly flags a new noun and allows a
    #     synonym/morphology variant. -----------------------------------
    print("[4b] Guard 3 detector (new noun flagged, synonym allowed)")
    led = _make_ledger()
    voiced0 = voiced_beats(led)
    flag_plan = RadioEditPlan(
        edits=[BeatEdit(
            beat_index=1, action="RECOMPOSE_BEAT_SAME_INTENT",
            new_line="Stay calm, Edna, the submarine is surfacing.")],
        projected_word_total=300,
    )
    _check("guard3 flags 'submarine'",
           guard3_visual_noun(flag_plan, voiced0, led) is not None)
    # 'pier' is in ROY's portrait_prompt ("by the pier"); a recompose of
    # EDNA's line that says 'piers' (plural) must be ALLOWED (morphology).
    syn_plan = RadioEditPlan(
        edits=[BeatEdit(
            beat_index=0, action="RECOMPOSE_BEAT_SAME_INTENT",
            new_line="Roy, the lights by the piers went dark an hour ago.")],
        projected_word_total=300,
    )
    _check("guard3 allows morphological variant 'piers'",
           guard3_visual_noun(syn_plan, voiced0, led) is None,
           str(guard3_visual_noun(syn_plan, voiced0, led)))

    # --- Test 5: deterministic actual total owns the band; the model's
    #     projection remains forensic only. ----------------------------
    print("[5] simulated actual word-total gate")
    led = _make_ledger()
    voiced0 = voiced_beats(led)
    in_band = RadioEditPlan(
        edits=[BeatEdit(beat_index=2, action="KEEP")],
        projected_word_total=350,
    )
    _check("good actual is accepted", post_validate_plan(in_band, voiced0, led) is None)
    too_long = RadioEditPlan(
        edits=[BeatEdit(beat_index=2, action="KEEP")],
        projected_word_total=800,
    )
    err = post_validate_plan(too_long, voiced0, led)
    _check("false high projection does not reject a good actual",
           err is None, str(err))
    too_short = RadioEditPlan(
        edits=[BeatEdit(beat_index=2, action="KEEP")],
        projected_word_total=100,
    )
    _check("false low projection does not reject a good actual",
           post_validate_plan(too_short, voiced0, led) is None)
    _check("280 (lower edge) in band", word_total_in_band(280))
    _check("420 (upper edge) in band", word_total_in_band(420))

    # --- Test 6: Guard 2 length cap + SPLIT_LINE remedy. -------------
    print("[6] Guard 2 length cap + SPLIT_LINE remedy")
    led = _make_ledger()
    voiced0 = voiced_beats(led)
    long_line = (
        "Roy I have to tell you that the entire harbor district has gone "
        "completely dark tonight and the men at the station refuse to "
        "explain a single thing about what is happening out there on the "
        "water near the old lighthouse."
    )
    over = RadioEditPlan(
        edits=[BeatEdit(beat_index=0, action="SHORTEN_LINE", new_line=long_line)],
        projected_word_total=300,
    )
    _check("guard2 rejects an over-cap SHORTEN", guard2_text_length(over, voiced0) is not None)
    # SPLIT_LINE of the same text into sentence units must pass (each
    # fragment under cap) -- build a 2-sentence over-long line.
    split_src = (
        "The harbor district has gone completely dark tonight. "
        "The men at the station refuse to explain anything at all."
    )
    split_plan = RadioEditPlan(
        edits=[BeatEdit(beat_index=0, action="SPLIT_LINE", new_line=split_src)],
        projected_word_total=300,
    )
    _check("guard2 accepts a clean SPLIT_LINE", guard2_text_length(split_plan, voiced0) is None,
           str(guard2_text_length(split_plan, voiced0)))
    # Apply the split: voiced count must grow by one, both halves stamped.
    led_s = _make_ledger()
    pre_v = len(voiced_beats(led_s))
    rep_s = apply_plan(led_s, split_plan)
    post_v = len(voiced_beats(led_s))
    _check("SPLIT grew voiced count by one", post_v == pre_v + 1, f"{pre_v}->{post_v}")
    _check("SPLIT added a beat id", len(rep_s["added_beat_ids"]) == 1, str(rep_s["added_beat_ids"]))
    split_rows = [r for r in led_s["lines"] if r["line_id"] in ("b001", "b001_s1")]
    _check("both split halves present", len(split_rows) == 2, str([r["line_id"] for r in led_s["lines"]]))
    _check("split child inherits speaker binding",
           all(r["char_id"] == "c01" for r in split_rows))
    _check("split halves stamped realign",
           all(r.get("needs_render_realign") for r in split_rows))

    # --- Test 7: post_validator refuses to cut the turn / button. ----
    print("[7] post_validator protects turn + button beats")
    led = _make_ledger()
    voiced0 = voiced_beats(led)
    # Declare voiced index 3 (announcer 'button') as the button beat,
    # voiced index 0 as the turn.
    cut_button = RadioEditPlan(
        edits=[BeatEdit(beat_index=3, action="CUT_LINE")],
        projected_word_total=300,
    )
    err = post_validate_plan(cut_button, voiced0, led,
                             turn_beat_index=0, button_beat_index=3)
    _check("cutting the button beat is rejected", err is not None, str(err))
    _check("rejection names the button", err is not None and "button" in err, str(err))
    cut_turn = RadioEditPlan(
        edits=[BeatEdit(beat_index=0, action="REMOVE_REDUNDANT_BEAT")],
        projected_word_total=300,
    )
    err = post_validate_plan(cut_turn, voiced0, led,
                             turn_beat_index=0, button_beat_index=3)
    _check("cutting the turning_point beat is rejected", err is not None, str(err))

    # --- Test 8: needs_length_normalization precondition probe. ------
    print("[8] needs_length_normalization (band + per-line cap triggers)")

    def _ledger_with_total(num_lines: int, words_each: int,
                           cap_bust: bool = False) -> Dict[str, Any]:
        """Build a ledger of ``num_lines`` voiced lines of ``words_each``
        words each (so the total is controllable). ``cap_bust`` makes the
        first line a 60-word run-on to trip the per-line cap while leaving
        the total in band."""
        filler = " ".join(["word"] * words_each)
        rows: List[Dict[str, Any]] = [
            {"line_id": "m0", "beat_id": "m0", "char_id": "music_open",
             "speaker_role": "music_open", "text": "[MUSIC] open"}
        ]
        for k in range(num_lines):
            t = " ".join(["word"] * 60) if (cap_bust and k == 0) else filler
            rows.append({
                "line_id": f"L{k}", "beat_id": f"b{k}", "char_id": "c01",
                "speaker_role": "character", "text": t,
                "word_count": _word_count(t),
            })
        return {"cast": [{"char_id": "c01", "name": "ALICE"}],
                "lines": rows, "meta": {}}

    led_in = _ledger_with_total(12, 28)        # total 336 -> in band
    led_short = _ledger_with_total(2, 10)       # total 20 -> under fallback band
    led_cap = _ledger_with_total(11, 28, cap_bust=True)  # in band, line over cap
    _check("in-spec draft does not need normalization",
           needs_length_normalization(led_in) is False)
    _check("under-band draft needs normalization",
           needs_length_normalization(led_short) is True)
    _check("over-cap line needs normalization (even in band)",
           needs_length_normalization(led_cap) is True)

    # --- Test 9: normalize_length SKIP path on an in-spec draft -------
    print("[9] normalize_length skip path (no LLM call when in spec)")

    def _must_not_call(*args, **kwargs):
        raise AssertionError("slot_fn must not run on an in-spec draft")

    def _noop_recompose(beat_index, original_text, hint):
        return original_text

    plan_skip, rep_skip = normalize_length(
        led_in, editor_model="test-creative", slot_fn=_must_not_call,
        recompose_fn=_noop_recompose,
    )
    _check("in-spec -> SKIPPED_IN_SPEC, no plan, no LLM call",
           plan_skip is None and rep_skip["status"] == "SKIPPED_IN_SPEC")

    # --- Test 10: normalize_length ACTIVE path tightens an over-long
    #     draft (Tier-2 cuts) and applies. -----------------------------
    print("[10] normalize_length active path (over-long draft -> cut to band)")
    led_long = _ledger_with_total(20, 28)       # total 560 -> over band
    _check("over-long draft needs normalization",
           needs_length_normalization(led_long) is True)
    cut_plan_json = json.dumps({
        "edits": [{"beat_index": i, "action": "CUT_LINE"} for i in range(8)],
        "projected_word_total": 350,
    })

    def _cut_slot(messages, *, temperature, max_new_tokens, stop=None):
        return cut_plan_json

    pre_long = len(voiced_beats(led_long))
    plan_long, rep_long = normalize_length(
        led_long, editor_model="test-creative", slot_fn=_cut_slot,
        recompose_fn=_noop_recompose, structured_call_fn=_fake_structured_call,
    )
    post_long = len(voiced_beats(led_long))
    _check("active normalize_length applied + status APPLIED",
           plan_long is not None and rep_long["status"] == "APPLIED")
    _check("normalize_length cut voiced beats (Tier-2)",
           post_long == pre_long - 8, f"{pre_long}->{post_long}")

    # --- Test 11: micro_repair no-beats skip path. -------------------
    print("[11] micro_repair skip path (empty / invalid flagged beats)")
    led_mr = _make_ledger()
    plan_nb, rep_nb = micro_repair(
        led_mr, [], editor_model="test-creative", slot_fn=_must_not_call,
        recompose_fn=_noop_recompose,
    )
    _check("empty flagged_beats -> SKIPPED_NO_BEATS, no LLM call",
           plan_nb is None and rep_nb["status"] == "SKIPPED_NO_BEATS")
    plan_oob, rep_oob = micro_repair(
        led_mr, [99], editor_model="test-creative", slot_fn=_must_not_call,
        recompose_fn=_noop_recompose,
    )
    _check("out-of-range flagged beat -> SKIPPED_NO_BEATS",
           plan_oob is None and rep_oob["status"] == "SKIPPED_NO_BEATS")

    # --- Test 12: micro_repair scoped fence (action + flagged-beat). -
    print("[12] micro_repair scoped validator (action + beat fence)")
    led12 = _make_ledger()
    voiced12 = voiced_beats(led12)
    base12 = make_post_validator(voiced12, led12)
    scoped12 = _make_scoped_validator(
        base12, allowed_actions=MICRO_REPAIR_ACTIONS, allowed_beat_indices={1},
    )
    edit_unflagged = RadioEditPlan(
        edits=[BeatEdit(beat_index=2, action="SHORTEN_LINE", new_line="Short line.")],
        projected_word_total=350,
    )
    _check("edit on a non-flagged beat is rejected",
           scoped12(edit_unflagged) is not None)
    edit_badaction = RadioEditPlan(
        edits=[BeatEdit(beat_index=1, action="CUT_LINE")],
        projected_word_total=350,
    )
    _check("disallowed action (CUT_LINE) is rejected in micro-repair",
           scoped12(edit_badaction) is not None)
    edit_ok = RadioEditPlan(
        edits=[BeatEdit(beat_index=1, action="SMOOTH_DIALOGUE",
                        new_line="Stay calm, Edna.")],
        projected_word_total=350,
    )
    _check("SMOOTH_DIALOGUE on the flagged beat is accepted",
           scoped12(edit_ok) is None, str(scoped12(edit_ok)))

    # --- Test 13: micro_repair end-to-end + Guard 3 fences a new noun
    #     in SMOOTH_DIALOGUE / CLARIFY_TURN fresh prose. ---------------
    print("[13] micro_repair applies SMOOTH_DIALOGUE; Guard 3 fences new noun")
    led13 = _make_ledger()
    voiced13 = voiced_beats(led13)
    smooth_json = json.dumps({
        "edits": [{"beat_index": 1, "action": "SMOOTH_DIALOGUE",
                   "new_line": "Stay calm now, Edna."}],
        "projected_word_total": 350,
    })

    def _smooth_slot(messages, *, temperature, max_new_tokens, stop=None):
        return smooth_json

    plan13, rep13 = micro_repair(
        led13, [1], editor_model="test-creative", slot_fn=_smooth_slot,
        recompose_fn=_noop_recompose, structured_call_fn=_fake_structured_call,
    )
    b002 = next(r for r in led13["lines"] if r["line_id"] == "b002")
    _check("micro_repair applied SMOOTH_DIALOGUE to the flagged beat",
           rep13["status"] == "APPLIED" and "Stay calm now" in b002["text"],
           b002["text"])
    bad_smooth = RadioEditPlan(
        edits=[BeatEdit(beat_index=1, action="SMOOTH_DIALOGUE",
                        new_line="Stay calm, the helicopter is coming.")],
        projected_word_total=350,
    )
    _check("Guard 3 fences a new noun in SMOOTH_DIALOGUE fresh prose",
           guard3_visual_noun(bad_smooth, voiced13, led13) is not None)
    clarify_new_noun = RadioEditPlan(
        edits=[BeatEdit(beat_index=1, action="CLARIFY_TURN",
                        new_line="Stay calm, the submarine decides it.")],
        projected_word_total=350,
    )
    _check("Guard 3 fences a new noun in CLARIFY_TURN fresh prose",
           guard3_visual_noun(clarify_new_noun, voiced13, led13) is not None)

    # --- Summary -----------------------------------------------------
    print()
    print(f"SELF-TEST PASS: {_passed}/{_total}")
    if _passed != _total:
        sys.exit(1)
    sys.exit(0)
