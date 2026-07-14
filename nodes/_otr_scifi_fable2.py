"""scifi_fable2 runner -- S2 full loop (LLM-first multipass sci-fi lane).

Architecture doc: docs/2026-07-10-scifi-fable2-architecture.md (sections
3/5/7/8/9/11/13). Operator law: **Python judges; the LLM writes.** Every
spoken ledger row is traceable to a named LLM artifact (the per-constituent
proof gate); this module never writes, trims, or repairs a spoken word.

S2 scope (doc s13): P0 dossier -> deal -> P1 pitch (one pitch below 120
words; three pitches at 120-900) -> P2a selection in full mode -> P2b
treatment -> P3 script (markup ladder + budget gate + truncation retry) ->
P4 critic + P5 revision in full mode -> deterministic keep-better judge ->
P6 casting/voices -> P7 pure-python assembly (proof gates, incremental
saves) -> P8 ledger audit (audit-only, fail loud). Compact mode stamps
P2a/P4/P5 as explicit skips. Requests above 900 fail LOUD before any
creative call; no supported request silently degrades to compact mode.

Posture: PURE module -- stdlib + pydantic + the shared structured_call
ladder + the fable2 markup parser + config.cast_pools + _otr_canon. No
ComfyUI imports, no GPU, and NEVER an import of OTR_LedgerScriptWriter
(r4/M3: the runner returns a plain `Fable2TailParts`; the WRITER builds
its own WriterTailContext from the parts, keeping the import graph
acyclic -- pinned by the pure-import test). Every failure raises a
`Fable2Error` subclass naming the pass. **No fallback to
legacy_many_pass, ever.**

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random
import re
import warnings
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from ._otr_structured_call import (
        StructuredCallFailedError,
        structured_call,
    )
    from ._otr_repair_prompts import make_dispatching_repair_factory
    from ._otr_story_rules import StoryRules, resolve_story_rules
    from ._otr_fable2_markup import (
        ANNOUNCER_NAME,
        Fable2ParseDefect,
        ParsedScript,
        normalize_fable2_markup_text,
        parse_fable2_markup,
        render_defects,
    )
    from . import _otr_canon as _OTRC
    # Shared operator kill-lexicons (lexicon-only kill policy, ratified
    # 2026-07-10 on the original_radio lane: a judge flag for a
    # closed-vocabulary class kills the episode ONLY when the lexicon
    # corroborates it in the script -- judge hallucinations are
    # discarded LOUDLY). One source of truth across lanes: extend the
    # lists there, not code here.
    from ._otr_original_radio import (
        WEAPON_SMOKING_EVIDENCE,
        lexicon_hits,
    )
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_structured_call import (  # type: ignore
        StructuredCallFailedError,
        structured_call,
    )
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore
    from _otr_story_rules import (  # type: ignore
        StoryRules,
        resolve_story_rules,
    )
    from _otr_fable2_markup import (  # type: ignore
        ANNOUNCER_NAME,
        Fable2ParseDefect,
        ParsedScript,
        normalize_fable2_markup_text,
        parse_fable2_markup,
        render_defects,
    )
    import _otr_canon as _OTRC  # type: ignore
    from _otr_original_radio import (  # type: ignore
        WEAPON_SMOKING_EVIDENCE,
        lexicon_hits,
    )

# Cast pools: relative in production (package load), absolute under test.
try:
    from ..config import cast_pools as _POOLS  # type: ignore[no-redef]
except (ImportError, ValueError):
    import sys as _sys
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(_REPO_ROOT))
    from config import cast_pools as _POOLS  # type: ignore[no-redef]


log = logging.getLogger("OTR")

# The LLM contract field "register" (HOW a character speaks -- doc s5) is
# load-bearing prompt vocabulary and cannot be renamed; pydantic warns that
# it shadows a BaseModel attribute. Silence exactly that warning -- clean
# logs, and nothing in this module ever calls the shadowed attribute.
warnings.filterwarnings(
    "ignore", message='Field name "register"', category=UserWarning)


# ---------------------------------------------------------------------------
# Errors (doc s2: ctor (pass_id, reason, attempts))
# ---------------------------------------------------------------------------

class Fable2Error(Exception):
    """Base: any fail-loud fable2 lane problem. Names the pass."""

    def __init__(self, pass_id: str, reason: str, attempts: int = 0) -> None:
        self.pass_id = pass_id
        self.reason = reason
        self.attempts = attempts
        att = f" after {attempts} attempt(s)" if attempts else ""
        super().__init__(
            f"[scifi_fable2] pass {pass_id!r} failed{att}: {reason} "
            f"(no fallback to legacy_many_pass)"
        )


class Fable2DossierError(Fable2Error):
    pass


class Fable2PitchError(Fable2Error):
    pass


class Fable2SelectError(Fable2Error):
    pass


class Fable2TreatmentError(Fable2Error):
    pass


class Fable2ScriptError(Fable2Error):
    pass


class Fable2ParseError(Fable2Error):
    pass


class Fable2CastError(Fable2Error):
    pass


class Fable2AssembleError(Fable2Error):
    pass


class Fable2AuditError(Fable2Error):
    pass


# ---------------------------------------------------------------------------
# Constants (doc s3/s8)
# ---------------------------------------------------------------------------

_TEMP = {
    "dossier": 0.30, "pitch_room": 0.90, "pitch_select": 0.30,
    "treatment": 0.40, "script": 0.75, "critic": 0.30, "revision": 0.60,
    "casting_voices": 0.40,
}


def _MARKUP_LADDER_TEMPS(t: float) -> "tuple[float, ...]":
    """The markup ladder NEVER raises temperature (2B principle). Four
    rungs (10th live smoke 2026-07-10: with the format wars won, rolls
    were dying on ONE small skeleton slip per attempt -- e.g. a missing
    closing MUSIC line -- so depth, not temperature, is the lever; the
    final rung repeats 0.30 with the defect quote)."""
    return (t, round(t * 0.66, 2), 0.30, 0.30)


_MAX_BUDGET_REROLLS = 2
_TOTAL_WORD_BAND = 0.20      # +/-20% band on CHARACTER words (r4/M4)
_SCENE_WORD_BAND = 0.30      # prompt-stated per-scene tolerance
_ONE_DRAFT_THRESHOLD = 120   # below this: one-pitch + one-draft mode
_SUPPORTED_WORD_CEILING = 900  # act-chunk mode deferred post-S3
_DIGEST_CHAR_CAP = 3600

_COMPACT_MODE = "one_pitch_one_draft"
_FULL_MODE = "three_pitch_two_draft"
Fable2Mode = Literal["one_pitch_one_draft", "three_pitch_two_draft"]

# S2 pins scene count as an explicit word-band table. Do not recover the old
# `round(words / 110)` formula: its half-boundaries depend on Python banker's
# rounding and were never an auditable production contract.
_SCENE_COUNT_TABLE: "tuple[tuple[int, int], ...]" = (
    (164, 1),
    (274, 2),
    (384, 3),
    (494, 4),
    (604, 5),
    (714, 6),
    (824, 7),
    (_SUPPORTED_WORD_CEILING, 8),
)

# Fixed passes: plain ints (r2 anchor: no lambdas in a dict). P3/P5 use
# _script_token_budget instead.
_MAX_NEW_TOKENS = {
    "dossier": 700, "pitch_room": 900, "pitch_select": 300,
    "treatment": 1100, "critic": 800,
    # 18th live smoke 2026-07-10: 1000 truncated a verbose 2-cast JSON
    # mid-object -> the extractor salvaged an inner entry -> schema fail.
    "casting_voices": 1400,
}

# Named overlap thresholds (unit-fixtured both directions, r2 anchor).
_REGISTER_OVERLAP_MAX = 0.5
_LOGLINE_OVERLAP_MAX = 0.6

_SCHEMA_VERSION = "fable2_v1"

_DECK_PATH = (
    Path(__file__).resolve().parent / "story_packs" / "scifi_fable2"
    / "frame_deck.json"
)


def _script_token_budget(target_words: int) -> int:
    """P3/P5 output budget: markup overhead ~1.5x words, ~1.35 tok/word,
    +200 skeleton overhead; floor 1200, cap 4200."""
    return min(4200, max(1200, int(target_words * 2.2) + 200))


_WORD_BAND_ABS_FLOOR = 25


def _word_band(target_words: int) -> "tuple[float, float]":
    """CHARACTER-word acceptance band: +/-20% (r4/M4) with an ABSOLUTE
    slack floor of +/-25 words (17th live smoke 2026-07-10: at 30 words
    the proportional band is 12 words wide -- whole-play word-count
    precision no 12B has; the band's purpose is render-length control,
    and +/-25 words is ~15 seconds of dialogue). At production lengths
    (>=125 words) the proportional band governs unchanged."""
    tw = int(target_words)
    slack = max(tw * _TOTAL_WORD_BAND, _WORD_BAND_ABS_FLOOR)
    return (max(1.0, tw - slack), tw + slack)


def assert_supported_target_words(target_words: int) -> None:
    """The lane's word-budget entry gate (r3/M4: runs in the WRITER at
    run() ENTRY, before the RSS fetch and the D.1 skeleton save; the
    runner re-asserts it defensively).

    Requests above the supported ceiling fail LOUD. The full S2 path owns
    120-900 words; compact mode owns only requests below 120. The caller
    must resolve the mode explicitly and never fall back between them.
    """
    tw = int(target_words)
    if tw > _SUPPORTED_WORD_CEILING:
        raise Fable2ScriptError(
            "script",
            f"target_words {tw} above supported ceiling "
            f"{_SUPPORTED_WORD_CEILING} (act-chunked long-episode mode is "
            f"deferred post-S3; lower target_words)",
        )


def _resolve_mode(target_words: int) -> Fable2Mode:
    """Return the executable S2 mode for a requested word count."""
    tw = int(target_words)
    if tw > _SUPPORTED_WORD_CEILING:
        raise Fable2ScriptError(
            "script",
            f"target_words {tw} above supported ceiling "
            f"{_SUPPORTED_WORD_CEILING}",
        )
    return (
        _COMPACT_MODE
        if tw < _ONE_DRAFT_THRESHOLD
        else _FULL_MODE
    )


# ---------------------------------------------------------------------------
# Artifact models (doc s5; pydantic v2)
# ---------------------------------------------------------------------------

class NamedEntities(BaseModel):
    people: list[str] = Field(default_factory=list, max_length=10)
    places: list[str] = Field(default_factory=list, max_length=10)
    things: list[str] = Field(default_factory=list, max_length=10)


class DossierLLM(BaseModel):
    """The LLM-facing dossier contract (r2/M2+CUT2). Provenance
    (headline/source/date/link) is PYTHON-STAMPED into
    meta.fable2.dossier after validation -- never in the LLM JSON."""

    facts_to_keep: list[str] = Field(min_length=3, max_length=10)
    allowed_numbers: list[str] = Field(default_factory=list, max_length=10)
    named_entities: NamedEntities
    dramatizable_vectors: list[str] = Field(min_length=3, max_length=5)

    @field_validator("facts_to_keep")
    @classmethod
    def _fact_lengths(cls, v: list[str]) -> list[str]:
        for f in v:
            if not 8 <= len(f) <= 200:
                raise ValueError(f"fact length {len(f)} outside 8-200: {f!r}")
        return v

    @field_validator("dramatizable_vectors")
    @classmethod
    def _vector_lengths(cls, v: list[str]) -> list[str]:
        for x in v:
            if not 10 <= len(x) <= 160:
                raise ValueError(f"vector length {len(x)} outside 10-160")
        return v


class Pitch(BaseModel):
    pitch_id: int = Field(ge=1, le=3, strict=True)
    frame_card: str
    logline: str = Field(min_length=15, max_length=240)
    hook: str = Field(min_length=10, max_length=200)
    scifi_device: str = Field(min_length=10, max_length=160)
    cast_size: int = Field(ge=1, le=8)
    ending_shape: Literal[
        "paid_victory", "quiet_loss", "ironic_turn", "open_question"]


class PitchSlate(BaseModel):
    """Exactly 3 pitches in full mode; exactly 1 in one-pitch (low-budget)
    mode -- the expectation is a post_validator parameter (r2/M3)."""

    pitches: list[Pitch] = Field(min_length=1, max_length=3)


class PitchSelect(BaseModel):
    chosen_pitch_id: int = Field(ge=1, le=3, strict=True)
    selection_rationale: str = Field(min_length=20, max_length=300)


class CastShape(BaseModel):
    """THE canonical cast-name source (r2/M1): names are born HERE so the
    script pass has legal cast names BEFORE P3 and the parser can gate
    speakers. P6 is voices/portraits only."""

    name: str
    role: str = Field(min_length=3, max_length=60)
    want: str = Field(min_length=5, max_length=120)
    pressure: str = Field(min_length=5, max_length=120)
    register: str = Field(min_length=5, max_length=90)

    @field_validator("name", mode="before")
    @classmethod
    def _normalize_name_label(cls, v):
        """Deterministic LABEL normalization (19th live smoke
        2026-07-10: the model titles its scientists reflexively --
        'DR. HARRIS' -- and the repair rung would not converge). The
        surname IS the LLM's chosen name; python strips the banned
        title tokens and keeps the LAST word (radio-billing surname
        convention -- the doc's own example is VOSS). Never invents a
        character; only normalizes the label shape."""
        if not isinstance(v, str):
            return v
        tokens = [t for t in v.strip().upper().split()
                  if t.rstrip(".") not in _HONORIFIC_TOKENS]
        if tokens:
            return tokens[-1].replace(".", "")
        return v

    @field_validator("name")
    @classmethod
    def _name_all_caps_one_word_not_announcer(cls, v: str) -> str:
        v = v.strip()
        if not v or v != v.upper():
            raise ValueError(f"cast name must be ALL CAPS, got {v!r}")
        if v == ANNOUNCER_NAME:
            raise ValueError("cast name must never be ANNOUNCER")
        # kibitz r2 S1 + 12th live smoke ("DR. VERONICA VOSS" shortened
        # to "VOSS" mid-script -> UNKNOWN_SPEAKER): one word, no titles,
        # no initials -- normalized above; anything still multiword here
        # is unrecoverable.
        if " " in v or "." in v:
            raise ValueError(
                f"cast name must be ONE invented word with no titles or "
                f"initials, got {v!r}")
        return v


class Treatment(BaseModel):
    title: str = Field(min_length=3, max_length=80)
    dramatic_question: str = Field(min_length=10, max_length=200)
    setting: str = Field(min_length=4, max_length=120)
    cast_shapes: list[CastShape] = Field(min_length=1, max_length=8)
    turn: str = Field(min_length=10, max_length=250)
    priced_ending: dict[str, str]
    news_thread: str = Field(min_length=10, max_length=200)
    # S1b read-split (kibitz r2 Q1 ruling, pulled forward 2026-07-10
    # after 13 live rolls: 7 died in the combined pass's read gates and
    # the typed repair could not converge): the factual close is now
    # authored by its OWN low-temp technical pass (P2c, seam
    # fable2_news_read_system) and stamped onto the treatment by the
    # runner -- so the field is optional at P2b and the downstream
    # contract (assembly + proof artifact name) is unchanged.
    news_close_read: str = Field(default="", max_length=420)

    @field_validator("dramatic_question")
    @classmethod
    def _has_question_mark(cls, v: str) -> str:
        if "?" not in v:
            raise ValueError("dramatic_question must contain '?'")
        return v

    @model_validator(mode="after")
    def _names_unique_registers_distinct_ending_priced(self) -> "Treatment":
        names = [c.name for c in self.cast_shapes]
        if len(set(names)) != len(names):
            raise ValueError(f"cast names must be unique, got {names}")
        regs = [c.register for c in self.cast_shapes]
        for i in range(len(regs)):
            for j in range(i + 1, len(regs)):
                ov = _token_overlap(regs[i], regs[j])
                if ov >= _REGISTER_OVERLAP_MAX:
                    raise ValueError(
                        f"registers {i} and {j} overlap {ov:.2f} >= "
                        f"{_REGISTER_OVERLAP_MAX} -- registers must differ "
                        f"in mechanism: {regs[i]!r} vs {regs[j]!r}"
                    )
        for key in ("choice", "cost_paid"):
            val = str(self.priced_ending.get(key) or "")
            if not 10 <= len(val) <= 200:
                raise ValueError(
                    f"priced_ending.{key} length {len(val)} outside 10-200")
        return self


class CriticNote(BaseModel):
    scene: int = Field(ge=0, strict=True)
    speaker: str
    frame_scope: "Literal['intro', 'outro'] | None" = None
    problem: Literal[
        "register_bleed", "on_the_nose", "stakes_sag", "ending_unearned",
        "continuity_break", "cast_unused", "announcer_contract",
        "word_budget", "subtext_flat"]
    note: str = Field(min_length=15, max_length=200)

    @field_validator("note")
    @classmethod
    def _never_replacement_dialogue(cls, v: str) -> str:
        if re.search(r"(?:^|[.!?]\s+)[A-Z][A-Z0-9 .'\-]{1,24}:\s", v):
            raise ValueError(
                "a note is never replacement dialogue (NAME-colon shape)")
        return v

    @model_validator(mode="after")
    def _frame_scope_is_explicit(self) -> "CriticNote":
        if self.scene == 0 and self.frame_scope is None:
            raise ValueError(
                "scene 0 requires frame_scope='intro' or 'outro'")
        if self.scene > 0 and self.frame_scope is not None:
            raise ValueError("frame_scope is only legal for scene 0 notes")
        return self


class CriticNotes(BaseModel):
    verdict: Literal["ship", "revise"]
    notes: list[CriticNote] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _revise_needs_notes(self) -> "CriticNotes":
        if self.verdict == "revise" and not self.notes:
            raise ValueError('"revise" requires >= 1 note')
        if self.verdict == "ship" and self.notes:
            raise ValueError('"ship" requires an empty notes array')
        return self


class CastVoice(BaseModel):
    name: str
    role: str = Field(min_length=3, max_length=60)
    character_description: str = Field(min_length=40, max_length=240)
    gender: Literal["male", "female"]
    age_band: Literal["20s", "30s", "40s", "50s", "60s"]
    # register/want/pressure are descriptive PAPERWORK (r1/A3: register
    # authority stays with the treatment; these never drive the voice or
    # the portrait). 31st live smoke 2026-07-10: a casting that omitted
    # them schema-aborted a run -- paperwork tolerance, defaults allowed.
    register: str = Field(default="", max_length=90)
    timbre: str
    want: str = Field(default="", max_length=120)
    pressure: str = Field(default="", max_length=120)


class NewsCloseRead(BaseModel):
    """P2c: the 1-2 sentence factual close (read-split, S1b). Floor 40,
    not the doc's 80 -- the 27th live smoke killed a perfectly factual
    78-char read; brevity is a style preference, never a correctness
    gate (the seam still asks for 1-2 wire-desk sentences)."""

    news_close_read: str = Field(min_length=40, max_length=420)


class CastingVoices(BaseModel):
    cast: list[CastVoice] = Field(min_length=1, max_length=8)

    @model_validator(mode="before")
    @classmethod
    def _accept_unwrapped(cls, data):
        """Wrapper tolerance (18th live smoke 2026-07-10): a truncated
        or envelope-dropping response can arrive as a BARE list of cast
        entries or a single entry dict -- wrap it so the SPEAKER-SET
        equality gate (which teaches) judges it instead of a raw schema
        error the repair cannot act on."""
        if isinstance(data, list):
            return {"cast": data}
        if isinstance(data, dict) and "cast" not in data and "name" in data:
            return {"cast": [data]}
        return data


# ---------------------------------------------------------------------------
# Small shared helpers
# ---------------------------------------------------------------------------

def _norm_ws(text: str) -> str:
    """Whitespace-collapsed form for verbatim proof matching (exact words,
    exact order; ignores how a phrase wrapped)."""
    return " ".join(str(text).split())


def _token_overlap(a: str, b: str) -> float:
    """Normalized token overlap: |A & B| / min(|A|, |B|); 0.0 when either
    side is empty. Casefolded, punctuation-stripped word sets."""
    ta = {t.strip(".,;:!?'\"()").casefold() for t in str(a).split()}
    tb = {t.strip(".,;:!?'\"()").casefold() for t in str(b).split()}
    ta.discard("")
    tb.discard("")
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _seam(pack: Any, name: str) -> str:
    """Direct prompt_stages read (original_radio accessor pattern)."""
    stages = getattr(pack, "prompt_stages", None) or {}
    value = str(stages.get(name) or "")
    if not value.strip():
        raise Fable2Error(
            "seam",
            f"pack seam {name!r} missing/empty (pack "
            f"{getattr(pack, 'story_model_id', '?')!r}); author it in the "
            f"pack -- the accessors reject declared-but-empty seams",
        )
    return value


def _helper_ctx(slot_scheduler: Any, name: str):
    """slot_scheduler.helper_context attribution when available; inert
    under unit tests that pass slot_scheduler=None."""
    if slot_scheduler is None:
        return nullcontext()
    return slot_scheduler.helper_context(name)


def _counting(slot_fn: Callable[..., str]):
    """Runner-local counting wrapper (media-interpreter precedent, r2/S2):
    structured_call does not return attempt counts on success, so the
    receipt reads the wrapper's counter."""
    box = {"calls": 0}

    def _fn(msgs, *, temperature, max_new_tokens):
        box["calls"] += 1
        return slot_fn(
            msgs, temperature=temperature, max_new_tokens=max_new_tokens)

    return _fn, box


def _ledger_meta(led: Any) -> dict:
    """Return the live meta mapping after any Ledger.save() rebind."""
    return led.data.setdefault("meta", {})


def _fable2_meta(led: Any) -> dict:
    """Return the live Fable2 mapping; never retain it across save()."""
    return _ledger_meta(led).setdefault("fable2", {})


def _resolve_seed() -> int:
    """OTR_FABLE2_SEED reproduces the frame-card/stance deal AND the
    announcer voice draw (r3/S2); OS entropy otherwise. The resolved
    value is stamped in meta.fable2.seed either way."""
    raw = os.environ.get("OTR_FABLE2_SEED", "").strip()
    if raw:
        try:
            return int(raw)
        except ValueError:
            raise Fable2Error(
                "deal", f"OTR_FABLE2_SEED must be an int, got {raw!r}")
    return random.SystemRandom().randrange(2 ** 63)


_RE_NUMERAL = re.compile(r"\d[\d,.]*")
_RE_PROPER = re.compile(r"\b[A-Z][a-zA-Z'\-]+\b")
_PROPER_STOPWORDS = frozenset({
    "The", "A", "An", "In", "On", "At", "It", "Its", "This", "That",
    "These", "Those", "They", "We", "But", "And", "Or", "For", "From",
    "With", "When", "Where", "While", "After", "Before", "Tonight",
    "Scientists", "Researchers", "Astronomers", "Engineers", "Doctors",
    # Calendar vocabulary is era-neutral factual date language, not an
    # invented entity (first live smoke 2026-07-10: a legitimate read
    # died on 'June').
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
    "Sunday", "Spring", "Summer", "Autumn", "Fall", "Winter",
    # Honorifics are titles, not entities (sixth live smoke 2026-07-10:
    # a legitimate read died on 'Dr' before 'Raman', which WAS in the
    # dossier). The surname beside them still faces the corpus check.
    "Dr", "Doctor", "Mr", "Mrs", "Ms", "Miss", "Professor", "Prof",
    "Sir", "Madam", "Captain", "Commander", "Colonel", "Lieutenant",
    "Sergeant", "Reverend",
    # Sentence-opening discourse words (kibitz r2 M2, 2026-07-10: the
    # blanket sentence-initial skip was an INVENTION HOLE -- an invented
    # source name at sentence start sailed through -- so every capital
    # now faces the corpus and only genuinely generic words are exempt).
    "However", "Meanwhile", "Across", "Together", "Still", "Yet",
    "According", "Since", "Because", "Despite", "Beyond", "Inside",
    "Outside", "Under", "Over", "Near", "Now", "Then", "Here", "There",
    "Every", "Each", "Some", "Many", "Most", "More", "One", "Two",
    "Three", "First", "Second", "Third", "Last", "Next", "New", "Old",
    "Today", "Yesterday", "Tomorrow", "Earlier", "Later", "Recently",
    "Soon", "Instruments", "Teams", "Officials",
})


def _word_re(term: str) -> "re.Pattern[str]":
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


def _scan_lexicon(text: str, lexicon) -> "str | None":
    """First word-boundary lexicon hit in text, or None. Word-boundary by
    design: 'gun' must never hit 'begun'."""
    for term in lexicon:
        if _word_re(term).search(text):
            return term
    return None


# SFW early gates on pitch/treatment prose (weapons/smoking only -- the
# audit's other kill classes need the performed script to exist).
_EARLY_SFW_LEXICON = WEAPON_SMOKING_EVIDENCE


# Spelled-number equivalence (16th live smoke 2026-07-10: the story said
# "seven days" / "Eighth day"; the dossier correctly extracted 7 and 8 as
# digits and the verbatim gate killed a faithful extraction).
_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven",
         "eight", "nine", "ten", "eleven", "twelve", "thirteen",
         "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
         "nineteen")
_TENS = {20: "twenty", 30: "thirty", 40: "forty", 50: "fifty",
         60: "sixty", 70: "seventy", 80: "eighty", 90: "ninety"}
_ORDINALS = ("zeroth", "first", "second", "third", "fourth", "fifth",
             "sixth", "seventh", "eighth", "ninth", "tenth", "eleventh",
             "twelfth", "thirteenth", "fourteenth", "fifteenth",
             "sixteenth", "seventeenth", "eighteenth", "nineteenth")

# Title tokens stripped from cast-name LABELS (19th live smoke).
_HONORIFIC_TOKENS = frozenset({
    "DR", "DOCTOR", "MR", "MRS", "MS", "MISS", "PROF", "PROFESSOR",
    "CAPTAIN", "CAPT", "COMMANDER", "CMDR", "COLONEL", "COL",
    "LIEUTENANT", "LT", "SERGEANT", "SGT", "REVEREND", "REV", "SIR",
    "MADAM", "MAJOR", "MAJ", "GENERAL", "GEN",
})

# Names that appear in the pack's FORMAT/schema EXAMPLES (kibitz r4 M1:
# roll 22 aired a VERA/DOKU cast copied from the few-shot example). A
# generated cast may not reuse them unless the SOURCE story itself
# carries the name.
_RESERVED_EXAMPLE_NAMES = frozenset({"VERA", "DOKU", "BRANNIGAN", "VOSS"})


def _spelled_forms(tok: str) -> "tuple[str, ...]":
    """Cardinal + ordinal word forms for a small integer token (0-100),
    hyphen and space compound variants included; () for anything else."""
    if not tok.isdigit():
        return ()
    n = int(tok)
    if n > 100:
        return ()
    forms: "list[str]" = []
    if n < 20:
        forms.append(_ONES[n])
        forms.append(_ORDINALS[n])
    elif n == 100:
        forms += ["hundred", "one hundred", "hundredth"]
    else:
        tens, ones = (n // 10) * 10, n % 10
        base = _TENS[tens]
        if ones == 0:
            forms += [base, base.rstrip("y") + "ieth"]
        else:
            forms += [f"{base}-{_ONES[ones]}", f"{base} {_ONES[ones]}",
                      f"{base}-{_ORDINALS[ones]}",
                      f"{base} {_ORDINALS[ones]}"]
    return tuple(forms)


# ---------------------------------------------------------------------------
# Frame deck + deal (doc s9/s13; entropy data, JSON owns content)
# ---------------------------------------------------------------------------

def _load_frame_deck(path: "Path | None" = None) -> dict:
    p = Path(path) if path is not None else _DECK_PATH
    try:
        deck = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Fable2PitchError(
            "pitch_room", f"frame deck unreadable at {p}: {exc}") from exc
    if deck.get("schema_version") != "fable2_deck_v1":
        raise Fable2PitchError(
            "pitch_room",
            f"frame deck schema_version "
            f"{deck.get('schema_version')!r} != 'fable2_deck_v1'")
    cards = deck.get("cards") or []
    stances = deck.get("stances") or []
    names = [c.get("name") for c in cards]
    if len(cards) < 3 or len(set(names)) != len(names):
        raise Fable2PitchError(
            "pitch_room",
            f"frame deck needs >= 3 uniquely-named cards, got {len(cards)}")
    if not stances:
        raise Fable2PitchError("pitch_room", "frame deck has no stances")
    return deck


def _deal(rng: random.Random, deck: dict, *, mode: Fable2Mode):
    """Deal (frame_cards, stance). one_pitch mode deals 1 card; full mode
    deals 3 distinct cards (pitch i uses card i)."""
    if mode == _COMPACT_MODE:
        count = 1
    elif mode == _FULL_MODE:
        count = 3
    else:  # pragma: no cover - Literal plus runtime fail-loud boundary
        raise Fable2PitchError("pitch_room", f"unknown fable2 mode {mode!r}")
    cards = tuple(rng.sample(list(deck["cards"]), count))
    stance = rng.choice(list(deck["stances"]))
    return cards, stance


# ---------------------------------------------------------------------------
# Scene envelope (python computes BEFORE any LLM call; doc s3/s13)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneEnvelope:
    scene_count: int
    scene_word_targets: "tuple[int, ...]"
    total_words: int
    band_frac: float = _TOTAL_WORD_BAND

    def __post_init__(self) -> None:
        if type(self.scene_count) is not int or type(self.total_words) is not int:
            raise TypeError("scene_count and total_words must be strict ints")
        if self.total_words < 1 or self.total_words > _SUPPORTED_WORD_CEILING:
            raise ValueError(
                f"total_words must be 1-{_SUPPORTED_WORD_CEILING}, got "
                f"{self.total_words}")
        if (not isinstance(self.scene_word_targets, tuple)
                or any(type(value) is not int or value < 1
                       for value in self.scene_word_targets)):
            raise TypeError(
                "scene_word_targets must be an immutable tuple of positive "
                "strict ints")
        if type(self.band_frac) is not float or self.band_frac != _TOTAL_WORD_BAND:
            raise ValueError(
                f"band_frac must equal the canonical {_TOTAL_WORD_BAND}")
        expected_scenes = next(
            scene_count for upper_bound, scene_count in _SCENE_COUNT_TABLE
            if self.total_words <= upper_bound
        )
        quotient, remainder = divmod(self.total_words, expected_scenes)
        expected_targets = tuple(
            quotient + (1 if index < remainder else 0)
            for index in range(expected_scenes)
        )
        if self.scene_count != expected_scenes:
            raise ValueError(
                f"scene_count {self.scene_count} does not match canonical "
                f"table value {expected_scenes} for {self.total_words} words")
        if self.scene_word_targets != expected_targets:
            raise ValueError(
                "scene_word_targets do not match the canonical "
                "quotient/remainder vector")

    @property
    def per_scene_words(self) -> int:
        """Compatibility view for compact callers; exact code uses vector."""
        return max(1, round(self.total_words / self.scene_count))

    @property
    def scene_word_bands(self) -> "tuple[tuple[int, int], ...]":
        return tuple(
            (
                math.ceil(target * (1.0 - _SCENE_WORD_BAND)),
                math.floor(target * (1.0 + _SCENE_WORD_BAND)),
            )
            for target in self.scene_word_targets
        )


def _scene_count_for_words(target_words: int) -> int:
    tw = int(target_words)
    for upper_bound, scene_count in _SCENE_COUNT_TABLE:
        if tw <= upper_bound:
            return scene_count
    return _SCENE_COUNT_TABLE[-1][1]


def _build_envelope(target_words: int) -> SceneEnvelope:
    """Build the pinned scene table plus exact quotient/remainder vector."""
    tw = int(target_words)
    if tw > _SUPPORTED_WORD_CEILING:
        raise Fable2ScriptError(
            "script",
            f"target_words {tw} above supported ceiling "
            f"{_SUPPORTED_WORD_CEILING}",
        )
    scenes = _scene_count_for_words(tw)
    quotient, remainder = divmod(tw, scenes)
    targets = tuple(
        quotient + (1 if i < remainder else 0)
        for i in range(scenes)
    )
    return SceneEnvelope(
        scene_count=scenes,
        scene_word_targets=targets,
        total_words=tw,
    )


def _validate_scene_envelope(envelope: Any) -> None:
    if type(envelope) is not SceneEnvelope:
        raise Fable2ScriptError(
            "revision_contract",
            "envelope must be an exact canonical SceneEnvelope artifact",
        )
    canonical = _build_envelope(envelope.total_words)
    if envelope != canonical:
        raise Fable2ScriptError(
            "revision_contract",
            "envelope does not match the pinned scene-count/vector contract",
        )


@dataclass(frozen=True)
class PassAttemptTrace:
    """One observed P3/P5 markup call; never an inferred attempt count."""

    attempt: int
    temperature: float
    requested_max_new_tokens: int
    outcome: Literal[
        "parse_rejected", "budget_rejected", "truncated", "accepted"
    ]
    truncation: bool = False
    selected: bool = False
    parse_defects: "tuple[str, ...]" = ()
    character_words: "int | None" = None
    scene_character_words: "tuple[int, ...]" = ()

    def __post_init__(self) -> None:
        valid_outcomes = {
            "parse_rejected", "budget_rejected", "truncated", "accepted",
        }
        if type(self.attempt) is not int or type(
                self.requested_max_new_tokens) is not int:
            raise TypeError(
                "attempt and requested_max_new_tokens must be strict ints")
        if self.attempt < 1 or self.requested_max_new_tokens < 1:
            raise ValueError("attempt and requested_max_new_tokens must be positive")
        if (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(float(self.temperature))):
            raise TypeError("temperature must be one finite number")
        if self.outcome not in valid_outcomes:
            raise ValueError(
                f"outcome must be one of {sorted(valid_outcomes)}, got "
                f"{self.outcome!r}")
        if type(self.truncation) is not bool or type(self.selected) is not bool:
            raise TypeError("truncation and selected must be strict booleans")
        if (not isinstance(self.parse_defects, tuple)
                or any(not isinstance(value, str)
                       for value in self.parse_defects)):
            raise TypeError("parse_defects must be an immutable string tuple")
        if (self.character_words is not None
                and (type(self.character_words) is not int
                     or self.character_words < 0)):
            raise TypeError("character_words must be a non-negative strict int")
        if (not isinstance(self.scene_character_words, tuple)
                or any(type(value) is not int or value < 0
                       for value in self.scene_character_words)):
            raise TypeError(
                "scene_character_words must be an immutable tuple of "
                "non-negative strict ints")
        if self.selected != (self.outcome == "accepted"):
            raise ValueError("selected must be true exactly for an accepted attempt")
        if self.truncation != (self.outcome == "truncated"):
            raise ValueError("truncation must be true exactly for a truncated attempt")


@dataclass(frozen=True)
class ConstituentProof:
    text: str
    artifact: str
    span: "tuple[int, int]"


@dataclass(frozen=True)
class ProofMapEntry:
    line_id: str
    constituents: "tuple[ConstituentProof, ...]"


@dataclass(frozen=True)
class FinalDraftHashes:
    raw_sha256: str
    normalized_sha256: str
    parsed_sha256: str
    proof_map_sha256: str
    artifact_sha256: str


DraftScore = tuple[int, int, int, int, int]


@dataclass(frozen=True)
class FinalDraft:
    """One atomic script authority selected before any ledger mutation."""

    raw_source: str
    normalized_source: str
    parsed: ParsedScript
    proof_map: "tuple[ProofMapEntry, ...]"
    p3_attempts: "tuple[PassAttemptTrace, ...]"
    p5_attempts: "tuple[PassAttemptTrace, ...]"
    score: DraftScore
    scoring_context_sha256: str
    hashes: FinalDraftHashes


@dataclass(frozen=True)
class RevisionContractResult:
    eligible: bool
    reasons: "tuple[str, ...]"
    addressed_problem_classes: "tuple[str, ...]"
    changed_scopes: "tuple[str, ...]"
    protected_field_differences: "tuple[str, ...]"
    total_word_count: int
    total_word_band: "tuple[int, int]"
    scene_word_counts: "tuple[int, ...]"
    scene_word_bands: "tuple[tuple[int, int], ...]"
    draft1_sha256: str
    draft2_sha256: str
    scoring_context_sha256: str
    contract_sha256: str


@dataclass(frozen=True)
class DraftSelection:
    winner: FinalDraft
    reason: str
    scoring_context_sha256: str


# ---------------------------------------------------------------------------
# Voice menu (doc s9: stable ids; python owns the larynx)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VoiceMenuEntry:
    menu_id: str
    gender: str
    description: str
    preset: str          # python-side only; never shown to the LLM


@dataclass(frozen=True)
class VoiceMenu:
    entries: "tuple[VoiceMenuEntry, ...]"

    def by_id(self) -> "dict[str, VoiceMenuEntry]":
        return {e.menu_id: e for e in self.entries}

    def prompt_lines(self) -> str:
        return "\n".join(
            f"- {e.menu_id} ({e.gender}): {e.description}"
            for e in self.entries
        )


def _deal_voice_menu(cast_size: int) -> VoiceMenu:
    """Menu derived at runtime from config/cast_pools (r1/S3 + r2/S4):
    the menu only ever offers what exists. Entries carry STABLE IDS
    (descriptions are not guaranteed unique); P6 orders BY menu_id and
    python validates BY menu_id. Preflight (r4/S1 stage 1): total usable
    capacity >= cast_size BEFORE P6 (gender is P6's choice, so
    gender-compatibility cannot be pre-checked)."""
    gender_by_preset = {
        p: g for p, g, _lang, _tags in _POOLS.VOICE_PROFILES
    }
    pool = _POOLS.open_voice_pool(set())
    entries = tuple(
        VoiceMenuEntry(
            menu_id=f"m{i + 1:02d}",
            gender=gender_by_preset.get(preset, "male"),
            description=short,
            preset=preset,
        )
        for i, (preset, short) in enumerate(pool)
    )
    if len(entries) < int(cast_size):
        raise Fable2CastError(
            "casting_voices",
            f"voice stock capacity {len(entries)} < cast size {cast_size} "
            f"(preflight, before any P6 LLM call)")
    return VoiceMenu(entries)


# ---------------------------------------------------------------------------
# post_validator factories (r2/S1: runtime-state checks live here, not in
# model validators)
# ---------------------------------------------------------------------------

def _make_dossier_validator(digest: str):
    """Copy-never-invent gate: every allowed number must appear verbatim
    in the capped source digest; every named entity's CONTENT TOKENS must
    all appear there (word-boundary). Token-level for entities because a
    faithful extraction may reorder a phrase -- the second S1b live smoke
    (2026-07-10) killed a grounded dossier over "Amsterdam's canals" vs
    the story's "the canals of Amsterdam". Typographic apostrophes are
    normalized; possessive 's is stripped before the token check."""
    hay = _norm_ws(digest).casefold().replace("’", "'")

    def _check(m: DossierLLM) -> "str | None":
        # Neither entities NOR numbers are rerolled here: unverifiable
        # ones are DROPPED by _filter_dossier_entities after the call
        # (28th + 30th live smokes 2026-07-10: world-knowledge
        # expansions and converted figures are unbounded; no reroll can
        # fix knowledge). Delete-only filtering keeps the anti-invention
        # property -- dropping an allowed number only SHRINKS what the
        # factual read may speak.
        return None

    return _check


def _entity_in_source(ent: str, hay: str, hay_words: "set[str]") -> bool:
    for tok in _norm_ws(ent).replace("’", "'").split():
        tok = tok.strip(".,;:!?'\"()").casefold()
        if tok.endswith("'s"):
            tok = tok[:-2]
        if len(tok) <= 2:
            continue
        if _word_re(tok).search(hay):
            continue
        # Demonym/inflection tolerance (24th live smoke: 'Scottish' vs
        # 'Scotland'): a shared prefix of >= 4 chars with any source
        # word is the same referent.
        if any(len(tok) >= 4 and w[:4] == tok[:4]
               for w in hay_words if len(w) >= 4):
            continue
        return False
    return True


def _filter_dossier_entities(dossier: DossierLLM,
                             digest: str) -> "tuple[DossierLLM, list[str]]":
    """Delete-only entity filtering (python judges): entities the source
    text cannot corroborate are DROPPED from the extraction and reported
    -- never rerolled (world-knowledge expansions like CMU -> Carnegie
    Mellon are unbounded) and never allowed to widen the READ gate's
    legality corpus."""
    hay = _norm_ws(digest).casefold().replace("’", "'")
    hay_words = {w.strip(".,;:!?'\"()") for w in hay.split()}
    dropped: "list[str]" = []
    kept: "dict[str, list[str]]" = {}
    for field_name in ("people", "places", "things"):
        vals = getattr(dossier.named_entities, field_name)
        kept[field_name] = []
        for ent in vals:
            if _entity_in_source(ent, hay, hay_words):
                kept[field_name].append(ent)
            else:
                dropped.append(ent)
    # Numbers too (30th live smoke): an allowed number the source text
    # cannot corroborate (verbatim or spelled) is dropped -- shrinking
    # the read's numeral legality, never widening it.
    kept_numbers: "list[str]" = []
    for num in dossier.allowed_numbers:
        norm = _norm_ws(num).casefold().replace("’", "'")
        if norm in hay or any(
                _word_re(f).search(hay)
                for f in _spelled_forms(norm.rstrip(".,"))):
            kept_numbers.append(num)
        else:
            dropped.append(num)
    if not dropped:
        return dossier, []
    return dossier.model_copy(update={
        "named_entities": NamedEntities(**kept),
        "allowed_numbers": kept_numbers}), dropped


def _make_pitch_validator(cards, mode: Fable2Mode, n_max: int):
    """Dealt-card + divergence + SFW gates. `cards` is the dealt tuple;
    the count expectation is a parameter (r2/M3)."""
    if mode == _COMPACT_MODE:
        expected = 1
    elif mode == _FULL_MODE:
        expected = 3
    else:
        raise Fable2PitchError("pitch_room", f"unknown fable2 mode {mode!r}")
    card_names = [c["name"] for c in cards]

    def _check(m: PitchSlate) -> "str | None":
        if len(m.pitches) != expected:
            return f"need exactly {expected} pitch(es), got {len(m.pitches)}"
        expected_ids = list(range(1, expected + 1))
        actual_ids = [p.pitch_id for p in m.pitches]
        if actual_ids != expected_ids:
            return (
                f"pitch ids must be exactly {expected_ids} in dealt-card "
                f"order, got {actual_ids}"
            )
        for i, p in enumerate(m.pitches):
            if p.frame_card != card_names[i]:
                return (f"pitch {i + 1} frame_card {p.frame_card!r} != dealt "
                        f"card {card_names[i]!r} (pitch i uses card i)")
            if p.cast_size > n_max:
                return f"pitch {i + 1} cast_size {p.cast_size} > N_MAX {n_max}"
            hit = _scan_lexicon(
                " ".join([p.logline, p.hook, p.scifi_device]),
                _EARLY_SFW_LEXICON)
            if hit:
                return f"pitch {i + 1} carries forbidden term {hit!r} (SFW)"
        if expected == 3:
            shapes = {(p.cast_size, p.ending_shape) for p in m.pitches}
            if len(shapes) == 1:
                return ("all three pitches share cast_size AND ending_shape "
                        "-- they must diverge")
            for i in range(3):
                for j in range(i + 1, 3):
                    ov = _token_overlap(
                        m.pitches[i].logline, m.pitches[j].logline)
                    if ov >= _LOGLINE_OVERLAP_MAX:
                        return (f"loglines {i + 1}/{j + 1} overlap "
                                f"{ov:.2f} >= {_LOGLINE_OVERLAP_MAX}")
        return None

    return _check


def _make_select_validator(slate: PitchSlate):
    expected_ids = [1, 2, 3]
    actual_ids = [pitch.pitch_id for pitch in slate.pitches]
    if actual_ids != expected_ids:
        raise Fable2SelectError(
            "pitch_select",
            f"selection requires exact pitch slate {expected_ids}, got "
            f"{actual_ids}",
        )
    allowed = frozenset(actual_ids)

    def _check(selection: PitchSelect) -> "str | None":
        if selection.chosen_pitch_id not in allowed:
            return (
                f"chosen_pitch_id {selection.chosen_pitch_id} is not an "
                f"existing slate member {sorted(allowed)}"
            )
        return None

    return _check


def _make_critic_validator(parsed: ParsedScript):
    """Every P4 note must name a real, explicitly-owned draft scope."""
    if type(parsed) is not ParsedScript:
        raise Fable2ScriptError(
            "critic", "critic input must be an exact ParsedScript artifact")

    def _check(notes: CriticNotes) -> "str | None":
        errors = []
        for index, note in enumerate(notes.notes, start=1):
            error = _note_scope_error(note, parsed)
            if error:
                errors.append(f"note {index}: {error}")
        return "; ".join(errors) if errors else None

    return _check


def _make_treatment_validator(dossier: DossierLLM, n_max: int,
                              provenance: "dict[str, str]",
                              digest: str = ""):
    """Grounding gates for P2b: cast ceiling, news_thread grounded in the
    dossier, SFW. The factual-read subset laws moved to
    ``_make_read_validator`` with the P2c read-split (S1b, 2026-07-10)."""
    ents = (dossier.named_entities.people + dossier.named_entities.places
            + dossier.named_entities.things)
    # Thread grounding corpus includes the SOURCE DIGEST (29th live
    # smoke 2026-07-10: a thread naming the story's own subject failed
    # because the filtered dossier alone was the corpus -- the source is
    # the legality authority everywhere).
    dossier_corpus = _norm_ws(" ".join(
        dossier.facts_to_keep + dossier.dramatizable_vectors + ents
        + dossier.allowed_numbers + [str(digest)])).casefold()

    example_hay = _norm_ws(str(digest)).casefold()

    def _check(m: Treatment) -> "str | None":
        # kibitz r2 S3: ACCUMULATE every content failure so the single
        # typed-repair attempt sees the whole target.
        errs: "list[str]" = []
        if len(m.cast_shapes) > n_max:
            errs.append(f"cast_shapes {len(m.cast_shapes)} > N_MAX {n_max}")
        # kibitz r4 M1 (roll 22 aired the few-shot example's cast): the
        # FORMAT-example names are reserved unless the source story
        # itself carries the name.
        for shape in m.cast_shapes:
            if (shape.name in _RESERVED_EXAMPLE_NAMES
                    and not _word_re(shape.name).search(example_hay)):
                errs.append(
                    f"cast name {shape.name!r} copies a FORMAT-example "
                    f"name -- invent a period-plausible name of your own")
        thread_tokens = [
            t.strip(".,;:!?'\"()").casefold()
            for t in m.news_thread.split() if len(t) > 3
        ]
        if not any(t and t in dossier_corpus for t in thread_tokens):
            errs.append(
                "news_thread shares no content noun with the dossier "
                "-- it must extrapolate the real science")
        hit = _scan_lexicon(
            " ".join([m.title, m.setting, m.turn,
                      m.priced_ending.get("choice", ""),
                      m.priced_ending.get("cost_paid", "")]
                     + [f"{c.role} {c.want} {c.pressure}"
                        for c in m.cast_shapes]),
            _EARLY_SFW_LEXICON)
        if hit:
            errs.append(f"treatment carries forbidden term {hit!r} (SFW)")
        if errs:
            return "; ".join(errs[:6])
        return None

    return _check


def _make_read_validator(dossier: DossierLLM,
                         provenance: "dict[str, str]",
                         digest: str,
                         cast_names: "list[str]"):
    """P2c factual-read subset laws (r4/S2 anti-invention direction).

    LEGALITY AUTHORITY = THE SOURCE (S1b live-smoke hardening 2026-07-10,
    rolls 1/6/7/8): the noun + numeral corpora are the WHOLE dossier +
    the python-stamped provenance + the python-capped SOURCE DIGEST.
    Nothing outside the source can pass; invented numerals and names die
    here. Word-boundary matching (never substring); numerals token-exact
    (kibitz r2 M3); NO sentence-initial skip (kibitz r2 M2 -- it was an
    invention hole); possessives are grammar; the drama's characters are
    hard-banned CASE-SENSITIVELY (a cast named HOPE must not outlaw the
    word 'hope'). All failures accumulate into ONE teaching message."""
    ents = (dossier.named_entities.people + dossier.named_entities.places
            + dossier.named_entities.things)
    noun_corpus = _norm_ws(" ".join(
        dossier.facts_to_keep + dossier.dramatizable_vectors + ents
        + list(provenance.values()) + [str(digest)]))
    legal_numerals = {
        t.rstrip(".,")
        for src in (list(dossier.allowed_numbers) + [str(digest)])
        for t in _RE_NUMERAL.findall(src)
    }
    source_hay = _norm_ws(
        " ".join(dossier.facts_to_keep + [str(digest)])).casefold()

    def _check(m: NewsCloseRead) -> "str | None":
        errs: "list[str]" = []
        text = m.news_close_read
        for name in cast_names:
            # SOURCE LEGALITY BEATS THE FICTIONAL BAN (14th live smoke
            # 2026-07-10: the drama named its character after the REAL
            # person in the story -- 'JIM' for NASA's Jim Ross -- and the
            # ban outlawed factual reporting of the real name).
            if _word_re(name).search(noun_corpus):
                continue
            name_re = re.compile(
                r"\b(?:" + re.escape(name) + r"|"
                + re.escape(name.title()) + r")\b")
            if name_re.search(text):
                errs.append(
                    f"the read names the fictional character {name!r} -- "
                    f"this is REAL NEWS about the source story: use only "
                    f"real names from the source, never the drama's "
                    f"characters")
        for tok in _RE_NUMERAL.findall(text):
            tok = tok.rstrip(".,")
            if tok in legal_numerals:
                continue
            # spelled-number equivalence, same as the dossier gate (23rd
            # live smoke 2026-07-10: the story spelled 'twenty')
            if any(_word_re(f).search(source_hay)
                   for f in _spelled_forms(tok)):
                continue
            errs.append(
                f"numeral {tok!r} not in allowed_numbers "
                f"{dossier.allowed_numbers} or the source text -- "
                f"never compute or invent a number")
        for match in _RE_PROPER.finditer(text):
            word = match.group(0)
            if word.endswith("'s"):
                word = word[:-2]
            if word in _PROPER_STOPWORDS:
                continue
            if not _word_re(word).search(noun_corpus):
                errs.append(
                    f"proper noun {word!r} not in the dossier or "
                    f"provenance -- use only real names from the source "
                    f"story")
        if errs:
            return "; ".join(errs[:6])
        return None

    return _check


def _make_casting_validator(menu: VoiceMenu, speakers: "list[str]"):
    """Speaker-set equality + menu-id legality + gender/timbre feasibility
    (r4/S1 stage 2: validated AFTER P6, rerolled via the ladder)."""
    by_id = menu.by_id()
    want = set(speakers)

    def _check(m: CastingVoices) -> "str | None":
        got = [c.name for c in m.cast]
        if len(set(got)) != len(got):
            return f"duplicate cast names {got}"
        if set(got) != want:
            return (f"cast names {sorted(got)} != script speakers "
                    f"{sorted(want)} -- cast EXACTLY the script's speakers")
        taken: set[str] = set()
        for c in m.cast:
            entry = by_id.get(c.timbre)
            if entry is None:
                return (f"{c.name}: timbre {c.timbre!r} is not a menu id "
                        f"(pick from the AVAILABLE VOICE STOCK)")
            if entry.gender != c.gender:
                legal = [e.menu_id for e in menu.entries
                         if e.gender == c.gender]
                return (f"{c.name}: menu id {c.timbre} is {entry.gender}, "
                        f"but gender is {c.gender!r} -- pick one of the "
                        f"{c.gender} ids instead: {', '.join(legal)}")
            if c.timbre in taken:
                return (f"{c.name}: timbre {c.timbre} already taken -- "
                        f"two characters never share a voice")
            taken.add(c.timbre)
        return None

    return _check


# ---------------------------------------------------------------------------
# LLM passes (doc s3/s8/s10)
# ---------------------------------------------------------------------------

def _build_digest(payload: "dict[str, Any]") -> str:
    """Python-capped source digest for P0/P3 (never the raw feed)."""
    head = (
        f"HEADLINE: {payload.get('headline', '')}\n"
        f"SOURCE: {payload.get('source', '')} {payload.get('date', '')}\n"
        f"SUMMARY: {payload.get('summary', '')}\n\n"
    )
    body = str(payload.get("full_text") or payload.get("seed_text") or "")
    return (head + body)[:_DIGEST_CHAR_CAP]


def _pass_dossier(technical_fn, pack, digest: str) -> DossierLLM:
    try:
        # LLM slot: technical -- P0 dossier (structured JSON extraction).
        return structured_call(
            prompt=[
                {"role": "system",
                 "content": _seam(pack, "fable2_dossier_system")},
                {"role": "user", "content": f"SCIENCE STORY:\n{digest}"},
            ],
            schema=DossierLLM,
            slot_fn=technical_fn,
            base_temperature=_TEMP["dossier"],
            structural_retry_temperature=_TEMP["dossier"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_dossier_validator(digest),
            max_new_tokens=_MAX_NEW_TOKENS["dossier"],
            helper_name="fable2_dossier",
        )
    except StructuredCallFailedError as exc:
        raise Fable2DossierError(
            "dossier", str(exc.last_error), exc.attempts) from exc


def _pass_pitch(creative_fn, pack, dossier: DossierLLM, cards, stance,
                *, n_max: int, mode: Fable2Mode) -> PitchSlate:
    if mode == _COMPACT_MODE:
        count = 1
    elif mode == _FULL_MODE:
        count = 3
    else:
        raise Fable2PitchError("pitch_room", f"unknown fable2 mode {mode!r}")
    cards_text = "\n".join(
        f"CARD {i + 1}: {c['name']} -- {c['shape']}"
        for i, c in enumerate(cards))
    user = (
        f"SOURCE DOSSIER:\n"
        f"{json.dumps(dossier.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"FRAME CARD(S) DEALT (pitch i uses card i):\n{cards_text}\n\n"
        f"EDITORIAL STANCE: {stance['name']} -- {stance['note']}\n\n"
        f"REQUESTED NUMBER OF PITCHES: {count}\n"
        f"N_MAX (speaking-character ceiling): {n_max}\n"
        f"Pitch now."
    )
    try:
        # LLM slot: creative -- P1 pitch room (CREATIVE 1).
        return structured_call(
            prompt=[
                {"role": "system",
                 "content": _seam(pack, "fable2_pitch_system")},
                {"role": "user", "content": user},
            ],
            schema=PitchSlate,
            slot_fn=creative_fn,
            base_temperature=_TEMP["pitch_room"],
            structural_retry_temperature=round(_TEMP["pitch_room"] / 2, 2),
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_pitch_validator(cards, mode, n_max),
            max_new_tokens=_MAX_NEW_TOKENS["pitch_room"],
            helper_name="fable2_pitch_room",
        )
    except StructuredCallFailedError as exc:
        raise Fable2PitchError(
            "pitch_room", str(exc.last_error), exc.attempts) from exc


def _pass_select(technical_fn, pack, dossier: DossierLLM,
                 slate: PitchSlate, stance) -> PitchSelect:
    """P2a: select one existing member of the exact three-pitch slate."""
    user = (
        "SOURCE DOSSIER:\n"
        f"{json.dumps(dossier.model_dump(), ensure_ascii=False, indent=2)}"
        "\n\nEDITORIAL STANCE:\n"
        f"{json.dumps(stance, ensure_ascii=False, indent=2)}"
        "\n\nTHREE-PITCH SLATE:\n"
        f"{json.dumps(slate.model_dump(), ensure_ascii=False, indent=2)}"
        "\n\nSelect one existing pitch now."
    )
    try:
        # LLM slot: technical -- P2a pitch selection.
        return structured_call(
            prompt=[
                {"role": "system",
                 "content": _seam(pack, "fable2_select_system")},
                {"role": "user", "content": user},
            ],
            schema=PitchSelect,
            slot_fn=technical_fn,
            base_temperature=_TEMP["pitch_select"],
            structural_retry_temperature=_TEMP["pitch_select"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_select_validator(slate),
            max_new_tokens=_MAX_NEW_TOKENS["pitch_select"],
            helper_name="fable2_pitch_select",
        )
    except StructuredCallFailedError as exc:
        raise Fable2SelectError(
            "pitch_select", str(exc.last_error), exc.attempts) from exc


def _pass_treatment(creative_fn, pack, dossier: DossierLLM, pitch: Pitch,
                    stance, *, n_max: int,
                    provenance: "dict[str, str]",
                    digest: str = "") -> Treatment:
    user = (
        f"SOURCE DOSSIER:\n"
        f"{json.dumps(dossier.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"EDITORIAL STANCE: {stance['name']} -- {stance['note']}\n\n"
        f"WINNING PITCH:\n"
        f"{json.dumps(pitch.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"N_MAX (speaking-character ceiling): {n_max}\n"
        f"Write the treatment now."
    )
    try:
        # LLM slot: creative -- P2b treatment (cast names born here).
        return structured_call(
            prompt=[
                {"role": "system",
                 "content": _seam(pack, "fable2_treatment_system")},
                {"role": "user", "content": user},
            ],
            schema=Treatment,
            slot_fn=creative_fn,
            base_temperature=_TEMP["treatment"],
            structural_retry_temperature=_TEMP["treatment"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_treatment_validator(
                dossier, n_max, provenance, digest),
            max_new_tokens=_MAX_NEW_TOKENS["treatment"],
            helper_name="fable2_treatment",
        )
    except StructuredCallFailedError as exc:
        raise Fable2TreatmentError(
            "treatment", str(exc.last_error), exc.attempts) from exc


def _pass_news_read(technical_fn, pack, dossier: DossierLLM,
                    provenance: "dict[str, str]", digest: str,
                    cast_names: "list[str]") -> NewsCloseRead:
    """P2c (read-split, S1b 2026-07-10): the factual close gets its own
    single-purpose low-temp technical pass -- 7 of the first 13 live
    rolls died when one combined P2b call had to satisfy the whole
    treatment AND a subset-law-compliant read."""
    user = (
        f"SOURCE DOSSIER:\n"
        f"{json.dumps(dossier.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"PROVENANCE: {json.dumps(provenance, ensure_ascii=False)}\n\n"
        # kibitz r3 M1: the validator authorizes source-digest facts, so
        # the model must SEE the digest it may quote from.
        f"SOURCE STORY TEXT (you may use its names and numbers "
        f"verbatim):\n{digest}\n\n"
        f"FORBIDDEN NAMES (the drama's fictional characters -- never use "
        f"them here): {', '.join(cast_names) or '(none)'}\n\n"
        f"Write the closing news read now."
    )
    validator = _make_read_validator(dossier, provenance, digest, cast_names)
    # Bounded outer retry (32nd live smoke 2026-07-10: on a thin source
    # the model keeps reaching for a figure; the SECOND outer attempt
    # demands a NUMERAL-FREE read, which cannot fail the numeral gate).
    last_exc: "Exception | None" = None
    for outer in range(2):
        prompt_user = user if outer == 0 else (
            user + "\n\nHARD CONSTRAINT for this attempt: write the read "
            "with NO numerals at all -- no figures, no dates; state the "
            "finding in plain words only."
        )
        try:
            # LLM slot: technical -- P2c factual close (read-split).
            return structured_call(
                prompt=[
                    {"role": "system",
                     "content": _seam(pack, "fable2_news_read_system")},
                    {"role": "user", "content": prompt_user},
                ],
                schema=NewsCloseRead,
                slot_fn=technical_fn,
                base_temperature=0.20,
                structural_retry_temperature=0.10,
                repair_prompt_factory=make_dispatching_repair_factory(),
                post_validator=validator,
                max_new_tokens=300,
                helper_name="fable2_news_read",
            )
        except StructuredCallFailedError as exc:
            last_exc = exc
            log.warning(
                "[scifi_fable2] news_read outer attempt %d exhausted "
                "(%s)%s", outer + 1, exc.last_error,
                "; retrying with the numeral-free hard constraint"
                if outer == 0 else "",
            )
    raise Fable2TreatmentError(
        "news_read", str(getattr(last_exc, "last_error", last_exc)),
        getattr(last_exc, "attempts", 0)) from last_exc


def _micro_episode_line_cap(target_words: int) -> int:
    """Tiny-budget structural cap: dialogue LINES, not word arithmetic
    (20th live smoke 2026-07-10: the model wrote 107 character words for
    target 30 through every numeric hint -- small models follow concrete
    line counts, not word counts)."""
    return max(4, int(target_words) // 7)


def _script_user_prompt(treatment: Treatment, digest: str,
                        envelope: SceneEnvelope,
                        cast_names: "list[str]") -> str:
    cast_block = "\n".join(
        f"- {c.name} ({c.role}): wants {c.want}; pressure {c.pressure}; "
        f"register: {c.register}"
        for c in treatment.cast_shapes)
    micro = ""
    if envelope.total_words < 60:
        cap = _micro_episode_line_cap(envelope.total_words)
        micro = (
            f" MICRO-EPISODE (hard): this is under a minute of air -- "
            f"write AT MOST {cap} character dialogue lines TOTAL across "
            f"all scenes, each line under 10 words; every line must "
            f"carry the whole turn."
        )
    vector = ", ".join(
        f"scene {index}: target {target}, accepted {band[0]}-{band[1]}"
        for index, (target, band) in enumerate(zip(
            envelope.scene_word_targets,
            envelope.scene_word_bands), start=1)
    )
    return (
        f"TREATMENT:\n"
        f"{json.dumps(treatment.model_dump(), ensure_ascii=False, indent=2)}"
        f"\n\nCAST (the ONLY legal speakers besides ANNOUNCER; copy names "
        f"EXACTLY):\n{cast_block}\n\n"
        f"SOURCE DIGEST (fiction fuel, never quoted as news):\n{digest}\n\n"
        f"SCENE ENVELOPE: exactly {envelope.scene_count} scene(s); exact "
        f"ordered target/band vector [{vector}]; total CHARACTER dialogue "
        f"target {envelope.total_words} words (announcer lines are metered "
        f"separately and stay lean: 1-2 intro lines, 1-2 outro lines)."
        f"{micro}\n\n"
        # Recency anchor (first S1b live smokes, 2026-07-10): the local
        # 12B model reverts to screenplay habits unless the LAST thing it
        # reads re-states the format law.
        f"FORMAT REMINDER (hard): every line is exactly LABEL: spoken "
        f"words. PLAIN TEXT ONLY -- no markdown, no asterisks. NO "
        f"parentheses or performance notes anywhere: never (pauses), "
        f"never (raspy), never (over radio); delivery lives in the words "
        f"themselves. The ANNOUNCER speaks ONLY in the intro and the "
        f"outro -- never inside or between scenes (a MUSIC line may sit "
        f"between scenes instead). End with ANNOUNCER outro -> CODA -> "
        f"MUSIC (closing) -> END.\n\n"
        f"Write the complete episode now."
    )


def _extract_format_example(seam_text: str) -> str:
    """Pull the seam's FORMAT example play (TITLE: The Long Count .. END.)
    for the few-shot assistant turn. The seam remains the single source
    of truth; extraction failing = the pack drifted = fail loud."""
    start = seam_text.find("TITLE: The Long Count")
    end = seam_text.find("\nEND.", start)
    if start < 0 or end < 0:
        raise Fable2ScriptError(
            "script",
            "fable2_script_system seam lost its FORMAT example play -- "
            "the few-shot anchor cannot be built")
    return seam_text[start:end + len("\nEND.")]


def _script_budget_defects(parsed: ParsedScript,
                           envelope: SceneEnvelope) -> "tuple[str, ...]":
    """Return exact total, scene-count, and per-scene vector defects."""
    _validate_scene_envelope(envelope)
    errors = []
    lo, hi = _word_band(envelope.total_words)
    if not (lo <= parsed.character_word_count <= hi):
        errors.append(
            f"WORD_BUDGET: character words {parsed.character_word_count} "
            f"outside {int(lo)}-{int(hi)} (target {envelope.total_words})"
        )
    counts = _scene_character_word_counts(parsed)
    if len(counts) != envelope.scene_count:
        errors.append(
            f"SCENE_COUNT: parsed {len(counts)} scene(s), expected "
            f"{envelope.scene_count}"
        )
    for index, (count, target, band) in enumerate(zip(
            counts, envelope.scene_word_targets,
            envelope.scene_word_bands), start=1):
        if not (band[0] <= count <= band[1]):
            errors.append(
                f"SCENE_WORD_BUDGET: scene {index} character words {count} "
                f"outside {band[0]}-{band[1]} (target {target})"
            )
    return tuple(errors)


def _run_markup_ladder(
    creative_fn,
    *,
    pass_id: Literal["script", "revision"],
    system: str,
    base_user: str,
    envelope: SceneEnvelope,
    cast_names: "list[str]",
    initial_temperature: float,
    format_example: "str | None" = None,
) -> "tuple[str, ParsedScript, dict]":
    """Shared P3/P5 whole-play ladder with observed immutable traces."""
    temps = _MARKUP_LADDER_TEMPS(initial_temperature)
    tokens = _script_token_budget(envelope.total_words)
    defects_by_attempt: "list[list[str]]" = []
    traces: "list[PassAttemptTrace]" = []
    budget_rerolls = 0
    truncation_retry_used = False
    extra_user: "str | None" = None
    last_defect_text = ""

    for temp in temps:
        while True:
            attempt = len(traces) + 1
            requested_tokens = tokens
            user_content = (
                f"{base_user}\n\n{extra_user}" if extra_user else base_user)
            if format_example is None:
                msgs = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ]
            else:
                msgs = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": (
                        "Before the real assignment: show the exact output "
                        "FORMAT once, as a tiny example episode.")},
                    {"role": "assistant", "content": format_example},
                    {"role": "user", "content": user_content},
                ]
            # LLM slot: creative -- P3/P5 whole-play markup ladder.
            raw = creative_fn(
                msgs, temperature=temp, max_new_tokens=requested_tokens)
            parsed, defects = parse_fable2_markup(raw, cast_names)
            if parsed is None:
                defect_rows = tuple(str(defect) for defect in defects)
                rendered = render_defects(defects)
                codes = {defect.code for defect in defects}
                missing_end = Fable2ParseDefect.MISSING_END in codes
                traces.append(PassAttemptTrace(
                    attempt=attempt,
                    temperature=float(temp),
                    requested_max_new_tokens=requested_tokens,
                    outcome="truncated" if missing_end else "parse_rejected",
                    truncation=missing_end,
                    parse_defects=defect_rows,
                ))
                defects_by_attempt.append(list(defect_rows))
                last_defect_text = rendered
                if missing_end and not truncation_retry_used:
                    truncation_retry_used = True
                    tokens = int(tokens * 1.25)
                    extra_user = (
                        "Your previous draft was TRUNCATED (no END. line). "
                        "Write the COMPLETE episode again, top to bottom, "
                        "slightly tighter, and stop after END."
                    )
                    continue
                extra_user = (
                    "Your previous draft violated the FORMAT. Fix EVERY "
                    "defect below and output the COMPLETE episode again "
                    "(top to bottom, TITLE: first, END. last). PLAIN TEXT "
                    "ONLY: never asterisks, never markdown, never bold "
                    "labels, no parenthetical stage directions. The "
                    "REQUIRED SKELETON, in order: TITLE -> MUSIC (opening) "
                    "-> ANNOUNCER intro -> SCENE 1..N -> ANNOUNCER outro "
                    "-> CODA -> MUSIC (closing) -> END.\n"
                    f"{rendered}"
                )
                break

            scene_counts = _scene_character_word_counts(parsed)
            budget_defects = _script_budget_defects(parsed, envelope)
            if budget_defects:
                traces.append(PassAttemptTrace(
                    attempt=attempt,
                    temperature=float(temp),
                    requested_max_new_tokens=requested_tokens,
                    outcome="budget_rejected",
                    parse_defects=budget_defects,
                    character_words=parsed.character_word_count,
                    scene_character_words=scene_counts,
                ))
                defects_by_attempt.append(list(budget_defects))
                last_defect_text = "; ".join(budget_defects)
                if budget_rerolls >= _MAX_BUDGET_REROLLS:
                    raise Fable2ScriptError(
                        pass_id, last_defect_text, len(traces))
                budget_rerolls += 1
                lo, hi = _word_band(envelope.total_words)
                verb = (
                    "EXPAND the dialogue"
                    if parsed.character_word_count < lo
                    else "TIGHTEN the dialogue"
                )
                structural = ""
                if (parsed.character_word_count > hi
                        and envelope.total_words < 60):
                    cap = _micro_episode_line_cap(envelope.total_words)
                    structural = (
                        f" Cut LINES, not words: at most {cap} character "
                        f"dialogue lines total, each under 10 words.")
                vector = ", ".join(
                    f"scene {index}={target} words "
                    f"({band[0]}-{band[1]})"
                    for index, (target, band) in enumerate(zip(
                        envelope.scene_word_targets,
                        envelope.scene_word_bands), start=1)
                )
                extra_user = (
                    f"Your draft's CHARACTER dialogue totals "
                    f"{parsed.character_word_count} words; the target is "
                    f"{envelope.total_words} (acceptable {int(lo)} to "
                    f"{int(hi)}). {verb} and satisfy the exact scene "
                    f"vector [{vector}]. Output the COMPLETE episode "
                    f"again.{structural}\nDEFECTS: {last_defect_text}"
                )
                continue

            traces.append(PassAttemptTrace(
                attempt=attempt,
                temperature=float(temp),
                requested_max_new_tokens=requested_tokens,
                outcome="accepted",
                selected=True,
                character_words=parsed.character_word_count,
                scene_character_words=scene_counts,
            ))
            trace_tuple = tuple(traces)
            _validate_attempt_sequence(pass_id, trace_tuple)
            return raw, parsed, {
                "defects_by_attempt": defects_by_attempt,
                "rerolls": len(traces) - 1,
                "attempts": len(traces),
                "budget_rerolls": budget_rerolls,
                "truncation_retry": truncation_retry_used,
                "attempt_trace": trace_tuple,
                "actual_max_new_tokens": max(
                    row.requested_max_new_tokens for row in trace_tuple),
            }

    raise Fable2ScriptError(
        pass_id,
        f"markup ladder exhausted; last defects:\n{last_defect_text}",
        len(traces))


def _pass_script(creative_fn, pack, treatment: Treatment, digest: str,
                 envelope: SceneEnvelope, cast_names: "list[str]",
                 ) -> "tuple[str, ParsedScript, dict]":
    """P3 whole-play markup through the shared observed-attempt ladder."""
    system = _seam(pack, "fable2_script_system")
    return _run_markup_ladder(
        creative_fn,
        pass_id="script",
        system=system,
        base_user=_script_user_prompt(
            treatment, digest, envelope, cast_names),
        envelope=envelope,
        cast_names=cast_names,
        initial_temperature=_TEMP["script"],
        format_example=_extract_format_example(system),
    )


def _pass_critic(technical_fn, pack, *, draft1: FinalDraft,
                 treatment: Treatment, selected_pitch: Pitch,
                 envelope: SceneEnvelope,
                 story_rules: StoryRules) -> CriticNotes:
    """P4: critique the exact sealed first draft without rewriting it."""
    _validate_final_draft_integrity("P4 draft1", draft1)
    user_payload = {
        "selected_pitch": selected_pitch.model_dump(mode="json"),
        "treatment": treatment.model_dump(mode="json"),
        "scene_envelope": _envelope_payload(envelope),
        "resolved_story_rules": _story_rules_payload(story_rules),
        "normalized_full_draft": draft1.normalized_source,
        "parsed_full_draft": _parsed_payload(draft1.parsed),
    }
    try:
        # LLM slot: technical -- P4 scoped critic.
        return structured_call(
            prompt=[
                {"role": "system",
                 "content": _seam(pack, "fable2_critic_system")},
                {"role": "user", "content": (
                    "AUTHORITATIVE P4 INPUTS:\n"
                    + json.dumps(
                        user_payload, ensure_ascii=False, indent=2)
                    + "\n\nReturn the CriticNotes object now.")},
            ],
            schema=CriticNotes,
            slot_fn=technical_fn,
            base_temperature=_TEMP["critic"],
            structural_retry_temperature=_TEMP["critic"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_critic_validator(draft1.parsed),
            max_new_tokens=_MAX_NEW_TOKENS["critic"],
            helper_name="fable2_critic",
        )
    except StructuredCallFailedError as exc:
        raise Fable2ScriptError(
            "critic", str(exc.last_error), exc.attempts) from exc


def _revision_protected_payload(parsed: ParsedScript,
                                treatment: Treatment) -> dict:
    """The exact P5 fields Python protects after the model writes."""
    return {
        "title": parsed.title,
        "cast_shapes": [
            shape.model_dump(mode="json") for shape in treatment.cast_shapes
        ],
        "scene_order": [scene.n for scene in parsed.scenes],
        "scene_settings": [scene.setting for scene in parsed.scenes],
        "speaker_topology": [
            [line.speaker for line in scene.lines]
            for scene in parsed.scenes
        ],
        "announcer_skeleton": {
            "intro_lines": len(parsed.announcer_intro),
            "outro_lines": len(parsed.announcer_outro),
        },
        "music_queue": {
            "opening": parsed.music_open,
            "interstitials": [list(row) for row in parsed.music_inter],
            "closing": parsed.music_close,
        },
        "coda_news_boundary": {
            "coda": parsed.coda,
            "news_close_read": treatment.news_close_read,
        },
    }


def _pass_revision(creative_fn, pack, *, draft1: FinalDraft,
                   treatment: Treatment, critic_notes: CriticNotes,
                   selected_pitch: Pitch, envelope: SceneEnvelope,
                   story_rules: StoryRules,
                   cast_names: "list[str]") -> "tuple[str, ParsedScript, dict]":
    """P5: rewrite the complete play, then let Python judge the result."""
    _validate_final_draft_integrity("P5 draft1", draft1)
    payload = {
        "selected_pitch": selected_pitch.model_dump(mode="json"),
        "treatment": treatment.model_dump(mode="json"),
        "scene_envelope": _envelope_payload(envelope),
        "resolved_story_rules": _story_rules_payload(story_rules),
        "critic_notes": critic_notes.model_dump(mode="json"),
        "protected_artifacts": _revision_protected_payload(
            draft1.parsed, treatment),
        "normalized_full_draft": draft1.normalized_source,
        "parsed_full_draft": _parsed_payload(draft1.parsed),
    }
    base_user = (
        "AUTHORITATIVE P5 INPUTS:\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n\nRewrite the COMPLETE episode now. Only critic-authorized "
          "intro/outro or scene/speaker scopes may change. If the verdict "
          "is ship with no notes, reproduce normalized_full_draft "
          "byte-for-byte."
    )
    return _run_markup_ladder(
        creative_fn,
        pass_id="revision",
        system=_seam(pack, "fable2_revision_system"),
        base_user=base_user,
        envelope=envelope,
        cast_names=cast_names,
        initial_temperature=_TEMP["revision"],
    )


def _speakers_in_order(parsed: ParsedScript) -> "list[str]":
    seen: "list[str]" = []
    for scene in parsed.scenes:
        for ln in scene.lines:
            if ln.speaker not in seen:
                seen.append(ln.speaker)
    return seen


def _pass_casting(slot_fn, pack, final_draft: FinalDraft,
                  treatment: Treatment, menu: VoiceMenu, *,
                  envelope: SceneEnvelope, story_rules: StoryRules,
                  mode: Fable2Mode) -> CastingVoices:
    """P6 (registry slot: technical; temp 0.40): voices + portraits ONLY --
    register authority stays with the treatment (r1/A3)."""
    _validate_selected_final_draft(
        final_draft, envelope, story_rules, mode, treatment=treatment)
    parsed = final_draft.parsed
    speakers = _speakers_in_order(parsed)
    script_view = _script_view(parsed, treatment, include_news_read=False)
    shapes = json.dumps(
        [c.model_dump() for c in treatment.cast_shapes],
        ensure_ascii=False, indent=2)
    user = (
        f"THE FINISHED SCRIPT:\n{script_view}\n\n"
        f"TREATMENT CAST SHAPES:\n{shapes}\n\n"
        f"AVAILABLE VOICE STOCK (order by menu id; ids only):\n"
        f"{menu.prompt_lines()}\n\n"
        f"Cast now: exactly one entry per distinct script speaker "
        f"({', '.join(speakers)})."
    )
    try:
        # LLM slot: technical -- P6 casting/voices (registry-legal slot;
        # register authority stays with the treatment).
        return structured_call(
            prompt=[
                {"role": "system",
                 "content": _seam(pack, "fable2_casting_system")},
                {"role": "user", "content": user},
            ],
            schema=CastingVoices,
            slot_fn=slot_fn,
            base_temperature=_TEMP["casting_voices"],
            structural_retry_temperature=_TEMP["casting_voices"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_casting_validator(menu, speakers),
            max_new_tokens=_MAX_NEW_TOKENS["casting_voices"],
            helper_name="fable2_casting_voices",
        )
    except StructuredCallFailedError as exc:
        raise Fable2CastError(
            "casting_voices", str(exc.last_error), exc.attempts) from exc


def _assign_voices(casting: CastingVoices, menu: VoiceMenu,
                   rng: random.Random,
                   speaker_order: "list[str]") -> "list[dict]":
    """r3/M8: returns COMPLETE ledger cast rows (set_cast contract):
    python-prebaked ANNOUNCER c01 (kokoro) + characters c02.. in
    first-appearance order (bark presets from the validated menu picks).
    The LLM invents the person; Python picks the larynx."""
    by_id = menu.by_id()
    by_name = {c.name: c for c in casting.cast}
    announcer = dict(_POOLS.pick_announcer(rng))
    announcer["char_id"] = "c01"
    rows: "list[dict]" = [announcer]
    taken: set[str] = set()
    for i, name in enumerate(speaker_order):
        cv = by_name.get(name)
        if cv is None:
            raise Fable2CastError(
                "casting_voices",
                f"speaker {name!r} missing from validated casting -- "
                f"speaker-set equality gate should have caught this")
        entry = by_id.get(cv.timbre)
        if entry is None or cv.timbre in taken:
            raise Fable2CastError(
                "casting_voices",
                f"{name}: timbre {cv.timbre!r} unknown or already taken "
                f"post-validation -- validator drift")
        taken.add(cv.timbre)
        rows.append({
            "char_id": f"c{i + 2:02d}",
            "name": cv.name,
            "character_description": cv.character_description,
            "gender": cv.gender,
            "tts_model": "bark",
            "voice_preset": entry.preset,
            "voice_params": None,
        })
    return rows


# ---------------------------------------------------------------------------
# P7 -- pure-python assembly (doc s7)
# ---------------------------------------------------------------------------

def _scene_character_word_counts(parsed: ParsedScript) -> "tuple[int, ...]":
    return tuple(
        sum(len(line.text.split()) for line in scene.lines)
        for scene in parsed.scenes
    )


def _parsed_payload(parsed: ParsedScript, *, normalizations: bool = True) -> dict:
    payload = {
        "title": parsed.title,
        "music_open": parsed.music_open,
        "music_inter": [list(row) for row in parsed.music_inter],
        "music_close": parsed.music_close,
        "announcer_intro": list(parsed.announcer_intro),
        "scenes": [
            {
                "n": scene.n,
                "setting": scene.setting,
                "lines": [
                    {"speaker": line.speaker, "text": line.text}
                    for line in scene.lines
                ],
            }
            for scene in parsed.scenes
        ],
        "announcer_outro": list(parsed.announcer_outro),
        "coda": parsed.coda,
        "character_word_count": parsed.character_word_count,
        "announcer_word_count": parsed.announcer_word_count,
    }
    if normalizations:
        payload["normalizations"] = list(parsed.normalizations)
    return payload


def _canonical_sha256(value: Any) -> str:
    return _sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def _story_rules_signature(story_rules: StoryRules) -> tuple:
    """Immutable content fingerprint for an already shape-checked pack."""
    pattern_fields = (
        "cliche", "stage_business", "on_the_nose", "banned_thesis",
        "personal_cost",
    )
    return (
        story_rules.rules_id,
        story_rules.label,
        story_rules.schema_version,
        tuple(
            (name, tuple((rx.pattern, int(rx.flags))
                         for rx in getattr(story_rules, name)))
            for name in pattern_fields
        ),
        tuple(
            (rx.pattern, int(rx.flags), replacement)
            for rx, replacement in story_rules.cliche_replacements
        ),
        tuple(story_rules.banned_phrases),
    )


def _validate_fable2_story_rules(story_rules: Any) -> None:
    """Fail loud if the caller did not resolve the exact Fable2 rules pack."""
    if not isinstance(story_rules, StoryRules):
        raise Fable2ScriptError(
            "revision_contract",
            "story_rules must be the resolved StoryRules artifact, got "
            f"{type(story_rules).__name__}",
        )
    if getattr(story_rules, "rules_id", None) != "scifi_fable2":
        raise Fable2ScriptError(
            "revision_contract",
            "story_rules.rules_id must be 'scifi_fable2', got "
            f"{getattr(story_rules, 'rules_id', None)!r}",
        )
    if getattr(story_rules, "schema_version", None) != "v1":
        raise Fable2ScriptError(
            "revision_contract",
            "story_rules.schema_version must be 'v1', got "
            f"{getattr(story_rules, 'schema_version', None)!r}",
        )
    if not isinstance(story_rules.label, str) or not story_rules.label:
        raise Fable2ScriptError(
            "revision_contract", "story_rules.label must be non-empty")
    for name in (
        "cliche", "stage_business", "on_the_nose", "banned_thesis",
        "personal_cost", "cliche_replacements",
    ):
        value = getattr(story_rules, name, None)
        if not isinstance(value, tuple):
            raise Fable2ScriptError(
                "revision_contract",
                f"story_rules.{name} must be a resolved tuple",
            )
    for name in (
        "cliche", "stage_business", "on_the_nose", "banned_thesis",
        "personal_cost",
    ):
        if any(not isinstance(rx, re.Pattern)
               for rx in getattr(story_rules, name)):
            raise Fable2ScriptError(
                "revision_contract",
                f"story_rules.{name} contains a non-re.Pattern entry",
            )
    replacements = story_rules.cliche_replacements
    if any(
        not isinstance(row, tuple)
        or len(row) != 2
        or not isinstance(row[0], re.Pattern)
        or not isinstance(row[1], str)
        for row in replacements
    ):
        raise Fable2ScriptError(
            "revision_contract",
            "story_rules.cliche_replacements must contain compiled-pattern "
            "and string tuples",
        )
    phrases = getattr(story_rules, "banned_phrases", None)
    if (not isinstance(phrases, tuple)
            or any(not isinstance(p, str) or not p for p in phrases)):
        raise Fable2ScriptError(
            "revision_contract",
            "story_rules.banned_phrases must be a resolved string tuple",
        )
    canonical = resolve_story_rules("scifi_fable2")
    if _story_rules_signature(story_rules) != _story_rules_signature(canonical):
        raise Fable2ScriptError(
            "revision_contract",
            "story_rules content does not match the canonical resolved "
            "scifi_fable2 pack",
        )


def _story_rules_payload(story_rules: StoryRules) -> dict:
    _validate_fable2_story_rules(story_rules)
    pattern_fields = (
        "cliche", "stage_business", "on_the_nose", "banned_thesis",
        "personal_cost",
    )
    return {
        "rules_id": story_rules.rules_id,
        "label": story_rules.label,
        "schema_version": story_rules.schema_version,
        **{
            name: [
                {"pattern": rx.pattern, "flags": int(rx.flags)}
                for rx in getattr(story_rules, name)
            ]
            for name in pattern_fields
        },
        "cliche_replacements": [
            {"pattern": rx.pattern, "flags": int(rx.flags),
             "replacement": replacement}
            for rx, replacement in story_rules.cliche_replacements
        ],
        "banned_phrases": list(story_rules.banned_phrases),
    }


def _envelope_payload(envelope: SceneEnvelope) -> dict:
    _validate_scene_envelope(envelope)
    return {
        "scene_count": envelope.scene_count,
        "scene_word_targets": list(envelope.scene_word_targets),
        "scene_word_bands": [list(band) for band in envelope.scene_word_bands],
        "total_words": envelope.total_words,
        "total_band_frac": envelope.band_frac,
        "scene_band_frac": _SCENE_WORD_BAND,
    }


def _scoring_context_sha256(envelope: SceneEnvelope,
                            story_rules: StoryRules) -> str:
    return _canonical_sha256({
        "envelope": _envelope_payload(envelope),
        "story_rules": _story_rules_payload(story_rules),
    })


def _spoken_script_text(parsed: ParsedScript) -> str:
    parts = list(parsed.announcer_intro)
    parts.extend(
        line.text for scene in parsed.scenes for line in scene.lines)
    parts.extend(parsed.announcer_outro)
    parts.append(parsed.coda)
    return "\n".join(parts)


def _rule_pattern_hits(parsed: ParsedScript, story_rules: Any) -> int:
    _validate_fable2_story_rules(story_rules)
    text = _spoken_script_text(parsed)
    hits = sum(
        len(tuple(rx.finditer(text)))
        for name in ("cliche", "stage_business", "on_the_nose",
                     "banned_thesis")
        for rx in getattr(story_rules, name)
    )
    folded = text.casefold()
    hits += sum(
        folded.count(phrase.casefold())
        for phrase in story_rules.banned_phrases
    )
    return hits


def _draft_score(parsed: ParsedScript, envelope: SceneEnvelope,
                 story_rules: Any) -> DraftScore:
    _validate_scene_envelope(envelope)
    counts = _scene_character_word_counts(parsed)
    bands = envelope.scene_word_bands
    paired = min(len(counts), len(bands))
    violations = abs(len(counts) - len(bands)) + sum(
        not (bands[i][0] <= counts[i] <= bands[i][1])
        for i in range(paired)
    )
    distance = sum(
        abs(counts[i] - envelope.scene_word_targets[i])
        for i in range(paired)
    )
    distance += sum(counts[paired:])
    distance += sum(envelope.scene_word_targets[paired:])
    return (
        _rule_pattern_hits(parsed, story_rules),
        int(violations),
        int(distance),
        abs(parsed.character_word_count - envelope.total_words),
        len(parsed.normalizations),
    )


def _proof_entry_payload(entry: ProofMapEntry) -> dict:
    return {
        "line_id": entry.line_id,
        "constituents": [
            {
                "text": proof.text,
                "artifact": proof.artifact,
                "span": list(proof.span),
            }
            for proof in entry.constituents
        ],
    }

def _prove_constituents(constituents, artifact_name: str, artifact_norm: str,
                        merged_text: str, line_id: str) -> "list[dict]":
    """Gate (a), per CONSTITUENT (r2/M4): every constituent line text
    (whitespace-normalized) must be a substring of its NAMED LLM
    artifact, and the merged row text must equal the space-join of its
    proven constituents."""
    proofs: "list[dict]" = []
    for text in constituents:
        norm = _norm_ws(text)
        idx = artifact_norm.find(norm)
        if idx < 0:
            raise Fable2AssembleError(
                "assemble",
                f"verbatim proof FAILED for line {line_id}: constituent "
                f"{norm[:80]!r} is not a substring of artifact "
                f"{artifact_name!r} -- python never writes dialogue")
        proofs.append({
            "text": norm,
            "artifact": artifact_name,
            "span": [idx, idx + len(norm)],
        })
    joined = " ".join(_norm_ws(t) for t in constituents)
    if _norm_ws(merged_text) != joined:
        raise Fable2AssembleError(
            "assemble",
            f"merged row {line_id} text != space-join of its proven "
            f"constituents")
    return proofs


def _build_proof_map(parsed: ParsedScript, treatment: Treatment,
                     normalized_source: str) -> "tuple[ProofMapEntry, ...]":
    """Build every spoken-row proof without touching a ledger or meta."""
    artifact_norm = _norm_ws(normalized_source)
    if not artifact_norm:
        raise Fable2AssembleError(
            "assemble", "winning draft artifact is empty")
    news_norm = _norm_ws(treatment.news_close_read)
    entries: "list[ProofMapEntry]" = []

    def _append(line_id: str, constituents, merged_text: str,
                artifact_name: str, source_norm: str) -> None:
        rows = _prove_constituents(
            constituents, artifact_name, source_norm, merged_text, line_id)
        entries.append(ProofMapEntry(
            line_id=line_id,
            constituents=tuple(
                ConstituentProof(
                    text=row["text"], artifact=row["artifact"],
                    span=(int(row["span"][0]), int(row["span"][1])),
                )
                for row in rows
            ),
        ))

    for index, text in enumerate(parsed.announcer_intro, start=1):
        _append(
            f"shot_000_b{index}", [text], text,
            "winning_draft", artifact_norm,
        )

    for scene in parsed.scenes:
        runs: "list[tuple[str, list[str]]]" = []
        for line in scene.lines:
            if runs and runs[-1][0] == line.speaker:
                runs[-1][1].append(line.text)
            else:
                runs.append((line.speaker, [line.text]))
        shot_id = f"shot_{scene.n:03d}"
        for index, (_speaker, texts) in enumerate(runs, start=1):
            merged = " ".join(_norm_ws(text) for text in texts)
            _append(
                f"{shot_id}_b{index}", texts, merged,
                "winning_draft", artifact_norm,
            )

    if not parsed.scenes:
        raise Fable2AssembleError("assemble", "script has no scenes")
    post_shot = f"shot_{parsed.scenes[-1].n + 1:03d}"
    post_index = 0
    for text in parsed.announcer_outro:
        post_index += 1
        _append(
            f"{post_shot}_b{post_index}", [text], text,
            "winning_draft", artifact_norm,
        )
    post_index += 1
    _append(
        f"{post_shot}_b{post_index}", [parsed.coda], parsed.coda,
        "winning_draft", artifact_norm,
    )
    post_index += 1
    _append(
        f"{post_shot}_b{post_index}", [treatment.news_close_read],
        treatment.news_close_read, "treatment.news_close_read", news_norm,
    )
    ids = [entry.line_id for entry in entries]
    if len(ids) != len(set(ids)):
        raise Fable2AssembleError(
            "assemble", f"proof map contains duplicate line ids: {ids}")
    return tuple(entries)


def _attempt_payload(trace: PassAttemptTrace) -> dict:
    return {
        "attempt": trace.attempt,
        "temperature": trace.temperature,
        "requested_max_new_tokens": trace.requested_max_new_tokens,
        "outcome": trace.outcome,
        "truncation": trace.truncation,
        "selected": trace.selected,
        "parse_defects": list(trace.parse_defects),
        "character_words": trace.character_words,
        "scene_character_words": list(trace.scene_character_words),
    }


def _final_draft_seal_payload(
    *,
    raw_sha256: str,
    normalized_sha256: str,
    parsed_sha256: str,
    proof_map_sha256: str,
    score: DraftScore,
    scoring_context_sha256: str,
    p3_attempts: "tuple[PassAttemptTrace, ...]",
    p5_attempts: "tuple[PassAttemptTrace, ...]",
) -> dict:
    return {
        "raw_sha256": raw_sha256,
        "normalized_sha256": normalized_sha256,
        "parsed_sha256": parsed_sha256,
        "proof_map_sha256": proof_map_sha256,
        "score": list(score),
        "scoring_context_sha256": scoring_context_sha256,
        "p3_attempts": [_attempt_payload(row) for row in p3_attempts],
        "p5_attempts": [_attempt_payload(row) for row in p5_attempts],
    }


def _validate_attempt_sequence(label: str,
                               rows: "tuple[PassAttemptTrace, ...]") -> None:
    if not rows:
        return
    if any(type(row) is not PassAttemptTrace for row in rows):
        raise Fable2ScriptError(
            "final_draft",
            f"{label} attempts must contain exact PassAttemptTrace artifacts",
        )
    if tuple(row.attempt for row in rows) != tuple(range(1, len(rows) + 1)):
        raise Fable2ScriptError(
            "final_draft", f"{label} attempts must be contiguous from 1")
    selected = [row for row in rows if row.selected]
    if len(selected) != 1 or selected[0] is not rows[-1]:
        raise Fable2ScriptError(
            "final_draft",
            f"{label} must have exactly one selected final attempt",
        )


def build_final_draft(
    raw_source: str,
    parsed: ParsedScript,
    treatment: Treatment,
    envelope: SceneEnvelope,
    story_rules: Any,
    *,
    p3_attempts: "tuple[PassAttemptTrace, ...]" = (),
    p5_attempts: "tuple[PassAttemptTrace, ...]" = (),
) -> FinalDraft:
    """Pure parse/proof/score constructor for one candidate whole play."""
    _validate_scene_envelope(envelope)
    _validate_fable2_story_rules(story_rules)
    p3_attempts = tuple(p3_attempts)
    p5_attempts = tuple(p5_attempts)
    _validate_attempt_sequence("P3", p3_attempts)
    _validate_attempt_sequence("P5", p5_attempts)
    normalized = normalize_fable2_markup_text(raw_source)
    cast_names = [shape.name for shape in treatment.cast_shapes]
    authoritative, defects = parse_fable2_markup(raw_source, cast_names)
    if authoritative is None:
        raise Fable2ParseError(
            "final_draft",
            "raw source does not parse: " + render_defects(defects),
        )
    if _parsed_payload(authoritative) != _parsed_payload(parsed):
        raise Fable2ParseError(
            "final_draft",
            "provided ParsedScript does not match the authoritative raw parse",
        )
    proof_map = _build_proof_map(authoritative, treatment, normalized)
    score = _draft_score(authoritative, envelope, story_rules)
    parsed_payload = _parsed_payload(authoritative)
    proof_payload = [_proof_entry_payload(entry) for entry in proof_map]
    context_hash = _scoring_context_sha256(envelope, story_rules)
    raw_hash = _sha256(raw_source)
    normalized_hash = _sha256(normalized)
    parsed_hash = _canonical_sha256(parsed_payload)
    proof_hash = _canonical_sha256(proof_payload)
    artifact_hash = _canonical_sha256(_final_draft_seal_payload(
        raw_sha256=raw_hash,
        normalized_sha256=normalized_hash,
        parsed_sha256=parsed_hash,
        proof_map_sha256=proof_hash,
        score=score,
        scoring_context_sha256=context_hash,
        p3_attempts=p3_attempts,
        p5_attempts=p5_attempts,
    ))
    return FinalDraft(
        raw_source=raw_source,
        normalized_source=normalized,
        parsed=authoritative,
        proof_map=proof_map,
        p3_attempts=p3_attempts,
        p5_attempts=p5_attempts,
        score=score,
        scoring_context_sha256=context_hash,
        hashes=FinalDraftHashes(
            raw_sha256=raw_hash,
            normalized_sha256=normalized_hash,
            parsed_sha256=parsed_hash,
            proof_map_sha256=proof_hash,
            artifact_sha256=artifact_hash,
        ),
    )


def _assert_source_matches_parsed(
    normalized_source: str,
    parsed: ParsedScript,
    cast_names: "list[str]",
    label: str,
) -> None:
    reparsed, defects = parse_fable2_markup(normalized_source, cast_names)
    if reparsed is None:
        raise Fable2ScriptError(
            "revision_contract",
            f"{label} normalized source does not parse: "
            + render_defects(defects),
        )
    if (_parsed_payload(reparsed, normalizations=False)
            != _parsed_payload(parsed, normalizations=False)):
        raise Fable2ScriptError(
            "revision_contract",
            f"{label} ParsedScript does not match its normalized source",
        )


def _validate_final_draft_integrity(label: str, draft: Any) -> None:
    """Recompute every FinalDraft seal before it can influence selection."""
    if type(draft) is not FinalDraft:
        raise Fable2ScriptError(
            "final_draft", f"{label} must be an exact FinalDraft artifact")
    if type(draft.hashes) is not FinalDraftHashes:
        raise Fable2ScriptError(
            "final_draft", f"{label}.hashes must be FinalDraftHashes")
    if type(draft.parsed) is not ParsedScript:
        raise Fable2ScriptError(
            "final_draft", f"{label}.parsed must be ParsedScript")
    if (type(draft.score) is not tuple or len(draft.score) != 5
            or any(type(value) is not int or value < 0
                   for value in draft.score)):
        raise Fable2ScriptError(
            "final_draft", f"{label}.score must be five non-negative ints")
    if not isinstance(draft.raw_source, str) or not isinstance(
            draft.normalized_source, str):
        raise Fable2ScriptError(
            "final_draft", f"{label} source fields must be strings")
    if not re.fullmatch(r"[0-9a-f]{64}", draft.scoring_context_sha256):
        raise Fable2ScriptError(
            "final_draft", f"{label} scoring context hash is malformed")
    if type(draft.p3_attempts) is not tuple or type(
            draft.p5_attempts) is not tuple:
        raise Fable2ScriptError(
            "final_draft", f"{label} attempt receipts must be tuples")
    _validate_attempt_sequence(f"{label} P3", draft.p3_attempts)
    _validate_attempt_sequence(f"{label} P5", draft.p5_attempts)

    normalized = normalize_fable2_markup_text(draft.raw_source)
    if normalized != draft.normalized_source:
        raise Fable2ScriptError(
            "final_draft", f"{label} normalized source seal is stale")
    authoritative, defects = parse_fable2_markup(
        draft.raw_source, _speakers_in_order(draft.parsed))
    if authoritative is None:
        raise Fable2ScriptError(
            "final_draft",
            f"{label} raw source no longer parses: {render_defects(defects)}",
        )
    if _parsed_payload(authoritative) != _parsed_payload(draft.parsed):
        raise Fable2ScriptError(
            "final_draft", f"{label} parsed artifact seal is stale")

    if type(draft.proof_map) is not tuple:
        raise Fable2ScriptError(
            "final_draft", f"{label}.proof_map must be an immutable tuple")
    line_ids = []
    source_norm = _norm_ws(draft.normalized_source)
    for entry in draft.proof_map:
        if type(entry) is not ProofMapEntry or type(
                entry.constituents) is not tuple:
            raise Fable2ScriptError(
                "final_draft", f"{label} proof map contains a forged row")
        if not isinstance(entry.line_id, str) or not entry.line_id:
            raise Fable2ScriptError(
                "final_draft", f"{label} proof line id is invalid")
        line_ids.append(entry.line_id)
        for proof in entry.constituents:
            if (type(proof) is not ConstituentProof
                    or type(proof.span) is not tuple
                    or len(proof.span) != 2
                    or any(type(value) is not int for value in proof.span)
                    or proof.span[0] < 0
                    or proof.span[1] < proof.span[0]
                    or not isinstance(proof.text, str)
                    or not isinstance(proof.artifact, str)):
                raise Fable2ScriptError(
                    "final_draft", f"{label} contains a forged proof")
            if proof.artifact == "winning_draft" and (
                proof.span[1] > len(source_norm)
                or source_norm[proof.span[0]:proof.span[1]] != proof.text
            ):
                raise Fable2ScriptError(
                    "final_draft", f"{label} proof span no longer resolves")
    if len(line_ids) != len(set(line_ids)):
        raise Fable2ScriptError(
            "final_draft", f"{label} proof line ids are not unique")

    raw_hash = _sha256(draft.raw_source)
    normalized_hash = _sha256(draft.normalized_source)
    parsed_hash = _canonical_sha256(_parsed_payload(draft.parsed))
    proof_hash = _canonical_sha256([
        _proof_entry_payload(entry) for entry in draft.proof_map
    ])
    artifact_hash = _canonical_sha256(_final_draft_seal_payload(
        raw_sha256=raw_hash,
        normalized_sha256=normalized_hash,
        parsed_sha256=parsed_hash,
        proof_map_sha256=proof_hash,
        score=draft.score,
        scoring_context_sha256=draft.scoring_context_sha256,
        p3_attempts=draft.p3_attempts,
        p5_attempts=draft.p5_attempts,
    ))
    expected = FinalDraftHashes(
        raw_sha256=raw_hash,
        normalized_sha256=normalized_hash,
        parsed_sha256=parsed_hash,
        proof_map_sha256=proof_hash,
        artifact_sha256=artifact_hash,
    )
    if draft.hashes != expected:
        raise Fable2ScriptError(
            "final_draft", f"{label} hash seal does not match its content")


def _validate_selected_final_draft(
    draft: FinalDraft,
    envelope: SceneEnvelope,
    story_rules: StoryRules,
    mode: Fable2Mode,
    *,
    treatment: "Treatment | None" = None,
) -> None:
    """Validate the one script authority before every downstream seam."""
    _validate_final_draft_integrity("selected final draft", draft)
    _validate_scene_envelope(envelope)
    _validate_fable2_story_rules(story_rules)
    expected_context = _scoring_context_sha256(envelope, story_rules)
    if draft.scoring_context_sha256 != expected_context:
        raise Fable2ScriptError(
            "final_draft", "selected draft scoring context is stale")
    expected_score = _draft_score(draft.parsed, envelope, story_rules)
    if draft.score != expected_score:
        raise Fable2ScriptError(
            "final_draft", "selected draft score does not match its content")
    if not draft.p3_attempts:
        raise Fable2ScriptError(
            "final_draft", "selected draft has no observed P3 attempts")
    if mode == _FULL_MODE and not draft.p5_attempts:
        raise Fable2ScriptError(
            "final_draft", "full-mode selected draft has no P5 attempts")
    if mode == _COMPACT_MODE and draft.p5_attempts:
        raise Fable2ScriptError(
            "final_draft", "compact selected draft cannot carry P5 attempts")
    if mode not in (_COMPACT_MODE, _FULL_MODE):
        raise Fable2ScriptError(
            "final_draft", f"unknown fable2 mode {mode!r}")
    if treatment is not None:
        expected_proof = _build_proof_map(
            draft.parsed, treatment, draft.normalized_source)
        if draft.proof_map != expected_proof:
            raise Fable2ScriptError(
                "final_draft",
                "selected draft proof map does not match treatment/source",
            )


def _final_draft_ledger_payload(draft: FinalDraft) -> dict:
    """Durable, content-free receipt for the selected sealed artifact."""
    _validate_final_draft_integrity("ledger final draft", draft)
    return {
        "raw_sha256": draft.hashes.raw_sha256,
        "normalized_sha256": draft.hashes.normalized_sha256,
        "parsed_sha256": draft.hashes.parsed_sha256,
        "proof_map_sha256": draft.hashes.proof_map_sha256,
        "artifact_sha256": draft.hashes.artifact_sha256,
        "score": list(draft.score),
        "scoring_context_sha256": draft.scoring_context_sha256,
        "p3_attempts": [
            _attempt_payload(row) for row in draft.p3_attempts
        ],
        "p5_attempts": [
            _attempt_payload(row) for row in draft.p5_attempts
        ],
    }


def _normalized_line_scopes(normalized_source: str):
    """Return exact nonblank source bytes with explicit revision ownership."""
    rows = []
    blank_positions = []
    current_scene = 0
    seen_scene = False
    for source_index, line in enumerate(normalized_source.splitlines()):
        if not line.strip():
            blank_positions.append(source_index)
            continue
        scene_match = re.match(r"^SCENE\s+(\d+):", line)
        if scene_match:
            current_scene = int(scene_match.group(1))
            seen_scene = True
            scope = ("protected", 0, "", "")
        elif line.startswith(("TITLE:", "MUSIC:", "CODA:", "END.")):
            scope = ("protected", 0, "", "")
        elif ":" not in line:
            scope = ("protected", 0, "", "")
        else:
            speaker = line.split(":", 1)[0]
            if speaker == ANNOUNCER_NAME:
                frame = "outro" if seen_scene else "intro"
                scope = ("frame", 0, ANNOUNCER_NAME, frame)
            else:
                scope = ("scene", current_scene, speaker, "")
        rows.append((source_index, line, scope))
    return tuple(rows), tuple(blank_positions)


def _note_scope_error(note: CriticNote, parsed: ParsedScript) -> "str | None":
    scenes = {scene.n: scene for scene in parsed.scenes}
    if note.scene == 0:
        if note.speaker not in ("", ANNOUNCER_NAME):
            return (
                f"scene-zero note speaker must be empty or {ANNOUNCER_NAME}, "
                f"got {note.speaker!r}"
            )
        if note.frame_scope not in ("intro", "outro"):
            return "scene-zero note requires explicit intro/outro frame_scope"
        return None
    scene = scenes.get(note.scene)
    if scene is None:
        return f"critic note names missing scene {note.scene}"
    speakers = {line.speaker for line in scene.lines}
    if note.speaker and note.speaker not in speakers:
        return (
            f"critic note scene {note.scene} names speaker "
            f"{note.speaker!r} outside that scene"
        )
    return None


def _revision_contract_payload(contract: RevisionContractResult) -> dict:
    return {
        "eligible": contract.eligible,
        "reasons": list(contract.reasons),
        "addressed_problem_classes": list(
            contract.addressed_problem_classes),
        "changed_scopes": list(contract.changed_scopes),
        "protected_field_differences": list(
            contract.protected_field_differences),
        "total_word_count": contract.total_word_count,
        "total_word_band": list(contract.total_word_band),
        "scene_word_counts": list(contract.scene_word_counts),
        "scene_word_bands": [list(row) for row in contract.scene_word_bands],
        "draft1_sha256": contract.draft1_sha256,
        "draft2_sha256": contract.draft2_sha256,
        "scoring_context_sha256": contract.scoring_context_sha256,
    }


def validate_revision_contract(
    draft1_normalized: str,
    parsed1: ParsedScript,
    draft2_normalized: str,
    parsed2: ParsedScript,
    critic_notes: CriticNotes,
    envelope: SceneEnvelope,
    resolved_rules: Any,
) -> RevisionContractResult:
    """Pure S2 revision-law validator; it never repairs or rewrites text."""
    _validate_scene_envelope(envelope)
    _validate_fable2_story_rules(resolved_rules)
    if not isinstance(critic_notes, CriticNotes):
        raise Fable2ScriptError(
            "revision_contract", "critic_notes must be a CriticNotes artifact")
    if normalize_fable2_markup_text(draft1_normalized) != draft1_normalized:
        raise Fable2ScriptError(
            "revision_contract", "draft1 source is not normalized markup")
    if normalize_fable2_markup_text(draft2_normalized) != draft2_normalized:
        raise Fable2ScriptError(
            "revision_contract", "draft2 source is not normalized markup")
    cast_names = sorted(set(
        _speakers_in_order(parsed1) + _speakers_in_order(parsed2)
    ))
    _assert_source_matches_parsed(
        draft1_normalized, parsed1, cast_names, "draft1")
    _assert_source_matches_parsed(
        draft2_normalized, parsed2, cast_names, "draft2")

    reasons: "list[str]" = []
    protected: "list[str]" = []

    def _protect(name: str, left: Any, right: Any) -> None:
        if left != right:
            protected.append(name)
            reasons.append(f"protected field changed: {name}")

    _protect("title", parsed1.title, parsed2.title)
    _protect("cast_speaker_set", tuple(_speakers_in_order(parsed1)),
             tuple(_speakers_in_order(parsed2)))
    _protect("scene_count", len(parsed1.scenes), len(parsed2.scenes))
    _protect("scene_order", tuple(s.n for s in parsed1.scenes),
             tuple(s.n for s in parsed2.scenes))
    _protect("scene_settings", tuple(s.setting for s in parsed1.scenes),
             tuple(s.setting for s in parsed2.scenes))
    _protect(
        "speaker_topology",
        tuple(tuple(line.speaker for line in scene.lines)
              for scene in parsed1.scenes),
        tuple(tuple(line.speaker for line in scene.lines)
              for scene in parsed2.scenes),
    )
    _protect(
        "music_queue",
        (parsed1.music_open, parsed1.music_inter, parsed1.music_close),
        (parsed2.music_open, parsed2.music_inter, parsed2.music_close),
    )
    _protect("coda_news_boundary", parsed1.coda, parsed2.coda)
    _protect(
        "announcer_skeleton",
        (len(parsed1.announcer_intro), len(parsed1.announcer_outro)),
        (len(parsed2.announcer_intro), len(parsed2.announcer_outro)),
    )

    scope_errors = []
    for index, note in enumerate(critic_notes.notes):
        error = _note_scope_error(note, parsed1)
        if error:
            scope_errors.append(f"critic note {index + 1}: {error}")
    reasons.extend(scope_errors)

    left_lines, left_blanks = _normalized_line_scopes(draft1_normalized)
    right_lines, right_blanks = _normalized_line_scopes(draft2_normalized)
    if left_blanks != right_blanks:
        reasons.append(
            "normalized-source blank-line layout changed outside authored "
            "revision scopes"
        )
    if len(left_lines) != len(right_lines):
        reasons.append(
            "normalized-source nonblank line count changed outside the "
            "protected skeleton"
        )
    changed: "list[tuple[str, int, str, str]]" = []
    for left, right in zip(left_lines, right_lines):
        left_index, left_bytes, left_scope = left
        right_index, right_bytes, right_scope = right
        if left_bytes == right_bytes:
            continue
        if left_scope != right_scope:
            reasons.append(
                f"normalized-source ownership changed at lines "
                f"{left_index + 1}/{right_index + 1}"
            )
            continue
        kind, scene_no, speaker, frame_scope = left_scope
        if kind == "protected":
            reasons.append(
                f"protected normalized-source bytes changed at line "
                f"{left_index + 1}"
            )
            continue
        label = (
            f"frame:{frame_scope}:line:{left_index}"
            if kind == "frame"
            else f"scene:{scene_no}:speaker:{speaker}:line:{left_index}"
        )
        changed.append((label, scene_no, speaker, frame_scope))

    note_changed: "list[bool]" = [False] * len(critic_notes.notes)
    for scope, scene_no, speaker, frame_scope in changed:
        authorized = False
        for index, note in enumerate(critic_notes.notes):
            applies = (
                (scene_no == 0 and note.scene == 0
                 and note.frame_scope == frame_scope
                 and note.speaker in ("", ANNOUNCER_NAME))
                or
                (scene_no > 0 and note.scene == scene_no
                 and note.speaker in ("", speaker))
            )
            if applies:
                authorized = True
                note_changed[index] = True
        if not authorized:
            reasons.append(f"unnoted change outside critic scope: {scope}")

    addressed: "list[str]" = []
    for index, note in enumerate(critic_notes.notes):
        if note_changed[index]:
            if note.problem not in addressed:
                addressed.append(note.problem)
        else:
            reasons.append(
                f"critic note {index + 1} ({note.problem}) produced no "
                "change in its authorized scope"
            )

    counts = _scene_character_word_counts(parsed2)
    bands = envelope.scene_word_bands
    lo_float, hi_float = _word_band(envelope.total_words)
    total_band = (math.ceil(lo_float), math.floor(hi_float))
    if not (total_band[0] <= parsed2.character_word_count <= total_band[1]):
        reasons.append(
            f"total character words {parsed2.character_word_count} outside "
            f"{total_band[0]}-{total_band[1]}"
        )
    if len(counts) != len(bands):
        reasons.append(
            f"scene count {len(counts)} does not match envelope "
            f"{len(bands)}"
        )
    for index, (count, band) in enumerate(zip(counts, bands), start=1):
        if not (band[0] <= count <= band[1]):
            reasons.append(
                f"scene {index} character words {count} outside "
                f"{band[0]}-{band[1]}"
            )

    # Keep the receipt concise and deterministic when one defect produces
    # more than one comparison symptom.
    reasons = list(dict.fromkeys(reasons))
    result = RevisionContractResult(
        eligible=not reasons,
        reasons=tuple(reasons),
        addressed_problem_classes=tuple(addressed),
        changed_scopes=tuple(
            scope for scope, _scene, _speaker, _frame in changed),
        protected_field_differences=tuple(protected),
        total_word_count=parsed2.character_word_count,
        total_word_band=total_band,
        scene_word_counts=counts,
        scene_word_bands=bands,
        draft1_sha256=_sha256(draft1_normalized),
        draft2_sha256=_sha256(draft2_normalized),
        scoring_context_sha256=_scoring_context_sha256(
            envelope, resolved_rules),
        contract_sha256="",
    )
    return replace(
        result,
        contract_sha256=_canonical_sha256(
            _revision_contract_payload(result)),
    )


def _validate_revision_contract_integrity(contract: Any) -> None:
    if type(contract) is not RevisionContractResult:
        raise Fable2ScriptError(
            "revision_contract",
            "contract must be an exact RevisionContractResult artifact",
        )
    tuple_fields = (
        "reasons", "addressed_problem_classes", "changed_scopes",
        "protected_field_differences", "total_word_band",
        "scene_word_counts", "scene_word_bands",
    )
    if any(type(getattr(contract, name)) is not tuple for name in tuple_fields):
        raise Fable2ScriptError(
            "revision_contract", "contract tuple fields are not immutable")
    if type(contract.eligible) is not bool or contract.eligible != (
            not contract.reasons):
        raise Fable2ScriptError(
            "revision_contract",
            "contract eligibility does not match its reasons",
        )
    if (type(contract.total_word_count) is not int
            or contract.total_word_count < 0
            or len(contract.total_word_band) != 2
            or any(type(value) is not int
                   for value in contract.total_word_band)
            or any(type(value) is not int or value < 0
                   for value in contract.scene_word_counts)
            or any(type(row) is not tuple or len(row) != 2
                   or any(type(value) is not int for value in row)
                   for row in contract.scene_word_bands)):
        raise Fable2ScriptError(
            "revision_contract", "contract word-budget receipt is malformed")
    for name in (
        "draft1_sha256", "draft2_sha256", "scoring_context_sha256",
        "contract_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", getattr(contract, name)):
            raise Fable2ScriptError(
                "revision_contract", f"contract {name} is malformed")
    expected_hash = _canonical_sha256(_revision_contract_payload(contract))
    if contract.contract_sha256 != expected_hash:
        raise Fable2ScriptError(
            "revision_contract", "revision contract hash seal is stale")


def choose_final_draft(draft1: FinalDraft, draft2: FinalDraft,
                       contract: RevisionContractResult) -> DraftSelection:
    """Keep draft 2 only when the revision is eligible and strictly better."""
    _validate_final_draft_integrity("draft1", draft1)
    _validate_final_draft_integrity("draft2", draft2)
    _validate_revision_contract_integrity(contract)
    if contract.draft1_sha256 != draft1.hashes.normalized_sha256:
        raise Fable2ScriptError(
            "revision_contract", "draft1 hash does not match contract")
    if contract.draft2_sha256 != draft2.hashes.normalized_sha256:
        raise Fable2ScriptError(
            "revision_contract", "draft2 hash does not match contract")
    if not (
        contract.scoring_context_sha256
        == draft1.scoring_context_sha256
        == draft2.scoring_context_sha256
    ):
        raise Fable2ScriptError(
            "revision_contract",
            "revision contract and drafts use different envelope/rules "
            "scoring contexts",
        )
    if not contract.eligible:
        detail = "; ".join(contract.reasons[:4]) or "unspecified defect"
        return DraftSelection(
            winner=draft1,
            reason=f"draft1 retained: revision ineligible ({detail})",
            scoring_context_sha256=contract.scoring_context_sha256,
        )
    if draft2.score < draft1.score:
        return DraftSelection(
            winner=draft2,
            reason=(f"draft2 selected: score {draft2.score} strictly lower "
                    f"than {draft1.score}"),
            scoring_context_sha256=contract.scoring_context_sha256,
        )
    if draft2.score == draft1.score:
        return DraftSelection(
            winner=draft1,
            reason=f"draft1 retained: score tie {draft1.score}",
            scoring_context_sha256=contract.scoring_context_sha256,
        )
    return DraftSelection(
        winner=draft1,
        reason=(f"draft1 retained: revision score {draft2.score} is worse "
                f"than {draft1.score}"),
        scoring_context_sha256=contract.scoring_context_sha256,
    )


def _assemble(led: Any, final_draft: FinalDraft, treatment: Treatment,
              cast_rows: "list[dict]", payload: "dict[str, Any]", *,
              envelope: SceneEnvelope, story_rules: StoryRules,
              mode: Fable2Mode) -> None:
    """Emit ALL FIVE ledger hierarchies (r1/S2) with the proof gates.
    Incremental led.save() after the preamble and after each scene.
    Ambiguity = upstream reroll, never silent-fix; timing stays unset
    (SceneSequencer owns it downstream)."""
    _validate_selected_final_draft(
        final_draft, envelope, story_rules, mode, treatment=treatment)
    parsed = final_draft.parsed
    winning_draft_text = final_draft.normalized_source
    draft_norm = _norm_ws(winning_draft_text)
    if not draft_norm:
        raise Fable2AssembleError(
            "assemble", "selected FinalDraft normalized source is empty")
    news_read_norm = _norm_ws(treatment.news_close_read)
    proof_by_id = {
        entry.line_id: entry for entry in final_draft.proof_map
    }
    consumed_proof_ids: "set[str]" = set()

    # Gate (b): parsed speaker set == cast row names (minus ANNOUNCER).
    spoken = set(_speakers_in_order(parsed))
    cast_names = {r["name"] for r in cast_rows} - {ANNOUNCER_NAME}
    if spoken != cast_names:
        raise Fable2AssembleError(
            "assemble",
            f"speaker set {sorted(spoken)} != cast rows {sorted(cast_names)}")
    # Gate (c): skeleton complete (the parser guarantees this on a
    # defect-free parse; re-assert cheaply).
    if not (parsed.music_open and parsed.music_close and parsed.coda
            and parsed.announcer_intro and parsed.announcer_outro
            and parsed.scenes):
        raise Fable2AssembleError("assemble", "skeleton incomplete")
    # Gate (d): CHARACTER words within the band (post-P3-gate re-assert).
    lo, hi = _word_band(envelope.total_words)
    if not (lo <= parsed.character_word_count <= hi):
        raise Fable2AssembleError(
            "assemble",
            f"character_word_count {parsed.character_word_count} outside "
            f"{int(lo)}-{int(hi)}")
    # Gate (e): every cast member speaks (parser CAST_MEMBER_SILENT
    # guarantees; equality in gate (b) re-covers it).

    char_id_by_name = {
        r["name"]: r["char_id"] for r in cast_rows
        if r["name"] != ANNOUNCER_NAME
    }

    led.set_cast(cast_rows)
    meta = _ledger_meta(led)
    meta["cast_status"] = "locked"

    scene_rows: "list[dict]" = []
    shot_rows: "list[dict]" = []
    beat_rows: "list[dict]" = []
    line_rows: "list[dict]" = []
    music_rows: "list[dict]" = []
    proof_map: "list[dict]" = []

    def _music_sentinel(shot_id: str, role: str, seq: int = 0) -> dict:
        # Exact sentinel row shape (r2/M5 + r3/M1): text "", char_id ==
        # speaker_role == the role string, NO line-level cue_id (music[]
        # is the cue authority). `seq` disambiguates MULTIPLE inter cues
        # after one scene (kibitz r2 M4); the first keeps the stable id.
        lid = f"{shot_id}_music" if seq == 0 else f"{shot_id}_music_{seq + 1}"
        return {
            "line_id": lid, "beat_id": lid, "shot_id": shot_id,
            "char_id": role, "speaker_role": role, "boundary": None,
            "text": "",
        }

    def _spoken_row(line_id: str, shot_id: str, char_id: str, role: str,
                    text: str, boundary: "str | None",
                    constituents, artifact_name: str,
                    artifact_norm: str) -> dict:
        del artifact_norm  # pure proof construction already consumed it
        entry = proof_by_id.get(line_id)
        if entry is None or line_id in consumed_proof_ids:
            raise Fable2AssembleError(
                "assemble",
                f"spoken row {line_id} has no unique prebuilt proof entry",
            )
        expected = tuple(_norm_ws(value) for value in constituents)
        actual = tuple(proof.text for proof in entry.constituents)
        if expected != actual:
            raise Fable2AssembleError(
                "assemble",
                f"spoken row {line_id} constituents disagree with its "
                "prebuilt proof entry",
            )
        if any(proof.artifact != artifact_name
               for proof in entry.constituents):
            raise Fable2AssembleError(
                "assemble",
                f"spoken row {line_id} proof artifact drifted from "
                f"{artifact_name!r}",
            )
        if _norm_ws(text) != " ".join(expected):
            raise Fable2AssembleError(
                "assemble",
                f"spoken row {line_id} text differs from proven constituents",
            )
        consumed_proof_ids.add(line_id)
        proof_map.append(_proof_entry_payload(entry))
        return {
            "line_id": line_id, "beat_id": line_id, "shot_id": shot_id,
            "char_id": char_id, "speaker_role": role, "boundary": boundary,
            "text": _norm_ws(text),
        }

    def _beat(line_row: dict, scene_id: "str | None", speaker: str) -> None:
        beat_rows.append({
            "beat_id": line_row["beat_id"], "shot_id": line_row["shot_id"],
            "scene_id": scene_id, "speaker": speaker,
            "char_id": line_row["char_id"],
            "line_ids": [line_row["line_id"]],
        })

    # --- preamble: shot_000 (fixture-truth intro char_id "announcer") ---
    shot_rows.append({
        "shot_id": "shot_000", "scene_id": None, "description": "preamble",
    })
    line_rows.append(_music_sentinel("shot_000", "music_open"))
    music_rows.append({
        "cue_id": "opening",
        "description": parsed.music_open,
        "generation_prompt": parsed.music_open,
    })
    for k, text in enumerate(parsed.announcer_intro):
        row = _spoken_row(
            f"shot_000_b{k + 1}", "shot_000", "announcer", "announcer",
            text, "shot_start" if k == 0 else "beat_start",
            [text], "winning_draft", draft_norm)
        line_rows.append(row)
        _beat(row, None, ANNOUNCER_NAME)
    led.set_lines(line_rows)
    led.save()

    # --- scenes -------------------------------------------------------
    # kibitz r2 M4 (2026-07-10): a dict keyed on the scene silently
    # dropped every inter cue after the first; keep them ALL, in order.
    inter_by_scene: "dict[int, list[str]]" = {}
    for n, cue in parsed.music_inter:
        inter_by_scene.setdefault(n, []).append(cue)
    inter_seq = 0
    for scene in parsed.scenes:
        scene_id = f"s{scene.n:02d}"
        shot_id = f"shot_{scene.n:03d}"
        scene_rows.append({
            "scene_id": scene_id, "description": scene.setting,
        })
        shot_rows.append({
            "shot_id": shot_id, "scene_id": scene_id,
            "description": scene.setting,
        })
        # Merge consecutive same-speaker constituent runs (r2/M4).
        runs: "list[tuple[str, list[str]]]" = []
        for ln in scene.lines:
            if runs and runs[-1][0] == ln.speaker:
                runs[-1][1].append(ln.text)
            else:
                runs.append((ln.speaker, [ln.text]))
        for k, (speaker, texts) in enumerate(runs):
            merged = " ".join(_norm_ws(t) for t in texts)
            row = _spoken_row(
                f"{shot_id}_b{k + 1}", shot_id, char_id_by_name[speaker],
                "character", merged,
                "shot_start" if k == 0 else "beat_start",
                texts, "winning_draft", draft_norm)
            line_rows.append(row)
            _beat(row, scene_id, speaker)
        for k, cue in enumerate(inter_by_scene.get(scene.n, [])):
            inter_seq += 1
            line_rows.append(_music_sentinel(shot_id, "music_inter", k))
            music_rows.append({
                "cue_id": f"inter_{inter_seq:02d}",
                "description": cue,
                "generation_prompt": cue,
            })
        led.set_lines(line_rows)
        led.save()

    # --- postamble: final shot. ALL fable2 announcer rows carry the
    # SENTINEL char_id "announcer" (22nd live smoke 2026-07-10: a
    # cast-keyed downstream mutator in the freeze cascade flipped a c01
    # postamble row to character+skip with no breadcrumb -> Phase 10
    # critical gap. The sentinel id is exempt from every cast-keyed
    # code path by design; the announcer TTS bus keys on speaker_role.
    # The legacy fixture's c01 postamble was a legacy-lane quirk fable2
    # does not copy.) --------------------------------------------------
    post_shot = f"shot_{parsed.scenes[-1].n + 1:03d}"
    shot_rows.append({
        "shot_id": post_shot, "scene_id": None, "description": "postamble",
    })
    k = 0
    for text in parsed.announcer_outro:
        k += 1
        row = _spoken_row(
            f"{post_shot}_b{k}", post_shot, "announcer", "announcer", text,
            "shot_start" if k == 1 else "beat_start",
            [text], "winning_draft", draft_norm)
        line_rows.append(row)
        _beat(row, None, ANNOUNCER_NAME)
    # CODA bridge row (announcer-spoken pivot, from the draft).
    k += 1
    row = _spoken_row(
        f"{post_shot}_b{k}", post_shot, "announcer", "announcer",
        parsed.coda, "beat_start", [parsed.coda], "winning_draft",
        draft_norm)
    line_rows.append(row)
    _beat(row, None, ANNOUNCER_NAME)
    # News-read row: LLM-authored (treatment.news_close_read),
    # python-APPENDED (r1/C1). The legacy coda append lives in legacy
    # composition (writer I.5), which this lane never runs -- no double
    # append by construction (doc s14 item 7).
    k += 1
    row = _spoken_row(
        f"{post_shot}_b{k}", post_shot, "announcer", "announcer",
        treatment.news_close_read, "beat_start",
        [treatment.news_close_read], "treatment.news_close_read",
        news_read_norm)
    line_rows.append(row)
    _beat(row, None, ANNOUNCER_NAME)
    line_rows.append(_music_sentinel(post_shot, "music_close"))
    music_rows.append({
        "cue_id": "closing",
        "description": parsed.music_close,
        "generation_prompt": parsed.music_close,
    })

    led.set_scenes(scene_rows)
    led.set_shots(shot_rows)
    led.set_beats(beat_rows)
    led.set_lines(line_rows)
    led.set_music(music_rows)

    missing_proofs = sorted(set(proof_by_id) - consumed_proof_ids)
    if missing_proofs:
        raise Fable2AssembleError(
            "assemble",
            f"prebuilt proof entries were not consumed: {missing_proofs}",
        )

    _fable2_meta(led)["proof_map"] = proof_map

    from ._otr_content_authorship import stamp_receipt
    meta = _ledger_meta(led)
    meta.setdefault("source_bank", "scifi_fable2")
    stamp_receipt(
        led.data, owner_bank="scifi_fable2",
        accepted_artifacts={
            "winning_script": winning_draft_text,
            "final_treatment": (
                treatment.model_dump(mode="json")
                if hasattr(treatment, "model_dump") else treatment.__dict__
            ),
        },
    )

    # Content-owned Phase 7 (720-bakeoff C2 / S2 P1.3): stamp the TTS
    # DELIVERY text (Dr. -> Doctor, digits -> words) onto a separate
    # text_for_tts field WITHOUT touching the sealed canonical text /
    # counts / proof map. This restores the pronunciation the P0 fold
    # switched off (the freeze cascade skips legacy Phase 7 under
    # content_owned_readonly) and is what the voice node speaks via the
    # delivery resolver. Stamped BEFORE save so the C1 durable-identity
    # merge carries it forward across the downstream cast-lock re-saves
    # (canonical text is asserted unchanged for this lane, so the stamp
    # is retained; a changed line would invalidate it -> resolver fails
    # loud rather than speaking stale delivery text).
    from ._otr_readiness import stamp_text_for_tts_delivery
    stamp_text_for_tts_delivery(led)

    led.save()


# ---------------------------------------------------------------------------
# P8 -- ledger audit (audit-only through S3; r2/M6+CUT1)
# ---------------------------------------------------------------------------

def _script_view(parsed: ParsedScript, treatment: Treatment,
                 *, include_news_read: bool = True) -> str:
    """Human-shaped assembled view for P6/P8 prompts (python-rendered
    from the parsed artifacts; no ledger dependency -- pure module)."""
    lines = [
        f"TITLE: {parsed.title}",
        f"MUSIC: {parsed.music_open}",
    ]
    for t in parsed.announcer_intro:
        lines.append(f"{ANNOUNCER_NAME}: {t}")
    inter_by_scene: "dict[int, list[str]]" = {}
    for scene_after, cue in parsed.music_inter:
        inter_by_scene.setdefault(scene_after, []).append(cue)
    for scene in parsed.scenes:
        lines.append(f"SCENE {scene.n}: {scene.setting}")
        for ln in scene.lines:
            lines.append(f"{ln.speaker}: {ln.text}")
        for cue in inter_by_scene.get(scene.n, []):
            lines.append(f"MUSIC: {cue}")
    for t in parsed.announcer_outro:
        lines.append(f"{ANNOUNCER_NAME}: {t}")
    lines.append(f"CODA: {parsed.coda}")
    if include_news_read:
        lines.append(f"{ANNOUNCER_NAME} (closing news read): "
                     f"{treatment.news_close_read}")
    lines.extend([f"MUSIC: {parsed.music_close}", "END."])
    return "\n".join(lines)


def _assert_no_weapons_or_smoking(parsed: ParsedScript) -> None:
    """The one content rule that must never reach the air, proven not judged.

    Word-boundary scan of the shared weapons/tobacco lexicon over the SPOKEN
    text. No model opinion, no self-corroborating "evidence", no episode thrown
    away because an auditor felt uneasy about a clean line.
    """
    spoken = "\n".join(
        ln.text for scene in parsed.scenes for ln in scene.lines
    )
    hits = lexicon_hits(spoken, WEAPON_SMOKING_EVIDENCE)
    if hits:
        raise Fable2AuditError(
            "ledger_audit",
            "spoken text names weapons or smoking: " + ", ".join(hits),
            1,
        )


# ---------------------------------------------------------------------------
# Tail handoff (r4/M3: plain, runner-local; the WRITER builds
# WriterTailContext from these parts -- acyclic import graph)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Fable2OutlineView:
    """The tail's outline duck-type: .premise + .title (+ .setting for
    forensics). fable2: premise = the treatment's dramatic-question line,
    title = the treatment title."""

    premise: str
    title: str
    setting: str = ""


@dataclass
class Fable2TailParts:
    outline_view: Fable2OutlineView
    canon: Any
    final_title_override: str
    run_story_spine: bool = False   # the P4/P5/P8 loop is this lane's spine
    refine_active: bool = False     # refine loop unsupported S1-S3
    fable2_meta: "dict | None" = None


_TIME_OF_DAY_WORDS = (
    ("midnight", "night"), ("night", "night"), ("dawn", "dawn"),
    ("sunrise", "dawn"), ("morning", "morning"), ("noon", "midday"),
    ("midday", "midday"), ("afternoon", "afternoon"), ("dusk", "dusk"),
    ("sunset", "dusk"), ("evening", "evening"),
)


def _derive_time_of_day(treatment: Treatment, parsed: ParsedScript) -> str:
    """Deterministic time-of-day derivation (doc s14 item 5): first match
    in setting text, then scene settings; period-radio default 'night'."""
    hay = " ".join(
        [treatment.setting] + [s.setting for s in parsed.scenes]).casefold()
    for word, value in _TIME_OF_DAY_WORDS:
        if word in hay:
            return value
    return "night"


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------

def run_scifi_fable2_episode(
    *,
    payload: "dict[str, Any]",
    pack: Any,
    resolved: dict,
    led: Any,
    meta: dict,
    creative_fn: Callable[..., str],
    technical_fn: Callable[..., str],
    slot_scheduler: Any,
    source_bank_row: Any,
    story_rules: Any,
    episode_root: Any,
    episode_id: str,
) -> Fable2TailParts:
    """Fill led + meta to the legacy writer's endpoint; return
    Fable2TailParts. `resolved` is consumed AS-IS (never rebuilt). The
    runner builds the episode-canon OBJECT only -- the tail remains the
    ONLY canon WRITER (single-writer rule; the tail's J.5 block re-titles
    and writes it). Raises Fable2Error subclasses; NO fallback."""
    target = int(resolved["target_words"])
    assert_supported_target_words(target)  # defensive re-assert (r3/M4)
    mode = _resolve_mode(target)
    _validate_fable2_story_rules(story_rules)
    n_max = max(1, int(resolved["num_characters"]))
    creative_model = str(resolved["creative_writing_model"])
    technical_model = str(resolved["technical_model"])

    seed = _resolve_seed()
    rng = random.Random(seed)
    # Credits receipt (25th live smoke 2026-07-10: OTR_CreditsRoll
    # requires meta.cast_contract.cast_seed OR meta.episode_seed; the
    # legacy cast-lock stamps the former, fable2 has no cast lock). The
    # fable2 seed GOVERNS the deal + the voice draw -- it IS this
    # episode's seed receipt.
    meta["episode_seed"] = seed

    receipts: "list[dict]" = []

    def _receipt(pass_id: str, model_id: str, attempts: int, temp: float,
                 max_new_tokens: int, *, status: str = "completed",
                 reason: str = "",
                 attempt_trace: "tuple[PassAttemptTrace, ...]" = ()) -> None:
        row = {
            "pass_id": pass_id, "model_id": model_id, "attempts": attempts,
            "temp": temp, "max_new_tokens": max_new_tokens, "mode": mode,
            "status": status,
        }
        if reason:
            row["reason"] = reason
        if attempt_trace:
            _validate_attempt_sequence(pass_id, attempt_trace)
            row["attempt_trace"] = [
                _attempt_payload(trace) for trace in attempt_trace
            ]
        receipts.append(row)

    f2: dict = {
        "schema_version": _SCHEMA_VERSION,
        "mode": mode,
        "seed": seed,
        "notes": [
            "news_briefs_required is a documented NO-OP for this lane "
            "(interpreter empty by design; the treatment IS the "
            "interpretation).",
            "S2 full mode executes P2a/P4/P5 at 120-900 words; compact "
            "mode stamps those passes skipped below 120; P8 is audit-only.",
        ],
    }
    meta["fable2"] = f2
    # The treatment IS the interpretation: no interpreter briefs exist for
    # this lane. None is the production-proven degrade-lane value every
    # downstream meta['news'] reader already tolerates.
    meta["news"] = None

    # --- P0: dossier ----------------------------------------------------
    digest = _build_digest(payload)
    counting_tech, tech_box = _counting(technical_fn)
    with _helper_ctx(slot_scheduler, "fable2_dossier"):
        dossier = _pass_dossier(counting_tech, pack, digest)
    _receipt("dossier", technical_model, tech_box["calls"],
             _TEMP["dossier"], _MAX_NEW_TOKENS["dossier"])
    dossier, dropped_entities = _filter_dossier_entities(dossier, digest)
    if dropped_entities:
        log.warning(
            "[scifi_fable2] dossier: DROPPED %d entity(ies) the source "
            "text cannot corroborate (delete-only; world-knowledge "
            "expansions never widen the read corpus): %s",
            len(dropped_entities), "; ".join(dropped_entities))
    provenance = {
        "headline": str(payload.get("headline") or ""),
        "source": str(payload.get("source") or ""),
        "date": str(payload.get("date") or ""),
        "link": str(payload.get("link") or ""),
    }
    f2["dossier"] = {**dossier.model_dump(), "provenance": provenance,
                     "dropped_entities": dropped_entities}

    # --- deal + P1: pitch room --------------------------------------------
    deck = _load_frame_deck()
    cards, stance = _deal(rng, deck, mode=mode)
    f2["cards_dealt"] = [dict(c) for c in cards]
    f2["stance"] = dict(stance)
    counting_cre, cre_box = _counting(creative_fn)
    with _helper_ctx(slot_scheduler, "fable2_pitch_room"):
        slate = _pass_pitch(
            counting_cre, pack, dossier, cards, stance,
            n_max=n_max, mode=mode)
    _receipt("pitch_room", creative_model, cre_box["calls"],
             _TEMP["pitch_room"], _MAX_NEW_TOKENS["pitch_room"])
    f2["pitches"] = [p.model_dump() for p in slate.pitches]
    if mode == _FULL_MODE:
        counting_select, select_box = _counting(technical_fn)
        with _helper_ctx(slot_scheduler, "fable2_pitch_select"):
            pitch_selection = _pass_select(
                counting_select, pack, dossier, slate, stance)
        _receipt(
            "pitch_select", technical_model, select_box["calls"],
            _TEMP["pitch_select"], _MAX_NEW_TOKENS["pitch_select"])
        pitch = next(
            candidate for candidate in slate.pitches
            if candidate.pitch_id == pitch_selection.chosen_pitch_id)
        f2["selection"] = {
            **pitch_selection.model_dump(),
            "rationale": pitch_selection.selection_rationale,
            "selected_pitch_sha256": _canonical_sha256(
                pitch.model_dump(mode="json")),
        }
    else:
        pitch = slate.pitches[0]
        rationale = (
            "one-pitch mode: the sole dealt pitch wins "
            "(P2a skipped by design)"
        )
        f2["selection"] = {
            "chosen_pitch_id": pitch.pitch_id,
            "selection_rationale": rationale,
            "rationale": rationale,
            "selected_pitch_sha256": _canonical_sha256(
                pitch.model_dump(mode="json")),
        }
        _receipt(
            "pitch_select", "skipped", 0, _TEMP["pitch_select"],
            _MAX_NEW_TOKENS["pitch_select"], status="skipped",
            reason="compact mode: the sole dealt pitch wins",
        )

    # --- P2b: treatment (cast names are BORN here, r2/M1) ----------------
    counting_cre2, cre2_box = _counting(creative_fn)
    with _helper_ctx(slot_scheduler, "fable2_treatment"):
        treatment = _pass_treatment(
            counting_cre2, pack, dossier, pitch, stance,
            n_max=n_max, provenance=provenance, digest=digest)
    _receipt("treatment", creative_model, cre2_box["calls"],
             _TEMP["treatment"], _MAX_NEW_TOKENS["treatment"])
    cast_names = [c.name for c in treatment.cast_shapes]
    meta["num_characters_locked"] = len(cast_names)

    # --- P2c: factual close (read-split; S1b deviation, kibitz r2 Q1) ----
    counting_tech_read, tech_read_box = _counting(technical_fn)
    with _helper_ctx(slot_scheduler, "fable2_news_read"):
        read = _pass_news_read(
            counting_tech_read, pack, dossier, provenance, digest,
            cast_names)
    _receipt("news_read", technical_model, tech_read_box["calls"],
             0.20, 300)
    treatment = treatment.model_copy(
        update={"news_close_read": read.news_close_read})
    f2["treatment"] = treatment.model_dump()

    # --- P3: script (markup ladder) ---------------------------------------
    envelope = _build_envelope(target)
    counting_cre3, cre3_box = _counting(creative_fn)
    with _helper_ctx(slot_scheduler, "fable2_script"):
        draft1_text, parsed1, p3_meta = _pass_script(
            counting_cre3, pack, treatment, digest, envelope, cast_names)
    p3_attempts = tuple(p3_meta["attempt_trace"])
    if cre3_box["calls"] != len(p3_attempts):
        raise Fable2ScriptError(
            "script", "P3 observed attempt trace/call count drift")
    _receipt(
        "script", creative_model, len(p3_attempts), _TEMP["script"],
        p3_meta["actual_max_new_tokens"], attempt_trace=p3_attempts)
    draft1_pre = build_final_draft(
        draft1_text, parsed1, treatment, envelope, story_rules,
        p3_attempts=p3_attempts)

    if mode == _FULL_MODE:
        # --- P4: critic ----------------------------------------------------
        counting_critic, critic_box = _counting(technical_fn)
        with _helper_ctx(slot_scheduler, "fable2_critic"):
            critic_notes = _pass_critic(
                counting_critic, pack, draft1=draft1_pre,
                treatment=treatment, selected_pitch=pitch,
                envelope=envelope, story_rules=story_rules)
        _receipt(
            "critic", technical_model, critic_box["calls"],
            _TEMP["critic"], _MAX_NEW_TOKENS["critic"])
        f2["critic"] = critic_notes.model_dump(mode="json")

        # --- P5: revision --------------------------------------------------
        counting_revision, revision_box = _counting(creative_fn)
        with _helper_ctx(slot_scheduler, "fable2_revision"):
            draft2_text, parsed2, p5_meta = _pass_revision(
                counting_revision, pack, draft1=draft1_pre,
                treatment=treatment, critic_notes=critic_notes,
                selected_pitch=pitch, envelope=envelope,
                story_rules=story_rules, cast_names=cast_names)
        p5_attempts = tuple(p5_meta["attempt_trace"])
        if revision_box["calls"] != len(p5_attempts):
            raise Fable2ScriptError(
                "revision", "P5 observed attempt trace/call count drift")
        _receipt(
            "revision", creative_model, len(p5_attempts),
            _TEMP["revision"], p5_meta["actual_max_new_tokens"],
            attempt_trace=p5_attempts)

        revision_contract = validate_revision_contract(
            draft1_pre.normalized_source, parsed1,
            normalize_fable2_markup_text(draft2_text), parsed2,
            critic_notes, envelope, story_rules)
        # Both candidates carry the whole observed P3/P5 history. A tied,
        # worse, or ineligible revision that retains draft1 therefore never
        # drops the real P5 attempt record from the selected artifact.
        draft1 = build_final_draft(
            draft1_text, parsed1, treatment, envelope, story_rules,
            p3_attempts=p3_attempts, p5_attempts=p5_attempts)
        draft2 = build_final_draft(
            draft2_text, parsed2, treatment, envelope, story_rules,
            p3_attempts=p3_attempts, p5_attempts=p5_attempts)
        draft_selection = choose_final_draft(
            draft1, draft2, revision_contract)
        selected_draft_name = (
            "draft2" if draft_selection.winner is draft2 else "draft1")
        revision_receipt = _revision_contract_payload(revision_contract)
        revision_receipt["contract_sha256"] = (
            revision_contract.contract_sha256)
        f2["revision_contract"] = revision_receipt
        f2["draft2_sha256"] = draft2.hashes.raw_sha256
        f2["parse_p5"] = {
            "defects_by_attempt": p5_meta["defects_by_attempt"],
            "rerolls": p5_meta["rerolls"],
            "normalizations": list(parsed2.normalizations),
            "attempt_trace": [
                _attempt_payload(row) for row in p5_attempts
            ],
        }
    else:
        draft1 = draft1_pre
        draft_selection = DraftSelection(
            winner=draft1,
            reason="draft1 selected: compact one-draft mode",
            scoring_context_sha256=draft1.scoring_context_sha256,
        )
        selected_draft_name = "draft1"
        f2["critic"] = None
        f2["revision_contract"] = None
        _receipt(
            "critic", "skipped", 0, _TEMP["critic"],
            _MAX_NEW_TOKENS["critic"], status="skipped",
            reason="compact mode uses one accepted draft",
        )
        _receipt(
            "revision", "skipped", 0, _TEMP["revision"],
            _script_token_budget(target), status="skipped",
            reason="compact mode uses one accepted draft",
        )

    final_draft = draft_selection.winner
    _validate_selected_final_draft(
        final_draft, envelope, story_rules, mode, treatment=treatment)
    f2["draft1_sha256"] = draft1.hashes.raw_sha256
    f2["final_sha256"] = final_draft.hashes.raw_sha256
    f2["selected_draft"] = selected_draft_name
    f2["better_draft_choice"] = draft_selection.reason
    f2["final_draft"] = _final_draft_ledger_payload(final_draft)
    f2["parse"] = {
        "defects_by_attempt": p3_meta["defects_by_attempt"],
        "rerolls": p3_meta["rerolls"],
        "normalizations": list(final_draft.parsed.normalizations),
        "attempt_trace": [
            _attempt_payload(row) for row in p3_attempts
        ],
    }
    if final_draft.parsed.normalizations:
        log.warning(
            "[scifi_fable2] parser stripped %d decoration(s) from the "
            "winning draft (delete-only; stamped at "
            "meta.fable2.parse.normalizations for the operator eyeball): %s",
            len(final_draft.parsed.normalizations),
            "; ".join(final_draft.parsed.normalizations[:12]),
        )

    # --- P6: casting/voices ------------------------------------------------
    menu = _deal_voice_menu(len(cast_names))
    f2["casting_stock_dealt"] = [
        {"menu_id": e.menu_id, "gender": e.gender,
         "description": e.description}
        for e in menu.entries
    ]
    counting_tech2, tech2_box = _counting(technical_fn)
    with _helper_ctx(slot_scheduler, "fable2_casting_voices"):
        casting = _pass_casting(
            counting_tech2, pack, final_draft, treatment, menu,
            envelope=envelope, story_rules=story_rules, mode=mode)
    _receipt("casting_voices", technical_model, tech2_box["calls"],
             _TEMP["casting_voices"], _MAX_NEW_TOKENS["casting_voices"])
    speaker_order = _speakers_in_order(final_draft.parsed)
    cast_rows = _assign_voices(casting, menu, rng, speaker_order)
    f2["casting"] = [c.model_dump() for c in casting.cast]

    # --- P7: assembly (pure python; proof gates; incremental saves) --------
    _assemble(
        led, final_draft, treatment, cast_rows, payload,
        envelope=envelope, story_rules=story_rules, mode=mode)
    _receipt("assemble", "python", 0, 0.0, 0)
    # Ledger.save() replaces led.data with its merged payload. Reacquire
    # both mappings before every post-assembly mutation; caller aliases are
    # intentionally not trusted across the incremental save boundary.
    meta = _ledger_meta(led)
    f2 = _fable2_meta(led)

    # --- P8: deterministic safety scan on the spoken drama ------------------
    # The LLM ledger audit is gone. It could only ever kill on a lexicon hit, so
    # the lexicon was always the real authority -- and asking a model first meant
    # a complete, twice-persisted ledger was thrown away on its opinion. Worse,
    # its corroboration lexicons were the PERIOD lane's: ordinary sci-fi words
    # ("machine", "computer", "report") "proved" a hallucinated flag.
    #
    # What survives is the one thing that must never reach the air, checked
    # directly on the spoken text with word boundaries.
    _assert_no_weapons_or_smoking(final_draft.parsed)
    f2["pass_receipts"] = receipts
    led.save()
    meta = _ledger_meta(led)
    f2 = _fable2_meta(led)

    # --- tail parts ---------------------------------------------------------
    canon = _OTRC.episode_canon_from_outline_dict({
        "title": treatment.title,
        "premise": treatment.dramatic_question,
        "setting": treatment.setting,
        "time_of_day": _derive_time_of_day(treatment, final_draft.parsed),
        "sound_palette": [],  # no style contract on this lane
    })
    outline_view = Fable2OutlineView(
        premise=treatment.dramatic_question,
        title=treatment.title,
        setting=treatment.setting,
    )
    log.info(
        "[scifi_fable2] spine complete: mode=%s seed=%d cast=%d scenes=%d "
        "character_words=%d receipts=%d",
        mode, seed, len(cast_names), len(final_draft.parsed.scenes),
        final_draft.parsed.character_word_count, len(receipts),
    )
    return Fable2TailParts(
        outline_view=outline_view,
        canon=canon,
        final_title_override=final_draft.parsed.title,
        run_story_spine=False,
        refine_active=False,
        fable2_meta=f2,
    )
