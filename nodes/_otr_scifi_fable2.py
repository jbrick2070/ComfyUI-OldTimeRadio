"""scifi_fable2 runner -- S1b spine (LLM-first multipass sci-fi lane).

Architecture doc: docs/2026-07-10-scifi-fable2-architecture.md (sections
3/5/7/8/9/11/13). Operator law: **Python judges; the LLM writes.** Every
spoken ledger row is traceable to a named LLM artifact (the per-constituent
proof gate); this module never writes, trims, or repairs a spoken word.

S1b scope (doc s13): P0 dossier -> deal -> P1 pitch (ONE-PITCH mode) ->
P2b treatment -> P3 script (markup ladder + budget gate + truncation
retry) -> P6 casting/voices -> P7 pure-python assembly (proof gates,
incremental saves) -> P8 ledger audit (audit-only, fail loud). The full
loop (P1 three-pitch, P2a select, P4 critic, P5 revision) lands at S2;
until then any run that would need it fails LOUD at the writer's entry
gate (`assert_supported_target_words`) -- never a silent degrade.

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
import os
import random
import re
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

try:
    from ._otr_structured_call import (
        StructuredCallFailedError,
        structured_call,
    )
    from ._otr_repair_prompts import make_dispatching_repair_factory
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
        MACHINE_ATTRIBUTION_EVIDENCE,
        NEWS_FRAMING_EVIDENCE,
        WEAPON_SMOKING_EVIDENCE,
    )
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_structured_call import (  # type: ignore
        StructuredCallFailedError,
        structured_call,
    )
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore
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
        MACHINE_ATTRIBUTION_EVIDENCE,
        NEWS_FRAMING_EVIDENCE,
        WEAPON_SMOKING_EVIDENCE,
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
    "casting_voices": 0.40, "ledger_audit": 0.20,
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

# Fixed passes: plain ints (r2 anchor: no lambdas in a dict). P3/P5 use
# _script_token_budget instead.
_MAX_NEW_TOKENS = {
    "dossier": 700, "pitch_room": 900, "pitch_select": 300,
    "treatment": 1100, "critic": 800,
    # 18th live smoke 2026-07-10: 1000 truncated a verbose 2-cast JSON
    # mid-object -> the extractor salvaged an inner entry -> schema fail.
    "casting_voices": 1400,
    "ledger_audit": 700,
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

    Two gates, both LOUD:
    * > _SUPPORTED_WORD_CEILING: act-chunked long-episode mode is
      deferred post-S3 -- never silently degrade a long request.
    * >= _ONE_DRAFT_THRESHOLD: S1b ships the ONE-PITCH / ONE-DRAFT spine
      only; the full loop (three-pitch room, select, critic, revision)
      lands at S2. Running a full-mode request through the one-draft
      spine would be a silent quality degrade -- refuse instead.
    """
    tw = int(target_words)
    if tw > _SUPPORTED_WORD_CEILING:
        raise Fable2ScriptError(
            "script",
            f"target_words {tw} above supported ceiling "
            f"{_SUPPORTED_WORD_CEILING} (act-chunked long-episode mode is "
            f"deferred post-S3; lower target_words)",
        )
    if tw >= _ONE_DRAFT_THRESHOLD:
        raise Fable2ScriptError(
            "script",
            f"target_words {tw} needs the FULL fable2 loop (three-pitch "
            f"room + select + critic + revision), which lands at S2; S1b "
            f"supports the low-budget one-draft spine only "
            f"(target_words < {_ONE_DRAFT_THRESHOLD})",
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
    pitch_id: int = Field(ge=1, le=3)
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
    chosen_pitch_id: int = Field(ge=1, le=3)
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
    scene: int = Field(ge=0)
    speaker: str
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


class CriticNotes(BaseModel):
    verdict: Literal["ship", "revise"]
    notes: list[CriticNote] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def _revise_needs_notes(self) -> "CriticNotes":
        if self.verdict == "revise" and not self.notes:
            raise ValueError('"revise" requires >= 1 note')
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


_AUDIT_CLASSES = (
    "register_bleed", "on_the_nose", "stakes_sag", "ending_unearned",
    "continuity_break", "cast_unused", "announcer_contract",
    "word_budget", "subtext_flat",
    "speaker_not_in_cast", "verbatim_break", "skeleton_break",
    "news_source_framing", "machine_attribution", "weapons_smoking",
)


class AuditFinding(BaseModel):
    finding_class: Literal[
        "register_bleed", "on_the_nose", "stakes_sag", "ending_unearned",
        "continuity_break", "cast_unused", "announcer_contract",
        "word_budget", "subtext_flat",
        "speaker_not_in_cast", "verbatim_break", "skeleton_break",
        "news_source_framing", "machine_attribution", "weapons_smoking"]
    # P8 is a REPORTING pass: a finding without a speaker/scene is still
    # triage-able (speaker "" is the documented scene-level form; 21st
    # live smoke 2026-07-10: a schema abort over a missing speaker field
    # killed an otherwise-complete episode).
    scene: int = Field(default=0, ge=0)
    speaker: str = ""
    detail: str = Field(min_length=10, max_length=200)


class AuditFindings(BaseModel):
    findings: list[AuditFinding] = Field(default_factory=list, max_length=12)


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


def _deal(rng: random.Random, deck: dict, *, mode: str):
    """Deal (frame_cards, stance). one_pitch mode deals 1 card; full mode
    deals 3 distinct cards (pitch i uses card i)."""
    count = 1 if mode == "one_pitch" else 3
    cards = tuple(rng.sample(list(deck["cards"]), count))
    stance = rng.choice(list(deck["stances"]))
    return cards, stance


# ---------------------------------------------------------------------------
# Scene envelope (python computes BEFORE any LLM call; doc s3/s13)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneEnvelope:
    scene_count: int
    per_scene_words: int
    total_words: int
    band_frac: float = _TOTAL_WORD_BAND


def _build_envelope(target_words: int) -> SceneEnvelope:
    """scenes = clamp(round(target_words/110), 1, 8) (30w -> 1, 350w -> 3;
    tuned against the S1b/S2 smokes)."""
    tw = int(target_words)
    scenes = max(1, min(8, round(tw / 110)))
    return SceneEnvelope(
        scene_count=scenes,
        per_scene_words=max(1, round(tw / scenes)),
        total_words=tw,
    )


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


def _make_pitch_validator(cards, mode: str, n_max: int):
    """Dealt-card + divergence + SFW gates. `cards` is the dealt tuple;
    the count expectation is a parameter (r2/M3)."""
    expected = 1 if mode == "one_pitch" else 3
    card_names = [c["name"] for c in cards]

    def _check(m: PitchSlate) -> "str | None":
        if len(m.pitches) != expected:
            return f"need exactly {expected} pitch(es), got {len(m.pitches)}"
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
                *, n_max: int, mode: str) -> PitchSlate:
    count = 1 if mode == "one_pitch" else 3
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
    return (
        f"TREATMENT:\n"
        f"{json.dumps(treatment.model_dump(), ensure_ascii=False, indent=2)}"
        f"\n\nCAST (the ONLY legal speakers besides ANNOUNCER; copy names "
        f"EXACTLY):\n{cast_block}\n\n"
        f"SOURCE DIGEST (fiction fuel, never quoted as news):\n{digest}\n\n"
        f"SCENE ENVELOPE: exactly {envelope.scene_count} scene(s); about "
        f"{envelope.per_scene_words} character-dialogue words per scene "
        f"(plus or minus 30%); total CHARACTER dialogue target "
        f"{envelope.total_words} words (announcer lines are metered "
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


def _pass_script(creative_fn, pack, treatment: Treatment, digest: str,
                 envelope: SceneEnvelope, cast_names: "list[str]",
                 ) -> "tuple[str, ParsedScript, dict]":
    """P3: whole-play markup with the markup ladder (defect-quoting
    reroll at falling temperature), the truncation retry (+25% tokens,
    ONCE, on MISSING_END), and the budget gate reroll (max 2, numeric
    hint). Python never repairs a word -- defective drafts reroll.

    Few-shot anchor (live-smoke hardening 2026-07-10): the local 12B
    model kept decorating labels (**bold**) and injecting parenthetical
    delivery tags through EVERY prompt-side ban, but small models
    faithfully imitate their OWN prior turns -- so the seam's FORMAT
    example play rides in an ASSISTANT message before the real request
    (system/user/assistant/user keeps the template's role alternation)."""
    system = _seam(pack, "fable2_script_system")
    example = _extract_format_example(system)
    base_user = _script_user_prompt(treatment, digest, envelope, cast_names)
    temps = _MARKUP_LADDER_TEMPS(_TEMP["script"])
    tokens = _script_token_budget(envelope.total_words)
    lo, hi = _word_band(envelope.total_words)

    defects_by_attempt: "list[list[str]]" = []
    budget_rerolls = 0
    truncation_retry_used = False
    attempts = 0
    extra_user: "str | None" = None
    last_defect_text = ""

    for temp in temps:
        while True:
            attempts += 1
            # Strict role alternation: local chat templates (Mistral-Nemo
            # jinja) raise TemplateError on consecutive same-role messages
            # (caught by the first S1b live smoke). Reroll text is folded
            # INTO the final user message; the few-shot example play rides
            # an assistant turn.
            user_content = (
                f"{base_user}\n\n{extra_user}" if extra_user else base_user)
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": (
                    "Before the real assignment: show the exact output "
                    "FORMAT once, as a tiny example episode.")},
                {"role": "assistant", "content": example},
                {"role": "user", "content": user_content},
            ]
            # LLM slot: creative -- P3 whole-play markup (CREATIVE 2;
            # raw text, not structured_call: the markup ladder above is
            # this pass's own retry law).
            raw = creative_fn(msgs, temperature=temp, max_new_tokens=tokens)
            parsed, defects = parse_fable2_markup(raw, cast_names)
            if parsed is None:
                rendered = render_defects(defects)
                defects_by_attempt.append([str(d) for d in defects])
                last_defect_text = rendered
                codes = {d.code for d in defects}
                if (Fable2ParseDefect.MISSING_END in codes
                        and not truncation_retry_used):
                    # Truncation defect: ONE retry with +25% tokens at the
                    # SAME temperature, then the ladder continues.
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
                break  # next rung: same prompt at LOWER temperature
            # Budget gate: CHARACTER words only (r4/M4).
            if not (lo <= parsed.character_word_count <= hi):
                msg = (
                    f"WORD_BUDGET: character words "
                    f"{parsed.character_word_count} outside "
                    f"{int(lo)}-{int(hi)} (target {envelope.total_words})"
                )
                defects_by_attempt.append([msg])
                last_defect_text = msg
                if budget_rerolls >= _MAX_BUDGET_REROLLS:
                    raise Fable2ScriptError("script", msg, attempts)
                budget_rerolls += 1
                verb = ("EXPAND the dialogue"
                        if parsed.character_word_count < lo
                        else "TIGHTEN the dialogue")
                structural = ""
                if (parsed.character_word_count > hi
                        and envelope.total_words < 60):
                    cap = _micro_episode_line_cap(envelope.total_words)
                    structural = (
                        f" Cut LINES, not words: at most {cap} character "
                        f"dialogue lines total, each under 10 words.")
                extra_user = (
                    f"Your draft's CHARACTER dialogue totals "
                    f"{parsed.character_word_count} words; the target is "
                    f"{envelope.total_words} (acceptable {int(lo)} to "
                    f"{int(hi)}). {verb} to hit the target and output the "
                    f"COMPLETE episode again.{structural}"
                )
                continue  # same rung, numeric hint
            return raw, parsed, {
                "defects_by_attempt": defects_by_attempt,
                "rerolls": attempts - 1,
                "attempts": attempts,
                "budget_rerolls": budget_rerolls,
                "truncation_retry": truncation_retry_used,
            }

    raise Fable2ScriptError(
        "script",
        f"markup ladder exhausted; last defects:\n{last_defect_text}",
        attempts)


def _speakers_in_order(parsed: ParsedScript) -> "list[str]":
    seen: "list[str]" = []
    for scene in parsed.scenes:
        for ln in scene.lines:
            if ln.speaker not in seen:
                seen.append(ln.speaker)
    return seen


def _pass_casting(slot_fn, pack, parsed: ParsedScript, treatment: Treatment,
                  menu: VoiceMenu) -> CastingVoices:
    """P6 (registry slot: technical; temp 0.40): voices + portraits ONLY --
    register authority stays with the treatment (r1/A3)."""
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


def _assemble(led: Any, parsed: ParsedScript, treatment: Treatment,
              cast_rows: "list[dict]", payload: "dict[str, Any]",
              meta: dict, *, target_words: int) -> None:
    """Emit ALL FIVE ledger hierarchies (r1/S2) with the proof gates.
    Incremental led.save() after the preamble and after each scene.
    Ambiguity = upstream reroll, never silent-fix; timing stays unset
    (SceneSequencer owns it downstream)."""
    # The winning draft artifact is threaded via meta.fable2 (stamped by
    # the runner just before assembly) -- ParsedScript is frozen and
    # deliberately does not carry the raw markup.
    f2 = meta.setdefault("fable2", {})
    draft_norm = _norm_ws(f2.get("_winning_draft_text") or "")
    if not draft_norm:
        raise Fable2AssembleError(
            "assemble", "winning draft artifact missing from meta.fable2")
    news_read_norm = _norm_ws(treatment.news_close_read)

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
    lo, hi = _word_band(target_words)
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
        proofs = _prove_constituents(
            constituents, artifact_name, artifact_norm, text, line_id)
        proof_map.append({"line_id": line_id, "constituents": proofs})
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

    f2["proof_map"] = proof_map
    f2.pop("_winning_draft_text", None)
    led.save()


# ---------------------------------------------------------------------------
# P8 -- ledger audit (audit-only through S3; r2/M6+CUT1)
# ---------------------------------------------------------------------------

def _script_view(parsed: ParsedScript, treatment: Treatment,
                 *, include_news_read: bool = True) -> str:
    """Human-shaped assembled view for P6/P8 prompts (python-rendered
    from the parsed artifacts; no ledger dependency -- pure module)."""
    lines = [f"TITLE: {parsed.title}"]
    for t in parsed.announcer_intro:
        lines.append(f"{ANNOUNCER_NAME}: {t}")
    for scene in parsed.scenes:
        lines.append(f"SCENE {scene.n}: {scene.setting}")
        for ln in scene.lines:
            lines.append(f"{ln.speaker}: {ln.text}")
    for t in parsed.announcer_outro:
        lines.append(f"{ANNOUNCER_NAME}: {t}")
    lines.append(f"CODA: {parsed.coda}")
    if include_news_read:
        lines.append(f"{ANNOUNCER_NAME} (closing news read): "
                     f"{treatment.news_close_read}")
    return "\n".join(lines)


def _pass_audit(technical_fn, pack, view: str,
                treatment: Treatment) -> AuditFindings:
    user = (
        f"ASSEMBLED EPISODE:\n{view}\n\n"
        f"TREATMENT:\n"
        f"{json.dumps(treatment.model_dump(), ensure_ascii=False, indent=2)}"
        f"\n\nAudit now."
    )
    try:
        # LLM slot: technical -- P8 ledger audit (audit-only through S3).
        return structured_call(
            prompt=[
                {"role": "system",
                 "content": _seam(pack, "fable2_audit_system")},
                {"role": "user", "content": user},
            ],
            schema=AuditFindings,
            slot_fn=technical_fn,
            base_temperature=_TEMP["ledger_audit"],
            structural_retry_temperature=_TEMP["ledger_audit"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=_MAX_NEW_TOKENS["ledger_audit"],
            helper_name="fable2_ledger_audit",
        )
    except StructuredCallFailedError as exc:
        raise Fable2AuditError(
            "ledger_audit", str(exc.last_error), exc.attempts) from exc


# Kill authority (operator lexicon-only kill policy, 2026-07-10): the
# closed-vocabulary classes take the shared lexicons as their ONLY kill
# authority; the python-provable structural classes are re-verified
# against facts python already gated. Everything else is critic-class
# taste -- REPORTED in meta, never fatal at S1b (the coalesced repair is
# deferred post-S3; r4/CUT1 keeps no live-looking repair field).
_AUDIT_LEXICON_BY_CLASS = {
    "weapons_smoking": WEAPON_SMOKING_EVIDENCE,
    "news_source_framing": NEWS_FRAMING_EVIDENCE,
    "machine_attribution": MACHINE_ATTRIBUTION_EVIDENCE,
}
_AUDIT_STRUCTURAL_CLASSES = frozenset({
    "speaker_not_in_cast", "verbatim_break", "skeleton_break",
})


def _triage(findings: AuditFindings, parsed: ParsedScript, view: str,
            cast_names: "list[str]"):
    """Evidence bar (doc P8): returns (confirmed, discarded, reported).
    confirmed = [(finding_dict, evidence_str)] -- the caller FAILS LOUD;
    discarded = uncorroborated hard flags (stamped LOUDLY);
    reported = taste-class notes (meta only, never fatal at S1b)."""
    confirmed: "list[tuple[dict, str]]" = []
    discarded: "list[dict]" = []
    reported: "list[dict]" = []
    # Only the CHARACTER-spoken drama is kill-scannable for news/machine
    # framing -- the announcer's closing news read is REAL news by design
    # (and the coda deliberately pivots toward it); scanning those for
    # 'news' words would kill every episode.
    drama_only = "\n".join(
        f"{ln.speaker}: {ln.text}"
        for scene in parsed.scenes for ln in scene.lines
    )
    for f in findings.findings:
        row = f.model_dump()
        cls = f.finding_class
        if cls in _AUDIT_LEXICON_BY_CLASS:
            scan_hay = (view if cls == "weapons_smoking" else drama_only)
            hit = _scan_lexicon(scan_hay, _AUDIT_LEXICON_BY_CLASS[cls])
            if hit:
                confirmed.append((row, f"lexicon term {hit!r} present"))
            else:
                discarded.append(row)
        elif cls in _AUDIT_STRUCTURAL_CLASSES:
            # Python already gated these facts (parser speaker gate,
            # per-constituent proof, skeleton checks). Python is the
            # judge of record: an LLM flag that contradicts a python-
            # verified fact is a hallucination -- discard LOUDLY.
            discarded.append(row)
        else:
            reported.append(row)
    if discarded:
        log.warning(
            "[scifi_fable2] ledger_audit: DISCARDED %d uncorroborated hard "
            "finding(s) (no lexicon evidence / contradicts python-verified "
            "facts): %s",
            len(discarded),
            "; ".join(f"{d['finding_class']}: {d['detail']}"
                      for d in discarded),
        )
    return confirmed, discarded, reported


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
    n_max = max(1, int(resolved["num_characters"]))
    creative_model = str(resolved["creative_writing_model"])
    technical_model = str(resolved["technical_model"])
    mode = "one_pitch_one_draft"           # S1b low-budget spine

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
                 max_new_tokens: int) -> None:
        receipts.append({
            "pass_id": pass_id, "model_id": model_id, "attempts": attempts,
            "temp": temp, "max_new_tokens": max_new_tokens, "mode": mode,
        })

    f2: dict = {
        "schema_version": _SCHEMA_VERSION,
        "mode": mode,
        "seed": seed,
        "notes": [
            "news_briefs_required is a documented NO-OP for this lane "
            "(interpreter empty by design; the treatment IS the "
            "interpretation).",
            "S1b spine: P2a/P4/P5 land at S2; P8 is audit-only through S3.",
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

    # --- deal + P1: pitch room (ONE-PITCH mode, r2/M3) --------------------
    deck = _load_frame_deck()
    cards, stance = _deal(rng, deck, mode="one_pitch")
    f2["cards_dealt"] = [dict(c) for c in cards]
    f2["stance"] = dict(stance)
    counting_cre, cre_box = _counting(creative_fn)
    with _helper_ctx(slot_scheduler, "fable2_pitch_room"):
        slate = _pass_pitch(
            counting_cre, pack, dossier, cards, stance,
            n_max=n_max, mode="one_pitch")
    _receipt("pitch_room", creative_model, cre_box["calls"],
             _TEMP["pitch_room"], _MAX_NEW_TOKENS["pitch_room"])
    pitch = slate.pitches[0]
    f2["pitches"] = [p.model_dump() for p in slate.pitches]
    f2["selection"] = {
        "chosen_pitch_id": pitch.pitch_id,
        "rationale": "one-pitch mode: the sole dealt pitch wins "
                     "(P2a skipped by design)",
    }

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
        draft_text, parsed, parse_meta = _pass_script(
            counting_cre3, pack, treatment, digest, envelope, cast_names)
    _receipt("script", creative_model, cre3_box["calls"],
             _TEMP["script"], _script_token_budget(target))
    f2["draft1_sha256"] = _sha256(draft_text)
    f2["final_sha256"] = f2["draft1_sha256"]
    f2["better_draft_choice"] = "draft1_one_draft_mode"
    f2["critic"] = None  # P4/P5 land at S2
    f2["parse"] = {
        "defects_by_attempt": parse_meta["defects_by_attempt"],
        "rerolls": parse_meta["rerolls"],
        "normalizations": list(parsed.normalizations),
    }
    if parsed.normalizations:
        log.warning(
            "[scifi_fable2] parser stripped %d decoration(s) from the "
            "winning draft (delete-only; stamped at "
            "meta.fable2.parse.normalizations for the operator eyeball): %s",
            len(parsed.normalizations),
            "; ".join(parsed.normalizations[:12]),
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
        casting = _pass_casting(counting_tech2, pack, parsed, treatment, menu)
    _receipt("casting_voices", technical_model, tech2_box["calls"],
             _TEMP["casting_voices"], _MAX_NEW_TOKENS["casting_voices"])
    speaker_order = _speakers_in_order(parsed)
    cast_rows = _assign_voices(casting, menu, rng, speaker_order)
    f2["casting"] = [c.model_dump() for c in casting.cast]

    # --- P7: assembly (pure python; proof gates; incremental saves) --------
    # The proof artifact is the SAME-normalized draft (delete-only strip;
    # every word LLM-authored) so constituent spans and parsed text share
    # one normalization. draft1_sha256 above hashes the RAW draft.
    f2["_winning_draft_text"] = normalize_fable2_markup_text(draft_text)
    _assemble(led, parsed, treatment, cast_rows, payload, meta,
              target_words=target)
    _receipt("assemble", "python", 0, 0.0, 0)

    # --- P8: ledger audit (audit-only; fail loud on confirmed) -------------
    view = _script_view(parsed, treatment)
    counting_tech3, tech3_box = _counting(technical_fn)
    with _helper_ctx(slot_scheduler, "fable2_ledger_audit"):
        findings = _pass_audit(counting_tech3, pack, view, treatment)
    _receipt("ledger_audit", technical_model, tech3_box["calls"],
             _TEMP["ledger_audit"], _MAX_NEW_TOKENS["ledger_audit"])
    confirmed, discarded_rows, reported = _triage(
        findings, parsed, view, cast_names)
    f2["audit"] = {
        "findings": reported + [row for row, _ev in confirmed],
        "confirmed": [
            {**row, "evidence": ev} for row, ev in confirmed
        ],
        "discarded": discarded_rows,
    }
    f2["pass_receipts"] = receipts
    led.save()
    if confirmed:
        raise Fable2AuditError(
            "ledger_audit",
            "confirmed audit defect(s): " + "; ".join(
                f"{row['finding_class']} (scene {row['scene']}, "
                f"{row['speaker'] or 'scene'}): {row['detail']} "
                f"[{ev}]"
                for row, ev in confirmed
            ))

    # --- tail parts ---------------------------------------------------------
    canon = _OTRC.episode_canon_from_outline_dict({
        "title": treatment.title,
        "premise": treatment.dramatic_question,
        "setting": treatment.setting,
        "time_of_day": _derive_time_of_day(treatment, parsed),
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
        mode, seed, len(cast_names), len(parsed.scenes),
        parsed.character_word_count, len(receipts),
    )
    return Fable2TailParts(
        outline_view=outline_view,
        canon=canon,
        final_title_override=parsed.title,
        run_story_spine=False,
        refine_active=False,
        fable2_meta=f2,
    )
