"""Fixed-topology scifi_news_pro story producer.

Every requested length follows the same path: source dossier, one pitch,
treatment, factual close, first structurally clean script, narrow same-story
safety cleanup, casting, and proof-backed assembly. Requested and actual word
counts are telemetry only. Prose style, visual vocabulary, and word drift
never replace the accepted story or fail the episode.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

try:
    from . import _otr_episode_budget as _OTRB
    from ._otr_structured_call import (
        StructuredCallFailedError,
        structured_call,
    )
    from ._otr_repair_prompts import make_dispatching_repair_factory
    from ._otr_text_metrics import canonical_word_count, set_line_text_metrics
    from ._otr_generation_budget import ProviderCapacityMessages
    from ._otr_scifi_p0_contract import (
        MAX_QUOTE_CHARS,
        p0_source_chunks,
    )
    from ._otr_fable2_markup import (
        ANNOUNCER_NAME,
        Fable2ParseDefect,
        ParseDefect,
        ParsedScript,
        _strip_role_parenthetical,
        normalize_fable2_markup_text,
        parse_fable2_markup,
        render_defects,
    )
    from . import _otr_canon as _OTRC
    from . import _otr_word_delivery as _OTRWD
except ImportError:  # pragma: no cover -- flat test/standalone load
    import _otr_episode_budget as _OTRB  # type: ignore
    from _otr_structured_call import (  # type: ignore
        StructuredCallFailedError,
        structured_call,
    )
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore
    from _otr_text_metrics import (  # type: ignore
        canonical_word_count,
        set_line_text_metrics,
    )
    from _otr_generation_budget import ProviderCapacityMessages  # type: ignore
    from _otr_scifi_p0_contract import (  # type: ignore
        MAX_QUOTE_CHARS,
        p0_source_chunks,
    )
    from _otr_fable2_markup import (  # type: ignore
        ANNOUNCER_NAME,
        Fable2ParseDefect,
        ParseDefect,
        ParsedScript,
        _strip_role_parenthetical,
        normalize_fable2_markup_text,
        parse_fable2_markup,
        render_defects,
    )
    import _otr_canon as _OTRC  # type: ignore
    import _otr_word_delivery as _OTRWD  # type: ignore

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
# load-bearing prompt vocabulary and cannot be renamed, so pydantic's
# name-shadowing warning is expected and is silenced for clean logs.
#
# THE WARNING WAS TELLING THE TRUTH, AND THIS FILTER HELPED IT HIDE
# (PBUG-20260812-02). "nothing in this module ever calls the shadowed
# attribute" was the wrong reassurance: the danger was never a call, it was
# that pydantic read the inherited attribute as the field's DEFAULT and quietly
# made a required contract field optional. That is fixed at the field itself --
# `CastShape.register` is declared `Field(...)` -- and `Field(...)` is what
# makes this filter safe. The suite pins both halves; see
# tests/test_writer_model_field_shadowing.py.
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
    "dossier": 0.30,
    "pitch": 0.90,
    "treatment": 0.40,
    "script": 0.75,
    "casting_voices": 0.40,
}


def _MARKUP_LADDER_TEMPS(t: float) -> "tuple[float, ...]":
    """The markup ladder NEVER raises temperature (2B principle). Four
    rungs (10th live smoke 2026-07-10: with the format wars won, rolls
    were dying on ONE small skeleton slip per attempt -- e.g. a missing
    closing MUSIC line -- so depth, not temperature, is the lever; the
    final rung repeats 0.30 with the defect quote)."""
    return (t, round(t * 0.66, 2), 0.30, 0.30)


_DIGEST_CHAR_CAP = 3600
_DOSSIER_WINDOW_OVERLAP_CHARS = MAX_QUOTE_CHARS - 1
_DOSSIER_FACTS_MAX = 20
_DOSSIER_NUMBERS_MAX = 20
_DOSSIER_ENTITIES_PER_BUCKET_MAX = 10

#: THE SCHEMA CEILING IS A BACKSTOP, NOT THE LIMIT.
#:
#: `_DOSSIER_ENTITIES_PER_BUCKET_MAX` is how many entities the merged dossier
#: KEEPS, and `_merge_window_dossiers` already enforces it deterministically
#: (`_balanced_window_values(limit=...)`). Putting the same number on the
#: pydantic field made it a REFUSAL that fired first: a source naming more
#: than ten places could never be extracted at all, so the trim that exists to
#: handle exactly that case never got to run.
#:
#: Measured 2026-08-14: a UCLA Health story names 11 places and 14
#: institutions. The model extracted them correctly, the schema rejected the
#: reply three times, and the episode died with
#: `Fable2DossierError: 2 validation errors for DossierLLM`. The model was
#: right and the cap was wrong -- and news stories with more than ten proper
#: nouns are ordinary, not exotic.
#:
#: This is the same doctrine the score compiler already applies to a stale cue
#: anchor ("a stale index must never be a fatal count gate") and the same
#: reason the word budget's upper bound was removed: a cap that refuses a
#: legitimate source is a defect, not a safeguard. The ceiling below stays
#: only to keep a degenerate decode finite.
_DOSSIER_ENTITIES_PER_BUCKET_CEILING = 60

#: THE SAME CONTRADICTION, THREE MORE TIMES. `facts_to_keep`,
#: `allowed_numbers` and `dramatizable_vectors` each carried a pydantic cap
#: equal to the number `_merge_dossiers` already trims them to
#: (`_balanced_window_values(limit=...)`), so a source richer than the keep
#: limit was refused before the trim could run -- exactly the defect that
#: killed two `scifi_news_pro` episodes on a UCLA Health story. Found by
#: sweeping every bounded list field in the story lanes after the first one
#: was fixed, which is the only reason these were not three more live
#: failures waiting for a richer source.
_DOSSIER_FACTS_CEILING = 60
_DOSSIER_NUMBERS_CEILING = 60
_DOSSIER_VECTORS_CEILING = 40
_DOSSIER_VECTORS_MAX = 10
_DIGEST_HEADLINE_CHAR_CAP = 240
_DIGEST_SOURCE_CHAR_CAP = 120
_DIGEST_DATE_CHAR_CAP = 64
_DIGEST_SUMMARY_CHAR_CAP = 720
_DIGEST_FRAME_TRIM_MARK = " [...TRIMMED]"

# RETIRED 2026-08-14: this mapped a WORD TOTAL to a scene count. The scene
# count follows the act topology now. Kept only as a tombstone reference.
_RETIRED_SCENE_COUNT_TABLE: "tuple[tuple[int, int], ...]" = (
    (164, 1),
    (274, 2),
    (384, 3),
    (494, 4),
    (604, 5),
    (714, 6),
    (824, 7),
    (2_147_483_647, 8),
)

# The finite P0 source dossier may retain a source-proof reservation. Every
# prose-bearing pass below uses the provider's remaining output capacity.
_MAX_NEW_TOKENS = {"dossier": 700}

_SCHEMA_VERSION = "fable2_v1"

_DECK_PATH = (
    Path(__file__).resolve().parent / "story_packs" / "scifi_news_pro"
    / "frame_deck.json"
)


# ---------------------------------------------------------------------------
# Artifact models (doc s5; pydantic v2)
# ---------------------------------------------------------------------------

class NamedEntities(BaseModel):
    """Extraction metadata. The per-bucket LIMIT is applied by the merge, not
    by this schema -- see `_DOSSIER_ENTITIES_PER_BUCKET_CEILING`."""

    people: list[str] = Field(
        default_factory=list,
        max_length=_DOSSIER_ENTITIES_PER_BUCKET_CEILING,
    )
    places: list[str] = Field(
        default_factory=list,
        max_length=_DOSSIER_ENTITIES_PER_BUCKET_CEILING,
    )
    things: list[str] = Field(
        default_factory=list,
        max_length=_DOSSIER_ENTITIES_PER_BUCKET_CEILING,
    )


class DossierLLM(BaseModel):
    """Source-grounding artifact; prose length is never judged."""

    facts_to_keep: list[str] = Field(
        min_length=1,
        max_length=_DOSSIER_FACTS_CEILING,
    )
    allowed_numbers: list[str] = Field(
        default_factory=list,
        max_length=_DOSSIER_NUMBERS_CEILING,
    )
    named_entities: NamedEntities
    dramatizable_vectors: list[str] = Field(
        default_factory=list,
        max_length=_DOSSIER_VECTORS_CEILING,
    )

    @field_validator("facts_to_keep")
    @classmethod
    def _facts_must_be_nonblank(cls, values):
        if any(not str(value).strip() for value in values):
            raise ValueError("facts_to_keep entries must be nonblank")
        return values


class Pitch(BaseModel):
    frame_card: str
    logline: str
    hook: str
    scifi_device: str
    cast_size: int = Field(ge=1, le=8)
    ending_shape: Literal[
        "paid_victory", "quiet_loss", "ironic_turn", "open_question"
    ]


#: The ONLY real ceiling on speaking characters: one per distinct voice.
#:
#: OPERATOR DIRECTIVE 2026-08-12, ALL BANKS: *"remove the cap, the story can
#: have as many characters as it wants."* So `num_characters` is now a REQUEST,
#: not a gate -- exactly like `target_words` (see the no-word-count-chasing
#: rule). Nothing refuses a story for wanting a third voice.
#:
#: What survives is PHYSICAL, not policy. `_make_casting_validator` enforces
#: *"two characters never share a voice"*, and `_deal_voice_menu` refuses when
#: stock is short, so the cast cannot exceed the voice pool without either
#: giving two characters one voice (a correctness defect the operator calls out
#: by name) or failing at casting AFTER the whole script is written. Gating here
#: fails closed EARLY and says why.
#:
#: Kept as a constant rather than read from the pool at import time, because
#: pydantic needs a literal when the class is built and an import-order
#: dependency here would be worse than a number with a test on it.
#: `tests/test_cast_size_is_a_request.py` asserts this equals the live stock, so
#: a pool that grows or shrinks reports the drift instead of hiding it.
MAX_SPEAKING_CAST = 10


class CastShape(BaseModel):
    """Canonical structural cast-name source."""

    name: str
    role: str
    want: str
    pressure: str
    #: `Field(...)` is not decoration -- it is the whole fix for
    #: PBUG-20260812-02, and REMOVING it silently re-breaks the writer.
    #: `register` also names a method reachable on BaseModel (pydantic's
    #: ModelMetaclass inherits ABCMeta), so a bare `register: str` does NOT
    #: declare a required field: pydantic reads the inherited attribute as the
    #: field's DEFAULT. The field then goes OPTIONAL, the JSON schema handed to
    #: the writer stops requiring it, and any cast shape that omits it carries a
    #: bound method -- which reaches the prompt as "<bound method ...>" and kills
    #: the node on the next json.dumps. `Field(...)` shadows the inherited
    #: attribute and restores the declared intent: required, no default.
    register: str = Field(...)

    @field_validator("name")
    @classmethod
    def _legal_speaker_label(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("cast name must not be blank")
        if value.strip().casefold() == ANNOUNCER_NAME.casefold():
            raise ValueError("cast name must never collide with ANNOUNCER")
        return value.strip()


class Treatment(BaseModel):
    title: str
    dramatic_question: str
    setting: str
    cast_shapes: list[CastShape] = Field(min_length=1,
                                         max_length=MAX_SPEAKING_CAST)
    turn: str
    priced_ending: dict[str, str]
    news_thread: str
    news_close_read: str = ""

    @model_validator(mode="after")
    def _structural_contract(self) -> "Treatment":
        names = [shape.name for shape in self.cast_shapes]
        identity_keys = [" ".join(name.split()).casefold() for name in names]
        if len(set(identity_keys)) != len(identity_keys):
            raise ValueError(
                "cast names must be unique under case-insensitive identity, "
                f"got {names}"
            )
        for key in ("choice", "cost_paid"):
            if not str(self.priced_ending.get(key) or "").strip():
                raise ValueError(f"priced_ending.{key} is required")
        return self


class CastVoice(BaseModel):
    name: str
    role: str
    character_description: str
    gender: Literal["male", "female"]
    age_band: Literal["20s", "30s", "40s", "50s", "60s", "n/a"] = "n/a"

    @field_validator("age_band", mode="before")
    @classmethod
    def _normalize_ageless(cls, value):
        if value is None:
            return "n/a"
        text = str(value).strip().lower().replace("\\", "/")
        if text in {"", "n/a", "na", "none", "null", "unknown", "n.a.", "-"}:
            return "n/a"
        return text

    register: str = ""
    timbre: str
    want: str = ""
    pressure: str = ""


class NewsCloseRead(BaseModel):
    """P2c: the 1-2 sentence factual close (read-split, S1b). Floor 40,
    not the doc's 80 -- the 27th live smoke killed a perfectly factual
    78-char read; brevity is a style preference, never a correctness
    gate (the seam still asks for 1-2 wire-desk sentences)."""

    news_close_read: str = Field(min_length=1)


class CastingVoices(BaseModel):
    cast: list[CastVoice] = Field(min_length=1, max_length=MAX_SPEAKING_CAST)

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


def _stamp_cast_contract(led: Any, *, resolved: Mapping[str, Any],
                         source_bank_row: Any) -> dict:
    """Stamp ``meta.cast_contract`` -- the cameo DECISION this lane never made.

    This lane derives its cast from the script the model already wrote
    (``_speakers_in_order`` -> ``_assign_voices``), so it never reaches
    ``lock_cast()`` and, until now, never stamped this key: every episode
    shipped ``cast_contract: {}`` (PBUG-20260811-03, confirmed on
    ``signal_lost_blood_red_water_20260815_195226``). Nothing failed and
    nothing logged, and a reader could not tell a declined cameo from one that
    was never considered.

    Called AFTER ``_assemble``, which saves incrementally -- ``Ledger.save()``
    rebinds ``led.data``, so the live meta is reacquired here rather than
    written through the caller's stale mapping. ``require_ledger_save`` further
    down persists it.

    No render-path behaviour changes: no cameo is cast, and the contract
    carries no ``cast_seed``, so CastLock's replay branch stays untaken and
    credits keep falling back to ``meta.episode_seed``.
    """
    from . import _otr_casting as _OTRCAST

    contract = _OTRCAST.content_owned_cast_contract(
        source_bank_id=str(
            getattr(source_bank_row, "source_bank_id", "") or ""
        ),
        num_characters_request=int(resolved.get("num_characters") or 0),
        num_characters_locked=_OTRCAST.count_locked_characters(
            led.data.get("cast") or []
        ),
    )
    _ledger_meta(led)["cast_contract"] = contract
    log.info(
        "[scifi_fable2] cast contract stamped: lemmy_hit=%s policy=%s "
        "characters requested=%d locked=%d",
        contract["lemmy_hit"], contract["lemmy_policy"],
        contract["num_characters_request"],
        contract["num_characters_locked"],
    )
    return contract


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


def _word_re(term: str) -> "re.Pattern[str]":
    return re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)


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
            "pitch", f"frame deck unreadable at {p}: {exc}") from exc
    if deck.get("schema_version") != "fable2_deck_v1":
        raise Fable2PitchError(
            "pitch",
            f"frame deck schema_version "
            f"{deck.get('schema_version')!r} != 'fable2_deck_v1'")
    cards = deck.get("cards") or []
    stances = deck.get("stances") or []
    names = [c.get("name") for c in cards]
    if len(cards) < 3 or len(set(names)) != len(names):
        raise Fable2PitchError(
            "pitch",
            f"frame deck needs >= 3 uniquely-named cards, got {len(cards)}")
    if not stances:
        raise Fable2PitchError("pitch", "frame deck has no stances")
    return deck


def _deal(rng: random.Random, deck: dict):
    """Deal one frame card and one stance for every requested length."""
    return (
        (rng.choice(list(deck["cards"])),),
        rng.choice(list(deck["stances"])),
    )


# ---------------------------------------------------------------------------
# Scene envelope (python computes BEFORE any LLM call; doc s3/s13)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SceneEnvelope:
    """Advisory scene plan derived before generation; never a delivery gate.

    2026-08-14: `scene_word_targets` and `total_words` were removed with the
    word authority. This lane carried its OWN copy of the word machinery --
    a word-total-to-scene-count table plus a per-scene word split -- on top
    of the one in `_otr_episode_budget` and the one in the codex circuit.
    The scene count follows the ACT TOPOLOGY now, like every other lane.
    """

    scene_count: int

    def __post_init__(self) -> None:
        if type(self.scene_count) is not int or self.scene_count < 1:
            raise TypeError("scene_count must be a positive strict int")


def _build_envelope(act_count: int) -> SceneEnvelope:
    """Advisory scene count for this act count. Never a delivery gate."""
    acts = int(act_count)
    if not (_OTRB.MIN_ACT_COUNT <= acts <= _OTRB.MAX_ACT_COUNT):
        raise Fable2ScriptError(
            "script",
            f"act_count must be {_OTRB.MIN_ACT_COUNT}..{_OTRB.MAX_ACT_COUNT}, "
            f"got {acts}",
        )
    return SceneEnvelope(scene_count=acts)


def _validate_scene_envelope(envelope: Any) -> None:
    if type(envelope) is not SceneEnvelope:
        raise Fable2ScriptError(
            "final_draft",
            "envelope must be an exact SceneEnvelope artifact",
        )
    if envelope != _build_envelope(envelope.scene_count):
        raise Fable2ScriptError(
            "final_draft",
            "envelope does not match its advisory scene plan",
        )


@dataclass(frozen=True)
class PassAttemptTrace:
    """One observed P3/P5 markup call; never an inferred attempt count."""

    attempt: int
    temperature: float
    outcome: Literal["parse_rejected", "accepted"]
    output_budget_mode: Literal["provider_capacity"] = "provider_capacity"
    selected: bool = False
    parse_defects: "tuple[str, ...]" = ()
    character_words: "int | None" = None
    scene_character_words: "tuple[int, ...]" = ()

    def __post_init__(self) -> None:
        valid_outcomes = {"parse_rejected", "accepted"}
        if type(self.attempt) is not int:
            raise TypeError("attempt must be a strict int")
        if self.attempt < 1:
            raise ValueError("attempt must be positive")
        if (isinstance(self.temperature, bool)
                or not isinstance(self.temperature, (int, float))
                or not math.isfinite(float(self.temperature))):
            raise TypeError("temperature must be one finite number")
        if self.outcome not in valid_outcomes:
            raise ValueError(
                f"outcome must be one of {sorted(valid_outcomes)}, got "
                f"{self.outcome!r}")
        if self.output_budget_mode != "provider_capacity":
            raise ValueError("output_budget_mode must be provider_capacity")
        if type(self.selected) is not bool:
            raise TypeError("selected must be a strict boolean")
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


@dataclass(frozen=True)
class FinalDraft:
    """The first structurally clean, safety-cleaned script authority."""

    raw_source: str
    normalized_source: str
    parsed: ParsedScript
    proof_map: "tuple[ProofMapEntry, ...]"
    p3_attempts: "tuple[PassAttemptTrace, ...]"
    hashes: FinalDraftHashes


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
        # Neither entities nor numbers open a retry here: unverifiable
        # ones are DROPPED by _filter_dossier_entities after the call
        # (28th + 30th live smokes 2026-07-10: world-knowledge
        # expansions and converted figures are unbounded; no structural retry can
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
    -- never retried (world-knowledge expansions like CMU -> Carnegie
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


def _make_pitch_validator(cards, n_max: int):
    """Restore the dealt card identity, and gate on the REAL ceiling only.

    ``n_max`` is the operator's REQUESTED cast size and is deliberately NOT
    enforced (operator directive 2026-08-12, all banks: *"remove the cap, the
    story can have as many characters as it wants"*). It survives as the number
    the prompt ASKS for; see :data:`MAX_SPEAKING_CAST` for what actually binds.
    """
    del n_max
    card_name = cards[0]["name"]

    def _check(pitch: Pitch) -> "str | None":
        pitch.frame_card = card_name
        if pitch.cast_size > MAX_SPEAKING_CAST:
            return (f"pitch cast_size {pitch.cast_size} > {MAX_SPEAKING_CAST}, "
                    f"the number of distinct voices in stock")
        return None

    return _check


def _make_treatment_validator(
    dossier: DossierLLM,
    n_max: int,
    provenance: "dict[str, str]",
    digest: str = "",
):
    """Gate the cast on the REAL ceiling -- the voice stock, not the request."""
    del dossier, provenance, digest, n_max

    def _check(model: Treatment) -> "str | None":
        if len(model.cast_shapes) > MAX_SPEAKING_CAST:
            return (f"cast_shapes {len(model.cast_shapes)} > "
                    f"{MAX_SPEAKING_CAST}, the number of distinct voices in "
                    f"stock")
        return None

    return _check


def _make_casting_validator(menu: VoiceMenu, speakers: "list[str]"):
    """Speaker-set equality + menu-id legality + gender/timbre feasibility
    (validated after the script seal; malformed casting data alone retries)."""
    by_id = menu.by_id()
    want = set(speakers)

    def _check(m: CastingVoices) -> "str | None":
        # The speaker set is LOCKED and Python owns it: these are the names in
        # the sealed script. A name the model re-typed with different case or
        # spacing is a copy of our own string, not a casting decision -- restore
        # it. A genuinely different name is a real defect and still goes back.
        by_normal = {_norm_ws(name).casefold(): name for name in speakers}
        for row in m.cast:
            canonical = by_normal.get(_norm_ws(row.name).casefold())
            if canonical is not None and row.name != canonical:
                row.name = canonical
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

@dataclass(frozen=True)
class _DossierWindow:
    index: int
    body_start: int
    body_end: int
    digest: str


def _raw_digest_header(payload: "Mapping[str, Any]") -> str:
    return (
        f"HEADLINE: {payload.get('headline', '')}\n"
        f"SOURCE: {payload.get('source', '')} {payload.get('date', '')}\n"
        f"SUMMARY: {payload.get('summary', '')}\n\n"
    )


def _bounded_frame_value(value: Any, *, limit: int) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    keep = limit - len(_DIGEST_FRAME_TRIM_MARK)
    return text[:keep].rstrip() + _DIGEST_FRAME_TRIM_MARK


def _digest_header(payload: "Mapping[str, Any]") -> str:
    """Bound repeated framing while leaving the selected body untouched."""
    raw = _raw_digest_header(payload)
    body = str(payload.get("full_text") or payload.get("seed_text") or "")
    if len(raw) + len(body) <= _DIGEST_CHAR_CAP:
        return raw
    return (
        "HEADLINE: "
        + _bounded_frame_value(
            payload.get("headline"),
            limit=_DIGEST_HEADLINE_CHAR_CAP,
        )
        + "\nSOURCE: "
        + _bounded_frame_value(
            payload.get("source"),
            limit=_DIGEST_SOURCE_CHAR_CAP,
        )
        + " "
        + _bounded_frame_value(
            payload.get("date"),
            limit=_DIGEST_DATE_CHAR_CAP,
        )
        + "\nSUMMARY: "
        + _bounded_frame_value(
            payload.get("summary"),
            limit=_DIGEST_SUMMARY_CHAR_CAP,
        )
        + "\n\n"
    )


def _source_body(payload: "Mapping[str, Any]") -> "tuple[str, str]":
    full_text = str(payload.get("full_text") or "")
    if full_text:
        return "full_text", full_text
    return "seed_text", str(payload.get("seed_text") or "")


def _build_source_preview(payload: "Mapping[str, Any]") -> str:
    """Bounded raw-text preview; the merged dossier owns full coverage."""
    _route, body = _source_body(payload)
    return (_digest_header(payload) + body)[:_DIGEST_CHAR_CAP]


def _validate_dossier_windows(
    payload: "Mapping[str, Any]",
    windows: "Sequence[_DossierWindow]",
    *,
    overlap_chars: int,
) -> None:
    """Prove that a proposed window set covers the selected body exactly."""
    if not windows:
        raise Fable2DossierError(
            "dossier", "complete-source window set is empty")
    header = _digest_header(payload)
    _route, body = _source_body(payload)
    covered_end = 0
    previous: "_DossierWindow | None" = None
    for expected_index, window in enumerate(windows):
        if window.index != expected_index:
            raise Fable2DossierError(
                "dossier",
                "complete-source windows have non-sequential indices",
            )
        if (
            window.body_start < 0
            or window.body_end < window.body_start
            or window.body_end > len(body)
        ):
            raise Fable2DossierError(
                "dossier",
                f"source window {window.index} has invalid body coordinates",
            )
        if expected_index == 0 and window.body_start != 0:
            raise Fable2DossierError(
                "dossier", "complete-source coverage does not start at zero")
        if previous is not None:
            if window.body_start > covered_end:
                raise Fable2DossierError(
                    "dossier",
                    f"source window {window.index} leaves a coverage gap",
                )
            if previous.body_end - window.body_start != overlap_chars:
                raise Fable2DossierError(
                    "dossier",
                    f"source window {window.index} overlap drifted from "
                    f"{overlap_chars} characters",
                )
        if body and window.body_end <= covered_end:
            raise Fable2DossierError(
                "dossier",
                f"source window {window.index} does not advance coverage",
            )
        expected_digest = header + body[window.body_start:window.body_end]
        if window.digest != expected_digest:
            raise Fable2DossierError(
                "dossier",
                f"source window {window.index} digest is not its exact slice",
            )
        if len(window.digest) > _DIGEST_CHAR_CAP:
            raise Fable2DossierError(
                "dossier",
                f"source window {window.index} exceeds "
                f"{_DIGEST_CHAR_CAP} characters",
            )
        covered_end = max(covered_end, window.body_end)
        previous = window
    if covered_end != len(body):
        raise Fable2DossierError(
            "dossier",
            f"complete-source coverage ends at {covered_end}, "
            f"not {len(body)}",
        )


def _build_digest_windows(
    payload: "Mapping[str, Any]",
) -> "tuple[_DossierWindow, ...]":
    """Split the complete selected body into bounded overlapping digests."""
    header = _digest_header(payload)
    _route, body = _source_body(payload)
    allowance = _DIGEST_CHAR_CAP - len(header)
    if allowance < 1 and body:
        raise Fable2DossierError(
            "dossier",
            "source framing leaves no room for the selected article body",
        )
    overlap = min(
        _DOSSIER_WINDOW_OVERLAP_CHARS,
        max(0, allowance - 1),
    )
    pairs = p0_source_chunks(
        {"frame": header, "full_text": body},
        budget_chars=_DIGEST_CHAR_CAP,
        overlap_chars=overlap,
    )
    windows = tuple(
        _DossierWindow(
            index=index,
            body_start=offset,
            body_end=offset + len(str(window_payload["full_text"])),
            digest=(
                str(window_payload["frame"])
                + str(window_payload["full_text"])
            ),
        )
        for index, (offset, window_payload) in enumerate(pairs)
    )
    _validate_dossier_windows(payload, windows, overlap_chars=overlap)
    return windows


def _window_value_key(value: str) -> str:
    return _norm_ws(value).casefold()


def _evenly_spaced_positions(count: int, limit: int) -> "list[int]":
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [0]
    return [
        (position * (count - 1)) // (limit - 1)
        for position in range(limit)
    ]


def _balanced_window_values(
    values_by_window: "Sequence[Sequence[str]]",
    *,
    limit: int,
) -> "tuple[list[str], dict[str, int]]":
    """Deduplicate overlap repeats and retain stable cross-window breadth."""
    groups: "list[tuple[int, list[str]]]" = []
    seen: "set[str]" = set()
    seen_count = 0
    for window_index, values in enumerate(values_by_window):
        group: "list[str]" = []
        for value in values:
            seen_count += 1
            key = _window_value_key(value)
            if not key or key in seen:
                continue
            seen.add(key)
            group.append(value)
        if group:
            groups.append((window_index, group))

    selected: "list[str]" = []
    if len(groups) > limit:
        selected = [
            groups[position][1][0]
            for position in _evenly_spaced_positions(len(groups), limit)
        ]
    else:
        depth = 0
        while len(selected) < limit:
            moved = False
            for _window_index, values in groups:
                if depth < len(values):
                    selected.append(values[depth])
                    moved = True
                    if len(selected) == limit:
                        break
            if not moved:
                break
            depth += 1
    return selected, {
        "seen": seen_count,
        "unique": len(seen),
        "retained": len(selected),
    }


def _merge_dossiers(
    window_dossiers: "Sequence[DossierLLM]",
) -> "tuple[DossierLLM, dict[str, dict[str, int]]]":
    """Merge complete-source extractions without widening the schema."""
    if not window_dossiers:
        raise Fable2DossierError(
            "dossier", "complete-source extraction produced no dossiers")

    facts, facts_count = _balanced_window_values(
        [row.facts_to_keep for row in window_dossiers],
        limit=_DOSSIER_FACTS_MAX,
    )
    numbers, numbers_count = _balanced_window_values(
        [row.allowed_numbers for row in window_dossiers],
        limit=_DOSSIER_NUMBERS_MAX,
    )
    vectors, vectors_count = _balanced_window_values(
        [row.dramatizable_vectors for row in window_dossiers],
        limit=_DOSSIER_VECTORS_MAX,
    )
    entity_values: "dict[str, list[str]]" = {}
    counts: "dict[str, dict[str, int]]" = {
        "facts_to_keep": facts_count,
        "allowed_numbers": numbers_count,
        "dramatizable_vectors": vectors_count,
    }
    for field_name in ("people", "places", "things"):
        values, field_count = _balanced_window_values(
            [
                getattr(row.named_entities, field_name)
                for row in window_dossiers
            ],
            limit=_DOSSIER_ENTITIES_PER_BUCKET_MAX,
        )
        entity_values[field_name] = values
        counts[f"named_entities.{field_name}"] = field_count

    if len(window_dossiers) == 1:
        return window_dossiers[0], counts
    return DossierLLM(
        facts_to_keep=facts,
        allowed_numbers=numbers,
        named_entities=NamedEntities(**entity_values),
        dramatizable_vectors=vectors,
    ), counts


def _dossier_sha256(dossier: DossierLLM) -> str:
    return _sha256(json.dumps(
        dossier.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ))


def _ordered_unique(values: "Sequence[str]") -> "list[str]":
    selected: "list[str]" = []
    seen: "set[str]" = set()
    for value in values:
        key = _window_value_key(value)
        if key and key not in seen:
            selected.append(value)
            seen.add(key)
    return selected


def _source_coverage_receipt(
    payload: "Mapping[str, Any]",
    windows: "Sequence[_DossierWindow]",
    *,
    overlap_chars: int,
    attempts_by_window: "Sequence[int]",
    local_dossiers: "Sequence[DossierLLM]",
    merged_dossier: DossierLLM,
    merge_counts: "Mapping[str, Mapping[str, int]]",
    news_seed_receipt: "Mapping[str, Any] | None" = None,
) -> "dict[str, Any]":
    """Validate and receipt complete selected-body extraction."""
    _validate_dossier_windows(
        payload, windows, overlap_chars=overlap_chars)
    if not (
        len(windows)
        == len(attempts_by_window)
        == len(local_dossiers)
    ):
        raise Fable2DossierError(
            "dossier", "source-window extraction receipt lengths disagree")
    if any(
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        for value in attempts_by_window
    ):
        raise Fable2DossierError(
            "dossier", "every source window requires a completed model call")

    route, body = _source_body(payload)
    header = _digest_header(payload)
    raw_header = _raw_digest_header(payload)
    body_sha = _sha256(body)
    receipt_match: "bool | None" = None
    body_source: "str | None" = None
    rss_content_index: "int | None" = None
    rss_content_count: "int | None" = None
    if news_seed_receipt:
        receipt_body_source = news_seed_receipt.get("body_source")
        receipt_index = news_seed_receipt.get("rss_content_index")
        receipt_count = news_seed_receipt.get("rss_content_count")
        if receipt_body_source not in {
            "rss_full", "url_scrape", "summary_fallback", "summary_only",
        }:
            raise Fable2DossierError(
                "dossier", "meta.news_seed has an invalid body_source")
        if (
            type(receipt_count) is not int
            or receipt_count < 0
            or (
                receipt_index is not None
                and (
                    type(receipt_index) is not int
                    or receipt_index < 0
                    or receipt_index >= receipt_count
                )
            )
        ):
            raise Fable2DossierError(
                "dossier", "meta.news_seed has invalid RSS coordinates")
        expected = {
            "body_chars": len(body),
            "body_bytes_utf8": len(body.encode("utf-8")),
            "body_sha256": body_sha,
        }
        if any(news_seed_receipt.get(key) != value
               for key, value in expected.items()):
            raise Fable2DossierError(
                "dossier",
                "complete-source coverage disagrees with meta.news_seed",
            )
        body_source = str(receipt_body_source)
        rss_content_index = receipt_index
        rss_content_count = receipt_count
        receipt_match = True

    return {
        "schema_version": "fable2_source_coverage_v1",
        "coverage_complete": True,
        "source_field": route,
        "body_source": body_source,
        "rss_content_index": rss_content_index,
        "rss_content_count": rss_content_count,
        "body_chars": len(body),
        "body_bytes_utf8": len(body.encode("utf-8")),
        "body_sha256": body_sha,
        "framing_truncated": raw_header != header,
        "framing_original_chars": len(raw_header),
        "framing_original_sha256": _sha256(raw_header),
        "framing_used_chars": len(header),
        "framing_used_sha256": _sha256(header),
        "canonical_source_chars": len(header + body),
        "canonical_source_bytes_utf8": len(
            (header + body).encode("utf-8")
        ),
        "canonical_source_sha256": _sha256(header + body),
        "digest_char_cap": _DIGEST_CHAR_CAP,
        "window_overlap_chars": overlap_chars,
        "window_count": len(windows),
        "news_seed_receipt_match": receipt_match,
        "windows": [
            {
                "index": window.index,
                "body_start": window.body_start,
                "body_end": window.body_end,
                "body_sha256": _sha256(
                    body[window.body_start:window.body_end]
                ),
                "digest_sha256": _sha256(window.digest),
                "dossier_sha256": _dossier_sha256(dossier),
                "attempts": attempts,
            }
            for window, attempts, dossier in zip(
                windows, attempts_by_window, local_dossiers
            )
        ],
        "merge_counts": {
            key: dict(value) for key, value in merge_counts.items()
        },
        "merged_dossier_sha256": _dossier_sha256(merged_dossier),
    }


def _extract_complete_source_dossier(
    technical_fn,
    pack,
    payload: "Mapping[str, Any]",
    *,
    slot_scheduler: Any,
    news_seed_receipt: "Mapping[str, Any] | None" = None,
) -> "tuple[DossierLLM, list[str], dict[str, Any], int]":
    """Extract every bounded source window, then merge deterministically."""
    windows = _build_digest_windows(payload)
    allowance = _DIGEST_CHAR_CAP - len(_digest_header(payload))
    overlap = min(
        _DOSSIER_WINDOW_OVERLAP_CHARS,
        max(0, allowance - 1),
    )
    # Revalidate here so a replaced or monkeypatched builder cannot silently
    # omit the tail before any model call fires.
    _validate_dossier_windows(payload, windows, overlap_chars=overlap)

    fn, box = _counting(technical_fn)
    local_dossiers: "list[DossierLLM]" = []
    attempts_by_window: "list[int]" = []
    dropped: "list[str]" = []
    for window in windows:
        before = box["calls"]
        with _helper_ctx(slot_scheduler, "fable2_dossier"):
            local = _pass_dossier(fn, pack, window.digest)
        attempts = box["calls"] - before
        local, local_dropped = _filter_dossier_entities(
            local, window.digest)
        local_dossiers.append(local)
        attempts_by_window.append(attempts)
        dropped.extend(local_dropped)

    dossier, merge_counts = _merge_dossiers(local_dossiers)
    coverage = _source_coverage_receipt(
        payload,
        windows,
        overlap_chars=overlap,
        attempts_by_window=attempts_by_window,
        local_dossiers=local_dossiers,
        merged_dossier=dossier,
        merge_counts=merge_counts,
        news_seed_receipt=news_seed_receipt,
    )
    return (
        dossier,
        _ordered_unique(dropped),
        coverage,
        box["calls"],
    )


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


def _pass_pitch(
    creative_fn, pack, dossier: DossierLLM, cards, stance, *, n_max: int,
) -> Pitch:
    card = cards[0]
    user = (
        "SOURCE DOSSIER:\n"
        f"{json.dumps(dossier.model_dump(), ensure_ascii=False, indent=2)}\n\n"
        f"FRAME CARD DEALT: {card['name']} -- {card['shape']}\n\n"
        f"EDITORIAL STANCE: {stance['name']} -- {stance['note']}\n\n"
        f"REQUESTED cast size: {n_max}. This is a REQUEST, not a limit -- use "
        f"as many speaking characters as the story genuinely needs, up to "
        f"{MAX_SPEAKING_CAST}.\nWrite the pitch now."
    )
    try:
        # LLM slot: creative -- P1 pitch.
        return structured_call(
            prompt=ProviderCapacityMessages([
                {"role": "system", "content": _seam(pack, "fable2_pitch_system")},
                {"role": "user", "content": user},
            ]),
            schema=Pitch,
            slot_fn=creative_fn,
            base_temperature=_TEMP["pitch"],
            structural_retry_temperature=round(_TEMP["pitch"] / 2, 2),
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_pitch_validator(cards, n_max),
            max_new_tokens=None,
            helper_name="fable2_pitch",
        )
    except StructuredCallFailedError as exc:
        raise Fable2PitchError(
            "pitch", str(exc.last_error), exc.attempts,
        ) from exc


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
        f"REQUESTED cast size: {n_max}. This is a REQUEST, not a limit -- use "
        f"as many speaking characters as the story genuinely needs, up to "
        f"{MAX_SPEAKING_CAST}.\n"
        f"Write the treatment now."
    )
    try:
        # LLM slot: creative -- P2b treatment (cast names born here).
        return structured_call(
            prompt=ProviderCapacityMessages([
                {"role": "system",
                 "content": _seam(pack, "fable2_treatment_system")},
                {"role": "user", "content": user},
            ]),
            schema=Treatment,
            slot_fn=creative_fn,
            base_temperature=_TEMP["treatment"],
            structural_retry_temperature=_TEMP["treatment"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_treatment_validator(
                dossier, n_max, provenance, digest),
            max_new_tokens=None,
            helper_name="fable2_treatment",
        )
    except StructuredCallFailedError as exc:
        raise Fable2TreatmentError(
            "treatment", str(exc.last_error), exc.attempts) from exc


def _news_read_source_anchors(dossier: DossierLLM) -> "tuple[str, ...]":
    """Verbatim strings that would PROVE the factual close named its source.

    The twin of `_otr_scifi_codex._news_coda_source_anchors`, reading this
    lane's dossier instead of that lane's fact index. Names shorter than three
    characters are dropped for the same reason: a one- or two-letter "entity"
    matches inside ordinary words and would let a close that names nothing
    pass by accident.
    """
    anchors: "list[str]" = []
    entities = dossier.named_entities
    for bucket in (entities.people, entities.places, entities.things):
        for raw in bucket:
            name = str(raw or "").strip()
            if len(name) >= 3:
                anchors.append(name)
    for raw in dossier.allowed_numbers:
        token = str(raw or "").strip()
        if token:
            anchors.append(token)
    seen: "set[str]" = set()
    ordered: "list[str]" = []
    for anchor in anchors:
        key = anchor.casefold()
        if key not in seen:
            seen.add(key)
            ordered.append(anchor)
    return tuple(ordered)


def _make_news_read_validator(dossier: DossierLLM, cast_names: "list[str]"):
    """The source-attribution gate this pass shipped without.

    P6's codex twin has verified and cleaned its coda since it was built;
    this lane passed NO post_validator at all, so the pro lane's closing read
    -- the one that tells a listener where the fact stopped and the fiction
    started -- was never checked for naming its source or for leaking a
    fictional character into a factual report.

    Both findings are PROVENANCE, not prose: whether a real source is named,
    and whether the fact/fiction boundary held. Length, register, sentence
    count and craft are not inspected and never will be here -- THE LAW.
    """
    anchors = _news_read_source_anchors(dossier)
    fiction = tuple(
        name for name in (str(n or "").strip() for n in cast_names)
        if len(name) >= 3
    )

    def _check(read: NewsCloseRead) -> "str | None":
        text = _norm_ws(read.news_close_read)
        findings: "list[str]" = []
        # An empty dossier cannot prove anything, so it does not accuse:
        # a close is only asked to name a source when a source was indexed.
        if anchors and not any(_word_re(a).search(text) for a in anchors):
            findings.append(
                "the closing read never names the real source -- none of the "
                "dossier's entities or numbers appears in it, so the listener "
                "is never told where the fact stopped and the fiction "
                "started. Name at least one of them verbatim"
            )
        spoken_fiction = [n for n in fiction if _word_re(n).search(text)]
        if spoken_fiction:
            findings.append(
                "the closing read is a FACTUAL report and it names invented "
                f"characters ({', '.join(sorted(spoken_fiction))}). Report "
                "only what the source says, using the source's own names"
            )
        return "; ".join(findings) if findings else None

    return _check


def _pass_news_read(
    technical_fn,
    pack,
    dossier: DossierLLM,
    provenance: "dict[str, str]",
    digest: str,
    cast_names: "list[str]",
) -> NewsCloseRead:
    """Author one source-grounded factual close through a typed ladder."""
    user = (
        "SOURCE DOSSIER:\n"
        + json.dumps(dossier.model_dump(), ensure_ascii=False, indent=2)
        + "\n\nPROVENANCE: "
        + json.dumps(provenance, ensure_ascii=False)
        + "\n\nBOUNDED SOURCE PREVIEW (names and numbers may be quoted; "
        + "the dossier above owns complete-source coverage):\n"
        + digest
        + "\n\nFICTIONAL CAST NAMES (never use these in the factual read): "
        + (", ".join(cast_names) or "(none)")
        + "\n\nWrite the closing news read now."
    )
    try:
        # LLM slot: technical -- source-grounded factual close.
        return structured_call(
            prompt=ProviderCapacityMessages([
                {
                    "role": "system",
                    "content": _seam(pack, "fable2_news_read_system"),
                },
                {"role": "user", "content": user},
            ]),
            schema=NewsCloseRead,
            slot_fn=technical_fn,
            base_temperature=0.20,
            structural_retry_temperature=0.10,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_news_read_validator(dossier, cast_names),
            max_new_tokens=None,
            helper_name="fable2_news_read",
        )
    except StructuredCallFailedError as exc:
        raise Fable2TreatmentError(
            "news_read", str(exc.last_error), exc.attempts
        ) from exc


def _script_user_prompt(
    treatment: Treatment,
    digest: str,
    envelope: SceneEnvelope,
    cast_names: "list[str]",
) -> str:
    cast_block = "\n".join(
        f"- {shape.name} ({shape.role}): wants {shape.want}; "
        f"pressure {shape.pressure}; register: {shape.register}"
        for shape in treatment.cast_shapes
    )
    return (
        "TREATMENT:\n"
        + json.dumps(
            treatment.model_dump(), ensure_ascii=False, indent=2
        )
        + "\n\nCAST (the only legal speakers besides ANNOUNCER; use these "
        + f"canonical roster labels):\n{cast_block}\n\n"
        + "BOUNDED SOURCE PREVIEW (fiction fuel; the treatment's dossier "
        + f"owns complete-source coverage):\n{digest}\n\n"
        + f"SHAPE: this episode runs {envelope.scene_count} scene"
        + ("s" if envelope.scene_count != 1 else "")
        + ". Let each scene run as long as it needs; never count, pad, "
        + "compress, or mention a word quota.\n\n"
        + "FORMAT REMINDER: every nonblank row uses one structural label: "
        + "TITLE, MUSIC, SCENE, ANNOUNCER, a cast roster label, CODA, or END. "
        + "Spoken text may use ordinary punctuation, including quotation marks, "
        + "parentheses, brackets, and internal colons. Add no unlabelled stage-"
        + "direction rows. ANNOUNCER speaks only in the intro and outro. End in "
        + "this order: ANNOUNCER outro, CODA, closing MUSIC, END.\n\n"
        + "Write the complete episode now."
    )


_TITLE_LINE_RE = re.compile(r"^\s*TITLE\s*:", re.IGNORECASE)
_END_LINE_RE = re.compile(r"^\s*END\s*\.\s*$", re.IGNORECASE)


def _strip_conversational_wrapper(raw: str) -> str:
    """Drop the chatter a model wraps around the script -- never the script.

    The markup format's OWN skeleton is TITLE: first, END. last. So anything
    BEFORE the first TITLE: line, or AFTER the END. line, is by definition not
    part of the episode: it is the model talking to us about the episode.

    A live 420-word run showed a chatty model opening its otherwise valid
    episode with a sentence announcing that it would now write the script.
    -> BAD_LINE_SHAPE (line 1) + SKELETON_BREAK (TITLE is not the first line).
    The markup ladder is a RAW-TEXT pass -- there is no schema to constrain the
    shape -- so it simply re-asked, five times, and the model politely announced
    itself every single time. Five full episode generations, then a dead leg,
    because of one sentence of throat-clearing in front of a script that was
    otherwise fine.

    This is NOT Python authoring story text, and it is not a shim: it removes a
    conversational wrapper the model put AROUND the artifact, exactly as
    `_strip_reasoning_tags` already removes a <think> preamble in the OpenRouter
    backend. Same class -- a chatty/reasoning model wraps its answer. Python owns
    the SHAPE; the LLM owns the WORDS, and not one word of the script is touched.

    Fail-safe by construction: if there is no TITLE: line at all we return the
    text UNCHANGED, so a genuinely malformed draft still raises its real defects
    (SKELETON_BREAK) instead of being silently massaged into something else.
    """
    if not raw:
        return raw
    lines = raw.splitlines()

    start = next(
        (i for i, ln in enumerate(lines) if _TITLE_LINE_RE.match(ln)), None)
    if start is None:
        return raw  # no script here -- let the parser report the real defects

    end = next(
        (i for i in range(len(lines) - 1, start - 1, -1)
         if _END_LINE_RE.match(lines[i])), None)
    stop = (end + 1) if end is not None else len(lines)

    if start == 0 and stop == len(lines):
        return raw  # already clean -- byte-identical, never touched

    trimmed = "\n".join(lines[start:stop])
    log.info(
        "[scifi_fable2] stripped a conversational wrapper: %d line(s) before "
        "TITLE:, %d after END. (the script itself is untouched)",
        start, len(lines) - stop,
    )
    return trimmed


#: A token that OPENS with one of these is a stage direction wearing a label,
#: not a speaker. ``*`` joined ``(`` and ``[`` on 2026-08-12 -- see
#: :func:`_standalone_stage_direction_repair_note`.
_STAGE_DIRECTION_PREFIXES = ("(", "[", "*")

#: The two defect codes a stage-direction-shaped row can raise. Compared as ENUM
#: MEMBERS, never as strings: ``Fable2ParseDefect`` is a plain ``enum.Enum``, not
#: a ``str`` enum, so ``defect.code in ("UNKNOWN_SPEAKER", ...)`` is False
#: FOREVER -- the note would silently never fire and every stage-direction leg
#: would burn its four attempts again, from code that reads correctly.
#:
#: ``SKELETON_BREAK`` IS DELIBERATELY ABSENT and listing it would be dead code.
#: Every ``skeleton()`` detail in ``_otr_fable2_markup`` is a descriptive English
#: sentence that merely CONTAINS the token -- the row reads ``character line
#: (*SFX) after the last scene`` -- so it can never open with a marker. It was
#: briefly listed here on 2026-08-12 and matched nothing. Nothing is lost: a
#: stage-direction row that trips a skeleton break always raises
#: ``UNKNOWN_SPEAKER`` for the same line first.
_STAGE_DIRECTION_CODES = (
    Fable2ParseDefect.BAD_LINE_SHAPE,
    Fable2ParseDefect.UNKNOWN_SPEAKER,
)


def require_ledger_save(led, what: str) -> None:
    """Save the ledger and REFUSE if it did not land.

    ``Ledger.save()`` returns the path on success and **None on failure, and
    never raises** (`production_ledger.py`). So a bare ``led.save()`` is a write
    that can silently not happen -- and the ledger is the source of truth every
    downstream node reads from DISK. A boundary that "saved" without saving
    hands the next node stale or absent state and the render goes wrong
    somewhere else entirely, with nothing pointing back here.

    This is the third form of one defect this session: PBUG-20260812-02 (a
    value that could not serialize), PBUG-20260812-04 (a model reaching the
    ledger), and the deal receipt that could silently not persist. In each case
    the write failed quietly and the consequence surfaced far away.

    Use at REQUIRED boundaries -- where the next stage reads what this one
    wrote. Diagnostic checkpoints may still warn and continue, but they must
    say so out loud rather than swallowing the result.
    """
    if led.save() is None:
        raise Fable2Error(
            "ledger_save",
            "the ledger did not persist after %s -- refusing to continue, "
            "because every downstream node reads it from disk and would read "
            "stale or missing state. The preceding [OTR_Ledger] warning names "
            "the cause and the offending field." % what,
        )


def _speaker_identity_key(value: str) -> str:
    """Case/spacing-insensitive speaker identity, matching the parser's own."""
    return " ".join(str(value).split()).casefold()


def _resolves_to_cast(label: str, roster: "set[str]") -> bool:
    """Whether ``label`` names someone on the roster, the parser's way.

    Mirrors the two-step lookup at `_otr_fable2_markup` `on_speaker`: exact
    identity first, then the same trailing-role-parenthetical fallback, so
    ``Ada (Engineer)`` resolves to Ada exactly as it does during parsing.
    Imported rather than reimplemented -- the parser is the one authority on
    what counts as the same speaker, and a second copy of that rule is a second
    answer waiting to disagree.
    """
    if _speaker_identity_key(label) in roster:
        return True
    bare = _strip_role_parenthetical(label)
    return bare != label and _speaker_identity_key(bare) in roster


def _undecorated_label(label: str) -> str:
    """A decorated speaker label with its decoration removed, else ``""``.

    ``*Ada`` -> ``Ada``; ``(NARRATOR)`` -> ``NARRATOR``; ``Ada`` -> ``""``,
    because an undecorated label is not this note's business.

    Only ever called with an ``UNKNOWN_SPEAKER`` detail, which the parser sets
    to the SUPPLIED LABEL alone. It must not be pointed at a ``BAD_LINE_SHAPE``
    detail, which is a line fragment (``line[:80]``) rather than a label.

    ONLY THE CLOSER THAT MATCHES THE OPENER comes off. Stripping any trailing
    bracket instead turns ``*Ada (Engineer)`` into ``Ada (Engineer`` -- it eats
    the role parenthetical's own closer, the roster lookup misses, and a real
    character is handed the delete-me rule. The trailing role parenthetical is
    the PARSER's to resolve (`_strip_role_parenthetical`), not this function's.
    """
    token = str(label).strip()
    if not token.startswith(_STAGE_DIRECTION_PREFIXES):
        return ""
    closer = {"(": ")", "[": "]", "*": "*"}[token[0]]
    token = token[1:].strip()
    if token.endswith(closer):
        token = token[:-1].strip()
    return token


def _standalone_stage_direction_repair_note(defects, *, cast_names):
    """Give the repair rung an explicit rule for illegal stage-direction rows.

    The parser reports the offending source row and line number. Keep that
    evidence in the repair instruction so a retry repairs the named format
    defect instead of regenerating a similarly shaped whole-play artifact.

    SCOPED TOO NARROWLY UNTIL 2026-08-12 (PBUG-20260812-03), and it cost a live
    leg. It fired only on ``BAD_LINE_SHAPE`` whose detail opened with ``(`` or
    ``[``. A model wrote ``*SFX: ...``, which HAS a colon and therefore parses
    as a SPEAKER -- so the defects were ``UNKNOWN_SPEAKER: *SFX (line 25)`` and
    ``SKELETON_BREAK``, this note returned "", and the repair rung got only the
    generic "repair the defects below". The model re-emitted the same shape four
    attempts running, the ladder exhausted, and the episode died in the writer
    having never reached a video engine -- the third time this module's own
    docstring records that exact ending.

    STILL DELIBERATELY NARROW where it matters: the note only fires when the
    offending TOKEN actually looks like a stage direction. An
    ``UNKNOWN_SPEAKER`` for a misspelled cast name must NOT be told to fold the
    row into a neighbouring line -- that would be wrong advice, and losing a
    real character's line is worse than the failure it replaces.

    ``cast_names`` IS WHAT MAKES THAT PROMISE TRUE, and the shape check alone
    did not. ``_canonicalize_transport_line`` strips only BALANCED decoration,
    so a real cast member wearing a stray unmatched marker -- ``*Ada: Hello`` --
    survives to the roster lookup, misses, and raises ``UNKNOWN_SPEAKER: *Ada``:
    a real character's line that LOOKS exactly like a stage direction.

    SO THERE ARE TWO RULES, NOT ONE RULE AND A MUTE BUTTON. Going silent on a
    roster hit was the first design, and it was wrong for the same reason the
    original defect was wrong: silence hands back the generic "repair the
    defects below", which is exactly the instruction that failed four attempts
    running. A decorated REAL name has an obvious, safe repair -- restore the
    canonical label and keep the dialogue -- so say that instead.

    Both branches take their token from ``UNKNOWN_SPEAKER`` details only.
    ``BAD_LINE_SHAPE`` carries a LINE FRAGMENT rather than a label, so a roster
    lookup against it is a category error and is not attempted.
    """
    roster = {_speaker_identity_key(name) for name in cast_names}
    roster.add(_speaker_identity_key(ANNOUNCER_NAME))

    for defect in defects:
        if defect.code not in _STAGE_DIRECTION_CODES:
            continue
        detail = str(defect.detail).strip()
        where = (" (reported at line %d)" % defect.line_no
                 if defect.line_no is not None else "")

        if defect.code is Fable2ParseDefect.UNKNOWN_SPEAKER:
            label = _undecorated_label(detail)
            if not label:
                continue  # an undecorated unknown name -- not our business
            if _resolves_to_cast(label, roster):
                return (
                    "\nFORMAT REPAIR RULE: a legal cast member was given a "
                    "decorated speaker label. The parser reported the label "
                    f"{detail!r}{where}, which is the character {label!r} "
                    "wearing stray markup. Restore the plain canonical label "
                    f"-- write the row as '{label}: ' followed by the dialogue "
                    "-- and KEEP THE DIALOGUE EXACTLY AS WRITTEN. Do not delete "
                    "this line, do not merge it into a neighbouring line, and "
                    "do not rename the character. A speaker label carries no "
                    "asterisks, parentheses or brackets."
                    "\nReturn no explanation, markdown, or repair commentary."
                )
            evidence = (f"The parser reported the illegal speaker label "
                        f"{detail!r}{where}.")
        else:
            if not detail.startswith(_STAGE_DIRECTION_PREFIXES):
                continue
            evidence = (f"The parser reported this malformed line, which may be "
                        f"truncated: {detail!r}{where}.")

        return (
            "\nFORMAT REPAIR RULE: a standalone parenthetical, bracketed "
            "or asterisked row is an illegal stage direction. Do not return "
            "it as its own line, and do not use it as a SPEAKER LABEL -- a "
            "row like '*SFX: a door slams' is a stage direction wearing a "
            "label, not a character, and there is no sound-effect speaker. "
            "Preserve a necessary event only by folding it into the "
            "nearest legal labelled spoken line as natural words, or omit "
            "it when nonessential. Every nonblank row must begin with a "
            "legal label; do not invent an unlabeled narration row."
            f"\n{evidence} It must not appear as a standalone output row. "
            "Return no explanation, markdown, or repair commentary."
        )
    return ""


#: The four forms the parser accepts, quoted exactly as it will read them.
#: Sourced from `_otr_fable2_markup._RE_END`'s grammar rather than restated by
#: hand -- a diagnostic that drifts from its validator teaches the model the
#: wrong target, which is worse than saying nothing.
_END_ACCEPTED_FORMS = ("END", "END.", "[END]", "[END.]")


def _end_delimiter_repair_note(defects) -> str:
    """Tell the repair rung the SHAPE it must produce, not just the offence.

    THE DEFECT (PBUG-20260815-03). The ladder burned all four rungs re-emitting
    a bare ``END`` while the parser demanded ``END.``. The plumbing was fine --
    each rung really did carry the rejected draft plus its defect list forward.
    What it carried was useless: ``BAD_LINE_SHAPE`` and ``MISSING_END`` describe
    what is WRONG and never once state the one-character fix that would satisfy
    them. A model cannot infer "add a period" from "your line's shape is bad"
    and "the terminal marker is missing" -- those are two symptoms of one
    omission, not two facts to reason from.

    So this note names the accepted literals. Where the accepted set is small
    and fixed, showing the strings beats describing the pattern.

    ITS OWN HELPER, not an extra branch on
    `_standalone_stage_direction_repair_note`. That one answers a different
    question (a stage direction wearing a speaker label) from different defect
    data, and folding two unrelated diagnoses into one function is how the next
    reader learns the wrong rule for their defect.

    Returns "" when no END-shaped defect is present, so an unrelated repair turn
    is not handed a lecture about a delimiter it got right.
    """
    # Read the REAL record: `ParseDefect.code` is a `Fable2ParseDefect` enum,
    # not a string. An earlier cut compared `str(code)` to "MISSING_END" --
    # which never matches, because str() of that enum is
    # "Fable2ParseDefect.MISSING_END". It would have shipped as a helper that
    # could not fire, with a green unit suite behind it, because the test stub
    # invented a `.kind` attribute production never sets.
    missing_end = False
    end_shaped_bad_line = False
    for defect in defects or ():
        code = getattr(defect, "code", None)
        if code is Fable2ParseDefect.MISSING_END:
            missing_end = True
        elif code is Fable2ParseDefect.BAD_LINE_SHAPE:
            # The bare `END` the model wrote arrives here as the defect detail.
            if "END" in str(getattr(defect, "detail", "") or "").upper():
                end_shaped_bad_line = True
    if not (missing_end or end_shaped_bad_line):
        return ""
    forms = ", ".join("'%s'" % f for f in _END_ACCEPTED_FORMS)
    return (
        "\nFORMAT REPAIR RULE -- THE TERMINAL MARKER. The script must end with "
        f"a line that is EXACTLY one of these, and nothing else: {forms}. "
        "The marker sits alone on the final line with no trailing words and no "
        "unpaired bracket -- 'END' followed by anything else, '[END' without "
        "its closing bracket, and 'END]' without its opening one are all "
        "rejected. Emit the marker once, as the last line of the episode."
    )


#: Characters per token for English prose, which tokenizes near 4.0 on the
#: models this pipeline runs. The three factors below MULTIPLY, so stacking
#: a pessimistic value here on top of two safety margins is not caution --
#: it compounds into dropping drafts that fit comfortably. The first version
#: did exactly that and lost the ceiling-length episodes the guard exists
#: for. Sanity-checked at the structural ceiling: even if real tokenization
#: were 3.5, a 1520-word draft plus its prompt and reply still fits 8192.
_CHARS_PER_TOKEN = 4.0

#: Headroom on the estimated reply. The corrected episode is about the same
#: length as the draft being corrected, plus slack for a model that expands
#: slightly while repairing.
_REPAIR_REPLY_MARGIN = 1.15

#: Safety margin on the whole turn, covering the system prompt, the defect list
#: and tokenizer variance -- none of which are measured here.
_REPAIR_TURN_HEADROOM = 1.10


def _draft_fits_repair_turn(base_user: str, draft: str,
                            system: str = "") -> bool:
    """Whether the rejected draft can ride along without crowding the reply.

    Budgets the WHOLE TURN -- prompt plus the reply the model still has to
    write -- rather than allowing the prompt a flat fraction of the window.

    THE FLAT FRACTION WAS WRONG AND QA CAUGHT IT (2026-08-12). The first
    version allowed the prompt 55% of an 8192-token window at a pessimistic 3
    chars/token, i.e. ~13.5k characters for base_user + draft. Measured against
    REAL inputs, a full-length episode blows straight through that: a maxed
    3600-char digest gives a ~6.1k-char `base_user`, and a 1520-word draft (the
    documented structural ceiling) is ~9.2k characters -- 15.3k together. So
    the guard dropped the draft for exactly the full-length episodes its own
    comment said it existed to serve, silently restoring the cold-regeneration
    bug this change was written to fix. Worse, the test that was supposed to
    cover it asserted on a 400-character toy play, so it passed throughout.

    Approximate by design: an exact tokenizer count would bind this to one
    provider's tokenizer, and the decision only has to be SAFE. The asymmetry
    still holds -- "no" costs one cold repair turn, "yes" when it does not fit
    truncates the prompt -- which is what the margins are for.
    """
    if not draft:
        return False
    # Read lazily with a floor. The catalog owns the real number and honours
    # OTR_HARD_VRAM_CONTEXT_LIMIT for bigger hardware; a module-level import
    # here would add an import-order dependency for a heuristic, and a missing
    # catalog must not be able to break script generation.
    cap = 8192
    try:
        from . import _otr_model_catalog as _catalog

        cap = int(_catalog.HARD_VRAM_CONTEXT_LIMIT) or cap
    except Exception:                       # pragma: no cover -- flat/test load
        try:
            import _otr_model_catalog as _catalog  # type: ignore

            cap = int(_catalog.HARD_VRAM_CONTEXT_LIMIT) or cap
        except Exception:
            pass
    # COUNT THE WHOLE TURN, INCLUDING WHAT THE OTHER FIXES ADDED.
    #
    # This counted only `base_user + draft`, and that was already wrong by the
    # time it shipped: the one-shot FORMAT EXAMPLE is injected as its own
    # user+assistant pair on EVERY call, and the system prompt rides along too.
    # Neither was in the arithmetic. So the fix that shows the model the grammar
    # silently ate part of the budget the fix that shows it the draft depends
    # on -- and this guard's failure mode is to drop the draft, i.e. to revert
    # the repair loop to cold regeneration without saying so.
    #
    # Both are now counted. `_FABLE2_FORMAT_EXAMPLE` is the real string rather
    # than an estimate, and the system prompt is passed in.
    overhead = len(_FABLE2_FORMAT_EXAMPLE) + len(system or "")
    prompt_tokens = (len(base_user) + len(draft) + overhead) / _CHARS_PER_TOKEN
    reply_tokens = (len(draft) / _CHARS_PER_TOKEN) * _REPAIR_REPLY_MARGIN
    needed = (prompt_tokens + reply_tokens) * _REPAIR_TURN_HEADROOM
    return needed <= cap


def _run_markup_ladder(
    creative_fn,
    *,
    pass_id: Literal["script"],
    system: str,
    base_user: str,
    envelope: SceneEnvelope,
    cast_names: "list[str]",
    initial_temperature: float,
    format_example: "str | None" = None,
) -> "tuple[str, ParsedScript, dict]":
    """Return the first structurally clean whole-play parse.

    Every attempt receives the provider's full remaining output capacity.
    Malformed markup may consume another rung; authored prose, punctuation,
    casing, and word count never do.
    """
    del envelope
    temps = _MARKUP_LADDER_TEMPS(initial_temperature)
    defects_by_attempt: "list[list[str]]" = []
    traces: "list[PassAttemptTrace]" = []
    extra_user: "str | None" = None
    last_defect_text = ""
    #: The draft the NEXT attempt has to repair. Empty until an attempt is
    #: rejected, and reset to "" whenever the draft could not be carried.
    rejected_draft = ""
    cold_regenerations = 0
    last_temp: "float | None" = None

    for temp in temps:
        attempt = len(traces) + 1
        # A REPAIR TURN CARRIES THE DRAFT IT IS REPAIRING (r1 2026-08-12).
        #
        # It did not, and that was the defect. The retry said "Keep the same
        # story, cast, events, and wording wherever the format is already
        # valid" -- to a model that had never been shown the text. Every
        # attempt after the first was a COLD REGENERATION carrying a list of
        # complaints about an invisible draft, which is why four attempts
        # produced four DIFFERENT malformed shapes instead of converging on one
        # repaired script, and why a precisely-worded repair rule
        # (PBUG-20260812-03) bought so little: it named the offending row to a
        # model that then had to rewrite the whole play from nothing.
        #
        # `_MARKUP_LADDER_TEMPS` is the proof this was never intended: it
        # decays to 0.30 and its own docstring calls that rung "repeats 0.30
        # WITH THE DEFECT QUOTE". The temperature was lowered toward
        # determinism FOR a repair context the code never supplied.
        draft_block = ""
        if extra_user and rejected_draft:
            draft_block = (
                "\n\nTHE DRAFT YOU MUST REPAIR (your previous attempt, "
                "verbatim). Return the corrected FULL episode. Keep every "
                "line that is already legal exactly as written -- change only "
                "what the defects below name:\n"
                "-----BEGIN REJECTED DRAFT-----\n"
                f"{rejected_draft}\n"
                "-----END REJECTED DRAFT-----"
            )
        user_content = (
            f"{base_user}{draft_block}\n\n{extra_user}"
            if extra_user else base_user
        )
        # TEMPERATURE DECAYS ONLY WHEN THE DRAFT ACTUALLY WENT WITH IT.
        # Near-deterministic sampling is right for surgical repair and wrong
        # for regeneration: a cold rung at 0.30 re-derives the same wrong
        # structure it just produced.
        #
        # HOLD THE PREVIOUS RUNG, DO NOT RESET TO THE OPENING TEMPERATURE.
        # Resetting was the first version and QA caught it: with the draft
        # dropped on attempt 2 the real sequence became 0.75, 0.49, 0.75, 0.30
        # -- a RISE, breaking `_MARKUP_LADDER_TEMPS`'s documented invariant
        # ("the markup ladder NEVER raises temperature") and putting the
        # last-chance attempt at full exploration instead of the narrowed rung
        # the ladder's whole "depth, not temperature" design calls for.
        # Holding gives 0.75, 0.49, 0.49, 0.30: still non-increasing, and still
        # away from the near-deterministic rung that a cold retry must avoid.
        if attempt > 1 and not draft_block and last_temp is not None:
            temp = max(temp, last_temp)
            cold_regenerations += 1
        last_temp = temp
        if format_example is None:
            messages = ProviderCapacityMessages([
                {"role": "system", "content": system},
                {"role": "user", "content": user_content},
            ])
        else:
            messages = ProviderCapacityMessages([
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": (
                        "Before the real assignment: show the exact output "
                        "FORMAT once, as a tiny example episode on any subject "
                        "at all. It is a FORMAT sample only -- the assignment "
                        "that follows will have its own title, its own cast and "
                        "its own subject, and you must not carry this example's "
                        "characters, setting or story into it."
                    ),
                },
                {"role": "assistant", "content": format_example},
                {"role": "user", "content": user_content},
            ])
        # LLM slot: creative -- whole-play markup authoring/repair.
        raw = creative_fn(
            messages,
            temperature=temp,
            max_new_tokens=None,
        )
        raw = _strip_conversational_wrapper(raw)
        parsed, defects = parse_fable2_markup(raw, cast_names)
        if parsed is not None:
            traces.append(PassAttemptTrace(
                attempt=attempt,
                temperature=float(temp),
                outcome="accepted",
                selected=True,
                character_words=parsed.character_word_count,
                scene_character_words=_scene_character_word_counts(parsed),
            ))
            trace_tuple = tuple(traces)
            _validate_attempt_sequence(pass_id, trace_tuple)
            return raw, parsed, {
                "defects_by_attempt": defects_by_attempt,
                "structural_retries": len(traces) - 1,
                "attempts": len(traces),
                "output_budget_mode": "provider_capacity",
                "requested_max_new_tokens": None,
                "attempt_trace": trace_tuple,
                # How many retries had to regenerate COLD because the
                # rejected draft did not fit the repair turn. Was declared
                # and incremented but never read -- and a comment claimed
                # it reached the trace, which nothing did. A budget guard
                # whose activations are invisible is a silent revert to the
                # cold-regeneration bug.
                "cold_regenerations": cold_regenerations,
            }

        defect_rows = tuple(str(defect) for defect in defects)
        rendered = render_defects(defects)
        traces.append(PassAttemptTrace(
            attempt=attempt,
            temperature=float(temp),
            outcome="parse_rejected",
            parse_defects=defect_rows,
        ))
        defects_by_attempt.append(list(defect_rows))
        last_defect_text = rendered
        # Carry this draft into the next turn -- unless it will not fit. The
        # budget check is deliberate rather than assumed: a 45-word episode's
        # draft is small, but this ladder also serves full-length episodes, and
        # silently overflowing the window would trade a diagnosable format
        # failure for a truncated prompt. When it does not fit we say so in the
        # diagnostics and fall back to regeneration WITHOUT decaying the
        # temperature.
        rejected_draft = (raw if _draft_fits_repair_turn(base_user, raw, system)
                          else "")
        extra_user = (
            "Repair only the malformed FORMAT defects below. Return the "
            "complete episode from TITLE through END as plain text. Keep "
            "the same story, cast, events, and wording wherever the format "
            "is already valid. Do not repeat a rejected malformed row."
            # The NOTE reads typed defects; `defect_rows` stays a string tuple
            # because `PassAttemptTrace.parse_defects` enforces exactly that.
            + _standalone_stage_direction_repair_note(
                defects, cast_names=cast_names)
            # The required SHAPE, not only the offence. Both notes are
            # self-silencing when their defect is absent, so a repair turn
            # carries exactly the rules its own defects call for.
            + _end_delimiter_repair_note(defects)
            + "\n" + rendered
        )

    raise Fable2ScriptError(
        pass_id,
        f"markup ladder exhausted; last defects:\n{last_defect_text}",
        len(traces),
    )


#: One valid episode, shown to the model as its own prior output.
#:
#: THE GRAMMAR WAS DESCRIBED AND NEVER DEMONSTRATED. `_run_markup_ladder` has
#: always accepted a `format_example` and built a one-shot user/assistant turn
#: from it -- and NOTHING EVER PASSED ONE, so that path was dead code and the
#: pack's own `"examples": []` was empty. The r1 review arc reached the same
#: conclusion from two directions: showing one conversion is worth more than any
#: rule sentence, and the failing legs were structural-compliance failures
#: rather than instruction-following failures.
#:
#: DELIBERATELY A DIFFERENT DOMAIN. A sci-fi example beside a sci-fi assignment
#: invites the model to lift the example's cast or premise, which would be a
#: FIDELITY defect worse than the format error it fixes. A gardening programme
#: cannot be mistaken for the assignment.
#:
#: THE SECOND SCENE IS THE WHOLE POINT. It shows the conversion the failing legs
#: kept getting wrong -- an event that a screenplay would narrate ("she crosses
#: to the window") carried instead by somebody SAYING it. Radio has no camera:
#: anything the audience must know is spoken or scored.
_FABLE2_FORMAT_EXAMPLE = """TITLE: The Frost Warning
MUSIC: a slow fiddle, up and under
ANNOUNCER: Tonight, from the allotments: The Frost Warning.
SCENE 1: a potting shed, before dawn
MAUD: You are up early for a woman who hates mornings.
PERCY: The radio said frost. I came to cover the seedlings.
MAUD: Then hand me the fleece and stop apologising to a tray of beans.
SCENE 2: outside, the beds
PERCY: Careful -- you are walking straight through the onion row.
MAUD: I am not, I am walking round it, and you are holding the torch wrong.
PERCY: There. The last row is covered.
ANNOUNCER: The frost came at four, and found nothing to take.
CODA: The beans lived. The argument continued.
MUSIC: the fiddle returns, and out
END."""


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
        format_example=_FABLE2_FORMAT_EXAMPLE,
    )


def _speakers_in_order(parsed: ParsedScript) -> "list[str]":
    seen: "list[str]" = []
    for scene in parsed.scenes:
        for ln in scene.lines:
            if ln.speaker not in seen:
                seen.append(ln.speaker)
    return seen


def _pass_casting(
    slot_fn,
    pack,
    final_draft: FinalDraft,
    treatment: Treatment,
    menu: VoiceMenu,
) -> CastingVoices:
    """Assign one legal voice-stock entry to each finished-script speaker."""
    _validate_selected_final_draft(final_draft, treatment=treatment)
    parsed = final_draft.parsed
    speakers = _speakers_in_order(parsed)
    script_view = _script_view(parsed, treatment, include_news_read=False)
    shapes = json.dumps(
        [shape.model_dump() for shape in treatment.cast_shapes],
        ensure_ascii=False,
        indent=2,
    )
    user = (
        f"THE FINISHED SCRIPT:\n{script_view}\n\n"
        f"TREATMENT CAST SHAPES:\n{shapes}\n\n"
        f"AVAILABLE VOICE STOCK (menu ids only):\n{menu.prompt_lines()}\n\n"
        f"Return exactly one entry per script speaker: {', '.join(speakers)}."
    )
    try:
        # LLM slot: technical -- voice-stock assignment only.
        return structured_call(
            prompt=ProviderCapacityMessages([
                {
                    "role": "system",
                    "content": _seam(pack, "fable2_casting_system"),
                },
                {"role": "user", "content": user},
            ]),
            schema=CastingVoices,
            slot_fn=slot_fn,
            base_temperature=_TEMP["casting_voices"],
            structural_retry_temperature=_TEMP["casting_voices"] / 2.0,
            repair_prompt_factory=make_dispatching_repair_factory(),
            post_validator=_make_casting_validator(menu, speakers),
            max_new_tokens=None,
            helper_name="fable2_casting_voices",
        )
    except StructuredCallFailedError as exc:
        raise Fable2CastError(
            "casting_voices", str(exc.last_error), exc.attempts
        ) from exc


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
        sum(canonical_word_count(line.text) for line in scene.lines)
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
        "output_budget_mode": trace.output_budget_mode,
        "requested_max_new_tokens": None,
        "outcome": trace.outcome,
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
    p3_attempts: "tuple[PassAttemptTrace, ...]",
) -> dict:
    return {
        "raw_sha256": raw_sha256,
        "normalized_sha256": normalized_sha256,
        "parsed_sha256": parsed_sha256,
        "proof_map_sha256": proof_map_sha256,
        "p3_attempts": [_attempt_payload(row) for row in p3_attempts],
    }


def _validate_attempt_sequence(
    label: str,
    rows: "tuple[PassAttemptTrace, ...]",
) -> None:
    if not rows:
        return
    if any(type(row) is not PassAttemptTrace for row in rows):
        raise Fable2ScriptError(
            "final_draft",
            f"{label} attempts must contain exact PassAttemptTrace artifacts",
        )
    if tuple(row.attempt for row in rows) != tuple(range(1, len(rows) + 1)):
        raise Fable2ScriptError(
            "final_draft", f"{label} attempts must be contiguous from 1"
        )
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
    *,
    p3_attempts: "tuple[PassAttemptTrace, ...]" = (),
) -> FinalDraft:
    """Build proof and hash seals for the sole accepted P3 artifact."""
    p3_attempts = tuple(p3_attempts)
    _validate_attempt_sequence("P3", p3_attempts)
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
    raw_hash = _sha256(raw_source)
    normalized_hash = _sha256(normalized)
    parsed_hash = _canonical_sha256(_parsed_payload(authoritative))
    proof_hash = _canonical_sha256([
        _proof_entry_payload(entry) for entry in proof_map
    ])
    artifact_hash = _canonical_sha256(_final_draft_seal_payload(
        raw_sha256=raw_hash,
        normalized_sha256=normalized_hash,
        parsed_sha256=parsed_hash,
        proof_map_sha256=proof_hash,
        p3_attempts=p3_attempts,
    ))
    return FinalDraft(
        raw_source=raw_source,
        normalized_source=normalized,
        parsed=authoritative,
        proof_map=proof_map,
        p3_attempts=p3_attempts,
        hashes=FinalDraftHashes(
            raw_sha256=raw_hash,
            normalized_sha256=normalized_hash,
            parsed_sha256=parsed_hash,
            proof_map_sha256=proof_hash,
            artifact_sha256=artifact_hash,
        ),
    )


def _validate_final_draft_integrity(label: str, draft: Any) -> None:
    """Recompute every structural/proof seal on the accepted artifact."""
    if type(draft) is not FinalDraft:
        raise Fable2ScriptError(
            "final_draft", f"{label} must be an exact FinalDraft artifact"
        )
    if type(draft.hashes) is not FinalDraftHashes:
        raise Fable2ScriptError(
            "final_draft", f"{label}.hashes must be FinalDraftHashes"
        )
    if type(draft.parsed) is not ParsedScript:
        raise Fable2ScriptError(
            "final_draft", f"{label}.parsed must be ParsedScript"
        )
    if not isinstance(draft.raw_source, str) or not isinstance(
        draft.normalized_source, str
    ):
        raise Fable2ScriptError(
            "final_draft", f"{label} source fields must be strings"
        )
    if type(draft.p3_attempts) is not tuple:
        raise Fable2ScriptError(
            "final_draft", f"{label} attempt receipts must be a tuple"
        )
    _validate_attempt_sequence(f"{label} P3", draft.p3_attempts)

    normalized = normalize_fable2_markup_text(draft.raw_source)
    if normalized != draft.normalized_source:
        raise Fable2ScriptError(
            "final_draft", f"{label} normalized source seal is stale"
        )
    authoritative, defects = parse_fable2_markup(
        draft.raw_source, _speakers_in_order(draft.parsed)
    )
    if authoritative is None:
        raise Fable2ScriptError(
            "final_draft",
            f"{label} raw source no longer parses: {render_defects(defects)}",
        )
    if _parsed_payload(authoritative) != _parsed_payload(draft.parsed):
        raise Fable2ScriptError(
            "final_draft", f"{label} parsed artifact seal is stale"
        )

    if type(draft.proof_map) is not tuple:
        raise Fable2ScriptError(
            "final_draft", f"{label}.proof_map must be an immutable tuple"
        )
    line_ids: list[str] = []
    source_norm = _norm_ws(draft.normalized_source)
    for entry in draft.proof_map:
        if type(entry) is not ProofMapEntry or type(
            entry.constituents
        ) is not tuple:
            raise Fable2ScriptError(
                "final_draft", f"{label} proof map contains a forged row"
            )
        if not isinstance(entry.line_id, str) or not entry.line_id:
            raise Fable2ScriptError(
                "final_draft", f"{label} proof line id is invalid"
            )
        line_ids.append(entry.line_id)
        for proof in entry.constituents:
            if (
                type(proof) is not ConstituentProof
                or type(proof.span) is not tuple
                or len(proof.span) != 2
                or any(type(value) is not int for value in proof.span)
                or proof.span[0] < 0
                or proof.span[1] < proof.span[0]
                or not isinstance(proof.text, str)
                or not isinstance(proof.artifact, str)
            ):
                raise Fable2ScriptError(
                    "final_draft", f"{label} contains a forged proof"
                )
            if proof.artifact == "winning_draft" and (
                proof.span[1] > len(source_norm)
                or source_norm[proof.span[0]:proof.span[1]] != proof.text
            ):
                raise Fable2ScriptError(
                    "final_draft", f"{label} proof span no longer resolves"
                )
    if len(line_ids) != len(set(line_ids)):
        raise Fable2ScriptError(
            "final_draft", f"{label} proof line ids are not unique"
        )

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
        p3_attempts=draft.p3_attempts,
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
            "final_draft", f"{label} hash seal does not match its content"
        )


def _validate_selected_final_draft(
    draft: FinalDraft,
    *,
    treatment: "Treatment | None" = None,
) -> None:
    _validate_final_draft_integrity("selected final draft", draft)
    if not draft.p3_attempts:
        raise Fable2ScriptError(
            "final_draft", "selected draft has no observed P3 attempts"
        )
    if treatment is not None:
        expected_proof = _build_proof_map(
            draft.parsed, treatment, draft.normalized_source
        )
        if draft.proof_map != expected_proof:
            raise Fable2ScriptError(
                "final_draft",
                "selected draft proof map does not match treatment/source",
            )


def _final_draft_ledger_payload(draft: FinalDraft) -> dict:
    _validate_final_draft_integrity("ledger final draft", draft)
    return {
        "raw_sha256": draft.hashes.raw_sha256,
        "normalized_sha256": draft.hashes.normalized_sha256,
        "parsed_sha256": draft.hashes.parsed_sha256,
        "proof_map_sha256": draft.hashes.proof_map_sha256,
        "artifact_sha256": draft.hashes.artifact_sha256,
        "p3_attempts": [
            _attempt_payload(row) for row in draft.p3_attempts
        ],
    }


def _assemble(
    led: Any,
    final_draft: FinalDraft,
    treatment: Treatment,
    cast_rows: "list[dict]",
    payload: "dict[str, Any]",
    *,
    owner_bank: str,
) -> None:
    """Emit ALL FIVE ledger hierarchies (r1/S2) with the proof gates.
    Incremental led.save() after the preamble and after each scene.
    Ambiguity is a structural upstream failure, never a silent fix; timing stays unset
    (SceneSequencer owns it downstream)."""
    _validate_selected_final_draft(final_draft, treatment=treatment)
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
    meta = _ledger_meta(led)

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
                    artifact_norm: str, *, speaker: str) -> dict:
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
            # PBUG-20260814-01: the spoken row names its own speaker. The
            # beat below reads it back off this row rather than being
            # handed it a second time, so the two can never disagree.
            "speaker": speaker,
            "char_id": char_id, "speaker_role": role, "boundary": boundary,
            "text": _norm_ws(text),
        }

    def _beat(line_row: dict, scene_id: "str | None") -> None:
        beat_rows.append({
            "beat_id": line_row["beat_id"], "shot_id": line_row["shot_id"],
            "scene_id": scene_id, "speaker": line_row["speaker"],
            "char_id": line_row["char_id"],
            "line_ids": [line_row["line_id"]],
        })

    # --- preamble: shot_000 (fixture-truth intro char_id "announcer") ---
    shot_rows.append({
        "shot_id": "shot_000", "scene_id": None, "description": "preamble",
    })
    opening_sentinel = _music_sentinel("shot_000", "music_open")
    line_rows.append(opening_sentinel)
    music_rows.append({
        "cue_id": "opening",
        "description": parsed.music_open,
        "generation_prompt": parsed.music_open,
        "placement": "opening",
        "anchor_line_id": opening_sentinel["line_id"],
    })
    for k, text in enumerate(parsed.announcer_intro):
        row = _spoken_row(
            f"shot_000_b{k + 1}", "shot_000", "announcer", "announcer",
            text, "shot_start" if k == 0 else "beat_start",
            [text], "winning_draft", draft_norm, speaker=ANNOUNCER_NAME)
        line_rows.append(row)
        _beat(row, None)
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
                texts, "winning_draft", draft_norm, speaker=speaker)
            line_rows.append(row)
            _beat(row, scene_id)
        for k, cue in enumerate(inter_by_scene.get(scene.n, [])):
            inter_seq += 1
            inter_sentinel = _music_sentinel(
                shot_id, "music_inter", k,
            )
            line_rows.append(inter_sentinel)
            music_rows.append({
                "cue_id": f"inter_{inter_seq:02d}",
                "description": cue,
                "generation_prompt": cue,
                "placement": "interstitial",
                "anchor_line_id": inter_sentinel["line_id"],
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
            [text], "winning_draft", draft_norm, speaker=ANNOUNCER_NAME)
        line_rows.append(row)
        _beat(row, None)
    # CODA bridge row (announcer-spoken pivot, from the draft).
    k += 1
    row = _spoken_row(
        f"{post_shot}_b{k}", post_shot, "announcer", "announcer",
        parsed.coda, "beat_start", [parsed.coda], "winning_draft",
        draft_norm, speaker=ANNOUNCER_NAME)
    line_rows.append(row)
    _beat(row, None)
    # News-read row: LLM-authored (treatment.news_close_read),
    # python-APPENDED (r1/C1). The legacy coda append lives in legacy
    # composition (writer I.5), which this lane never runs -- no double
    # append by construction (doc s14 item 7).
    k += 1
    row = _spoken_row(
        f"{post_shot}_b{k}", post_shot, "announcer", "announcer",
        treatment.news_close_read, "beat_start",
        [treatment.news_close_read], "treatment.news_close_read",
        news_read_norm, speaker=ANNOUNCER_NAME)
    line_rows.append(row)
    _beat(row, None)
    closing_sentinel = _music_sentinel(post_shot, "music_close")
    line_rows.append(closing_sentinel)
    music_rows.append({
        "cue_id": "closing",
        "description": parsed.music_close,
        "generation_prompt": parsed.music_close,
        "placement": "closing",
        "anchor_line_id": closing_sentinel["line_id"],
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
    # B1 (bake-off): provenance is the SELECTED lane id (scifi_fable2_v2/v3),
    # never the base -- the writer already stamped meta.source_bank to it, so
    # this setdefault is a no-op on the live path and keeps the direct-call
    # tests self-consistent.
    meta.setdefault("source_bank", owner_bank)
    stamp_receipt(
        led.data, owner_bank=owner_bank,
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

    # REQUIRED: the TTS delivery stamp is what the voice nodes read.
    require_ledger_save(led, "the TTS delivery-text stamp")


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


def _apply_fable_safety_cleanup(
    raw_source: str,
    parsed: ParsedScript,
    treatment: Treatment,
    *,
    technical_fn,
) -> "tuple[str, ParsedScript, Treatment, dict[str, Any]]":
    """Return the script unchanged. The content cleanup this ran is retired.

    Same-story SAFETY CLEANUP RETIRED 2026-08-05 (operator directive: no
    content guardrails on generated episodes). This built a projection of every
    spoken row -- intro, scene lines, outro, coda, news read -- scanned it,
    asked the model to rewrite any row matching the profanity / weapon / sexual
    list, and raised Fable2AuditError when a term survived: two terminal content
    failures inside the fable2 lane.

    The projection fed that scan and nothing else -- no receipt, no treatment
    mutation, no patched_news propagation read it -- so it is gone with the scan
    rather than left to walk every line and be discarded. The script is returned
    exactly as written, and the receipt below keeps its established shape so the
    caller's ledger field never loses a value.
    """
    del technical_fn  # the retired scan was this function's only model call
    return raw_source, parsed, treatment, {
        "status": "retired_no_content_policy",
        "hits": [],
        "patch_count": 0,
    }


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
    episode_root: Any,
    episode_id: str,
) -> Fable2TailParts:
    """Produce one proof-backed story with target-independent topology."""
    del episode_root, episode_id
    # `target = int(resolved["target_words"])` and its `< 1` guard were removed
    # 2026-08-14 with the word authority. The act count is validated where it
    # is used, by `_build_envelope`, against the real topology range.
    n_max = max(1, int(resolved["num_characters"]))
    creative_model = str(resolved["creative_writing_model"])
    technical_model = str(resolved["technical_model"])
    seed = _resolve_seed()
    rng = random.Random(seed)
    meta["episode_seed"] = seed
    _OTRWD.stamp_contract(meta, owner="scifi_news_pro")
    receipts: "list[dict[str, Any]]" = []

    def receipt(pass_id, model_id, attempts, temp, tokens, *, trace=()):
        row = {
            "pass_id": pass_id, "model_id": model_id,
            "attempts": attempts, "temp": temp,
            "status": "completed",
        }
        if tokens is None:
            row["output_budget_mode"] = "provider_capacity"
            row["requested_max_new_tokens"] = None
        else:
            row["max_new_tokens"] = tokens
        if trace:
            _validate_attempt_sequence(pass_id, trace)
            row["attempt_trace"] = [_attempt_payload(value) for value in trace]
        receipts.append(row)

    f2: "dict[str, Any]" = {
        "schema_version": _SCHEMA_VERSION,
        "seed": seed,
        "notes": [
            "Every target uses one fixed story-construction topology.",
            "Requested and actual words are telemetry only.",
        ],
    }
    meta["fable2"] = f2
    meta["news"] = None

    source_preview = _build_source_preview(payload)
    dossier, dropped, source_coverage, dossier_calls = (
        _extract_complete_source_dossier(
            technical_fn,
            pack,
            payload,
            slot_scheduler=slot_scheduler,
            news_seed_receipt=(
                meta.get("news_seed")
                if isinstance(meta.get("news_seed"), Mapping)
                else None
            ),
        )
    )
    receipt("dossier", technical_model, dossier_calls,
            _TEMP["dossier"], _MAX_NEW_TOKENS["dossier"])
    provenance = {
        key: str(payload.get(key) or "")
        for key in ("headline", "source", "date", "link")
    }
    f2["source_coverage"] = source_coverage
    f2["dossier"] = {
        **dossier.model_dump(), "provenance": provenance,
        "dropped_entities": dropped,
    }

    cards, stance = _deal(rng, _load_frame_deck())
    f2["cards_dealt"] = [dict(card) for card in cards]
    f2["stance"] = dict(stance)

    # PERSIST THE DEAL BEFORE ANY PASS THAT CAN DIE ON IT.
    #
    # `seed`, `cards_dealt` and `stance` were written into `meta["fable2"]` in
    # memory and the ledger was not saved again until the whole pipeline had
    # finished -- so an episode that died in the SCRIPT pass recorded none of
    # them. Three consecutive `scifi_news_pro` failures on 2026-08-12 were
    # therefore UNREPRODUCIBLE: the on-disk ledger had no `fable2` block, the
    # exception message did not carry the deal, and nothing logged it. We could
    # not even say which of the 14 frame cards or 6 stances had been dealt.
    #
    # That is the same defect class as a truncated traceback: the failure path
    # discards the evidence needed to diagnose it. It also blocks the one thing
    # the review panel asked for -- holding the deal constant while varying one
    # input -- because you cannot hold constant what a failed run never wrote
    # down.
    #
    # Logged as well as saved: a leg that dies before `obs_publish` still leaves
    # its deal in the server log, which is where a stuck run gets read first.
    log.info(
        "[scifi_fable2] deal: seed=%s frame_card=%r stance=%r",
        seed,
        cards[0].get("name") if cards else None,
        stance.get("name") if isinstance(stance, Mapping) else stance,
    )
    # CHECK THE RETURN. `Ledger.save()` returns the path on success and **None
    # on failure, and never raises** -- so an unchecked call is a receipt that
    # can silently not happen, which is precisely the defect this block exists
    # to fix. The episode is still renderable without the receipt, so this warns
    # rather than raising; killing an episode over a diagnostic would be a worse
    # trade than losing the diagnostic. The log line above already carries the
    # deal, so a failed save degrades to log-only rather than to nothing.
    if led.save() is None:
        log.warning(
            "[scifi_fable2] the deal receipt did NOT persist -- if this leg "
            "dies later, seed=%s / frame_card=%r / stance=%r are recoverable "
            "only from the line above, not from the ledger",
            seed,
            cards[0].get("name") if cards else None,
            stance.get("name") if isinstance(stance, Mapping) else stance,
        )
    fn, box = _counting(creative_fn)
    with _helper_ctx(slot_scheduler, "fable2_pitch"):
        pitch = _pass_pitch(fn, pack, dossier, cards, stance, n_max=n_max)
    receipt("pitch", creative_model, box["calls"],
            _TEMP["pitch"], None)
    f2["pitch"] = pitch.model_dump()

    fn, box = _counting(creative_fn)
    with _helper_ctx(slot_scheduler, "fable2_treatment"):
        treatment = _pass_treatment(
            fn, pack, dossier, pitch, stance, n_max=n_max,
            provenance=provenance, digest=source_preview,
        )
    receipt("treatment", creative_model, box["calls"],
            _TEMP["treatment"], None)
    cast_names = [shape.name for shape in treatment.cast_shapes]
    meta["num_characters_locked"] = len(cast_names)

    fn, box = _counting(technical_fn)
    with _helper_ctx(slot_scheduler, "fable2_news_read"):
        read = _pass_news_read(
            fn, pack, dossier, provenance, source_preview, cast_names
        )
    receipt("news_read", technical_model, box["calls"], 0.20, None)
    treatment = treatment.model_copy(
        update={"news_close_read": read.news_close_read}
    )
    envelope = _build_envelope(int(resolved["act_count"]))
    f2["episode_shape"] = {
        "act_count": int(resolved["act_count"]),
        "suggested_scenes": envelope.scene_count,
        "gating": False,
    }
    fn, box = _counting(creative_fn)
    with _helper_ctx(slot_scheduler, "fable2_script"):
        script_text, parsed, p3_meta = _pass_script(
            fn, pack, treatment, source_preview, envelope, cast_names
        )
    p3_attempts = tuple(p3_meta["attempt_trace"])
    if box["calls"] != len(p3_attempts):
        raise Fable2ScriptError("script", "P3 attempt/call count drift")
    receipt(
        "script", creative_model, box["calls"], _TEMP["script"],
        None, trace=p3_attempts,
    )

    with _helper_ctx(slot_scheduler, "fable2_same_story_safety"):
        script_text, parsed, treatment, safety = (
            _apply_fable_safety_cleanup(
                script_text, parsed, treatment,
                technical_fn=technical_fn,
            )
        )
    f2["safety_cleanup"] = safety
    f2["treatment"] = treatment.model_dump()
    final_draft = build_final_draft(
        script_text, parsed, treatment, p3_attempts=p3_attempts
    )
    _validate_selected_final_draft(final_draft, treatment=treatment)
    f2["final_draft"] = _final_draft_ledger_payload(final_draft)
    f2["parse"] = {
        "defects_by_attempt": p3_meta["defects_by_attempt"],
        "structural_retries": p3_meta["structural_retries"],
        "normalizations": list(final_draft.parsed.normalizations),
        "attempt_trace": [_attempt_payload(value) for value in p3_attempts],
    }

    menu = _deal_voice_menu(len(cast_names))
    f2["casting_stock_dealt"] = [
        {"menu_id": item.menu_id, "gender": item.gender,
         "description": item.description}
        for item in menu.entries
    ]
    fn, box = _counting(technical_fn)
    with _helper_ctx(slot_scheduler, "fable2_casting_voices"):
        casting = _pass_casting(fn, pack, final_draft, treatment, menu)
    receipt("casting", technical_model, box["calls"],
            _TEMP["casting_voices"], None)
    cast_rows = _assign_voices(
        casting, menu, rng, _speakers_in_order(final_draft.parsed)
    )
    f2["casting"] = [item.model_dump() for item in casting.cast]

    _assemble(
        led, final_draft, treatment, cast_rows, payload,
        owner_bank=source_bank_row.source_bank_id,
    )
    # `_assemble` saves incrementally and Ledger.save() rebinds `led.data`.
    # Reacquire the live mapping and preserve the complete-source receipt.
    _fable2_meta(led)["source_coverage"] = source_coverage
    _stamp_cast_contract(
        led, resolved=resolved, source_bank_row=source_bank_row,
    )
    delivery = _OTRWD.stamp_actual(
        led.data, stage="scifi_news_pro_assembled"
    )
    f2 = _fable2_meta(led)
    f2["delivery_telemetry"] = delivery
    # Terminal "safety_scan" over the assembled ledger DELETED 2026-08-05
    # (operator directive). It was the last of the fable2 lane's content
    # refusals -- a fully assembled episode thrown away for a word.
    f2["pass_receipts"] = receipts
    # REQUIRED: the completed episode's receipts and every f2 field the
    # downstream consumers key on.
    require_ledger_save(led, "the fable2 pass receipts")
    f2 = _fable2_meta(led)

    canon = _OTRC.episode_canon_from_outline_dict({
        "title": treatment.title,
        "premise": treatment.dramatic_question,
        "setting": treatment.setting,
        "time_of_day": _derive_time_of_day(treatment, final_draft.parsed),
        "sound_palette": [],
    })
    log.info(
        "[scifi_fable2] fixed story complete: seed=%d cast=%d scenes=%d "
        "character_words=%d", seed, len(cast_names),
        len(final_draft.parsed.scenes),
        final_draft.parsed.character_word_count,
    )
    return Fable2TailParts(
        outline_view=Fable2OutlineView(
            premise=treatment.dramatic_question,
            title=treatment.title,
            setting=treatment.setting,
        ),
        canon=canon,
        final_title_override=final_draft.parsed.title,
        run_story_spine=False,
        refine_active=False,
        fable2_meta=f2,
    )
