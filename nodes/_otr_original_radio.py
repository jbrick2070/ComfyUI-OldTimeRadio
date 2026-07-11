"""nodes/_otr_original_radio.py -- the original_radio creative front + QA.

The no-source original-drama lane (ARCHITECTURE_V4 2026-07-09, kibitz
r2-r4): a spark-deck entropy draw feeds a three-pass creative front
(CONCEPT -> SELECT -> BRIEF) that emits the SAME interpreter-result
contract the source lanes produce (validate_interpreter_result duck
contract; news_close_brief hardwired "" so the close routes to the
announcer outro -- the Hitchcock epilogue seam). A post-composition
whole-script QA pass (original_qa_system) judges the enumerated hard-rule
classes; the WRITER owns the one bounded repair + fail-loud.

Posture: PURE module -- stdlib + pydantic + the shared structured_call
ladder. No I/O beyond the deck file read, no GPU, no ComfyUI imports, no
writer imports. Every failure RAISES (OriginalDeckError /
OriginalBriefsError / OriginalQAError or StructuredCallFailedError from
the ladder): this lane accepts hard fails and has NO degrade surface --
the science news_briefs_required lever does not apply here.

Pack seams are read DIRECTLY from pack.prompt_stages (kibitz r3 D1): the
shared pack-prompt accessors hard-check the PRODUCTION seam allowlist
and reject this pipeline's declared original_* seams by design.

Forbidden-term defense (V4 lock 4): the lane-local lexicon below scans
the deck at load and every brief-field at the gate -- weapons/smoking,
modern-IP wording, news framing, and a small anachronism lexicon. Data
here, not Python branches: extend the lists, not the code.

UTF-8 no BOM. No em-dashes (Windows cp1252 subprocess decode trap).
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import random
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field, ValidationError, field_validator

try:
    from ._otr_json import parse_first_json_object
    from ._otr_structured_call import (
        StructuredCallFailedError,
        structured_call,
    )
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover -- flat test/standalone load
    from _otr_json import parse_first_json_object  # type: ignore
    from _otr_structured_call import (  # type: ignore
        StructuredCallFailedError,
        structured_call,
    )
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore


log = logging.getLogger("OTR")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class OriginalRadioError(Exception):
    """Base: any fail-loud original_radio lane problem."""


class OriginalDeckError(OriginalRadioError):
    """Spark deck missing/malformed/forbidden content."""


class OriginalBriefsError(OriginalRadioError):
    """The creative front could not produce a valid brief set."""


class OriginalQAError(OriginalRadioError):
    """Whole-script QA still dirty after the bounded repair."""


# ---------------------------------------------------------------------------
# Forbidden-term lexicon (lane data; scanned at deck load + brief gate)
# ---------------------------------------------------------------------------

FORBIDDEN_TERMS: tuple[str, ...] = (
    # weapons / smoking (operator hard rule)
    "gun", "pistol", "rifle", "revolver", "knife", "dagger", "blade",
    "shotgun", "bullet", "ammunition", "cigarette", "cigar", "smoking",
    "tobacco", "pipe smoke",
    # news / source framing (this lane has no source)
    "breaking news", "according to reports", "our correspondent",
    "news bulletin", "the headlines",
)

ANACHRONISM_TERMS: tuple[str, ...] = (
    # small modern-tech lexicon (V4 lock 4) -- plot must not depend on these
    "internet", "smartphone", "cell phone", "cellphone", "mobile phone",
    "computer", "laptop", "email", "e-mail", "website", "online",
    "television", "webcam", "wifi", "wi-fi", "app ", "software",
    "database", "satellite", "drone", "credit card", "social media",
)

_ALL_FORBIDDEN = FORBIDDEN_TERMS + ANACHRONISM_TERMS

# Evidence lexicons for the episode-KILLING QA classes (live-smoke
# hardening 2026-07-10: the first original_radio smoke died on a judge
# that invented 'weapons_smoking' for a weaponless threat line and
# 'news_source_framing' for a plain dramatic teaser). A hard-class
# finding may kill the episode only when (a) one of these terms
# corroborates it in the quoted detail or the script, or (b) the ONE
# bounded confirm re-judge produces a verbatim script quote for it.
# Data here, not Python branches: extend the lists, not the code.
WEAPON_SMOKING_EVIDENCE: tuple[str, ...] = (
    "gun", "pistol", "rifle", "revolver", "shotgun", "firearm",
    "derringer", "holster", "bullet", "ammunition", "weapon",
    "six-shooter", "sixgun", "luger", "musket", "carbine",
    "knife", "dagger", "blade", "bayonet", "switchblade", "stiletto",
    "machete", "saber", "sabre", "sword",
    "cigarette", "cigar", "smoking", "tobacco", "pipe smoke",
    "ashtray", "lit a pipe", "lights a pipe", "smokes a", "lit up a",
)
NEWS_FRAMING_EVIDENCE: tuple[str, ...] = (
    "news", "report", "reported", "reporter", "bulletin", "headline",
    "correspondent", "according to", "true story", "sources say",
    "dispatch", "journalist", "press conference", "this really happened",
)
MACHINE_ATTRIBUTION_EVIDENCE: tuple[str, ...] = (
    "machine", "computer", "generated", "algorithm", "artificial",
    "automaton", "automated", "synthetic", "fabricated by",
)
HARD_CLASS_EVIDENCE: "dict[str, tuple[str, ...]]" = {
    "weapons_smoking": WEAPON_SMOKING_EVIDENCE,
    "news_source_framing": NEWS_FRAMING_EVIDENCE,
    "machine_attribution": MACHINE_ATTRIBUTION_EVIDENCE,
    "anachronism_dependency": ANACHRONISM_TERMS,
}

# Kill authority per hard class (run-3 live catch 2026-07-10: the
# confirm judge "proved" weapons_smoking with the verbatim quote
# 'I don't care if the coils start to smoke' -- a grounded quote proves
# the LINE EXISTS, not that it belongs to the CLASS; the first 420w
# roll repeated the hole on news_source_framing by quoting the entire
# -- perfectly clean -- intro line). Radio is pure dialogue: a weapon,
# a tobacco act, NEWS FRAMING, or MACHINE ATTRIBUTION must be SAID in
# words to exist on air, so every class takes the lexicon as its only
# kill authority and an uncorroborated flag is discarded LOUDLY (meta
# stamp + log keep it auditable; the operator eyeball reviews discards).
# Data here, not Python branches.
#
# RIPPED 2026-07-11 (operator directive: "no more IP gates"). The pipeline
# had ONE open-vocabulary class -- brand / franchise wording -- and it was
# the only one whose kill authority was a confirm-quote rather than a
# lexicon. Unenumerable by nature, it could never be corroborated: it killed
# an episode on an unbounded hunch, with nothing to check it against and no
# repair path. It took down the original_radio bank in the all-banks 30w
# sweep. A gate with no enumerable evidence and no way to fix the finding is
# not a gate; it is a coin flip the episode can only lose. Every surviving
# class corroborates against a lexicon.
KILL_POLICY_BY_CLASS: "dict[str, str]" = {
    "weapons_smoking": "lexicon_only",
    "anachronism_dependency": "lexicon_only",
    "news_source_framing": "lexicon_only",
    "machine_attribution": "lexicon_only",
}


def scan_forbidden(text: str, *, origin: str) -> None:
    """Case-insensitive whole-ish word scan; raises OriginalRadioError
    naming the first hit. Data-driven -- extend the lexicons, not this."""
    low = f" {str(text).lower()} "
    for term in _ALL_FORBIDDEN:
        if term in low:
            raise OriginalRadioError(
                f"{origin}: forbidden term {term!r} present"
            )


# ---------------------------------------------------------------------------
# Spark deck -- load + entropy draw
# ---------------------------------------------------------------------------

DECK_AXES: tuple[str, ...] = (
    "place", "object", "occupation", "tension_verb", "texture",
    "radio_device",
)
_DECK_MIN_ATOMS_PER_AXIS = 6

_DECK_PATH = Path(__file__).resolve().parent / "story_packs" / \
    "original_radio" / "spark_deck.json"


@dataclass(frozen=True)
class SparkDraw:
    atoms: "dict[str, str]"       # axis -> drawn atom
    digest: str                    # short human seed line (payload seed_text)
    digest_long: str               # multi-line axis listing (payload full_text)
    deck_version: str
    deck_hash: str


def load_spark_deck(path: "Path | None" = None) -> dict:
    """Read + validate the deck (fail loud). Returns the parsed dict."""
    p = Path(path) if path is not None else _DECK_PATH
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise OriginalDeckError(f"spark deck unreadable at {p}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OriginalDeckError(f"spark deck malformed JSON: {exc}") from exc
    axes = data.get("axes")
    if not isinstance(axes, dict):
        raise OriginalDeckError("spark deck: 'axes' must be an object")
    for axis in DECK_AXES:
        atoms = axes.get(axis)
        if (not isinstance(atoms, list)
                or len(atoms) < _DECK_MIN_ATOMS_PER_AXIS
                or any(not isinstance(a, str) or not a.strip()
                       for a in atoms)):
            raise OriginalDeckError(
                f"spark deck axis {axis!r}: needs >= "
                f"{_DECK_MIN_ATOMS_PER_AXIS} non-empty string atoms"
            )
        for a in atoms:
            scan_forbidden(a, origin=f"spark deck axis {axis!r}")
    unknown = sorted(set(axes) - set(DECK_AXES))
    if unknown:
        raise OriginalDeckError(f"spark deck: unknown axes {unknown}")
    if not str(data.get("deck_version") or "").strip():
        raise OriginalDeckError("spark deck: deck_version required")
    data["_deck_hash"] = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return data


def draw_spark_atoms(deck: "dict | None" = None) -> SparkDraw:
    """One atom per axis. OS entropy (SystemRandom) by repo convention;
    the OTR_ORIGINAL_SEED env reproduces the DRAW ONLY (stated limit:
    model outputs downstream are not deterministic)."""
    d = deck if deck is not None else load_spark_deck()
    seed = os.environ.get("OTR_ORIGINAL_SEED", "").strip()
    rng: "random.Random" = (
        random.Random(seed) if seed else random.SystemRandom()
    )
    atoms = {axis: rng.choice(d["axes"][axis]) for axis in DECK_AXES}
    digest = (
        f"{atoms['occupation']} {atoms['tension_verb']} "
        f"{atoms['object']} at {atoms['place']}"
    )
    digest_long = "\n".join(f"{axis}: {atoms[axis]}" for axis in DECK_AXES)
    return SparkDraw(
        atoms=atoms,
        digest=digest,
        digest_long=digest_long,
        deck_version=str(d.get("deck_version")),
        deck_hash=str(d.get("_deck_hash", "")),
    )


# ---------------------------------------------------------------------------
# Creative-front schemas (loose shells; content gates are post-validators)
# ---------------------------------------------------------------------------

#: Title tokens stripped from the LEADING edge of an LLM-generated cast name.
#: Same token set fable2 uses (`_otr_scifi_fable2._HONORIFIC_TOKENS`, 19th live
#: smoke 2026-07-10); kept lane-local rather than imported so this lane does not
#: take a hard dependency on the fable2 module.
_HONORIFIC_TOKENS = frozenset({
    "DR", "DOCTOR", "MR", "MRS", "MS", "MISS", "PROF", "PROFESSOR",
    "CAPTAIN", "CAPT", "COMMANDER", "CMDR", "COLONEL", "COL",
    "LIEUTENANT", "LT", "SERGEANT", "SGT", "REVEREND", "REV", "SIR",
    "MADAM", "MAJOR", "MAJ", "GENERAL", "GEN",
})


def strip_leading_honorifics(name: str) -> str:
    """Deterministic cast-name LABEL normalization: drop LEADING honorific
    tokens ('DR. ELI CROSS' -> 'ELI CROSS').

    Local 12B-class models title their scientists/officers reflexively. Fable2
    hit the same class (roll 19: 'DR. HARRIS' broke repair convergence) and
    fixed it with a deterministic strip; original_radio's cast names ride
    through every gate literally today (`_normalize_speaker` in _otr_outline.py
    preserves INTERNAL punctuation by design, so it never removes the title).

    Shape differs from fable2 on purpose: fable2 bills a ONE-WORD surname and
    keeps only the last token, while original_radio bills a two-word period
    personal name ("MARTHA VANE", "ELI CROSS"), so we strip only the leading
    title tokens and keep the WHOLE remaining name. Never invents a character;
    only normalizes the label shape (Python judges, the LLM writes). A name
    that is nothing BUT a title is returned unchanged so this normalizer never
    manufactures an empty cast name -- the existing empty-name gates own that.
    """
    if not isinstance(name, str):
        return name
    tokens = name.strip().split()
    i = 0
    while i < len(tokens) and tokens[i].rstrip(".").upper() in _HONORIFIC_TOKENS:
        i += 1
    kept = tokens[i:]
    return " ".join(kept) if kept else name.strip()


class CastSketchEntry(BaseModel):
    name: str = Field(default="")
    role: str = Field(default="")

    @field_validator("name", mode="before")
    @classmethod
    def _strip_honorifics(cls, v):
        return strip_leading_honorifics(v) if isinstance(v, str) else v


class ConceptPitch(BaseModel):
    logline: str = Field(default="")
    hook: str = Field(default="")
    cast_sketch: list[CastSketchEntry] = Field(default_factory=list)
    radio_device: str = Field(default="")


class ConceptPitches(BaseModel):
    pitches: list[ConceptPitch] = Field(default_factory=list)


class SelectCastEntry(BaseModel):
    name: str = Field(default="")
    want: str = Field(default="")
    pressure: str = Field(default="")
    relationship: str = Field(default="")

    @field_validator("name", mode="before")
    @classmethod
    def _strip_honorifics(cls, v):
        return strip_leading_honorifics(v) if isinstance(v, str) else v


class SelectedPitch(BaseModel):
    logline: str = Field(default="")
    hook: str = Field(default="")
    radio_device: str = Field(default="")


class SelectedConcept(BaseModel):
    selected_index: int = Field(default=-1)
    pitch: SelectedPitch = Field(default_factory=SelectedPitch)
    cast: list[SelectCastEntry] = Field(default_factory=list)
    selection_rationale: str = Field(default="")


class OriginalBriefsModel(BaseModel):
    """The interpreter-result compat contract (duck-typed pin in
    _otr_source_payload.validate_interpreter_result): direct attrs
    casting_brief / script_brief / key_terms / attempts + model_dump()
    string news_close_brief -- HARDWIRED "" so the writer's nc_brief
    hoist routes the close to compose_announcer_outro (V4 lock 1)."""
    casting_brief: str = Field(default="")
    script_brief: str = Field(default="")
    key_terms: list[str] = Field(default_factory=list)
    news_close_brief: str = Field(default="")
    attempts: int = Field(default=0)


class QAFinding(BaseModel):
    finding_class: str = Field(default="", alias="class")
    detail: str = Field(default="")

    model_config = {"populate_by_name": True}


class QAFindings(BaseModel):
    findings: list[QAFinding] = Field(default_factory=list)


# The brand/franchise class is GONE (operator 2026-07-11: "no more IP gates").
# A QA class the judge can flag but nobody can enumerate, corroborate, or repair
# is not a safety net -- it is a lottery the episode can only lose.
QA_CLASSES: frozenset = frozenset({
    "weapons_smoking", "news_source_framing",
    "machine_attribution", "epilogue_missing", "epilogue_moralizes",
    "epilogue_contradicts", "anachronism_dependency",
})

# kibitz r3 D8 + r4 P4: the bounded-repair interface is DATA. Epilogue
# classes coalesce into ONE compose_announcer_outro re-run (writer-owned);
# everything else fails the episode outright.
REPAIR_ACTION_BY_CLASS: "dict[str, str]" = {
    "weapons_smoking": "fail",
    "news_source_framing": "fail",
    "machine_attribution": "fail",
    "epilogue_missing": "recompose_outro",
    "epilogue_moralizes": "recompose_outro",
    "epilogue_contradicts": "recompose_outro",
    "anachronism_dependency": "fail",
}


# ---------------------------------------------------------------------------
# Seam access (kibitz r3 D1: direct prompt_stages reads)
# ---------------------------------------------------------------------------

def _seam(pack: Any, name: str) -> str:
    stages = getattr(pack, "prompt_stages", None) or {}
    value = str(stages.get(name) or "")
    if not value.strip():
        raise OriginalBriefsError(
            f"original pack seam {name!r} missing/empty (pack "
            f"{getattr(pack, 'story_model_id', '?')!r}); the production "
            f"accessors reject declared seams by design -- author it in "
            f"the pack"
        )
    return value


# ---------------------------------------------------------------------------
# The creative front: CONCEPT -> SELECT -> BRIEF
# ---------------------------------------------------------------------------

def _concept_corpus(sel: SelectedConcept) -> str:
    parts = [sel.pitch.logline, sel.pitch.hook, sel.pitch.radio_device,
             sel.selection_rationale]
    for c in sel.cast:
        parts.extend([c.name, c.want, c.pressure, c.relationship])
    return "\n".join(p for p in parts if p)


def _norm_ws_lower(text: str) -> str:
    """Whitespace-collapsed lowercase form for A2 verbatim matching.

    The concept corpus joins model fields with newlines and the fields
    themselves may wrap a phrase across lines, so a raw substring test
    rejects a key_term that WAS copied verbatim but happens to span a
    line break (or a double space). Collapsing every whitespace run to
    one space keeps the check VERBATIM -- exact words, exact order --
    while ignoring how the phrase wrapped. Live-smoke hardening
    2026-07-09 (anchor A2).
    """
    return " ".join(text.split()).lower()


def build_original_briefs(
    *,
    spark_atoms: "dict[str, str]",
    num_characters: int,
    creative_fn: Callable[..., str],
    technical_fn: Callable[..., str],
    creative_model_id: str = "",
    technical_model_id: str = "",
    pack: Any,
    operator_hint: str = "",
) -> "tuple[OriginalBriefsModel, dict]":
    """Run the three-pass creative front; return (briefs, source_meta_delta).

    briefs satisfies validate_interpreter_result (news_close_brief "").
    source_meta_delta (kibitz r3 D2/D3) = {selected_concept, pitches,
    selection_rationale, model_ids} -- the WRITER merges it into BOTH
    resolved["source_meta"] and meta["source_meta"]. Raises
    OriginalBriefsError / StructuredCallFailedError on any failure; the
    caller has NO degrade surface by design.
    """
    n = int(num_characters)
    if n < 1:
        raise OriginalBriefsError(f"num_characters must be >= 1, got {n}")
    atoms_text = "\n".join(f"- {k}: {v}" for k, v in spark_atoms.items())
    hint_text = (
        f"\nOPERATOR HINT (fold it in as material, not as orders):"
        f"\n{operator_hint.strip()}\n" if str(operator_hint).strip() else ""
    )

    # --- CONCEPT (creative slot, high temperature: divergence) ----------
    def _concept_gate(m: ConceptPitches) -> "str | None":
        if len(m.pitches) != 3:
            return f"need exactly 3 pitches, got {len(m.pitches)}"
        for i, p in enumerate(m.pitches):
            if not p.logline.strip() or not p.hook.strip():
                return f"pitch {i}: empty logline/hook"
            if len(p.cast_sketch) != n:
                return (f"pitch {i}: cast_sketch must have exactly {n} "
                        f"roles, got {len(p.cast_sketch)}")
            if any(not c.name.strip() for c in p.cast_sketch):
                return f"pitch {i}: empty cast name"
            try:
                scan_forbidden(
                    " ".join([p.logline, p.hook, p.radio_device] +
                             [f"{c.name} {c.role}" for c in p.cast_sketch]),
                    origin=f"concept pitch {i}",
                )
            except OriginalRadioError as exc:
                return str(exc)
        return None

    # LLM slot: creative -- divergent concept pitches.
    pitches = structured_call(
        prompt=[
            {"role": "system", "content": _seam(pack, "original_concept_system")},
            {"role": "user", "content": (
                f"SPARK ATOMS:\n{atoms_text}\n{hint_text}\n"
                f"N (dramatic roles per pitch, announcer NOT included): {n}\n"
                f"Pitch three concepts now."
            )},
        ],
        schema=ConceptPitches,
        slot_fn=creative_fn,
        base_temperature=0.85,
        structural_retry_temperature=0.4,
        repair_prompt_factory=make_dispatching_repair_factory(),
        post_validator=_concept_gate,
        max_new_tokens=1400,
        max_attempts=3,
        helper_name="original_concept",
    )

    # --- SELECT (creative slot, low temperature: judgment) --------------
    def _select_gate(m: SelectedConcept) -> "str | None":
        if m.selected_index not in (0, 1, 2):
            return f"selected_index must be 0..2, got {m.selected_index}"
        if len(m.cast) != n:
            return f"cast must have exactly {n} entries, got {len(m.cast)}"
        if any(not c.name.strip() for c in m.cast):
            return "empty cast name"
        if not m.pitch.logline.strip():
            return "empty winning logline"
        if not m.selection_rationale.strip():
            return "empty selection_rationale"
        try:
            scan_forbidden(_concept_corpus(m), origin="selected concept")
        except OriginalRadioError as exc:
            return str(exc)
        return None

    # LLM slot: creative -- convergent selection.
    sel = structured_call(
        prompt=[
            {"role": "system", "content": _seam(pack, "original_select_system")},
            {"role": "user", "content": (
                f"N (dramatic roles): {n}\n\nTHE THREE PITCHES:\n"
                + json.dumps(pitches.model_dump(), ensure_ascii=False,
                             indent=2)
                + "\n\nChoose now."
            )},
        ],
        schema=SelectedConcept,
        slot_fn=creative_fn,
        base_temperature=0.3,
        structural_retry_temperature=0.2,
        repair_prompt_factory=make_dispatching_repair_factory(),
        post_validator=_select_gate,
        max_new_tokens=1000,
        max_attempts=3,
        helper_name="original_select",
    )
    corpus = _concept_corpus(sel)
    corpus_norm = _norm_ws_lower(corpus)

    # --- BRIEF (technical slot: the compat contract) ---------------------
    def _brief_gate(m: OriginalBriefsModel) -> "str | None":
        if len(m.casting_brief.strip()) < 60:
            return "casting_brief too short (>= 60 chars; QA trap 3)"
        if len(m.script_brief.strip()) < 60:
            return "script_brief too short (>= 60 chars; QA trap 3)"
        if not (2 <= len(m.key_terms) <= 6):
            return f"key_terms must be 2-6, got {len(m.key_terms)}"
        for t in m.key_terms:
            ts = str(t).strip()
            if not ts:
                return "empty key_term"
            if _norm_ws_lower(ts) not in corpus_norm:
                return (f"key_term {ts!r} not grounded in the concept "
                        f"text (anchor A2)")
        if m.news_close_brief.strip():
            return ("news_close_brief must stay EMPTY for this lane "
                    "(V4 lock 1: the close is the announcer outro)")
        try:
            scan_forbidden(
                m.casting_brief + " " + m.script_brief +
                " " + " ".join(m.key_terms),
                origin="original briefs",
            )
        except OriginalRadioError as exc:
            return str(exc)
        return None

    def _prune_ungrounded_key_terms(
        failed_output: str, error: BaseException,
    ) -> "OriginalBriefsModel | None":
        """Deterministic A2 repair: drop ungrounded key_terms, keep the rest.

        Mirrors the cast-membership Levenshtein split (see the repair-
        prompts module docstring): the dispatcher consults this BEFORE
        building the LLM repair turn. Pruning to the grounded subset is
        strictly stronger anti-drift than a paraphrase re-roll -- every
        surviving term is a verbatim concept phrase -- and it is pure
        text filtering: deterministic, seed-independent, no LLM spend.
        Declines (returns None -> typed LLM repair) unless the pruned
        instance passes the FULL brief gate, so a briefs set with any
        other latent problem keeps its LLM repair rung instead of
        burning the ladder's last attempt on a half-fix. Live-smoke
        hardening 2026-07-09: the technical model paraphrased one
        key_term twice and exhausted the ladder.
        """
        try:
            data = parse_first_json_object(failed_output or "")
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None
        raw_terms = data.get("key_terms")
        if not isinstance(raw_terms, list):
            return None
        kept = [
            str(t) for t in raw_terms
            if str(t).strip() and _norm_ws_lower(str(t)) in corpus_norm
        ]
        if len(kept) < 2:
            return None
        try:
            candidate = OriginalBriefsModel.model_validate(
                {**data, "key_terms": kept}
            )
        except ValidationError:
            return None
        if _brief_gate(candidate) is not None:
            return None
        log.info(
            "[OTR_OriginalRadio] anchor A2 deterministic repair: pruned "
            "%d ungrounded key_term(s), kept %d verbatim term(s)",
            len(raw_terms) - len(kept), len(kept),
        )
        return candidate

    # LLM slot: technical -- structured compat briefs.
    briefs = structured_call(
        prompt=[
            {"role": "system", "content": _seam(pack, "original_brief_system")},
            {"role": "user", "content": (
                "THE SELECTED CONCEPT:\n"
                + json.dumps(sel.model_dump(), ensure_ascii=False, indent=2)
                + "\n\nEmit the production briefs now."
            )},
        ],
        schema=OriginalBriefsModel,
        slot_fn=technical_fn,
        base_temperature=0.3,
        structural_retry_temperature=0.2,
        repair_prompt_factory=make_dispatching_repair_factory(
            deterministic_repair=_prune_ungrounded_key_terms,
        ),
        post_validator=_brief_gate,
        max_new_tokens=900,
        max_attempts=3,
        helper_name="original_brief",
    )
    briefs = briefs.model_copy(update={
        "news_close_brief": "",   # belt + suspenders: the lane hardwire
        "attempts": 3,            # ladder-bounded; exact count is internal
    })

    delta = {
        "selected_concept": sel.model_dump(),
        "pitches": [p.model_dump() for p in pitches.pitches],
        "selection_rationale": sel.selection_rationale,
        "model_ids": {
            "creative": str(creative_model_id),
            "technical": str(technical_model_id),
        },
        "concept_corpus": corpus,
    }
    log.info(
        "[OTR_OriginalRadio] creative front OK: %d key_terms, cast=%d, "
        "pick=%d", len(briefs.key_terms), len(sel.cast), sel.selected_index,
    )
    return briefs, delta


# ---------------------------------------------------------------------------
# Whole-script QA (the writer owns repair + fail-loud)
# ---------------------------------------------------------------------------

def script_excerpt(led: Any, cap: int = 6000) -> str:
    """Public alias of the QA excerpt builder -- the writer's QA gate
    reuses the exact text the judge saw for evidence corroboration."""
    return _script_excerpt(led, cap=cap)


def _script_excerpt(led: Any, cap: int = 6000) -> str:
    data = getattr(led, "data", led)
    rows = []
    used = 0
    for ln in (data.get("lines") or []):
        if not isinstance(ln, dict):
            continue
        role = str(ln.get("speaker_role") or "").lower()
        text = str(ln.get("text") or "").strip()
        if role not in ("character", "announcer") or not text:
            continue
        speaker = "ANNOUNCER" if role == "announcer" else str(
            ln.get("char_id") or "CHARACTER").upper()
        rows.append(f"{speaker}: {text}")
        used += len(text)
        if used >= cap:
            break
    return "\n".join(rows)


def run_original_qa(
    led: Any,
    *,
    technical_fn: Callable[..., str],
    technical_model_id: str = "",
    concept_text: str = "",
    pack: Any,
) -> QAFindings:
    """Judge the finished script against the enumerated hard-rule classes.

    Returns the (possibly empty) findings; unknown classes are rejected by
    the post-validator. The WRITER maps findings through
    REPAIR_ACTION_BY_CLASS (coalesced single outro re-compose, kibitz r4
    P4), re-judges ONCE, and raises OriginalQAError if still dirty."""
    script = _script_excerpt(led)
    if not script.strip():
        raise OriginalQAError("original QA: empty script excerpt")

    def _qa_gate(m: QAFindings) -> "str | None":
        # A finding whose class we do not recognise -- a retired class, or one the
        # judge invented -- is DROPPED, not fatal.
        # Rejecting the whole artifact over it sends the pass round the retry
        # ladder and then kills the episode: the judge naming a class we no longer
        # act on is not a defect in the SCRIPT, and it must never cost a broadcast.
        kept = []
        for f in m.findings:
            if f.finding_class not in QA_CLASSES:
                log.warning(
                    "[original_qa] discarding finding of unknown/retired class %r "
                    "(detail=%r) -- it is not a class this pipeline acts on",
                    f.finding_class, f.detail,
                )
                continue
            if not f.detail.strip():
                return f"finding {f.finding_class!r}: empty detail"
            kept.append(f)
        m.findings = kept
        return None

    # LLM slot: technical -- structured audit, not narrative.
    return structured_call(
        prompt=[
            {"role": "system", "content": _seam(pack, "original_qa_system")},
            {"role": "user", "content": (
                f"THE CONCEPT:\n{concept_text or '(none provided)'}\n\n"
                f"THE FULL SPOKEN SCRIPT:\n{script}\n\nAudit now."
            )},
        ],
        schema=QAFindings,
        slot_fn=technical_fn,
        base_temperature=0.2,
        structural_retry_temperature=0.1,
        repair_prompt_factory=make_dispatching_repair_factory(),
        post_validator=_qa_gate,
        max_new_tokens=800,
        max_attempts=3,
        helper_name="original_qa",
    )


# ---------------------------------------------------------------------------
# Hard-finding evidence triage (live-smoke hardening 2026-07-10)
# ---------------------------------------------------------------------------
#
# The first live original_radio smoke died on a single uncorroborated
# 12B judgment: THREE findings in one pass ('weapons_smoking' for a
# lever-and-sirens threat, 'news_source_framing' for a dramatic teaser,
# 'epilogue_missing' with the ironic outro plainly present). The
# episode-killing decision now demands EVIDENCE: a lexicon hit
# (deterministic, data above) or a verbatim script quote from ONE
# bounded confirm re-judge. Unproven findings are discarded LOUDLY
# (caller logs + stamps meta) -- never silently. Confirmed findings
# kill exactly as before: a dead episode is fine, a shipped bad one is
# not. Epilogue-class repair mechanics are untouched (r4 P3/P4).


def corroborate_hard_finding(finding: QAFinding, script: str) -> "str | None":
    """Deterministic evidence check for an episode-killing finding.

    Returns the evidence term found in the finding's quoted detail or
    the script excerpt (whitespace-normalized, case-insensitive), or
    None when the class has no lexicon hit -- the caller then runs the
    bounded confirm re-judge. Pure text; no LLM."""
    terms = HARD_CLASS_EVIDENCE.get(finding.finding_class) or ()
    hay = " " + _norm_ws_lower(finding.detail + " " + script) + " "
    for term in terms:
        if term in hay:
            return term
    return None


def triage_hard_findings(
    findings: "Sequence[QAFinding]",
    *,
    script: str,
    technical_fn: Callable[..., str],
) -> "tuple[list[tuple[QAFinding, str]], list[QAFinding]]":
    """Evidence triage for episode-killing findings.

    EVERY class corroborates against a LEXICON. A weapon, a tobacco act, news
    framing, machine attribution, an anachronism -- radio is pure dialogue, so
    each must be SAID in words to exist on air, and the words are enumerable.
    A flag with no lexicon hit is judge noise: it is discarded (the caller logs
    it loudly and stamps meta for the operator eyeball).

    Nothing kills an episode on an open-vocabulary judgement. The confirm-quote
    ladder was ripped with the brand/franchise class (operator 2026-07-11): a quote
    proves the LINE EXISTS, never that it belongs to the class, and it was the
    only path by which an unenumerable hunch could take a broadcast off the air.

    Returns (kills, discarded); each kill pairs the finding with a human-readable
    evidence string for the OriginalQAError / meta stamp.
    """
    kills: "list[tuple[QAFinding, str]]" = []
    discarded: "list[QAFinding]" = []
    for f in findings:
        term = corroborate_hard_finding(f, script)
        if term is not None:
            kills.append((f, f"lexicon evidence {term!r}"))
        else:
            discarded.append(f)
    return kills, discarded


__all__ = [
    "OriginalRadioError", "OriginalDeckError", "OriginalBriefsError",
    "OriginalQAError", "SparkDraw", "load_spark_deck", "draw_spark_atoms",
    "OriginalBriefsModel", "QAFindings", "QAFinding", "QA_CLASSES",
    "REPAIR_ACTION_BY_CLASS", "build_original_briefs", "run_original_qa",
    "scan_forbidden", "DECK_AXES", "FORBIDDEN_TERMS", "ANACHRONISM_TERMS",
    "HARD_CLASS_EVIDENCE", "KILL_POLICY_BY_CLASS",
    "corroborate_hard_finding",
    "triage_hard_findings", "script_excerpt",
]
