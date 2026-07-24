"""Original-radio source construction.

A spark-deck draw feeds the fixed CONCEPT -> SELECT -> BRIEF front and emits
the interpreter-compatible source metadata consumed by the shared writer.
These passes establish the original source/cast authority before a complete
story exists. Post-script prose QA, vocabulary policy, and re-authoring are
owned nowhere in this module; the shared narrow safety cleanup and structural
freeze are the only publication gates.
"""
from __future__ import annotations

import json
import hashlib
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

try:
    from ._otr_structured_call import (
        StructuredCallFailedError,
        structured_call,
    )
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover -- flat test/standalone load
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


# ---------------------------------------------------------------------------
# Spark deck -- load + entropy draw
# ---------------------------------------------------------------------------

DECK_AXES: tuple[str, ...] = (
    "place", "object", "occupation", "tension_verb", "texture",
    "radio_device",
)
_DECK_MIN_ATOMS_PER_AXIS = 6

_DECK_PATH = Path(__file__).resolve().parent / "story_packs" / \
    "original" / "spark_deck.json"


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


class CastSketchEntry(BaseModel):
    name: str = Field(default="")
    role: str = Field(default="")



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
        if not m.pitches:
            return "need at least one pitch"
        for i, p in enumerate(m.pitches):
            if not p.logline.strip() or not p.hook.strip():
                return f"pitch {i}: empty logline/hook"
            if len(p.cast_sketch) != n:
                return (f"pitch {i}: cast_sketch must have exactly {n} "
                        f"roles, got {len(p.cast_sketch)}")
            if any(not c.name.strip() for c in p.cast_sketch):
                return f"pitch {i}: empty cast name"
        return None

    # LLM slot: creative -- divergent concept pitches.
    pitches = structured_call(
        prompt=[
            {"role": "system", "content": _seam(pack, "original_concept_system")},
            {"role": "user", "content": (
                f"SPARK ATOMS:\n{atoms_text}\n{hint_text}\n"
                f"N (dramatic roles per pitch, announcer NOT included): {n}\n"
                "Pitch episode concepts now."
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
        if not (0 <= m.selected_index < len(pitches.pitches)):
            return (
                f"selected_index must address one of {len(pitches.pitches)} "
                f"pitches, got {m.selected_index}"
            )
        if len(m.cast) != n:
            return f"cast must have exactly {n} entries, got {len(m.cast)}"
        if any(not c.name.strip() for c in m.cast):
            return "empty cast name"
        if not m.pitch.logline.strip():
            return "empty winning logline"
        return None

    # LLM slot: creative -- convergent selection.
    sel = structured_call(
        prompt=[
            {"role": "system", "content": _seam(pack, "original_select_system")},
            {"role": "user", "content": (
                f"N (dramatic roles): {n}\n\nTHE CANDIDATE PITCHES:\n"
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
        repair_prompt_factory=make_dispatching_repair_factory(),
        max_new_tokens=900,
        max_attempts=3,
        helper_name="original_brief",
    )
    raw_terms = list(briefs.key_terms)
    grounded_terms = [
        str(term).strip() for term in raw_terms
        if str(term).strip()
        and _norm_ws_lower(str(term)) in corpus_norm
    ]
    if grounded_terms != raw_terms:
        log.warning(
            "[OTR_OriginalRadio] dropped %d unsupported/empty key_term(s); "
            "source-proof terms are optional and never have a count floor",
            len(raw_terms) - len(grounded_terms),
        )
    briefs = briefs.model_copy(update={
        "key_terms": grounded_terms,
        "news_close_brief": "",   # lane hardwire; no retry for model prose
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


__all__ = [
    "OriginalRadioError", "OriginalDeckError", "OriginalBriefsError",
    "SparkDraw", "load_spark_deck", "draw_spark_atoms",
    "OriginalBriefsModel", "build_original_briefs", "DECK_AXES",
]
