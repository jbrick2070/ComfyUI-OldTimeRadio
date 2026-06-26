"""nodes/_otr_ledger_consistency.py -- deterministic CROSS-STAGE consistency.

story-ledger DRIFT chunk 2 (2026-06-25,
docs/2026-06-25-story-ledger-integrity/). THE core drift fix: a PURE, offline,
NO-LLM parity check that asserts every non-optional UPSTREAM field
(StoryContract / Outline / locked cast) has a populated DOWNSTREAM equivalent in
the written canon / ledger.

Motivating regression: ``StoryContract.sound_world`` was a rich per-style audio
palette, but the writer DROPPED it at the canon stamp so
``episode_canon.sound_palette`` always shipped ``[]``. There was NO canon-parity
test, so the drift shipped silently for weeks (fixed in 2baba3a4; this module is
the guard that would have caught it).

Design (per the spec):
  * an explicit SOURCE-OF-TRUTH MATRIX (``field | source | canon/ledger path |
    normalizer | required?``) -- ``SOURCE_OF_TRUTH_MATRIX`` below, so a test can
    assert every non-optional model field is covered (a NEW required field with
    no parity row fails the parity test).
  * DUCK-TYPED: every argument may be a dataclass / pydantic object OR a plain
    dict OR ``None``; a missing source row is SKIPPED, never an error.
  * an ACCURACY GUARD MUST NEVER BE AN LLM (an LLM that fail-OPENs is the bug
    this kills). Pure stdlib; deterministic; ``assert_ledger_consistency`` never
    raises. The driver ``evaluate_consistency`` is the ONLY thing that may raise
    -- and only when the caller asks for strict mode -- so the live writer /
    freeze path stays audio-safe (LOUD warn + ``meta.consistency_status``).

GROUNDING CORRECTION (verify-at-build): the spec assumed ``CastLock`` is in scope
at the freeze call site. It is NOT -- ``OTR_CastLock`` is a DOWNSTREAM node that
re-locks the FROZEN ledger (it reads OTR_LedgerFreezeCascade's output). At
pre-freeze the cast source-of-truth is the locked ``ledger.cast`` rows the writer
already populated, so ``castlock`` is OPTIONAL here: pass the locked-cast view
when you have it (a test, or a future CastLock-stage call), else ``None`` and the
cast-slot row is skipped (the ledger.cast well-formedness + line-speaker rows
still run).

UTF-8 no BOM. 4-space indentation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Source-of-truth matrix (data -- the parity test introspects this).
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MatrixRow:
    """One parity rule. ``required`` rows fire a Defect when the source is
    populated but the mapped target is empty / missing / divergent."""
    field: str
    source: str
    target: str
    required: bool = True
    note: str = ""


#: The explicit matrix. Order = check order. The parity test asserts that every
#: non-optional EpisodeCanon / StoryContract field is covered here.
SOURCE_OF_TRUTH_MATRIX: tuple[MatrixRow, ...] = (
    MatrixRow("title", "outline.title", "canon.title"),
    MatrixRow("premise", "outline.premise", "canon.premise"),
    MatrixRow("setting", "outline.setting", "canon.setting"),
    MatrixRow("time_of_day", "outline.time_of_day", "canon.time_of_day"),
    MatrixRow("sound_palette", "contract.sound_world", "canon.sound_palette",
              note="the motivating regression class"),
    MatrixRow("style", "contract.slug", "ledger.meta.story_contract.slug"),
    MatrixRow("cast", "castlock.slots / ledger.cast", "ledger.cast",
              note="cast rows well-formed + locked slots have a ledger row"),
    MatrixRow("beat_id", "ledger.lines[].beat_id", "outline.beats[].beat_id"),
)

#: Fields compared for EQUALITY (not just populated). ``title`` is excluded: the
#: writer deliberately renders a TBD composition title, so canon.title may
#: legitimately diverge from outline.title.
_EQUALITY_FIELDS: frozenset = frozenset({"premise", "setting", "time_of_day"})


@dataclass(frozen=True)
class Defect:
    """One cross-stage consistency defect. ``field`` matches a matrix row."""
    field: str
    source: str
    detail: str
    expected: str = ""
    found: str = ""


class ConsistencyError(Exception):
    """Raised by ``evaluate_consistency`` in STRICT mode when defects exist.
    The live pipeline never runs strict (audio is never gated on this); CI /
    the parity test does."""


# ---------------------------------------------------------------------------
# Duck-typed accessors (object | dict | None).
# ---------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _nonempty(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return bool(v.strip())
    if isinstance(v, (list, tuple, set, dict)):
        return len(v) > 0
    return bool(v)


def _norm(v: Any) -> str:
    return str(v if v is not None else "").strip()


# ---------------------------------------------------------------------------
# The check.
# ---------------------------------------------------------------------------
def assert_ledger_consistency(
    contract: Any = None,
    outline: Any = None,
    castlock: Any = None,
    canon: Any = None,
    ledger: Any = None,
) -> list[Defect]:
    """Return the list of cross-stage consistency Defects (empty == clean).

    PURE + duck-typed + NEVER raises. Each matrix row is checked only when the
    arguments it needs are present, so a partial call (e.g. no ``castlock`` at
    the freeze call site) simply skips that row rather than failing."""
    defects: list[Defect] = []
    try:
        meta = _get(ledger, "meta", {}) or {}
        lines = _get(ledger, "lines", []) or []
        cast = _get(ledger, "cast", []) or []

        # --- outline -> canon: title / premise / setting / time_of_day ---
        if outline is not None and canon is not None:
            for fld in ("title", "premise", "setting", "time_of_day"):
                src = _get(outline, fld, "")
                tgt = _get(canon, fld, "")
                if _nonempty(src) and not _nonempty(tgt):
                    defects.append(Defect(
                        fld, f"outline.{fld}",
                        f"outline.{fld} is set but canon.{fld} is empty",
                        expected=_norm(src)[:80], found=repr(tgt)))
                elif (fld in _EQUALITY_FIELDS and _nonempty(src)
                      and _nonempty(tgt) and _norm(src) != _norm(tgt)):
                    defects.append(Defect(
                        fld, f"outline.{fld}",
                        f"canon.{fld} diverges from outline.{fld}",
                        expected=_norm(src)[:80], found=_norm(tgt)[:80]))

        # --- contract.sound_world -> canon.sound_palette (the regression) ---
        if contract is not None and canon is not None:
            sw = _get(contract, "sound_world", "")
            sp = _get(canon, "sound_palette", [])
            if _nonempty(sw) and not _nonempty(sp):
                defects.append(Defect(
                    "sound_palette", "contract.sound_world",
                    "StoryContract carries a sound_world but the canon "
                    "sound_palette is empty -- the audio world was selected "
                    "but never reached the canon/ledger",
                    expected=_norm(sw)[:80], found=repr(sp)))

        # --- contract.slug recorded on meta.story_contract ---
        if contract is not None:
            slug = _get(contract, "slug", "")
            recorded = _get(_get(meta, "story_contract", {}) or {}, "slug", "")
            if _nonempty(slug) and not _nonempty(recorded):
                defects.append(Defect(
                    "style", "contract.slug",
                    "contract slug is set but meta.story_contract.slug is not "
                    "recorded on the ledger",
                    expected=_norm(slug), found=repr(recorded)))

        # --- cast rows well-formed + locked slots resolve to a ledger row ---
        cast_ids: set[str] = set()
        for row in cast:
            cid = _get(row, "char_id", "")
            nm = _get(row, "name", "")
            if _nonempty(cid):
                cast_ids.add(_norm(cid))
            if not _nonempty(cid) or not _nonempty(nm):
                defects.append(Defect(
                    "cast", "ledger.cast",
                    "a cast row is missing char_id or name",
                    found=repr(row)[:120]))
        if castlock is not None:
            slots = (castlock if isinstance(castlock, (list, tuple))
                     else _get(castlock, "slots", []) or [])
            for slot in slots:
                sid = _get(slot, "char_id", "")
                snm = _get(slot, "name", "")
                if _nonempty(sid) and _norm(sid) not in cast_ids:
                    defects.append(Defect(
                        "cast", "castlock.slots",
                        f"locked cast slot {_norm(sid)!r} ({_norm(snm)!r}) has "
                        "no matching ledger.cast row",
                        expected=_norm(sid)))

        # --- every line.beat_id exists in the outline beats ---
        if outline is not None and lines:
            beat_ids: set[str] = set()
            for b in (_get(outline, "beats", []) or []):
                bid = _get(b, "beat_id", "")
                if _nonempty(bid):
                    beat_ids.add(_norm(bid))
            if beat_ids:
                for ln in lines:
                    bid = _get(ln, "beat_id", "")
                    if _nonempty(bid) and _norm(bid) not in beat_ids:
                        defects.append(Defect(
                            "beat_id", "ledger.lines",
                            f"line references beat_id {_norm(bid)!r} absent "
                            "from the outline beats",
                            found=_norm(bid)))
    except Exception as exc:  # noqa: BLE001 -- the guard must never raise
        # A guard that crashes is worse than no guard; record it as a defect.
        defects.append(Defect(
            "internal", "assert_ledger_consistency",
            f"consistency check raised {type(exc).__name__}: {exc}"))
    return defects


def evaluate_consistency(
    contract: Any = None,
    outline: Any = None,
    castlock: Any = None,
    canon: Any = None,
    ledger: Any = None,
    *,
    strict: bool = False,
) -> dict:
    """Run the check, stamp ``ledger.meta.consistency_status``, and return that
    status dict. In ``strict`` mode (CI / the parity gate) it RAISES
    ``ConsistencyError`` on any defect; otherwise it is audio-safe -- the live
    writer / freeze path calls it non-strict so a defect is LOUD + observable
    (``meta.consistency_status``) but NEVER breaks the render. Never silently
    passes: the status is stamped either way."""
    defects = assert_ledger_consistency(contract, outline, castlock, canon, ledger)
    status = {
        "ran": True,
        "clean": not defects,
        "defect_count": len(defects),
        "defects": [
            {"field": d.field, "source": d.source, "detail": d.detail}
            for d in defects
        ],
    }
    meta = _get(ledger, "meta", None)
    if isinstance(meta, dict):
        meta["consistency_status"] = status
    if defects and strict:
        raise ConsistencyError(
            "; ".join(f"{d.field}: {d.detail}" for d in defects)
        )
    return status
