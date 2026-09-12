"""The verbatim executor's PLAN step: a play's own speeches become the dialogue.

Operator ruling (docs/OTR_STANDING_RULINGS.md, "THE PASSAGE LANE"): *"For
shakespeare I'm open to a version that is very strict and finds, based on word
count and random choice, hones in on a specific part of a play to get real
specific dialogue, no paraphrasing."* And: *"`shakespeare` is VERBATIM and gets
the executor."* Before this module the lane's model wrote Shakespeare from a
beat's intent and mood -- the newest live ledger on 2026-09-11 had HERO holding
"this blade high between us" in a scene with no blade.

THE SHAPE. ``_otr_passage_selector`` chooses one contiguous window of
consecutive speeches (seeded, replayable) and cuts it into exactly the voiced
beats the operator's act dial bought (``build_beat_plan``, cap-to-fill). This
module runs that at the ONE input-resolution site, where the dials, the scene
metadata and the raw scene text are all in hand, and hands the writer a plan
plus a body-free receipt. The writer then: seats the passage's speakers as the
cast, builds the outline from the plan's speaker order, writes each chunk into
its beat instead of composing a line, and protects those rows from every later
rewriter (``VERBATIM_SOURCE_FLAG``).

THREE SIZES, ONE OWNER EACH. The OPERATOR REQUEST (``num_characters`` as
asked) is recorded on the receipt and never reassigned. The EXECUTABLE size is
``len(passage.speakers)``: it is what ``lock_cast`` is handed, hence what it
stamps as ``num_characters_request`` and what voice replay reads -- consistent
with the cast that actually exists. The LOCKED size is what the writer's count
guard reads (``num_characters_effective``), unchanged.

THE TOPOLOGY IS NOT DERIVED FROM THE PASSAGE. ``compute_episode_budget`` stays
the one topology owner and the act dial is honoured exactly; the passage FILLS
the beats. The alternative -- deriving an act count from the words selected --
is the word-to-act veto deleted on 2026-08-14 wearing new clothes.

THE SEED. Same helper and same ``OTR_CAST_SEED`` pin as the cast draw, but a
SEPARATE draw taken here, on this lane only: the cast mint at the writer's own
site is untouched and no other bank draws entropy it did not draw before.

A MISS NEVER KILLS. No vendored scene misses at any legal dial (measured:
14 scenes x cast 1-10 x acts 1-6, zero failures). A source that does --
a custom play, a snapshot cut from a file that has since changed, a
custom-premise run on the bank -- gets today's LLM path plus a receipt that
says so, and the printed credits say "freely adapted from" (a phrase borrowed
from the prose lane's ruling for the non-verbatim variant).

Nothing here calls a model, touches the ledger, or reads the network.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Mapping

try:
    from . import _otr_passage_selector as _PS
except ImportError:  # pragma: no cover -- flat import harnesses
    import _otr_passage_selector as _PS  # type: ignore

try:
    from ._otr_episode_budget import BEAT_WORD_HARD_MAX, voiced_beat_count
except ImportError:  # pragma: no cover -- flat import harnesses
    from _otr_episode_budget import BEAT_WORD_HARD_MAX, voiced_beat_count  # type: ignore

log = logging.getLogger("OTR")

RECEIPT_VERSION = "otr_verbatim_passage_v1"
STATUS_PLANNED = "planned"
STATUS_UNAVAILABLE = "unavailable"

# The bank-default gate. A typed bool validated by `_otr_story_routing`; never a
# bank-id compare, and never `style_pool_class` (public_domain shares that and
# is ruled fuzzy prose).
GATE_KEY = "verbatim_passage"

# Printed on the credits when the lane could not perform the source verbatim,
# so a non-verbatim publication is visibly one. The phrase is the standing
# ruling's own wording for the prose lane's paraphrased variant, borrowed here.
NON_VERBATIM_CREDIT_PREFIX = "freely adapted from"


def non_verbatim_credit_line(existing: str) -> str:
    """The printed credit for a verbatim bank that could not perform its
    source verbatim. ``printed_credit_line`` writes "adapted from <work>
    (<terms>)"; this makes it "freely adapted from ...". Never raises."""
    line = str(existing or "").strip()
    if line.lower().startswith("adapted from"):
        return f"freely {line}"
    if line:
        return f"{NON_VERBATIM_CREDIT_PREFIX} the source; {line}"
    return f"{NON_VERBATIM_CREDIT_PREFIX} the source"


def bank_is_verbatim(bank_row: Any) -> bool:
    """Does this bank perform its source text verbatim? Reads only the typed
    default; a missing key means no, byte-identically to every other bank."""
    defaults = getattr(bank_row, "defaults", None) or {}
    return bool(defaults.get(GATE_KEY)) if isinstance(defaults, Mapping) else False


@dataclass(frozen=True)
class VerbatimPlan:
    """What the writer executes. ``entries`` are the ledger's future character
    rows in outline order; ``receipt`` is durable and body-free."""

    entries: tuple
    speakers: tuple[str, ...]
    passage_text: str
    seed: str
    receipt: dict

    @property
    def beat_count(self) -> int:
        return len(self.entries)


def speaking_cast_seats() -> int:
    """The largest cast the voice stock seats -- `lock_cast`'s own clamp value."""
    try:
        from . import _otr_casting as _OTRCAST
    except ImportError:  # pragma: no cover
        import _otr_casting as _OTRCAST  # type: ignore
    return int(_OTRCAST._LEGACY_MAX_SPEAKING_CAST)


def draw_passage_seed(source_ref: str) -> tuple[str, str]:
    """A replay-stable seed for the window draw: the cast-seed helper (fresh OS
    entropy per episode, pinned by OTR_CAST_SEED for C7 runs), called here a
    second time so the writer's own cast mint is untouched."""
    try:
        from ._otr_writer_tail import _resolve_cast_rng_seed
    except ImportError:  # pragma: no cover
        from _otr_writer_tail import _resolve_cast_rng_seed  # type: ignore
    seed, source = _resolve_cast_rng_seed()
    return f"{int(seed)}|{source_ref}", str(source)


def _raw_sha256(text: str) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def _base_receipt(*, source_ref: str, num_characters: int, act_count: int) -> dict:
    return {
        "receipt_version": RECEIPT_VERSION,
        "source_ref": str(source_ref or ""),
        "num_characters_operator_request": int(num_characters),
        "act_count": int(act_count),
    }


def unavailable_receipt(reason: str, **base: Any) -> dict:
    out = _base_receipt(**base)
    out["status"] = STATUS_UNAVAILABLE
    out["reason"] = str(reason)
    return out


def plan_verbatim_passage(
    *,
    source_text: str | None,
    source_meta: Mapping[str, Any],
    num_characters: int,
    act_count: int,
    source_ref: str,
    unavailable_reason: str = "",
) -> tuple[VerbatimPlan | None, dict]:
    """Select and cut the passage. Returns ``(plan, receipt)``; a miss returns
    ``(None, receipt)`` with ``status: unavailable`` and never raises for a
    source-shaped reason."""
    base = dict(source_ref=source_ref, num_characters=num_characters,
                act_count=act_count)
    if unavailable_reason:
        return None, unavailable_receipt(unavailable_reason, **base)
    if not str(source_text or "").strip():
        return None, unavailable_receipt(
            "no line-structured source text reached the plan step (a custom "
            "premise, a user story, or a fetcher that carries none)", **base)
    try:
        target_words = int((source_meta or {}).get("recommended_word_budget") or 0)
    except (TypeError, ValueError):
        target_words = 0
    if target_words <= 0:
        return None, unavailable_receipt(
            "scene metadata carries no recommended_word_budget", **base)

    cast_ceiling = min(max(1, int(num_characters)), speaking_cast_seats())
    min_speakers = min(2, cast_ceiling)
    max_beats = voiced_beat_count(int(act_count))
    seed, seed_source = draw_passage_seed(source_ref)
    try:
        passage = _PS.select_passage(
            str(source_text),
            target_words=target_words,
            cast_ceiling=cast_ceiling,
            max_beats=max_beats,
            seed=seed,
            min_speakers=min_speakers,
        )
        entries = _PS.build_beat_plan(passage, beat_count=max_beats)
    except _PS.PassageError as exc:
        return None, unavailable_receipt(str(exc), **base)

    receipt = _base_receipt(**base)
    receipt.update({
        "status": STATUS_PLANNED,
        "raw_sha256": _raw_sha256(str(source_text)),
        "selector_version": _PS.SELECTOR_VERSION
        if hasattr(_PS, "SELECTOR_VERSION") else "",
        "chunker_version": _PS.CHUNKER_VERSION,
        "seed": seed,
        "seed_source": seed_source,
        "target_words": target_words,
        "cast_ceiling": cast_ceiling,
        "min_speakers": min_speakers,
        "beat_word_cap": int(BEAT_WORD_HARD_MAX),
        "first_index": passage.first_index,
        "last_index": passage.last_index,
        "speech_count": passage.speech_count,
        "word_count": passage.word_count,
        "beat_cost": passage.beat_cost,
        "beat_count": len(entries),
        "eligible_count": passage.eligible_count,
        "speakers": list(passage.speakers),
        "beats": [
            {
                "speech_index": e.speech_index,
                "chunk_ordinal": e.chunk_ordinal,
                "chunk_count": e.chunk_count,
                "speaker": e.speaker,
                "words": len(e.text.split()),
            }
            for e in entries
        ],
    })
    plan = VerbatimPlan(
        entries=tuple(entries),
        speakers=tuple(passage.speakers),
        passage_text=_PS.render_passage_text(passage),
        seed=seed,
        receipt=receipt,
    )
    return plan, receipt


def project_payload(news_article: Mapping[str, Any], plan: VerbatimPlan) -> dict:
    """The interpreter, the outline's macro pass and the cast seed all read the
    lane's payload. With a plan in hand the passage IS the material, so
    ``full_text`` and the seed's excerpt become the passage; the headline, the
    scene synopsis and the rights stay the scene's, so the announcer still
    names the work and the scene. A projection of the material, not a change of
    source identity -- the receipt's hashes say which is which."""
    out = dict(news_article)
    out["full_text"] = plan.passage_text
    seed_text = str(out.get("seed_text") or "")
    for marker in ("\nSpeakers:", "\nExcerpt:"):
        if marker in seed_text:
            seed_text = seed_text.split(marker, 1)[0]
            break
    out["seed_text"] = (
        f"{seed_text}\n"
        f"Speakers: {', '.join(plan.speakers)}\n"
        f"Excerpt (a verbatim passage, performed as written):\n"
        f"{plan.passage_text}"
    )
    return out


def raw_text_for_snapshot(snapshot: Any, *, source_ref: str) -> tuple[str | None, str]:
    """A source-snapshot replay carries the payload, never the raw file. Re-read
    the scene by its pinned reference through the BASE bank's own fetcher and
    prove the bytes are the ones the snapshot was cut from. Returns
    ``(raw_text, "")`` or ``(None, reason)``; never raises for a source-shaped
    reason."""
    meta = getattr(snapshot, "source_meta", None) or {}
    want = str(meta.get("raw_sha256") or "").strip()
    if not want:
        return None, ("the snapshot carries no raw_sha256 (captured before the "
                      "verbatim executor); the file behind its reference cannot "
                      "be proven to be the same bytes")
    base = str(getattr(snapshot, "base_source_bank_id", "") or "")
    try:
        try:
            from . import _otr_story_routing as _R
            from . import _otr_source_payload as _P
        except ImportError:  # pragma: no cover
            import _otr_story_routing as _R  # type: ignore
            import _otr_source_payload as _P  # type: ignore
        bank = _R.get_bank(base)
        entry = _P.resolve_fetcher(bank, owner=_R.user_bank_bundle(bank.source_bank_id))
        result = entry.fetch(bank=bank, technical_model="", source_ref=str(source_ref or ""))
    except Exception as exc:  # noqa: BLE001 -- a miss is receipted, never fatal
        return None, f"re-reading {source_ref!r} through bank {base!r} failed: {exc}"
    raw = getattr(result, "source_text", None)
    if not str(raw or "").strip():
        return None, f"bank {base!r} carries no raw source text"
    got = _raw_sha256(str(raw))
    if got != want:
        return None, (f"the scene file behind {source_ref!r} ({got[:12]}) is not the "
                      f"one the snapshot was cut from ({want[:12]})")
    return str(raw), ""


__all__ = [
    "GATE_KEY",
    "NON_VERBATIM_CREDIT_PREFIX",
    "RECEIPT_VERSION",
    "STATUS_PLANNED",
    "STATUS_UNAVAILABLE",
    "VerbatimPlan",
    "bank_is_verbatim",
    "draw_passage_seed",
    "non_verbatim_credit_line",
    "plan_verbatim_passage",
    "project_payload",
    "raw_text_for_snapshot",
    "speaking_cast_seats",
    "unavailable_receipt",
]
