"""The shared writer tail's ledger completion pass (independent source banks, wave 6).

WHY THIS EXISTS
---------------
Everything downstream of the writer -- TTS, per-beat slicing, shot direction,
captions, the credits roll, assembly and ``obs_publish`` -- reads LEDGER
FIELDS, not intentions. A client-authored source bank runs through the trusted
shared writer precisely so it can never touch the ledger itself; the price of
that safety is that the WRITER must hand every downstream consumer a complete
ledger even when the client's source material was thin.

This module is that completion pass. It runs once, at the one shared producer
boundary in ``OTR_LedgerScriptWriter._run_writer_tail`` -- after every
writer-side text mutation, BEFORE the TTS delivery stamp and before the freeze
cascade -- and it does three things in order:

  1. DETERMINISTIC COMPLETION. Fill every required field that has a derivable
     owner: mint a missing ``line_id``, resolve a blank ``speaker_role`` from a
     cast ``char_id``, turn an unspeakable row into an EXPLICIT skip carrying
     its reason, re-stamp text metrics. No prose is authored here.
  2. (RETIRED 2026-08-05.) This step used to be SAFETY REPAIR IN PLACE --
     rewriting a delivered spoken row whose words matched a profanity / weapon /
     sexual list. It is GONE by operator directive: no content guardrails on
     generated episodes, and on an adaptation lane the author's own language is
     carried as written. The terminal G9 freeze gate it fed was deleted in the
     same change. The receipt key ``safety`` survives with a retired status so
     no consumer of ``meta.ledger_cleanup`` loses a field.
  3. PROSE COMPLETION. A required prose field that survives step 1 with no
     value (today: ``meta.episode_title``) gets one bounded same-story LLM
     fill, then a deterministic source-grounded backstop.

Only after all three does it look for holes. A required field with NO owner and
NO value is a STRUCTURALLY incomplete ledger and raises ``LedgerIncompleteError``
naming every offending field -- the one hard failure this pass owns.

THE LAW (operator, 2026-07-22) governs the split: length, language, style,
visual vocabulary, craft and quality may never fail a story. Structure may.
Content is repaired; structure is named.

SCOPE -- what this pass does NOT own
------------------------------------
Fields stamped by nodes DOWNSTREAM of the writer are deliberately absent from
the required set: ``meta.render_engines`` / ``image_engines`` / ``music_engine``
(render nodes), ``meta.style`` (the style authority), per-line ``start_s`` /
``dur_s`` and the clips manifest (the timeline). The writer cannot fill what it
does not produce, and claiming ownership of another producer's field would turn
a normal mid-pipeline state into a false hard failure here.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Mapping, MutableMapping

log = logging.getLogger("OTR.ledger_cleanup")


__all__ = [
    "LEDGER_CLEANUP_VERSION",
    "LedgerIncompleteError",
    "required_field_gaps",
    "run_ledger_cleanup",
]


LEDGER_CLEANUP_VERSION = "ledger_cleanup_v1"

#: Roles whose rows are actually delivered to a voice engine.
_SPOKEN_ROLES = frozenset({"character", "announcer"})

#: Stamped on rows this pass converted into an explicit skip.
_EMPTY_TEXT_SKIP_REASON = "empty_spoken_text_at_ledger_cleanup"
#: Stamped when a row arrived already skipped but carried no reason.
_UNRECORDED_SKIP_REASON = "reason_unrecorded_at_ledger_cleanup"


class LedgerIncompleteError(RuntimeError):
    """A required ledger field has no owner and no value after cleanup.

    Carries the full field list so the operator sees every hole at once
    instead of fixing them one render at a time.
    """

    def __init__(self, missing: "list[str]", *, bank_id: str = "") -> None:
        self.missing = list(missing)
        self.bank_id = str(bank_id or "")
        where = f" for source bank {self.bank_id!r}" if self.bank_id else ""
        super().__init__(
            f"ledger is structurally incomplete{where} after cleanup: "
            f"{len(self.missing)} required field(s) have no owner and no "
            f"value -- " + "; ".join(self.missing[:20])
            + (f" (+{len(self.missing) - 20} more)"
               if len(self.missing) > 20 else "")
        )


# ---------------------------------------------------------------------------
# small shared helpers
# ---------------------------------------------------------------------------


def _rows(ledger_data: Mapping[str, Any], key: str) -> "list":
    value = ledger_data.get(key)
    return value if isinstance(value, list) else []


def _text_of(row: Mapping[str, Any]) -> str:
    value = row.get("text")
    return value if isinstance(value, str) else ""


def _clean_spoken_text(text: str) -> str:
    """The shared stage-direction stripper, imported function-locally."""
    try:
        from ._otr_script_prep import clean_spoken_text
    except ImportError:  # pragma: no cover -- flat test/standalone load
        from _otr_script_prep import clean_spoken_text  # type: ignore
    return clean_spoken_text(text)


def _cast_ids(ledger_data: Mapping[str, Any]) -> "set[str]":
    ids: "set[str]" = set()
    for row in _rows(ledger_data, "cast"):
        if isinstance(row, Mapping):
            char_id = row.get("char_id")
            if isinstance(char_id, str) and char_id.strip():
                ids.add(char_id.strip())
    return ids


# ---------------------------------------------------------------------------
# step 1 -- deterministic completion
# ---------------------------------------------------------------------------


def _mint_line_id(index: int, taken: "set[str]") -> str:
    """A stable, collision-free id for a row that arrived without one."""
    candidate = f"lc{index:04d}"
    suffix = 0
    while candidate in taken:
        suffix += 1
        candidate = f"lc{index:04d}_{suffix}"
    taken.add(candidate)
    return candidate


def _complete_deterministic(
    ledger_data: MutableMapping[str, Any],
) -> "list[dict]":
    """Fill every required field that has a derivable owner. Authors nothing."""
    try:
        from ._otr_text_metrics import set_line_text_metrics
    except ImportError:  # pragma: no cover -- flat test/standalone load
        from _otr_text_metrics import set_line_text_metrics  # type: ignore

    actions: "list[dict]" = []
    cast_ids = _cast_ids(ledger_data)

    # --- cast rows: a row with no id can never be referenced -------------
    taken_char_ids = set(cast_ids)
    for index, row in enumerate(_rows(ledger_data, "cast")):
        if not isinstance(row, MutableMapping):
            continue
        char_id = row.get("char_id")
        if not isinstance(char_id, str) or not char_id.strip():
            minted = f"cc{index:02d}"
            bump = 0
            while minted in taken_char_ids:
                bump += 1
                minted = f"cc{index:02d}_{bump}"
            taken_char_ids.add(minted)
            row["char_id"] = minted
            cast_ids.add(minted)
            actions.append({
                "field": f"cast[{index}].char_id",
                "action": "minted",
                "value": minted,
            })

    # --- line rows -------------------------------------------------------
    taken_line_ids: "set[str]" = set()
    for row in _rows(ledger_data, "lines"):
        if isinstance(row, Mapping):
            line_id = row.get("line_id")
            if isinstance(line_id, str) and line_id:
                taken_line_ids.add(line_id)

    seen_line_ids: "set[str]" = set()
    for index, row in enumerate(_rows(ledger_data, "lines")):
        if not isinstance(row, MutableMapping):
            continue

        # line_id: the key every audio clip matches back to its script row.
        line_id = row.get("line_id")
        if not isinstance(line_id, str) or not line_id:
            minted = _mint_line_id(index, taken_line_ids)
            row["line_id"] = minted
            actions.append({
                "field": f"lines[{index}].line_id",
                "action": "minted",
                "value": minted,
            })
        elif line_id in seen_line_ids:
            minted = _mint_line_id(index, taken_line_ids)
            row["line_id"] = minted
            actions.append({
                "field": f"lines[{index}].line_id",
                "action": "deduplicated",
                "value": minted,
                "was": line_id,
            })
        seen_line_ids.add(str(row.get("line_id") or ""))
        line_id = str(row.get("line_id") or "")

        # speaker_role: derivable ONLY from a char_id that is a real cast id.
        # Anything else would be a guess about who is talking.
        role = row.get("speaker_role")
        if not isinstance(role, str) or not role.strip():
            char_id = row.get("char_id")
            if isinstance(char_id, str) and char_id.strip() in cast_ids:
                row["speaker_role"] = "character"
                actions.append({
                    "field": f"line_id={line_id}.speaker_role",
                    "action": "resolved_from_cast_char_id",
                    "value": "character",
                })
            # else: left blank on purpose -> reported as a gap below.
        role = str(row.get("speaker_role") or "")

        # tts_skip_reason must be a string when present, never null.
        if "tts_skip_reason" in row and row.get("tts_skip_reason") is None:
            row["tts_skip_reason"] = ""
            actions.append({
                "field": f"line_id={line_id}.tts_skip_reason",
                "action": "nulled_to_empty_string",
            })

        text = _text_of(row)
        skipped = bool(row.get("skip"))

        if skipped and text.strip():
            # Contradictory state. Keep the STORY: a row that has words is a
            # row somebody wrote. Drop the skip rather than drop the text.
            row["skip"] = False
            row["tts_skip_reason"] = ""
            skipped = False
            actions.append({
                "field": f"line_id={line_id}.skip",
                "action": "cleared_skip_on_row_with_text",
            })

        if (not skipped and role in _SPOKEN_ROLES
                and not _clean_spoken_text(text).strip()):
            # A voiced row with nothing sayable cannot be voiced. Making the
            # skip EXPLICIT (with its reason) is completion; leaving it as a
            # silent hole is what breaks slicing and captions downstream.
            row["skip"] = True
            # `set_line_text_metrics` is the canonical atomic owner of text +
            # its counts; assigning row["text"] directly would leave the two
            # able to disagree (tests/test_text_metric_ownership.py forbids it).
            set_line_text_metrics(row, "")
            row["tts_skip_reason"] = _EMPTY_TEXT_SKIP_REASON
            skipped = True
            text = ""
            actions.append({
                "field": f"line_id={line_id}.skip",
                "action": "marked_explicit_skip_no_spoken_surface",
            })

        if skipped:
            reason = row.get("tts_skip_reason")
            if not isinstance(reason, str) or not reason.strip():
                row["tts_skip_reason"] = _UNRECORDED_SKIP_REASON
                actions.append({
                    "field": f"line_id={line_id}.tts_skip_reason",
                    "action": "filled_missing_skip_reason",
                })

        # Counts are telemetry, never a gate -- but a stale count is a lie,
        # and re-deriving one from the text costs nothing.
        set_line_text_metrics(row, _text_of(row))

    return actions


# ---------------------------------------------------------------------------
# step 2 -- safety repair in place (NEVER a story-fail)
# ---------------------------------------------------------------------------


# _repair_safety() was DELETED 2026-08-05 (operator directive). Its receipt
# key survives with a retired status -- see run_ledger_cleanup.

# ---------------------------------------------------------------------------
# step 3 -- prose completion (bounded; deterministic backstop)
# ---------------------------------------------------------------------------


def _source_title_backstop(ledger_data: Mapping[str, Any]) -> str:
    """A truthful episode title derived from what the ledger already knows."""
    meta = ledger_data.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    for path in (
        ("news", "headline"),
        ("source_meta", "source_label"),
        ("story_contract", "logline"),
    ):
        node: Any = meta
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        if isinstance(node, str) and node.strip():
            return " ".join(node.split())[:120]
    return ""


def _complete_prose(
    ledger_data: MutableMapping[str, Any],
    slot_fn: "Callable[..., str] | None",
) -> "list[dict]":
    """Fill required PROSE fields that survived deterministic completion.

    Today that is exactly one field: `meta.episode_title`. The freeze gate
    only WARNS on a missing title, but `otr_credits_roll` is NO-FALLBACK and
    raises on it -- so a hole here surfaces as a crash minutes later, in a
    node that has no idea which bank caused it. One bounded LLM fill, then a
    deterministic source-grounded backstop, then it is a named gap.
    """
    actions: "list[dict]" = []
    meta = ledger_data.setdefault("meta", {})
    if not isinstance(meta, MutableMapping):
        return actions

    title = meta.get("episode_title")
    if isinstance(title, str) and title.strip():
        return actions

    if slot_fn is not None:
        filled = _llm_episode_title(ledger_data, slot_fn)
        if filled:
            meta["episode_title"] = filled
            actions.append({
                "field": "meta.episode_title",
                "action": "llm_filled",
                "value": filled,
            })
            return actions

    backstop = _source_title_backstop(ledger_data)
    if backstop:
        meta["episode_title"] = backstop
        actions.append({
            "field": "meta.episode_title",
            "action": "derived_from_source",
            "value": backstop,
        })
    return actions


def _llm_episode_title(
    ledger_data: Mapping[str, Any],
    slot_fn: "Callable[..., str]",
) -> str:
    """One bounded same-story title call. Returns "" on any failure."""
    try:
        from pydantic import BaseModel, Field

        from ._otr_structured_call import structured_call
    except ImportError:  # pragma: no cover -- flat test/standalone load
        try:
            from pydantic import BaseModel, Field
            from _otr_structured_call import structured_call  # type: ignore
        except Exception:
            return ""

    class _EpisodeTitle(BaseModel):
        episode_title: str = Field(min_length=1, max_length=120)

    spoken = [
        _text_of(row)
        for row in _rows(ledger_data, "lines")
        if isinstance(row, Mapping)
        and not bool(row.get("skip"))
        and _text_of(row).strip()
    ][:24]
    if not spoken:
        return ""

    prompt = [
        {
            "role": "system",
            "content": (
                "You title an already-written radio episode. Return JSON "
                "only. The title must describe THIS script -- do not invent "
                "events, characters, or a sequel hook, and do not add "
                "profanity, explicit weapon language, or explicit sexual or "
                "nudity language."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return {\"episode_title\": \"...\"} -- a single plain-text "
                "title of at most twelve words for this script:\n\n"
                + "\n".join(spoken)
            ),
        },
    ]

    def _validate(result: "_EpisodeTitle") -> "str | None":
        # The unsafe-language rejection here was DELETED 2026-08-05 (operator
        # directive). A title drawn from the story's own words is a good title;
        # re-rolling it for containing one of them is the same fidelity defect
        # as the lane-level clauses. Emptiness is STRUCTURAL and still refuses.
        candidate = " ".join(str(result.episode_title or "").split())
        if not candidate:
            return "episode_title is empty after whitespace normalization"
        return None

    try:
        # LLM slot: technical -- naming an artifact that already exists.
        result = structured_call(
            prompt=prompt,
            schema=_EpisodeTitle,
            slot_fn=slot_fn,
            base_temperature=0.4,
            structural_retry_temperature=0.1,
            post_validator=_validate,
            max_new_tokens=128,
            max_attempts=3,
            helper_name="ledger_cleanup_episode_title",
        )
    except Exception as exc:  # noqa: BLE001 -- a backstop follows; never fatal
        log.warning(
            "[ledger_cleanup] episode-title fill failed (%s: %s); falling "
            "back to the source-derived title",
            type(exc).__name__, str(exc)[:200],
        )
        return ""
    return " ".join(str(result.episode_title or "").split())[:120]


# ---------------------------------------------------------------------------
# step 4 -- the one hard failure: structural incompleteness
# ---------------------------------------------------------------------------


def required_field_gaps(ledger_data: Mapping[str, Any]) -> "list[str]":
    """Required WRITER-OWNED fields that still have no owner and no value.

    Deliberately narrower than the freeze gate's error set: this reports only
    fields the shared writer itself produces. A field another producer stamps
    later (render engines, the style slug, the timeline, the clips manifest)
    is legitimately absent here and is NOT a gap -- see the module docstring.

    Returns human-readable field paths, most useful first, so the raised
    error tells the operator which row to look at.
    """
    gaps: "list[str]" = []
    if not isinstance(ledger_data, Mapping):
        return ["<ledger>: not a mapping"]

    for index, row in enumerate(_rows(ledger_data, "cast")):
        if not isinstance(row, Mapping):
            gaps.append(f"cast[{index}]: not a mapping")
            continue
        char_id = row.get("char_id")
        if not isinstance(char_id, str) or not char_id.strip():
            gaps.append(f"cast[{index}].char_id: empty")
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            gaps.append(
                f"cast[{index}] (char_id={char_id!r}).name: empty -- captions "
                "and the credits roll label speakers from this field"
            )

    seen: "set[str]" = set()
    for index, row in enumerate(_rows(ledger_data, "lines")):
        if not isinstance(row, Mapping):
            gaps.append(f"lines[{index}]: not a mapping")
            continue
        line_id = row.get("line_id")
        if not isinstance(line_id, str) or not line_id.strip():
            gaps.append(f"lines[{index}].line_id: empty")
        elif line_id in seen:
            gaps.append(f"lines[{index}].line_id: {line_id!r} is duplicated")
        else:
            seen.add(line_id)

        role = row.get("speaker_role")
        if not isinstance(role, str) or not role.strip():
            gaps.append(
                f"lines[{index}] (line_id={line_id!r}).speaker_role: empty -- "
                "shot direction maps this to a video role with no fallback"
            )
        elif role not in _ALLOWED_SPEAKER_ROLES():
            gaps.append(
                f"lines[{index}] (line_id={line_id!r}).speaker_role: "
                f"{role!r} is not an allowed role"
            )
        elif role in _SPOKEN_ROLES:
            char_id = row.get("char_id")
            if not isinstance(char_id, str) or not char_id.strip():
                gaps.append(
                    f"line_id={line_id!r}.char_id: empty on a voiced row"
                )
            # A voiced char_id deliberately does NOT have to name a cast row.
            # The ANNOUNCER is the standing counter-example: it speaks on
            # nearly every episode, carries char_id="announcer", lives in the
            # Kokoro voice namespace rather than the cast's Bark one, and
            # legitimately has no cast[] entry -- which is why the freeze
            # gate's own per-line invariant requires a non-empty char_id and
            # stops there. Requiring cast backing here would have invented a
            # structural failure the authority does not recognize. An
            # unlabeled caption is a quality cost, not a hole in the ledger,
            # and THE LAW keeps quality out of the terminal set.

        if bool(row.get("skip")):
            reason = row.get("tts_skip_reason")
            if not isinstance(reason, str) or not reason.strip():
                gaps.append(
                    f"line_id={line_id!r}.tts_skip_reason: empty on a skip row"
                )

    meta = ledger_data.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    title = meta.get("episode_title")
    if not isinstance(title, str) or not title.strip():
        gaps.append(
            "meta.episode_title: empty -- otr_credits_roll raises on it"
        )
    # The credits roll's seed receipt (meta.episode_seed OR
    # meta.cast_contract.cast_seed) is deliberately NOT checked here. It has
    # one owner per lane family and neither of them is this pass: a legacy
    # lane's seeded cast picker stamps cast_seed upstream, and a content-owned
    # lane's seed is stamped by the writer tail immediately after this call.
    # Claiming it here also made the tail non-reproducible -- a freshly minted
    # seed is by construction not derivable from the inputs, so a pass that
    # mints one for every lane cannot be byte-identical across two runs of the
    # same inputs (tests/test_fable2_tail_context.py pins exactly that).
    return gaps


def _ALLOWED_SPEAKER_ROLES() -> frozenset:
    """The single allowed-role authority, read from the freeze module.

    Imported here rather than copied so a future role change cannot leave
    this pass silently disagreeing with the gate that enforces it.
    """
    try:
        from ._otr_ledger_freeze import ALLOWED_SPEAKER_ROLES
    except ImportError:  # pragma: no cover -- flat test/standalone load
        from _otr_ledger_freeze import ALLOWED_SPEAKER_ROLES  # type: ignore
    return ALLOWED_SPEAKER_ROLES


# ---------------------------------------------------------------------------
# the pass
# ---------------------------------------------------------------------------


def run_ledger_cleanup(
    ledger_data: MutableMapping[str, Any],
    *,
    slot_fn: "Callable[..., str] | None" = None,
    bank_id: str = "",
) -> dict:
    """Complete, sanitize and verify the ledger. Stamps `meta.ledger_cleanup`.

    Order is load-bearing. Deterministic completion runs FIRST so the safety
    scan sees final rows and the gap report sees everything that could be
    filled. Safety repair runs BEFORE the caller stamps TTS delivery text --
    sanitizing after that stamp would leave the delivery string carrying the
    language the canonical row no longer has.

    `slot_fn` is the technical LLM slot; without one the deterministic half
    still runs and the receipt says which halves were skipped.

    Raises `LedgerIncompleteError` if any required writer-owned field still
    has no owner and no value. That is the only failure this pass owns:
    content is repaired, structure is named.
    """
    receipt: dict = {
        "schema_version": LEDGER_CLEANUP_VERSION,
        "bank": str(bank_id or ""),
    }

    deterministic = _complete_deterministic(ledger_data)
    receipt["deterministic_fills"] = deterministic

    # Safety repair REMOVED 2026-08-05 (operator directive: no content
    # guardrails on generated episodes). The pass used to rewrite a delivered
    # spoken row whose words matched the profanity / weapon / sexual list, which
    # on an adaptation lane meant editing the author. The receipt KEY stays with
    # its established shape and a defined value on every path -- a ripped pass
    # may not leave an unowned field, and meta.ledger_cleanup.safety is read by
    # the freeze cascade's receipt plumbing.
    receipt["safety"] = {
        "status": "retired",
        "patch_count": 0,
        "residual_hits": 0,
    }

    meta = ledger_data.setdefault("meta", {})
    receipt["prose_fills"] = _complete_prose(ledger_data, slot_fn)

    gaps = required_field_gaps(ledger_data)
    receipt["missing"] = gaps
    receipt["status"] = "incomplete" if gaps else "complete"
    if isinstance(meta, MutableMapping):
        meta["ledger_cleanup"] = receipt

    if gaps:
        log.error(
            "[ledger_cleanup] ledger incomplete after cleanup: %d field(s)",
            len(gaps),
        )
        raise LedgerIncompleteError(gaps, bank_id=bank_id)

    log.info(
        "[ledger_cleanup] complete: %d deterministic fill(s), %d prose "
        "fill(s), safety=%s",
        len(deterministic), len(receipt["prose_fills"]),
        receipt["safety"].get("status"),
    )
    return receipt
