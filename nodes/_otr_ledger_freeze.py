"""nodes/_otr_ledger_freeze.py — Ledger Freeze Cascade Phase 0 + Phase 10.

Scope of THIS module (sprint commit 1):

  Phase 0 (gap_audit_pre)
      Deterministic, read-only audit of the ledger BEFORE any cleanup
      runs. Stamps `meta.gap_audit_pre`. Never raises -- soft gaps
      surface as warnings and the cascade advances.

  Phase 10 (gap_audit_post + freeze)
      Deterministic, read-only audit AFTER the cascade. If any
      CRITICAL invariant is violated, raises `FreezeAssertionError`
      and stamps `meta.freeze_verdict = "needs_full_rerun"`. On
      success stamps:
          meta.cleanup_locked   = True
          meta.freeze_timestamp = iso8601 UTC
          meta.freeze_verdict   = "frozen_clean" | "frozen_with_warns"

The full Ledger Freeze Cascade (Phases 1-9) is layered on top of this
module in subsequent commits. The orchestrator that wires these phases
together lives in `nodes/OTR_LedgerFreezeCascade.py` (commit 2).

Reference: docs/2026-05-11-multi-turn-polish-adr.md (Ledger Freeze
Cascade ADR).

Schema mapping (ADR §6.16 reality-check vs L3 ledger):
    ADR §6.16 references `sfx_cues` / `music_cues` (plural lists) on
    each line. The live L3 schema has a top-level `music` list on the
    ledger root. The legacy top-level `sfx` list was deleted in S26
    (CD-3), and the whole sfx subsystem (the `sfx` speaker-role +
    `sfx_cue` field) followed 2026-07-01 (rip-sfx-broll). The
    null-rejection invariants here apply to the fields that actually
    exist:

      * top-level `cast` / `lines` / `beats` / `scenes` / `shots`
        / `music` / `clips`  -- must be list, never null.
      * `meta.news.key_terms`  -- must be list when present, never null.
      * `meta.episode_title` / `meta.style` / `meta.episode_id`
        -- must be non-empty string when present.
      * optional string fields (e.g. line.tts_skip_reason) must be ""
        when unset, never null.

Status: Phase 0 + Phase 10 of the Multi-Turn Polish sprint (2026-05-11).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, List, Literal, Optional

log = logging.getLogger("OTR.ledger_freeze")


__all__ = [
    "FreezeAssertionError",
    "GapAuditReport",
    "FreezeVerdict",
    "ALLOWED_FREEZE_VERDICTS",
    "ALLOWED_SPEAKER_ROLES",
    "EXPECTED_SCHEMA_VERSION",
    "run_gap_audit",
    "phase_0_gap_audit_pre",
    "phase_10_gap_audit_post_and_freeze",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Pinned by ADR §7. Pulled live from production_ledger so the two
# stay in lockstep; falls back to the hardcoded string if the import
# fails (defensive -- same pattern as Ledger.SCHEMA_VERSION).
try:
    from . import production_ledger as _PL_FOR_SCHEMA  # type: ignore

    EXPECTED_SCHEMA_VERSION: str = _PL_FOR_SCHEMA.Ledger.SCHEMA_VERSION
    del _PL_FOR_SCHEMA
except Exception:  # pragma: no cover -- defensive fallback
    EXPECTED_SCHEMA_VERSION = "l3-2026-05-14"


# Allowed values for line.speaker_role. Drawn from
# `_otr_ledger_reviewer._ALLOWED_SPEAKER_ROLES`; kept local so this
# module does not import the reviewer. "sfx" REMOVED 2026-07-01
# (rip-sfx-broll): an old ledger carrying speaker_role="sfx" now
# fails the per-line invariant as a hard ERROR (Phase 10 raises).
ALLOWED_SPEAKER_ROLES: frozenset[str] = frozenset({
    "character",
    "announcer",
    "music_open",
    "music_close",
    "music_inter",
})


# Pinned freeze verdicts (extends ADR §9 list -- only the literals
# THIS module can stamp on Phase 0 / Phase 10 are listed here).
FreezeVerdict = Literal[
    "frozen_clean",
    "frozen_with_warns",
    "needs_full_rerun",
]


ALLOWED_FREEZE_VERDICTS: frozenset[str] = frozenset({
    "frozen_clean",
    "frozen_with_warns",
    "needs_full_rerun",
})


# Top-level fields that must be present AND must be a list (never null,
# never omitted, never another type). Per ADR §6.16 null-rejection
# (adapted to L3 schema).
_REQUIRED_TOP_LEVEL_LISTS: tuple[str, ...] = (
    "cast",
    "lines",
    "beats",
    "scenes",
    "shots",
    # S26-A3: legacy top-level "sfx" list removed from schema; the
    # whole sfx subsystem followed 2026-07-01 (rip-sfx-broll) -- a
    # speaker_role="sfx" line is now an invariant ERROR.
    "music",
    "clips",
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class FreezeAssertionError(RuntimeError):
    """Raised by Phase 10 when a CRITICAL invariant is violated.

    Attributes:
        errors: list[str] -- one human-readable line per critical gap.
        report: GapAuditReport -- the full Phase 10 audit report for
            forensic inspection by the caller.
    """

    def __init__(self, errors: List[str], report: "GapAuditReport") -> None:
        self.errors: List[str] = list(errors)
        self.report: "GapAuditReport" = report
        msg = (
            f"Ledger freeze rejected: {len(errors)} critical gap(s); "
            f"first: {errors[0] if errors else '(empty)'}"
        )
        super().__init__(msg)


@dataclass
class GapAuditReport:
    """One run of the deterministic gap audit.

    `label` is "pre" (Phase 0) or "post" (Phase 10). On Phase 0 the
    audit is warn-only; on Phase 10 a non-empty `errors` list causes
    `FreezeAssertionError` to fire from the caller.

    Severity buckets:
        errors    -- CRITICAL invariant violations (e.g. voiced line
                     missing char_id; line_id collision; null where a
                     list is required). Phase 10 hard-fails on these.
        warnings  -- SOFT invariant violations (e.g. word_count out of
                     sync, optional meta key missing, beat without any
                     line). Both phases advance through these.
        info      -- numeric counters and audit metadata (line count,
                     voiced beat count, etc.) for forensic logging.
    """

    label: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    info: dict = field(default_factory=dict)

    @property
    def is_clean(self) -> bool:
        return not self.errors and not self.warnings

    @property
    def has_critical_gaps(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# Invariant checks (private)
# ---------------------------------------------------------------------------


def _check_null_rejection(
    ledger_data: dict,
    errors: List[str],
    warnings: List[str],
) -> None:
    """ADR §6.16: no null values where a list / dict / string is expected.

    Hard errors on the top-level list fields (cast / lines / beats / ...);
    everything else is a warning so the cascade can advance.
    """
    if not isinstance(ledger_data, dict):
        errors.append(
            f"ledger_data is not a dict: type={type(ledger_data).__name__}"
        )
        return
    for key in _REQUIRED_TOP_LEVEL_LISTS:
        if key not in ledger_data:
            errors.append(f"top-level key {key!r} is MISSING")
            continue
        value = ledger_data[key]
        if value is None:
            errors.append(f"top-level key {key!r} is null; expected list")
            continue
        if not isinstance(value, list):
            errors.append(
                f"top-level key {key!r} has type {type(value).__name__}; "
                f"expected list"
            )

    meta = ledger_data.get("meta")
    if meta is None:
        warnings.append("meta is missing or null")
        return
    if not isinstance(meta, dict):
        errors.append(f"meta has type {type(meta).__name__}; expected dict")
        return

    # meta.news.key_terms is allowed-empty but never null and never wrong-type.
    news = meta.get("news")
    if news is not None:
        if not isinstance(news, dict):
            errors.append(
                f"meta.news has type {type(news).__name__}; expected dict"
            )
        else:
            key_terms = news.get("key_terms")
            if key_terms is None:
                # Treat missing as [] (forward-compat) -- writer may not
                # have stamped news yet. Warn so soak diagnostics surface.
                if "key_terms" in news:
                    errors.append("meta.news.key_terms is null; expected list")
            elif not isinstance(key_terms, list):
                errors.append(
                    f"meta.news.key_terms has type {type(key_terms).__name__}; "
                    f"expected list"
                )


def _check_per_line_invariants(
    ledger_data: dict,
    errors: List[str],
    warnings: List[str],
    info: dict,
) -> None:
    """ADR §7 per-line invariants.

    line_id unique; char_id non-empty for voiced beats; speaker_role
    in the allowed enum; (text non-empty) OR (skip=True with reason);
    beat_id references an existing beat in meta.outline.beats (or
    top-level beats[]); word_count / char_count match text; no
    inconsistent text!="" AND skip=True combo.
    """
    lines = ledger_data.get("lines")
    if not isinstance(lines, list):
        # The null-rejection check above already errored this; skip
        # further per-line work to avoid double-reporting.
        info["line_count"] = 0
        info["voiced_beat_count"] = 0
        return
    info["line_count"] = len(lines)

    # Build the set of valid beat ids from top-level beats.
    # S28 cleanbreak: dropped the meta.outline.beats walk that was a
    # back-compat fallback for caller-shaped ledgers (pre-D-inversion,
    # pre-2026-05-10). OTR_LedgerScriptWriter always stamps top-level
    # `ledger["beats"]` from the outline pass.
    valid_beat_ids: set[str] = set()
    top_beats = ledger_data.get("beats")
    if isinstance(top_beats, list):
        for b in top_beats:
            if isinstance(b, dict):
                bid = b.get("beat_id")
                if isinstance(bid, str) and bid:
                    valid_beat_ids.add(bid)

    seen_line_ids: set[str] = set()
    voiced = 0
    for idx, ln in enumerate(lines):
        if not isinstance(ln, dict):
            errors.append(
                f"lines[{idx}] is type {type(ln).__name__}; expected dict"
            )
            continue
        line_id = ln.get("line_id")
        if not isinstance(line_id, str) or not line_id:
            errors.append(f"lines[{idx}] has empty/missing line_id")
        elif line_id in seen_line_ids:
            errors.append(f"lines[{idx}] line_id={line_id!r} is duplicated")
        else:
            seen_line_ids.add(line_id)

        speaker_role = ln.get("speaker_role")
        if not isinstance(speaker_role, str) or not speaker_role:
            errors.append(
                f"line_id={line_id!r} has empty/missing speaker_role"
            )
        elif speaker_role not in ALLOWED_SPEAKER_ROLES:
            errors.append(
                f"line_id={line_id!r} speaker_role={speaker_role!r} not in "
                f"{sorted(ALLOWED_SPEAKER_ROLES)}"
            )

        # Voiced beats (character + announcer) require char_id.
        if speaker_role in ("character", "announcer"):
            voiced += 1
            char_id = ln.get("char_id")
            if not isinstance(char_id, str) or not char_id:
                errors.append(
                    f"line_id={line_id!r} (voiced) has empty/missing char_id"
                )

        # text vs skip consistency.
        text = ln.get("text")
        if text is None:
            text = ""
        if not isinstance(text, str):
            errors.append(
                f"line_id={line_id!r} text is type {type(text).__name__}; "
                f"expected str"
            )
            text = ""
        skip = bool(ln.get("skip"))
        if skip:
            if text != "":
                errors.append(
                    f"line_id={line_id!r} has skip=True AND non-empty text "
                    f"(inconsistent state)"
                )
            tsr = ln.get("tts_skip_reason")
            if not isinstance(tsr, str) or not tsr:
                # S28 cleanbreak: promoted warning -> error. Phantom-skip
                # and reviewer-skip both stamp tts_skip_reason on every
                # production path. The pre-S28 warn-only tolerance for
                # "some legacy fallbacks set skip without the reason"
                # is extinct under Rule A + Rule B (uniform shape).
                errors.append(
                    f"line_id={line_id!r} skip=True but tts_skip_reason "
                    f"empty/missing (writer contract violation; every "
                    f"skip path must stamp the reason)"
                )
        else:
            # Non-skipped voiced beats must have non-empty text.
            if speaker_role in ("character", "announcer") and text == "":
                errors.append(
                    f"line_id={line_id!r} (voiced, not skipped) has empty text"
                )

        # beat_id reference (warn-only when valid_beat_ids is empty --
        # some unit-test fixtures omit beats[] entirely; we don't want
        # to false-fail Phase 10 on those).
        beat_id = ln.get("beat_id")
        if valid_beat_ids and isinstance(beat_id, str) and beat_id:
            if beat_id not in valid_beat_ids:
                warnings.append(
                    f"line_id={line_id!r} beat_id={beat_id!r} does not "
                    f"reference an existing beat"
                )

        # word_count / char_count vs text -- WARN-only so ledger writes
        # that haven't re-stamped counts don't false-fail.
        if isinstance(text, str) and text != "":
            wc = ln.get("word_count")
            cc = ln.get("char_count")
            real_wc = sum(1 for _ in text.split())
            real_cc = len(text)
            if isinstance(wc, int) and wc != real_wc:
                warnings.append(
                    f"line_id={line_id!r} word_count={wc} but text has "
                    f"{real_wc} word(s)"
                )
            if isinstance(cc, int) and cc != real_cc:
                warnings.append(
                    f"line_id={line_id!r} char_count={cc} but text has "
                    f"{real_cc} char(s)"
                )

        # tts_skip_reason: must be string when present (never null).
        if "tts_skip_reason" in ln:
            tsr_raw = ln.get("tts_skip_reason")
            if tsr_raw is None:
                errors.append(
                    f"line_id={line_id!r} tts_skip_reason is null; "
                    f"expected str"
                )
            elif not isinstance(tsr_raw, str):
                errors.append(
                    f"line_id={line_id!r} tts_skip_reason has type "
                    f"{type(tsr_raw).__name__}; expected str"
                )

    info["voiced_beat_count"] = voiced


def _check_per_cast_invariants(
    ledger_data: dict,
    errors: List[str],
    warnings: List[str],
) -> None:
    """ADR §7 per-cast-entry invariants.

    char_id / name / traits / voice_preset present and non-empty;
    each char_id is referenced by ≥ 1 non-skipped line (unless
    announcer-only-fallback flag set in meta).
    """
    cast = ledger_data.get("cast")
    if not isinstance(cast, list):
        return  # null-rejection already errored
    meta = ledger_data.get("meta") or {}
    announcer_only_fallback = (
        isinstance(meta, dict)
        and bool(meta.get("announcer_only_fallback"))
    )
    if not cast:
        if not announcer_only_fallback:
            errors.append(
                "cast is empty (and meta.announcer_only_fallback not set)"
            )
        return

    seen_char_ids: set[str] = set()
    for idx, row in enumerate(cast):
        if not isinstance(row, dict):
            errors.append(
                f"cast[{idx}] is type {type(row).__name__}; expected dict"
            )
            continue
        char_id = row.get("char_id")
        name = row.get("name")
        if not isinstance(char_id, str) or not char_id:
            errors.append(f"cast[{idx}] has empty/missing char_id")
        elif char_id in seen_char_ids:
            errors.append(
                f"cast[{idx}] char_id={char_id!r} is duplicated"
            )
        else:
            seen_char_ids.add(char_id)
        if not isinstance(name, str) or not name:
            errors.append(
                f"cast[{idx}] (char_id={char_id!r}) has empty/missing name"
            )
        # traits / voice_preset are WARN-only -- some retro test
        # fixtures populate only the minimum cast row shape.
        traits = row.get("traits") or row.get("character_description")
        if traits is None:
            warnings.append(
                f"cast[{idx}] (char_id={char_id!r}) has no traits/"
                f"character_description"
            )
        elif not isinstance(traits, (str, list)):
            errors.append(
                f"cast[{idx}] (char_id={char_id!r}) traits has type "
                f"{type(traits).__name__}; expected str or list"
            )
        # G6 invariant (voice-path-cleanbreak, Gate 2 of 3).
        # Every non-ANNOUNCER cast row must carry a non-empty
        # ``voice_preset`` starting with ``v2/`` (the Bark preset
        # namespace). ANNOUNCER is intentionally excluded because it
        # lives in the Kokoro namespace (bm_* / bf_*) by construction.
        # Empty / None / non-v2 preset is a writer contract violation.
        # S28 cleanbreak: dropped the "promoted from the legacy
        # WARN-fallback (tts_model / speaker_role substitutes) which
        # was a back-compat shim" framing — that retirement happened
        # in voice-path-cleanbreak Gate 2 and the shim is long gone;
        # the comment was carrying forensic history that's now in the
        # git log instead.
        is_announcer = (
            isinstance(char_id, str) and char_id.strip().lower() == "announcer"
        ) or (
            isinstance(name, str) and name.strip().upper() == "ANNOUNCER"
        )
        if is_announcer:
            # Kokoro namespace -- voice_preset still must be non-empty
            # but the v2/ prefix does not apply.
            if not row.get("voice_preset"):
                warnings.append(
                    f"cast[{idx}] (char_id={char_id!r}, ANNOUNCER) has no "
                    f"voice_preset"
                )
        else:
            voice_preset = row.get("voice_preset")
            if not voice_preset:
                # Sprint 2 (a): bark voice_preset is assigned by OTR_CastLock
                # AFTER this freeze (the writer no longer stamps it), so an empty
                # preset here is EXPECTED, not a contract violation. WARN only --
                # CastLock's exit invariant is the hard gate now.
                warnings.append(
                    f"cast[{idx}] (char_id={char_id!r}) has no voice_preset yet "
                    f"(assigned by OTR_CastLock post-freeze)"
                )
            elif not str(voice_preset).startswith("v2/"):
                errors.append(
                    f"G6: cast[{idx}] (char_id={char_id!r}) voice_preset "
                    f"{voice_preset!r} does not start with 'v2/' "
                    f"(Bark requires v2/* presets on non-ANNOUNCER rows)"
                )

    # Each char_id must be referenced by >= 1 non-skipped line (unless
    # announcer-only-fallback).
    if announcer_only_fallback:
        return
    lines = ledger_data.get("lines") or []
    referenced: set[str] = set()
    for ln in lines:
        if not isinstance(ln, dict):
            continue
        if ln.get("skip"):
            continue
        cid = ln.get("char_id")
        if isinstance(cid, str) and cid:
            referenced.add(cid)
    for char_id in seen_char_ids:
        if char_id == "announcer":
            # Announcer cast row commonly auto-stamped; missing
            # reference is OK if there are non-skipped announcer
            # lines OR the writer chose not to use it.
            continue
        if char_id not in referenced:
            warnings.append(
                f"cast char_id={char_id!r} is not referenced by any "
                f"non-skipped line"
            )


def _check_meta_invariants(
    ledger_data: dict,
    errors: List[str],
    warnings: List[str],
) -> None:
    """ADR §7 meta invariants.

    schema_version pinned; episode_title / style / episode_id stamped
    (warn-only); meta.audit_passes / cleanup_passes / readiness_passes
    are lists when present (B6 bucket split, 2026-05-12).
    """
    schema_version = ledger_data.get("schema_version")
    if schema_version != EXPECTED_SCHEMA_VERSION:
        errors.append(
            f"schema_version={schema_version!r}; expected "
            f"{EXPECTED_SCHEMA_VERSION!r}"
        )

    meta = ledger_data.get("meta")
    if not isinstance(meta, dict):
        # Already errored / warned by null-rejection.
        return

    for key in ("episode_title", "style"):
        val = meta.get(key)
        if (
            key == "style"
            and val is None
            and meta.get("story_scaffold_enabled") is False
            and meta.get("story_style_status") == "story_scaffold_off"
        ):
            continue
        if val is None:
            warnings.append(f"meta.{key} is missing")
            continue
        if not isinstance(val, str):
            errors.append(
                f"meta.{key} has type {type(val).__name__}; expected str"
            )
        elif val == "":
            warnings.append(f"meta.{key} is empty string")

    # Style-engine consolidation (2026-07-05): meta.gen_params_initial.style
    # (and the whole gen_params_initial style/style_combo/style_custom/
    # style_source quartet) is DELETED along with the retired resolver --
    # this block used to validate that field's slug shape (S25/MG-6
    # BUG-LOCAL-216, relaxed BUG-LOCAL-240) and is removed with it. The
    # canonical `meta.style` (already checked above) is the only surviving
    # style record now, derived from meta.story_contract.slug.

    # Phase-history bucket lists (B6 split, 2026-05-12). Optional
    # at Phase 0; present-as-list at Phase 10. Validate-when-present
    # for each bucket.
    for bucket_key in ("audit_passes", "cleanup_passes", "readiness_passes"):
        bucket = meta.get(bucket_key)
        if bucket is not None and not isinstance(bucket, list):
            errors.append(
                f"meta.{bucket_key} has type {type(bucket).__name__}; "
                f"expected list"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_gap_audit(ledger_data: dict, *, label: str) -> GapAuditReport:
    """Run the full deterministic gap-audit pass.

    Same function for Phase 0 (label="pre") and Phase 10 (label="post").
    Read-only -- mutates nothing on `ledger_data`. The caller decides
    whether to stamp `meta.gap_audit_pre` / `meta.gap_audit_post`.
    """
    report = GapAuditReport(label=label)
    _check_null_rejection(ledger_data, report.errors, report.warnings)
    # If null-rejection already failed at the top level the deeper
    # checks would just double-report; gate them on top-level shape
    # being well-formed enough to walk.
    if isinstance(ledger_data, dict):
        _check_per_line_invariants(
            ledger_data, report.errors, report.warnings, report.info,
        )
        _check_per_cast_invariants(
            ledger_data, report.errors, report.warnings,
        )
        _check_meta_invariants(
            ledger_data, report.errors, report.warnings,
        )
        _check_g8_line_id_uniqueness(
            ledger_data, report.errors, report.warnings,
        )
        _check_g9_sfw_spoken_text(
            ledger_data, report.errors, report.warnings,
        )
        _check_g10_genre_spoken_text(
            ledger_data, report.errors, report.warnings,
        )
    return report


# G9 SFW ship-stop on spoken text. Phase 0 collect / Phase 10 raise.
#
# PD4 is a hard rule ("Safe for work. No profanity.") and until now nothing
# enforced it on the episode that actually ships. `_otr_ledger_scrub` runs only
# on the lanes with `run_story_spine=True` -- never on the content-owned lanes --
# and even where it runs, its verdict is stamped and read by nobody. The lanes
# that had any profanity check at all had it as an LLM opinion.
#
# This is the deterministic version: one word-boundary scan over the spoken text
# that reaches the microphone, on the one path EVERY lane crosses. It runs inside
# the gap audit, so Phase 10 raises FreezeAssertionError on it exactly like any
# other critical gap -- a verdict an existing caller already acts on.
#
# Announcer lines are in scope: the bookends ship too.


def _profanity_terms() -> List[str]:
    """The live SFW vocabulary. Global policy -- never pack-tunable."""
    try:
        from ._otr_stage3_validators import DEFAULT_PROFANITY_TERMS
    except ImportError:  # pragma: no cover -- flat test/standalone load
        from _otr_stage3_validators import DEFAULT_PROFANITY_TERMS  # type: ignore
    return list(DEFAULT_PROFANITY_TERMS)


def _sfw_pattern() -> "re.Pattern[str]":
    """Whole-word, case-insensitive alternation over the live term list.

    Word-boundary, so "hell" fires on "go to hell" and never on "hello",
    "shellfish", or "Michelle". Multi-word entries ("screw you") keep their
    internal spacing and stay bounded at both ends.
    """
    terms = sorted(
        (t.strip() for t in _profanity_terms() if t and t.strip()),
        key=len, reverse=True,
    )
    if not terms:
        return re.compile(r"(?!x)x")  # matches nothing
    body = "|".join(re.escape(term) for term in terms)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def _check_g9_sfw_spoken_text(
    ledger_data: dict,
    errors: List[str],
    warnings: List[str],
) -> None:
    """G9: no profanity in the spoken text of any ledger line."""
    lines = ledger_data.get("lines")
    if not isinstance(lines, list):
        return
    pattern = _sfw_pattern()
    hits: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        text = line.get("text")
        if not isinstance(text, str) or not text:
            continue
        found = sorted({match.group(0).lower() for match in pattern.finditer(text)})
        if found:
            lid = line.get("line_id") or "<no line_id>"
            hits.append(f"{lid}: {', '.join(found)}")
    if hits:
        sample = hits[:5]
        more = "" if len(hits) <= 5 else f" (+{len(hits) - 5} more)"
        errors.append(
            f"G9: profanity in the spoken text of {len(hits)} line(s): "
            f"{'; '.join(sample)}{more}. PD4 is a hard ship rule -- the "
            f"episode does not freeze with these words on the microphone. "
            f"The vocabulary is DEFAULT_PROFANITY_TERMS in "
            f"_otr_stage3_validators."
        )


# G10 GENRE spoken-text ship-stop (v4 campaign P1(iii)). Phase 0 collect / Phase
# 10 raise -- the deterministic terminal for a banned GENRE phrase surviving on
# the microphone. Bank-aware + OPT-IN: reads meta["genre_banned_phrases"], which
# the writer stamps ONLY when the bank sets defaults.genre_guard_spoken=True. The
# key is absent for every current bank, so G10 is INERT until a v4 lane opts in.
#
# THE LAW: an audit may improve a story, never fail one; only a DETERMINISTIC
# validator ends an episode. The upstream authored repair (_otr_genre_guard.
# apply_genre_spoken_repair, run at the writer creative boundary) is the "improve"
# side; this read-only scan is the terminal for a surviving hit. It uses the
# boundary matcher (casefolded, Unicode-aware) so "gun" never fires on "begun".
# Runs on the one path EVERY execution family crosses (codex phase_10 finalizer,
# inline run_freeze_cascade, fable2 finalizer), exactly like G9.


def _check_g10_genre_spoken_text(
    ledger_data: dict,
    errors: List[str],
    warnings: List[str],
) -> None:
    """G10: no banned GENRE phrase in the spoken text of any ledger line."""
    meta = ledger_data.get("meta")
    if not isinstance(meta, dict):
        return
    phrases = meta.get("genre_banned_phrases")
    if not isinstance(phrases, list) or not phrases:
        return  # opt-in only; inert for every current bank
    try:
        from ._otr_genre_guard import scan_spoken_lines
    except ImportError:  # pragma: no cover -- flat test/standalone load
        from _otr_genre_guard import scan_spoken_lines  # type: ignore
    hits = [
        f"{lid}: {', '.join(found)}"
        for lid, found in scan_spoken_lines(ledger_data, phrases)
    ]
    if hits:
        sample = hits[:5]
        more = "" if len(hits) <= 5 else f" (+{len(hits) - 5} more)"
        errors.append(
            f"G10: banned genre phrase in the spoken text of {len(hits)} "
            f"line(s): {'; '.join(sample)}{more}. The bank sets "
            f"genre_guard_spoken; the vocabulary is its story_rules."
            f"banned_phrases. An audit may improve a story, never fail one -- "
            f"the fix is the upstream authored repair, not a relaxed gate."
        )


# G7 (SFX per-cue dur_s bounds) DELETED 2026-07-01 (rip-sfx-broll):
# the sfx speaker-role no longer exists, so there are no sfx lines to
# bound. An old ledger carrying speaker_role="sfx" now fails the
# per-line invariant (ALLOWED_SPEAKER_ROLES) as a hard ERROR instead.


# G8 line_id uniqueness. Phase 0 collect / Phase 10 raise.
#
# Voice-path-cleanbreak Sprint 13.2 (2026-05-12): added per IMP-8
# from the S6-S8 round-robin. Every ledger write-back
# path that stamps by line_id assumes uniqueness. Without G8 enforcing
# it, two lines with the same line_id silently overwrite each other
# both on disk and in the ledger.


def _check_g8_line_id_uniqueness(
    ledger_data: dict,
    errors: List[str],
    warnings: List[str],
) -> None:
    """G8: line_id uniqueness across ledger.lines[].

    Every ledger write-back path
    (ledger.apply_line_timings, ledger.patch_line_fields, etc.) keys
    by line_id. Duplicates produce silent overwrites in BOTH places:
    on disk the second render replaces the first wav, in the ledger
    the second patch_line_fields call clobbers the first row's
    stamps. The user notices when an episode renders with the wrong
    audio in a slot -- too late to abort cleanly.

    Skips lines without line_id (already caught by per-line type
    invariants; no need to double-report). Empty list / non-list
    `lines` value handled defensively.
    """
    lines = ledger_data.get("lines")
    if not isinstance(lines, list):
        return
    seen: set[str] = set()
    duplicates: list[str] = []
    for line in lines:
        if not isinstance(line, dict):
            continue
        lid = line.get("line_id")
        if not lid or not isinstance(lid, str):
            continue
        if lid in seen:
            duplicates.append(lid)
        else:
            seen.add(lid)
    if duplicates:
        # Cap the displayed list at 5 to keep the diagnostic short.
        # The full count gives the operator a sense of severity.
        sample = duplicates[:5]
        more = "" if len(duplicates) <= 5 else f" (+{len(duplicates) - 5} more)"
        errors.append(
            f"G8: {len(duplicates)} duplicate line_id(s) across "
            f"ledger.lines[]: {', '.join(sample)}{more}. "
            f"Render filenames and ledger write-back paths key by "
            f"line_id; duplicates produce silent overwrites in BOTH "
            f"places. The writer must emit unique line_ids."
        )


def _coerce_ledger_data(led_or_data) -> dict:
    """Return ledger_data for both Ledger objects and bare dicts.

    The cascade is invoked on a `production_ledger.Ledger` in
    production; unit tests often hand a bare dict. Accept both.
    """
    if isinstance(led_or_data, dict):
        return led_or_data
    return led_or_data.data  # type: ignore[attr-defined]


def _safe_stamp_meta(ledger_data: dict, key: str, value) -> bool:
    """Stamp ledger_data['meta'][key] = value when meta is a dict.

    Returns True if the stamp landed. When meta is the wrong type
    (a list, a string, etc.) the audit has already errored that --
    stamping would just crash with TypeError. Bail and let the
    caller raise FreezeAssertionError on the existing errors.
    """
    meta = ledger_data.get("meta")
    if not isinstance(meta, dict):
        return False
    meta[key] = value
    return True


def _report_dict(report: GapAuditReport) -> dict:
    return {
        "label": report.label,
        "errors": list(report.errors),
        "warnings": list(report.warnings),
        "info": dict(report.info),
    }


def phase_0_gap_audit_pre(led) -> GapAuditReport:
    """LFC Phase 0. Deterministic warn-mode audit on entry to the cascade.

    Stamps `meta.gap_audit_pre = {label, errors, warnings, info}` when
    `meta` is a dict. NEVER raises -- soft gaps surface as warnings;
    even CRITICAL errors at this stage advance to the cascade so the
    cleanup phases have a chance to repair them. Phase 10 is the hard
    gate.
    """
    ledger_data = _coerce_ledger_data(led)
    report = run_gap_audit(ledger_data, label="pre")
    _safe_stamp_meta(ledger_data, "gap_audit_pre", _report_dict(report))
    if report.errors:
        log.warning(
            "[LFC:phase_0] %d critical gap(s) detected -- advancing to "
            "cascade (Phase 10 will hard-fail if unrepaired). First: %s",
            len(report.errors), report.errors[0],
        )
    elif report.warnings:
        log.info(
            "[LFC:phase_0] %d soft gap(s) detected; advancing.",
            len(report.warnings),
        )
    else:
        log.info("[LFC:phase_0] ledger is gap-clean.")
    return report


def phase_10_gap_audit_post_and_freeze(led) -> GapAuditReport:
    """LFC Phase 10. Hard-asserts critical gaps and stamps the freeze.

    On critical gap (`report.errors` non-empty):
        meta.freeze_verdict        = "needs_full_rerun"  (when stampable)
        meta.gap_audit_post        = report dict          (when stampable)
        raises FreezeAssertionError(errors=..., report=...)

    On no critical gaps:
        meta.gap_audit_post        = report dict
        meta.cleanup_locked        = True
        meta.freeze_timestamp      = ISO-8601 UTC string
        meta.freeze_verdict        = "frozen_clean" if no warnings,
                                     else "frozen_with_warns"

    When `meta` is the wrong type (list / str / etc.), the audit has
    already flagged it as a critical gap -- the function then raises
    without attempting to stamp (stamping a non-dict would just crash
    with TypeError and obscure the real diagnostic).

    Idempotent on a well-shaped ledger: calling Phase 10 twice on an
    already-frozen ledger updates the stamps in place and keeps the
    freeze lock raised.
    """
    ledger_data = _coerce_ledger_data(led)
    report = run_gap_audit(ledger_data, label="post")
    stamped = _safe_stamp_meta(
        ledger_data, "gap_audit_post", _report_dict(report),
    )
    if report.errors:
        if stamped:
            _safe_stamp_meta(ledger_data, "freeze_verdict", "needs_full_rerun")
        log.error(
            "[LFC:phase_10] %d critical gap(s) -- FREEZE REJECTED. "
            "First: %s",
            len(report.errors), report.errors[0],
        )
        raise FreezeAssertionError(list(report.errors), report)

    # Successful path -- meta must be a dict (no errors means
    # null-rejection passed, which guarantees meta is a dict or None;
    # ensure-dict here covers the missing-meta case).
    meta = ledger_data.setdefault("meta", {})
    if not isinstance(meta, dict):  # pragma: no cover -- defensive
        raise FreezeAssertionError(
            [f"meta has type {type(meta).__name__}; cannot stamp freeze"],
            report,
        )
    meta["cleanup_locked"] = True
    meta["freeze_timestamp"] = datetime.now(timezone.utc).isoformat()
    verdict: FreezeVerdict = (
        "frozen_with_warns" if report.warnings else "frozen_clean"
    )
    meta["freeze_verdict"] = verdict
    if report.warnings:
        log.info(
            "[LFC:phase_10] frozen_with_warns -- %d soft gap(s).",
            len(report.warnings),
        )
    else:
        log.info("[LFC:phase_10] frozen_clean.")
    return report
