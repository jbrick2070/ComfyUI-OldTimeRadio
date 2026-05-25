"""nodes/_otr_ledger_reviewer.py — Phase 3 two-pass cast-gated reviewer.

Pipeline architecture (per script-writing-architecture synthesis §3
Phase 3):

  LOCKED CAST -> PRE-AUDIT -> SCRIPT DOCTOR -> PATCH CANDIDATE
              -> POST-AUDIT -> SAVE

Pass 1 (Cast Auditor pre-check) -- LLM #1
    Reads meta.cast_contract + every line. Returns structured
    CastViolation list. Python deterministic-repair pass then fixes
    bad_casing / wrong_char_id / role_mismatch / alias_used and runs
    invented-name auto-remap via Levenshtein.

Pass 2 (Script Doctor) -- LLM #2
    Reads the repaired ledger. Proposes rewrite / skip / annotate
    edits, bounded by the scaled edit_cap. Python applies the patch
    to a CANDIDATE copy (the original ledger on disk is untouched
    until Pass 3 clears).

Step 2.5 RETIRED in S33 B4 (2026-05-15)
    Phantom-skip fallback was a mute (set line.skip=True), not a
    story edit; pipeline cut under the refined no-auditors rule.

Pass 3 RETIRED in S33 B3 (2026-05-15)
    Post-edit auditor LLM call had no editor consumer once the
    `post_audit_pass` rollback gate was retired in S33 B2.

Final disposition stamped on meta.reviewer_verdict. Only Phase 1
audit -> deterministic repairs -> Phase 2 Script Doctor remain.

Status: Phase 3 of v2.0 sprint (2026-05-11). New module.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

log = logging.getLogger("OTR")


__all__ = [
    "ReviewerVerdict",
    "CastViolation",
    "PreAuditReport",
    "ReviewerEdit",
    "ScriptDoctorReport",
    "ReviewerDisposition",
    "audit_cast_contract",
    "apply_deterministic_cast_repairs",
    "auto_remap_phantom",
    "_resolve_cast_member",
    "compute_edit_cap",
    "review_ledger",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Pinned reviewer verdicts. Imported everywhere meta.reviewer_verdict
# is read or written (per synthesis G7).
#
# S33 B2 (2026-05-15): `cast_unrecoverable` and `post_audit_failed`
# removed per refined no-auditors rule. Both verdicts were rollback-
# gate outputs (pipeline cuts, not story edits). The cast-unrecoverable
# verdict required the speaker_unknowns rollback gate (deleted same
# commit); the post_audit_failed verdict required the post_audit_pass
# rollback gate (also deleted same commit). Per Jeffrey's phantom-ship
# policy, occasional phantoms reaching the audience is the accepted
# trade-off vs preserving the rollback gates.
ReviewerVerdict = Literal[
    "clean_no_edits",
    "improved",
    "too_many_edits",
    "needs_full_rerun",
]


# Levenshtein threshold for invented-name auto-remap (per §6.A).
# Names ≥ 5 chars allow distance ≤ 3; shorter names use a tighter cap.
# Substring-containment is a secondary fast path (case-insensitive).
_LEVENSHTEIN_THRESHOLD = 3


# Post-Phase-3 review Gap 2 follow-up (2026-05-11): `_TITLED_PHANTOM_RE`
# constant removed. All call sites used the canonical
# `_otr_line_composer.detect_phantom_names` which covers titled names
# + ALL-CAPS tokens + Title-Case bigrams. One detector, one roster,
# everywhere. (S33 B4 retired `apply_phantom_skip_fallback` and
# `_final_phantom_check`; only `_detect_titled_phantoms` remains as
# an active call site.)


# Generation params for the three LLM calls (kept conservative; the
# auditor + doctor benefit from low-temp determinism, not creativity).
_AUDIT_TEMPERATURE = 0.2
_AUDIT_MAX_NEW_TOKENS = 2000
_DOCTOR_TEMPERATURE = 0.5
_DOCTOR_MAX_NEW_TOKENS = 3500

# Structural-retry temperatures for the structured_call ladder (Sprint
# 2A/2B). Attempt 2 re-rolls at a temperature STRICTLY BELOW the base
# attempt -- lowering entropy during a JSON-schema retry, never raising
# it. structured_call asserts this invariant at entry.
_AUDIT_RETRY_TEMPERATURE = 0.1
_DOCTOR_RETRY_TEMPERATURE = 0.3


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CastViolation(BaseModel):
    """Pure anomaly extraction from the Cast Contract Auditor.

    Sprint 3F (2026-05-25): the `confidence: float` field was removed.
    Small models cannot reliably distinguish 0.7 from 0.8 or 0.8 from
    0.9, so the auditor no longer scores its own certainty. The
    auditor's job is anomaly extraction only -- `found` (the literal
    that drifted), `expected` (the cast member it should have been, or
    "" when the auditor cannot name one), and `kind`. Python alone
    decides whether and how to repair: exact case-fold match first,
    then Levenshtein distance via `auto_remap_phantom`. Per the
    Operating Philosophy -- the LLM proposes anomalies; deterministic
    wrappers commit repairs.
    """
    line_id: str
    kind: Literal[
        "bad_casing",
        "alias_used",
        "invented_name",
        "wrong_char_id",
        "role_mismatch",
        "speaker_unknown",
    ]
    found: str
    expected: str = ""


class PreAuditReport(BaseModel):
    violations: list[CastViolation] = Field(default_factory=list)
    pass_clean: bool = True
    # Wiring-review #8 (2026-05-11): audit fail-loud. When the
    # auditor LLM call raises, returns garbage that doesn't parse,
    # or returns JSON that doesn't validate, `audit_cast_contract`
    # constructs a SENTINEL report with audit_failed=True +
    # pass_clean=False so the caller can branch to needs_full_rerun.
    # (S33 B2: post_audit_failed branch retired with the rollback
    # gate.) The default pass_clean=True is reserved for "LLM ran
    # cleanly and reported no violations" -- never for "audit didn't
    # run". Pure pydantic field-only changes; no downstream rewrite
    # needed beyond branching on these fields.
    audit_failed: bool = False
    audit_failure_reason: str = ""


def _audit_failed_sentinel(reason: str) -> "PreAuditReport":
    """Synthetic PreAuditReport stamped when the auditor LLM path
    failed. pass_clean=False so the caller's `if not pass_clean`
    branch fires and the verdict maps to needs_full_rerun.

    S33 B2 (2026-05-15): the historical "post" label path mapped to
    `post_audit_failed`; that branch was retired together with the
    `post_audit_pass` rollback gate. Only the "pre" label still
    consults this sentinel."""
    return PreAuditReport(
        violations=[CastViolation(
            line_id="__audit__",
            kind="invented_name",     # closest pinned kind; audit_failed
                                      # surfaces via audit_failed flag
            found=reason[:200],
            expected="",
        )],
        pass_clean=False,
        audit_failed=True,
        audit_failure_reason=reason,
    )


class ReviewerEdit(BaseModel):
    line_id: str
    action: Literal["rewrite", "skip", "annotate"]
    payload: str
    rationale: str = ""


class ScriptDoctorReport(BaseModel):
    edits: list[ReviewerEdit] = Field(default_factory=list)
    overall_verdict: Literal["clean", "improved", "needs_full_rerun"] = "clean"


# ---------------------------------------------------------------------------
# ReviewerDisposition (final return)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReviewerDisposition:
    """End-of-review summary. Stamped to meta.reviewer_disposition
    alongside meta.reviewer_verdict."""

    # S33 B4 (2026-05-15): `phantom_skip_count` field retired with
    # `apply_phantom_skip_fallback`. The dataclass field is gone; no
    # caller writes it.
    verdict: str  # ReviewerVerdict literal
    pre_audit_violations: int
    pre_audit_repairs_applied: int
    doctor_edits_proposed: int
    doctor_edits_applied: int
    post_audit_violations: int


# ---------------------------------------------------------------------------
# Levenshtein distance (pure Python, no external dep)
# ---------------------------------------------------------------------------


def _levenshtein(a: str, b: str) -> int:
    """Iterative Levenshtein distance. Case-folded by caller."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins = cur[j - 1] + 1
            dele = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(ins, dele, sub))
        prev = cur
    return prev[-1]


# ---------------------------------------------------------------------------
# auto_remap_phantom
# ---------------------------------------------------------------------------


def auto_remap_phantom(
    phantom: str,
    cast_roster: Iterable[str],
    *,
    threshold: int = _LEVENSHTEIN_THRESHOLD,
) -> Optional[str]:
    """Map a phantom name to the closest cast roster member, or None.

    Strategy:
      1. Case-folded substring containment fast-path. If the phantom
         contains a cast name as a substring (or vice versa) of
         length ≥ 3, that's a confident match.
      2. Levenshtein distance on case-folded forms. The smallest
         distance ≤ threshold wins. Ties drop to None (ambiguous).

    Caller is responsible for substituting the phantom literal in the
    line text. This function does NOT mutate anything.

    Per §6.A: cast is LOCKED -- no LLM reroll for invented names. If
    auto_remap returns None, the line falls through to the Script
    Doctor's rewrite/skip step. (S33 B4 retired the Step 2.5
    deterministic phantom-skip fallback that previously caught
    residual phantoms.)
    """
    if not phantom:
        return None
    roster_list = [r for r in (cast_roster or ()) if isinstance(r, str) and r]
    if not roster_list:
        return None
    p_fold = phantom.casefold()
    # Fast path: substring containment in either direction (>= 3 chars).
    for member in roster_list:
        m_fold = member.casefold()
        if len(m_fold) >= 3 and m_fold in p_fold:
            return member
        if len(p_fold) >= 3 and p_fold in m_fold:
            return member
    # Levenshtein. Smallest distance ≤ threshold wins; ties -> None.
    best: tuple[int, Optional[str]] = (threshold + 1, None)
    second_best_dist = threshold + 1
    for member in roster_list:
        d = _levenshtein(p_fold, member.casefold())
        if d < best[0]:
            second_best_dist = best[0]
            best = (d, member)
        elif d == best[0]:
            second_best_dist = d
    if best[0] > threshold:
        return None
    if best[0] == second_best_dist:
        return None
    return best[1]


def _resolve_cast_member(
    candidate: str,
    cast_roster: Iterable[str],
) -> Optional[str]:
    """Resolve an auditor-suggested name to a real cast member.

    Sprint 3F (2026-05-25): the deterministic resolver Python uses in
    place of the retired `confidence` gate. Strategy:

      1. Exact case-fold match against the roster. If exactly one
         member case-folds equal to `candidate`, that is the answer
         (and it is the canonical roster spelling, not the auditor's
         casing).
      2. Otherwise fall through to `auto_remap_phantom` (case-fold
         substring fast-path, then Levenshtein <= 3). Ambiguous ties
         already resolve to None inside `auto_remap_phantom`.

    Returns the canonical roster spelling, or None when nothing
    resolves or the match is an ambiguous tie -- the caller escalates
    a None exactly as it escalates any unresolved violation (leave the
    row for the Script Doctor). One Levenshtein implementation; this
    reuses `auto_remap_phantom` / `_levenshtein` rather than adding a
    second matcher.
    """
    if not candidate:
        return None
    roster_list = [r for r in (cast_roster or ()) if isinstance(r, str) and r]
    if not roster_list:
        return None
    cand_fold = candidate.casefold()
    exact = [m for m in roster_list if m.casefold() == cand_fold]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        # Two roster members case-fold identical -- a genuine tie.
        # Escalate rather than guess.
        return None
    return auto_remap_phantom(candidate, roster_list)


# ---------------------------------------------------------------------------
# Cast contract auditor (Pass 1 + Pass 3 use the same function)
# ---------------------------------------------------------------------------


def _render_cast_contract_table(cast_rows: list[dict]) -> str:
    """One row per character, plain text. Reused by the auditor prompt."""
    if not cast_rows:
        return "(no cast contract present)"
    lines = []
    for row in cast_rows:
        char_id = row.get("char_id", "")
        name = row.get("name", "")
        role = row.get("speaker_role") or row.get("tts_model") or ""
        desc = row.get("character_description") or ""
        lines.append(
            f"- char_id={char_id} canonical_name={name!r} "
            f"role={role!r} description={desc!r}"
        )
    return "\n".join(lines)


def _render_lines_for_audit(lines: list[dict]) -> str:
    """Compact one-line-per-line listing for auditor prompts."""
    if not lines:
        return "(no lines in ledger)"
    out = []
    for ln in lines:
        out.append(
            f"- line_id={ln.get('line_id','')} "
            f"speaker_role={ln.get('speaker_role','')} "
            f"char_id={ln.get('char_id','')} "
            f"text={(ln.get('text') or '')[:200]!r}"
        )
    return "\n".join(out)


_AUDITOR_SYSTEM_PROMPT = """\
You are a cast contract auditor for an audio drama script. Your job is
to detect deviations from the locked cast contract in the script ledger.

You DO NOT rewrite dialogue. You DO NOT propose creative changes. You
DO NOT score your own certainty. You only extract anomalies: the
literal that drifted, the cast member it should have been, and the
kind of drift. Python decides the repair downstream.

VIOLATION TYPES:
- bad_casing: speaker is the right name but cased wrong (e.g. "alice"
  when the cast says "ALICE").
- alias_used: line uses an alias (e.g. "AL") that exists in cast_rows
  but the canonical name should appear.
- invented_name: dialogue text references a proper noun that is
  neither in cast_contract nor in meta.key_terms.
- wrong_char_id: lines[k].char_id does not correspond to the speaker
  name in cast_contract.
- role_mismatch: lines[k].speaker_role does not match the speaker's
  role in cast_contract.
- speaker_unknown: speaker field is not in cast_contract at all.

For every violation you find, output one anomaly object:
- found: the exact literal in the ledger that drifted (the misspelled
  name, the alias, the invented proper noun, the wrong char_id, etc.).
- expected: the cast member or canonical value it should have been.
  Leave it "" when you cannot name one (e.g. an invented_name with no
  obvious cast match) -- Python resolves it.
- kind: one of the six violation types above.

Report only anomalies that are real drift from the locked cast
contract. Do not score, rank, or weight your findings -- emit the
anomaly facts only and let the downstream step decide the repair.

Return EXACTLY one JSON object matching this schema:
{
  "violations": [
    {"line_id": "...", "kind": "...", "found": "...", "expected": "..."},
    ...
  ],
  "pass_clean": true|false
}

No prose before or after the JSON. No markdown fences.
"""


# JSON extraction: the naive first-'{'-to-last-'}' extractor was removed
# in the BUG-LOCAL-261 consolidation; reviewer JSON is now parsed via the
# shared _otr_json.parse_first_json_object. Package import in production;
# flat import when loaded standalone / under test.
try:
    from . import _otr_json
except ImportError:  # pragma: no cover - standalone / test load
    import _otr_json  # type: ignore

# Sprint 2A: the shared structured-JSON retry ladder. Both LLM passes
# in this module (audit_cast_contract, run_script_doctor) route through
# it. Package import in production; flat import under standalone test.
try:
    from ._otr_structured_call import (
        structured_call,
        StructuredCallFailedError,
    )
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_structured_call import (  # type: ignore
        structured_call,
        StructuredCallFailedError,
    )

# Sprint 2C: typed repair-prompt factories. audit_cast_contract and
# run_script_doctor pass a dispatching factory so structured_call's
# Attempt 3 routes the repair turn by failure class. Package import in
# production; flat import under standalone test.
try:
    from ._otr_repair_prompts import make_dispatching_repair_factory
except ImportError:  # pragma: no cover - standalone / test load
    from _otr_repair_prompts import make_dispatching_repair_factory  # type: ignore


def audit_cast_contract(
    generate_fn,
    ledger_data: dict,
    label: str = "pre",
) -> PreAuditReport:
    """Run the Cast Contract Auditor LLM pass.

    `label` is retained in the signature for forensic log clarity
    (it surfaces in [OTR_LedgerReviewer:%s] messages). S33 B3
    (2026-05-15) retired the only non-"pre" call site -- the
    historical Phase 9 `label="post"` post-edit audit. Only Phase 1
    (label="pre") now invokes this function. The default value
    matches the only surviving call site.

    Output `PreAuditReport.violations` is consumed by
    `apply_deterministic_cast_repairs` (an editor that rewrites
    line text to fix phantom names / bad casing / wrong char_id),
    so the auditor survives the S33 no-auditors rule -- its output
    USES the audit to develop the story.

    Returns a PreAuditReport. On LLM failure or unparseable JSON,
    returns `_audit_failed_sentinel()` (pass_clean=False,
    audit_failed=True). Downstream Python guards still catch
    structural drift.
    """
    cast_rows = ledger_data.get("cast") or []
    lines = ledger_data.get("lines") or []
    user_prompt = (
        "LOCKED CAST CONTRACT (authoritative):\n"
        f"{_render_cast_contract_table(cast_rows)}\n\n"
        "LEDGER LINES TO AUDIT:\n"
        f"{_render_lines_for_audit(lines)}\n\n"
        "Output a single JSON PreAuditReport. No prose, no commentary."
    )
    messages = [
        {"role": "system", "content": _AUDITOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # Wiring-review #8 (2026-05-11): EVERY failure path below returns
    # _audit_failed_sentinel() (pass_clean=False, audit_failed=True),
    # NOT the bare default PreAuditReport(). S33 B2 (2026-05-15) the
    # caller maps the failure to needs_full_rerun (post_audit_failed
    # branch retired with the rollback gate); the deterministic
    # cast repairs still consume the sentinel violations list.
    # Sprint 2A/2D: the single-shot call + parse + validate is now the
    # shared structured_call 3-attempt ladder (base -> structural retry
    # -> typed repair). An exhausted ladder raises
    # StructuredCallFailedError -- the converted form of the prior
    # three failure arms. The slot fn (the LLM call) can still raise
    # arbitrary loader exceptions, which structured_call does not catch;
    # the broad `except Exception` below keeps this function's
    # never-raises contract (every failure -> _audit_failed_sentinel).
    # LLM slot: technical -- cast-contract auditor is structured validation
    try:
        report = structured_call(
            prompt=messages,
            schema=PreAuditReport,
            slot_fn=generate_fn,
            base_temperature=_AUDIT_TEMPERATURE,
            structural_retry_temperature=_AUDIT_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=_AUDIT_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name=f"audit_cast_contract:{label}",
        )
    except StructuredCallFailedError as exc:
        log.warning(
            "[OTR_LedgerReviewer:%s] structured_call exhausted the retry "
            "ladder after %d attempt(s) (last error: %s); returning "
            "audit_failed sentinel", label, exc.attempts, exc.last_error,
        )
        return _audit_failed_sentinel(f"structured_call failed: {exc}")
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        log.warning(
            "[OTR_LedgerReviewer:%s] structured_call raised %s: %s; "
            "returning audit_failed sentinel",
            label, type(exc).__name__, exc,
        )
        return _audit_failed_sentinel(
            f"structured_call raised: {type(exc).__name__}: {exc}"
        )
    log.info(
        "[OTR_LedgerReviewer:%s] audit complete: %d violation(s), "
        "pass_clean=%s",
        label, len(report.violations), report.pass_clean,
    )
    return report


# ---------------------------------------------------------------------------
# Deterministic cast repair (between Pass 1 and Pass 2)
# ---------------------------------------------------------------------------


# Wiring-review #11 (2026-05-11): structural fields are locked from
# the LLM. Deterministic repair NEVER writes a raw LLM-provided
# `violation.expected` to char_id or speaker_role. For wrong_char_id:
# look up `expected` (LLM-suggested name) in the locked cast_contract
# and use the canonical char_id from that lookup only if exact match.
# For role_mismatch: validate `expected` against the allowed-role
# enum. On any miss, leave the row unrepaired -- Pass 2 Script Doctor
# decides.
_ALLOWED_SPEAKER_ROLES: frozenset[str] = frozenset({
    "character", "announcer",
    "music_open", "music_close", "music_inter",
    "sfx",
})


def apply_deterministic_cast_repairs(
    candidate_ledger: dict,
    pre_audit: PreAuditReport,
    cast_rows: list[dict],
) -> int:
    """Apply Python-only repairs to the candidate ledger.

    Returns the number of violations successfully repaired. Does NOT
    raise on a row lookup miss -- the violation is logged and skipped.

    Sprint 3F (2026-05-25): confidence gating removed. The auditor no
    longer scores `confidence` (small models cannot reliably tell
    0.7 from 0.8 or 0.8 from 0.9), so Python alone decides whether a
    repair is safe to commit. The decision is deterministic:

      * bad_casing / alias_used: the literal-substitution repairs.
        Python verifies `found` actually appears in the line text and
        that `expected` is a real cast member (exact case-fold match
        against the roster, then Levenshtein <= 3 via
        `auto_remap_phantom`). If the auditor's `expected` does not
        resolve to a cast member, the repair is NOT applied -- the row
        falls through to the Script Doctor.
      * wrong_char_id: resolve `expected` (auditor-suggested NAME) to
        a cast member by exact case-fold match, then Levenshtein. On a
        unique resolution, write the canonical char_id. On no match or
        an ambiguous tie, leave the row for the Script Doctor.
      * role_mismatch: validate `expected` against the allowed-role
        enum. On match, write. On miss, leave for the Script Doctor.
        (Roles are a fixed enum, not a fuzzy-matched name space, so
        there is nothing for Levenshtein to do here.)
      * invented_name: `auto_remap_phantom` fuzzy match against the
        full roster; on a miss or ambiguous tie, leave for the Script
        Doctor.
      * speaker_unknown: no-op here; flows into Phase 2 Script Doctor
        as an ordinary violation. S33 B2 (2026-05-15) retired the
        speaker_unknowns rollback gate per refined no-auditors rule.

    Ambiguous ties (two cast members equally close to the auditor's
    `expected` / `found`) escalate exactly as an unresolved violation
    does: the row is left untouched for the Script Doctor. Python
    never silently picks one of a tie.
    """
    cast_names = [
        row.get("name", "") for row in cast_rows
        if row.get("name") and row.get("name") != "ANNOUNCER"
    ]
    full_roster = set(cast_names) | {"ANNOUNCER"}
    # Name -> canonical char_id lookup for the wrong_char_id repair
    # (wiring-review #11). Build once; ANNOUNCER routes to the
    # writer's hardcoded "announcer" cid.
    char_id_by_name: dict[str, str] = {}
    for row in cast_rows or []:
        name = row.get("name") or ""
        cid = row.get("char_id") or ""
        if name and cid:
            char_id_by_name[name] = cid
            char_id_by_name[name.upper()] = cid
    char_id_by_name.setdefault("ANNOUNCER", "announcer")
    repaired = 0
    lines_by_id: dict[str, dict] = {
        ln.get("line_id", ""): ln
        for ln in candidate_ledger.get("lines", []) or []
    }

    for v in pre_audit.violations:
        line = lines_by_id.get(v.line_id)
        if line is None:
            log.warning(
                "[OTR_LedgerReviewer] violation references unknown "
                "line_id=%s; skipping", v.line_id,
            )
            continue
        if v.kind == "speaker_unknown":
            # S33 B2 (2026-05-15): speaker_unknowns rollback gate
            # retired; nothing to patch here either. Phase 2 Script
            # Doctor sees this row as an ordinary violation.
            continue
        if v.kind == "bad_casing":
            # Sprint 3F: no confidence gate. Python resolves the
            # auditor's `expected` to a real cast member (exact
            # case-fold, then Levenshtein <= 3). An unresolved or
            # ambiguous `expected` escalates -- the row is left for
            # the Script Doctor.
            text = line.get("text") or ""
            target = _resolve_cast_member(v.expected, full_roster)
            if target and v.found and v.found in text:
                line["text"] = text.replace(v.found, target)
                repaired += 1
            elif v.found and v.found in text:
                log.warning(
                    "[OTR_LedgerReviewer] bad_casing violation on "
                    "line_id=%s suggested expected=%r did not resolve "
                    "to a cast member; leaving row unrepaired for the "
                    "Script Doctor.",
                    v.line_id, v.expected,
                )
            continue
        if v.kind == "wrong_char_id":
            # Wiring-review #11: NEVER write violation.expected raw.
            # Sprint 3F: resolve `expected` (auditor-suggested NAME)
            # to a real cast member deterministically (exact case-fold
            # then Levenshtein <= 3), then write that member's
            # canonical char_id. On no match or an ambiguous tie,
            # leave the row unchanged -- Pass 2 doctor decides.
            target_name = _resolve_cast_member(v.expected, full_roster)
            cid = None
            if target_name:
                cid = char_id_by_name.get(target_name)
                if cid is None:
                    cid = char_id_by_name.get(target_name.upper())
            if cid:
                line["char_id"] = cid
                repaired += 1
            else:
                log.warning(
                    "[OTR_LedgerReviewer] wrong_char_id violation on "
                    "line_id=%s suggested expected=%r but no cast "
                    "member resolves; leaving row unrepaired for the "
                    "Script Doctor.",
                    v.line_id, v.expected,
                )
            continue
        if v.kind == "role_mismatch":
            # Wiring-review #11: validate `expected` against the
            # allowed-role enum. Reject any other string -- the LLM
            # cannot invent new speaker_role values. Sprint 3F: roles
            # are a fixed enum, not a fuzzy name space, so exact
            # membership is the only deterministic check -- there is
            # nothing for Levenshtein to resolve here.
            expected_role = (v.expected or "").strip()
            if expected_role in _ALLOWED_SPEAKER_ROLES:
                line["speaker_role"] = expected_role
                repaired += 1
            else:
                log.warning(
                    "[OTR_LedgerReviewer] role_mismatch violation on "
                    "line_id=%s suggested expected=%r not in allowed "
                    "roles %r; leaving row unrepaired.",
                    v.line_id, v.expected, sorted(_ALLOWED_SPEAKER_ROLES),
                )
            continue
        if v.kind == "alias_used":
            # Sprint 3F: resolve the auditor's `expected` to a real
            # cast member before substituting. An alias the auditor
            # cannot map to a cast member escalates to the Script
            # Doctor rather than writing an unverified literal.
            text = line.get("text") or ""
            target = _resolve_cast_member(v.expected, full_roster)
            if target and v.found and v.found in text:
                line["text"] = text.replace(v.found, target)
                repaired += 1
            elif v.found and v.found in text:
                log.warning(
                    "[OTR_LedgerReviewer] alias_used violation on "
                    "line_id=%s suggested expected=%r did not resolve "
                    "to a cast member; leaving row unrepaired for the "
                    "Script Doctor.",
                    v.line_id, v.expected,
                )
            continue
        if v.kind == "invented_name":
            remap = auto_remap_phantom(v.found, full_roster)
            if remap is not None:
                text = line.get("text") or ""
                if v.found and v.found in text:
                    line["text"] = text.replace(v.found, remap)
                    repaired += 1
                # else: phantom isn't in the visible text; Pass 2
                # Script Doctor handles. (S33 B4 retired Step 2.5.)
            # On no match: leave phantom in place. Pass 2 Script
            # Doctor owns the next step. (S33 B4 retired Step 2.5.)
            continue
    return repaired


# ---------------------------------------------------------------------------
# Edit cap (per synthesis G1)
# ---------------------------------------------------------------------------


def compute_edit_cap(voiced_beats: int) -> int:
    """edit_cap = min(8, max(3, voiced_beats // 3)).

    Per §3 Phase 3 -- scales with episode size so a 19-beat 7-act
    episode can accommodate ~6 plausible rewrites without flipping
    `too_many_edits`, while a 6-beat 1-act episode caps at 3.
    """
    return min(8, max(3, voiced_beats // 3))


# ---------------------------------------------------------------------------
# Script Doctor (Pass 2)
# ---------------------------------------------------------------------------


_DOCTOR_SYSTEM_PROMPT = """\
You are a script doctor for an audio drama. The cast contract has
already been validated and any cast drift has been deterministically
repaired. Your job is to improve the dialogue: pacing, voice
consistency, arc adherence, beat-intent alignment.

You may propose: rewrite (replace a line's text), skip (mark a line
as muted), annotate (leave a diagnostic note for the user, no text
change).

You MAY NOT propose: inserting new beats, reordering beats,
renumbering line_id / beat_id / shot_id, or any structural change.
The ledger structure is locked.

Output a single JSON ScriptDoctorReport:
{
  "edits": [
    {"line_id": "...", "action": "rewrite|skip|annotate",
     "payload": "...", "rationale": "..."},
    ...
  ],
  "overall_verdict": "clean" | "improved" | "needs_full_rerun"
}

Set overall_verdict to "needs_full_rerun" only if the script is
structurally broken (entire act misses its arc phase, multiple
characters undifferentiated, etc.) -- in that case propose no edits.

No prose outside the JSON. No markdown fences.
"""


def _render_doctor_user_prompt(
    candidate_ledger: dict,
    cast_rows: list[dict],
    titled_phantoms: list[tuple[str, str]],
    edit_cap: int,
) -> str:
    """Build the Script Doctor user prompt.

    `titled_phantoms` is a pre-filled work list of (line_id, phantom)
    tuples the Pass 1 auto-remap couldn't resolve -- the doctor MUST
    rewrite or skip these per the belt-and-braces guarantee.
    """
    meta = candidate_ledger.get("meta") or {}
    budget_lines: list[str] = []
    target_words = meta.get("requested_target_words") or meta.get("target_words")
    if target_words is not None:
        budget_lines.append(f"- target_words: {target_words}")
    style = meta.get("style") or ""
    if style:
        budget_lines.append(f"- style: {style}")
    news_meta = meta.get("news") or {}
    script_brief = news_meta.get("script_brief") or ""
    key_terms = news_meta.get("key_terms") or []
    parts: list[str] = []
    if budget_lines:
        parts.append("EPISODE BUDGET (from meta):")
        parts.extend(budget_lines)
        parts.append("")
    if script_brief:
        parts.append("EPISODE SUMMARY:")
        parts.append(f"- script_brief: {script_brief}")
        if key_terms:
            parts.append(f"- key_terms: {', '.join(key_terms)}")
        parts.append("")
    parts.append("LOCKED CAST:")
    parts.append(_render_cast_contract_table(cast_rows))
    parts.append("")
    if titled_phantoms:
        parts.append(
            "TITLED PHANTOMS (auto-remap could not resolve -- you MUST "
            "rewrite or skip each of these; annotation alone is not "
            "acceptable):"
        )
        for line_id, phantom in titled_phantoms:
            parts.append(f"  {line_id}: {phantom!r}")
        parts.append("")
    # Fix 4 (post-Phase-3 review, 2026-05-11): doctor sees ONLY
    # character-role beats. Announcer / music / sfx beats are locked
    # structural content; showing them tempts the doctor to invent
    # edits the apply-time guard would reject anyway. Belt + suspenders:
    # the prompt filter is suspenders, the apply guard at
    # apply_doctor_edits is the belt. Together: the doctor can't see,
    # can't edit, can't reference structural beats.
    character_lines = [
        ln for ln in (candidate_ledger.get("lines", []) or [])
        if ln.get("speaker_role") == "character"
    ]
    parts.append(
        "LEDGER (CHARACTER DIALOGUE BEATS ONLY -- "
        "post-cast-repair, ready for creative edits):"
    )
    parts.append(_render_lines_for_audit(character_lines))
    parts.append("")
    parts.append(
        "You are seeing ONLY character dialogue beats. Announcer "
        "beats, music beats, and SFX beats are locked structural "
        "content and are NOT in your input. Do not propose edits "
        "referencing line_ids outside the list above."
    )
    parts.append("")
    parts.append(
        f"Propose at most {edit_cap} edits total. Rewrite only when a "
        f"line fails its beat intent. Provide one-sentence rationale "
        f"on each edit."
    )
    return "\n".join(parts)


def _detect_titled_phantoms(
    candidate_ledger: dict,
    cast_roster: set[str],
) -> list[tuple[str, str]]:
    """Return [(line_id, phantom_token), ...] for phantoms present
    in candidate text.

    Post-Phase-3 review Gap 2 follow-up (2026-05-11): migrated from
    the narrow `_TITLED_PHANTOM_RE` to the canonical
    `_otr_line_composer.detect_phantom_names` so the doctor's
    pre-filled work list catches ALL-CAPS phantoms (CARLA) and
    Title-Case bigrams (Joe Smith), not just titled names (Dr.
    Patel). One detector, one roster, everywhere — the name kept
    as `_detect_titled_phantoms` for callsite stability; the
    behavior is the canonical detector's full three-regex pass.
    """
    from ._otr_line_composer import detect_phantom_names  # type: ignore

    found: list[tuple[str, str]] = []
    roster_frozen = frozenset(cast_roster)
    for line in candidate_ledger.get("lines", []) or []:
        text = line.get("text") or ""
        if not text:
            continue
        speaker = line.get("char_id") or line.get("speaker") or ""
        phantoms = detect_phantom_names(text, speaker, roster_frozen)
        for tok in phantoms:
            found.append((line.get("line_id", ""), tok))
    return found


def run_script_doctor(
    generate_fn,
    candidate_ledger: dict,
    cast_rows: list[dict],
    edit_cap: int,
) -> ScriptDoctorReport:
    """One LLM call. Returns ScriptDoctorReport.

    On LLM / JSON / schema failure, returns a report with
    overall_verdict="needs_full_rerun" so the caller can branch
    on the failure (cascade routes to needs_full_rerun verdict;
    caller decides whether to retry the writer or surface the
    failure loud).

    S33 B3 + B4 retired Pass 3 post-audit and Step 2.5 phantom-
    skip fallback. The doctor IS the final structural pass; it
    must therefore fail loud with needs_full_rerun so downstream
    commits don't ship corrupted candidates. S34 B1 corrected the
    prior fail-soft behavior that S33 had assumed was already
    loud (which it wasn't).
    """
    # Wiring-review #7 / #9 (2026-05-11): prefer canonical
    # meta.allowed_roster (cast + ANNOUNCER + key_terms). Fallback
    # to cast-only matches pre-canonical-roster ledgers.
    canonical = candidate_ledger.get("meta", {}).get("allowed_roster") or []
    full_roster_upper: set[str] = {
        str(r).strip().upper() for r in canonical if str(r).strip()
    }
    if not full_roster_upper:
        for row in cast_rows:
            n = row.get("name", "")
            if n:
                full_roster_upper.add(n.upper())
        full_roster_upper.add("ANNOUNCER")
    titled_phantoms = _detect_titled_phantoms(
        candidate_ledger, full_roster_upper,
    )
    user_prompt = _render_doctor_user_prompt(
        candidate_ledger, cast_rows, titled_phantoms, edit_cap,
    )
    messages = [
        {"role": "system", "content": _DOCTOR_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    # Sprint 2A/2D: the single-shot call + parse + validate is now the
    # shared structured_call 3-attempt ladder (base -> structural retry
    # -> typed repair). S34 B1 (2026-05-15) requires this pass to
    # fail loud with needs_full_rerun: an exhausted ladder
    # (StructuredCallFailedError) and a raising slot fn -- which
    # structured_call does not catch -- both map to that verdict,
    # preserving the loud-failure contract S33 B3/B4 depend on.
    # LLM slot: technical -- Script Doctor structural edits
    try:
        report = structured_call(
            prompt=messages,
            schema=ScriptDoctorReport,
            slot_fn=generate_fn,
            base_temperature=_DOCTOR_TEMPERATURE,
            structural_retry_temperature=_DOCTOR_RETRY_TEMPERATURE,
            repair_prompt_factory=make_dispatching_repair_factory(),
            max_new_tokens=_DOCTOR_MAX_NEW_TOKENS,
            max_attempts=3,
            helper_name="run_script_doctor",
        )
    except StructuredCallFailedError as exc:
        log.warning(
            "[OTR_LedgerReviewer:doctor] structured_call exhausted the "
            "retry ladder after %d attempt(s) (last error: %s); "
            "returning needs_full_rerun report",
            exc.attempts, exc.last_error,
        )
        return ScriptDoctorReport(overall_verdict="needs_full_rerun")
    except Exception as exc:  # noqa: BLE001 -- slot fn (LLM call) varies
        log.warning(
            "[OTR_LedgerReviewer:doctor] structured_call raised %s: %s; "
            "returning needs_full_rerun report", type(exc).__name__, exc,
        )
        return ScriptDoctorReport(overall_verdict="needs_full_rerun")
    return report


# Wiring-review (Pass 2 doctor scope guard) per synthesis §3 Phase 3:
# the Script Doctor only sees character-role beats and MUST NOT touch
# announcer / music / sfx beats. Pre-apply guard rejects any edit
# targeting a non-character row -- it indicates the doctor invented
# the line_id or hallucinated structural-scope. Log to
# meta.reviewer_doctor_rejected_edits for forensics.
_ALLOWED_DOCTOR_ROLES: frozenset[str] = frozenset({"character"})


def apply_doctor_edits(
    candidate_ledger: dict,
    report: ScriptDoctorReport,
    *,
    edit_cap: int,
) -> int:
    """Mutate `candidate_ledger` per `report.edits`. Returns the
    number of edits applied.

    Returns -1 if the doctor proposed MORE than `edit_cap` edits --
    caller stamps `too_many_edits` and skips applying anything.

    Wiring-review (2026-05-11): edits targeting a non-character beat
    are REJECTED, stamped on `meta.reviewer_doctor_rejected_edits[]`,
    and DO NOT count toward `applied`. Belt-and-suspenders: when an
    edit is `skip`, also clear `text` to "" so any TTS consumer that
    doesn't honor `skip` still emits nothing.
    """
    if len(report.edits) > edit_cap:
        return -1
    lines_by_id: dict[str, dict] = {
        ln.get("line_id", ""): ln
        for ln in candidate_ledger.get("lines", []) or []
    }
    applied = 0
    rejected: list[dict] = []
    for edit in report.edits:
        line = lines_by_id.get(edit.line_id)
        if line is None:
            log.warning(
                "[OTR_LedgerReviewer:doctor] edit references unknown "
                "line_id=%s; skipping", edit.line_id,
            )
            rejected.append({"line_id": edit.line_id,
                             "reason": "unknown_line_id"})
            continue
        # Scope guard: doctor MAY ONLY edit character beats.
        role = line.get("speaker_role") or ""
        if role not in _ALLOWED_DOCTOR_ROLES:
            log.warning(
                "[OTR_LedgerReviewer:doctor] edit targets non-character "
                "beat (line_id=%s, role=%r) -- REJECTED.",
                edit.line_id, role,
            )
            rejected.append({
                "line_id": edit.line_id,
                "reason": f"doctor_targeted_non_character_beat(role={role!r})",
            })
            continue
        if edit.action == "rewrite":
            # BUG-LOCAL-267: route the doctor's rewrite through the
            # composer's format strip. This branch previously wrote
            # `edit.payload` verbatim, so a doctor LLM that emitted a
            # leading "SPEAKER:" prefix (e.g. "HAYES VANCE: ...")
            # re-injected the speaker label the composer had already
            # stripped at compose time -- and Bark then voiced the
            # character's name aloud. Same strip the composer applies
            # to its own output; consistency, no behaviour change for
            # a clean payload.
            try:
                from ._otr_line_composer import strip_line_formatting
                rewritten = strip_line_formatting(edit.payload or "")
            except Exception as exc:  # noqa: BLE001
                log.warning(
                    "[OTR_LedgerReviewer:doctor] strip_line_formatting "
                    "unavailable (%s); applying rewrite payload raw", exc,
                )
                rewritten = edit.payload or ""
            line["text"] = rewritten
            # Recompute counts in lockstep with the stripped text.
            line["char_count"] = len(rewritten)
            line["word_count"] = len(re.findall(
                r"[A-Za-z][A-Za-z0-9'\-]*", rewritten,
            ))
        elif edit.action == "skip":
            line["skip"] = True
            line["reviewer_skip_reason"] = edit.payload or "skip"
            # Wiring-review #14 belt-and-suspenders.
            line["text"] = ""
            line["char_count"] = 0
            line["word_count"] = 0
        elif edit.action == "annotate":
            line["reviewer_note"] = edit.payload
        applied += 1
    if rejected:
        meta = candidate_ledger.setdefault("meta", {})
        meta["reviewer_doctor_rejected_edits"] = rejected
    return applied


# ---------------------------------------------------------------------------
# Step 2.5 phantom-skip fallback + final phantom check
# ---------------------------------------------------------------------------
#
# S33 B4 (2026-05-15): both `apply_phantom_skip_fallback` and
# `_final_phantom_check` retired per B1.5 classification.
#
# * `apply_phantom_skip_fallback` mutated `line["skip"] = True` +
#   cleared text -- a mute, not a story edit. Skipping a line is a
#   pipeline cut under the refined no-auditors rule.
# * `_final_phantom_check` was pure report-only; its only consumer
#   was the `post_audit_pass` rollback gate retired in S33 B2.
#
# Per Jeffrey's phantom-ship policy (2026-05-15), occasional
# phantoms reaching the audience is the accepted trade-off.
# `apply_deterministic_cast_repairs` + Phase 2 Script Doctor (both
# editors) still rewrite phantom name violations into real cast
# names; only the post-Phase-2 mute / report layer is gone.


# ---------------------------------------------------------------------------
# review_ledger -- top-level entrypoint
# ---------------------------------------------------------------------------


def _stamp_word_counts_safe(led) -> None:
    """Lazy wrapper around production_ledger.stamp_word_counts.

    Lazy import so this module stays loadable without
    production_ledger present (matches the rest of the module's
    deferred-import pattern). Best-effort: any failure logs and
    skips -- a stamping miss must never break the reviewer's
    commit/restore flow.
    """
    try:
        from . import production_ledger as _PL  # type: ignore
        _PL.stamp_word_counts(led)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_LedgerReviewer] stamp_word_counts skipped: %s", exc,
        )


def review_ledger(
    generate_fn,
    led,
) -> ReviewerDisposition:
    """Run the three-pass cast-gated reviewer on `led`.

    `led` is a production_ledger.Ledger (or any object exposing
    `.data` dict + `.save()` method).

    On success the ORIGINAL led is mutated in place to reflect the
    Script Doctor's edits (post-audit + final phantom check
    cleared); led.save() persists. On any failure path the original
    ledger fields are restored before returning, and `led.save()` is
    a no-op visually (the bytes match the pre-review state).

    Stamps `meta.reviewer_verdict` (one of the ReviewerVerdict
    literals) and `meta.reviewer_disposition` on `led.data`.

    Returns the ReviewerDisposition for the caller's forensic log.
    """
    ledger_data: dict = led.data
    meta: dict = ledger_data.setdefault("meta", {})

    # Programmatic bypass per synthesis G9. Production runs never set
    # this; tests of OTR_LedgerScriptWriter that don't want to pay
    # 3 LLM calls per test can set it directly.
    if meta.get("skip_reviewer"):
        meta["reviewer_verdict"] = "clean_no_edits"
        meta["reviewer_disposition"] = {
            "verdict": "clean_no_edits",
            "skipped": True,
            "skipped_reason": "meta.skip_reviewer=True",
        }
        log.info("[OTR_LedgerReviewer] skip_reviewer flag set; no-op")
        return ReviewerDisposition(
            verdict="clean_no_edits",
            pre_audit_violations=0,
            pre_audit_repairs_applied=0,
            doctor_edits_proposed=0,
            doctor_edits_applied=0,
            post_audit_violations=0,
        )

    cast_rows = list(ledger_data.get("cast") or [])
    # S33 B4 (2026-05-15): the `cast_roster_upper` set construction
    # block was the input to `apply_phantom_skip_fallback` +
    # `_final_phantom_check`, both retired this commit. Dead code
    # removed. Phase 1's audit prompt still consults the cast contract
    # via `_render_cast_contract_table(cast_rows)`; no production
    # behavior depends on the upper-case roster set.

    voiced_beats = sum(
        1 for ln in ledger_data.get("lines", []) or []
        if (ln.get("speaker_role") or "") in ("character", "announcer")
    )
    edit_cap = compute_edit_cap(voiced_beats)

    # Snapshot the ledger so we can restore on any failure path.
    import copy
    original_snapshot = copy.deepcopy(ledger_data)
    candidate = copy.deepcopy(ledger_data)

    # ---- Pass 1: Cast Auditor pre-check -------------------
    pre_audit = audit_cast_contract(generate_fn, candidate, label="pre")
    pre_audit_violations = len(pre_audit.violations)

    # Wiring-review #8 (2026-05-11): if the auditor LLM call itself
    # failed (parse / schema / generation exception), the sentinel
    # returns audit_failed=True. Map to needs_full_rerun; do not run
    # the doctor on garbage data.
    if getattr(pre_audit, "audit_failed", False):
        led.data.clear()
        led.data.update(original_snapshot)
        meta_after = led.data.setdefault("meta", {})
        meta_after["reviewer_verdict"] = "needs_full_rerun"
        meta_after["reviewer_audit_failure_reason"] = getattr(
            pre_audit, "audit_failure_reason", "unknown",
        )
        disp = ReviewerDisposition(
            verdict="needs_full_rerun",
            pre_audit_violations=pre_audit_violations,
            pre_audit_repairs_applied=0,
            doctor_edits_proposed=0,
            doctor_edits_applied=0,
            post_audit_violations=0,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
        # Fix 3 (2026-05-11): re-stamp §6.G word counts after every
        # commit / restore so meta.character_word_count stays in sync
        # with whatever is on disk.
        _stamp_word_counts_safe(led)
        led.save()
        return disp

    # S33 B2 (2026-05-15): `speaker_unknowns` rollback gate retired.
    # Per refined no-auditors rule, gates that cut the pipeline (halt,
    # rollback, report-only) are forbidden -- only audit calls that
    # feed editors survive. The deterministic cast repairs below
    # still consume `pre_audit.violations` to develop the story;
    # high-confidence speaker_unknown rows now flow into Phase 2
    # Script Doctor as ordinary violations rather than triggering an
    # early `cast_unrecoverable` rollback.

    # ---- Python deterministic repairs (between Pass 1 and Pass 2) ----
    repairs_applied = apply_deterministic_cast_repairs(
        candidate, pre_audit, cast_rows,
    )

    # ---- Pass 2: Script Doctor -------------------------------
    doctor_report = run_script_doctor(
        generate_fn, candidate, cast_rows, edit_cap,
    )
    doctor_edits_proposed = len(doctor_report.edits)

    if doctor_report.overall_verdict == "needs_full_rerun":
        led.data.clear()
        led.data.update(original_snapshot)
        meta_after = led.data.setdefault("meta", {})
        meta_after["reviewer_verdict"] = "needs_full_rerun"
        disp = ReviewerDisposition(
            verdict="needs_full_rerun",
            pre_audit_violations=pre_audit_violations,
            pre_audit_repairs_applied=repairs_applied,
            doctor_edits_proposed=doctor_edits_proposed,
            doctor_edits_applied=0,
            post_audit_violations=0,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
        # Fix 3 (2026-05-11): re-stamp §6.G word counts after every
        # commit / restore so meta.character_word_count stays in sync
        # with whatever is on disk.
        _stamp_word_counts_safe(led)
        led.save()
        return disp

    edits_applied = apply_doctor_edits(
        candidate, doctor_report, edit_cap=edit_cap,
    )
    if edits_applied == -1:
        led.data.clear()
        led.data.update(original_snapshot)
        meta_after = led.data.setdefault("meta", {})
        meta_after["reviewer_verdict"] = "too_many_edits"
        disp = ReviewerDisposition(
            verdict="too_many_edits",
            pre_audit_violations=pre_audit_violations,
            pre_audit_repairs_applied=repairs_applied,
            doctor_edits_proposed=doctor_edits_proposed,
            doctor_edits_applied=0,
            post_audit_violations=0,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
        # Fix 3 (2026-05-11): re-stamp §6.G word counts after every
        # commit / restore so meta.character_word_count stays in sync
        # with whatever is on disk.
        _stamp_word_counts_safe(led)
        led.save()
        return disp

    # S33 B4 (2026-05-15): Step 2.5 phantom-skip fallback retired
    # together with `apply_phantom_skip_fallback`. Setting `skip=True`
    # on a line is a pipeline cut (mute), not a story edit, so the
    # refined no-auditors rule forbids it. Per Jeffrey's phantom-ship
    # policy, occasional phantoms reaching the audience is the
    # accepted trade-off.

    # S33 B3 (2026-05-15): Phase 9 `audit_cast_contract(label="post")`
    # call retired. Its only consumer was the `post_audit_pass`
    # rollback gate, which B2 already retired. With the gate gone the
    # post-edit LLM audit had no editor consumer -- it was a pure
    # pipeline cut. Per Jeffrey's phantom-ship policy (2026-05-15),
    # occasional phantoms reaching the audience is the accepted
    # trade-off. The shared `audit_cast_contract` function is still
    # called once (label="pre", Phase 1) -- the function survives
    # because its output feeds `apply_deterministic_cast_repairs`
    # (an editor).

    # ---- Commit candidate to disk ---------------------------
    # Wiring-review #10: clear+update (NOT bare shallow update -- bare
    # update leaves any keys removed by the doctor still on led.data,
    # which would be a state leak.)
    led.data.clear()
    led.data.update(candidate)
    meta_after = led.data.setdefault("meta", {})
    verdict_str = "improved" if edits_applied > 0 else "clean_no_edits"
    meta_after["reviewer_verdict"] = verdict_str
    disp = ReviewerDisposition(
        verdict=verdict_str,
        pre_audit_violations=pre_audit_violations,
        pre_audit_repairs_applied=repairs_applied,
        doctor_edits_proposed=doctor_edits_proposed,
        doctor_edits_applied=edits_applied,
        post_audit_violations=0,
    )
    meta_after["reviewer_disposition"] = disp.__dict__
    # Fix 3 (2026-05-11): §6.G word counts re-stamped on the commit
    # path too so post-review meta.character_word_count reflects any
    # rewrites / skips the doctor applied.
    _stamp_word_counts_safe(led)
    led.save()
    return disp
