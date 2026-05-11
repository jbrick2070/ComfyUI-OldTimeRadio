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

Step 2.5 (Deterministic phantom-skip fallback)
    Catches titled phantoms ("Dr. Patel") that survived Pass 1's
    Levenshtein auto-remap AND the Script Doctor. Sets line.skip=True
    + tts_skip_reason. Pure Python; no LLM.

Pass 3 (Cast Auditor post-check) -- LLM #3 (same function, different
        label)
    Confirms the Script Doctor's patch didn't reintroduce cast drift
    on the candidate. Hard reject -> keep original on disk.

Final disposition stamped on meta.reviewer_verdict. Original ledger
preserved exactly on every failure path.

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
    "apply_phantom_skip_fallback",
    "compute_edit_cap",
    "review_ledger",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# Pinned reviewer verdicts. Imported everywhere meta.reviewer_verdict
# is read or written (per synthesis G7).
ReviewerVerdict = Literal[
    "clean_no_edits",
    "improved",
    "cast_unrecoverable",
    "too_many_edits",
    "needs_full_rerun",
    "post_audit_failed",
]


# Levenshtein threshold for invented-name auto-remap (per §6.A).
# Names ≥ 5 chars allow distance ≤ 3; shorter names use a tighter cap.
# Substring-containment is a secondary fast path (case-insensitive).
_LEVENSHTEIN_THRESHOLD = 3


# Titled-phantom regex for Step 2.5 deterministic skip-fallback.
_TITLED_PHANTOM_RE = re.compile(
    r"\b(Dr|Mr|Ms|Mrs|Prof|Lt|Capt|Cmdr|Adm|Sen|Sgt|Col|Gen)"
    r"\.\s+[A-Z]\w+"
)


# Generation params for the three LLM calls (kept conservative; the
# auditor + doctor benefit from low-temp determinism, not creativity).
_AUDIT_TEMPERATURE = 0.2
_AUDIT_MAX_NEW_TOKENS = 2000
_DOCTOR_TEMPERATURE = 0.5
_DOCTOR_MAX_NEW_TOKENS = 3500


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class CastViolation(BaseModel):
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
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class PreAuditReport(BaseModel):
    violations: list[CastViolation] = Field(default_factory=list)
    pass_clean: bool = True
    # Wiring-review #8 (2026-05-11): audit fail-loud. When the
    # auditor LLM call raises, returns garbage that doesn't parse,
    # or returns JSON that doesn't validate, `audit_cast_contract`
    # constructs a SENTINEL report with audit_failed=True +
    # pass_clean=False so the caller can branch to needs_full_rerun
    # / post_audit_failed. The default pass_clean=True is reserved
    # for "LLM ran cleanly and reported no violations" -- never for
    # "audit didn't run". Pure pydantic field-only changes; no
    # downstream rewrite needed beyond branching on these fields.
    audit_failed: bool = False
    audit_failure_reason: str = ""


def _audit_failed_sentinel(reason: str) -> "PreAuditReport":
    """Synthetic PreAuditReport stamped when the auditor LLM path
    failed. pass_clean=False so the caller's `if not pass_clean`
    branch fires and the verdict maps to needs_full_rerun (pre) or
    post_audit_failed (post)."""
    return PreAuditReport(
        violations=[CastViolation(
            line_id="__audit__",
            kind="invented_name",     # closest pinned kind; audit_failed
                                      # surfaces via audit_failed flag
            found=reason[:200],
            expected="",
            confidence=1.0,
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

    verdict: str  # ReviewerVerdict literal
    pre_audit_violations: int
    pre_audit_repairs_applied: int
    doctor_edits_proposed: int
    doctor_edits_applied: int
    post_audit_violations: int
    phantom_skip_count: int


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
    Doctor's rewrite/skip step or the deterministic phantom-skip
    fallback (Step 2.5).
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
only flag violations.

VIOLATION TYPES:
- bad_casing: speaker is the right name but cased wrong (e.g. "alice"
  when the cast says "ALICE"). High confidence.
- alias_used: line uses an alias (e.g. "AL") that exists in cast_rows
  but the canonical name should appear. Medium confidence.
- invented_name: dialogue text references a proper noun that is
  neither in cast_contract nor in meta.key_terms.
- wrong_char_id: lines[k].char_id does not correspond to the speaker
  name in cast_contract.
- role_mismatch: lines[k].speaker_role does not match the speaker's
  role in cast_contract.
- speaker_unknown: speaker field is not in cast_contract at all.

For every violation you find, output one CastViolation object. Set
confidence based on how certain you are that the violation is real
(1.0 = certain; 0.5 = ambiguous; below 0.3 = do not report).

Return EXACTLY one JSON object matching this schema:
{
  "violations": [
    {"line_id": "...", "kind": "...", "found": "...", "expected": "...", "confidence": 0.0-1.0},
    ...
  ],
  "pass_clean": true|false
}

No prose before or after the JSON. No markdown fences.
"""


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL | re.IGNORECASE)


def _extract_json_block(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip()
    m = _FENCE_RE.search(s)
    if m:
        return m.group(1).strip()
    first = s.find("{")
    last = s.rfind("}")
    if first != -1 and last != -1 and last > first:
        return s[first:last + 1]
    return s


def audit_cast_contract(
    generate_fn,
    ledger_data: dict,
    label: str,
) -> PreAuditReport:
    """Run a single Cast Contract Auditor LLM pass.

    `label` distinguishes Pass 1 (label="pre") from Pass 3
    (label="post") in soak logs. Per synthesis G4 this is ONE
    function called twice; the underlying prompt + schema are
    identical.

    Returns a PreAuditReport. On LLM failure or unparseable JSON,
    returns an empty clean report and logs a warning -- the
    Python deterministic-repair pass then runs on whatever
    violations were found (i.e. none). This is fail-soft on the
    auditor itself; the downstream Python guards still catch
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
    # NOT the bare default PreAuditReport(). The caller maps the
    # failure to needs_full_rerun / post_audit_failed and skips
    # downstream LLM calls.
    try:
        raw = generate_fn(
            messages,
            temperature=_AUDIT_TEMPERATURE,
            max_new_tokens=_AUDIT_MAX_NEW_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_LedgerReviewer:%s] generate_fn raised: %s; "
            "returning audit_failed sentinel", label, exc,
        )
        return _audit_failed_sentinel(
            f"generate_fn raised: {type(exc).__name__}: {exc}"
        )
    json_str = _extract_json_block(raw or "")
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.warning(
            "[OTR_LedgerReviewer:%s] JSON parse failed (%s); "
            "raw=%r; returning audit_failed sentinel",
            label, exc, (raw or "")[:200],
        )
        return _audit_failed_sentinel(f"json.JSONDecodeError: {exc}")
    try:
        report = PreAuditReport.model_validate(data)
    except ValidationError as exc:
        log.warning(
            "[OTR_LedgerReviewer:%s] schema validation failed (%s); "
            "returning audit_failed sentinel", label, exc,
        )
        return _audit_failed_sentinel(f"ValidationError: {exc}")
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

    Per synthesis §3 Phase 3 -- repair table (wiring-review #11
    tightened):
        bad_casing (conf >= 0.8)    replace literal in text
        wrong_char_id (conf >= 0.9) LOOK UP `expected` (LLM name)
                                    in cast_contract; on match,
                                    write the canonical char_id.
                                    On miss, leave for Script Doctor.
        role_mismatch (conf >= 0.9) Validate `expected` against the
                                    allowed-role enum. On match,
                                    write. On miss, leave for Script
                                    Doctor.
        alias_used (any conf)       substitute alias -> canonical
        invented_name (any conf)    auto_remap fuzzy match; on miss,
                                    leave for Script Doctor
        speaker_unknown (any conf)  STOP; caller stamps
                                    cast_unrecoverable verdict
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
            # Caller decides outcome (cast_unrecoverable). Do not
            # patch here.
            continue
        if v.kind == "bad_casing" and v.confidence >= 0.8:
            text = line.get("text") or ""
            if v.found and v.expected and v.found in text:
                line["text"] = text.replace(v.found, v.expected)
                repaired += 1
            continue
        if v.kind == "wrong_char_id" and v.confidence >= 0.9:
            # Wiring-review #11: NEVER write violation.expected raw.
            # Look it up as a NAME in cast_contract and write the
            # canonical char_id from the lookup. On miss, leave the
            # row unchanged -- Pass 2 doctor decides.
            expected_name = (v.expected or "").strip()
            cid = char_id_by_name.get(expected_name)
            if cid is None:
                cid = char_id_by_name.get(expected_name.upper())
            if cid:
                line["char_id"] = cid
                repaired += 1
            else:
                log.warning(
                    "[OTR_LedgerReviewer] wrong_char_id violation on "
                    "line_id=%s suggested expected=%r but no cast "
                    "member matches; leaving row unrepaired for the "
                    "Script Doctor.",
                    v.line_id, v.expected,
                )
            continue
        if v.kind == "role_mismatch" and v.confidence >= 0.9:
            # Wiring-review #11: validate `expected` against the
            # allowed-role enum. Reject any other string -- the LLM
            # cannot invent new speaker_role values.
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
            text = line.get("text") or ""
            if v.found and v.expected and v.found in text:
                line["text"] = text.replace(v.found, v.expected)
                repaired += 1
            continue
        if v.kind == "invented_name":
            remap = auto_remap_phantom(v.found, full_roster)
            if remap is not None:
                text = line.get("text") or ""
                if v.found and v.found in text:
                    line["text"] = text.replace(v.found, remap)
                    repaired += 1
                # else: phantom isn't in the visible text; let Pass 2
                # / Step 2.5 handle.
            # On no match: leave phantom in place. Pass 2 / Step 2.5
            # owns the next step.
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
    parts.append("LEDGER (post-cast-repair, ready for creative edits):")
    parts.append(_render_lines_for_audit(candidate_ledger.get("lines", []) or []))
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
    """Return [(line_id, phantom_token), ...] for titled phantoms
    present in candidate text. Cast roster is the UPPERCASE set."""
    found: list[tuple[str, str]] = []
    for line in candidate_ledger.get("lines", []) or []:
        text = line.get("text") or ""
        if not text:
            continue
        for m in _TITLED_PHANTOM_RE.finditer(text):
            tok = m.group(0)
            if tok.upper() in cast_roster:
                continue
            found.append((line.get("line_id", ""), tok))
    return found


def run_script_doctor(
    generate_fn,
    candidate_ledger: dict,
    cast_rows: list[dict],
    edit_cap: int,
) -> ScriptDoctorReport:
    """One LLM call. Returns ScriptDoctorReport.

    On any LLM / JSON / schema failure, returns a clean report with
    overall_verdict="clean" so the caller commits the post-repair
    candidate without further edits. This is fail-soft on the doctor;
    Pass 3 post-audit and Step 2.5 phantom-skip are the structural
    guarantees.
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
    try:
        raw = generate_fn(
            messages,
            temperature=_DOCTOR_TEMPERATURE,
            max_new_tokens=_DOCTOR_MAX_NEW_TOKENS,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_LedgerReviewer:doctor] generate_fn raised: %s; "
            "returning clean no-op report", exc,
        )
        return ScriptDoctorReport()
    json_str = _extract_json_block(raw or "")
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        log.warning(
            "[OTR_LedgerReviewer:doctor] JSON parse failed (%s); "
            "raw=%r; returning clean no-op report",
            exc, (raw or "")[:200],
        )
        return ScriptDoctorReport()
    try:
        report = ScriptDoctorReport.model_validate(data)
    except ValidationError as exc:
        log.warning(
            "[OTR_LedgerReviewer:doctor] schema validation failed (%s); "
            "returning clean no-op report", exc,
        )
        return ScriptDoctorReport()
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
            line["text"] = edit.payload
            # Recompute counts in lockstep with the new text.
            line["char_count"] = len(edit.payload)
            line["word_count"] = len(re.findall(
                r"[A-Za-z][A-Za-z0-9'\-]*", edit.payload,
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
# Step 2.5 -- deterministic phantom-skip fallback
# ---------------------------------------------------------------------------


def apply_phantom_skip_fallback(
    candidate_ledger: dict,
    cast_roster_upper: set[str],
) -> int:
    """Mute any line still carrying a titled phantom after Pass 2.

    Returns the number of lines newly marked skip=True. Sets
    `tts_skip_reason` for forensic logging. Per synthesis Step 2.5 +
    M2.

    Wiring-review #14 (2026-05-11): when setting skip=True, ALSO
    clear `text` to "" (belt-and-suspenders). A muted line therefore
    has BOTH skip=True AND text=="" — either alone is sufficient to
    mute the line; both together guarantee that any consumer that
    fails to honor one signal still emits nothing.
    """
    n_skipped = 0
    for line in candidate_ledger.get("lines", []) or []:
        if line.get("skip"):
            continue
        text = line.get("text") or ""
        if not text:
            continue
        for m in _TITLED_PHANTOM_RE.finditer(text):
            full = m.group(0)
            if full.upper() in cast_roster_upper:
                continue
            line["skip"] = True
            line["tts_skip_reason"] = f"phantom_titled_name:{full}"
            # Belt-and-suspenders: every TTS consumer that checks
            # text==""  emits nothing for this line.
            line["text"] = ""
            line["char_count"] = 0
            line["word_count"] = 0
            n_skipped += 1
            break
    return n_skipped


# ---------------------------------------------------------------------------
# Final python validation (post-audit safety net)
# ---------------------------------------------------------------------------


def _final_phantom_check(
    candidate_ledger: dict,
    cast_roster_upper: set[str],
) -> list[tuple[str, str]]:
    """Return [(line_id, phantom_token), ...] for any titled phantom
    still present in NON-skipped lines."""
    out: list[tuple[str, str]] = []
    for line in candidate_ledger.get("lines", []) or []:
        if line.get("skip"):
            continue
        text = line.get("text") or ""
        for m in _TITLED_PHANTOM_RE.finditer(text):
            tok = m.group(0)
            if tok.upper() in cast_roster_upper:
                continue
            out.append((line.get("line_id", ""), tok))
    return out


# ---------------------------------------------------------------------------
# review_ledger -- top-level entrypoint
# ---------------------------------------------------------------------------


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
            phantom_skip_count=0,
        )

    cast_rows = list(ledger_data.get("cast") or [])
    # Wiring-review #7 / #9 (2026-05-11): prefer the canonical
    # meta.allowed_roster (writer-stamped, includes cast + ANNOUNCER
    # + key_terms). Fall back to cast-only when the field is absent
    # (e.g. ledger built by a pre-Phase-0 writer). Key_terms inclusion
    # matters: it stops Step 2.5 phantom-skip from incorrectly muting
    # lines that legitimately mention "Dr. Hawking" when Hawking is
    # in news_interpreter's key_terms.
    canonical_roster_raw = ledger_data.get("meta", {}).get("allowed_roster") or []
    cast_roster_upper: set[str] = {
        str(r).strip().upper() for r in canonical_roster_raw if str(r).strip()
    }
    if not cast_roster_upper:
        for row in cast_rows:
            n = row.get("name", "")
            if n:
                cast_roster_upper.add(n.upper())
        cast_roster_upper.add("ANNOUNCER")

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
            phantom_skip_count=0,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
        led.save()
        return disp

    # If any speaker_unknown shows up at high confidence, that's
    # unrecoverable: stop, restore original ledger, return.
    speaker_unknowns = [
        v for v in pre_audit.violations
        if v.kind == "speaker_unknown" and v.confidence >= 0.7
    ]
    if speaker_unknowns:
        # Wiring-review #10: clear+update (NOT bare shallow update --
        # bare update leaks any keys candidate added since the snapshot).
        led.data.clear()
        led.data.update(original_snapshot)
        meta_after = led.data.setdefault("meta", {})
        meta_after["reviewer_verdict"] = "cast_unrecoverable"
        disp = ReviewerDisposition(
            verdict="cast_unrecoverable",
            pre_audit_violations=pre_audit_violations,
            pre_audit_repairs_applied=0,
            doctor_edits_proposed=0,
            doctor_edits_applied=0,
            post_audit_violations=0,
            phantom_skip_count=0,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
        led.save()
        return disp

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
            phantom_skip_count=0,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
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
            phantom_skip_count=0,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
        led.save()
        return disp

    # ---- Step 2.5: Deterministic phantom-skip fallback -------
    phantom_skip_count = apply_phantom_skip_fallback(
        candidate, cast_roster_upper,
    )

    # ---- Pass 3: Cast Auditor post-check ---------------------
    post_audit = audit_cast_contract(generate_fn, candidate, label="post")
    final_phantoms = _final_phantom_check(candidate, cast_roster_upper)

    post_audit_pass = (
        post_audit.pass_clean
        and not post_audit.violations
        and not final_phantoms
    )
    # Wiring-review #8: if Pass 3 itself failed (audit_failed sentinel),
    # the violations list contains a synthetic entry. Either way (real
    # violations OR audit_failed), the patch is rejected and the
    # original ledger is restored.
    if not post_audit_pass:
        led.data.clear()
        led.data.update(original_snapshot)
        meta_after = led.data.setdefault("meta", {})
        meta_after["reviewer_verdict"] = "post_audit_failed"
        if getattr(post_audit, "audit_failed", False):
            meta_after["reviewer_audit_failure_reason"] = getattr(
                post_audit, "audit_failure_reason", "unknown",
            )
        disp = ReviewerDisposition(
            verdict="post_audit_failed",
            pre_audit_violations=pre_audit_violations,
            pre_audit_repairs_applied=repairs_applied,
            doctor_edits_proposed=doctor_edits_proposed,
            doctor_edits_applied=edits_applied,
            post_audit_violations=len(post_audit.violations) + len(final_phantoms),
            phantom_skip_count=phantom_skip_count,
        )
        meta_after["reviewer_disposition"] = disp.__dict__
        led.save()
        return disp

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
        phantom_skip_count=phantom_skip_count,
    )
    meta_after["reviewer_disposition"] = disp.__dict__
    led.save()
    return disp
