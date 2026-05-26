"""Sprint 10A step 4 -- Stage 1 cast audit.

Deterministic name -> gender -> pronouns -> voice_id validator that
runs AFTER a Stage 1 plan parses cleanly. Catches the gender-
inversion class of bug the writer pipeline has been producing
intermittently:

  * pending_20260525_223109: COLE=female, MIRA=male
  * pending_20260526_053338: OSCAR=female
  * pending_20260526_114111: ROBINSON VOSS=female (operator-flagged)

The audit produces a structured `CastAuditResult` with categorized
findings. Step 4 is MEASUREMENT only -- the audit findings are
stamped on `meta.stage1_cast_audit` for the soak gate (0 mismatches
across 10 runs). Later steps decide whether to REPAIR the plan in
place or REGENERATE; this module does neither.

Checks (per cast member):
  1. name_gender_mismatch -- canonical name lookup conflicts with
     LLM-emitted gender. ROBINSON=female is the canonical case.
  2. gender_pronouns_mismatch -- LLM emitted gender/pronouns combo
     that doesn't match the canonical mapping (male->he/him,
     female->she/her, nonbinary->they/them).
  3. gender_voice_mismatch -- voice_id's canonical gender (from
     config/cast_pools.VOICE_PROFILES) doesn't match the LLM's
     emitted gender.
  4. voice_unknown -- voice_id not in the Bark v2 pool (rare; the
     Stage 1 schema's regex catches most of these at parse time).
  5. name_unknown_soft -- name not in the curated NAME_GENDER pool
     AND not a known-unknown ("unknown" canonical). Soft signal --
     does NOT count as an error finding (LLM may legitimately
     invent novel names).

Module is PURE: no I/O, no LLM, no transformers. Stage 1 cast audit
runs in milliseconds; no model load required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ._otr_name_gender import (
    GENDER_LITERAL,
    NAME_GENDER,
    is_inversion,
    lookup_first_name_gender,
)
from ._otr_stage1_plan import Stage1Plan


# ---------------------------------------------------------------------------
# Pronoun + voice mapping tables
# ---------------------------------------------------------------------------

# Canonical gender -> pronouns mapping. Stage1Plan's Literal pins both
# fields, so this map covers the entire output space.
CANONICAL_PRONOUNS: dict[str, str] = {
    "male":      "he/him",
    "female":    "she/her",
    "nonbinary": "they/them",
}


def _build_voice_gender_lookup() -> dict[str, str]:
    """Build voice_id -> canonical gender lookup from cast_pools.

    Imported lazily so the audit module stays importable even if the
    config package layout shifts (the project has had pure-Python /
    sandbox path variants over time).
    """
    try:
        from config import cast_pools as _POOLS
    except ImportError:
        try:
            from ..config import cast_pools as _POOLS  # type: ignore
        except ImportError:
            # In sandbox / standalone test environments the config
            # package may not be importable. Fall back to the static
            # mapping we know is current (kept in sync with
            # config/cast_pools.VOICE_PROFILES). The fallback is the
            # SAME data, hard-coded so the audit still runs.
            return _FALLBACK_VOICE_GENDER.copy()
    return {
        preset: gender
        for preset, gender, _lang, _tags in _POOLS.VOICE_PROFILES
    }


# Hard-coded mirror of config/cast_pools.VOICE_PROFILES gender column.
# Used only when the config package can't be imported (sandbox / test).
# Test below pins these in sync with the live config.
_FALLBACK_VOICE_GENDER: dict[str, str] = {
    "v2/en_speaker_0": "male",
    "v2/en_speaker_1": "male",
    "v2/en_speaker_2": "female",
    "v2/en_speaker_3": "male",
    "v2/en_speaker_4": "female",
    "v2/en_speaker_5": "male",
    "v2/en_speaker_6": "male",
    "v2/en_speaker_7": "female",
    "v2/en_speaker_8": "male",
    "v2/en_speaker_9": "female",
}


# ---------------------------------------------------------------------------
# Finding + result types
# ---------------------------------------------------------------------------


# Finding severity: 'error' counts toward the 0-mismatch soak gate;
# 'warn' is informational only and does NOT count.
_SEVERITY_ERROR: str = "error"
_SEVERITY_WARN: str = "warn"

# Finding codes (stable strings -- consumers grep these)
CODE_NAME_GENDER_MISMATCH: str = "name_gender_mismatch"
CODE_GENDER_PRONOUNS_MISMATCH: str = "gender_pronouns_mismatch"
CODE_GENDER_VOICE_MISMATCH: str = "gender_voice_mismatch"
CODE_VOICE_UNKNOWN: str = "voice_unknown"
CODE_NAME_UNKNOWN_SOFT: str = "name_unknown_soft"


@dataclass
class CastAuditFinding:
    """One audit finding on one cast member."""

    severity: str           # "error" | "warn"
    code: str               # one of the CODE_* constants above
    char_name: str          # cast member name as emitted by the LLM
    message: str            # human-readable description
    # Optional diagnostic context for soak forensics.
    expected: Optional[str] = None
    got: Optional[str] = None


@dataclass
class CastAuditResult:
    """Aggregate result for one Stage 1 plan's cast list."""

    findings: List[CastAuditFinding] = field(default_factory=list)

    @property
    def errors(self) -> List[CastAuditFinding]:
        return [f for f in self.findings if f.severity == _SEVERITY_ERROR]

    @property
    def warns(self) -> List[CastAuditFinding]:
        return [f for f in self.findings if f.severity == _SEVERITY_WARN]

    @property
    def ok(self) -> bool:
        """True iff no error-level findings.

        Soak gate (0 mismatches across 10 runs) reads this property.
        Warn-level findings (e.g. name_unknown_soft) do NOT flip ok.
        """
        return len(self.errors) == 0

    def to_dict(self) -> dict:
        """Serialize for meta.stage1_cast_audit ledger stamp."""
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warn_count": len(self.warns),
            "findings": [
                {
                    "severity": f.severity,
                    "code": f.code,
                    "char_name": f.char_name,
                    "message": f.message,
                    "expected": f.expected,
                    "got": f.got,
                }
                for f in self.findings
            ],
        }


# ---------------------------------------------------------------------------
# Audit entry point
# ---------------------------------------------------------------------------


def audit_cast(plan: Stage1Plan) -> CastAuditResult:
    """Run the deterministic name/gender/pronouns/voice audit.

    Per-cast-member checks (see module docstring for full list):
        1. name -> gender match
        2. gender -> pronouns match
        3. gender -> voice_id gender match
        4. voice_id is in the Bark v2 pool
        5. name is in the curated NAME_GENDER pool (soft warn only)

    Returns:
        CastAuditResult with all findings. result.ok is True iff
        no error-level findings landed; the soak gate keys off this.

    Pure function; no I/O.
    """
    result = CastAuditResult()
    voice_gender = _build_voice_gender_lookup()

    for cm in plan.cast:
        name = cm.name
        canonical = lookup_first_name_gender(name)

        # ---- (1) name -> gender ----
        if is_inversion(canonical, cm.gender):
            result.findings.append(CastAuditFinding(
                severity=_SEVERITY_ERROR,
                code=CODE_NAME_GENDER_MISMATCH,
                char_name=name,
                message=(
                    f"Name {name!r} is canonically {canonical!r}; "
                    f"LLM emitted gender={cm.gender!r}. Likely a "
                    f"gender-inversion failure -- see operator-flagged "
                    f"cases in docs/2026-05-26-sprint-10a-backlog.md."
                ),
                expected=canonical,
                got=cm.gender,
            ))
        elif canonical == "unknown":
            # Name not in the curated pool. Soft warn -- LLM may
            # legitimately invent novel names, but we want the soak
            # log to show which names are slipping past the audit
            # so the pool can be expanded over time.
            result.findings.append(CastAuditFinding(
                severity=_SEVERITY_WARN,
                code=CODE_NAME_UNKNOWN_SOFT,
                char_name=name,
                message=(
                    f"Name {name!r} not in the curated NAME_GENDER "
                    f"pool. LLM gender={cm.gender!r} accepted on "
                    f"faith. If this name recurs in soak, add it to "
                    f"the pool."
                ),
                expected=None,
                got=cm.gender,
            ))

        # ---- (2) gender -> pronouns ----
        expected_pronouns = CANONICAL_PRONOUNS.get(cm.gender)
        if expected_pronouns is not None and cm.pronouns != expected_pronouns:
            result.findings.append(CastAuditFinding(
                severity=_SEVERITY_ERROR,
                code=CODE_GENDER_PRONOUNS_MISMATCH,
                char_name=name,
                message=(
                    f"Cast member {name!r} has gender={cm.gender!r} "
                    f"but pronouns={cm.pronouns!r}; canonical "
                    f"mapping requires {expected_pronouns!r}."
                ),
                expected=expected_pronouns,
                got=cm.pronouns,
            ))

        # ---- (3) gender -> voice_id ----
        v_gender = voice_gender.get(cm.voice_id)
        if v_gender is None:
            # ---- (4) voice_id unknown ----
            result.findings.append(CastAuditFinding(
                severity=_SEVERITY_ERROR,
                code=CODE_VOICE_UNKNOWN,
                char_name=name,
                message=(
                    f"voice_id={cm.voice_id!r} is not in the Bark v2 "
                    f"pool (config/cast_pools.VOICE_PROFILES). "
                    f"Schema-level regex should have caught this; "
                    f"audit catches it as a backstop."
                ),
                expected="one of v2/en_speaker_0..9",
                got=cm.voice_id,
            ))
        elif cm.gender in ("male", "female") and v_gender != cm.gender:
            result.findings.append(CastAuditFinding(
                severity=_SEVERITY_ERROR,
                code=CODE_GENDER_VOICE_MISMATCH,
                char_name=name,
                message=(
                    f"voice_id={cm.voice_id!r} is a {v_gender}-coded "
                    f"Bark preset, but cast member {name!r} has "
                    f"gender={cm.gender!r}."
                ),
                expected=cm.gender,
                got=v_gender,
            ))
        # NOTE: nonbinary characters do NOT fire gender_voice_mismatch.
        # The Bark pool is binary (male / female); a nonbinary character
        # gets whichever Bark voice the LLM picked, and Stage 2's TTS
        # path handles the resulting voicing. This is a known gap that
        # a future TTS-replacement step may close.

    return result
