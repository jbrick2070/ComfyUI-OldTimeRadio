"""Narrow Stage-3 validation for accepted spoken rows.

Only unambiguous transport/ledger markup and the shared terminal safety policy
are blocking. Word length, pronouns, style, banned phrases, continuity guesses,
on-beat similarity, and ordinary narrator-like wording are not judged here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any


SEVERITY_ERROR = "error"
CODE_EMPTY_SPOKEN_LINE = "empty_spoken_line"
CODE_SPEAKER_MARKUP = "speaker_markup"
CODE_SFW_VIOLATION = "sfw_violation"

_SPOKEN_ROLES = frozenset({"character", "announcer"})
def _has_transport_speaker_prefix(text: str, speaker: str) -> bool:
    """Match only authoritative speaker labels, never ordinary WORD: prose."""
    labels = [value for value in (speaker, "ANNOUNCER") if str(value).strip()]
    for label in labels:
        escaped = re.escape(str(label).strip())
        if re.match(rf"^\s*{escaped}\s*:", text, flags=re.IGNORECASE):
            return True
        if re.match(
            rf"^\s*\[\s*(?:VOICE\s*:\s*)?{escaped}\s*\]",
            text,
            flags=re.IGNORECASE,
        ):
            return True
    return False


@dataclass(frozen=True)
class ValidationFinding:
    severity: str
    code: str
    beat_id: str
    speaker: str
    message: str
    expected: str | None = None
    got: str | None = None


@dataclass
class ValidationResult:
    findings: list[ValidationFinding] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationFinding]:
        return [row for row in self.findings if row.severity == SEVERITY_ERROR]

    @property
    def warns(self) -> list[ValidationFinding]:
        return []

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "error_count": len(self.errors),
            "warn_count": 0,
            "findings": [
                {
                    "severity": row.severity,
                    "code": row.code,
                    "beat_id": row.beat_id,
                    "speaker": row.speaker,
                    "message": row.message,
                    "expected": row.expected,
                    "got": row.got,
                }
                for row in self.findings
            ],
        }


def _beat_identity(beat: Any) -> tuple[str, str, str]:
    return (
        str(getattr(beat, "beat_id", "") or ""),
        str(getattr(beat, "speaker", "") or ""),
        str(getattr(beat, "speaker_role", "character") or "character").lower(),
    )


def validate_spoken_structure(beat: Any, line_text: Any) -> ValidationFinding | None:
    """Reject only empty or unmistakably transport-decorated spoken rows."""
    beat_id, speaker, role = _beat_identity(beat)
    if role not in _SPOKEN_ROLES:
        return None
    text = str(line_text or "")
    if not text.strip():
        return ValidationFinding(
            SEVERITY_ERROR,
            CODE_EMPTY_SPOKEN_LINE,
            beat_id,
            speaker,
            "Spoken ledger row is empty.",
            "nonempty spoken text",
            "",
        )
    if ("\n" in text or "\r" in text
            or _has_transport_speaker_prefix(text, speaker)):
        return ValidationFinding(
            SEVERITY_ERROR,
            CODE_SPEAKER_MARKUP,
            beat_id,
            speaker,
            "Spoken text contains a transport speaker label or multiple rows.",
            "one label-free spoken row",
            text[:160],
        )
    return None


def validate_sfw(beat: Any, line_text: Any) -> ValidationFinding | None:
    """Return one blocking finding from the shared narrow safety policy."""
    from ._otr_content_safety import find_text_hits

    beat_id, speaker, role = _beat_identity(beat)
    if role not in _SPOKEN_ROLES:
        return None
    hits = find_text_hits(line_text)
    if not hits:
        return None
    category, term = hits[0]
    return ValidationFinding(
        SEVERITY_ERROR,
        CODE_SFW_VIOLATION,
        beat_id,
        speaker,
        f"Spoken safety violation: {category} term {term!r}.",
        (
            "no profanity, explicit guns/knives/weapons, or explicit "
            "sexual/nudity language"
        ),
        term,
    )


def validate_line(beat: Any, line_text: Any) -> ValidationResult:
    """Validate one accepted line without a synthetic story plan."""
    findings = [
        finding
        for finding in (
            validate_spoken_structure(beat, line_text),
            validate_sfw(beat, line_text),
        )
        if finding is not None
    ]
    return ValidationResult(findings)



__all__ = [
    "CODE_EMPTY_SPOKEN_LINE",
    "CODE_SFW_VIOLATION",
    "CODE_SPEAKER_MARKUP",
    "ValidationFinding",
    "ValidationResult",
    "validate_line",
    "validate_sfw",
    "validate_spoken_structure",
]
