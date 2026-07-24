"""Final deterministic ledger scrub.

This module owns only JSON/ledger shape, exact transport cleanup, cast binding,
and the shared narrow spoken-safety scan. It deliberately has no word-count,
style, craft, visual-vocabulary, name-guessing, or story-quality policy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Dict, List, Optional

from ._otr_content_safety import scan_spoken_ledger
from ._otr_text_metrics import set_line_text_metrics


_SPOKEN_ROLES = frozenset({"character", "announcer"})


def is_spoken_role(role: Any) -> bool:
    """Return whether a ledger role owns spoken delivery text."""
    return str(role or "").strip().lower() in _SPOKEN_ROLES


def append_compose_flag(row: Any, flag: str) -> None:
    """Append one diagnostic flag using the ledger's list semantics."""
    if not isinstance(row, dict):
        return
    flags = row.get("compose_flags")
    if not isinstance(flags, list):
        flags = []
        row["compose_flags"] = flags
    flags.append(str(flag))


@dataclass(frozen=True)
class ScrubFinding:
    code: str
    where: str
    detail: str
    line_id: Optional[str] = None


@dataclass
class ScrubResult:
    status: str
    ledger: Dict[str, Any]
    findings: List[ScrubFinding] = field(default_factory=list)
    normalized: bool = False
    safety_violations: List[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "normalized": bool(self.normalized),
            "safety_violations": list(self.safety_violations),
            "findings": [
                {
                    "code": finding.code,
                    "where": finding.where,
                    "detail": finding.detail,
                    "line_id": finding.line_id,
                }
                for finding in self.findings
            ],
        }


def _shape_findings(ledger: Any) -> list[ScrubFinding]:
    findings: list[ScrubFinding] = []
    if not isinstance(ledger, dict):
        return [
            ScrubFinding(
                "ledger_shape", "ledger", "ledger root must be an object"
            )
        ]
    try:
        json.dumps(ledger, ensure_ascii=False)
    except Exception as exc:
        findings.append(
            ScrubFinding(
                "json_invalid",
                "ledger",
                f"ledger is not JSON serializable: {type(exc).__name__}: {exc}",
            )
        )
        return findings

    cast = ledger.get("cast")
    lines = ledger.get("lines")
    if not isinstance(cast, list):
        findings.append(
            ScrubFinding("ledger_shape", "cast", "cast must be a list")
        )
        cast = []
    if not isinstance(lines, list):
        findings.append(
            ScrubFinding("ledger_shape", "lines", "lines must be a list")
        )
        return findings

    cast_ids: set[str] = set()
    cast_names: set[str] = set()
    for index, row in enumerate(cast):
        if not isinstance(row, dict):
            findings.append(
                ScrubFinding(
                    "ledger_shape", f"cast[{index}]", "cast row must be an object"
                )
            )
            continue
        char_id = str(row.get("char_id") or "").strip()
        name = str(row.get("name") or "").strip()
        if not char_id or not name:
            findings.append(
                ScrubFinding(
                    "cast_binding",
                    f"cast[{index}]",
                    "cast row requires non-empty char_id and name",
                )
            )
            continue
        if char_id in cast_ids:
            findings.append(
                ScrubFinding(
                    "cast_binding", f"cast[{index}]", f"duplicate char_id {char_id!r}"
                )
            )
        if name.casefold() in cast_names:
            findings.append(
                ScrubFinding(
                    "cast_binding", f"cast[{index}]", f"duplicate cast name {name!r}"
                )
            )
        cast_ids.add(char_id)
        cast_names.add(name.casefold())

    line_ids: set[str] = set()
    for index, row in enumerate(lines):
        where = f"lines[{index}]"
        if not isinstance(row, dict):
            findings.append(
                ScrubFinding("ledger_shape", where, "line row must be an object")
            )
            continue
        line_id = str(row.get("line_id") or "").strip()
        if not line_id:
            findings.append(
                ScrubFinding("ledger_shape", where, "line_id is required")
            )
        elif line_id in line_ids:
            findings.append(
                ScrubFinding(
                    "ledger_shape", where, f"duplicate line_id {line_id!r}", line_id
                )
            )
        else:
            line_ids.add(line_id)

        role = str(row.get("speaker_role") or "").strip().lower()
        if not is_spoken_role(role) or bool(row.get("skip")):
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            findings.append(
                ScrubFinding(
                    "empty_spoken_line",
                    where,
                    "non-skipped spoken row requires non-empty string text",
                    line_id or None,
                )
            )
        if role == "character":
            char_id = str(row.get("char_id") or "").strip()
            if char_id not in cast_ids:
                findings.append(
                    ScrubFinding(
                        "cast_binding",
                        where,
                        f"character row references unknown char_id {char_id!r}",
                        line_id or None,
                    )
                )
    return findings


def scrub_ledger(ledger: Dict[str, Any]) -> ScrubResult:
    """Normalize exact transport wrappers, then validate structure and safety."""
    findings = _shape_findings(ledger)
    if any(f.code in {"json_invalid", "ledger_shape"} for f in findings):
        return ScrubResult("FAIL", ledger, findings=findings)

    try:
        from ._otr_line_composer import strip_line_formatting
    except ImportError:  # pragma: no cover
        from _otr_line_composer import strip_line_formatting  # type: ignore

    speaker_names = {"ANNOUNCER"}
    for cast_row in ledger.get("cast") or []:
        if isinstance(cast_row, dict):
            name = str(cast_row.get("name") or "").strip()
            if name:
                speaker_names.add(name)

    normalized = False
    for index, row in enumerate(ledger.get("lines") or []):
        if not isinstance(row, dict) or row.get("skip"):
            continue
        if not is_spoken_role(row.get("speaker_role")):
            continue
        canonical = row.get("text")
        if not isinstance(canonical, str):
            continue
        cleaned = strip_line_formatting(canonical, speaker_names)
        if cleaned != canonical and cleaned.strip():
            set_line_text_metrics(row, cleaned)
            append_compose_flag(row, "transport_markup_stripped")
            findings.append(
                ScrubFinding(
                    "transport_normalized",
                    f"lines[{index}]",
                    "removed explicit response transport markup",
                    str(row.get("line_id") or "") or None,
                )
            )
            normalized = True

    # Recheck after transport normalization so an emptied spoken row is a
    # structural failure rather than invented replacement prose.
    findings.extend(
        finding
        for finding in _shape_findings(ledger)
        if finding not in findings
    )
    safety_hits = scan_spoken_ledger(ledger)
    safety_rows = [hit.to_dict() for hit in safety_hits]
    for hit in safety_hits:
        findings.append(
            ScrubFinding(
                "safety_violation",
                f"lines[{hit.line_index}]",
                f"{hit.category} term {hit.term!r}",
                hit.line_id,
            )
        )

    blocking_codes = {
        "json_invalid",
        "ledger_shape",
        "cast_binding",
        "empty_spoken_line",
        "safety_violation",
    }
    status = "FAIL" if any(
        finding.code in blocking_codes for finding in findings
    ) else "PASS"
    return ScrubResult(
        status,
        ledger,
        findings=findings,
        normalized=normalized,
        safety_violations=safety_rows,
    )


__all__ = [
    "ScrubFinding",
    "ScrubResult",
    "append_compose_flag",
    "is_spoken_role",
    "scrub_ledger",
]
