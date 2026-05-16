"""Sprint D D1a forensic fixture.

Pinned canonical placeholder values that may appear in
workflow JSON `widgets_values` arrays of the default workflow.

These appear in widgets_values of the default workflow JSON. They
are the BUG-LOCAL-032 preserved-mode fix (commit dabcebd
2026-04-14) for the widget-drift bug class BUG-LOCAL-027 / 029 /
030 / 031. They are NOT placeholder data. Removing them
re-introduces widget drift -- the deep-research retrospective
of 2026-05-15 misread them as a no-dummy-data violation; that
finding was REFUTED in triage adjudication and the workflow
validator's stance is REJECT only divergence from the canonical
shape, never reject zero-length string entries by themselves.

See also docs/retrospectives/2026-05-15-sprint-c-triage-findings.md
section 1 and SA-100 in the Sprint A acceptance backlog for the
schema-positive canonical-shape gate.
"""
from __future__ import annotations

CANONICAL_PRESERVED_PLACEHOLDERS: tuple[str, ...] = ("", "[]", "{}")

__all__ = ["CANONICAL_PRESERVED_PLACEHOLDERS"]
