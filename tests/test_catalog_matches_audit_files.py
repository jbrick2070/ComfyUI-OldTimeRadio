"""Sprint D D1a -- the catalog row license fields must agree with
the on-disk D0b audit files at docs/model-license-<sanitized>.md.

This is the tie that makes the audit framework load-bearing: the
audit file is the source of truth for the operator-reviewed verdict;
the catalog row's `license` + `license_audit_status` fields mirror it
for in-process consumption (workflow validator, dropdown filtering,
downstream routing decisions). Either surface drifting from the
other is a contract violation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_model_catalog as catalog  # noqa: E402
from tools.audit_model_license import (  # noqa: E402
    load_audit_record,
    read_targets,
)


def test_catalog_license_fields_match_audit_files_for_every_row() -> None:
    """Walk every CURATED_LLM_MODELS row. For each, look up the
    corresponding `docs/model-license-<sanitized>.md` audit file and
    assert `license` + `license_audit_status` match.
    """
    audit_targets = set(read_targets())
    catalog_rows = list(catalog.CURATED_LLM_MODELS)
    failures: list[str] = []

    # Every catalog row must have an audit file.
    for row in catalog_rows:
        if row.repo_id not in audit_targets:
            failures.append(
                f"catalog row {row.repo_id!r} has no entry in "
                f"docs/model-license-audit-targets.txt; D0b "
                f"framework cannot validate it"
            )
            continue
        record = load_audit_record(row.repo_id)
        fm = record.frontmatter
        if fm.get("license") != row.license:
            failures.append(
                f"{row.repo_id!r}: catalog.license = {row.license!r} "
                f"but audit file says {fm.get('license')!r} "
                f"({record.path.name})"
            )
        if fm.get("license_audit_status") != row.license_audit_status:
            failures.append(
                f"{row.repo_id!r}: catalog.license_audit_status = "
                f"{row.license_audit_status!r} but audit file says "
                f"{fm.get('license_audit_status')!r} "
                f"({record.path.name})"
            )

    # Every audit target must have a catalog row (catches dangling
    # audit files for retired models).
    catalog_repo_ids = {r.repo_id for r in catalog_rows}
    for target in audit_targets:
        if target not in catalog_repo_ids:
            failures.append(
                f"audit target {target!r} has no corresponding catalog "
                f"row in CURATED_LLM_MODELS; either add the row or "
                f"remove the audit target"
            )

    assert not failures, "\n  " + "\n  ".join(failures)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
