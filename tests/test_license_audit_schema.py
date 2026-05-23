"""Sprint D D0b -- license audit framework schema-positive gate.

Required pytest assertions per v3 plan:

1. Every audit_target has a corresponding markdown file.
2. Every audit file's YAML frontmatter has the required keys.
3. Every audit file's frontmatter repo_id matches the sanitized filename.
4. audit_all() returns no errors and one record per target.

(The talkie-specific research_lane assertion was retired 2026-05-22
when the broken talkie-lm/talkie-1930-13b-it row + its audit file
were removed.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.audit_model_license import (  # noqa: E402
    AUDIT_DIR,
    AUDIT_FILENAME_TEMPLATE,
    REQUIRED_KEYS,
    audit_all,
    parse_frontmatter,
    read_targets,
    sanitize_repo_id,
)


def test_every_audit_target_has_a_corresponding_markdown_file() -> None:
    targets = read_targets()
    assert targets, "audit targets list is empty"
    missing: list[str] = []
    for repo_id in targets:
        sanitized = sanitize_repo_id(repo_id)
        expected = AUDIT_DIR / AUDIT_FILENAME_TEMPLATE.format(
            sanitized=sanitized,
        )
        if not expected.is_file():
            missing.append(f"{repo_id} -> {expected.name}")
    assert not missing, (
        "audit files missing for targets:\n  " + "\n  ".join(missing)
    )


def test_audit_yaml_frontmatter_has_required_keys() -> None:
    targets = read_targets()
    failures: list[str] = []
    for repo_id in targets:
        sanitized = sanitize_repo_id(repo_id)
        path = AUDIT_DIR / AUDIT_FILENAME_TEMPLATE.format(
            sanitized=sanitized,
        )
        fm = parse_frontmatter(path.read_text(encoding="utf-8"))
        missing_keys = [k for k in REQUIRED_KEYS if k not in fm]
        if missing_keys:
            failures.append(f"{path.name} missing {missing_keys}")
    assert not failures, (
        "audit frontmatter missing required keys:\n  "
        + "\n  ".join(failures)
    )


def test_audit_repo_id_matches_filename() -> None:
    targets = read_targets()
    failures: list[str] = []
    for repo_id in targets:
        sanitized = sanitize_repo_id(repo_id)
        path = AUDIT_DIR / AUDIT_FILENAME_TEMPLATE.format(
            sanitized=sanitized,
        )
        fm = parse_frontmatter(path.read_text(encoding="utf-8"))
        fm_repo_id = fm.get("repo_id", "")
        if fm_repo_id != repo_id:
            failures.append(
                f"{path.name}: frontmatter repo_id "
                f"{fm_repo_id!r} != target {repo_id!r}"
            )
    assert not failures, (
        "frontmatter repo_id mismatch:\n  " + "\n  ".join(failures)
    )


def test_audit_all_returns_no_errors_at_d0b_landing() -> None:
    """Convenience smoke: the framework as wired at D0b passes all rows."""
    passing, errors = audit_all()
    assert errors == [], (
        "license audit framework errors at D0b landing:\n  "
        + "\n  ".join(str(e) for e in errors)
    )
    assert len(passing) == len(read_targets()), (
        f"audit framework returned {len(passing)} passing records; "
        f"expected {len(read_targets())} (one per target)."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
