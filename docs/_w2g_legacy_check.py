"""Wave 2 Agent G dev helper: list unclassified legacy hits.

One-shot diagnostic. Reuses the audit's classifier so the output
mirrors what the test sees.
"""
from __future__ import annotations

import subprocess
import sys

sys.path.insert(0, ".")

from tests.test_legacy_audit_clean import (  # noqa: E402
    LEGACY_PATTERN,
    _is_excluded,
    _is_forensic,
    _is_forensic_in_context,
    _is_generic_english,
)


def main() -> int:
    out = subprocess.run(
        ["git", "grep", "-nE", LEGACY_PATTERN, "--", "*.py", "*.json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    cache: dict = {}
    unclassified: list[str] = []
    for raw in out.stdout.splitlines():
        parts = raw.split(":", 2)
        if len(parts) != 3:
            continue
        path, lineno, content = parts
        if _is_excluded(path):
            continue
        if _is_forensic(content) or _is_generic_english(content):
            continue
        if _is_forensic_in_context(path, lineno, cache):
            continue
        unclassified.append(f"{path}:{lineno}: {content.strip()}")

    print(f"count: {len(unclassified)}")
    for line in unclassified[:60]:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
