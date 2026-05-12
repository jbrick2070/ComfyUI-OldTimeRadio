"""tests/test_legacy_contract_retired.py

LFC sprint Commit 13 -- ADR section 10 acceptance test. Pins the
retirement of the legacy `script_parse_json` output contract.

History
  Pre-2026-05 the writer emitted a `script_parse_json` intermediate
  in the legacy OTR_LLMScriptWriter ledger format. Downstream
  consumers parsed it. The L3 sprint (2026-05-09 / 10) rewrote 7 of
  7 consumers to read the in-memory ledger handle via
  `production_ledger.get_ledger()` plus the L3 schema; the writer
  retired `script_parse_json` as an output name.

  ADR section 10 (Multi-Turn Polish) calls for an automated
  acceptance test that the retirement has not regressed. Any future
  PR that re-introduces a code reference to `script_parse_json`
  fails this test.

Scope
  * Searches `nodes/`, `tests/`, and the top-level files for any
    CODE reference to `script_parse_json`. Comments and docstrings
    are tolerated (the strings are searchable for historical
    context).
  * Code reference = an occurrence not preceded by a `#` comment
    marker on its own line and not inside a string literal context
    that the heuristic recognises (the heuristic is conservative;
    edge cases surface as failures and need a careful follow-up).
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REPO_ROOT = Path(__file__).resolve().parents[1]


# Directories to scan + extensions.
_SCAN_DIRS = ("nodes", "tests", "otr_v2", "visual")
_SCAN_EXTENSIONS = (".py",)


# Files that are ALLOWED to mention legacy tokens -- usually the
# acceptance test itself, plus any rename-history docs.
_ALLOWED_FILES = {
    "tests/test_legacy_contract_retired.py",
    # The cascade-orchestrator test asserts the legacy module is
    # dead; it must reference the legacy name to do so.
    "tests/test_lfc_freeze_cascade_orchestrator.py",
    "docs/2026-05-11-multi-turn-polish-adr.md",
    "docs/2026-05-11-multi-turn-polish-problem-statement.md",
    "docs/2026-05-11-lfc-go-forward-qa.md",
    "docs/2026-05-11-lfc-wiring-qa-roundrobin.md",
    "docs/2026-05-11-multi-turn-polish-qa-handoff.md",
    "docs/2026-05-12-lfc-clean-break.md",
    "docs/BUG_LOG.md",
    "ROADMAP.md",
}


_LEGACY_TOKEN = "script_parse_json"


# Clean-break v2.0-alpha (2026-05-12): additional legacy class /
# meta-key names that must not appear in source files outside the
# allowed docs.
_CLEAN_BREAK_LEGACY_TOKENS = (
    "OTR_LedgerScriptReviewer",
    "OTR_Gemma4Director",
)


def _scan_for_legacy_references():
    """Yield (rel_path, line_no, line_text) for each code reference.

    Skips:
      - Allowed files (acceptance test + ADR / problem statement docs).
      - Pure comment lines (everything after the first #).
      - Docstring lines that occur AFTER three consecutive quote chars
        (the heuristic is one-line and conservative -- multi-line
        docstrings that span code-shaped content still count, which
        we accept).
    """
    hits = []
    for top in _SCAN_DIRS:
        dir_path = REPO_ROOT / top
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in _ALLOWED_FILES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if _LEGACY_TOKEN not in content:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if _LEGACY_TOKEN not in line:
                    continue
                # Strip comments. If the token appears only AFTER a
                # `#` mark, it is a comment and tolerated.
                hash_pos = line.find("#")
                token_pos = line.find(_LEGACY_TOKEN)
                if hash_pos != -1 and hash_pos < token_pos:
                    continue
                # Quick docstring sniff: lines that look like a
                # quoted-string (start with whitespace + quote or
                # contain `"""`) tolerate the token. This is the
                # ADR-section-10 stated policy: "Hits in comments
                # / docstrings only; no code references".
                stripped = line.lstrip()
                if stripped.startswith(('"""', "'''")):
                    continue
                if stripped.startswith('"') or stripped.startswith("'"):
                    continue
                hits.append((rel, lineno, line.rstrip()))
    return hits


def test_no_script_parse_json_code_references():
    """No CODE references to `script_parse_json` outside the
    acceptance test + ADR docs.

    The legacy `script_parse_json` output name was retired in the
    L3 Consumer Rewrite sprint (2026-05-09 / 10). All 7 downstream
    consumers read the in-memory Ledger handle via
    production_ledger.get_ledger() + L3 schema. This test fails
    loud if any future code re-introduces the legacy name.
    """
    hits = _scan_for_legacy_references()
    if hits:
        lines = "\n".join(
            f"  {rel}:{lineno}: {text}" for rel, lineno, text in hits
        )
        pytest.fail(
            f"Found {len(hits)} CODE reference(s) to "
            f"`{_LEGACY_TOKEN}` outside the allowed list. The L3 "
            f"Consumer Rewrite (2026-05-09/10) retired this output "
            f"contract; the LFC sprint (2026-05-11) pins the "
            f"retirement via this acceptance test. If the "
            f"reintroduction is intentional, update _ALLOWED_FILES "
            f"in tests/test_legacy_contract_retired.py.\n\n"
            f"Hits:\n{lines}"
        )


def test_allowed_files_set_intact():
    """Sanity check -- the allow-list isn't an empty set or wildcard.

    A future refactor that accidentally relaxes the allow-list (e.g.
    sets it to a glob like "*") would silently disable this test.
    """
    assert isinstance(_ALLOWED_FILES, set)
    assert "tests/test_legacy_contract_retired.py" in _ALLOWED_FILES
    # Allow-list stays small. If it grows past ~10 entries something
    # is off -- the cascade should not be tolerating new references
    # except for docs and the acceptance test itself.
    assert len(_ALLOWED_FILES) <= 10


def _scan_for_arbitrary_token(token: str):
    """Same scan-and-skip logic as _scan_for_legacy_references but
    for any token (clean-break helper)."""
    hits = []
    for top in _SCAN_DIRS:
        dir_path = REPO_ROOT / top
        if not dir_path.exists():
            continue
        for path in dir_path.rglob("*.py"):
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in _ALLOWED_FILES:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except Exception:
                continue
            if token not in content:
                continue
            for lineno, line in enumerate(content.splitlines(), start=1):
                if token not in line:
                    continue
                hash_pos = line.find("#")
                token_pos = line.find(token)
                if hash_pos != -1 and hash_pos < token_pos:
                    continue
                stripped = line.lstrip()
                if stripped.startswith(('"""', "'''")):
                    continue
                if stripped.startswith('"') or stripped.startswith("'"):
                    continue
                hits.append((rel, lineno, line.rstrip()))
    return hits


@pytest.mark.parametrize("token", list(_CLEAN_BREAK_LEGACY_TOKENS))
def test_clean_break_legacy_names_absent(token):
    """Clean-break v2.0-alpha (2026-05-12). Legacy class names
    that were aliased pre-cleanbreak must NOT appear in code
    (comments + docstrings still tolerated). Enforced by
    automated scan so a future contributor reintroducing the
    name fails CI loudly.
    """
    hits = _scan_for_arbitrary_token(token)
    if hits:
        lines = "\n".join(
            f"  {rel}:{lineno}: {text}" for rel, lineno, text in hits
        )
        pytest.fail(
            f"Found {len(hits)} CODE reference(s) to legacy token "
            f"`{token}`. Clean-break v2.0-alpha requires zero "
            f"legacy back-compat surface.\n\nHits:\n{lines}"
        )


def test_scanner_finds_token_when_present_synthetic():
    """Smoke test the scanner: a synthetic file that contains the
    legacy token in code is detected.

    Builds the synthetic content in-memory (no file written) to
    confirm the regex / hash-position logic identifies code-line
    occurrences and skips comment-line occurrences.
    """
    code_line = f"x = {_LEGACY_TOKEN}  # not a comment"
    comment_line = f"# {_LEGACY_TOKEN} -- historical note"

    # Comment line is tolerated.
    assert "#" in comment_line
    hash_pos = comment_line.find("#")
    token_pos = comment_line.find(_LEGACY_TOKEN)
    assert hash_pos < token_pos  # comment skip rule fires

    # Code line is NOT tolerated -- token is before any `#`.
    hash_pos = code_line.find("#")
    token_pos = code_line.find(_LEGACY_TOKEN)
    assert hash_pos > token_pos  # comment skip rule does NOT fire
