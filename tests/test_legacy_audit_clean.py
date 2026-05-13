"""S15.5.1: the canonical legacy audit must return zero unclassified
hits after S16-S23 complete. Run as the final gate before declaring
voice-path-cleanbreak closed.

Scope choice (deviation from plan):
- Scans ``*.py`` + ``*.json`` only. Historical ``*.md`` documentation
  (docs/*, ROADMAP.md, ROADMAP_HISTORY.md, BUG_LOG.md, README.md) is by
  definition forensic migration commentary; tracked separately in the
  deferred S23.10 "README + reference fixture docs rewrite" task in
  ROADMAP.md.
- Excludes ``docs/``, ``tests/_reports/`` (test output artifacts),
  and ``tests/fixtures/`` (data files, not source code).
- Uses word-boundary anchors on each pattern token so ``Director``
  does not match ``Directory`` / ``TemporaryDirectory()``.

A line passes if it contains a forensic-marker substring (legacy,
deleted, removed in, after the deletion, after the
production_plan_json socket deletion, DELETED_NODE_TYPES,
FORBIDDEN_INPUT_SOCKETS, post-cleanbreak, voice-path-cleanbreak,
pre-cleanbreak) OR is on the GENERIC_ENGLISH_LINES allowlist
(specific known-keep lines from prompt templates).

EXCLUDED_PATHS discipline (C11 / IMP-38 / S24, 2026-05-13):
adding to ``EXCLUDED_PATHS`` requires a one-line ``# justification:``
comment naming the reason. PRs that add an entry without this
comment must be rejected. The justification rule exists because
EXCLUDED_PATHS has load-bearing semantics -- every file on the
list is invisible to the audit -- and a future contributor adding
a path without explaining why is silently widening the audit's
blind spot.
"""
from __future__ import annotations

import subprocess


# Bounded regex -- each alternative anchored at word boundaries so
# "Director" does not match "Directory" or "TemporaryDirectory".
LEGACY_PATTERN = (
    r"\bDirector\b|\bdirector_json\b|\bproduction_plan_json\b|"
    r"\bvoice_map_json\b|\bsfx_plan_json\b|\bmusic_plan_json\b|"
    r"\bLLMDirector\b|\bparser_list\b|\bparser-list\b|"
    r"\bdirector_raw_dump_dir\b"
)


# Forensic-anchor substrings that mark a legacy reference as
# documentation, not a survival. A line containing any of these
# (case-insensitive) is permitted to mention Director-era names.
FORENSIC_MARKERS = (
    "legacy",
    "deleted",
    "removed in",
    "after the deletion",
    "after the production_plan_json socket deletion",
    "deleted_node_types",
    "forbidden_input_sockets",
    # Sprint citations also count as forensic anchors -- they tie the
    # mention to a specific historical decision. Both hyphenated and
    # space-separated forms appear in shipped docstrings.
    "post-cleanbreak",
    "voice-path-cleanbreak",
    "voice-path cleanbreak",
    "pre-cleanbreak",
    # Catches "retired" / "retirement" common in test guardrail docs
    "retired",
)


# Lowercase generic-English uses inside prompt templates or doc
# strings -- not code symbols, not refactor targets. Curated.
GENERIC_ENGLISH_LINES = frozenset({
    # story_orchestrator.py SCAFFOLDING_PREAMBLE
    "- precise, timed, sound-first specifications that a director, a voice cast, and",
})


# Path prefixes that are out of scope for this audit.
EXCLUDED_PATH_PREFIXES = (
    "docs/",
    "tests/_reports/",
    "tests/fixtures/",
)


# Specific files that are inherently forensic. Every entry MUST
# carry a per-file ``# justification:`` comment explaining why the
# audit's substring rule can't be applied to it. Adding to
# EXCLUDED_PATHS without a justification comment is a contract
# breach -- PRs that do so must be rejected (C11 / IMP-38 / S24).
#
# The rule exists because EXCLUDED_PATHS is a small allowlist with
# load-bearing semantics: every file on the list is invisible to
# the audit. A future contributor adding a path without explaining
# why is silently widening the audit's blind spot.
EXCLUDED_PATHS = frozenset({
    # justification: this test file itself describes the legacy
    # tokens it scans for + classifies, so it has to mention them
    # verbatim in source.
    "tests/test_legacy_audit_clean.py",
    # justification: built specifically to assert Director surfaces
    # are gone from the workflow JSON; references the forbidden
    # names by string literal in every assertion.
    "tests/test_workflow_director_freedom.py",
    # justification: workflow contract guardrail tests assert
    # FORBIDDEN_INPUT_SOCKETS names are absent from the JSON. Each
    # name appears as a string literal in the assertion body.
    "tests/test_workflow_json_guardrails.py",
    # justification: extended-validator tests pass forbidden names
    # through synthetic workflow fixtures to verify the validator
    # raises. Forbidden names appear as widget.name / title / S&R
    # values in the test fixtures.
    "tests/test_workflow_validator_extended.py",
})


def _is_forensic(line_text: str) -> bool:
    lower = line_text.lower()
    return any(marker in lower for marker in FORENSIC_MARKERS)


def _is_generic_english(line_text: str) -> bool:
    stripped = line_text.strip().lower()
    return any(g in stripped for g in GENERIC_ENGLISH_LINES)


def _is_excluded(path: str) -> bool:
    if path in EXCLUDED_PATHS:
        return True
    return any(path.startswith(p) for p in EXCLUDED_PATH_PREFIXES)


# Number of preceding lines to scan for a forensic anchor when the
# current line lacks one on itself. Covers multi-line comment blocks
# where the marker word appears once and following lines elaborate.
_CONTEXT_WINDOW = 5


def _is_forensic_in_context(
    path: str, lineno_str: str, _content_cache: dict
) -> bool:
    """Walk up to _CONTEXT_WINDOW lines preceding the current line and
    check if any of them is forensic. Reads the file (cached per-path)
    and returns True if any preceding line in the window has a
    forensic marker substring.
    """
    try:
        lineno = int(lineno_str)
    except ValueError:
        return False
    if path not in _content_cache:
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                _content_cache[path] = fh.readlines()
        except OSError:
            _content_cache[path] = []
    lines = _content_cache[path]
    start = max(0, lineno - 1 - _CONTEXT_WINDOW)
    end = lineno - 1
    for i in range(start, end):
        if i < len(lines) and _is_forensic(lines[i]):
            return True
    return False


def test_no_unclassified_legacy_references():
    """Every Director-era symbol in code must be forensic or removed.

    Failure means there is a NEW legacy surface that none of S16-S23
    accounts for. Either it needs a sprint task, or the comment
    needs a forensic marker word ("legacy", "deleted", etc.), or the
    line needs adding to GENERIC_ENGLISH_LINES.
    """
    out = subprocess.run(
        ["git", "grep", "-nE", LEGACY_PATTERN, "--", "*.py", "*.json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    unclassified = []
    content_cache: dict = {}
    for raw in out.stdout.splitlines():
        # git grep format: path:lineno:content
        try:
            path, lineno, content = raw.split(":", 2)
        except ValueError:
            continue
        if _is_excluded(path):
            continue
        if _is_forensic(content) or _is_generic_english(content):
            continue
        # Multi-line forensic comment blocks: if any of the
        # _CONTEXT_WINDOW preceding lines has a marker, the current
        # line is part of the same documentation block.
        if _is_forensic_in_context(path, lineno, content_cache):
            continue
        unclassified.append(f"{path}:{lineno}: {content.strip()}")

    assert not unclassified, (
        f"{len(unclassified)} unclassified legacy references survived. "
        f"First 30:\n" + "\n".join(unclassified[:30])
    )
