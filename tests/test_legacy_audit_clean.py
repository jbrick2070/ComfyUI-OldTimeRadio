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
    # mention to a specific historical decision.
    "post-cleanbreak",
    "voice-path-cleanbreak",
    "pre-cleanbreak",
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


def _is_forensic(line_text: str) -> bool:
    lower = line_text.lower()
    return any(marker in lower for marker in FORENSIC_MARKERS)


def _is_generic_english(line_text: str) -> bool:
    stripped = line_text.strip().lower()
    return any(g in stripped for g in GENERIC_ENGLISH_LINES)


def _is_excluded(path: str) -> bool:
    return any(path.startswith(p) for p in EXCLUDED_PATH_PREFIXES)


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
        check=False,
    )
    unclassified = []
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
        unclassified.append(f"{path}:{lineno}: {content.strip()}")

    assert not unclassified, (
        f"{len(unclassified)} unclassified legacy references survived. "
        f"First 30:\n" + "\n".join(unclassified[:30])
    )
