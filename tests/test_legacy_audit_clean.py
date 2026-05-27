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

import ast
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
    # Sprint 10B Wave 1 (2026-05-27) writers'-room agents -- new
    # Director / Editor / Story Room nodes that intentionally carry
    # the word "Director" as the new role name (NOT the deleted
    # legacy LLMDirector). The 10B design lives in docs/2026-05-26-
    # good-story-writer-architecture__02_design.md Section 4 Wave 1.
    # The substrings below anchor a mention to that design:
    "wave 1 agent",      # "Sprint 10B Wave 1 Agent D / E / F"
    "agent d's",         # cross-references inside Wave 1 Agent E -> D
    "agent f's",         # cross-references inside Wave 1 Wave 2
    "section 3.1",       # the frozen DirectorBrief dataclass shape
    "directorbrief",     # mentions of the new typed brief shape
    # Per the same Sprint 10B design, the new node title strings
    # (e.g. workflow JSON `"title": "Director Brief (dormant, ...)"`)
    # carry "dormant" as the load-bearing marker that Wave 1's
    # nodes are register-only until Wave 2 Agent F wires them.
    "(dormant",
    # Sprint 10B Wave 2 (2026-05-27) writers'-room Story Room agents --
    # the loop body + transcript extractor in nodes/_otr_story_room.py
    # + nodes/OTR_StoryRoom.py + nodes/_otr_story_room_extract.py +
    # nodes/OTR_StoryRoomExtract.py + their tests all describe the
    # Director / Writer / Editor as ROLES inside the writers' room loop
    # (the design Section 1 architecture) rather than the deleted
    # legacy LLMDirector node. The substrings below anchor a mention
    # to that Wave 2 design.
    "wave 2 agent",      # "Sprint 10B Wave 2 Agent F / G"
    "story room",        # any mention of the Wave 2 loop body
    "storyroomturn",     # Section 3.3 transcript dataclass field
    "storyroomtranscript",   # Section 3.3 contract
    "storyroomextraction",   # Section 3.4 contract
    "section 3.3",       # the frozen StoryRoomTranscript shape
    "section 3.4",       # the frozen StoryRoomExtraction shape
    # Writers'-room role-phrase markers. Every one of these is a
    # phrase that names a ROLE inside the Wave 2 writers' room loop
    # (per design Section 1 architecture: Director / Writer / Editor
    # roles on the same small LLM, different system prompts per
    # turn). None of these phrases ever appeared in the deleted
    # LLMDirector code path -- the legacy node had no concept of
    # writer + editor as peer roles. So a line carrying one of these
    # substrings is unambiguously a Wave 2 surface.
    "director / editor",
    "director + writer",
    "director call",
    "director brief",
    "director turn",
    "director's brief",
    "section 1:",        # design Section 1 architecture quote anchor
    "run_story_room",    # the Wave 2 loop entry-point fn name
    "writers'-room",     # any prose phrase referencing the writers'-room
    "writers' room",     #   role architecture (Section 1)
)


# Lowercase generic-English uses inside prompt templates or doc
# strings -- not code symbols, not refactor targets. Curated.
GENERIC_ENGLISH_LINES = frozenset({
    # story_orchestrator.py SCAFFOLDING_PREAMBLE
    "- precise, timed, sound-first specifications that a director, a voice cast, and",
})


# Path prefixes that are out of scope for this audit. Per S29 §Phase
# 4.4, every entry carries a one-line ``# justification:`` comment.
EXCLUDED_PATH_PREFIXES = (
    # justification: forensic migration commentary tracked in the
    # deferred S23.10 docs-rewrite task; not source code.
    "docs/",
    # justification: pytest output artifacts (HTML reports, logs);
    # not source the audit can constrain.
    "tests/_reports/",
    # justification: data files (workflow JSON copies, golden audio),
    # not source code; legacy strings appear by design.
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
    # justification: Sprint 10B Wave 1 Agent D Director module IS
    # the writers'-room Director agent (design Section 4 Wave 1);
    # not the deleted legacy LLMDirector. Every line names the new
    # role by design (DirectorBrief dataclass, run_director, etc.).
    # The legacy LLMDirector was retired in voice-path-cleanbreak
    # S2 (2026-05-12) and is forbidden via DELETED_NODE_TYPES;
    # this NEW director is the dormant writers'-room agent.
    "nodes/_otr_director_brief.py",
    # justification: Sprint 10B Wave 1 Agent D Director ComfyUI
    # node wrapper around _otr_director_brief; same reasoning.
    "nodes/OTR_DirectorBrief.py",
    # justification: Sprint 10B Wave 1 Agent D Director test file;
    # asserts on the new Director surface, every line names it by
    # design (run_director, DirectorBrief, OTR_DirectorBrief).
    "tests/test_otr_director_brief.py",
    # justification: Sprint 10B Wave 1 Agent E Editor test file
    # references the new writers'-room Director brief shape as a
    # cross-reference (the Editor consumes the Director's brief per
    # design Section 3). Mock payloads embed prose like "the
    # Director's brief" which is content of the test fixture, not
    # a legacy LLMDirector reference. Mirrors the Director test
    # exclusion just above.
    "tests/test_otr_editor_pass.py",
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


def _scan_collection_entries(src: str, target_node: ast.AST) -> list[int]:
    """Return the 1-indexed line numbers of each entry in a module-
    level collection literal (Set/FrozenSet/Tuple/List/Dict). Only
    pure literals are scanned; anything dynamic (function calls,
    comprehensions, name references) yields an empty list so the
    test silently skips it.
    """
    if isinstance(target_node, ast.Call):
        # frozenset({...}) / set([...])
        if not target_node.args:
            return []
        inner = target_node.args[0]
        return _scan_collection_entries(src, inner)
    if isinstance(target_node, (ast.Set, ast.List, ast.Tuple)):
        return [e.lineno for e in target_node.elts]
    if isinstance(target_node, ast.Dict):
        return [k.lineno for k in target_node.keys if k is not None]
    return []


def test_excluded_allowed_collections_have_per_entry_justification():
    """S29 Phase 4.4 -- generalize the C11 / IMP-38 / S24 rule from
    EXCLUDED_PATHS (test_legacy_audit_clean.py) to every module-level
    ``EXCLUDED_*`` / ``ALLOWED_*`` collection across tests/.

    Each entry in any such collection MUST be preceded (within the
    3 lines above the entry) by a ``# justification:`` comment. The
    rule exists because these collections widen the audit's blind
    spot one entry at a time; without per-entry reasons, future
    contributors silently bypass the audit.
    """
    import pathlib

    tests_dir = pathlib.Path(__file__).resolve().parent
    offenders: list[str] = []
    for f in sorted(tests_dir.glob("*.py")):
        src = f.read_text(encoding="utf-8")
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        lines = src.splitlines()
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if not names:
                continue
            name = names[0]
            if not (name.startswith("EXCLUDED_") or name.startswith("ALLOWED_")):
                continue
            entry_linenos = _scan_collection_entries(src, node.value)
            for entry_lineno in entry_linenos:
                # Walk backwards from the line above the entry,
                # consuming the contiguous block of `#` comment lines
                # (and blank lines that may separate sub-comments).
                # If the block contains `# justification:` anywhere,
                # the entry is documented. Lines is 0-indexed;
                # entry_lineno is 1-indexed.
                idx = entry_lineno - 2
                found = False
                while idx >= 0:
                    raw = lines[idx].strip()
                    if raw.startswith("#"):
                        if "# justification:" in lines[idx]:
                            found = True
                            break
                        idx -= 1
                        continue
                    # Allow a single blank line between sub-comments,
                    # but stop at code / multi-blank breaks.
                    if raw == "" and idx - 1 >= 0 and lines[idx - 1].strip().startswith("#"):
                        idx -= 1
                        continue
                    break
                if not found:
                    offenders.append(
                        f"{f.name}:{entry_lineno} ({name}) -- "
                        f"{lines[entry_lineno - 1].strip()[:60]}"
                    )

    assert not offenders, (
        f"{len(offenders)} EXCLUDED_*/ALLOWED_* entries missing "
        f"# justification: comment.\n"
        + "\n".join(offenders[:30])
    )


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
