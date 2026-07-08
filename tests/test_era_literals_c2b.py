"""Sprint C C2b: era literal cleanbreak -- orchestrator (reduced scope).

C1 audit (executed at commit b01feda, recorded in `SPRINT.md` §B) plus
the C2b 8-search audit (chat thread, 2026-05-15) confirmed that
`SCRIPT_SYSTEM_PROMPT` and `SCAFFOLDING_PREAMBLE` were orphan constants
from the `eec4718` LPL extraction sprint -- the new
`OTR_LedgerScriptWriter` pipeline composes prompts from per-phase
modules (`_otr_outline`, `_otr_line_composer`, `_otr_ledger_reviewer`,
`_otr_period_prompts`) and never imports either constant. The OMNI-RETRO
5-pillar block at lines 3160-3179 of HEAD (851f54e) lived INSIDE
`SCRIPT_SYSTEM_PROMPT` and was therefore forensic, not runtime.

C2b reduced scope per operator directive 2026-05-15 (Path A):
1. Live rerank prompt at `nodes/story_orchestrator.py:1596` rewritten
   to be style-neutral (`"for a 1940s-style radio drama"` ->
   `"for an audio drama"`).
2. `SCAFFOLDING_PREAMBLE` constant deleted entirely (no shim,
   no alias, per the no-legacy-back-compat standing directive).
3. `SCRIPT_SYSTEM_PROMPT` constant deleted entirely.
4. `_STYLE_WORLD_BLOCK` machinery SKIPPED -- no live consumer to wire
   it into. No dead code shipped to replace dead code.

The four tests below pin these decisions. The
`test_script_system_prompt_interpolates_style_world_block` and
`test_script_token_length_stability_with_dynamic_world_block` tests
from the original v3 plan are intentionally absent: there is no live
script-system prompt to interpolate, so there is nothing to stabilize.

Forensic attribution comments naming the deleted constants
(e.g. `tests/test_legacy_audit_clean.py:74` documenting where a test
fixture line originated) are intentional and preserved. The tests use
runtime `hasattr` assertions, not source-text scans, so attribution
comments do not produce false positives.
"""

from __future__ import annotations

import json
import os
import tokenize
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_ORCHESTRATOR = _REPO_ROOT / "nodes" / "story_orchestrator.py"


# ---------------------------------------------------------------------------
# Tokenize-aware code-context scanner (mirrors tests/_s28_forbidden_sweep)
# ---------------------------------------------------------------------------


def _classify_lines(path: Path) -> dict[int, str]:
    """Return {1-indexed line: 'code' | 'comment' | 'string'} for a .py file."""
    src = path.read_text(encoding="utf-8", errors="replace")
    classifications: dict[int, str] = {}
    try:
        toks = list(
            tokenize.generate_tokens(
                iter(src.splitlines(keepends=True)).__next__
            )
        )
    except (tokenize.TokenizeError, SyntaxError, IndentationError):
        for i in range(1, src.count("\n") + 2):
            classifications[i] = "code"
        return classifications

    string_token_types = (
        tokenize.STRING,
        getattr(tokenize, "FSTRING_START", -1),
        getattr(tokenize, "FSTRING_MIDDLE", -1),
        getattr(tokenize, "FSTRING_END", -1),
    )
    string_lines: set[int] = set()
    comment_lines: set[int] = set()
    for tok in toks:
        if tok.type in string_token_types:
            start_l, _ = tok.start
            end_l, _ = tok.end
            for line_no in range(start_l, end_l + 1):
                string_lines.add(line_no)
        elif tok.type == tokenize.COMMENT:
            comment_lines.add(tok.start[0])

    total = src.count("\n") + 1
    for line_no in range(1, total + 1):
        if line_no in string_lines:
            classifications[line_no] = "string"
        elif line_no in comment_lines:
            classifications[line_no] = "comment"
        else:
            classifications[line_no] = "code"
    return classifications


def _scan_runtime_strings_for_literal(
    py_path: Path, literal: str
) -> list[tuple[int, str]]:
    """Return [(lineno, line_text)] for hits in code or in non-docstring
    string literals. Comment-line hits are filtered as forensic.

    The orchestrator's prompts are constructed via f-strings and
    concatenated text -- both classified as "string" by tokenize. We
    therefore count "string" hits as runtime (they MAY be part of a
    live prompt) and only filter "comment" hits. The C2b deletion
    eliminated the only orphan strings that contained the legacy
    OMNI-RETRO 5-pillar literals, so a remaining string hit indicates
    either a NEW orphan or an unwanted reintroduction.
    """
    if not py_path.exists():
        return []
    classifications = _classify_lines(py_path)
    src = py_path.read_text(encoding="utf-8", errors="replace")
    hits: list[tuple[int, str]] = []
    for lineno, line in enumerate(src.splitlines(), start=1):
        if literal not in line:
            continue
        kind = classifications.get(lineno, "code")
        if kind == "comment":
            continue
        hits.append((lineno, line))
    return hits


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRerankPromptEraNeutral:
    """C2b: live rerank prompt at story_orchestrator.py:1596 is era-neutral."""

    def test_no_1940s_literal_in_rerank_prompt(self):
        # `_llm_rerank_with_bodies` is the only orchestrator-layer
        # prompt-string site that was confirmed runtime-live by the C2b
        # 8-search audit. The `1940s-style radio drama` anchor was
        # rewritten to `audio drama` -- style-neutral, lets the chosen
        # style slug flow through `{genre_human}` instead.
        src = _ORCHESTRATOR.read_text(encoding="utf-8")
        assert "1940s-style radio drama" not in src, (
            "rerank prompt at story_orchestrator.py:1596 still contains "
            "the retired era literal '1940s-style radio drama'"
        )

    def test_rerank_prompt_audio_drama_anchor_present(self):
        # Locks the replacement text so a future refactor cannot
        # silently revert to a style-locked anchor.
        src = _ORCHESTRATOR.read_text(encoding="utf-8")
        assert '"for an audio drama: specific human stakes, "' in src, (
            "rerank prompt at story_orchestrator.py:1596 missing the "
            "expected style-neutral replacement text"
        )


class TestOrphanConstantsDeleted:
    """C2b: SCAFFOLDING_PREAMBLE + SCRIPT_SYSTEM_PROMPT constants deleted.

    The string names may still appear in forensic attribution comments
    (e.g. tests/test_legacy_audit_clean.py:74 documenting where test
    fixture lines originated; this commit's own forensic-note block;
    the _load_canon_for_writer docstring naming the deleted consumer
    as historical context until Sprint G's orphan sweep). Those are
    intentional and preserved. Runtime assertions catch reintroduction.
    """

    def test_scaffolding_preamble_constant_deleted(self):
        import importlib
        so = importlib.import_module("nodes.story_orchestrator")
        assert not hasattr(so, "SCAFFOLDING_PREAMBLE"), (
            "SCAFFOLDING_PREAMBLE should be deleted, not present "
            "(C2b orphan-constant cleanup)"
        )

    def test_script_system_prompt_constant_deleted(self):
        import importlib
        so = importlib.import_module("nodes.story_orchestrator")
        assert not hasattr(so, "SCRIPT_SYSTEM_PROMPT"), (
            "SCRIPT_SYSTEM_PROMPT should be deleted, not present "
            "(C2b orphan-constant cleanup)"
        )


class TestNoOrphan5PillarLiterals:
    """C2b: the 5-pillar / dramaturg era literals live nowhere in
    orchestrator runtime strings (code or string-context).

    The original literals lived only inside the now-deleted
    SCRIPT_SYSTEM_PROMPT and SCAFFOLDING_PREAMBLE constants. After
    deletion, they should appear nowhere except the C2b deletion's own
    forensic comment block (line ~2995, which the tokenize classifier
    flags as "comment" and is filtered).
    """

    @pytest.mark.parametrize(
        "literal",
        [
            "1950s Americana",
            "Afrofuturism",
            "Neo-Tokyo",
            "Thai Street Density",
            "Russian Dieselpunk",
            "Orson Welles",
            "Norman Corwin",
            "Lucille Fletcher",
            "golden age of radio",
            "OMNI-RETRO CULTURAL COLLISION",
        ],
    )
    def test_literal_not_in_runtime_strings(self, literal: str):
        hits = _scan_runtime_strings_for_literal(_ORCHESTRATOR, literal)
        assert not hits, (
            f"C2b cleanup failed -- {literal!r} still present in "
            f"runtime context (code or string-literal) of "
            f"nodes/story_orchestrator.py:\n  "
            + "\n  ".join(f"{ln}: {line.strip()[:120]}" for ln, line in hits)
        )
