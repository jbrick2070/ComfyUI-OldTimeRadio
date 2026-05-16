"""Sprint C C3b: meta.ltx_style_brief retirement (pulled forward per E-14).

The heavy retirement happened at S31 B3 -- `_generate_ltx_style_brief`
and `_LTX_STYLE_BRIEF_PROMPT` were deleted from
`nodes/story_orchestrator.py`, and the stamp at the writer was removed
in that sprint. C3b ships the residual cleanup so the legacy contract
is gone end-to-end before the new `meta.story_brief` system lands at
C5a1+. Per the no-legacy-back-compat standing directive: no alias, no
shim, no migration -- old ledgers without `meta.story_brief` will fall
through to `story_brief_status='absent'` per refinement §8.2 at C5b+.

These four assertions lock the retirement:

1. **Runtime symbol absence.** `_generate_ltx_style_brief` and
   `_LTX_STYLE_BRIEF_PROMPT` are not importable from
   `nodes.story_orchestrator`. Same pattern as C2b's orphan-constant
   deletion locks -- runtime `hasattr` is immune to forensic mentions
   in deletion comments and forbidden-sweep markers.

2. **Repo-wide source scan.** Zero references to `meta.ltx_style_brief`
   in `nodes/`, `visual/`, `workflows/`, or `scripts/` outside
   tokenize-suppressed forensic comments. The s28 forbidden-sweep
   tool's diff-based check is the prospective guard; this test is
   the static guard on current HEAD.

3. **Audio C7 byte-identical canary** (existing -- invoked via
   `pytest tests/test_audio_byte_identical.py` separately, not
   from this file).
"""

from __future__ import annotations

import importlib
import json
import tokenize
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent


# ---------------------------------------------------------------------------
# 1: deleted-symbol assertions (runtime hasattr)
# ---------------------------------------------------------------------------


class TestDeletedSymbols:
    """C3b: _generate_ltx_style_brief and _LTX_STYLE_BRIEF_PROMPT were
    deleted at S31 B3; this test pins their absence so any future
    contributor that tries to revive them gets a loud failure."""

    @pytest.fixture(scope="class")
    def orchestrator_mod(self):
        return importlib.import_module("nodes.story_orchestrator")

    def test_no_generate_ltx_style_brief_symbol(self, orchestrator_mod):
        assert not hasattr(orchestrator_mod, "_generate_ltx_style_brief"), (
            "_generate_ltx_style_brief should remain deleted "
            "(retired at S31 B3, marker armed at Sprint C C3b)"
        )

    def test_no_ltx_style_brief_prompt_constant(self, orchestrator_mod):
        assert not hasattr(orchestrator_mod, "_LTX_STYLE_BRIEF_PROMPT"), (
            "_LTX_STYLE_BRIEF_PROMPT should remain deleted "
            "(retired at S31 B3, marker armed at Sprint C C3b)"
        )


# ---------------------------------------------------------------------------
# 2: repo-wide static scan (tokenize-suppressed for .py forensic comments)
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


def _scan_python_for_literal_code_only(
    py_path: Path, literal: str
) -> list[tuple[int, str]]:
    """Return [(lineno, line_text)] for hits in CODE context only.

    Comment-line hits and string-literal hits are filtered. The s28
    forbidden-sweep tool uses the same classifier for its diff-based
    runtime/forensic split; this static-scan version applies the same
    filter to the current source to confirm the residual cleanup is
    complete and the only mentions are forensic.
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
        if kind in ("comment", "string"):
            continue
        hits.append((lineno, line))
    return hits


def _walk_json_strings(node, path: str = "$"):
    """Yield (path, str_value) for every string in a parsed JSON tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk_json_strings(value, f"{path}.{key}")
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            yield from _walk_json_strings(value, f"{path}[{idx}]")
    elif isinstance(node, str):
        yield (path, node)


class TestNoMetaLtxStyleBriefReferences:
    """C3b: repo-wide scan. Zero `meta.ltx_style_brief` references in
    nodes/, visual/, workflows/, or scripts/ outside forensic
    comment-context mentions."""

    @pytest.mark.parametrize(
        "scan_dir", ["nodes", "visual", "scripts"]
    )
    def test_no_python_code_reference(self, scan_dir: str):
        target = _REPO_ROOT / scan_dir
        if not target.is_dir():
            pytest.skip(f"{scan_dir}/ does not exist at HEAD")
        violations: list[str] = []
        for py_path in sorted(target.rglob("*.py")):
            for lineno, line in _scan_python_for_literal_code_only(
                py_path, "ltx_style_brief"
            ):
                violations.append(
                    f"{py_path.relative_to(_REPO_ROOT)}:{lineno} "
                    f"{line.strip()[:120]}"
                )
        assert not violations, (
            "C3b cleanup incomplete -- `ltx_style_brief` still present "
            f"in code context under {scan_dir}/:\n  "
            + "\n  ".join(violations)
        )

    def test_no_workflow_json_reference(self):
        wf_dir = _REPO_ROOT / "workflows"
        if not wf_dir.is_dir():
            pytest.skip("workflows/ does not exist at HEAD")
        violations: list[str] = []
        for json_path in sorted(wf_dir.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                violations.append(
                    f"{json_path.relative_to(_REPO_ROOT)} parse-error: {exc}"
                )
                continue
            for path, value in _walk_json_strings(data):
                if "ltx_style_brief" in value:
                    violations.append(
                        f"{json_path.relative_to(_REPO_ROOT)} {path} "
                        f"{value!r}"
                    )
        assert not violations, (
            "C3b cleanup incomplete -- `ltx_style_brief` still present "
            "in workflow JSON string values:\n  " + "\n  ".join(violations)
        )
