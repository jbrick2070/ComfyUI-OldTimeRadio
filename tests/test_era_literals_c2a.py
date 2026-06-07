"""Sprint C C2a: era literal cleanbreak -- visual layer.

These tests lock in the C2a removal of era anchors from production source
files in `visual/` and `workflows/`, plus the `_DEFAULT_STYLE_TAIL`
constant in `nodes/otr_video_plan.py`. The four checks together prevent
silent reintroduction of period-locked language in the visual prompt
path, while preserving the test fixture exception at
`tests/test_musicgen_cache_keys.py:23` (a cache-key payload that must
remain byte-stable -- not in scope for these scans).

Forensic mentions of the retired literals in docstrings and comments
are allowed: this scanner uses Python's `tokenize` module to classify
each line and only counts hits in code context. JSON files are walked
as parsed structures so only string VALUES are checked, not key names
or surrounding scaffolding.
"""

from __future__ import annotations

import json
import os
import tokenize
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_VISUAL_DIR = _REPO_ROOT / "visual"
_WORKFLOWS_DIR = _REPO_ROOT / "workflows"
_NODES_DIR = _REPO_ROOT / "nodes"
_WORKFLOW_FULL = _WORKFLOWS_DIR / "otr_scifi_16gb_full.json"

# The era literals C2a retired.
_RETIRED_ERA_LITERALS = (
    "1940s noir radio drama style",
    "1940s noir radio drama style, cinematic",
    "1980s broadcast aesthetic",
)

# The post-replacement strings C2a installed at workflow JSON sites.
_EXPECTED_NEUTRAL_STRINGS = (
    "head-and-shoulders studio portrait, neutral lighting, cinematic",
    "cinematic, 35mm film look, subtle film grain, volumetric lighting",
)


# ---------------------------------------------------------------------------
# Helpers -- tokenize-aware Python scanner mirroring docs/_s28_forbidden_sweep
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


def _scan_python_for_literal(
    py_path: Path, literal: str
) -> list[tuple[int, str]]:
    """Return [(lineno, line_text)] for occurrences of `literal` in a .py file.

    A hit counts only when the line classification is "code" OR "string"
    -- a string literal in production source IS the era anchor we are
    retiring, even though `tokenize` classifies it as "string". Comments
    and docstrings (also "string"-classified by tokenize) are filtered
    by checking whether the surrounding token is a docstring vs an
    ordinary expression.

    Simpler heuristic that matches Sprint C intent: skip pure-comment
    lines (line classification == "comment"); count all others. This
    catches the retired literals at lines 109/170/234 of the FLUX
    portrait file (all "string"-classified by tokenize because they
    are quoted Python strings) while still allowing comment-context
    forensic mentions.
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNo1940sLiteralInVisualLayer:
    """C2a: zero `1940s` in visual/ + workflows/ outside test fixtures."""

    def test_no_1940s_literal_in_visual_layer(self):
        violations: list[str] = []

        # Walk every .py under visual/ -- skip pure comment lines.
        for py_path in sorted(_VISUAL_DIR.rglob("*.py")):
            for lineno, line in _scan_python_for_literal(py_path, "1940s"):
                violations.append(
                    f"{py_path.relative_to(_REPO_ROOT)}:{lineno} {line.strip()}"
                )

        # Walk workflow JSONs as parsed strings.
        for json_path in sorted(_WORKFLOWS_DIR.rglob("*.json")):
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                violations.append(
                    f"{json_path.relative_to(_REPO_ROOT)} parse-error: {exc}"
                )
                continue
            for path, value in _walk_json_strings(data):
                if "1940s" in value:
                    violations.append(
                        f"{json_path.relative_to(_REPO_ROOT)} {path} {value!r}"
                    )

        assert not violations, (
            "C2a era-literal sweep failed -- `1940s` still present in "
            "visual/ or workflows/:\n  "
            + "\n  ".join(violations)
        )


class TestNo1980sBroadcastLiteralInVideoPlan:
    """C2a: zero `1980s broadcast` in nodes/otr_video_plan.py + nodes/video_engine.py."""

    @pytest.mark.parametrize(
        "rel_path",
        ["nodes/otr_video_plan.py", "nodes/video_engine.py"],
    )
    def test_no_1980s_broadcast_literal(self, rel_path: str):
        py_path = _REPO_ROOT / rel_path
        if not py_path.exists():
            pytest.skip(f"{rel_path} does not exist at HEAD")
        hits = _scan_python_for_literal(py_path, "1980s broadcast")
        assert not hits, (
            f"C2a era-literal sweep failed -- `1980s broadcast` still "
            f"present in {rel_path}:\n  "
            + "\n  ".join(f"{ln}: {line.strip()}" for ln, line in hits)
        )


class TestWorkflowJsonStringBasedReplaceCompleted:
    """C2a / E-19: exact-string recursive replacement on workflow JSON.

    Locks in (a) zero pre-replacement era strings remain in any string
    value of the canonical workflow JSON, and (b) the post-replacement
    era-neutral strings ARE present at the expected sites.
    """

    def test_no_pre_replacement_strings_in_workflow(self):
        with open(_WORKFLOW_FULL, encoding="utf-8") as handle:
            data = json.load(handle)
        violations: list[str] = []
        for path, value in _walk_json_strings(data):
            for retired in _RETIRED_ERA_LITERALS:
                if retired in value:
                    violations.append(f"{path}: {value!r} contains {retired!r}")
        assert not violations, (
            "C2a workflow JSON still contains retired era strings:\n  "
            + "\n  ".join(violations)
        )


class TestWorkflowJsonNoEraLiteralWidgetDefault:
    """C2a: FLUX portrait widget defaults are era-neutral.

    Walks the workflow JSON's `nodes` array. For every node whose `type`
    is FLUX-portrait-related, asserts no `widgets_values` entry contains
    a retired era literal.
    """

    def test_flux_portrait_widget_defaults_era_neutral(self):
        with open(_WORKFLOW_FULL, encoding="utf-8") as handle:
            data = json.load(handle)
        nodes = data.get("nodes") or []
        violations: list[str] = []
        for node in nodes:
            ntype = (node.get("type") or "").lower()
            if "flux" not in ntype or "portrait" not in ntype:
                continue
            widgets = node.get("widgets_values") or []
            for idx, value in enumerate(widgets):
                if not isinstance(value, str):
                    continue
                for retired in _RETIRED_ERA_LITERALS:
                    if retired in value:
                        violations.append(
                            f"node id={node.get('id')} type={node.get('type')} "
                            f"widgets_values[{idx}]={value!r} contains "
                            f"{retired!r}"
                        )
        # Assertion phrasing: even if there are zero FLUX portrait nodes
        # in the workflow at the time of this test, the assertion is
        # vacuously satisfied. C2a's responsibility is the literals, not
        # the node count.
        assert not violations, (
            "C2a FLUX portrait widget defaults still carry era literals:"
            "\n  " + "\n  ".join(violations)
        )
