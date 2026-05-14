"""S30 B4 -- standalone LFC Phase 4/5/6 node + phase-function extinction.

Three guardrails that prove the deletion stays deleted:

1. test_lfc_phase_nodes_unregistered -- the three node classes
   (OTR_LFCPhase4Scene, OTR_LFCPhase5Voice, OTR_LFCPhase6Arc) must
   NOT appear in NODE_CLASS_MAPPINGS or NODE_DISPLAY_NAME_MAPPINGS.
2. test_no_lfc_phase_callers -- the five phase function names
   (_phase_3_per_line_polish, _phase_4_per_scene_coherence,
   _phase_4_5_smart_suggestion, _phase_5_voice_drift,
   _phase_6_episode_arc) must NOT appear as Python identifiers
   anywhere under nodes/, visual/, scripts/.
3. test_lfc_phase_functions_extinct -- AST scan of
   _otr_freeze_cascade.py asserts the five function definitions
   are gone.
4. test_phase3polishreport_class_extinct -- same AST scan for the
   Phase3PolishReport dataclass.
5. test_lfc_phase_backing_files_deleted -- the four backing
   implementation files + the two shared helper modules are gone
   from nodes/.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


PACK_ROOT = Path(__file__).resolve().parent.parent

_DELETED_NODE_NAMES = (
    "OTR_LFCPhase4Scene",
    "OTR_LFCPhase5Voice",
    "OTR_LFCPhase6Arc",
)

_DELETED_PHASE_FUNCTIONS = (
    "_phase_3_per_line_polish",
    "_phase_4_per_scene_coherence",
    "_phase_4_5_smart_suggestion",
    "_phase_5_voice_drift",
    "_phase_6_episode_arc",
)

_DELETED_BACKING_FILES = (
    "_otr_lfc_phase_4_scene_coherence.py",
    "_otr_lfc_phase_5_voice_drift.py",
    "_otr_lfc_phase_6_episode_arc.py",
    "_otr_lfc_smart_suggestion.py",
    "_otr_lfc_phase_verdicts.py",
    "_otr_lfc_llm_helpers.py",
)


def _python_files() -> list[Path]:
    out: list[Path] = []
    for d in ("nodes", "visual", "scripts"):
        root = PACK_ROOT / d
        if not root.is_dir():
            continue
        out.extend(sorted(root.rglob("*.py")))
    return out


def test_lfc_phase_nodes_unregistered():
    """The three deleted standalone node classes must NOT be present
    in NODE_CLASS_MAPPINGS or NODE_DISPLAY_NAME_MAPPINGS exported by
    the package __init__.py."""
    # Parse __init__.py with AST -- avoid actually importing the
    # package (which pulls torch + transformers).
    init_path = PACK_ROOT / "__init__.py"
    src = init_path.read_text(encoding="utf-8")
    for name in _DELETED_NODE_NAMES:
        # Two contexts where the symbol would re-register: a top-level
        # dict-key in the rename map (rendered as a Constant("...") key
        # of a Dict literal) OR a NODE_CLASS_MAPPINGS literal. Forbid
        # both as Constant-string occurrences at the AST level inside
        # any Dict literal.
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Dict):
                continue
            for k in node.keys:
                if (
                    isinstance(k, ast.Constant)
                    and isinstance(k.value, str)
                    and k.value == name
                ):
                    raise AssertionError(
                        f"__init__.py still registers {name!r} as a "
                        f"node mapping key at line "
                        f"{getattr(k, 'lineno', '?')}"
                    )


def test_no_lfc_phase_callers():
    """The five phase function names must not appear as Python
    identifiers (Name, Attribute, FunctionDef) anywhere under
    nodes/, visual/, scripts/. Forensic mentions in string literals
    are permitted (e.g. the forbidden-sweep marker list).
    """
    offenders: list[str] = []
    for path in _python_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for sub in ast.walk(tree):
            if isinstance(sub, ast.Name) and sub.id in _DELETED_PHASE_FUNCTIONS:
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(sub, 'lineno', '?')} [Name {sub.id!r}]"
                )
            elif (
                isinstance(sub, ast.Attribute)
                and sub.attr in _DELETED_PHASE_FUNCTIONS
            ):
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(sub, 'lineno', '?')} [Attribute "
                    f"{sub.attr!r}]"
                )
            elif (
                isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef))
                and sub.name in _DELETED_PHASE_FUNCTIONS
            ):
                offenders.append(
                    f"{path.relative_to(PACK_ROOT).as_posix()}:"
                    f"{getattr(sub, 'lineno', '?')} [FunctionDef "
                    f"{sub.name!r}]"
                )
    assert not offenders, (
        "S30 B4 violation: deleted LFC phase function name "
        "reintroduced as a Python identifier:\n  "
        + "\n  ".join(offenders)
    )


def test_lfc_phase_functions_extinct():
    """AST scan of _otr_freeze_cascade.py: none of the five deleted
    phase functions may be redefined."""
    path = PACK_ROOT / "nodes" / "_otr_freeze_cascade.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bad: list[str] = []
    for sub in ast.walk(tree):
        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if sub.name in _DELETED_PHASE_FUNCTIONS:
                bad.append(f"line {sub.lineno}: def {sub.name}")
    assert not bad, (
        "_otr_freeze_cascade.py must not redefine the deleted phase "
        "functions:\n  " + "\n  ".join(bad)
    )


def test_phase3polishreport_class_extinct():
    """The Phase3PolishReport dataclass was the per-line polish
    report shape; B4 deletes both the class and its caller."""
    path = PACK_ROOT / "nodes" / "_otr_freeze_cascade.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for sub in ast.walk(tree):
        if isinstance(sub, ast.ClassDef) and sub.name == "Phase3PolishReport":
            raise AssertionError(
                f"_otr_freeze_cascade.py still defines Phase3PolishReport "
                f"at line {sub.lineno}"
            )


def test_lfc_phase_backing_files_deleted():
    """The four phase backing implementation files + the two shared
    helper modules must be gone from nodes/."""
    bad: list[str] = []
    for name in _DELETED_BACKING_FILES:
        p = PACK_ROOT / "nodes" / name
        if p.exists():
            bad.append(name)
    assert not bad, (
        f"Backing files still present in nodes/: {bad!r}"
    )


def test_lfc_phase_node_files_deleted():
    """The three standalone node class files must be gone from
    nodes/. Defends against a future rename that re-creates the file
    even with a different class inside."""
    bad: list[str] = []
    for name in (
        "OTR_LFCPhase4Scene.py",
        "OTR_LFCPhase5Voice.py",
        "OTR_LFCPhase6Arc.py",
    ):
        p = PACK_ROOT / "nodes" / name
        if p.exists():
            bad.append(name)
    assert not bad, (
        f"Standalone phase node files still present in nodes/: {bad!r}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
