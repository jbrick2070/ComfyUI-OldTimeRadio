"""Phase A byte-identity harness -- pins the 5 production constants and
the extractor's None-return semantics against the sibling mirror."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from upstream_story_lab.extractor import get_pack_prompt_or_none
from upstream_story_lab.registry import Registry

ROOT = Path(__file__).resolve().parents[1]
SNAP_ROOT = ROOT / "tests" / "snapshots" / "phase_a"

NEW_SEAMS = (
    "outline_macro_system",
    "outline_phase_system",
    "outline_beat_system",
    "line_composer_system",
)

MIRROR_CONSTANTS = (
    ("_otr_outline.py", "_SYSTEM_PROMPT", "outline_system"),
    ("_otr_outline.py", "_MACRO_SYSTEM_PROMPT", "outline_macro_system"),
    ("_otr_outline.py", "_PHASE_SYSTEM_PROMPT", "outline_phase_system"),
    ("_otr_outline.py", "_BEAT_SYSTEM_PROMPT", "outline_beat_system"),
    ("_otr_line_composer.py", "_SYSTEM_PROMPT", "line_composer_system"),
)


def _extract_constant(mirror_root: Path, filename: str, name: str) -> str:
    """AST-extract a module-level string constant from a mirror file.
    Never imports the module (mirror is dependency-incomplete)."""
    src = (mirror_root / filename).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    if isinstance(node.value, ast.Constant) and isinstance(
                        node.value.value, str
                    ):
                        return node.value.value
    raise AssertionError(f"{name} not found as module-level str in {filename}")


def test_phase_a_new_seams_absent_from_science():
    """Phase A new seams are absent from every science_news pack ->
    extractor returns None -> production keeps its Python literal."""
    reg = Registry(ROOT)
    science_keys = [k for k in reg.packs if k[0] == "science_news"]
    assert science_keys, "expected at least one science_news pack"
    for pack_key in science_keys:
        for seam in NEW_SEAMS:
            assert get_pack_prompt_or_none(reg, *pack_key, seam) is None, (
                f"unexpected override for {pack_key} seam {seam}"
            )


def test_mirror_constants_match_snapshot(mirror_nodes: Path):
    """The 5 Python constants Phase A defers to are byte-stable against
    the production_mirror at a7bdc42d."""
    SNAP_ROOT.mkdir(parents=True, exist_ok=True)
    for filename, const_name, seam_key in MIRROR_CONSTANTS:
        current = _extract_constant(mirror_nodes, filename, const_name)
        snap = SNAP_ROOT / f"{seam_key}.txt"
        if not snap.exists():
            # First run: commit the snapshot alongside the test.
            snap.write_text(current, encoding="utf-8")
            pytest.skip(f"snapshot created: {snap.name}; re-run to assert")
        assert snap.read_text(encoding="utf-8") == current, (
            f"drift: {const_name} in mirror != snapshot {snap.name}"
        )
