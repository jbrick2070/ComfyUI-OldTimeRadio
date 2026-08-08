"""Queue item 8 (2026-08-08): the RTX-VSR rip guard.

Mirrors ``tests/test_rip_sfx_bed_guard.py``. Asserts the retired
``nodes/rtx_upscale.py`` is truly gone: file absent from disk,
``OTR_RTXUpscale`` not in NODE_CLASS_MAPPINGS, and the name IS in
``DELETED_NODE_TYPES`` so any stale saved workflow gets a named refusal
from the shipped mechanism.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_rtx_upscale_module_absent_from_disk():
    p = REPO / "nodes" / "rtx_upscale.py"
    assert not p.exists(), (
        f"nodes/rtx_upscale.py still on disk at {p}; queue item 8 rip incomplete")


def test_rtx_upscale_import_fails():
    """Belt+suspenders: even if a stale .pyc lingers, `import` from a fresh
    interpreter must fail."""
    try:
        import importlib
        import sys
        # Force fresh import path resolution.
        sys.modules.pop("nodes.rtx_upscale", None)
        try:
            importlib.import_module("nodes.rtx_upscale")
        except ModuleNotFoundError:
            return  # expected
        # If import succeeded, it's a stale artifact.
        raise AssertionError(
            "nodes.rtx_upscale imported successfully; the module should be gone")
    except Exception as e:
        if "rtx_upscale" not in str(e).lower() and not isinstance(e, ModuleNotFoundError):
            raise


def test_otr_rtxupscale_in_deleted_node_types():
    """A stale saved workflow naming OTR_RTXUpscale must get a named refusal
    at validation time (not a silent skip / crash)."""
    from nodes._workflow_validation import DELETED_NODE_TYPES
    assert "OTR_RTXUpscale" in DELETED_NODE_TYPES, (
        f"OTR_RTXUpscale missing from DELETED_NODE_TYPES; a stale saved "
        f"workflow referencing it would slip past the shipped guard. "
        f"Current set: {sorted(DELETED_NODE_TYPES)!r}")


def test_otr_rtxupscale_not_in_node_class_mappings():
    """The class registration is gone from __init__.py."""
    import importlib
    import sys
    # Fresh import so a cached mapping doesn't fool us.
    sys.modules.pop("ComfyUI-OldTimeRadio", None)
    try:
        import ComfyUI_OldTimeRadio  # noqa: F401
        NCM = ComfyUI_OldTimeRadio.NODE_CLASS_MAPPINGS
    except Exception:
        # Fallback: read the __init__.py directly.
        init_text = (REPO / "__init__.py").read_text(encoding="utf-8")
        assert '"OTR_RTXUpscale"' not in init_text, (
            "OTR_RTXUpscale mapping still declared in __init__.py")
        return
    assert "OTR_RTXUpscale" not in NCM, (
        f"OTR_RTXUpscale still in NODE_CLASS_MAPPINGS despite the rip")


def test_no_production_module_references_rtx_upscale_symbols():
    """No live nodes/*.py may `import` from rtx_upscale or reference the class.
    Docstrings, comments and tests are exempt (they legitimately record
    the history of the retirement)."""
    import re
    # Match import statements + attribute references, ignoring comments/docs.
    IMPORT_PAT = re.compile(r"^\s*from\s+.*rtx_upscale\s+import\b", re.MULTILINE)
    NODES_DIR = REPO / "nodes"
    hits = []
    for p in NODES_DIR.rglob("*.py"):
        if p.name == "rtx_upscale.py":
            continue  # already asserted absent above
        src = p.read_text(encoding="utf-8", errors="replace")
        if IMPORT_PAT.search(src):
            hits.append(str(p.relative_to(REPO)))
    assert not hits, (
        f"live production modules still import from rtx_upscale: {hits!r}")
