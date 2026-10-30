"""ONE owner decides which ffmpeg runs: ``nodes/_otr_shared/ffmpeg.py``.

This is the ratchet, not the docstring. ``encode_sink.has_nvenc`` once called
itself "the ONLY nvenc decision in the pack" while a third string-test copy
sat in ``scope_draw`` -- a claim in prose reads as coverage and stops the next
reader from looking. So the claim here is made by an AST walk over ``nodes/``:

* ``OTR_FFMPEG`` is read in exactly one module, the ffmpeg owner.
* ``which(...)`` on ffmpeg, ffprobe or a variable is called only by the two
  tool owners -- the ffmpeg owner and the probe owner (``ffprobe.py``). As of
  2026-09-04 nothing else did, once the twelve ffmpeg copies were retired. A
  new copy fails this test the day it is written, whether it spells the tool
  as a constant or as a variable. A literal naming some other tool is not an
  ffmpeg decision (the carve-out was earned by the RTX VSR installer's
  ``which("uv")`` on 2026-10-16; that installer left on 2026-10-30 and the
  carve-out stays generic).
"""
from __future__ import annotations

import ast
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parent.parent
NODES = REPO / "nodes"
FFMPEG_OWNER = NODES / "_otr_shared" / "ffmpeg.py"
PROBE_OWNER = NODES / "_otr_shared" / "ffprobe.py"
_WHICH_OWNERS = {FFMPEG_OWNER, PROBE_OWNER}
_BARE = {"ffmpeg", "ffmpeg.exe"}


def _call_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


_MEDIA_STEMS = {"ffmpeg", "ffprobe"}


def _names_another_tool(node: ast.Call) -> bool:
    """A which() whose argument is a literal naming some OTHER tool is not an
    ffmpeg decision (the case that earned this, the RTX VSR installer's
    ``which("uv")``, has since left the tree). A variable
    argument still counts -- it could hold "ffmpeg" -- so the ratchet keeps
    its teeth for every spelling that can reach the media tools."""
    arg = node.args[0] if node.args else None
    if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
        return False
    # Compare the bare stem, so "/usr/bin/ffmpeg", "ffmpeg.cmd" and
    # "ffmpeg " still count as the media tool; a Windows path splits on any
    # host.
    bare = arg.value.strip().lower().replace("\\", "/")
    stem = os.path.splitext(os.path.basename(bare))[0]
    return stem not in _MEDIA_STEMS


def _offenders(path: pathlib.Path) -> list:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    rel = path.relative_to(REPO)
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and node.value == "OTR_FFMPEG" \
                and path != FFMPEG_OWNER:
            out.append(f"{rel}:{node.lineno} reads OTR_FFMPEG")
        elif isinstance(node, ast.Call) and _call_name(node) == "which" \
                and path not in _WHICH_OWNERS \
                and not _names_another_tool(node):
            out.append(f"{rel}:{node.lineno} calls which(...)")
        elif isinstance(node, (ast.List, ast.Tuple)) and node.elts \
                and isinstance(node.elts[0], ast.Constant) \
                and str(node.elts[0].value).lower() in _BARE \
                and path not in _WHICH_OWNERS:
            # An argv that STARTS with the bare tool name executes whatever
            # PATH says, having asked nobody -- the three sites r3 found
            # (post_upscale, lyria, foley_stems) read no env and called no
            # which(), which is exactly why the two rules above missed them.
            out.append(f"{rel}:{node.lineno} argv literal starts with "
                       f"{node.elts[0].value!r}")
    return out


def test_the_owner_exists_and_reads_the_pin():
    assert FFMPEG_OWNER.is_file(), "the ffmpeg owner module is missing"
    tree = ast.parse(FFMPEG_OWNER.read_text(encoding="utf-8"))
    reads = [n for n in ast.walk(tree)
             if isinstance(n, ast.Constant) and n.value == "OTR_FFMPEG"]
    assert reads, "the owner no longer reads OTR_FFMPEG -- who does?"


def test_no_second_ffmpeg_resolution_anywhere_under_nodes():
    offenders = []
    for py in sorted(NODES.rglob("*.py")):
        offenders.extend(_offenders(py))
    assert offenders == [], (
        "a second ffmpeg resolution decision exists outside "
        "nodes/_otr_shared/ffmpeg.py -- delegate to resolve_ffmpeg():\n  "
        + "\n  ".join(offenders)
    )
