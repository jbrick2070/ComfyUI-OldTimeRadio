"""A missing-node error must name the PACK, not just the class.

PBUG-20260829-09. The haunted lane needs ComfyUI-AnimateDiff-Evolved and that
was documented NOWHERE -- not README, not requirements.txt, not pyproject.toml
(and it cannot go in the latter two: a ComfyUI node pack is not a pip dist).
The runtime error did fail loud, but it named `ADE_AnimateDiffLoaderGen1` and
said "install the wrapper", which is only actionable to someone who already
knows which pack that is.

It went unnoticed because the box that proved the lane had git-cloned the pack
by hand while assembling it -- so the PROVEN path and the DOCUMENTED path had
quietly diverged, and every friction measurement was taken on a prepared box.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines.wrapper_bridge import (
    WrapperNodeMissing, resolve_graph_classes, resolve_node_class)

ADE_PACK = "ComfyUI-AnimateDiff-Evolved"


def test_graph_resolution_names_the_animatediff_pack():
    with pytest.raises(WrapperNodeMissing) as ei:
        resolve_graph_classes(
            {"ade": ("ADE_AnimateDiffLoaderGen1",),
             "ctx": ("ADE_StandardStaticContextOptions",)}, mapping={})
    msg = str(ei.value)
    assert ADE_PACK in msg, "error does not name the pack: %s" % msg
    assert "github.com" in msg, "error gives no way to obtain it: %s" % msg


def test_single_class_resolution_names_the_pack_too():
    with pytest.raises(WrapperNodeMissing) as ei:
        resolve_node_class(("ADE_AnimateDiffLoaderGen1",), mapping={})
    assert ADE_PACK in str(ei.value)


GGUF_PACK = "ComfyUI-GGUF"


def test_gguf_loader_resolution_names_the_gguf_pack():
    """2026-09-01 ship audit: ltx25_* and flux2_klein resolve UnetLoaderGGUF /
    CLIPLoaderGGUF from city96/ComfyUI-GGUF, which was named nowhere on
    failure. Both class names must map to the pack and its URL."""
    for cls in ("UnetLoaderGGUF", "CLIPLoaderGGUF"):
        with pytest.raises(WrapperNodeMissing) as ei:
            resolve_node_class((cls,), mapping={})
        msg = str(ei.value)
        assert GGUF_PACK in msg, "error does not name the pack for %s: %s" % (cls, msg)
        assert "github.com/city96/ComfyUI-GGUF" in msg, msg


def test_an_unknown_prefix_still_errors_without_inventing_a_pack():
    """No pack hint is better than a wrong one."""
    with pytest.raises(WrapperNodeMissing) as ei:
        resolve_node_class(("SomeVendor_Widget",), mapping={})
    msg = str(ei.value)
    assert "SomeVendor_Widget" in msg
    assert "provided by" not in msg, "invented a pack for an unknown prefix"


def test_the_readme_documents_the_prerequisite():
    """The error reaches the user mid-render; the README reaches them first."""
    import pathlib
    readme = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text("utf-8")
    assert ADE_PACK in readme, "the node-pack prerequisite is undocumented again"
    assert "otr_nvidia_8gb_haunted" in readme, (
        "the shipping 8GB profile is not tied to its node-pack requirement")
