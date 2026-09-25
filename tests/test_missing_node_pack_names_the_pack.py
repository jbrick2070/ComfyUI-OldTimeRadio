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
    # CORRECTED 2026-09-21. This asserted `otr_nvidia_8gb_haunted`, which no
    # shipped graph resolves and which `build_variants.SHIPPING_SET` does not
    # contain -- a leftover from before the 8 GB AnimateDiff profile was
    # renamed. The stale string pulled a FALSE sentence into the README to
    # satisfy it, which is the wrong direction: the doc followed the test
    # instead of the code. Pin the profile the shipped graph actually names.
    assert "otr_8gb_animatediff" in readme, (
        "the shipping 8GB AnimateDiff profile is not tied to its node-pack "
        "requirement")
