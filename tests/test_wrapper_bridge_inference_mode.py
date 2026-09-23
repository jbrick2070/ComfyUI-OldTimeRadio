# -*- coding: utf-8 -*-
"""Graph nodes run under ``torch.inference_mode()``, as ComfyUI runs them.

Written 2026-09-22 from a live failure on the native LTX 2.5 lane. The graph
loaded its weights, sampled both stages at 100% GPU, and died on its LAST node:

    node 'decode' (decode) raised RuntimeError: Inplace update to inference
    tensor outside InferenceMode is not allowed.

The mutation is ComfyUI's own -- ``comfy/sd.py`` builds the VAE's
``process_output`` as ``image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)``, three
in-place ops -- and it is legal there because ComfyUI's executor wraps the
whole run in ``inference_mode``. This bridge did not, so a node that is correct
under ComfyUI was wrong under us. The environment was the bug, not the node.
"""
import ast
import io

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb

torch = pytest.importorskip("torch")


class _AsksTheMode:
    """A node that reports whether torch considers it to be in inference mode."""

    FUNCTION = "go"

    def go(self):
        return (torch.is_inference_mode_enabled(),)


class _MakesATensor:
    FUNCTION = "go"

    def go(self):
        return (torch.ones(2, 4, 4, 3),)


@torch.inference_mode()
def _tiled_scale_multidim_like():
    """Stands in for ``comfy.utils.tiled_scale_multidim``, decorator and all.

    THE DECORATOR IS THE POINT. The returned tensor carries the inference flag
    and this function's context has already EXITED by the time the caller sees
    it, so the in-place update below is legal only while some OUTER mode is
    still in force.
    """
    return torch.ones(2, 4, 4, 3)


class _DecodesLikeTheLtxVae:
    """The node that actually died: tiled decode, then ``process_output``.

    ``comfy/sd.py`` defines that step as
    ``lambda image: image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)`` and the LTX
    VideoVAE branches never override it.
    """

    FUNCTION = "decode"

    def decode(self):
        image = _tiled_scale_multidim_like()
        return (image.add_(1.0).div_(2.0).clamp_(0.0, 1.0),)


class _MutatesWhatItIsGiven:
    FUNCTION = "go"

    def go(self, image):
        return (image.add_(1.0),)


def test_a_node_executes_inside_inference_mode():
    out = wb.run_graph({"n": {"class": _AsksTheMode, "inputs": {}}},
                       terminal="n")
    assert out[0] is True, "the node ran outside inference mode"


def test_a_tensor_a_node_produces_is_an_inference_tensor():
    """The precondition ComfyUI's in-place VAE output math relies on.

    NOT a restatement of the test above. That one asks the node what mode it
    is in; this asks what the node PRODUCED, which is the property the failure
    actually turned on -- ``process_output`` mutates the decode result, and
    torch permits that only because the tensor carries the inference flag and
    the mutation happens under the same mode.
    """
    out = wb.run_graph({"n": {"class": _MakesATensor, "inputs": {}}},
                       terminal="n")
    assert torch.is_inference(out[0])


def test_the_terminal_tensor_still_survives_the_encoder_path():
    """An inference tensor must still reach ffmpeg, or the fix trades one
    failure for another.

    ``images_to_uint8`` is what every video lane hands its frames to, and it
    goes ``.detach().cpu().numpy()`` then numpy math. Both are reads, so both
    are legal on an inference tensor -- but that is a fact about torch, not a
    thing to assume, and the whole defect was an assumption about torch.
    """
    out = wb.run_graph({"n": {"class": _MakesATensor, "inputs": {}}},
                       terminal="n")
    frames = wb.images_to_uint8(out[0])
    assert frames.shape == (2, 4, 4, 3)
    assert frames.dtype.name == "uint8"


def test_the_bridge_does_not_leave_the_caller_in_inference_mode():
    """The ``with`` exits before ``run_graph`` returns, so the caller is free.

    The docstring here used to say "wrapped at the CALL, not around the loop",
    which stopped being true when the wrap moved and is the same class of
    stale comment that talked the first version into narrowing the boundary.
    """
    assert not torch.is_inference_mode_enabled()
    wb.run_graph({"n": {"class": _AsksTheMode, "inputs": {}}}, terminal="n")
    assert not torch.is_inference_mode_enabled()


def test_on_result_runs_inside_inference_mode_too():
    """The callback is inside the SAME mode the node ran in.

    An earlier version of this fix wrapped only the node call so that
    ``on_result`` stayed in normal mode, on the reasoning that callbacks own
    the tensors they are handed and may mutate them. A cursor QA lane
    reproduced the exact error the fix exists to kill: the callback receives
    an INFERENCE tensor, and any in-place write raised
    ``GraphExecutionError: ... Inplace update to inference tensor outside
    InferenceMode``. ComfyUI does not draw that boundary and neither do we.
    """
    seen = []
    wb.run_graph({"n": {"class": _AsksTheMode, "inputs": {}}},
                 on_result=lambda nid, out: seen.append(
                     torch.is_inference_mode_enabled()))
    assert seen == [True]


def test_a_callback_may_mutate_what_it_is_handed():
    """A callback can write to its argument, because it shares the node's mode.

    This is the property the whole-loop boundary gives callers. It is NOT a
    model of any production callback -- see the clone test below, which is.
    """
    wb.run_graph({"n": {"class": _MakesATensor, "inputs": {}}},
                 on_result=lambda nid, out: out[0].add_(1.0))


# A test named test_a_harvested_clone_escapes_the_graphs_inference_mode used to
# sit here. It RE-SPELLED ``detach().cpu().clone()`` under
# ``inference_mode(False)`` in its own body instead of calling the shipping
# helper -- so it proved that the IDIOM works, which was never in doubt, and
# would have passed unchanged while ``_latent_to_cpu`` was broken. Both QA
# lanes flagged it independently on 2026-09-22. The test below covers the same
# property by calling the real function, which is the only version of this that
# can fail when the code does.


def test_the_real_foley_harvest_produces_a_writable_copy():
    """The same property, through the SHIPPING helper rather than a re-spell.

    A test that re-implements the copy proves the test. This one calls
    ``_latent_to_cpu`` itself, from inside a graph, which is the only way to
    know the shipped code escapes.
    """
    from nodes._otr_video_engines import eng_ltx25

    seen = {}

    class _EmitsALatent:
        FUNCTION = "go"

        def go(self):
            return (None, {"samples": torch.ones(2, 4)})

    def harvest(nid, out):
        seen["latent"] = eng_ltx25.Ltx25FoleyPlusEngine._latent_to_cpu(out[1])

    wb.run_graph({"n": {"class": _EmitsALatent, "inputs": {}}},
                 on_result=harvest)
    copied = seen["latent"]["samples"]
    assert not torch.is_inference(copied)
    copied.add_(1.0)


def test_the_ltx_decode_shape_that_actually_failed():
    """The regression lock: nested @inference_mode, then an in-place update.

    The first version of these tests asserted only that a node SAW the mode
    flag set, and that a tensor built with ``torch.ones`` inside the wrap
    carried the inference flag. Both are weaker properties than the defect,
    and neither reconstructs it -- which a QA lane pointed out, correctly.
    This one fails without the wrap and passes with it.
    """
    out = wb.run_graph({"decode": {"class": _DecodesLikeTheLtxVae,
                                   "inputs": {}}}, terminal="decode")
    assert torch.allclose(out[0], torch.ones(2, 4, 4, 3))


def test_a_consumer_node_may_mutate_an_upstream_tensor_in_place():
    """Node-to-node, the same property, since every node shares one mode."""
    graph = {
        "make": {"class": _MakesATensor, "inputs": {}},
        "post": {"class": _MutatesWhatItIsGiven,
                 "inputs": {"image": wb.Wire("make", 0)}},
    }
    out = wb.run_graph(graph, terminal="post")
    assert torch.allclose(out[0], torch.full((2, 4, 4, 3), 2.0))


def test_torch_is_still_imported_lazily():
    """A hard top-level torch import would break this module on a torchless box."""
    tree = ast.parse(io.open(wb.__file__, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            assert "torch" not in names and getattr(node, "module", "") != "torch"
