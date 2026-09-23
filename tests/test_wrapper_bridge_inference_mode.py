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
    """Wrapped at the CALL, not around the loop -- the caller is untouched."""
    assert not torch.is_inference_mode_enabled()
    wb.run_graph({"n": {"class": _AsksTheMode, "inputs": {}}}, terminal="n")
    assert not torch.is_inference_mode_enabled()


def test_on_result_runs_outside_inference_mode():
    """Callbacks hand tensors to owners who may legitimately mutate them."""
    seen = []
    wb.run_graph({"n": {"class": _AsksTheMode, "inputs": {}}},
                 on_result=lambda nid, out: seen.append(
                     torch.is_inference_mode_enabled()))
    assert seen == [False]


def test_torch_is_still_imported_lazily():
    """A hard top-level torch import would break this module on a torchless box."""
    tree = ast.parse(io.open(wb.__file__, encoding="utf-8").read())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            assert "torch" not in names and getattr(node, "module", "") != "torch"
