"""The Mac's AnimateDiff leg, both halves (PBUG-20260913-04 and -05).

Measured on a 16 GB M4 on 2026-09-13: `otr_mac16_animatediff` completed all
eight sampling steps and then died decoding 88 latents in one batch --
2.67 GiB refused against a 20.13 GiB ceiling with 14.29 GiB already held. Two
separate defects sit in that one traceback:

  1. Nothing bounded the decode batch. ComfyUI sizes its own from
     `get_free_memory`, which on Apple silicon reports system memory rather
     than the MPS allocator's working set, and its tiled retry never runs
     because `is_oom()` matches `torch.cuda.OutOfMemoryError` and
     `torch.AcceleratorError` while MPS raises a plain `RuntimeError`.
  2. The failure was then reported as `INVALID_DAG` -- an exhausted allocator
     wearing a graph-execution wrapper -- which sends the next reader hunting
     a wiring fault that does not exist.

Neither test needs a GPU, a Mac, or torch.
"""
from __future__ import annotations

import unittest
from pathlib import Path

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_video_engines import eng_ghost_signal as GS
from nodes._otr_video_engines import render_driver as RD
from nodes._otr_video_engines import wrapper_bridge as WB


#: The two canvases the shipped animatediff profiles actually set, as LATENT
#: dimensions (canvas over 8). They differ, which is the whole point of this
#: class: config/profiles/otr_mac16_animatediff.json is 832x480 while both
#: NVIDIA animatediff profiles are 512x288.
NVIDIA_LATENT = (36, 64)    # 512 x 288
MAC_LATENT = (60, 104)      # 832 x 480


class TheDecodeIsBoundedWhereItHasToBe(unittest.TestCase):
    def test_the_chunk_follows_the_canvas_and_is_not_a_constant(self):
        """The defect this replaced: a constant 4, chosen with arithmetic done
        at 512x288, applied to a Mac that renders 832x480. Four frames there is
        about 13 GiB of transient against a 20.13 GiB ceiling -- a fix that
        might not have fixed anything, on the only machine it exists for."""
        nvidia = GS.ghost_decode_chunk_frames("mps", *NVIDIA_LATENT)
        mac = GS.ghost_decode_chunk_frames("mps", *MAC_LATENT)
        self.assertGreater(nvidia, mac,
                           "the bigger canvas must get the smaller chunk")
        self.assertGreaterEqual(mac, 1)

    def test_the_budget_is_what_decides(self):
        # Each chunk's estimated cost stays inside the declared budget, at
        # both canvases, using ComfyUI's own per-frame formula.
        for h, w in (NVIDIA_LATENT, MAC_LATENT):
            frames = GS.ghost_decode_chunk_frames("mps", h, w)
            per_frame = 2178 * h * w * 64 * 4
            self.assertLessEqual(frames * per_frame,
                                 GS.GHOST_MPS_DECODE_BUDGET_BYTES,
                                 "chunk at %dx%d exceeds the budget" % (h, w))

    def test_cuda_is_untouched_and_still_decodes_in_one_call(self):
        # Zero means "the whole batch", which is what this lane always did --
        # and it stays zero whatever the canvas is.
        for h, w in (NVIDIA_LATENT, MAC_LATENT):
            self.assertEqual(GS.ghost_decode_chunk_frames("cuda", h, w), 0)
            self.assertEqual(GS.ghost_decode_chunk_frames("cpu", h, w), 0)

    def test_a_host_we_cannot_identify_is_never_given_a_changed_path(self):
        self.assertEqual(GS.ghost_decode_chunk_frames("", *MAC_LATENT), 0)
        self.assertEqual(GS.ghost_decode_chunk_frames("xpu", *MAC_LATENT), 0)

    def test_an_unmeasurable_latent_takes_the_conservative_end(self):
        # A guess is worst exactly when the latent cannot be read, so the
        # answer there is one frame, not a remembered constant.
        self.assertEqual(GS.ghost_decode_chunk_frames("mps"),
                         GS.GHOST_MPS_DECODE_FALLBACK_FRAMES)
        self.assertEqual(GS.GHOST_MPS_DECODE_FALLBACK_FRAMES, 1)


class AnOutOfMemoryIsAnOutOfMemory(unittest.TestCase):
    """`classify_failure` read only the outermost type, so an allocator
    refusal wrapped by the node executor was reported as an invalid graph."""

    def test_the_mac_traceback_shape_classifies_as_oom(self):
        inner = RuntimeError(
            "MPS backend out of memory (MPS allocated: 14.29 GiB, other "
            "allocations: 3.65 GiB, max allowed: 20.13 GiB). Tried to "
            "allocate 2.67 GiB on private pool.")
        outer = RuntimeError("graph execution failed at node 92")
        outer.__cause__ = inner
        self.assertIs(RD.out_of_memory_in_chain(outer), True)
        self.assertEqual(RD.classify_failure(outer), RD._rt.FailureKind.OOM)

    def test_a_cuda_out_of_memory_type_is_matched_by_name(self):
        class OutOfMemoryError(RuntimeError):
            pass
        self.assertIs(RD.out_of_memory_in_chain(OutOfMemoryError("boom")), True)

    def test_an_implicit_context_counts_too(self):
        # `raise X` inside `except Y` sets __context__, not __cause__.
        inner = RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")
        outer = ValueError("wrapped")
        outer.__context__ = inner
        self.assertIs(RD.out_of_memory_in_chain(outer), True)

    def test_a_real_wiring_fault_is_still_a_wiring_fault(self):
        self.assertIs(RD.out_of_memory_in_chain(
            RuntimeError("input 'vae' is not connected")), False)

    def test_a_self_referential_chain_terminates(self):
        a = RuntimeError("a")
        b = RuntimeError("b")
        a.__cause__ = b
        b.__cause__ = a
        self.assertIs(RD.out_of_memory_in_chain(a), False)

    def test_none_is_not_an_out_of_memory(self):
        self.assertIs(RD.out_of_memory_in_chain(None), False)


class TheChunkingIsActuallyWired(unittest.TestCase):
    """Correct code nothing calls is this repo's most repeated defect. Assert
    the call at its real site, not just the helper."""

    def test_the_decode_stage_goes_through_the_chunking_method(self):
        src = (Path(__file__).resolve().parents[1] / "nodes"
               / "_otr_video_engines" / "eng_ghost_signal.py").read_text(
                   encoding="utf-8")
        self.assertIn("self._decode_latents(decode_graph, sampled_latent", src)
        # The latent must reach the budget, or it is a constant again.
        self.assertIn("ghost_decode_chunk_frames(latent_h=lat_h, latent_w=lat_w)",
                      src, "the chunk must be derived from the real latent")

    def test_the_classifier_asks_before_it_reads_the_type_name(self):
        src = (Path(__file__).resolve().parents[1] / "nodes"
               / "_otr_video_engines" / "render_driver.py").read_text(
                   encoding="utf-8")
        start = src.index("def classify_failure(exc):")
        ask = src.index("out_of_memory_in_chain(exc)", start)
        name = src.index("name = type(exc).__name__", start)
        self.assertLess(ask, name,
                        "an OOM must be recognised before the type table")


class TheChunkedDecodeReturnsTheSameFrames(unittest.TestCase):
    """The behavioural half, and the limit of it. A test that only asserted
    the chunk SIZE would pass against a method that dropped the tail,
    reordered the pieces, or decoded the first chunk three times -- so this
    runs the real method against a recording stand-in for the executor and
    compares it, frame for frame, with what one unchunked call returns.

    IT PINS ORDER AND COVERAGE, NOT THE DECODER'S ARITHMETIC. The stand-in is
    not a VAE, so no test here can prove the pixels are identical; that rests
    on the decoder being per-frame, which the method's docstring argues from
    the source. The live Mac re-run is the proof of the picture."""

    def setUp(self):
        import torch
        self.torch = torch
        self.calls = []
        self._real_run = WB.run_graph

        def fake_run_graph(graph, external_results=None, terminal=None, **kw):
            latent = (external_results or {})["sampled_latent"][0]
            samples = latent["samples"]
            self.calls.append(int(samples.shape[0]))
            # A stand-in "decoder": one 2x2 pixel frame per latent, carrying
            # the latent's own value so a dropped or duplicated chunk shows up.
            return (samples[:, :1, :1, :1].expand(-1, 2, 2, 3).clone(),)

        WB.run_graph = fake_run_graph

    def tearDown(self):
        WB.run_graph = self._real_run

    def _latent(self, frames):
        t = self.torch.arange(frames, dtype=self.torch.float32)
        return ({"samples": t.reshape(frames, 1, 1, 1).expand(frames, 4, 36, 64)
                 .clone()},)

    def _decode(self, frames, chunk):
        engine = GS.GhostSignalEngine.__new__(GS.GhostSignalEngine)
        real = GS.ghost_decode_chunk_frames
        GS.ghost_decode_chunk_frames = lambda *a, **k: chunk
        try:
            return engine._decode_latents({}, self._latent(frames), object())
        finally:
            GS.ghost_decode_chunk_frames = real

    def test_chunked_and_unchunked_agree_frame_for_frame(self):
        whole = self._decode(29, 0)
        self.calls.clear()
        chunked = self._decode(29, 4)
        self.assertEqual(self.calls, [4, 4, 4, 4, 4, 4, 4, 1],
                         "the tail chunk is the one a naive range() drops")
        self.assertTrue(self.torch.equal(whole, chunked))

    def test_one_call_when_the_batch_already_fits(self):
        self._decode(3, 4)
        self.assertEqual(self.calls, [3])

    def test_zero_means_one_call_with_everything(self):
        self._decode(88, 0)
        self.assertEqual(self.calls, [88])


if __name__ == "__main__":
    unittest.main()
