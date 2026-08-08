"""Queue item 8 (2026-08-08): color-matrix symmetry guard for
_run_model_pipeline.

Fable final-gate MF-1: every clip source is bt709-TAGGED by the V-1 contract.
Without explicit bt709 declarations on BOTH sides of the model round-trip
(yuv->rgb24 decode + rgb24->yuv420p encode), the encoder side falls to
swscale's default matrix (bt601 historically) while `_color_args` TAGS
output bt709 -- an asymmetric round-trip that color-shifts exactly the
segments the model enhanced, sitting next to floor segments that never
left yuv. Six mechanical reviews audited time and shape; none audited
color.

This test pins the two ffmpeg VF flags that close the drift. If a future
refactor removes them, this test fires by name.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_run_model_pipeline_declares_bt709_on_decode_and_encode():
    """Source read of otr_silent_composite._run_model_pipeline must contain
    both `in_color_matrix=bt709` (decode -vf) and
    `out_color_matrix=bt709` (encode -vf). The literal strings are the
    contract; grep-level check is enough."""
    src = (REPO / "nodes" / "otr_silent_composite.py").read_text(encoding="utf-8")
    # Isolate the _run_model_pipeline function body so the assertion can't
    # be satisfied by an unrelated comment elsewhere in the file.
    fn_start = src.find("def _run_model_pipeline(")
    assert fn_start != -1, "_run_model_pipeline definition missing"
    # 5000 chars is a generous window; the whole function is much smaller.
    fn_body = src[fn_start:fn_start + 5000]
    assert "in_color_matrix=bt709" in fn_body, (
        "decode -vf lost `in_color_matrix=bt709`; the yuv->rgb decode drifts "
        "silently version-dependent -- Fable final-gate MF-1 regressed")
    assert "out_color_matrix=bt709" in fn_body, (
        "encode -vf lost `out_color_matrix=bt709`; the rgb->yuv encode falls "
        "to swscale default (bt601 historically) while _color_args tags "
        "output bt709 -- asymmetric round-trip color shift on every model-"
        "enhanced segment")
    assert "out_range=tv" in fn_body, (
        "encoder -vf lost `out_range=tv`; would produce full-range output "
        "against bt709 tv-range tagging (asymmetric range on top of matrix)")


def test_fast_path_does_not_apply_color_matrix_flags():
    """The fast path (off / floor / dir / black) MUST NOT gain the model-
    pipeline color flags -- byte-identity to today's shipping composite
    depends on the ffmpeg command staying unchanged. This test is a
    negative pin: `in_color_matrix` must NOT appear in _encode_segment's
    fast-path branch."""
    src = (REPO / "nodes" / "otr_silent_composite.py").read_text(encoding="utf-8")
    fn_start = src.find("def _encode_segment(")
    assert fn_start != -1
    # Search only between _encode_segment's def and the `_run_model_pipeline(`
    # call it makes at the end of the model branch (the fast-path is what
    # sits BEFORE that call).
    fast_path_end = src.find("_run_model_pipeline(", fn_start)
    assert fast_path_end != -1
    fast_body = src[fn_start:fast_path_end]
    # The fast path builds cmd via _seg_vf + _color_args; it must not use
    # the pipeline's explicit-matrix arguments.
    assert "in_color_matrix" not in fast_body, (
        "_encode_segment's fast path gained `in_color_matrix`; byte-identity "
        "to shipping composite is at risk. The fast path calls `_seg_vf` "
        "which is the ONE shared composite scale chain -- not this file's "
        "job to touch its color plumbing")
