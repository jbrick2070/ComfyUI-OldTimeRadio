"""Queue item 8 (2026-08-08): _fit_and_pad_bhwc helper.

The r3-amended helper must:
* decrease-fit in BOTH directions (upscale a small input, downscale a large one);
* mirror ffmpeg's `scale=W:H:force_original_aspect_ratio=decrease` +
  `pad=W:H:(ow-iw)/2:(oh-ih)/2:color=black` exactly at the tensor level;
* return a `.contiguous()` tensor (Antigravity r4 MF-2 -- non-contiguous ==
  downstream .numpy() crash).

Codex r4 MF-5: r2's downscale-only spec broke the ship-canvas case
(832 -> x2 = 1664 < 1920). This test pins the correct decrease-fit behavior.
"""
from __future__ import annotations

import pytest

pytest.importorskip("torch")


def _make_bhwc(b, h, w, c=3, fill=0.5):
    """Return a (b, h, w, c) float32 tensor on CPU with a known value."""
    import torch
    x = torch.full((b, h, w, c), float(fill), dtype=torch.float32)
    return x


def test_upscale_case_source_smaller_than_canvas():
    """832 -> x2 = 1664, must scale UP to hit at least one canvas dim (1920 or
    1080) with aspect preserved."""
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    src = _make_bhwc(1, 960, 1664)  # 832 -> x2 = 1664 wide, 480 -> x2 = 960 tall
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    assert tuple(out.shape) == (1, 1080, 1920, 3), (
        f"canvas mismatch: got {tuple(out.shape)}, expected (1,1080,1920,3)")


def test_downscale_case_source_larger_than_canvas():
    """2160 -> canvas 1080, decrease-fit downscales without cropping."""
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    src = _make_bhwc(1, 2160, 3840)
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    assert tuple(out.shape) == (1, 1080, 1920, 3)


def test_output_is_contiguous():
    """Antigravity r4 MF-2: non-contiguous tensor -> .numpy() crash downstream."""
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    src = _make_bhwc(2, 480, 832)
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    assert out.is_contiguous(), "output tensor is not contiguous"


def test_output_preserves_batch_dim():
    """N frames in, N frames out."""
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    src = _make_bhwc(4, 480, 832)
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    assert out.shape[0] == 4


def test_output_preserves_channels():
    """Channel dim preserved (3 = RGB)."""
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    src = _make_bhwc(1, 480, 832)
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    assert out.shape[3] == 3


def test_output_is_float32():
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    src = _make_bhwc(1, 480, 832)
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    assert out.dtype.__str__().endswith("float32"), f"dtype {out.dtype!r} not float32"


def test_padding_is_black():
    """When the fit dims are smaller than the canvas along one axis, the
    pad regions must be exactly zero (BLACK). Verify by summing corner
    pixels that must be zero for a 480x832 -> 1920x1080 case where the
    aspect-preserving fit lands at 1080x1872 (width-fit) OR similar."""
    import torch
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    # Use a solid-white input so the fitted region is 1.0 and the pad is 0.
    src = torch.ones((1, 480, 832, 3), dtype=torch.float32)
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    # 480/832 aspect * min(1920/832, 1080/480) = min(2.308, 2.25) = 2.25.
    # new_w = 832*2.25 = 1872; new_h = 480*2.25 = 1080. So no top/bot pad,
    # LEFT/RIGHT pad = (1920-1872)/2 = 24 each side.
    # Pixel (0, 500, 0, :) is in the leftmost pad column and MUST be 0.
    left_corner = out[0, 500, 0, :]  # any row, leftmost col
    assert torch.allclose(left_corner, torch.zeros(3)), (
        f"left pad column not zero: {left_corner.tolist()!r}")
    right_corner = out[0, 500, 1919, :]  # rightmost col
    assert torch.allclose(right_corner, torch.zeros(3)), (
        f"right pad column not zero: {right_corner.tolist()!r}")


def test_ship_case_832x480_to_1920x1080_after_2x_model():
    """The ship-profile flow: model 2x turns 832x480 into 1664x960; the
    helper's job then is to fit-with-pad to 1920x1080. Verify the exact
    output canvas."""
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc
    src = _make_bhwc(1, 960, 1664)
    out = _fit_and_pad_bhwc(src, canvas_w=1920, canvas_h=1080)
    assert tuple(out.shape) == (1, 1080, 1920, 3)
    # 1664/960 vs canvas: min(1920/1664, 1080/960) = min(1.154, 1.125) = 1.125.
    # new_w = 1664*1.125 = 1872; new_h = 960*1.125 = 1080. Pad = 24 left/right.
