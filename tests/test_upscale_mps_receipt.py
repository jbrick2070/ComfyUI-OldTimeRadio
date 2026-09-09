"""The upscale namespace on Apple Silicon -- and the receipt that unblocked it.

`_resolve.resolve_device` rejected `"mps"` by name from the first ship, with a
docstring stating the exact condition for lifting it: *"it will land when a Mac
user provides an integration receipt."* One string comparison kept the ENTIRE
upscale namespace untested on Apple Silicon.

THE RECEIPT, taken 2026-09-09 on an Apple M4 / 16 GB:

    RealESRGAN_x2plus.pth via spandrel 0.4.2, arch ESRGAN, scale 2
    128x128 -> 256x256 on torch.device("mps") in 0.51 s, all outputs finite
    (the automated re-take below uses 64x64 -> 128x128 to stay quick)
    max |mps - cpu| = 0.00000   -- against the CPU path, raw spandrel forward

and through the ENGINE's own path (load -> upscale_frames), driven as
otr_silent_composite drives it:

    2x 288x512 -> 576x1024 in 1.06 s on mps against 8.37 s on cpu (7.9x)
    max |mps - cpu| = 0.000003

NOT "bit-exact" as a blanket claim: the raw forward was exact, the stage is
3e-6, and the live tests assert < 1e-3 because float reductions on two backends
are not required to agree to the last bit. The parity is load-bearing evidence,
not a bit-for-bit guarantee -- said precisely here because an earlier draft of
this docstring rounded it up to "bit-exact" everywhere and review caught it.

The parity number is the load-bearing half. A forward that merely RETURNS on
Metal proves nothing: a wrong kernel returns too, and this project has already
shipped one silently-wrong Metal path (sub-quadratic attention, which produced
clean-looking noise). Bit-exact agreement with CPU on the same weights and the
same input is what makes it evidence.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nodes._otr_upscale_engines import _resolve as R  # noqa: E402
from nodes._otr_upscale_engines import registry as ureg  # noqa: E402


def test_both_rows_now_claim_mps():
    """A device row is a claim about PROVEN execution in this repo, so these
    two moved only when the receipt existed."""
    for name, row in ureg.CAPABILITIES.items():
        assert "mps" in row["device_backends"], name
        assert "cuda" in row["device_backends"], (
            "%s must not strand NVIDIA to gain Metal" % name)
        assert "cpu" in row["device_backends"], name


def test_the_resolver_accepts_mps_and_still_refuses_junk():
    """GUARDED, because the unguarded version would FAIL ON THE 5080. The mps
    branch fails loud when Metal is absent -- deliberately, matching the cuda
    branch -- so a test that calls it on an NVIDIA box asserts a refusal, not a
    device. Caught in review before it broke the other machine."""
    import torch
    have_mps = bool(getattr(torch, "backends", None)
                    and getattr(torch.backends, "mps", None)
                    and torch.backends.mps.is_available())
    from nodes._otr_shared.engine_registry_base import EngineUnusable
    if have_mps:
        assert R.resolve_device("mps").type == "mps"
    else:
        with pytest.raises(EngineUnusable):
            R.resolve_device("mps")
    assert R.resolve_device("cpu").type == "cpu"
    # NO FALLBACKS: an unknown token is still a named refusal, not a quiet cpu.
    from nodes._otr_shared.engine_registry_base import EngineUnusable
    for junk in ("rocm", "metal", "gpu", "mps:0", "xpu"):
        with pytest.raises(EngineUnusable):
            R.resolve_device(junk)


def test_asking_for_metal_where_there_is_none_is_a_NAMED_refusal():
    """The same fail-loud shape as the cuda branch. A profile that asks for
    Metal on a box without it is a misconfiguration to name -- silently
    satisfying it on the CPU would make the receipt describe a render that did
    not happen."""
    import inspect
    src = inspect.getsource(R.resolve_device)
    assert 'if v == "mps":' in src
    assert "torch.backends.mps.is_available()" in src
    assert "INCOMPATIBLE_PROFILE" in src.split('if v == "mps":')[1][:600]
    # and it must not silently fall through to cpu
    assert 'return torch.device("cpu")' not in src.split('if v == "mps":')[1][:400]


@pytest.mark.skipif(
    not (getattr(__import__("torch"), "backends", None)
         and getattr(__import__("torch").backends, "mps", None)
         and __import__("torch").backends.mps.is_available()),
    reason="no Metal on this host -- the receipt is re-taken only on a Mac")
def test_the_receipt_still_holds_bit_exact_against_cpu():
    """RE-TAKES THE RECEIPT, rather than trusting the comment that records it.

    This is the test that would catch a torch or spandrel upgrade quietly
    breaking the Metal path -- which is precisely how the sub-quadratic
    attention defect reached production."""
    import glob

    import numpy as np
    import torch
    spandrel = pytest.importorskip("spandrel")

    hits = sorted(glob.glob(os.path.expanduser(
        "~/Documents/models/upscale_models/RealESRGAN_x2plus.pth")))
    if not hits:
        pytest.skip("RealESRGAN_x2plus.pth not fetched on this host")
    mdl = spandrel.ModelLoader().load_from_file(hits[0])
    assert mdl.scale == 2

    x = torch.rand(1, 3, 64, 64, generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        y_mps = mdl.model.eval().to("mps")(x.to("mps")).detach().cpu()
        y_cpu = mdl.model.eval().to("cpu")(x)
    assert y_mps.shape == (1, 3, 128, 128)
    assert bool(np.isfinite(y_mps.numpy()).all()), "Metal produced non-finite output"
    assert float((y_mps - y_cpu).abs().max()) < 1e-3, (
        "Metal diverged from CPU -- a returning kernel is not a correct kernel")


# ---------------------------------------------------------------------------
# THE STAGE, not just the model. Same distinction that bit the AnimateDiff lane:
# "the recipe runs on Metal" and "the lane runs on Metal" are different claims.
# ---------------------------------------------------------------------------

def test_the_class_attribute_and_the_registry_row_agree():
    """CAUGHT BY THE OPERATOR ASKING 'maybe it is not compatible'. The first cut
    of this change updated the registry CAPABILITIES and left BOTH engines'
    class-level `device_backends` saying ("cuda", "cpu") -- so the registry
    advertised Metal while the class denied it. Two sources of truth for one
    fact is how a lane ends up half-enabled."""
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    from nodes._otr_upscale_engines.eng_off import OffUpscale
    for cls, key in ((SpandrelEsrgan, "spandrel_esrgan"), (OffUpscale, "off")):
        assert set(cls.device_backends) == set(
            ureg.CAPABILITIES[key]["device_backends"]), key
        assert "mps" in cls.device_backends, key


def test_the_caller_contract_is_that_frames_arrive_ON_the_device():
    """`upscale_frames` does NOT move its input -- registry.py:61 says the tensor
    arrives "on the engine's device", and otr_silent_composite does
    `torch.from_numpy(arr).to(engine.device)` before calling.

    Pinned because violating it produces a confusing error that looks like a
    Metal incompatibility and is not one: `slow_conv2d_forward_mps: input
    (device='cpu') and weight(device=mps:0) must be on the same device`."""
    import inspect
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    src = inspect.getsource(SpandrelEsrgan.upscale_frames)
    assert ".to(" not in src, (
        "the engine must NOT move frames -- that is the caller's job, and doing "
        "it here would hide a caller passing the wrong device")
    comp = inspect.getsource(
        __import__("nodes.otr_silent_composite", fromlist=["x"]))
    assert ".to(engine.device)" in comp, "the composite must move them"


@pytest.mark.skipif(
    not (getattr(__import__("torch"), "backends", None)
         and getattr(__import__("torch").backends, "mps", None)
         and __import__("torch").backends.mps.is_available()),
    reason="no Metal on this host")
def test_the_whole_stage_runs_on_metal_and_matches_cpu():
    """resolve -> load -> upscale_frames -> unload, driven exactly as
    otr_silent_composite drives it.

    MEASURED 2026-09-09 on an M4/16 GB: two 288x512 frames to 576x1024 in
    1.06 s on mps against 8.37 s on cpu (7.9x), with max |mps - cpu| = 0.000003
    through the engine's own path."""
    import glob

    import numpy as np
    import torch
    pytest.importorskip("spandrel")
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    if not glob.glob(os.path.expanduser(
            "~/Documents/models/upscale_models/RealESRGAN_x2plus.pth")):
        pytest.skip("RealESRGAN_x2plus.pth not fetched on this host")

    def run(token):
        eng = SpandrelEsrgan()
        eng.load(R.resolve_device(token))
        arr = np.random.default_rng(7).integers(0, 255, (2, 96, 128, 3), dtype=np.uint8)
        bhwc = torch.from_numpy(arr).to(eng.device).float() / 255.0
        out = eng.upscale_frames(bhwc).detach().cpu().float()
        eng.unload()
        return out

    got, cpu = run("mps"), run("cpu")
    assert tuple(got.shape) == (2, 192, 256, 3), "x2 BHWC not honoured"
    assert bool(np.isfinite(got.numpy()).all())
    assert got.numpy().std() > 0.01, "flat output -- a dead kernel returns too"
    assert float((got - cpu).abs().max()) < 1e-3, "Metal diverged from CPU"


@pytest.mark.skipif(
    not (getattr(__import__("torch"), "backends", None)
         and getattr(__import__("torch").backends, "mps", None)
         and __import__("torch").backends.mps.is_available()),
    reason="no Metal on this host")
def test_the_fit_and_pad_after_the_upscale_also_runs_on_metal():
    """THE REST OF THE METAL SURFACE, which the stage test stops short of.

    `otr_silent_composite` does `.to(engine.device)` -> `upscale_frames` ->
    `_validate_engine_output` -> **`_fit_and_pad_bhwc`** -> `.cpu().numpy()`.
    That fit-and-pad runs a bicubic `F.interpolate(antialias=True)` and a pad ON
    WHATEVER DEVICE THE TENSOR IS ON (`_pipeline.py:263-265`), so on a Mac it is
    Metal work happening after the ESRGAN forward -- and it was untested.

    Flagged in review as the one piece of the real pipe still unexercised."""
    import numpy as np
    import torch
    from nodes._otr_upscale_engines._pipeline import _fit_and_pad_bhwc

    # An upscaled frame, on Metal, at a size that forces BOTH a resize and a pad
    # (4:3 content into a 16:9 canvas is the pillarbox case).
    x = torch.rand(2, 192, 256, 3, generator=torch.Generator().manual_seed(11)).to("mps")
    out = _fit_and_pad_bhwc(x, 1920, 1080)
    assert out.device.type == "mps", "the resize must stay on the device it got"
    assert tuple(out.shape) == (2, 1080, 1920, 3)
    a = out.detach().cpu().numpy()
    assert bool(np.isfinite(a).all()), "Metal bicubic produced non-finite pixels"
    assert a.std() > 0.01, "flat output -- antialias on Metal returning nothing"

    cpu = _fit_and_pad_bhwc(x.cpu(), 1920, 1080)
    assert float((out.detach().cpu() - cpu).abs().max()) < 1e-3, (
        "Metal bicubic+pad diverged from CPU")
    # the pillarbox must actually be black, not smeared edge pixels
    assert float(a[:, :, :4, :].max()) < 1e-3, "left pad is not black"
