"""``rtx_vsr``: NVIDIA RTX Video Super Resolution as an opt-in upscaler.

A fake ``nvvfx`` stands in for NVIDIA's package (no RTX card in CI). The live
receipt -- a real 4060, real OTR frames -- is in the engine's docstring.
"""
from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
import torch

from nodes._otr_upscale_engines import eng_rtx_vsr as V
from nodes._otr_upscale_engines.registry import (
    CAPABILITIES, EngineUnusable, assert_usable, get_engine)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _FakeEffect:
    instances: list = []

    def __init__(self, quality=None, strength=None, device=None, image_encoding=None):
        self.quality, self.device = quality, device
        self.output_width = self.output_height = None
        self.loaded = self.closed = False
        self.runs = 0
        _FakeEffect.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True

    def load(self):
        assert self.output_width % 8 == 0 and self.output_height % 8 == 0
        self.loaded = True

    def run(self, chw):
        assert self.loaded and chw.dim() == 3 and chw.shape[0] == 3
        self.runs += 1
        # Like the real SDK, the WHOLE picture is scaled to the requested
        # (rounded-up) size -- nearest-neighbour, so exact multiples stay
        # byte-checkable.
        frame = torch.nn.functional.interpolate(
            chw[None], size=(self.output_height, self.output_width),
            mode="nearest")[0]
        return types.SimpleNamespace(image=frame.contiguous())


def _fake_nvvfx(monkeypatch, effect_cls=_FakeEffect):
    _FakeEffect.instances = []
    levels = {"LOW": 0, "HIGH": 1, "ULTRA": 2, "DENOISE_HIGH": 3}
    mod = types.SimpleNamespace(
        VideoSuperRes=effect_cls,
        QualityLevel=types.SimpleNamespace(__members__=levels),
        ImageEncoding=types.SimpleNamespace(RGB8="rgb8"))
    monkeypatch.setitem(sys.modules, "nvvfx", mod)
    return mod

# --------------------------------------------------------------------------- #
def test_registered_opt_in_never_default():
    eng = get_engine("rtx_vsr")
    assert assert_usable("rtx_vsr", "upscale_stage") == "rtx_vsr"
    assert eng.default_roles == () and eng.intrinsic_scale == 2
    assert eng.device_backends == ("cuda",) and eng.requires_vendor == "nvidia"
    assert eng.degrades_to_off is True
    row = CAPABILITIES["rtx_vsr"]
    assert row["device_backends"] == ["cuda"] and row["requires_vendor"] == "nvidia"
    assert row["practical_without_gpu"] is False


def test_not_a_declared_dependency():
    """Proprietary and ~540 MB: opt-in, never on an AMD/Intel/Mac install."""
    for name in ("requirements.txt", "pyproject.toml"):
        assert "nvidia-vfx" not in (REPO_ROOT / name).read_text(encoding="utf-8")


def test_nvidia_workflows_upscale_with_rtx_vsr_and_the_rest_stay_off():
    """Operator, 2026-10-15: "every nvidia lane to 1080p". Each NVIDIA
    workflow variant selects rtx_vsr on cuda; the canonical graph and every
    non-NVIDIA variant keep off (the 2026-10-14 default)."""
    matrix = json.loads((REPO_ROOT / "config" / "workflow_matrix.json")
                        .read_text(encoding="utf-8"))
    nvidia = {"otr_%s.json" % r["id"].replace("otr_", "", 1)
              for r in matrix["rows"] if r.get("gpu_vendor") == "nvidia"}
    assert len(nvidia) >= 10, nvidia
    # The NVIDIA graphs that stay all open weights (operator, 2026-10-20:
    # "we skip VSR, we stay open weight"; 2026-10-21: "yes, switch to off"): no
    # closed SDK -- ffmpeg Lanczos + unsharp, the canonical default. The second is
    # its reference twin (2026-10-21: "use reference photos ... no VSR").
    open_weights = {"otr_16gb_foley_open.json", "otr_16gb_foley_open_flux3_ref.json"}
    assert open_weights <= nvidia
    seen = set()
    for path in sorted((REPO_ROOT / "workflows").glob("*.json")):
        wf = json.loads(path.read_text(encoding="utf-8"))
        for node in wf.get("nodes", []):
            if node.get("type") != "OTR_SilentComposite":
                continue
            values = node.get("widgets_values", [])
            if path.name in open_weights:
                seen.add(path.name)
                assert values[4:6] == ["off", "cpu"], path.name
            elif path.name in nvidia:
                seen.add(path.name)
                assert values[4:6] == ["rtx_vsr", "cuda"], path.name
            else:
                assert "rtx_vsr" not in values, path.name
    assert seen == nvidia


def test_fingerprint_is_metadata_only_and_stable(monkeypatch):
    monkeypatch.setattr(V, "_package_version", lambda: "0.2.0.0")
    eng = V.RtxVsr()
    first = eng.model_fingerprint_parts()
    assert first == eng.model_fingerprint_parts()
    assert first == (("nvidia_vfx", "0.2.0.0"), ("quality", "ULTRA"),
                     ("sizing", "to_canvas"))
    monkeypatch.setattr(V, "_package_version", lambda: "absent")
    assert eng.model_fingerprint_parts() != first        # installing it moves the key
    monkeypatch.setenv("OTR_RTX_VSR_QUALITY", "high")
    assert ("quality", "HIGH") in eng.model_fingerprint_parts()


def test_fingerprint_does_not_import_torch_or_nvvfx():
    code = ("import sys;"
            "from nodes._otr_upscale_engines.eng_rtx_vsr import RtxVsr;"
            "RtxVsr().model_fingerprint_parts();"
            "heavy=[m for m in ('torch','nvvfx') if m in sys.modules];"
            "print('HEAVY',heavy); sys.exit(1 if heavy else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_non_cuda_device_refuses(monkeypatch):
    _fake_nvvfx(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EngineUnusable, match="NVIDIA CUDA device"):
        V.RtxVsr().load(torch.device("cpu"))


def test_a_rocm_torch_refuses(monkeypatch):
    _fake_nvvfx(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)
    with pytest.raises(EngineUnusable, match="ROCm"):
        V.RtxVsr().load(torch.device("cuda:0"))


@pytest.mark.parametrize("quality,match", [("BOGUS", "not one of"),
                                           ("DENOISE_HIGH", "keeps the input size")])
def test_a_bad_quality_level_refuses(monkeypatch, quality, match):
    _fake_nvvfx(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setenv("OTR_RTX_VSR_QUALITY", quality)
    with pytest.raises(EngineUnusable, match=match):
        V.RtxVsr().load(torch.device("cuda:0"))


def test_upscale_is_exact_x2_with_sides_rounded_for_the_sdk_and_resized():
    """1001x501 asks the SDK for 2008x1008 (multiples of 8) and RESIZES to the
    exact 2002x1002 the composite validates -- a crop would cut off the right
    and bottom edges of a picture the SDK scaled whole (2026-10-18). One
    effect serves a size."""
    eng = V.RtxVsr()
    eng._nvvfx = types.SimpleNamespace(
        VideoSuperRes=_FakeEffect,
        QualityLevel={"ULTRA": 2}, ImageEncoding=types.SimpleNamespace(RGB8="rgb8"))
    eng._quality, eng.device = "ULTRA", torch.device("cpu")
    _FakeEffect.instances = []
    import contextlib
    orig = torch.cuda.device
    torch.cuda.device = lambda *_a: contextlib.nullcontext()
    try:
        frames = torch.rand((2, 501, 1001, 3))
        out = eng.upscale_frames(frames)
        assert tuple(out.shape) == (2, 1002, 2002, 3)
        assert (_FakeEffect.instances[-1].output_width,
                _FakeEffect.instances[-1].output_height) == (2008, 1008)
        # The right-most input column must reach the right-most output
        # column; a crop to 2002 dropped it (it lands at 2006-2007 of 2008).
        edge = torch.zeros((1, 501, 1001, 3))
        edge[:, :, -1, :] = 1.0
        assert eng.upscale_frames(edge)[0, :, -1, :].min() > 0.5
        eng.upscale_frames(torch.rand((1, 501, 1001, 3)))
        assert len(_FakeEffect.instances) == 1                    # reused
        # An on-grid size needs no resize: exact pixel doubling survives.
        on_grid = torch.rand((1, 100, 100, 3))
        assert torch.equal(eng.upscale_frames(on_grid)[0, ::2, ::2], on_grid[0])
        assert len(_FakeEffect.instances) == 2 and _FakeEffect.instances[0].closed
        eng.unload()
        assert _FakeEffect.instances[1].closed and eng.device is None
    finally:
        torch.cuda.device = orig


def test_composite_falls_back_to_off_when_rtx_vsr_cannot_run(tmp_path, monkeypatch):
    """The composite is the episode's last stage: an optional accelerator that
    cannot run here costs the sharpening, never the episode."""
    import nodes.otr_silent_composite as SC

    class _Refusing:
        name, intrinsic_scale, degrades_to_off, device = "rtx_vsr", 2, True, None

        def load(self, device):
            raise EngineUnusable("rtx_vsr", "upscale_stage", "missing_model",
                                 "no nvvfx", kind="upscale")

        def unload(self):
            raise AssertionError("a refused load is never unloaded")

    seen = {}
    off = get_engine("off")
    monkeypatch.setattr(SC, "_get_upscale_engine",
                        lambda name: _Refusing() if name == "rtx_vsr" else off)
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device", lambda v: torch.device("cpu"))

    def assemble(*a, **kw):
        seen["engine"] = kw["engine"].name
        return "out.mp4", ["stubbed"]
    monkeypatch.setattr(SC, "assemble_silent_timeline", assemble)
    monkeypatch.setattr(SC, "_has_model_eligible_clips", lambda m, **kw: True)
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")
    silent, report = SC.OTRSilentComposite().composite(
        base_video_path=str(src), canvas_w=1920, canvas_h=1080, fps=25,
        output_path=str(tmp_path / "out.mp4"), gate_in="",
        clip_manifest_json=json.dumps({"fps": 25, "clips": [{"path": "x", "n_frames": 1}]}),
        upscale_engine="rtx_vsr", upscale_device="cpu")
    assert silent == "out.mp4" and seen["engine"] == "off"
    assert "rtx_vsr unavailable here" in report and "used off" in report


def test_a_cuda_typed_on_a_machine_without_cuda_also_falls_back(tmp_path, monkeypatch):
    """QA on aa7f11af: the device resolve sits inside the gate too, and a
    multi-line reason is folded to one line in the report."""
    import nodes.otr_silent_composite as SC

    class _Engine:
        name, intrinsic_scale, degrades_to_off, device = "rtx_vsr", 2, True, None

        def load(self, device):
            raise AssertionError("never reached: the device did not resolve")

        def unload(self):
            pass

    def no_cuda(v):
        raise EngineUnusable("<upscale>", "upscale_stage", "incompatible_profile",
                             "cuda requested\nbut torch.cuda.is_available() is False",
                             kind="upscale")
    off = get_engine("off")
    monkeypatch.setattr(SC, "_get_upscale_engine",
                        lambda name: _Engine() if name == "rtx_vsr" else off)
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device", no_cuda)
    monkeypatch.setattr(SC, "assemble_silent_timeline",
                        lambda *a, **kw: ("out.mp4", ["engine=" + kw["engine"].name]))
    monkeypatch.setattr(SC, "_has_model_eligible_clips", lambda m, **kw: True)
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")
    silent, report = SC.OTRSilentComposite().composite(
        base_video_path=str(src), canvas_w=1920, canvas_h=1080, fps=25,
        output_path=str(tmp_path / "out.mp4"), gate_in="",
        clip_manifest_json=json.dumps({"fps": 25, "clips": [{"path": "x", "n_frames": 1}]}),
        upscale_engine="rtx_vsr", upscale_device="cuda")
    first_line = report.splitlines()[0]
    assert "engine=off" in report and "rtx_vsr unavailable here" in first_line
    assert "torch.cuda.is_available() is False" in first_line   # folded, not split


def test_spandrel_refusal_is_still_loud(tmp_path, monkeypatch):
    import nodes.otr_silent_composite as SC

    class _Refusing:
        name, intrinsic_scale, device = "spandrel_esrgan", 2, None

        def load(self, device):
            raise RuntimeError("missing checkpoint")

        def unload(self):
            pass

    monkeypatch.setattr(SC, "_get_upscale_engine", lambda name: _Refusing())
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device", lambda v: torch.device("cpu"))
    monkeypatch.setattr(SC, "_has_model_eligible_clips", lambda m, **kw: True)
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")
    with pytest.raises(RuntimeError, match="missing checkpoint"):
        SC.OTRSilentComposite().composite(
            base_video_path=str(src), canvas_w=1920, canvas_h=1080, fps=25,
            output_path=str(tmp_path / "out.mp4"), gate_in="",
            clip_manifest_json=json.dumps({"fps": 25, "clips": [{"path": "x", "n_frames": 1}]}),
            upscale_engine="spandrel_esrgan", upscale_device="cpu")


def _composite_with(SC, tmp_path, engine_name):
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")
    return SC.OTRSilentComposite().composite(
        base_video_path=str(src), canvas_w=1920, canvas_h=1080, fps=25,
        output_path=str(tmp_path / "out.mp4"), gate_in="",
        clip_manifest_json=json.dumps({"fps": 25, "clips": [{"path": "x", "n_frames": 1}]}),
        upscale_engine=engine_name, upscale_device="cuda")


def test_rtx_vsr_failing_mid_composite_redoes_it_with_off(tmp_path, monkeypatch):
    """QA on ece6ed34: the load gate cannot prove the accelerator survives a
    real clip (rtx_vsr reopens its effect per output size). A failure THERE
    must also cost only the sharpening: unload, composite again on off."""
    import nodes.otr_silent_composite as SC

    unloaded = []

    class _DiesMidRun:
        name, intrinsic_scale, degrades_to_off, device = "rtx_vsr", 2, True, None

        def load(self, device):
            pass

        def unload(self):
            unloaded.append(True)

    off = get_engine("off")
    calls = []

    def assemble(*a, **kw):
        calls.append(kw["engine"].name)
        if kw["engine"].name == "rtx_vsr":
            raise SC.UpscaleEngineFailed("NvVFX effect.run failed\nat 3328x1920")
        return "out.mp4", ["engine=" + kw["engine"].name]

    monkeypatch.setattr(SC, "_get_upscale_engine",
                        lambda name: _DiesMidRun() if name == "rtx_vsr" else off)
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device", lambda v: torch.device("cpu"))
    monkeypatch.setattr(SC, "assemble_silent_timeline", assemble)
    monkeypatch.setattr(SC, "_has_model_eligible_clips", lambda m, **kw: True)
    silent, report = _composite_with(SC, tmp_path, "rtx_vsr")
    assert silent == "out.mp4" and calls == ["rtx_vsr", "off"]
    assert unloaded == [True]                 # unloaded once, not again in finally
    assert "failed mid-composite" in report and "redid it with off" in report
    assert "at 3328x1920" in report.splitlines()[0]          # folded to one line


def test_a_non_degrading_engine_failing_mid_composite_still_raises(tmp_path,
                                                                   monkeypatch):
    import nodes.otr_silent_composite as SC

    class _Spandrel:
        name, intrinsic_scale, device = "spandrel_esrgan", 2, None

        def load(self, device):
            pass

        def unload(self):
            pass

    def assemble(*a, **kw):
        raise RuntimeError("esrgan blew up")

    monkeypatch.setattr(SC, "_get_upscale_engine", lambda name: _Spandrel())
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device", lambda v: torch.device("cpu"))
    monkeypatch.setattr(SC, "assemble_silent_timeline", assemble)
    monkeypatch.setattr(SC, "_has_model_eligible_clips", lambda m, **kw: True)
    with pytest.raises(RuntimeError, match="esrgan blew up"):
        _composite_with(SC, tmp_path, "spandrel_esrgan")


# --- the SDK is the user's install (Comfy Registry standards, 2026-10-30) --- #
#
# From 2.3.29 to 2.3.44 load() pip-installed nvidia-vfx itself. The registry's
# reviewer flagged that "policy-v0.7: code-install" (2.3.34, 2.3.35), and the
# published standards forbid runtime package installation through subprocess
# calls (PBUG-20261030-01). So: a missing SDK is a REFUSAL that names the pip
# line, the pack spawns nothing to get it, and the hardware refusals still come
# first so a CPU, AMD, Mac or GTX box hears its real reason, not "install a
# package it cannot use".

def _sdk_absent(monkeypatch):
    """No nvvfx; a capable RTX host unless a test says otherwise; and a spawn
    boundary that FAILS THE TEST if the engine tries to run anything at all."""
    from nodes._otr_shared import proc as otr_proc

    def never(*argv, **kw):
        pytest.fail("rtx_vsr spawned %r: the pack must never install the SDK" % (argv,))
    monkeypatch.setattr(otr_proc, "run", never)
    monkeypatch.setattr(otr_proc, "popen", never)
    monkeypatch.setitem(sys.modules, "nvvfx", None)       # absent
    import platform
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (8, 9))
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda d=None: "NVIDIA GeForce RTX 4060 Laptop GPU")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "version_info", (3, 13, 0, "final", 0))


def test_without_the_package_load_refuses_with_the_pip_line(monkeypatch):
    _sdk_absent(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    for _ in range(2):                          # stateless: the same answer twice
        with pytest.raises(EngineUnusable) as exc:
            V.RtxVsr().load(torch.device("cuda:0"))
        assert "not installed" in str(exc.value)
        assert 'pip install "nvidia-vfx' in str(exc.value)
        assert V.REQUIREMENT in str(exc.value)


def test_a_cpu_box_is_refused_for_its_hardware_not_the_package(monkeypatch):
    _sdk_absent(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EngineUnusable, match="NVIDIA CUDA device"):
        V.RtxVsr().load(torch.device("cpu"))


def test_an_amd_box_is_refused_for_its_hardware_not_the_package(monkeypatch):
    _sdk_absent(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", "6.2", raising=False)
    with pytest.raises(EngineUnusable, match="ROCm"):
        V.RtxVsr().load(torch.device("cuda:0"))


def test_a_pre_turing_card_without_the_sdk_hears_the_real_reason(monkeypatch):
    _sdk_absent(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (6, 1))
    with pytest.raises(EngineUnusable, match="Turing"):
        V.RtxVsr().load(torch.device("cuda:0"))


def test_an_arm_or_mac_host_without_the_sdk_hears_the_real_reason(monkeypatch):
    import platform
    _sdk_absent(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(platform, "machine", lambda: "aarch64")
    with pytest.raises(EngineUnusable, match="x86-64"):
        V.RtxVsr().load(torch.device("cuda:0"))


def test_a_gtx_card_without_the_sdk_hears_the_real_reason(monkeypatch):
    _sdk_absent(monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d=None: (7, 5))
    monkeypatch.setattr(torch.cuda, "get_device_name",
                        lambda d=None: "NVIDIA GeForce GTX 1660 SUPER")
    with pytest.raises(EngineUnusable, match="GTX"):
        V.RtxVsr().load(torch.device("cuda:0"))


def test_a_present_but_broken_sdk_raises_its_own_error(monkeypatch):
    """A DLL or driver error is not "missing": it surfaces as itself, never as
    the install hint for a package that is already there."""
    import builtins
    real_import = builtins.__import__

    def broken(name, *a, **kw):
        if name == "nvvfx":
            raise ImportError("DLL load failed while importing nvvfx")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", broken)
    with pytest.raises(ImportError, match="DLL load failed"):
        V._import_nvvfx()


# --- AnimateDiff / Ghost rows (operator, 2026-10-15) ----------------------- #

def _clip_row(mode=None, **kw):
    row = {"exists": True, "path": "clip.mp4", "type": "clip",
           "delivery_scale_mode": mode}
    row.update(kw)
    return row


def test_a_clean_declared_row_is_eligible_only_for_a_clean_engine():
    import nodes.otr_silent_composite as SC
    ghost = {"clips": [_clip_row(SC.DELIVERY_SCALE_MODE_CLEAN)]}
    assert SC._has_model_eligible_clips(ghost) is False
    assert SC._has_model_eligible_clips(ghost, clean_engine=True) is True
    assert SC._has_model_eligible_clips({"clips": [_clip_row()]}) is True
    assert SC._engine_is_clean_full_frame("rtx_vsr") is True
    assert SC._engine_is_clean_full_frame("spandrel_esrgan") is False
    assert SC._engine_is_clean_full_frame("off") is False
    assert SC._engine_is_clean_full_frame("no_such_engine") is False


def _route(monkeypatch, engine, *, mode, src_dims=(512, 288)):
    import nodes.otr_silent_composite as SC
    from nodes._otr_upscale_engines import _pipeline as P
    seen = {"ffmpeg": [], "model": 0}

    def fake_run(cmd):
        seen["ffmpeg"].append(cmd[cmd.index("-vf") + 1])
        return types.SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(SC, "_run", fake_run)
    monkeypatch.setattr(SC, "_color_args", lambda fb, p: [p])
    monkeypatch.setattr(P, "_probe_video_dims", lambda fb, src: (*src_dims, 25))

    def fake_model(**kw):
        seen["model"] += 1
        seen["full_frame"] = kw.get("full_frame")
    monkeypatch.setattr(SC, "_run_model_pipeline", fake_model)
    SC._encode_segment("ffmpeg", "clip.mp4", 25, "seg.mp4", w=1920, h=1080,
                       fps=25, sharpen=True, engine=engine,
                       delivery_scale_mode=mode)
    return seen


class _Clean:
    name, clean_full_frame, device = "rtx_vsr", True, "cuda:0"


class _NotClean:
    name, device = "spandrel_esrgan", "cpu"


def test_a_ghost_row_goes_through_rtx_vsr(monkeypatch):
    import nodes.otr_silent_composite as SC
    seen = _route(monkeypatch, _Clean(), mode=SC.DELIVERY_SCALE_MODE_CLEAN)
    assert seen["model"] == 1 and seen["ffmpeg"] == []
    assert seen["full_frame"] is True              # filled exactly, never padded
    legacy = _route(monkeypatch, _Clean(), mode=None)
    assert legacy["model"] == 1 and legacy["full_frame"] is False


def test_a_ghost_row_keeps_clean_lanczos_under_esrgan_or_off(monkeypatch):
    import nodes.otr_silent_composite as SC
    for engine in (_NotClean(), None):
        seen = _route(monkeypatch, engine, mode=SC.DELIVERY_SCALE_MODE_CLEAN)
        assert seen["model"] == 0
        (vf,) = seen["ffmpeg"]
        assert "flags=lanczos" in vf and "unsharp" not in vf and "pad=1920" not in vf


def test_a_ghost_row_already_at_canvas_keeps_its_declared_chain(monkeypatch):
    import nodes.otr_silent_composite as SC
    seen = _route(monkeypatch, _Clean(), mode=SC.DELIVERY_SCALE_MODE_CLEAN,
                  src_dims=(1920, 1080))
    assert seen["model"] == 0
    (vf,) = seen["ffmpeg"]
    assert "flags=lanczos" in vf and "unsharp" not in vf and "pad=1920" not in vf


def test_the_full_frame_fit_fills_any_canvas_with_no_pad():
    from nodes._otr_upscale_engines._pipeline import (
        _fit_full_frame_bhwc, _fit_and_pad_bhwc)
    x = torch.full((1, 576, 1024, 3), 0.5)          # a Ghost clip after VSR x2
    for w, h in ((1920, 1080), (1472, 832), (1280, 1024)):
        y = _fit_full_frame_bhwc(x, w, h)
        assert tuple(y.shape) == (1, h, w, 3)
        assert float(y.min()) > 0.4                 # no black rows or columns
    padded = _fit_and_pad_bhwc(x, 1280, 1024)
    assert float(padded.min()) == 0.0               # the legacy fit letterboxes


def _ghost_manifest(tmp_path):
    import nodes.otr_silent_composite as SC
    clip = tmp_path / "ghost.mp4"
    clip.write_bytes(b"fake-mp4")
    return {"fps": 25, "clips": [{"path": str(clip), "n_frames": 25,
                                  "exists": True, "type": "clip",
                                  "delivery_scale_mode": SC.DELIVERY_SCALE_MODE_CLEAN}]}


def test_a_ghost_only_episode_loads_rtx_vsr_through_the_real_gate(tmp_path, monkeypatch):
    """The node's own eligibility call (not a stub) must pass clean_engine."""
    import nodes.otr_silent_composite as SC
    loaded = []

    class _CleanEngine:
        name, intrinsic_scale, degrades_to_off = "rtx_vsr", 2, True
        clean_full_frame, device = True, None

        def load(self, device):
            loaded.append(device)

        def unload(self):
            pass

    off = get_engine("off")
    monkeypatch.setattr(SC, "_get_upscale_engine",
                        lambda name: _CleanEngine() if name == "rtx_vsr" else off)
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device", lambda v: torch.device("cpu"))
    seen = {}

    def assemble(*a, **kw):
        seen["engine"] = kw["engine"].name
        return "out.mp4", ["stubbed"]
    monkeypatch.setattr(SC, "assemble_silent_timeline", assemble)
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")
    SC.OTRSilentComposite().composite(
        base_video_path=str(src), canvas_w=1920, canvas_h=1080, fps=25,
        output_path=str(tmp_path / "out.mp4"), gate_in="",
        clip_manifest_json=json.dumps(_ghost_manifest(tmp_path)),
        upscale_engine="rtx_vsr", upscale_device="cuda")
    assert loaded and seen["engine"] == "rtx_vsr"


def test_a_ghost_only_key_ignores_unsharp_but_carries_the_engine(tmp_path, monkeypatch):
    import nodes.otr_silent_composite as SC
    manifest = json.dumps(_ghost_manifest(tmp_path))
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")

    def key(engine, unsharp):
        monkeypatch.setenv("OTR_COMPOSITE_UNSHARP_AMOUNT", unsharp)
        return SC.OTRSilentComposite.IS_CHANGED(
            base_video_path=str(src), canvas_w=1920, canvas_h=1080, fps=25,
            output_path=str(tmp_path / "out.mp4"), clip_manifest_json=manifest,
            upscale_engine=engine, upscale_device="cuda")
    assert key("rtx_vsr", "0.4") == key("rtx_vsr", "0.9")    # no unsharp reaches a Ghost row
    assert "nvidia_vfx" in key("rtx_vsr", "0.4")             # the model the rows now use
    assert key("off", "0.4") == key("off", "0.9")


def test_a_mixed_manifest_key_still_tracks_unsharp(tmp_path, monkeypatch):
    import nodes.otr_silent_composite as SC
    manifest = _ghost_manifest(tmp_path)
    legacy = tmp_path / "legacy.mp4"
    legacy.write_bytes(b"fake-mp4")
    manifest["clips"].append({"path": str(legacy), "n_frames": 25,
                              "exists": True, "type": "clip"})
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")

    def key(unsharp):
        monkeypatch.setenv("OTR_COMPOSITE_UNSHARP_AMOUNT", unsharp)
        return SC.OTRSilentComposite.IS_CHANGED(
            base_video_path=str(src), canvas_w=1920, canvas_h=1080, fps=25,
            output_path=str(tmp_path / "out.mp4"),
            clip_manifest_json=json.dumps(manifest),
            upscale_engine="rtx_vsr", upscale_device="cuda")
    assert key("0.4") != key("0.9")                # the legacy row can carry unsharp


def test_a_non_upscaler_failure_mid_composite_is_not_redone(tmp_path, monkeypatch):
    """Gemini second opinion: only the model's own failure earns the redo. An
    ffmpeg error, a refused executable or an underrun propagates at once."""
    import nodes.otr_silent_composite as SC
    from nodes._otr_shared.proc import ExecutableNotAllowed

    class _CleanEng:
        name, intrinsic_scale, degrades_to_off, device = "rtx_vsr", 2, True, None

        def load(self, device):
            pass

        def unload(self):
            pass

    calls = []

    def assemble(*a, **kw):
        calls.append(kw["engine"].name)
        raise ExecutableNotAllowed("cmd.exe is not an allowed executable")
    off = get_engine("off")
    monkeypatch.setattr(SC, "_get_upscale_engine",
                        lambda name: _CleanEng() if name == "rtx_vsr" else off)
    monkeypatch.setattr(SC, "_assert_upscale_usable", lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device", lambda v: torch.device("cpu"))
    monkeypatch.setattr(SC, "assemble_silent_timeline", assemble)
    monkeypatch.setattr(SC, "_has_model_eligible_clips", lambda m, **kw: True)
    with pytest.raises(ExecutableNotAllowed):
        _composite_with(SC, tmp_path, "rtx_vsr")
    assert calls == ["rtx_vsr"]                    # no second, off pass


def test_every_gpu_spatial_step_sits_inside_the_typed_boundary():
    """Codex review 2026-10-16: a CUDA OOM in the device transfer, the fit or
    the conversion back is the upscaler's failure too, so it must raise
    UpscaleEngineFailed (and earn the off redo); the encoder write must not."""
    import inspect
    import nodes.otr_silent_composite as SC
    src = inspect.getsource(SC._run_model_pipeline)
    start = src.index("try:\n                # engine.device")
    end = src.index("raise UpscaleEngineFailed(", start)
    guarded = src[start:end]
    for step in (".to(engine.device)", "engine.upscale_frames(",
                 "_validate_engine_output(", "_fit(out_bhwc", ".cpu().numpy()"):
        assert step in guarded, step
    assert "enc.stdin.write" not in guarded
