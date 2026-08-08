"""Queue item 8 (2026-08-08): the single-base composite path MUST NOT load
the upscale engine.

Sonnet 5 QA-on-diff MF-2: `composite()`'s `engine.load(device)` used to
fire unconditionally whenever `upscale_engine != "off"`, BEFORE the
`assemble` branch was checked. But `normalize_to_silent_canonical`
(the single-base path when `clip_manifest_json` is empty / has no
`"clips"`) does not consume the engine at all -- loading the model there
was dead VRAM churn AND, worse, would abort a render that structurally
never needed the model if `engine.load()` raised (missing model file,
missing spandrel install, CUDA OOM).

This test drives `OTRSilentComposite.composite()` with `assemble=False`
(empty clip_manifest_json) + a non-off engine and asserts that
`engine.load()` is never called and `engine.unload()` is never called.
Uses a mock engine so we don't need a real spandrel/model on disk.
"""
from __future__ import annotations

import json
from unittest import mock

import pytest


class _RecordingEngine:
    """Mock upscale engine that records every load/unload call so the test
    can assert neither ever fires on the single-base path."""

    name = "spandrel_esrgan"
    intrinsic_scale = 2

    def __init__(self):
        self.load_calls = []
        self.unload_calls = 0
        self.device = None

    def load(self, device):
        self.load_calls.append(device)
        self.device = device

    def unload(self):
        self.unload_calls += 1
        self.device = None

    def upscale_frames(self, frames):
        raise AssertionError(
            "single-base composite path must not invoke upscale_frames")


def test_composite_single_base_never_loads_engine(tmp_path, monkeypatch):
    """assemble=False + non-off engine widget -> engine.load NOT called."""
    from nodes.otr_silent_composite import OTRSilentComposite
    from nodes._otr_upscale_engines import registry as reg

    mock_engine = _RecordingEngine()
    # Patch the registry's get_engine + assert_usable to return our recorder.
    monkeypatch.setattr(reg, "get_engine", lambda name: mock_engine)
    monkeypatch.setattr(reg, "assert_usable",
                          lambda name, role: name)
    # ALSO patch the SAME symbols on `otr_silent_composite` (imported at
    # module top; independent references).
    import nodes.otr_silent_composite as SC
    monkeypatch.setattr(SC, "_get_upscale_engine", lambda name: mock_engine)
    monkeypatch.setattr(SC, "_assert_upscale_usable",
                          lambda name, role: name)

    # Also stub normalize_to_silent_canonical so we don't need a real ffmpeg
    # to run through -- we only care about the load/unload gating.
    monkeypatch.setattr(SC, "normalize_to_silent_canonical",
                          lambda src, out, **kw: (out, ["stubbed"]))

    # Create a placeholder source file (existence checked before ffmpeg).
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")

    node = OTRSilentComposite()
    silent, report = node.composite(
        base_video_path=str(src),
        canvas_w=1920, canvas_h=1080, fps=25,
        ffmpeg="ffmpeg",
        output_path=str(tmp_path / "out.mp4"),
        gate_in="",
        clip_manifest_json="{}",  # NO "clips" key -> single-base path
        upscale_engine="spandrel_esrgan",
        upscale_device="cuda:0",
    )
    # The core assertion: no load call, no unload call.
    assert mock_engine.load_calls == [], (
        f"engine.load was called {mock_engine.load_calls!r} times on the "
        f"single-base composite path; Sonnet 5 QA-on-diff MF-2 regressed")
    assert mock_engine.unload_calls == 0, (
        f"engine.unload was called {mock_engine.unload_calls} times; "
        f"the load-gate + unload-gate must move together")


def test_composite_single_base_survives_engine_load_error_potential(tmp_path, monkeypatch):
    """If a hypothetical engine.load would raise (missing model / CUDA OOM),
    the single-base composite path must NOT surface that error to the user
    because the load never runs on this path -- verified by using an engine
    whose load() unconditionally raises."""
    from nodes.otr_silent_composite import OTRSilentComposite
    import nodes.otr_silent_composite as SC

    class _ExplodingEngine:
        name = "spandrel_esrgan"
        intrinsic_scale = 2

        def load(self, device):
            raise RuntimeError("simulated missing model / OOM at load time")

        def unload(self):
            pass

    monkeypatch.setattr(SC, "_get_upscale_engine",
                          lambda name: _ExplodingEngine())
    monkeypatch.setattr(SC, "_assert_upscale_usable",
                          lambda name, role: name)
    monkeypatch.setattr(SC, "normalize_to_silent_canonical",
                          lambda src, out, **kw: (out, ["stubbed"]))

    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")

    node = OTRSilentComposite()
    # This MUST succeed -- the exploding load must never fire.
    silent, report = node.composite(
        base_video_path=str(src),
        canvas_w=1920, canvas_h=1080, fps=25,
        ffmpeg="ffmpeg",
        output_path=str(tmp_path / "out.mp4"),
        gate_in="",
        clip_manifest_json="{}",  # single-base path
        upscale_engine="spandrel_esrgan",
        upscale_device="cuda:0",
    )
    assert silent, "single-base composite silently failed despite exploding engine"
    assert "OTR_SilentComposite OK" in report or "stubbed" in report


def test_composite_assemble_path_still_loads_engine(tmp_path, monkeypatch):
    """The load gate must NOT accidentally break the assemble path -- on
    a clip-manifest-bearing render with a non-off engine, engine.load
    MUST be called exactly once."""
    from nodes.otr_silent_composite import OTRSilentComposite
    from nodes._otr_upscale_engines import registry as reg
    import nodes.otr_silent_composite as SC

    mock_engine = _RecordingEngine()
    monkeypatch.setattr(SC, "_get_upscale_engine", lambda name: mock_engine)
    monkeypatch.setattr(SC, "_assert_upscale_usable",
                          lambda name, role: name)
    monkeypatch.setattr(SC, "_resolve_upscale_device",
                          lambda v: "cuda:0-resolved")
    monkeypatch.setattr(SC, "assemble_silent_timeline",
                          lambda *a, **kw: (kw.get("out_path", "out.mp4")
                                             if "out_path" in kw
                                             else "out.mp4", ["stubbed"]))
    # Force the assemble branch by providing a manifest with clips.
    manifest = {"fps": 25, "clips": [{"path": "x", "n_frames": 1}]}
    src = tmp_path / "source.mp4"
    src.write_bytes(b"fake-mp4")

    node = OTRSilentComposite()
    _silent, _report = node.composite(
        base_video_path=str(src),
        canvas_w=1920, canvas_h=1080, fps=25,
        ffmpeg="ffmpeg",
        output_path=str(tmp_path / "out.mp4"),
        gate_in="",
        clip_manifest_json=json.dumps(manifest),
        upscale_engine="spandrel_esrgan",
        upscale_device="cuda:0",
    )
    assert len(mock_engine.load_calls) == 1, (
        f"assemble path must load the engine exactly once; got "
        f"{mock_engine.load_calls!r}")
    assert mock_engine.unload_calls == 1
