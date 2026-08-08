"""Queue item 8 (2026-08-08): SpandrelEsrgan adapter (mocked spandrel loader).

Covers:
* device pinning: ModelLoader(device=<str>) is called with the resolved device
* failure taxonomy: absent model file -> EngineUnusable(MISSING_MODEL);
  wrong-SHA -> EngineUnusable(MISSING_MODEL); scale mismatch ->
  EngineUnusable(INCOMPATIBLE_PROFILE)
* per-frame descriptor invocation (spandrel enforces batch=1)
* BHWC -> BCHW permute is applied inside upscale_frames
* .contiguous() after the final BCHW -> BHWC permute (Antigravity r4 MF-2)
* graceful bare-venv fallback when folder_paths is not importable
  (Antigravity r4 MF-4)

Every test uses mocked spandrel.ModelLoader; no real weights are loaded and
no CUDA is exercised.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from unittest import mock

import pytest

torch = pytest.importorskip("torch")

from nodes._otr_shared.engine_registry_base import (
    EngineUnusable, EngineUsabilityReason)


class _MockDescriptor:
    """Stand-in for spandrel.ImageModelDescriptor."""

    def __init__(self, scale=2):
        self.scale = scale
        self.device = torch.device("cpu")
        # Track invocation for per-frame-loop assertions.
        self.calls = []

    def __call__(self, x):
        """Enforce (1,C,H,W) input (spandrel's real contract)."""
        assert x.ndim == 4 and x.shape[0] == 1, (
            f"descriptor got shape {tuple(x.shape)}; must be (1,C,H,W)")
        self.calls.append(tuple(x.shape))
        # Return 2x upsample: (1, C, H, W) -> (1, C, H*2, W*2).
        # Use nearest so we don't depend on interpolation nuances.
        return torch.nn.functional.interpolate(
            x, scale_factor=self.scale, mode="nearest")

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self


def _make_mock_loader(descriptor):
    """Build a stand-in for spandrel.ModelLoader that captures device."""
    mock_loader_cls = mock.MagicMock()
    mock_loader_instance = mock.MagicMock()
    mock_loader_instance.load_from_file = mock.MagicMock(return_value=descriptor)
    mock_loader_cls.return_value = mock_loader_instance
    return mock_loader_cls, mock_loader_instance


def _prepare_model_dir(tmp_path):
    """Create a fake model file. Real bytes so hash-check tests work."""
    d = tmp_path / "upscale_models"
    d.mkdir()
    p = d / "RealESRGAN_x2plus.pth"
    p.write_bytes(b"fake_weights_content")
    return d, p


def test_load_calls_model_loader_with_resolved_device(tmp_path):
    """The adapter must pass the resolved device to ModelLoader(device=...)."""
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    model_dir, _model_path = _prepare_model_dir(tmp_path)
    descriptor = _MockDescriptor(scale=2)
    mock_loader_cls, _mock_loader_instance = _make_mock_loader(descriptor)

    mock_spandrel = mock.MagicMock()
    mock_spandrel.ModelLoader = mock_loader_cls
    mock_folder_paths = mock.MagicMock()
    mock_folder_paths.get_folder_paths = mock.MagicMock(return_value=[str(model_dir)])

    engine = SpandrelEsrgan()
    with mock.patch.dict("sys.modules",
                          {"spandrel": mock_spandrel,
                           "folder_paths": mock_folder_paths}):
        engine.load(torch.device("cpu"))
    mock_loader_cls.assert_called_once()
    call_kwargs = mock_loader_cls.call_args.kwargs
    assert "device" in call_kwargs
    assert call_kwargs["device"] == "cpu"
    assert engine.device == torch.device("cpu")


def test_load_raises_missing_model_when_file_absent(tmp_path):
    """No RealESRGAN file anywhere -> classified EngineUnusable(MISSING_MODEL)."""
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    empty_dir = tmp_path / "upscale_models"
    empty_dir.mkdir()
    mock_spandrel = mock.MagicMock()
    mock_spandrel.ModelLoader = mock.MagicMock()
    mock_folder_paths = mock.MagicMock()
    mock_folder_paths.get_folder_paths = mock.MagicMock(return_value=[str(empty_dir)])
    engine = SpandrelEsrgan()
    with mock.patch.dict("sys.modules",
                          {"spandrel": mock_spandrel,
                           "folder_paths": mock_folder_paths}):
        with pytest.raises(EngineUnusable) as excinfo:
            engine.load(torch.device("cpu"))
    assert excinfo.value.reason == EngineUsabilityReason.MISSING_MODEL


def test_load_raises_missing_model_when_sha_mismatches(tmp_path):
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    model_dir, model_path = _prepare_model_dir(tmp_path)
    descriptor = _MockDescriptor(scale=2)
    mock_loader_cls, _mli = _make_mock_loader(descriptor)
    mock_spandrel = mock.MagicMock()
    mock_spandrel.ModelLoader = mock_loader_cls
    mock_folder_paths = mock.MagicMock()
    mock_folder_paths.get_folder_paths = mock.MagicMock(return_value=[str(model_dir)])

    engine = SpandrelEsrgan()
    engine._model_sha256 = "deadbeef" * 8  # pinned, but wrong

    with mock.patch.dict("sys.modules",
                          {"spandrel": mock_spandrel,
                           "folder_paths": mock_folder_paths}):
        with pytest.raises(EngineUnusable) as excinfo:
            engine.load(torch.device("cpu"))
    assert excinfo.value.reason == EngineUsabilityReason.MISSING_MODEL
    assert "sha256" in str(excinfo.value).lower()


def test_load_raises_incompatible_profile_on_scale_mismatch(tmp_path):
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    model_dir, _mp = _prepare_model_dir(tmp_path)
    # Descriptor reports scale=4 but engine expects scale=2 -> mismatch.
    descriptor = _MockDescriptor(scale=4)
    mock_loader_cls, _mli = _make_mock_loader(descriptor)
    mock_spandrel = mock.MagicMock()
    mock_spandrel.ModelLoader = mock_loader_cls
    mock_folder_paths = mock.MagicMock()
    mock_folder_paths.get_folder_paths = mock.MagicMock(return_value=[str(model_dir)])

    engine = SpandrelEsrgan()
    with mock.patch.dict("sys.modules",
                          {"spandrel": mock_spandrel,
                           "folder_paths": mock_folder_paths}):
        with pytest.raises(EngineUnusable) as excinfo:
            engine.load(torch.device("cpu"))
    assert excinfo.value.reason == EngineUsabilityReason.INCOMPATIBLE_PROFILE


def test_upscale_frames_calls_descriptor_per_frame(tmp_path):
    """spandrel enforces batch=1; the adapter must loop internally."""
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    model_dir, _mp = _prepare_model_dir(tmp_path)
    descriptor = _MockDescriptor(scale=2)
    mock_loader_cls, _mli = _make_mock_loader(descriptor)
    mock_spandrel = mock.MagicMock()
    mock_spandrel.ModelLoader = mock_loader_cls
    mock_folder_paths = mock.MagicMock()
    mock_folder_paths.get_folder_paths = mock.MagicMock(return_value=[str(model_dir)])

    engine = SpandrelEsrgan()
    with mock.patch.dict("sys.modules",
                          {"spandrel": mock_spandrel,
                           "folder_paths": mock_folder_paths}):
        engine.load(torch.device("cpu"))
    # 3-frame batch. spandrel should be invoked 3 times, each with shape (1,C,H,W).
    frames_bhwc = torch.rand((3, 8, 8, 3), dtype=torch.float32)
    out = engine.upscale_frames(frames_bhwc)
    assert len(descriptor.calls) == 3
    for shape in descriptor.calls:
        assert shape[0] == 1  # per-frame batch dim
    # Output is BHWC with 2x spatial, batch preserved, contiguous.
    assert tuple(out.shape) == (3, 16, 16, 3)
    assert out.is_contiguous(), "output not contiguous (Antigravity r4 MF-2)"


def test_unload_clears_device_and_descriptor(tmp_path):
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    model_dir, _mp = _prepare_model_dir(tmp_path)
    descriptor = _MockDescriptor(scale=2)
    mock_loader_cls, _mli = _make_mock_loader(descriptor)
    mock_spandrel = mock.MagicMock()
    mock_spandrel.ModelLoader = mock_loader_cls
    mock_folder_paths = mock.MagicMock()
    mock_folder_paths.get_folder_paths = mock.MagicMock(return_value=[str(model_dir)])

    engine = SpandrelEsrgan()
    with mock.patch.dict("sys.modules",
                          {"spandrel": mock_spandrel,
                           "folder_paths": mock_folder_paths}):
        engine.load(torch.device("cpu"))
    assert engine._descriptor is not None
    engine.unload()
    assert engine._descriptor is None
    assert engine.device is None


def test_load_falls_back_when_folder_paths_missing(tmp_path):
    """Antigravity r4 MF-4: bare venv without ComfyUI on sys.path -> folder_paths
    ImportError -> adapter must NOT crash; must fall back to a repo-relative
    models/upscale_models/ candidate.

    Sonnet 5 QA MF-1: the fallback resolves parents[4] from
    ``nodes/_otr_upscale_engines/eng_spandrel_esrgan.py`` to reach ComfyUI's
    root -- parents[3] would land under custom_nodes/ and silently miss the
    real models dir. The r4 first-draft used parents[3] and the test
    reproduced the same arithmetic so it went green against the wrong dir.
    This fixed version reproduces the CORRECTED arithmetic + asserts the
    fallback lands under the actual ComfyUI root."""
    from nodes._otr_upscale_engines.eng_spandrel_esrgan import SpandrelEsrgan
    # Prepare a fake model file at the FALLBACK location.
    # eng_spandrel_esrgan resolves parents[4] from its own path (ComfyUI root).
    from nodes._otr_upscale_engines import eng_spandrel_esrgan as EM
    fallback = Path(EM.__file__).resolve().parents[4] / "models" / "upscale_models"
    # Belt+suspenders: the fallback path MUST end in .../models/upscale_models
    # under a directory that ALSO contains a `custom_nodes/` sibling; that
    # sanity-checks the parents[4] arithmetic against ComfyUI's real layout.
    assert (fallback.parent.parent / "custom_nodes").is_dir(), (
        f"fallback {fallback} does not resolve under ComfyUI root -- "
        f"parents[N] arithmetic drifted (parents[3] regression?)")
    fallback.mkdir(parents=True, exist_ok=True)
    model_path = fallback / "RealESRGAN_x2plus.pth"
    _existed_before = model_path.exists()
    if not _existed_before:
        model_path.write_bytes(b"fake_fallback_content")
    try:
        descriptor = _MockDescriptor(scale=2)
        mock_loader_cls, _mli = _make_mock_loader(descriptor)
        mock_spandrel = mock.MagicMock()
        mock_spandrel.ModelLoader = mock_loader_cls
        # Simulate ImportError for folder_paths.
        with mock.patch.dict("sys.modules", {"spandrel": mock_spandrel}):
            import sys as _sys
            saved_fp = _sys.modules.pop("folder_paths", None)
            try:
                # Insert an ImportError-raising stub? No -- just leave it absent.
                engine = SpandrelEsrgan()
                engine.load(torch.device("cpu"))
                assert engine._descriptor is not None
            finally:
                if saved_fp is not None:
                    _sys.modules["folder_paths"] = saved_fp
    finally:
        # Only delete if this test created the file (don't nuke a real download).
        if not _existed_before:
            try:
                model_path.unlink()
            except OSError:
                pass
