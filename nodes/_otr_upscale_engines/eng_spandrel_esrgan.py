"""``spandrel_esrgan`` upscale engine -- pinned Real-ESRGAN x2plus (BSD-3-Clause)
loaded via spandrel's ModelLoader with explicit device pinning.

Real-ESRGAN x2plus was chosen for the first ship because it is:

* commercial-clean (BSD-3-Clause weights);
* device-agnostic through the spandrel abstraction (CUDA + CPU today; ROCm
  presents as CUDA to torch and works with no code change; MPS deferred);
* an established, widely-used super-resolution model with strong texture
  behaviour on the 832x480 -> 1080p canvas the OTR composite hands it;
* small enough (~64 MB) that ``ensure_upscale_models.py`` can fetch it once
  and pin its SHA-256 for reproducible provisioning.

The adapter enforces spandrel's ``(1, C, H, W)`` per-descriptor-call contract by
looping over the batch dim inside ``upscale_frames`` -- callers see a batch-N
interface but the model runs frame-by-frame internally.

Cold-import clean: torch + spandrel + folder_paths import LAZILY inside
``load`` / ``upscale_frames`` so this module can be imported (and its
CAPABILITIES row consulted) in a bare venv without ComfyUI or spandrel present.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

from .._otr_shared.engine_registry_base import (
    EngineUnusable,
    EngineUsabilityReason,
)
from .registry import register

_LOG = logging.getLogger("OTR.upscale.spandrel_esrgan")


@register
class SpandrelEsrgan:
    """Real-ESRGAN x2plus super-resolution engine, via spandrel."""

    name = "spandrel_esrgan"
    roles = ("upscale_stage",)
    default_roles = ()               # never an auto-default; explicit pick only
    commercial_clean = True          # Real-ESRGAN x2plus is BSD-3-Clause
    requires_flag = None

    # Upscale-namespace identity
    device_backends = ("cuda", "cpu")   # MPS deferred until Mac receipt
    requires_vendor = None
    intrinsic_scale = 2

    # PINNED CHECKPOINT. Filename lives under models/upscale_models/.
    # `scripts/ensure_upscale_models.py` handles URL + SHA-256 provisioning.
    _model_filename = "RealESRGAN_x2plus.pth"

    # SHA-256 PINNED 2026-08-08. An empty string SKIPS verification entirely,
    # so a truncated or substituted checkpoint would have loaded silently.
    #
    # The digest was not taken on trust: before pinning, the file was loaded
    # through spandrel on CPU and confirmed to be a genuine ESRGAN at
    # scale=2, 3->3 channels, tags ['64nf', '23nb', 'unshuffle'] -- the
    # RealESRGAN_x2plus signature -- at 67,061,725 bytes, matching the
    # upstream v0.2.1 release. Pinning the hash of a corrupt file would have
    # cemented the corruption, which is worse than no pin at all.
    #
    # Must stay in lockstep with ASSETS[...]["sha256"] in
    # scripts/ensure_upscale_models.py; a test pins that they agree.
    # Tests monkeypatch this to force verify-fail scenarios.
    _model_sha256 = "49fafd45f8fd7aa8d31ab2a22d14d91b536c34494a5cfe31eb5d89c2fa266abb"

    def __init__(self) -> None:
        self._descriptor = None
        self.device = None

    # -----------------------------------------------------------------
    # Lifecycle: load + unload
    # -----------------------------------------------------------------
    def load(self, device) -> None:
        """Load the pinned Real-ESRGAN x2plus weights onto ``device``.

        Lazy-imports spandrel, torch, and folder_paths. On any missing dep or
        missing model file, raises :class:`EngineUnusable` with a classified
        reason and an actionable message.
        """
        try:
            import spandrel  # noqa: WPS433 - deliberate lazy import
            import torch  # noqa: F401,WPS433 - deliberate lazy import
        except ImportError as e:
            raise EngineUnusable(
                self.name, "upscale_stage",
                EngineUsabilityReason.MALFORMED_CONFIG,
                f"spandrel/torch not installed: pip install 'spandrel~=0.4.1' "
                f"({e})",
                kind="upscale")

        # Model path resolution. Prefer folder_paths (ComfyUI-aware); fall back
        # to a repo-relative path when folder_paths is not importable (bare
        # venv / unit tests -- Antigravity r4 MF-4 / Sonnet 5 MF-5 guard).
        candidates: list[str] = []
        try:
            import folder_paths  # noqa: WPS433 - deliberate lazy import
            fp_candidates = folder_paths.get_folder_paths("upscale_models")
            if fp_candidates:
                candidates.extend(str(p) for p in fp_candidates)
        except (ImportError, ModuleNotFoundError):
            pass
        except Exception as e:  # noqa: BLE001
            _LOG.warning(
                "[OTR.upscale] folder_paths.get_folder_paths raised (%s); "
                "falling back to repo-relative models dir", e)
        # Repo-relative fallback: the ComfyUI install has
        # models/upscale_models/ at its root; our path is
        # <comfy_root>/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_upscale_engines/eng_spandrel_esrgan.py
        # so parents[0]=_otr_upscale_engines, parents[1]=nodes,
        # parents[2]=ComfyUI-OldTimeRadio, parents[3]=custom_nodes,
        # parents[4]=<comfy_root>. Sonnet 5 QA MF-1: r4 wrote parents[3],
        # which landed under custom_nodes/ (nothing there) and silently
        # missed the shipped models dir on a bare-venv path resolution.
        fallback = Path(__file__).resolve().parents[4] / "models" / "upscale_models"
        candidates.append(str(fallback))

        model_path = None
        for base in candidates:
            path = Path(base) / self._model_filename
            if path.is_file():
                model_path = str(path)
                break
        if model_path is None:
            raise EngineUnusable(
                self.name, "upscale_stage",
                EngineUsabilityReason.MISSING_MODEL,
                f"{self._model_filename} not found under any of {candidates}. "
                f"Run `python scripts/ensure_upscale_models.py` to download it.",
                kind="upscale")

        # SHA-256 verify if pinned. Empty string = skip (dev mode / first run).
        if self._model_sha256:
            try:
                actual = hashlib.sha256(Path(model_path).read_bytes()).hexdigest()
            except OSError as e:
                raise EngineUnusable(
                    self.name, "upscale_stage",
                    EngineUsabilityReason.MISSING_MODEL,
                    f"failed to read {model_path!r} for SHA verification: {e}",
                    kind="upscale")
            if actual != self._model_sha256:
                raise EngineUnusable(
                    self.name, "upscale_stage",
                    EngineUsabilityReason.MISSING_MODEL,
                    f"{self._model_filename} sha256 {actual!r} != pinned "
                    f"{self._model_sha256!r} -- corrupt or wrong checkpoint",
                    kind="upscale")

        loader = spandrel.ModelLoader(device=str(device))
        try:
            self._descriptor = loader.load_from_file(model_path)
        except Exception as e:  # noqa: BLE001
            raise EngineUnusable(
                self.name, "upscale_stage",
                EngineUsabilityReason.MISSING_MODEL,
                f"spandrel failed to load {model_path!r}: {type(e).__name__}: {e}",
                kind="upscale")
        # spandrel's ModelLoader.load_from_state_dict already .to(device)'s the
        # descriptor before returning (verified live in
        # spandrel/__helpers/loader.py). The extra .to(device) below is
        # defensive: a future spandrel version may change the internal call
        # order, and a no-op .to() is cheap.
        try:
            self._descriptor.to(device)
        except Exception as e:  # noqa: BLE001
            _LOG.warning(
                "[OTR.upscale] descriptor.to(%r) raised (%s); the model may "
                "already be resident on the target device", device, e)
        # Enforce our pinned intrinsic scale (fail loud on a mislabeled ckpt).
        descriptor_scale = int(getattr(self._descriptor, "scale", 0))
        if descriptor_scale != self.intrinsic_scale:
            # Fable final-gate SF-1: raise AFTER clearing the descriptor so
            # the registry singleton does not hold GPU weights when composite()
            # never calls unload() (composite's engine.load() sits outside its
            # try/finally by design -- a load-time failure has to shed the
            # descriptor itself). Same pattern for every post-descriptor
            # raise in this load() body.
            self._descriptor = None
            raise EngineUnusable(
                self.name, "upscale_stage",
                EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                f"{self._model_filename} reports scale={descriptor_scale}, "
                f"expected {self.intrinsic_scale}",
                kind="upscale")
        # Inference mode for the descriptor's underlying module. spandrel's
        # __call__ is already @torch.inference_mode()-decorated, so this is
        # belt+suspenders: it guarantees no autograd graph accumulates even
        # if a future spandrel version changes the decorator.
        try:
            self._descriptor.eval()
        except Exception:  # noqa: BLE001
            pass
        self.device = device

    def unload(self) -> None:
        """Release the descriptor and clear cached CUDA memory."""
        import gc

        if self._descriptor is not None:
            try:
                self._descriptor.to("cpu")
            except Exception as e:  # noqa: BLE001
                _LOG.warning(
                    "[OTR.upscale] spandrel_esrgan.unload: "
                    "descriptor.to('cpu') failed: %s", e)
            self._descriptor = None
        gc.collect()
        try:
            import torch

            if self.device is not None and str(self.device).startswith("cuda"):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:  # noqa: BLE001
            pass
        self.device = None

    # -----------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------
    def upscale_frames(self, frames):
        """Upscale ``frames`` (BHWC float32 [0,1]) by ``intrinsic_scale``.

        spandrel's ``ImageModelDescriptor.__call__`` enforces batch=1 at the
        descriptor layer; we loop over the batch dim internally so callers see
        a batch-N interface.
        """
        assert self._descriptor is not None, "engine.load(device) must run first"
        import torch

        with torch.inference_mode():
            # BHWC -> BCHW for spandrel's (1,C,H,W) contract
            bchw = frames.permute(0, 3, 1, 2).contiguous()
            outs: list = []
            for i in range(bchw.shape[0]):
                one = bchw[i:i + 1]  # (1, C, H, W)
                out_one = self._descriptor(one)  # spandrel enforces this shape
                outs.append(out_one)
            out_bchw = torch.cat(outs, dim=0)
            # BCHW -> BHWC. .contiguous() so downstream .numpy() doesn't crash
            # on stride mismatch (Antigravity r4 MF-2).
            out_bhwc = out_bchw.permute(0, 2, 3, 1).contiguous()
        return out_bhwc
