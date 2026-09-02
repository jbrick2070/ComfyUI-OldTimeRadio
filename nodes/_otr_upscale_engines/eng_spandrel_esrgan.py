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

#: Exception types already reported by _resolve_model's folder_paths guard.
#: That warning used to sit only on load() -- an occasional, explicit call.
#: model_fingerprint_parts() now calls the same resolver from IS_CHANGED, which
#: ComfyUI runs on EVERY prompt evaluation, so a persistently broken
#: folder_paths would emit one warning per evaluation forever. Keyed on the
#: exception class name: tiny, stable, and bounded (Bug Bible 06.04).
_RESOLVE_WARNED: set = set()
_RESOLVE_WARNED_MAX = 32


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
    # Model resolution -- ONE source of truth for two callers
    # -----------------------------------------------------------------
    def _resolve_model(self):
        """Locate the checkpoint. Returns ``(candidates, path_or_None)``.

        SINGLE-SOURCED DELIBERATELY. ``load()`` and
        ``model_fingerprint_parts()`` both need the answer, and before this
        existed they asked DIFFERENT questions: the loader walked this
        candidate list while the composite's cache key called
        ``folder_paths.get_full_path`` alone. On a box where the checkpoint is
        reachable only through the repo-relative fallback, the loader found it
        and the cache key did not -- so swapping those weights never
        invalidated the composite.

        Returns a plain tuple rather than a dataclass/NamedTuple on purpose:
        this module imports only stdlib + the registry base, and staying
        cold-import clean is the whole design claim of the namespace.

        Candidate paths are absolutised and normalised, then deduplicated on
        ``normcase`` with FIRST occurrence winning. The dedupe is not cosmetic:
        ``folder_paths`` normally returns the very directory the repo-relative
        fallback appends, and the absence marker EMBEDS this list in the cache
        key, so a duplicate would be baked into it.

        Never raises for ordinary absence -- that is ``None``, and the caller
        decides what it means. An UNREADABLE candidate is a different matter
        and does raise: ``Path.is_file()`` filters through pathlib's
        ``_ignore_error()``, which returns False only for the not-found family
        and re-raises everything else, so a ``PermissionError`` escapes here
        rather than masquerading as absence. That is the behaviour we want and
        it costs no code -- see ``model_fingerprint_parts`` for why an explicit
        classification pass was written and then deleted as unreachable.
        """
        raw: list = []
        try:
            import folder_paths  # noqa: WPS433 - deliberate lazy import
            fp_candidates = folder_paths.get_folder_paths("upscale_models")
            if fp_candidates:
                raw.extend(fp_candidates)
        except (ImportError, ModuleNotFoundError):
            pass
        except Exception as e:  # noqa: BLE001
            key = type(e).__name__
            if key not in _RESOLVE_WARNED:
                if len(_RESOLVE_WARNED) >= _RESOLVE_WARNED_MAX:
                    _RESOLVE_WARNED.clear()
                _RESOLVE_WARNED.add(key)
                _LOG.warning(
                    "[OTR.upscale] folder_paths.get_folder_paths raised (%s: "
                    "%r); falling back to repo-relative models dir. Logged "
                    "once per error type -- this path runs on every prompt "
                    "evaluation.", key, str(e))
        # Repo-relative fallback: the ComfyUI install has
        # models/upscale_models/ at its root; our path is
        # <comfy_root>/custom_nodes/ComfyUI-OldTimeRadio/nodes/_otr_upscale_engines/eng_spandrel_esrgan.py
        # so parents[0]=_otr_upscale_engines, parents[1]=nodes,
        # parents[2]=ComfyUI-OldTimeRadio, parents[3]=custom_nodes,
        # parents[4]=<comfy_root>. Sonnet 5 QA MF-1: r4 wrote parents[3],
        # which landed under custom_nodes/ (nothing there) and silently
        # missed the shipped models dir on a bare-venv path resolution.
        # The length guard covers a checkout shallower than five levels
        # (pip-install / symlink / tmp_path test tree), where indexing
        # parents[4] would raise IndexError instead of resolving.
        here = Path(__file__).resolve()
        if len(here.parents) > 4:
            raw.append(here.parents[4] / "models" / "upscale_models")

        candidates: list = []
        seen: set = set()
        for entry in raw:
            norm = os.path.abspath(os.path.normpath(os.fspath(entry)))
            key = os.path.normcase(norm)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(norm)

        model_path = None
        for base in candidates:
            path = Path(base) / self._model_filename
            # NOT wrapped in try/except OSError on purpose: is_file() already
            # returns False for the not-found family and re-raises anything
            # else, which is exactly the absence-vs-fault line we want. A
            # broad catch here would launder an unreadable checkpoint into
            # "absent" and cache a stale composite against it.
            if path.is_file():
                model_path = str(path)
                break
        return tuple(candidates), model_path

    # -----------------------------------------------------------------
    # Cache fingerprint (see UpscaleEngine.model_fingerprint_parts)
    # -----------------------------------------------------------------
    def model_fingerprint_parts(self) -> tuple:
        """Filesystem metadata for the pinned checkpoint + the declared digest.

        Pure with respect to load state: nothing here reads ``self.device`` or
        ``self._descriptor``, so the value is identical before ``load()``,
        after ``load()`` and after ``unload()``.

        THE ABSENCE / FAILURE SPLIT, which is where the care goes. A missing
        checkpoint is a NORMAL, stable state and must keep producing an EQUAL
        key (Bug Bible 06.07 -- an unstable key defeats the cache forever),
        whereas an UNREADABLE one is a real fault the caller should turn into a
        bare NaN. Nothing here has to draw that line by hand, because
        ``pathlib`` already draws it: ``Path.is_file()`` filters through
        ``_ignore_error()``, which whitelists only the not-found family
        (ENOENT / ENOTDIR / EBADF / ELOOP) and RE-RAISES everything else.
        Verified on this platform, Python 3.12.11: ``PermissionError`` and
        ``OSError(EIO)`` propagate out of ``is_file()``; ``FileNotFoundError``
        and ``NotADirectoryError`` return False. So ``_resolve_model`` returns
        ``None`` for genuine absence ONLY, and a permission fault has already
        escaped before we get here. An earlier draft of this method re-stat'ed
        every candidate to classify the failure itself; that pass was
        unreachable and was removed rather than kept as reassuring dead code.

        No live hashing -- ``_model_sha256`` is the DECLARED constant, folded
        in so that re-pinning it changes the key. What this key can prove is
        bounded and worth stating plainly: it is filesystem metadata plus a
        version salt, and it does not guarantee detection when the resulting
        path, size and mtime are all unchanged.
        """
        candidates, model_path = self._resolve_model()
        if model_path is not None:
            st = os.stat(model_path)          # unexpected OSError propagates
            return (
                ("model", self._model_filename, model_path,
                 st.st_mtime_ns, st.st_size),
                ("declared_sha256", self._model_sha256),
            )
        return (
            ("model_absent", self._model_filename, candidates),
            ("declared_sha256", self._model_sha256),
        )

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

        candidates, model_path = self._resolve_model()
        if model_path is None:
            raise EngineUnusable(
                self.name, "upscale_stage",
                EngineUsabilityReason.MISSING_MODEL,
                f"{self._model_filename} not found under any of "
                f"{list(candidates)}. Download it from "
                f"https://github.com/xinntao/Real-ESRGAN/releases/download/"
                f"v0.2.1/RealESRGAN_x2plus.pth into models/upscale_models "
                f"(sha256 {self._model_sha256}), or run the script at "
                f"https://github.com/jbrick2070/ComfyUI-OldTimeRadio/blob/"
                f"v2.0-alpha/scripts/ensure_upscale_models.py",
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
