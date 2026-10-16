"""``rtx_vsr`` upscale engine -- NVIDIA RTX Video Super Resolution (2026-10-14).

The same super-resolution an RTX card applies to browser video, reached through
NVIDIA's ``nvidia-vfx`` Python package (``import nvvfx``), the binding Comfy-Org's
own ComfyUI_Nvidia_VFX_Nodes is built on. An optional second upscaler beside
``spandrel_esrgan``. ``off`` stays the canonical workflow's default (operator,
2026-10-14: "one of 2 upscale options but do not activate by default for
now"); every NVIDIA workflow variant selects it on ``cuda`` (operator,
2026-10-15: "every nvidia lane to 1080p"). Where ``nvidia-vfx`` is not
installed it falls back to ``off`` with a warning, so a variant never fails on
it.

WHY IT EARNS A DROPDOWN ROW -- measured on the RTX 4060 Laptop (8 GB), 48 real
frames of a 1472x832 OTR viz clip, 2026-10-14:

* Full reference (downscale by 2, upscale by 2, compare with the original):
  VSR ULTRA 3.9 ms a frame, PSNR 35.82 / SSIM 0.936; Real-ESRGAN x2 fp32
  629 ms a frame, PSNR 34.47 / SSIM 0.899. About 160x faster, and closer to
  the original. The composite's own lanczos + unsharp (``off``) scored
  PSNR 36.04 / SSIM 0.944 at 15.8 ms: VSR's gain is crisper edges, not a
  higher fidelity score.
* Production shape (1472x832 straight to the 1920x1080 canvas): VSR 11-14
  ms a frame; Real-ESRGAN x2 then fit, 1,846 ms a frame on fp16 -- and at
  fp32 it spilled the 8 GB card into shared memory and stalled.
* Through THIS class, live (2026-10-14, same card): load() from the widget's
  shipped ``cpu`` landed on cuda:0 in 1.8 s; 48 frames x2 (1472x832 ->
  2944x1664) at 31 ms each including the copy back; torch peak 0.97 GB, the
  card at 2.5 GB with its desktop baseline; the same frame gave identical
  bytes on repeat and after other frames (held-frame reuse stays exact); a
  1001x501 input came back exactly 2002x1002.

WHAT IT NEEDS, AND HOW IT GETS THERE. An NVIDIA RTX card (Turing or newer;
driver 570.65+ on Windows) and the ``nvidia-vfx`` wheel (about 540 MB; Windows
and Linux x86-64; NVIDIA's proprietary licence). It is NOT in requirements.txt,
because that would put a proprietary 540 MB wheel on every AMD, Intel and Mac
install for nothing. Instead ``load`` INSTALLS IT ITSELF (operator,
2026-10-15): only when a workflow has picked this engine -- which only the
NVIDIA workflow variants do -- and only once the device has proved to be an
NVIDIA CUDA card, it runs ``python -m pip install nvidia-vfx`` into ComfyUI's
own interpreter, once per server process, then carries on. Choosing an NVIDIA
workflow on an NVIDIA machine is taken as accepting NVIDIA's licence for its
own runtime (operator: the user already accepted NVIDIA's terms with the
driver). An offline box, a pip failure or a wheel that will not load refuses
like any other unusable case, so the composite degrades to ``off``.

IT NEVER FAILS A RENDER FOR LACK OF HARDWARE. ``degrades_to_off = True``:
when this engine cannot run here -- no ``nvvfx``, no CUDA, an AMD card, a GTX
card the SDK refuses -- ``load`` raises :class:`EngineUnusable` with the reason
and the composite falls back to ``off`` (its lanczos path) with a warning. The
composite runs at the very end of an episode; losing a finished episode to an
optional sharpener is the wrong trade (operator rule: probes never fail a
workflow; an optional accelerator is used when it works).

THE DEVICE BOX. VSR runs only on an NVIDIA GPU, so a ``cpu`` in
``upscale_device`` -- the widget's shipped default -- means "you did not
choose", and the engine takes ``cuda:0`` and says so. ``mps`` cannot be this
engine's device and degrades as above.

Cold-import clean: ``nvvfx`` and ``torch`` import lazily inside ``load`` /
``upscale_frames``; the fingerprint reads package metadata only.
"""
from __future__ import annotations

import logging

from .._otr_shared import env as otr_env
from .._otr_shared.engine_registry_base import (
    EngineUnusable,
    EngineUsabilityReason,
)
from .registry import register

_LOG = logging.getLogger("OTR.upscale.rtx_vsr")

#: The pip line every refusal names.
INSTALL_HINT = "pip install nvidia-vfx (NVIDIA RTX card required)"

#: What ``load`` installs on an NVIDIA CUDA box that lacks the SDK. Bounded to
#: the 0.2 line, the one measured on the 4060 (0.2.0.0); a later minor release
#: is a deliberate bump, not something a first run picks up by itself.
AUTO_INSTALL_REQUIREMENT = "nvidia-vfx>=0.2,<0.3"

#: Ceiling on the one pip run. The wheel is ~540 MB; a slow line still lands
#: well inside this, and a hung pip must not hold the episode's last stage.
AUTO_INSTALL_TIMEOUT_S = 1200

#: ONCE PER PROCESS. ``None`` = not tried; otherwise the outcome text. A failed
#: install (offline box, pip missing) is not retried on every episode -- each
#: retry would cost the same wait for the same answer; a server restart tries
#: again.
_AUTO_INSTALL_OUTCOME = None


def _pip_install_nvidia_vfx():
    """Install the SDK into THIS interpreter. Returns ``(ok, detail)``; never
    raises. Spawned through the pack's one subprocess boundary
    (``otr_proc.run``, whose allowlist admits the ``python`` interpreter)."""
    import sys
    from .._otr_shared import proc as otr_proc
    argv = [sys.executable, "-m", "pip", "install", "--disable-pip-version-check",
            AUTO_INSTALL_REQUIREMENT]
    try:
        r = otr_proc.run(argv, capture_output=True, text=True,
                         timeout=AUTO_INSTALL_TIMEOUT_S)
    except Exception as exc:  # noqa: BLE001 -- timeout, spawn failure
        return False, f"{type(exc).__name__}: {exc}"
    tail = " ".join(((r.stdout or "") + " " + (r.stderr or "")).split())[-300:]
    return r.returncode == 0, f"pip exit {r.returncode}: {tail}"


def _import_nvvfx_installing_if_missing():
    """``import nvvfx``; on an absent package, install it once and retry.

    Only ``ModuleNotFoundError`` for ``nvvfx`` itself triggers the install: a
    package that is present but fails to load (a DLL error, an old driver) is
    not fixed by installing it again, so that raises as-is."""
    global _AUTO_INSTALL_OUTCOME
    try:
        import nvvfx
        return nvvfx
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "nvvfx") not in (None, "nvvfx"):
            raise
        if _AUTO_INSTALL_OUTCOME is not None:
            raise ModuleNotFoundError(
                f"nvvfx is not installed and this server already tried to "
                f"install it ({_AUTO_INSTALL_OUTCOME})") from exc
    _LOG.warning("[OTR.upscale] rtx_vsr: nvidia-vfx is not installed; installing "
                 "%s into %s (about 540 MB, once)", AUTO_INSTALL_REQUIREMENT,
                 __import__("sys").executable)
    ok, detail = _pip_install_nvidia_vfx()
    _AUTO_INSTALL_OUTCOME = ("installed" if ok else "failed") + f" -- {detail}"
    if not ok:
        raise ModuleNotFoundError(f"installing {AUTO_INSTALL_REQUIREMENT} failed "
                                  f"({detail})")
    import importlib
    importlib.invalidate_caches()
    import nvvfx
    _LOG.info("[OTR.upscale] rtx_vsr: nvidia-vfx %s installed and imported",
              _package_version())
    return nvvfx

#: ``nvvfx.QualityLevel`` member. ULTRA measured the highest SSIM of the
#: upscaling levels on OTR frames at under 14 ms a 1080p frame on an 8 GB
#: laptop card. ``OTR_RTX_VSR_QUALITY`` names another level.
DEFAULT_QUALITY = "ULTRA"

#: The SDK wants output sides in multiples of 8 (Comfy-Org's node rounds the
#: same way). The engine asks for the rounded-up size and crops to exact.
_SIDE_MULTIPLE = 8


def _round_up(value: int, multiple: int = _SIDE_MULTIPLE) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _package_version() -> str:
    """``nvidia-vfx``'s installed version, or ``"absent"``. Metadata only:
    never imports the package (this runs on every prompt evaluation)."""
    try:
        from importlib import metadata
        return str(metadata.version("nvidia-vfx"))
    except Exception:  # noqa: BLE001 -- PackageNotFoundError and friends
        return "absent"


def _quality() -> str:
    raw = str(otr_env.get("OTR_RTX_VSR_QUALITY", "") or "").strip().upper()
    return raw or DEFAULT_QUALITY


@register
class RtxVsr:
    """NVIDIA RTX Video Super Resolution, x2."""

    name = "rtx_vsr"
    roles = ("upscale_stage",)
    default_roles = ()               # never an auto-default; explicit pick only
    # The OUTPUT is the user's; the SDK runtime is NVIDIA-proprietary and is
    # never redistributed by this pack (the user installs it).
    commercial_clean = False
    requires_flag = None

    device_backends = ("cuda",)
    requires_vendor = "nvidia"
    intrinsic_scale = 2

    #: The composite falls back to ``off`` when ``load`` refuses (see module
    #: docstring). Spandrel does not set this: its refusal stays loud.
    degrades_to_off = True

    def __init__(self) -> None:
        self.device = None
        self._nvvfx = None
        self._quality = None
        self._effect = None          # the open VideoSuperRes for _effect_size
        self._effect_size = None     # (out_w, out_h) it was loaded for

    # -----------------------------------------------------------------
    def model_fingerprint_parts(self) -> tuple:
        """The SDK's models ship INSIDE the ``nvidia-vfx`` wheel, so its version
        is the checkpoint identity; the quality level picks the model. Pure:
        package metadata and the environment, never load state."""
        return (("nvidia_vfx", _package_version()), ("quality", _quality()))

    # -----------------------------------------------------------------
    def _refuse(self, reason, message):
        return EngineUnusable(self.name, "upscale_stage", reason,
                              f"{message}; {INSTALL_HINT}", kind="upscale")

    def load(self, device) -> None:
        """Bind to an NVIDIA GPU and prove the SDK runs there. Raises
        :class:`EngineUnusable` (which the composite turns into ``off``)."""
        try:
            import torch
        except ImportError as exc:
            raise self._refuse(EngineUsabilityReason.MALFORMED_CONFIG,
                               f"torch is not importable ({exc})")
        dev = torch.device(device) if not isinstance(device, torch.device) else device
        if dev.type == "cpu" and torch.cuda.is_available():
            _LOG.info("[OTR.upscale] rtx_vsr runs only on an NVIDIA GPU; "
                      "upscale_device is 'cpu', so it uses cuda:0")
            dev = torch.device("cuda:0")
        if dev.type != "cuda" or not torch.cuda.is_available():
            raise self._refuse(EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                               f"RTX VSR needs an NVIDIA CUDA device, got {dev}")
        if getattr(torch.version, "hip", None):
            raise self._refuse(EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                               "this is a ROCm (AMD) build of torch; RTX VSR is "
                               "NVIDIA-only")
        # THE SDK, AFTER THE DEVICE HAS PROVED NVIDIA: the auto-install below
        # must never put a 540 MB proprietary wheel on a CPU, Mac or AMD box,
        # so every hardware refusal above runs first.
        try:
            nvvfx = _import_nvvfx_installing_if_missing()
        except Exception as exc:  # noqa: BLE001 -- ImportError, a DLL load error
            raise self._refuse(EngineUsabilityReason.MISSING_MODEL,
                               f"the nvidia-vfx package is not usable here "
                               f"({type(exc).__name__}: {exc})")
        quality = _quality()
        if quality not in getattr(nvvfx.QualityLevel, "__members__", {}):
            raise self._refuse(
                EngineUsabilityReason.MALFORMED_CONFIG,
                f"OTR_RTX_VSR_QUALITY={quality!r} is not one of "
                f"{sorted(getattr(nvvfx.QualityLevel, '__members__', {}))}")
        if quality.startswith(("DENOISE", "DEBLUR")):
            raise self._refuse(
                EngineUsabilityReason.MALFORMED_CONFIG,
                f"OTR_RTX_VSR_QUALITY={quality!r} keeps the input size; the "
                f"upscale stage needs an upscaling level")
        self._nvvfx, self._quality, self.device = nvvfx, quality, dev
        # PROVE IT RUNS before the composite commits to it: a GTX card, an old
        # driver or a missing runtime DLL fails HERE, as a refusal, rather than
        # on the first real frame mid-segment.
        try:
            probe = torch.zeros((1, 64, 64, 3), dtype=torch.float32, device=dev)
            out = self.upscale_frames(probe)
            if tuple(out.shape) != (1, 128, 128, 3):
                raise RuntimeError("probe returned shape %s" % (tuple(out.shape),))
        except Exception as exc:  # noqa: BLE001 -- refused, the composite degrades
            self.unload()
            raise self._refuse(EngineUsabilityReason.INCOMPATIBLE_PROFILE,
                               f"the RTX VSR effect did not run on {dev} "
                               f"({type(exc).__name__}: {exc})")

    def unload(self) -> None:
        self._close_effect()
        self._nvvfx = None
        if self.device is not None:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001 -- teardown must never raise
                pass
        self.device = None

    # -----------------------------------------------------------------
    def _close_effect(self) -> None:
        effect, self._effect, self._effect_size = self._effect, None, None
        if effect is not None:
            try:
                effect.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 -- teardown must never raise
                _LOG.warning("[OTR.upscale] rtx_vsr: closing the effect raised %s", exc)

    def _effect_for(self, out_w: int, out_h: int):
        """One open effect per output size; a clip of another size reopens it."""
        if self._effect is not None and self._effect_size == (out_w, out_h):
            return self._effect
        self._close_effect()
        import torch
        nvvfx = self._nvvfx
        with torch.cuda.device(self.device):
            effect = nvvfx.VideoSuperRes(
                quality=nvvfx.QualityLevel[self._quality], strength=1.0,
                device=self.device.index or 0,
                image_encoding=nvvfx.ImageEncoding.RGB8)
            effect.__enter__()
            try:
                effect.output_width, effect.output_height = out_w, out_h
                effect.load()
            except BaseException:
                effect.__exit__(None, None, None)
                raise
        self._effect, self._effect_size = effect, (out_w, out_h)
        return effect

    def upscale_frames(self, frames):
        """BHWC float32 [0, 1] on ``self.device`` -> BHWC upscaled x2.

        Deterministic and stateless per frame (no temporal history is fed), so
        the composite's held-frame reuse stays exact."""
        assert self._nvvfx is not None, "engine.load(device) must run first"
        import torch

        b, h, w, _ = frames.shape
        want_w, want_h = int(w) * 2, int(h) * 2
        out_w, out_h = _round_up(want_w), _round_up(want_h)
        effect = self._effect_for(out_w, out_h)
        outs = []
        with torch.inference_mode(), torch.cuda.device(self.device):
            for i in range(int(b)):
                chw = frames[i, :, :, :3].to(self.device, torch.float32) \
                    .movedim(-1, 0).contiguous()
                img = torch.from_dlpack(effect.run(chw).image).clone()
                outs.append(img.movedim(0, -1)[:want_h, :want_w, :])
            out = torch.stack(outs).clamp_(0.0, 1.0).contiguous()
        return out
