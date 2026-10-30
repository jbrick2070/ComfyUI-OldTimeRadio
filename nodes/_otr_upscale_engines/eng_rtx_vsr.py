"""``rtx_vsr`` upscale engine -- NVIDIA RTX Video Super Resolution (2026-10-14).

The same super-resolution an RTX card applies to browser video, reached through
NVIDIA's ``nvidia-vfx`` Python package (``import nvvfx``), the binding Comfy-Org's
own ComfyUI_Nvidia_VFX_Nodes is built on. An optional second upscaler beside
``spandrel_esrgan``. ``off`` stays the canonical workflow's default (operator,
2026-10-14: "one of 2 upscale options but do not activate by default for
now"); every NVIDIA workflow variant selects it on ``cuda`` (operator,
2026-10-15: "every nvidia lane to 1080p"). Its NVIDIA package comes with the
pack's requirements where NVIDIA builds it (below); wherever it cannot run it
falls back to ``off`` with a warning, so a variant never fails on it.

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

WHAT IT NEEDS, AND HOW IT ARRIVES. An NVIDIA RTX card (Turing or newer;
driver 570.65+ on Windows) and the ``nvidia-vfx`` wheel (about 540 MB; Windows
and Linux x86-64; NVIDIA's proprietary licence). Since 2026-10-30 the wheel is
a DECLARED dependency: requirements.txt and pyproject carry
``nvidia-vfx>=0.2,<0.3`` under a marker for the two platforms NVIDIA publishes
it for, the way Comfy-Org's own ComfyUI_Nvidia_VFX_Nodes declares it, so a
registry or Manager install has it before the first render. AND THIS PACK NEVER
INSTALLS IT AT RUN TIME. From 2.3.29 to 2.3.44 ``load`` ran ``python -m pip
install`` itself the first time an NVIDIA workflow picked the engine. The Comfy
Registry's published standards forbid exactly that ("runtime package
installation through subprocess calls is not permitted"), its reviewer flagged
2.3.34 and 2.3.35 ``policy-v0.7: code-install`` for it, and no version after
2.3.20 reached Active while it shipped (PBUG-20261030-01). ``load`` imports
what is there; a git clone that skipped the requirements is told the one pip
line to run into ComfyUI's own Python (``INSTALL_HINT``, which the
missing-package refusal names). A box without the package refuses like any
other unusable case -- naming the hardware reason first where there is one, so
a CPU, Mac, AMD or GTX box is not sent to install a wheel it cannot use -- and
the composite degrades to ``off``. An AMD or Intel x86-64 box does receive the
wheel through the marker (PEP 508 has no GPU-vendor marker) and refuses here by
hardware; the operator took that cost over a run-time install (2026-10-30).

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

#: The one pip line, bounded to the 0.2 line measured on the 4060 (0.2.0.0); a
#: later minor release is a deliberate bump, not something a box picks up by
#: itself.
REQUIREMENT = "nvidia-vfx>=0.2,<0.3"

#: The missing-package refusal names it (``_refuse``'s ``hint``), and only on a
#: box that could run the SDK. The package is a declared dependency, so a
#: registry or Manager install already has it; the hint is for a git clone
#: that skipped `pip install -r requirements.txt`. The PACK never runs it --
#: the user does, with ComfyUI's own interpreter (a system pip installs where
#: ComfyUI never looks); the module docstring says why.
INSTALL_HINT = ("it comes with the pack's requirements; if it is missing, install "
                "it with ComfyUI's own interpreter, never a system pip: "
                'python -m pip install "%s" (NVIDIA RTX card required)'
                % REQUIREMENT)

#: The SDK's floor: RTX Video Super Resolution runs on Turing (sm_75) or newer.
#: A Pascal/Maxwell GTX card would be told to install a wheel it cannot use.
MIN_COMPUTE_CAPABILITY = (7, 5)


def _hardware_blocker(torch, dev):
    """Why RTX VSR cannot run on this host, or ``""``. Read when the SDK is
    missing, so the refusal names the real reason rather than the absent
    package: the wheel is win_amd64 / manylinux x86-64, built for CPython 3.10
    and 3.11 and abi3 from 3.12 (pypi.nvidia.com, read 2026-10-30), and the
    effect needs sm_75+. (GTX 16xx is Turing without tensor cores and
    passes this check; the load probe catches it.)"""
    import platform
    import sys
    machine = platform.machine().lower()
    if sys.platform not in ("win32", "linux") or machine not in ("amd64", "x86_64"):
        return f"nvidia-vfx ships for Windows/Linux x86-64 only ({sys.platform} {machine})"
    if sys.version_info < (3, 10):
        return "nvidia-vfx needs Python 3.10 or newer (this is %d.%d)" % sys.version_info[:2]
    try:
        cap = tuple(torch.cuda.get_device_capability(dev))
    except Exception as exc:  # noqa: BLE001 -- an unreadable card is still a refusal
        return f"could not read the card's compute capability ({type(exc).__name__}: {exc})"
    if cap < MIN_COMPUTE_CAPABILITY:
        return ("this card is compute capability %d.%d; RTX VSR needs %d.%d "
                "(Turing) or newer" % (cap + MIN_COMPUTE_CAPABILITY))
    # GTX 16xx is Turing (sm_75) without the RTX hardware VSR needs; it passes
    # the capability floor, so the product name is the only cheap tell.
    try:
        name = str(torch.cuda.get_device_name(dev))
    except Exception:  # noqa: BLE001 -- unknown name: let the probe decide
        name = ""
    if "GTX" in name.upper():
        return f"{name} is a GTX card; RTX VSR needs an RTX card"
    return ""


def _import_nvvfx(blocker=""):
    """``import nvvfx``, or a ``ModuleNotFoundError`` that says what is missing.

    THE PACK NEVER INSTALLS IT (module docstring). Only ``ModuleNotFoundError``
    for ``nvvfx`` itself is reworded: a package that is present but fails to
    load (a DLL error, an old driver) raises as itself, since the install hint
    would be wrong for it. ``blocker`` is the hardware reason, when there is
    one, so a box that could never run the SDK is not sent to install it."""
    try:
        import nvvfx
        return nvvfx
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", "nvvfx") not in (None, "nvvfx"):
            raise
        if blocker:
            raise ModuleNotFoundError(
                f"nvvfx is not installed, and it could not run here anyway: "
                f"{blocker}") from exc
        raise ModuleNotFoundError(
            "the nvidia-vfx package is not installed in ComfyUI's Python") from exc

#: ``nvvfx.QualityLevel`` member. ULTRA measured the highest SSIM of the
#: upscaling levels on OTR frames at under 14 ms a 1080p frame on an 8 GB
#: laptop card. ``OTR_RTX_VSR_QUALITY`` names another level.
DEFAULT_QUALITY = "ULTRA"

#: The SDK wants output sides in multiples of 8 (Comfy-Org's node rounds the
#: same way). The engine asks for the rounded-up size and resizes to exact.
_SIDE_MULTIPLE = 8

#: The largest per-side enlargement one VSR pass is used for. Measured: up to
#: 2x one pass wins; past it VSR x2 plus bicubic wins (see upscale_frames_to).
ONE_PASS_MAX_SCALE = 2


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
    """NVIDIA RTX Video Super Resolution: straight to the composite's target
    size in one pass (``upscale_frames_to``), x2 for the registry contract."""

    name = "rtx_vsr"
    roles = ("upscale_stage",)
    default_roles = ()               # never an auto-default; explicit pick only
    # The OUTPUT is the user's; the SDK runtime is NVIDIA-proprietary and is
    # never redistributed by this pack (the user installs it from NVIDIA's own
    # PyPI release; the pack never does -- see load()).
    commercial_clean = False
    requires_flag = None

    device_backends = ("cuda",)
    requires_vendor = "nvidia"
    intrinsic_scale = 2

    #: The composite falls back to ``off`` when ``load`` refuses (see module
    #: docstring). Spandrel does not set this: its refusal stays loud.
    degrades_to_off = True

    #: CLEAN FULL FRAME: an exact x2 (rounded up for the SDK, cropped back to
    #: exact) with no added sharpening filter, crop or pad, then the
    #: composite's exact full-frame resize to the canvas (never fit-and-pad).
    #: VSR does crisp edges itself -- that is its point -- but it adds no
    #: unsharp pass and invents no texture. That honours a row that declared
    #: ``lanczos_clean_full_frame`` (the AnimateDiff/Ghost lanes), so the
    #: composite upscales those rows with this engine (operator, 2026-10-15:
    #: "run some animated diffs and upscale those too"). Real-ESRGAN does not
    #: declare it: its texture invention is what that contract excludes.
    clean_full_frame = True

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
        # "to_canvas": the composite now asks for its target size in one pass
        # (2026-10-18), so composites cached from the x2 era are not reused.
        return (("nvidia_vfx", _package_version()), ("quality", _quality()),
                ("sizing", "to_canvas"))

    # -----------------------------------------------------------------
    def _refuse(self, reason, message, hint=False):
        """The composite's refusal. ``hint`` appends the pip line, and only the
        missing-package refusal on a box that could run the SDK sets it: a CPU,
        AMD, Mac or GTX box, a bad quality level, a failed probe, or a package
        that is present but will not load each get their own reason and
        nothing else, so nobody is sent to install a wheel that would not
        help (Sonnet and Codex reviews of 68d0a1417)."""
        text = f"{message}; {INSTALL_HINT}" if hint else message
        return EngineUnusable(self.name, "upscale_stage", reason, text, kind="upscale")

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
        # THE SDK, AFTER THE DEVICE HAS PROVED NVIDIA, so a CPU, Mac or AMD box
        # is refused for its hardware, never for a package it has no use for.
        blocker = _hardware_blocker(torch, dev)
        try:
            nvvfx = _import_nvvfx(blocker=blocker)
        except ModuleNotFoundError as exc:
            # The pip line helps exactly here: the package is absent and this
            # box could use it. A hardware blocker means it would not help; a
            # module OTHER than nvvfx missing is the SDK's own breakage.
            absent = getattr(exc, "name", None) in (None, "nvvfx")
            raise self._refuse(EngineUsabilityReason.MISSING_MODEL,
                               f"the nvidia-vfx package is not usable here "
                               f"({type(exc).__name__}: {exc})",
                               hint=absent and not blocker)
        except Exception as exc:  # noqa: BLE001 -- an ImportError, a DLL load error
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

        The registry's fixed-scale contract (``intrinsic_scale``); the load
        probe uses it. The composite calls :meth:`upscale_frames_to`."""
        assert self._nvvfx is not None, "engine.load(device) must run first"
        _, h, w, _ = frames.shape
        return self._run(frames, int(w) * 2, int(h) * 2)

    def upscale_frames_to(self, frames, out_w: int, out_h: int):
        """BHWC float32 [0, 1] -> BHWC at EXACTLY ``(out_h, out_w)``, in one
        VSR pass (2026-10-18), when one pass is the better picture.

        The fixed x2 doubled a 1472x832 LTX frame to 2944x1664 only for the
        composite to shrink it back to 1920x1080. Asking VSR for 1080p
        directly is better on every frame measured. Comfy-Org's
        Nvidia_RTX_Nodes_ComfyUI exposes the same target-size mode.

        MEASURED ON THE 4060, 16 native Klein stills (no upscaler had touched
        them), area-downscaled and brought back, luma SSIM:
          * 1.3x  -- one pass 0.869, x2 then shrink 0.863 (16/16 stills)
          * 2x    -- identical (the same pass)
          * 4x    -- one pass 0.514, x2 then bicubic 0.526 (0/16 for one pass)
        VSR is tuned for about 2x; past that, its own x2 plus a bicubic
        remainder is the better picture. So a target more than
        :data:`ONE_PASS_MAX_SCALE` on either side -- the AnimateDiff 512x288
        to 1080p case -- or smaller than the frame returns ``None``, and the
        caller takes the x2 path, as before."""
        assert self._nvvfx is not None, "engine.load(device) must run first"
        _, h, w, _ = frames.shape
        out_w, out_h = int(out_w), int(out_h)
        if (out_w < int(w) or out_h < int(h)
                or out_w > int(w) * ONE_PASS_MAX_SCALE
                or out_h > int(h) * ONE_PASS_MAX_SCALE):
            return None
        return self._run(frames, out_w, out_h)

    def _run(self, frames, want_w: int, want_h: int):
        """Deterministic and stateless per frame (no temporal history is fed),
        so the composite's held-frame reuse stays exact.

        The SDK wants sides in multiples of 8. Off that grid it is asked for
        the rounded-up size and the result is RESIZED back to exact, never
        cropped: the SDK scales the whole picture to the size it was given,
        so a crop cut the right and bottom edges off a picture that was also
        stretched by those pixels (measured 2026-10-18 on native Klein stills:
        a 393x222 frame doubled that way scored 18.3 dB, plain bicubic 23.4).
        The shipped sizes (1024x576, 2944x1664, 1920x1080) are on the grid."""
        import torch
        import torch.nn.functional as F

        b = frames.shape[0]
        out_w, out_h = _round_up(want_w), _round_up(want_h)
        effect = self._effect_for(out_w, out_h)
        outs = []
        with torch.inference_mode(), torch.cuda.device(self.device):
            for i in range(int(b)):
                chw = frames[i, :, :, :3].to(self.device, torch.float32) \
                    .movedim(-1, 0).contiguous()
                img = torch.from_dlpack(effect.run(chw).image).clone()
                if (out_w, out_h) != (want_w, want_h):
                    img = F.interpolate(img[None].float(), size=(want_h, want_w),
                                        mode="bicubic", align_corners=False,
                                        antialias=True)[0]
                outs.append(img.movedim(0, -1))
            out = torch.stack(outs).clamp_(0.0, 1.0).contiguous()
        return out
