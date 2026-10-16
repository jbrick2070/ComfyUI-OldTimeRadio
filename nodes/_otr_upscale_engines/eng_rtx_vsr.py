"""``rtx_vsr`` upscale engine -- NVIDIA RTX Video Super Resolution (2026-10-14).

The same super-resolution an RTX card applies to browser video, reached through
NVIDIA's ``nvidia-vfx`` Python package (``import nvvfx``), the binding Comfy-Org's
own ComfyUI_Nvidia_VFX_Nodes is built on. An optional second upscaler beside
``spandrel_esrgan``. ``off`` stays the canonical workflow's default (operator,
2026-10-14: "one of 2 upscale options but do not activate by default for
now"); every NVIDIA workflow variant selects it on ``cuda`` (operator,
2026-10-15: "every nvidia lane to 1080p"). It installs ``nvidia-vfx`` itself
on a capable NVIDIA card (below); wherever it still cannot run it falls back
to ``off`` with a warning, so a variant never fails on it.

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
NVIDIA workflow variants do -- and only once the machine has proved it can use
it (a CUDA device that is not a ROCm build, compute capability 7.5 / Turing or
newer, Windows or Linux on x86-64, Python 3.12+ for the abi3 wheel), it runs
``python -m pip install`` into ComfyUI's own interpreter -- or ``uv pip``
when that interpreter has no pip, as on ComfyUI Desktop installs that alias
pip to uv -- once per server process, then carries on. Choosing an NVIDIA
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

#: The SDK's floor: RTX Video Super Resolution runs on Turing (sm_75) or newer.
#: A Pascal/Maxwell GTX card would download the wheel only to fail the probe.
MIN_COMPUTE_CAPABILITY = (7, 5)


def _install_blocker(torch, dev):
    """Why installing the SDK here would be pointless, or ``""``. Checked
    BEFORE any download: the wheel is win_amd64 / manylinux x86-64, abi3 from
    CPython 3.12, and the effect needs sm_75+. (GTX 16xx is Turing without
    tensor cores and passes this check; the post-install probe catches it.)"""
    import platform
    import sys
    machine = platform.machine().lower()
    if sys.platform not in ("win32", "linux") or machine not in ("amd64", "x86_64"):
        return f"nvidia-vfx ships for Windows/Linux x86-64 only ({sys.platform} {machine})"
    if sys.version_info < (3, 12):
        return "nvidia-vfx needs Python 3.12 or newer (this is %d.%d)" % sys.version_info[:2]
    try:
        cap = tuple(torch.cuda.get_device_capability(dev))
    except Exception as exc:  # noqa: BLE001 -- an unreadable card is not a reason to download
        return f"could not read the card's compute capability ({type(exc).__name__}: {exc})"
    if cap < MIN_COMPUTE_CAPABILITY:
        return ("this card is compute capability %d.%d; RTX VSR needs %d.%d "
                "(Turing) or newer" % (cap + MIN_COMPUTE_CAPABILITY))
    return ""

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
    (``otr_proc.run``, whose allowlist admits the ``python`` interpreter).

    ``python -m pip`` first; when the interpreter has no pip module (ComfyUI
    Desktop venvs can be uv-managed with pip aliased to ``uv pip``), the same
    install through ``python -m uv pip install --python <this interpreter>``.
    ``--no-input`` so an index that wants credentials fails instead of waiting
    on stdin until the timeout."""
    import sys
    from .._otr_shared import proc as otr_proc

    def attempt(argv):
        try:
            r = otr_proc.run(argv, capture_output=True, text=True,
                             timeout=AUTO_INSTALL_TIMEOUT_S)
        except Exception as exc:  # noqa: BLE001 -- timeout, spawn failure
            return False, f"{type(exc).__name__}: {exc}"
        out = " ".join(((r.stdout or "") + " " + (r.stderr or "")).split())
        return r.returncode == 0, f"exit {r.returncode}: {out}"

    ok, detail = attempt([sys.executable, "-m", "pip", "install", "--no-input",
                          "--disable-pip-version-check", AUTO_INSTALL_REQUIREMENT])
    if ok or "No module named pip" not in detail:
        return ok, "pip " + detail
    uv_args = ["pip", "install", "--python", sys.executable,
               AUTO_INSTALL_REQUIREMENT]
    ok, uv_detail = attempt([sys.executable, "-m", "uv"] + uv_args)
    if not ok and "No module named uv" in uv_detail:
        # uv shipped as its own executable rather than a module in this venv.
        import shutil
        uv_exe = shutil.which("uv")
        if uv_exe:
            ok, uv_detail = attempt([uv_exe] + uv_args)
    return ok, "pip unavailable (%s); uv pip %s" % (detail[:120], uv_detail)


def _redacted(text):
    """Pip output with any credential-bearing run masked, via the pack's one
    redactor (index URLs or tokens can appear in pip's messages). Never
    raises, and fails CLOSED: if the redactor cannot run, the output is
    withheld rather than logged raw."""
    try:
        from .._otr_visual_assets import _redact_credentials
        return _redact_credentials(text)
    except Exception:  # noqa: BLE001 -- logging must never fail the install path
        return "<pip output withheld: the credential redactor was unavailable>"


def _import_nvvfx_installing_if_missing(blocker=""):
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
        if blocker:
            raise ModuleNotFoundError(
                f"nvvfx is not installed, and installing it here would not "
                f"help: {blocker}") from exc
        if _AUTO_INSTALL_OUTCOME is not None:
            raise ModuleNotFoundError(
                f"nvvfx is not installed and this server already tried to "
                f"install it ({_AUTO_INSTALL_OUTCOME})") from exc
    _LOG.warning("[OTR.upscale] rtx_vsr: nvidia-vfx is not installed; installing "
                 "%s into %s (about 540 MB, once)", AUTO_INSTALL_REQUIREMENT,
                 __import__("sys").executable)
    ok, detail = _pip_install_nvidia_vfx()
    detail = _redacted(detail)
    # The refusal is folded to one short line downstream, so the WHOLE pip
    # output goes to the log here, where a reader can see why it failed.
    _LOG.log(logging.INFO if ok else logging.WARNING,
             "[OTR.upscale] rtx_vsr: installing %s %s -- %s",
             AUTO_INSTALL_REQUIREMENT, "succeeded" if ok else "FAILED", detail)
    _AUTO_INSTALL_OUTCOME = ("installed" if ok else "failed") + f" -- {detail[-300:]}"
    if not ok:
        raise ModuleNotFoundError(f"installing {AUTO_INSTALL_REQUIREMENT} failed "
                                  f"({detail[-300:]})")
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
    # never redistributed by this pack (it is pip-installed from NVIDIA's own
    # PyPI release on the user's machine, see load()).
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
            nvvfx = _import_nvvfx_installing_if_missing(
                blocker=_install_blocker(torch, dev))
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
