"""
visual.backends.flux_anchor  --  Day 2, Stage 1 anchors
===============================================================

FLUX.1-dev torchao FP8 text-to-image anchor renderer.  Runs inside the
subprocess spawned by ``VisualBridge._spawn_sidecar`` when the user
picks ``backend="flux_anchor"`` in the node UI (or sets
``OTR_VISUAL_BACKEND=flux_anchor`` directly).

Dependencies (pinned in ``requirements.video.txt``):
    torch==2.10.0+cu130, diffusers==0.37.0, transformers==5.5.0,
    accelerate==1.13.0, sentencepiece, torchao==0.16.0

Model weights (NOT checked in):
    ``C:/Users/jeffr/Documents/ComfyUI/models/diffusers/FLUX.1-dev-torchao-fp8``
    (or override via ``OTR_FLUX_MODEL`` env var -- absolute path only).
    Pre-quantized via torchao ``float8_weight_only``, pickled ``.bin``
    files.  This is the only supported FLUX checkpoint on a 16 GB
    Blackwell card -- the stock BF16 path was removed 2026-04-19 because
    it PCIe-thrashed even under ``enable_sequential_cpu_offload()``
    when chained with a second pipeline (Wan2.1 / LTX).  See
    BUG-LOCAL-049 (FP8 swap) and BUG-LOCAL-050 (chained-stage
    pipe leak) for the full history.

Execution modes:

1. **Real mode** (default when weights exist and CUDA is available):
   builds a ``FluxPipeline`` from ``diffusers/FLUX.1-dev-torchao-fp8``
   via the diffusers folder loader (``use_safetensors=False``) with NO
   ``torch_dtype`` override (weights are already quantized), then
   ``enable_model_cpu_offload()``.  Peak VRAM ~10-11 GB.  Renders
   1024x1024 per shot.  Each shot's ``env_prompt`` is concatenated with
   ``camera`` and a small cinematic suffix before being fed to the
   pipeline.

2. **Stub mode** (``OTR_FLUX_STUB=1`` OR weights missing OR no CUDA OR
   pipeline load fails): emits a 1024x1024 solid PNG per shot tagged
   ``backend=flux_anchor`` with ``mode=stub``.  Lets the Day 2 dispatch
   path be smoke-tested without dragging down 17 GB of weights in CI.

Zero audio imports.  C7 audio byte-identical gate is unaffected.

BUG-LOCAL-050: ``_release_pipe()`` is called at the end of
``_render_real`` so the next chained backend (ltx_motion / wan21_loop)
starts with a clean VRAM slate.  Without this, CPU-offload hooks
from this pipeline stayed resident and collided with the next load,
producing PCIe thrashing at 100% GPU / 1 W / ~200 MB free.
"""

from __future__ import annotations

import gc
import os
import struct
import sys
import time
import traceback
import zlib
from pathlib import Path
from typing import Any

from ._base import (
    Backend,
    STATUS_ERROR,
    STATUS_OOM,
    STATUS_READY,
    STATUS_RUNNING,
    atomic_write_json,
    write_status,
)


# Model discovery.  User-overridable via env so conda env installs can
# point elsewhere without a code change.  Only the torchao FP8 folder
# is supported -- the BF16 path was removed 2026-04-19 (BUG-LOCAL-049
# /050).  If the folder is absent, ``_should_stub`` returns True.
_DEFAULT_MODEL_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\diffusers\FLUX.1-dev-torchao-fp8"
)
_MODEL_PATH = Path(os.environ.get("OTR_FLUX_MODEL", str(_DEFAULT_MODEL_PATH)))

# BUG-LOCAL-057 Step 2: Comfy-Org single-file safetensors (primary path).
# This is the same file ComfyUI-native loaded in 44.62s with peak 11350 MB
# VRAM tonight -- raw FP8 tensors, no torchao pickle, no v1/v2 config
# version drift. Preferred over the torchao folder because the pre-quantized
# diffusers/FLUX.1-dev-torchao-fp8 checkpoint was saved with
# Float8DynamicActivationFloat8WeightConfig v1 layout while torchao 0.16
# expects v2 -- triggers "size of tensor a (4096) must match tensor b (10240)"
# at pipe() call time. Single-file safetensors has neither layout concept.
_DEFAULT_SINGLE_FILE_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\checkpoints\flux1-dev-fp8.safetensors"
)
_SINGLE_FILE_PATH = Path(
    os.environ.get("OTR_FLUX_SINGLE_FILE", str(_DEFAULT_SINGLE_FILE_PATH))
)

# Day 2 gate: 1024x1024 square.  Must not change without amending the
# kill criteria table in ROADMAP.md.
_RENDER_WIDTH = 1024
_RENDER_HEIGHT = 1024

# Cinematic suffix applied to every env_prompt.  Matches the anchor_gen
# SD 1.5 path so Phase B / Day 2 outputs stay stylistically coherent.
_STYLE_SUFFIX = (
    "cinematic, 35mm film, anamorphic lens, volumetric lighting, "
    "heavy vignette, muted color grade, sharp focus"
)


def _stub_png(path: Path, r: int, g: int, b: int,
              width: int = _RENDER_WIDTH, height: int = _RENDER_HEIGHT) -> None:
    """Emit a valid 1024x1024 solid PNG with no external deps.

    Same minimal-PNG writer as ``placeholder_test`` and ``worker._create_placeholder_png``
    but scaled to the Day 2 render size so downstream nodes never need
    to conditionally resize based on mode.
    """
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes([0] + [r, g, b] * width)
    raw = row * height
    idat = zlib.compress(raw, 1)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", idat)
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def _build_prompt(shot: dict) -> str:
    env = (shot.get("env_prompt") or "").strip()
    camera = (shot.get("camera") or "").strip()
    parts = []
    if env:
        parts.append(env)
    if camera:
        parts.append(camera)
    parts.append(_STYLE_SUFFIX)
    return ", ".join(parts)


def _derive_seed(shot: dict, shot_idx: int, base: int = 0x0F1401) -> int:
    """Deterministic per-shot seed.  Same shot+base -> same image.

    ``shot_id`` is preferred because it survives re-orderings; fall back
    to shot index so shots with no explicit id still get a stable seed.
    """
    import hashlib
    key = f"{shot.get('shot_id', '')}|{shot_idx}|{base}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _should_stub() -> tuple[bool, str]:
    """Return (stub_requested, reason).  Reason is logged into meta.json.

    Stubs only when the torchao FP8 folder is missing (the only supported
    FLUX path since BUG-LOCAL-049 / 050 stripped the BF16 ladder).
    """
    if os.environ.get("OTR_FLUX_STUB", "").strip() == "1":
        return (True, "OTR_FLUX_STUB=1")
    if not _MODEL_PATH.exists():
        return (True, f"model_missing:{_MODEL_PATH}")
    # CUDA check is lazy -- deferred to real-mode entry so the import
    # stays torch-free here.  ``run()`` handles the fallback.
    return (False, "")


def _log_stderr(msg: str) -> None:
    """Loud log to sidecar stderr. No-op if stderr is unavailable.

    Added for BUG-LOCAL-047 / BUG-046 family so load-failure decisions
    leave a paper trail in ``sidecar_stderr.log`` instead of being silent.
    """
    try:
        sys.stderr.write(msg if msg.endswith("\n") else msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _install_torchao_compat_shim() -> str:
    """Install back-compat aliases for torchao symbols renamed in 0.16.

    BUG-LOCAL-051 (initial) -> 053 (hand-list expansion) -> 054
    (auto-discovery): the pre-quantized FLUX.1-dev-torchao-fp8 checkpoint
    was saved against an older torchao that exported every quantization
    Config as a matching lowercase helper function -- e.g.

        torchao.quantization.float8_dynamic_activation_float8_weight
        torchao.quantization.float8_static_activation_float8_weight
        torchao.quantization.float8_weight_only
        torchao.quantization.int4_weight_only

    torchao 0.16 removed the helpers; only the Config classes remain.
    Deserialising the checkpoint trips one ImportError per missing helper
    and walks the config tree one symbol at a time, so fixing one reveals
    the next.  053 tried to hand-maintain the list; the live 21:50:19 run
    tripped on *int4_weight_only* (BUG-LOCAL-054), proving whack-a-mole.

    054 switches to auto-discovery: walk every ``*Config`` class exported
    by ``torchao.quantization``, derive the expected snake_case helper
    name via CamelCase->snake_case with the trailing ``_config`` stripped,
    and alias each one whose helper is missing.  Pickle resolves callables
    by fully-qualified ``module.name``, so binding the class at the
    expected attribute is enough for unpickling; call sites that did
    ``helper(kwargs)`` now hit ``Config(kwargs)`` instead, which is the
    replacement API anyway.

    Idempotent: helpers that already exist (older torchao or a future
    re-introduction) are skipped.  Spurious aliases for non-checkpoint
    classes (e.g. ``Float8MMConfig``) are harmless because pickle only
    resolves what the checkpoint actually references.  Returns a
    structured tag tracking the canonical helper set so the stderr log
    tells future archaeologists exactly what was repaired.
    """
    import re
    try:
        import torchao.quantization as _q  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return f"torchao_import_failed:{type(exc).__name__}"

    _CAMEL_SPLIT = re.compile(r"([a-z0-9])([A-Z])")

    def _camel_to_snake(cls_name: str) -> str:
        # Strip trailing "Config" first, then insert underscore before every
        # uppercase letter that follows a lowercase letter or digit, then
        # lowercase.  Acronym runs ("AO", "MM", "UIntX") collapse to their
        # natural lowercase form -- good enough because pickle only asks
        # for the exact names the checkpoint needs.
        base = cls_name[: -len("Config")] if cls_name.endswith("Config") else cls_name
        return _CAMEL_SPLIT.sub(r"\1_\2", base).lower()

    installed: list[str] = []
    native: list[str] = []

    for attr in dir(_q):
        if not attr.endswith("Config"):
            continue
        if not attr[:1].isupper():
            continue
        cls = getattr(_q, attr, None)
        if not isinstance(cls, type):
            continue
        helper = _camel_to_snake(attr)
        if not helper:
            continue
        if hasattr(_q, helper):
            native.append(helper)
            continue
        setattr(_q, helper, cls)
        installed.append(helper)

    # BUG-LOCAL-055: acronym-to-word boundary edge cases that the
    # generic ``([a-z0-9])([A-Z])`` split can't tokenize.  torchao
    # exposes ``UIntXWeightOnlyConfig`` (the UIntX quantization format)
    # but the regex has no way to know that ``UIntX`` is a single token
    # -- it produces ``u_int_x_weight_only`` instead of the pickled
    # legacy name ``uintx_weight_only``.  Hand-alias the known edge
    # cases here.  Extend this table when new acronym-boundary symbols
    # surface in future checkpoints; the auto-discovery loop above
    # handles every case that CamelCase split can resolve unambiguously.
    _EXPLICIT_ACRONYM_ALIASES: dict[str, str] = {
        "uintx_weight_only": "UIntXWeightOnlyConfig",
    }
    explicit_state: dict[str, str] = {}
    for helper_name, cls_name in _EXPLICIT_ACRONYM_ALIASES.items():
        if hasattr(_q, helper_name):
            explicit_state[helper_name] = "N"
            native.append(helper_name)
            continue
        cls = getattr(_q, cls_name, None)
        if not isinstance(cls, type):
            explicit_state[helper_name] = "x"  # target class missing
            continue
        setattr(_q, helper_name, cls)
        installed.append(helper_name)
        explicit_state[helper_name] = "I"

    # Structured return tag: summary counts + explicit status for the
    # canonical helpers we know the FLUX checkpoint may reference.  The
    # "canonical" list is documentation, not enforcement -- auto-discovery
    # handles the rest.
    canonical = (
        "float8_dynamic_activation_float8_weight",
        "float8_static_activation_float8_weight",
        "float8_weight_only",
        "int4_weight_only",
        "int8_weight_only",
        "uintx_weight_only",
    )
    state: dict[str, str] = {}
    for h in canonical:
        if h in installed:
            state[h] = "I"
        elif h in native:
            state[h] = "N"
        else:
            state[h] = "-"  # neither installed nor native; likely class missing

    parts = [f"installed={len(installed)}", f"native={len(native)}"]
    parts.extend(f"{h}={state[h]}" for h in canonical)
    return "shim:" + ";".join(parts)


def _release_pipe(pipe) -> None:
    """Tear down a loaded FluxPipeline and its CPU-offload hooks.

    BUG-LOCAL-050: video_stack chains flux_anchor -> ltx_motion ->
    wan21_loop inside a single sidecar process.  Without explicit
    teardown, diffusers' accelerate-hooked modules stayed resident
    after ``run()`` returned -- the next pipeline loaded on top and the
    combined working set PCIe-thrashed at 100% GPU / ~1 W / ~200 MB
    free VRAM.  Calling this at the end of the render loop forces the
    hooks loose, drops the CUDA context for this pipe, and runs a
    conservative ``empty_cache()`` so the next stage starts from a
    clean baseline.

    Safe to call with ``pipe=None`` (no-op).
    """
    if pipe is None:
        return
    try:
        # ``remove_all_hooks`` is the accelerate-supported way to drop
        # offload hooks.  Not every diffusers version exposes it on the
        # pipeline object; fall through on AttributeError.
        remove = getattr(pipe, "remove_all_hooks", None)
        if callable(remove):
            remove()
    except Exception:
        pass
    try:
        # Best-effort component teardown.  del-ing the pipeline is
        # insufficient when accelerate has hooks on individual
        # submodules; we move them to CPU first so the CUDA allocator
        # drops their storage on the next empty_cache().
        for attr in ("transformer", "text_encoder", "text_encoder_2", "vae",
                     "scheduler", "tokenizer", "tokenizer_2"):
            mod = getattr(pipe, attr, None)
            if mod is None:
                continue
            to = getattr(mod, "to", None)
            if callable(to):
                try:
                    mod.to("cpu")
                except Exception:
                    pass
    except Exception:
        pass
    try:
        del pipe
    except Exception:
        pass
    try:
        gc.collect()
    except Exception:
        pass
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass
    _log_stderr("[flux_anchor] pipe released (del + gc + empty_cache)")


def _try_load_pipeline():
    """Load FluxPipeline with CPU offload.  Returns (pipe, reason, load_mode).

    Two-tier load (BUG-LOCAL-057 Step 2):

    **fp8_single_file** (PREFERRED) -- Comfy-Org's
    ``flux1-dev-fp8.safetensors`` (single 17.25 GB file, raw FP8 tensors,
    no pickle, no torchao dependency). Loaded via
    ``FluxPipeline.from_single_file(path, torch_dtype=torch.bfloat16)``.
    Matches the checkpoint ComfyUI-native renders in ~44s. Avoids the
    torchao v1/v2 tensor layout mismatch that the folder path hits.

    **fp8_torchao** (FALLBACK) -- legacy pre-quantized Diffusers folder
    at ``_MODEL_PATH``. Still attempted when the single-file path is
    missing OR ``from_single_file`` rejects the format. Weights are
    torchao ``float8_weight_only`` pickle .bin; we pass no ``torch_dtype``
    override and ``use_safetensors=False``. Known to raise RuntimeError
    on pipe() call with torchao 0.16 due to v1/v2 Float8 config layout
    drift (symptom: ``size of tensor a (4096) must match tensor b (10240)``).

    Returns ``(pipe, reason, load_mode)``. ``pipe`` is None when ALL paths
    failed -- caller falls back to stub mode and records the reason in
    meta.json.
    """
    try:
        import torch  # type: ignore
        from diffusers import FluxPipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, f"import_error:{type(exc).__name__}:{exc}", "none")

    if not torch.cuda.is_available():
        return (None, "cuda_unavailable", "none")

    failures: list[str] = []

    # Tier 1: Comfy-Org single-file safetensors via from_single_file.
    # This is the same checkpoint that rendered ComfyUI-native in 44s
    # with peak 11350 MB VRAM on the RTX 5080 tonight.
    #
    # Key decisions (per external diagnostic on meta-tensor failure):
    #   - Pass NO torch_dtype. Passing bfloat16 caused diffusers to
    #     create meta placeholders that never materialized from the
    #     FP8 safetensors, leading to the "Tensor on device meta is
    #     not on the expected device cpu!" error at pipe() call time.
    #     Letting diffusers use the safetensors' native dtype (FP8 for
    #     transformer, bf16 for text encoders) materializes cleanly.
    #   - Skip accelerate offload entirely and use pipe.to("cuda").
    #     The checkpoint's FP8 transformer is ~11 GB; text encoders
    #     + VAE bring peak to ~12-13 GB -- fits in 16 GB without the
    #     offload hooks that were failing on meta tensors.
    if _SINGLE_FILE_PATH.exists():
        _log_stderr(
            f"[flux_anchor] attempting FluxPipeline.from_single_file "
            f"load_mode=fp8_single_file path={_SINGLE_FILE_PATH}"
        )
        try:
            # low_cpu_mem_usage=False was tried in commit 312d1cd and
            # did NOT fix the meta-tensor issue -- Diffusers left params
            # in meta state regardless. Reverted per Jeffrey's request;
            # the 24 GB CPU RAM spike it caused wasn't buying us
            # anything. Back to Diffusers' default behavior.
            pipe = FluxPipeline.from_single_file(
                str(_SINGLE_FILE_PATH),
                # No torch_dtype -- let diffusers pick from safetensors.
                local_files_only=True,
            )
            _log_stderr("[flux_anchor] loaded pipeline load_mode=fp8_single_file")
            # Materialize the full pipeline on GPU. FP8 transformer +
            # bf16 encoders fit ~12-13 GB on a 16 GB card, matching the
            # ComfyUI-native baseline (peak 11350 MB).
            try:
                pipe.to("cuda")
                _log_stderr(
                    "[flux_anchor] pipeline materialized on cuda "
                    "(no offload -- single_file FP8 fits 16 GB directly)"
                )
            except Exception as place_exc:  # noqa: BLE001
                # If .to("cuda") OOMs, fall back to sequential offload.
                _log_stderr(
                    f"[flux_anchor] pipe.to('cuda') failed "
                    f"({type(place_exc).__name__}: {str(place_exc)[:120]}); "
                    f"falling back to enable_sequential_cpu_offload"
                )
                try:
                    pipe.enable_sequential_cpu_offload()
                    _log_stderr(
                        "[flux_anchor] enable_sequential_cpu_offload active"
                    )
                except Exception as off_exc:  # noqa: BLE001
                    _log_stderr(
                        f"[flux_anchor] enable_sequential_cpu_offload also "
                        f"failed ({type(off_exc).__name__}: "
                        f"{str(off_exc)[:120]}); continuing without offload"
                    )
            try:
                pipe.set_progress_bar_config(disable=True)
            except Exception:
                pass
            return (pipe, "loaded[fp8_single_file]", "fp8_single_file")
        except Exception as exc:  # noqa: BLE001
            err_text = str(exc)[:200]
            failures.append(
                f"fp8_single_file:{type(exc).__name__}:{err_text}"
            )
            _log_stderr(
                f"[flux_anchor] fp8_single_file load failed: "
                f"{type(exc).__name__}: {err_text}; falling back to fp8_torchao"
            )
    else:
        failures.append(f"fp8_single_file:missing:{_SINGLE_FILE_PATH}")
        _log_stderr(
            f"[flux_anchor] single-file safetensors not found at "
            f"{_SINGLE_FILE_PATH}; skipping to fp8_torchao fallback"
        )

    # Tier 2: legacy torchao FP8 Diffusers folder (BUG-LOCAL-049/050).
    if not _MODEL_PATH.exists():
        failures.append(f"fp8_torchao:missing:{_MODEL_PATH}")
        return (None, "|".join(failures), "none")

    # BUG-LOCAL-051: install the torchao back-compat shim BEFORE the
    # diffusers load, since from_pretrained unpickles weight metadata
    # that references helpers removed in torchao 0.16.
    shim_state = _install_torchao_compat_shim()
    _log_stderr(f"[flux_anchor] torchao compat shim: {shim_state}")

    _log_stderr(
        f"[flux_anchor] attempting FluxPipeline.from_pretrained "
        f"load_mode=fp8_torchao path={_MODEL_PATH}"
    )
    try:
        pipe = FluxPipeline.from_pretrained(
            str(_MODEL_PATH),
            use_safetensors=False,
            local_files_only=True,
        )
    except Exception as exc:  # noqa: BLE001
        err_text = str(exc)[:200]
        failures.append(f"fp8_torchao:{type(exc).__name__}:{err_text}")
        _log_stderr(
            f"[flux_anchor] fp8_torchao load failed: "
            f"{type(exc).__name__}: {err_text}"
        )
        return (None, "|".join(failures), "none")

    _log_stderr("[flux_anchor] loaded pipeline load_mode=fp8_torchao")

    try:
        pipe.enable_model_cpu_offload()
    except Exception as exc:  # noqa: BLE001
        _log_stderr(
            f"[flux_anchor] enable_model_cpu_offload failed "
            f"({type(exc).__name__}: {str(exc)[:120]}); continuing without offload"
        )
    try:
        pipe.set_progress_bar_config(disable=True)
    except Exception:
        pass
    return (pipe, "loaded[fp8_torchao]", "fp8_torchao")


class FluxAnchorBackend(Backend):
    """Day 2 -- FLUX.1-dev torchao FP8 text-to-image anchor renderer."""

    name = "flux_anchor"

    def run(self, job_dir: Path) -> None:
        out_dir = self.out_dir_for(job_dir)
        try:
            shots = self.load_shotlist(job_dir)
        except (FileNotFoundError, ValueError) as exc:
            write_status(out_dir, STATUS_ERROR, f"{type(exc).__name__}: {exc}")
            return

        if not shots:
            write_status(out_dir, STATUS_ERROR, "shotlist has zero shots")
            return

        stub, stub_reason = _should_stub()
        if stub:
            write_status(
                out_dir, STATUS_RUNNING,
                f"flux_anchor stub mode: {len(shots)} shots ({stub_reason})",
            )
            self._render_stub(shots, out_dir, stub_reason)
            return

        # Real mode -- attempt pipeline load.  Any failure falls back to
        # stub rather than surfacing ERROR, so downstream nodes always
        # have PNGs to work with even when weights are mid-download.
        # Single supported source: pre-quantized FP8 folder (BUG-LOCAL-049).
        write_status(
            out_dir, STATUS_RUNNING,
            f"flux_anchor real mode: loading FLUX.1-dev (fp8_torchao) "
            f"from {_MODEL_PATH}",
        )
        pipe, reason, load_mode = _try_load_pipeline()
        if pipe is None:
            write_status(
                out_dir, STATUS_RUNNING,
                f"flux_anchor load failed ({reason}); falling back to stub",
            )
            self._render_stub(shots, out_dir, reason)
            return

        _log_stderr(f"[flux_anchor] load_mode={load_mode}")
        try:
            self._render_real(pipe, shots, out_dir, load_mode)
        finally:
            # BUG-LOCAL-050: always release the pipe before returning so
            # the next chained backend (ltx_motion / wan21_loop) starts
            # with a clean VRAM slate.  Runs even on exception so a
            # mid-render failure doesn't leave hooks resident.
            _release_pipe(pipe)

    # ------------------------------------------------------------------
    # stub path -- CI-safe, no torch, no GPU
    # ------------------------------------------------------------------
    def _render_stub(self, shots: list[dict], out_dir: Path, reason: str) -> None:
        for i, shot in enumerate(shots):
            shot_id = shot.get("shot_id", f"shot_{i:03d}")
            shot_dir = out_dir / shot_id
            shot_dir.mkdir(parents=True, exist_ok=True)
            # Deterministic distinct colors per shot so visual review
            # can tell stub outputs apart at a glance.
            r = 40 + (i * 23) % 200
            g = 40 + (i * 41) % 200
            b = 40 + (i * 67) % 200
            _stub_png(shot_dir / "render.png", r, g, b)
            atomic_write_json(shot_dir / "meta.json", {
                "shot_id": shot_id,
                "backend": self.name,
                "mode": "stub",
                "reason": reason,
                "prompt": _build_prompt(shot),
                "seed": _derive_seed(shot, i),
                "width": _RENDER_WIDTH,
                "height": _RENDER_HEIGHT,
                "env_prompt": shot.get("env_prompt", ""),
                "camera": shot.get("camera", ""),
                "duration_sec": float(shot.get("duration_sec", 9)),
            })
        write_status(
            out_dir, STATUS_READY,
            f"flux_anchor stub READY: {len(shots)} shots ({reason})",
            backend=self.name, mode="stub",
        )

    # ------------------------------------------------------------------
    # real path -- loads torch lazily, uses VRAMCoordinator gate
    # ------------------------------------------------------------------
    def _render_real(self, pipe, shots: list[dict], out_dir: Path,
                     load_mode: str = "unknown") -> None:
        # Lazy imports so stub path stays torch-free.
        import torch  # type: ignore

        # VRAMCoordinator isolates the FLUX run from any concurrent
        # audio-side passes (Bark/MusicGen) so peak-VRAM spikes don't
        # collide.  Same import dance as worker.py.
        try:
            from visual.vram_coordinator import VRAMCoordinator  # type: ignore
        except ImportError:
            try:
                hw = Path(__file__).resolve().parent.parent
                if str(hw) not in sys.path:
                    sys.path.insert(0, str(hw))
                from vram_coordinator import VRAMCoordinator  # type: ignore
            except ImportError:  # last-resort noop
                class VRAMCoordinator:  # type: ignore
                    def acquire(self, *a, **kw):
                        from contextlib import contextmanager

                        @contextmanager
                        def _n():
                            yield self
                        return _n()

        coord = VRAMCoordinator()
        rendered = 0
        oom = 0
        errored = 0

        # BUG-LOCAL-057: previously this loop had zero stderr logging, so when
        # pipe() raised per-shot exceptions the errors were written ONLY to
        # meta.json -- which wan21_loop later overwrites with its own
        # "still_missing" meta, erasing the forensic trail. Log each stage
        # transition to stderr so the truth persists regardless of downstream
        # overwrites. Also write flux_anchor.meta.json alongside meta.json so
        # the per-stage history survives the chained-backend overwrite.
        # Single-shot isolation mode. Set OTR_FLUX_SINGLE_SHOT=1 to render
        # only the first shot and skip shots 2..N. Useful for isolating
        # load-path bugs (meta-tensor, dtype, OOM) from any loop or
        # chain-state issues. The batching loop code below is fully
        # preserved -- this just adds an early break when the flag is set.
        # Unset the env var to restore the full 9-shot run.
        _single_shot_mode = os.environ.get("OTR_FLUX_SINGLE_SHOT", "").strip() == "1"
        if _single_shot_mode:
            _log_stderr(
                "[flux_anchor] SINGLE-SHOT MODE active (OTR_FLUX_SINGLE_SHOT=1) "
                "-- rendering shot 1 only, skipping shots 2..N"
            )

        _log_stderr(
            f"[flux_anchor] _render_real entered: shots={len(shots)} "
            f"load_mode={load_mode} out_dir={out_dir}"
            + (" [single-shot]" if _single_shot_mode else "")
        )
        with coord.acquire(owner="flux_anchor", job_id=out_dir.name, timeout=1800):
            _log_stderr("[flux_anchor] VRAMCoordinator acquired")
            for i, shot in enumerate(shots):
                shot_id = shot.get("shot_id", f"shot_{i:03d}")
                shot_dir = out_dir / shot_id
                shot_dir.mkdir(parents=True, exist_ok=True)

                prompt = _build_prompt(shot)
                seed = _derive_seed(shot, i)
                generator = torch.Generator(device="cuda").manual_seed(seed)

                _log_stderr(
                    f"[flux_anchor] shot {i+1}/{len(shots)} {shot_id} "
                    f"prompt_len={len(prompt)} seed={seed} start"
                )

                t0 = time.perf_counter()
                try:
                    out = pipe(
                        prompt,
                        width=_RENDER_WIDTH,
                        height=_RENDER_HEIGHT,
                        num_inference_steps=20,
                        guidance_scale=3.5,
                        generator=generator,
                    )
                    img = out.images[0]
                except torch.cuda.OutOfMemoryError:
                    oom += 1
                    oom_meta = {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "load_mode": load_mode,
                        "error": "CUDA_OOM",
                        "prompt": prompt,
                        "seed": seed,
                    }
                    atomic_write_json(shot_dir / "meta.json", oom_meta)
                    atomic_write_json(shot_dir / "flux_anchor.meta.json", oom_meta)
                    _log_stderr(
                        f"[flux_anchor] shot {i+1}/{len(shots)} {shot_id} "
                        f"OOM after {time.perf_counter() - t0:.1f}s"
                    )
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    if _single_shot_mode:
                        _log_stderr(
                            "[flux_anchor] single-shot mode: stopping "
                            "after shot 1 (OOM'd)"
                        )
                        break
                    continue
                except Exception as exc:  # noqa: BLE001
                    errored += 1
                    err_meta = {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "load_mode": load_mode,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "traceback_tail": traceback.format_exc()[-800:],
                        "prompt": prompt,
                        "seed": seed,
                    }
                    atomic_write_json(shot_dir / "meta.json", err_meta)
                    atomic_write_json(shot_dir / "flux_anchor.meta.json", err_meta)
                    _log_stderr(
                        f"[flux_anchor] shot {i+1}/{len(shots)} {shot_id} "
                        f"ERRORED after {time.perf_counter() - t0:.1f}s: "
                        f"{type(exc).__name__}: {str(exc)[:200]}"
                    )
                    if _single_shot_mode:
                        _log_stderr(
                            "[flux_anchor] single-shot mode: stopping "
                            "after shot 1 (errored)"
                        )
                        break
                    continue

                elapsed = time.perf_counter() - t0
                img.save(shot_dir / "render.png", format="PNG")
                ok_meta = {
                    "shot_id": shot_id,
                    "backend": self.name,
                    "mode": "real",
                    "load_mode": load_mode,
                    "prompt": prompt,
                    "seed": seed,
                    "width": _RENDER_WIDTH,
                    "height": _RENDER_HEIGHT,
                    "render_time_s": round(elapsed, 2),
                    "env_prompt": shot.get("env_prompt", ""),
                    "camera": shot.get("camera", ""),
                    "duration_sec": float(shot.get("duration_sec", 9)),
                }
                atomic_write_json(shot_dir / "meta.json", ok_meta)
                atomic_write_json(shot_dir / "flux_anchor.meta.json", ok_meta)
                _log_stderr(
                    f"[flux_anchor] shot {i+1}/{len(shots)} {shot_id} "
                    f"rendered in {elapsed:.1f}s"
                )
                rendered += 1

                # Single-shot mode: stop after the first iteration
                # regardless of whether it succeeded, OOM'd, or raised.
                # The OOM and exception branches `continue` above, which
                # bypasses this check -- the same single-shot break
                # logic therefore lives inside both except blocks too.
                if _single_shot_mode:
                    _log_stderr(
                        "[flux_anchor] single-shot mode: stopping after "
                        "shot 1 (remaining shots skipped)"
                    )
                    break

        _log_stderr(
            f"[flux_anchor] _render_real done: rendered={rendered} "
            f"oom={oom} errored={errored}"
            + (" [single-shot]" if _single_shot_mode else "")
        )

        if oom > 0:
            write_status(
                out_dir, STATUS_OOM,
                f"flux_anchor OOM on {oom}/{len(shots)} shots "
                f"(rendered={rendered}, errored={errored}, load_mode={load_mode})",
                backend=self.name, mode="real", load_mode=load_mode,
                rendered=rendered, oom=oom, errored=errored,
            )
        elif rendered == 0 and errored > 0:
            # BUG-LOCAL-058: every shot raised an exception. Don't
            # write STATUS_READY here -- this stage IS a failure. Use
            # STATUS_RUNNING so video_stack's downstream stages still
            # run (they stub-fallback gracefully when render.png is
            # missing) and the FINAL STATUS is video_stack's own
            # "missing=N" composite READY, not a misleading intermediate
            # "flux_anchor READY: 0 rendered". Prevents VisualPoll from
            # unblocking on flux_anchor's premature terminal marker.
            write_status(
                out_dir, STATUS_RUNNING,
                f"flux_anchor failed all shots: 0/{len(shots)} rendered, "
                f"{errored} errored (load_mode={load_mode}); "
                f"chain continues with stub anchors",
                backend=self.name, mode="real", load_mode=load_mode,
                rendered=rendered, errored=errored,
            )
        else:
            write_status(
                out_dir, STATUS_READY,
                f"flux_anchor READY: {rendered}/{len(shots)} rendered"
                + (f", {errored} errored" if errored else "")
                + f" (load_mode={load_mode})",
                backend=self.name, mode="real", load_mode=load_mode,
                rendered=rendered, errored=errored,
            )
