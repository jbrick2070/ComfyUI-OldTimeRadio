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

    Single supported path (BUG-LOCAL-049 / 050):

    **fp8_torchao** -- pre-quantized checkpoint at ``_MODEL_PATH``.
    Weights are already torchao ``float8_weight_only`` on disk, so we
    pass *no* ``torch_dtype`` override and ``use_safetensors=False``
    (the repo ships pickled .bin files).  This bypasses the torch 2.10
    ``Float8_e4m3fnStorage`` safetensors deserialization bug entirely.
    Peak VRAM ~10-11 GB with ``enable_model_cpu_offload()``.

    Returns ``(pipe, reason, load_mode)``.  ``pipe`` is None when the
    load failed -- caller falls back to stub mode and records the
    reason in meta.json.  No dtype override env var is honoured:
    the BF16 ladder was removed with BUG-LOCAL-050 because a second
    pipeline (Wan2.1 / LTX) in the same sidecar thrashed the bus.
    """
    try:
        import torch  # type: ignore
        from diffusers import FluxPipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, f"import_error:{type(exc).__name__}:{exc}", "none")

    if not torch.cuda.is_available():
        return (None, "cuda_unavailable", "none")

    if not _MODEL_PATH.exists():
        return (None, f"model_missing:{_MODEL_PATH}", "none")

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
        reason = f"load_error[fp8_torchao]:{type(exc).__name__}:{err_text}"
        _log_stderr(
            f"[flux_anchor] fp8_torchao load failed: "
            f"{type(exc).__name__}: {err_text}"
        )
        return (None, reason, "none")

    _log_stderr("[flux_anchor] loaded pipeline load_mode=fp8_torchao")

    try:
        pipe.enable_model_cpu_offload()
    except Exception as exc:  # noqa: BLE001
        _log_stderr(
            f"[flux_anchor] enable_model_cpu_offload failed "
            f"({type(exc).__name__}: {str(exc)[:120]}); continuing without offload"
        )
    # Silence diffusers progress bars so the sidecar stdout log
    # doesn't blow up with tqdm carriage-returns.  Mirrors the
    # anchor_gen fix for BUG-LOCAL-043.
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

        with coord.acquire(owner="flux_anchor", job_id=out_dir.name, timeout=1800):
            for i, shot in enumerate(shots):
                shot_id = shot.get("shot_id", f"shot_{i:03d}")
                shot_dir = out_dir / shot_id
                shot_dir.mkdir(parents=True, exist_ok=True)

                prompt = _build_prompt(shot)
                seed = _derive_seed(shot, i)
                generator = torch.Generator(device="cuda").manual_seed(seed)

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
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "load_mode": load_mode,
                        "error": "CUDA_OOM",
                        "prompt": prompt,
                        "seed": seed,
                    })
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                    continue
                except Exception as exc:  # noqa: BLE001
                    errored += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "load_mode": load_mode,
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "traceback_tail": traceback.format_exc()[-800:],
                        "prompt": prompt,
                        "seed": seed,
                    })
                    continue

                elapsed = time.perf_counter() - t0
                img.save(shot_dir / "render.png", format="PNG")
                atomic_write_json(shot_dir / "meta.json", {
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
                })
                rendered += 1

        if oom > 0:
            write_status(
                out_dir, STATUS_OOM,
                f"flux_anchor OOM on {oom}/{len(shots)} shots "
                f"(rendered={rendered}, errored={errored}, load_mode={load_mode})",
                backend=self.name, mode="real", load_mode=load_mode,
                rendered=rendered, oom=oom, errored=errored,
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
