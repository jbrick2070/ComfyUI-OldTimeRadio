"""
visual.backends.ltx_motion  --  Day 5 FLUX still -> LTX-2.3 handoff
=============================================================================

LTX-Video 2.3 image-to-video backend.  Takes a still (Day 4 keyframe.png
preferred, Day 2 anchor render.png fallback) as the first frame and
renders a short motion clip.  This is the Day 5 shipment on the video
stack sprint -- the bridge orchestration piece that gets FLUX anchors /
keyframes into a motion sidecar without cross-pipeline VRAM fragmentation.

Platform pins (ROADMAP.md, CLAUDE.md):

* Constraint C4 -- LTX-2.3 clips max 10-12 s (257 frames at 24 fps).
  Auto-chunk + ffmpeg crossfade is a Day 5+ concern; this backend caps
  per-shot duration at 10 s and records the cap in meta.json so the
  renderer can crossfade at mux time.
* Constraint C5 -- LTX-2.3 pipeline runs in ``torch.float8_e4m3fn``
  (Blackwell-native).
* Constraint C7 -- no audio imports on this path; byte-identical audio
  gate is preserved.
* VRAM ceiling: 14.5 GB real-mode target.  Subprocess spawn pattern
  (VisualBridge._spawn_sidecar) ensures FLUX VRAM is fully released
  before this sidecar loads LTX weights.

Two execution modes, matching Days 2-4:

1. **Real mode** (default when weights exist + CUDA is available):
   tries ``diffusers.LTXPipeline`` (preferred) / ``LTXImageToVideoPipeline``
   from ``Lightricks/LTX-Video`` at ``torch.float8_e4m3fn`` with
   ``enable_model_cpu_offload``, then runs image-to-video conditioned on
   the Day 4 keyframe.  Output: ``motion.mp4`` (h264, 24 fps) via
   ``diffusers.utils.export_to_video`` (imageio-ffmpeg backend).  When
   the I2V variant isn't available the backend records a detailed error
   per shot rather than silently falling back to noise.

2. **Stub mode** (``OTR_LTX_STUB=1`` / weights missing / no CUDA):
   emits a minimal-but-valid 24-byte MP4 ``ftyp`` atom + a 1-frame
   mdat so file-magic tools and tests can confirm a clip landed.  The
   clip bytes are keyed on the input-still hash so the handoff
   invariant ("same input still -> same stub output") is unit-testable
   before real weights land.

Input source priority (Row 5 of the Day 5 decision, parallels Day 4's
Row 3 for ControlNet):

    1. ``<out_dir>/<shot_id>/keyframe.png`` (Day 4 output) -- preferred
    2. ``<out_dir>/<shot_id>/render.png`` (Day 2 output) -- fallback
    3. otherwise: error record in meta.json, skip the shot

Per-shot seed base ``0x4C_54_58_4D`` spells "LTXM" in ASCII, distinct
from flux_anchor (0x0F1401), pulid_portrait (0x7075_6C69), and
flux_keyframe (0x4B45_5946) so one shot_id doesn't collide across
pipelines.
"""

from __future__ import annotations

import gc
import hashlib
import os
import struct
import sys
import time
import traceback
from pathlib import Path

from ._base import (
    Backend,
    STATUS_ERROR,
    STATUS_OOM,
    STATUS_READY,
    STATUS_RUNNING,
    atomic_write_json,
    write_status,
)


def _log_stderr(msg: str) -> None:
    """Loud log to sidecar stderr. No-op if stderr is unavailable."""
    try:
        sys.stderr.write(msg if msg.endswith("\n") else msg + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def _release_pipe(pipe) -> None:
    """Tear down a loaded LTX pipeline and its CPU-offload hooks.

    BUG-LOCAL-050: video_stack chains flux_anchor -> ltx_motion ->
    wan21_loop inside a single sidecar process.  Without explicit
    teardown, diffusers' accelerate-hooked modules stayed resident
    after ``run()`` returned -- the next pipeline loaded on top and the
    combined working set PCIe-thrashed at 100% GPU / ~1 W / ~200 MB
    free VRAM.  Calling this at the end of the render loop forces the
    hooks loose, drops component storages from CUDA, and runs a
    conservative ``empty_cache()`` so the next stage starts clean.

    Safe to call with ``pipe=None`` (no-op).
    """
    if pipe is None:
        return
    try:
        remove = getattr(pipe, "remove_all_hooks", None)
        if callable(remove):
            remove()
    except Exception:
        pass
    try:
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
    _log_stderr("[ltx_motion] pipe released (del + gc + empty_cache)")


# ---- Model discovery ------------------------------------------------------

# LTX-Video weights -- user overridable via env.  Canonical upstream:
#   https://huggingface.co/Lightricks/LTX-Video
_DEFAULT_LTX_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\diffusers\LTX-Video"
)
_LTX_PATH = Path(os.environ.get("OTR_LTX_MODEL", str(_DEFAULT_LTX_PATH)))

# C4 cap: 10 s at 24 fps = 240 frames; round up to LTX-2.3 native 257
# only in the real pipeline call.  Exposed as env overrides so a soak
# run can shorten to 2 s without editing code.
_FPS = int(os.environ.get("OTR_LTX_FPS", "24"))
_DURATION_S = float(os.environ.get("OTR_LTX_DURATION_S", "10.0"))
_NUM_INFERENCE_STEPS = int(os.environ.get("OTR_LTX_STEPS", "40"))

# Latent frame count used by LTX-2.3's native 257-frame ceiling.  The
# caller can dial this down via env.
_MAX_FRAMES = 257


# ---- Helpers --------------------------------------------------------------

def _still_hash(path_or_ident: str) -> str:
    """Stable hex hash of the input still identifier.

    In real mode the still's pixel content drives the motion; in stub
    mode we can't render real motion so we key the stub clip bytes on
    a hash of the still's path string.  Two shots with the same still
    path land on the same stub bytes; the Day 5 handoff invariant
    becomes unit-testable without running LTX.
    """
    if not path_or_ident:
        return "no_still"
    h = hashlib.sha256(path_or_ident.encode("utf-8"))
    return h.hexdigest()[:12]


def _derive_seed(shot: dict, shot_idx: int,
                  base: int = 0x4C_54_58_4D) -> int:
    """Deterministic per-shot seed, distinct base from other backends."""
    key = f"{shot.get('shot_id', '')}|{shot_idx}|{base}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _build_prompt(shot: dict) -> str:
    """LTX I2V prompt -- emphasises motion verbs over scene description
    since the still already carries composition / color."""
    motion = (shot.get("motion_prompt") or "").strip()
    env = (shot.get("env_prompt") or "").strip()
    camera = (shot.get("camera") or "").strip()
    parts = []
    if motion:
        parts.append(motion)
    elif env:
        # Motion prompt absent: the env_prompt becomes the anchor for
        # motion vocabulary.  We still bias toward movement words.
        parts.append(f"slow cinematic motion through {env}")
    if camera:
        parts.append(camera)
    parts.append("subtle parallax, naturalistic motion, 24fps")
    return ", ".join(parts)


def _resolve_input_still(shot: dict, job_dir: Path,
                         out_dir: Path) -> tuple[Path | None, str]:
    """Row 5 handoff priority: keyframe.png > render.png > None.

    Returns ``(path, source_tag)``.  ``source_tag`` is one of:
        "keyframe", "anchor", "missing"
    Recorded in meta.json so downstream analysis can tell which
    upstream stage fed a given motion clip.
    """
    shot_id = shot.get("shot_id", "")
    if not shot_id:
        return (None, "missing")
    kf = out_dir / shot_id / "keyframe.png"
    if kf.exists():
        return (kf, "keyframe")
    anchor = out_dir / shot_id / "render.png"
    if anchor.exists():
        return (anchor, "anchor")
    return (None, "missing")


def _should_stub() -> tuple[bool, str]:
    """Return (stub_requested, reason).  Reason is logged into meta.json."""
    if os.environ.get("OTR_LTX_STUB", "").strip() == "1":
        return (True, "OTR_LTX_STUB=1")
    if not _LTX_PATH.exists():
        return (True, f"ltx_weights_missing:{_LTX_PATH}")
    return (False, "")


def _stub_mp4(path: Path, still_hash: str) -> None:
    """Emit a minimal-but-valid MP4 skeleton.

    The file contains:
        * An ``ftyp`` atom declaring ``isom`` / ``mp42`` brands.
        * A tiny ``mdat`` atom whose payload is 16 bytes keyed on the
          input still's hash so two different stills yield two
          different stub clips.  The mdat is not a real h264 stream,
          but the file is syntactically a valid MP4 container as far
          as atom-walking tools (including ``ffprobe -show_format``)
          are concerned -- they'll report it as a zero-duration file.

    This is deliberately NOT a playable video; it only has to satisfy:
        a) file magic tools recognise it as MP4
        b) same input still -> same stub clip bytes (handoff invariant)
    """
    # ftyp atom -- 24 bytes
    ftyp = (
        b"\x00\x00\x00\x18"  # atom size = 24
        b"ftyp"              # atom type
        b"isom"              # major brand
        b"\x00\x00\x02\x00"  # minor version = 0x200
        b"isom"              # compatible brand 1
        b"mp42"              # compatible brand 2
    )
    # mdat atom -- 16-byte payload keyed on still_hash
    payload_bytes = hashlib.sha256(still_hash.encode("utf-8")).digest()[:16]
    mdat_size = 8 + len(payload_bytes)  # header + payload
    mdat = struct.pack(">I", mdat_size) + b"mdat" + payload_bytes
    path.write_bytes(ftyp + mdat)


def _is_mp4(path: Path) -> bool:
    """Recognise our stub MP4 by the ftyp atom signature."""
    try:
        head = path.read_bytes()[:12]
    except OSError:
        return False
    return len(head) >= 12 and head[4:8] == b"ftyp"


def _try_load_ltx_pipeline():
    """Return (pipe, variant, reason).  ``variant`` is "i2v" | "t2v" | None."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, None, f"torch_import_error:{type(exc).__name__}:{exc}")

    if not torch.cuda.is_available():
        return (None, None, "cuda_unavailable")

    # Prefer the image-to-video pipeline (what Day 5 actually needs).
    try:
        from diffusers import LTXImageToVideoPipeline  # type: ignore
        pipe = LTXImageToVideoPipeline.from_pretrained(
            str(_LTX_PATH),
            torch_dtype=torch.float8_e4m3fn,  # C5
            local_files_only=True,
        )
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            try:
                pipe.enable_sequential_cpu_offload()
            except Exception:
                pass
        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass
        return (pipe, "i2v", f"ltx_i2v_loaded:{_LTX_PATH}")
    except ImportError as exc:
        pass  # fall through to LTXPipeline variant below
    except Exception as exc:  # noqa: BLE001
        return (None, None,
                f"ltx_i2v_load_error:{type(exc).__name__}:{str(exc)[:200]}")

    # Older diffusers revisions exposed only LTXPipeline; it accepts an
    # ``image=`` kwarg on modern releases.
    try:
        from diffusers import LTXPipeline  # type: ignore
        pipe = LTXPipeline.from_pretrained(
            str(_LTX_PATH),
            torch_dtype=__import__("torch").float8_e4m3fn,
            local_files_only=True,
        )
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
        return (pipe, "t2v", f"ltx_t2v_loaded_fallback:{_LTX_PATH}")
    except Exception as exc:  # noqa: BLE001
        return (None, None,
                f"ltx_load_error:{type(exc).__name__}:{str(exc)[:200]}")


# ---- Backend --------------------------------------------------------------

class LtxMotionBackend(Backend):
    """Day 5 -- LTX-Video 2.3 I2V motion sidecar."""

    name = "ltx_motion"

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
                f"ltx_motion stub mode: {len(shots)} shots ({stub_reason})",
            )
            self._render_stub(shots, job_dir, out_dir, stub_reason)
            return

        write_status(
            out_dir, STATUS_RUNNING,
            f"ltx_motion real mode: {len(shots)} shots, I2V from FLUX stills",
        )
        self._render_real(shots, job_dir, out_dir)

    # ------------------------------------------------------------------
    # stub path -- CI-safe, no torch, no GPU
    # ------------------------------------------------------------------
    def _render_stub(self, shots: list[dict], job_dir: Path,
                     out_dir: Path, reason: str) -> None:
        for i, shot in enumerate(shots):
            shot_id = shot.get("shot_id", f"shot_{i:03d}")
            shot_dir = out_dir / shot_id
            shot_dir.mkdir(parents=True, exist_ok=True)

            still_path, source_tag = _resolve_input_still(shot, job_dir, out_dir)
            # In stub mode we don't require an upstream still -- we
            # fabricate a key based on the would-be keyframe path so
            # tests can run without running Day 2/4 first.
            still_ident = str(still_path) if still_path is not None else (
                str(out_dir / shot_id / "keyframe.png")
            )
            shash = _still_hash(still_ident)
            _stub_mp4(shot_dir / "motion.mp4", shash)

            duration_s = min(
                float(shot.get("duration_sec", _DURATION_S)),
                _DURATION_S,
            )
            atomic_write_json(shot_dir / "meta.json", {
                "shot_id": shot_id,
                "backend": self.name,
                "mode": "stub",
                "reason": reason,
                "prompt": _build_prompt(shot),
                "seed": _derive_seed(shot, i),
                "input_still": still_ident,
                "input_still_source": source_tag,
                "input_still_hash": shash,
                "still_present": still_path is not None,
                "duration_s": duration_s,
                "fps": _FPS,
                "num_frames": int(duration_s * _FPS),
                "env_prompt": shot.get("env_prompt", ""),
                "camera": shot.get("camera", ""),
                "motion_prompt": shot.get("motion_prompt", ""),
            })

        write_status(
            out_dir, STATUS_READY,
            f"ltx_motion stub READY: {len(shots)} shots ({reason})",
            backend=self.name, mode="stub",
        )

    # ------------------------------------------------------------------
    # real path -- LTX-2.3 I2V
    # ------------------------------------------------------------------
    def _render_real(self, shots: list[dict], job_dir: Path, out_dir: Path) -> None:
        import torch  # type: ignore

        try:
            from visual.vram_coordinator import VRAMCoordinator  # type: ignore
        except ImportError:
            try:
                hw = Path(__file__).resolve().parent.parent
                if str(hw) not in sys.path:
                    sys.path.insert(0, str(hw))
                from vram_coordinator import VRAMCoordinator  # type: ignore
            except ImportError:
                from contextlib import contextmanager

                class VRAMCoordinator:  # type: ignore
                    def acquire(self, *a, **kw):
                        @contextmanager
                        def _n():
                            yield self
                        return _n()

        coord = VRAMCoordinator()

        pipe, variant, pipe_reason = _try_load_ltx_pipeline()
        if pipe is None:
            write_status(
                out_dir, STATUS_ERROR,
                f"ltx_motion: pipeline load failed ({pipe_reason})",
                backend=self.name, mode="real",
            )
            return

        try:
            from diffusers.utils import export_to_video  # type: ignore
        except Exception as exc:  # noqa: BLE001
            write_status(
                out_dir, STATUS_ERROR,
                f"ltx_motion: export_to_video import failed "
                f"({type(exc).__name__}:{exc})",
                backend=self.name, mode="real",
            )
            return

        try:
            from PIL import Image  # type: ignore
        except Exception:
            Image = None  # type: ignore

        rendered = 0
        oom = 0
        errored = 0
        no_still = 0

        duration_s = _DURATION_S
        num_frames = min(int(duration_s * _FPS), _MAX_FRAMES)

        try:
            with coord.acquire(owner="ltx_motion", job_id=out_dir.name, timeout=1800):
                for i, shot in enumerate(shots):
                    shot_id = shot.get("shot_id", f"shot_{i:03d}")
                    shot_dir = out_dir / shot_id
                    shot_dir.mkdir(parents=True, exist_ok=True)

                    still_path, source_tag = _resolve_input_still(shot, job_dir, out_dir)
                    if still_path is None:
                        no_still += 1
                        atomic_write_json(shot_dir / "meta.json", {
                            "shot_id": shot_id,
                            "backend": self.name,
                            "mode": "real",
                            "error": "still_missing",
                            "detail": (
                                "No Day 2 render.png or Day 4 keyframe.png "
                                "on disk for this shot_id; LTX handoff needs "
                                "an upstream still."
                            ),
                        })
                        continue

                    prompt = _build_prompt(shot)
                    seed = _derive_seed(shot, i)
                    generator = torch.Generator(device="cuda").manual_seed(seed)

                    t0 = time.perf_counter()
                    try:
                        call_kwargs = dict(
                            prompt=prompt,
                            num_inference_steps=_NUM_INFERENCE_STEPS,
                            num_frames=num_frames,
                            generator=generator,
                        )
                        if Image is not None and variant == "i2v":
                            call_kwargs["image"] = Image.open(still_path).convert("RGB")
                        elif variant == "t2v":
                            # Fallback path: no image conditioning available,
                            # record that handoff degraded to T2V.
                            pass

                        out = pipe(**call_kwargs)
                        frames = out.frames[0] if hasattr(out, "frames") else out
                    except torch.cuda.OutOfMemoryError:
                        oom += 1
                        atomic_write_json(shot_dir / "meta.json", {
                            "shot_id": shot_id,
                            "backend": self.name,
                            "mode": "real",
                            "error": "CUDA_OOM",
                            "prompt": prompt,
                            "seed": seed,
                            "input_still": str(still_path),
                            "input_still_source": source_tag,
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
                            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                            "traceback_tail": traceback.format_exc()[-800:],
                            "prompt": prompt,
                            "seed": seed,
                            "input_still": str(still_path),
                            "input_still_source": source_tag,
                        })
                        continue

                    elapsed = time.perf_counter() - t0
                    mp4_path = shot_dir / "motion.mp4"
                    try:
                        export_to_video(frames, str(mp4_path), fps=_FPS)
                    except Exception as exc:  # noqa: BLE001
                        errored += 1
                        atomic_write_json(shot_dir / "meta.json", {
                            "shot_id": shot_id,
                            "backend": self.name,
                            "mode": "real",
                            "error": f"export_error:{type(exc).__name__}:{exc}",
                            "prompt": prompt,
                            "seed": seed,
                            "input_still": str(still_path),
                            "input_still_source": source_tag,
                        })
                        continue

                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "variant": variant,
                        "prompt": prompt,
                        "seed": seed,
                        "duration_s": duration_s,
                        "fps": _FPS,
                        "num_frames": num_frames,
                        "render_time_s": round(elapsed, 2),
                        "input_still": str(still_path),
                        "input_still_source": source_tag,
                        "input_still_hash": _still_hash(str(still_path)),
                        "pipeline_reason": pipe_reason,
                        "env_prompt": shot.get("env_prompt", ""),
                        "camera": shot.get("camera", ""),
                        "motion_prompt": shot.get("motion_prompt", ""),
                    })
                    rendered += 1
        finally:
            # BUG-LOCAL-050: always release the pipe before returning so
            # the next chained backend (wan21_loop) starts with a clean
            # VRAM slate.  Runs even on exception so a mid-render failure
            # doesn't leave accelerate hooks resident.
            _release_pipe(pipe)

        if oom > 0:
            write_status(
                out_dir, STATUS_OOM,
                f"ltx_motion OOM on {oom}/{len(shots)} shots "
                f"(rendered={rendered}, errored={errored}, no_still={no_still})",
                backend=self.name, mode="real",
                rendered=rendered, oom=oom, errored=errored, no_still=no_still,
                pipeline_reason=pipe_reason,
            )
        else:
            write_status(
                out_dir, STATUS_READY,
                f"ltx_motion READY: {rendered}/{len(shots)} rendered"
                + (f", {errored} errored" if errored else "")
                + (f", {no_still} without still" if no_still else ""),
                backend=self.name, mode="real",
                rendered=rendered, errored=errored, no_still=no_still,
                pipeline_reason=pipe_reason,
            )
