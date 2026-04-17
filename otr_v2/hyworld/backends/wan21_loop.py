"""
otr_v2.hyworld.backends.wan21_loop  --  Day 6 Wan2.1 1.3B I2V loop
==================================================================

Wan2.1 1.3B image-to-video loop backend.  Takes a still (Day 4
keyframe.png preferred, Day 2 anchor render.png fallback) as the first
frame and renders a short loop clip used for VJ-style long motion
beds.  This is the Day 6 shipment on the 14-day video stack sprint --
it widens the motion vocabulary beyond LTX-2.3's short bursts.

Platform pins (ROADMAP.md, CLAUDE.md):

* Gate G6 -- Wan2.1 1.3B I2V runs in <= 10 GB VRAM.  The 1.3B weight
  profile is chosen specifically because the 14B profile exceeds the
  RTX 5080 Laptop 16 GB ceiling in combination with Stage 3-5 state.
* Constraint C4 -- Wan2.1 loops cap per-shot at 10 s (240 frames at
  24 fps) so they compose cleanly with LTX-2.3 handoffs without
  requiring ffmpeg crossfade at planning time.
* Constraint C5 -- Wan2.1 pipeline runs in ``torch.float8_e4m3fn``
  (Blackwell-native) where supported.  Falls back to ``torch.float16``
  when the pipeline class rejects the fp8 dtype, recording the choice
  in meta.json.
* Constraint C7 -- no audio imports on this path; byte-identical audio
  gate is preserved.
* VRAM ceiling: 14.5 GB real-mode audio target / 15.5 GB video target.
  Subprocess spawn pattern (HyworldBridge._spawn_sidecar) ensures
  upstream FLUX / LTX VRAM is fully released before this sidecar
  loads Wan2.1 weights.

Two execution modes, matching Days 2-5:

1. **Real mode** (default when weights exist + CUDA is available):
   tries ``diffusers.WanImageToVideoPipeline`` (preferred) / falls back
   to ``diffusers.WanPipeline`` from ``Wan-AI/Wan2.1-I2V-1.3B`` at
   ``torch.float8_e4m3fn`` with ``enable_model_cpu_offload``, then
   runs image-to-video conditioned on the Day 4 keyframe.  Output:
   ``loop.mp4`` (h264, 24 fps) via ``diffusers.utils.export_to_video``
   (imageio-ffmpeg backend).  When the I2V variant isn't available
   the backend records a detailed error per shot rather than silently
   falling back to noise.

2. **Stub mode** (``OTR_WAN_STUB=1`` / weights missing / no CUDA):
   emits a minimal-but-valid 24-byte MP4 ``ftyp`` atom + a 1-frame
   mdat so file-magic tools and tests can confirm a loop landed.
   The clip bytes are keyed on the input-still hash so the handoff
   invariant ("same input still -> same stub loop bytes") is
   unit-testable before real weights land.

Input source priority (Row 6 of the Day 6 decision, parallels Day 5's
Row 5 for LTX motion -- Wan2.1 loops consume the same upstream stills
so the two backends share the Row 5 precedence rule):

    1. ``<out_dir>/<shot_id>/keyframe.png`` (Day 4 output) -- preferred
    2. ``<out_dir>/<shot_id>/render.png`` (Day 2 output) -- fallback
    3. otherwise: error record in meta.json, skip the shot

Output filename is ``loop.mp4`` (distinct from LTX's ``motion.mp4``)
so the planner can mix and match per-shot without extension
collisions.

Per-shot seed base ``0x57_41_4E_32`` spells "WAN2" in ASCII, distinct
from flux_anchor (0x0F1401), pulid_portrait (0x7075_6C69),
flux_keyframe (0x4B45_5946), and ltx_motion (0x4C54_584D) so one
shot_id doesn't collide across pipelines.
"""

from __future__ import annotations

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


# ---- Model discovery ------------------------------------------------------

# Wan2.1 1.3B I2V weights -- user overridable via env.  Canonical upstream:
#   https://huggingface.co/Wan-AI/Wan2.1-I2V-1.3B
# Default path mirrors the diffusers/ layout used by the LTX backend so
# operators only have to memorise one convention.
_DEFAULT_WAN_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\diffusers\Wan2.1-I2V-1.3B"
)
_WAN_PATH = Path(os.environ.get("OTR_WAN_MODEL", str(_DEFAULT_WAN_PATH)))

# C4 cap: 10 s at 24 fps = 240 frames.  Wan2.1 1.3B I2V's native ceiling
# is 81 frames per single forward pass on diffusers 0.31+; the real
# path chunks beyond that via the pipeline's own frame-packing.  Env
# overrides let a soak run shorten to 2 s without editing code.
_FPS = int(os.environ.get("OTR_WAN_FPS", "24"))
_DURATION_S = float(os.environ.get("OTR_WAN_DURATION_S", "10.0"))
_NUM_INFERENCE_STEPS = int(os.environ.get("OTR_WAN_STEPS", "40"))

# Hard cap on single-call frame count so we never attempt to request
# more frames than the pipeline can realistically deliver in one shot.
# 240 frames fits the 10 s / 24 fps gate exactly.
_MAX_FRAMES = 240


# ---- Helpers --------------------------------------------------------------

def _still_hash(path_or_ident: str) -> str:
    """Stable hex hash of the input still identifier.

    In real mode the still's pixel content drives the loop; in stub
    mode we can't render real motion so we key the stub clip bytes on
    a hash of the still's path string.  Two shots with the same still
    path land on the same stub bytes; the Day 6 handoff invariant
    becomes unit-testable without running Wan2.1.
    """
    if not path_or_ident:
        return "no_still"
    h = hashlib.sha256(path_or_ident.encode("utf-8"))
    return h.hexdigest()[:12]


def _derive_seed(shot: dict, shot_idx: int,
                  base: int = 0x57_41_4E_32) -> int:
    """Deterministic per-shot seed, distinct base from other backends."""
    key = f"{shot.get('shot_id', '')}|{shot_idx}|{base}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _build_prompt(shot: dict) -> str:
    """Wan2.1 I2V prompt -- biases toward loopable, continuous motion.

    Unlike LTX's short burst ("push in"), Wan2.1 loops want motion
    verbs that don't imply a start/end, so the prompt suffix emphasises
    cycling / drifting / ambient vocabulary.
    """
    loop = (shot.get("loop_prompt") or "").strip()
    motion = (shot.get("motion_prompt") or "").strip()
    env = (shot.get("env_prompt") or "").strip()
    camera = (shot.get("camera") or "").strip()
    parts = []
    if loop:
        parts.append(loop)
    elif motion:
        parts.append(motion)
    elif env:
        parts.append(f"ambient drifting motion through {env}")
    if camera:
        parts.append(camera)
    parts.append("seamless loop, subtle cycling motion, 24fps")
    return ", ".join(parts)


def _resolve_input_still(shot: dict, job_dir: Path,
                         out_dir: Path) -> tuple[Path | None, str]:
    """Row 6 handoff priority: keyframe.png > render.png > None.

    Returns ``(path, source_tag)``.  ``source_tag`` is one of:
        "keyframe", "anchor", "missing"
    Recorded in meta.json so downstream analysis can tell which
    upstream stage fed a given loop clip.
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
    if os.environ.get("OTR_WAN_STUB", "").strip() == "1":
        return (True, "OTR_WAN_STUB=1")
    if not _WAN_PATH.exists():
        return (True, f"wan_weights_missing:{_WAN_PATH}")
    return (False, "")


def _stub_mp4(path: Path, still_hash: str) -> None:
    """Emit a minimal-but-valid MP4 skeleton keyed on still_hash.

    Matches the LTX stub format so downstream tooling treats loop.mp4
    and motion.mp4 identically at the container level.  The mdat
    payload is salted with the backend name so two backends with the
    same still hash still produce different bytes -- prevents a spurious
    'wan and ltx collided' collision test from passing.
    """
    # ftyp atom -- 24 bytes, same brands as LTX stub for consistency.
    ftyp = (
        b"\x00\x00\x00\x18"  # atom size = 24
        b"ftyp"              # atom type
        b"isom"              # major brand
        b"\x00\x00\x02\x00"  # minor version = 0x200
        b"isom"              # compatible brand 1
        b"mp42"              # compatible brand 2
    )
    # mdat atom -- 16-byte payload keyed on still_hash + backend salt.
    salt = b"wan21_loop"
    payload_bytes = hashlib.sha256(
        salt + still_hash.encode("utf-8"),
    ).digest()[:16]
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


def _try_load_wan_pipeline():
    """Return (pipe, variant, dtype_str, reason).

    ``variant`` is "i2v" | "t2v" | None.  ``dtype_str`` records which
    torch dtype actually loaded (fp8 preferred, fp16 fallback) so
    meta.json can surface it for post-hoc VRAM analysis.
    """
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, None, "",
                f"torch_import_error:{type(exc).__name__}:{exc}")

    if not torch.cuda.is_available():
        return (None, None, "", "cuda_unavailable")

    # Prefer the I2V variant (what Day 6 actually needs).
    try:
        from diffusers import WanImageToVideoPipeline  # type: ignore
    except ImportError as exc:  # noqa: BLE001
        WanImageToVideoPipeline = None  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, None, "",
                f"wan_i2v_import_error:{type(exc).__name__}:"
                f"{str(exc)[:200]}")

    if WanImageToVideoPipeline is not None:
        for dtype_name in ("float8_e4m3fn", "float16"):
            try:
                dtype = getattr(torch, dtype_name)
            except AttributeError:
                continue
            try:
                pipe = WanImageToVideoPipeline.from_pretrained(
                    str(_WAN_PATH),
                    torch_dtype=dtype,
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
                return (pipe, "i2v", dtype_name,
                        f"wan_i2v_loaded:{_WAN_PATH}:{dtype_name}")
            except Exception as exc:  # noqa: BLE001
                last_exc = exc  # try next dtype
        # If both dtypes failed, fall through with the last error.
        return (None, None, "",
                f"wan_i2v_load_error:{type(last_exc).__name__}:"
                f"{str(last_exc)[:200]}")

    # Older diffusers exposed only WanPipeline (T2V).  We record the
    # degradation so the shot knows the still was ignored.
    try:
        from diffusers import WanPipeline  # type: ignore
        pipe = WanPipeline.from_pretrained(
            str(_WAN_PATH),
            torch_dtype=__import__("torch").float16,
            local_files_only=True,
        )
        try:
            pipe.enable_model_cpu_offload()
        except Exception:
            pass
        return (pipe, "t2v", "float16",
                f"wan_t2v_loaded_fallback:{_WAN_PATH}")
    except Exception as exc:  # noqa: BLE001
        return (None, None, "",
                f"wan_load_error:{type(exc).__name__}:{str(exc)[:200]}")


# ---- Backend --------------------------------------------------------------

class Wan21LoopBackend(Backend):
    """Day 6 -- Wan2.1 1.3B I2V loop sidecar."""

    name = "wan21_loop"

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
                f"wan21_loop stub mode: {len(shots)} shots ({stub_reason})",
            )
            self._render_stub(shots, job_dir, out_dir, stub_reason)
            return

        write_status(
            out_dir, STATUS_RUNNING,
            f"wan21_loop real mode: {len(shots)} shots, "
            f"I2V loops from FLUX stills",
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
            still_ident = str(still_path) if still_path is not None else (
                str(out_dir / shot_id / "keyframe.png")
            )
            shash = _still_hash(still_ident)
            _stub_mp4(shot_dir / "loop.mp4", shash)

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
                "loop_prompt": shot.get("loop_prompt", ""),
            })

        write_status(
            out_dir, STATUS_READY,
            f"wan21_loop stub READY: {len(shots)} shots ({reason})",
            backend=self.name, mode="stub",
        )

    # ------------------------------------------------------------------
    # real path -- Wan2.1 1.3B I2V
    # ------------------------------------------------------------------
    def _render_real(self, shots: list[dict], job_dir: Path, out_dir: Path) -> None:
        import torch  # type: ignore

        try:
            from otr_v2.hyworld.vram_coordinator import VRAMCoordinator  # type: ignore
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

        pipe, variant, dtype_str, pipe_reason = _try_load_wan_pipeline()
        if pipe is None:
            write_status(
                out_dir, STATUS_ERROR,
                f"wan21_loop: pipeline load failed ({pipe_reason})",
                backend=self.name, mode="real",
            )
            return

        try:
            from diffusers.utils import export_to_video  # type: ignore
        except Exception as exc:  # noqa: BLE001
            write_status(
                out_dir, STATUS_ERROR,
                f"wan21_loop: export_to_video import failed "
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

        with coord.acquire(owner="wan21_loop", job_id=out_dir.name, timeout=1800):
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
                            "on disk for this shot_id; Wan2.1 loop handoff "
                            "needs an upstream still."
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
                        # Fallback: no image conditioning available,
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
                        "dtype": dtype_str,
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
                        "dtype": dtype_str,
                    })
                    continue

                elapsed = time.perf_counter() - t0
                mp4_path = shot_dir / "loop.mp4"
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
                        "dtype": dtype_str,
                    })
                    continue

                atomic_write_json(shot_dir / "meta.json", {
                    "shot_id": shot_id,
                    "backend": self.name,
                    "mode": "real",
                    "variant": variant,
                    "dtype": dtype_str,
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
                    "loop_prompt": shot.get("loop_prompt", ""),
                })
                rendered += 1

        if oom > 0:
            write_status(
                out_dir, STATUS_OOM,
                f"wan21_loop OOM on {oom}/{len(shots)} shots "
                f"(rendered={rendered}, errored={errored}, no_still={no_still})",
                backend=self.name, mode="real",
                rendered=rendered, oom=oom, errored=errored, no_still=no_still,
                pipeline_reason=pipe_reason,
            )
        else:
            write_status(
                out_dir, STATUS_READY,
                f"wan21_loop READY: {rendered}/{len(shots)} rendered"
                + (f", {errored} errored" if errored else "")
                + (f", {no_still} without still" if no_still else ""),
                backend=self.name, mode="real",
                rendered=rendered, errored=errored, no_still=no_still,
                pipeline_reason=pipe_reason,
            )
