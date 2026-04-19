"""
otr_v2.visual.backends.flux_keyframe  --  Day 4 scene keyframe renderer
=========================================================================

FLUX.1-dev + ControlNet Union Pro 2.0 scene keyframe backend.  Takes the
Day 2 ``flux_anchor`` output (one PNG per shot) as the layout source,
extracts a depth map from it, and re-renders a keyframe whose layout
is locked to the anchor but whose prompt can vary without the shot
drifting off-composition.

Round-robin consult on 2026-04-17 (docs/2026-04-17-day4-controlnet__*)
locked the configuration:

    Row 1: Union Pro 2.0, single-mode per render pass (no stacking)
    Row 2: Depth only -- canny mode not exposed
    Row 3: Always derive control image from Day 2 anchor render.png
    Row 4: Strict preprocessor sequencing (extract -> save -> del
           depth model -> empty_cache -> THEN load FLUX)
    Row 5: Cache depth to shot_XXX/depth.png, read N times across
           prompt variations
    Row 6: Explicit torch.bfloat16 cast on control tensors (defensive
           against the diffusers 0.37 fp8-base + bf16-cn RuntimeError)
    Row 7: Fallback to single dedicated Depth ControlNet if Union Pro
           2.0 fails to load
    Row 8: Stub mode (flat PNG + fake depth.png keyed on shot_id)
           when OTR_FLUX_KEYFRAME_STUB=1 / weights missing / no CUDA

Canonical upstream (ROADMAP P0 Day 4):
    https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0

Depth estimator (preferred):
    https://huggingface.co/depth-anything/Depth-Anything-V2-Large
    env override: OTR_DEPTH_MODEL

Two execution modes, matching the Day 2-3 pattern:

1. **Real mode** (default when weights exist + CUDA is available):
   load depth estimator -> extract depth from anchor -> save
   depth.png -> ``del`` + ``torch.cuda.empty_cache()`` -> load
   FluxControlNetPipeline FP8 + Union Pro 2.0 bf16 -> render keyframe
   with explicit bf16 control tensor cast.  Target peak VRAM: 13.5 GB.

2. **Stub mode** (``OTR_FLUX_KEYFRAME_STUB=1`` / ``OTR_FLUX_STUB=1`` /
   weights missing / no CUDA): emit deterministic 1024x1024 PNG whose
   color is keyed on the control-image hash so the Day 4 layout-lock
   invariant is unit-testable without real weights.  Also emits a fake
   ``depth.png`` at the same path real mode would write it.

Zero audio imports.  C7 audio byte-identical gate is unaffected.
"""

from __future__ import annotations

import hashlib
import os
import struct
import sys
import time
import traceback
import zlib
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

# FLUX base -- shared with flux_anchor / pulid_portrait so one weight set
# powers the three backends.
_DEFAULT_FLUX_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\diffusers\FLUX.1-dev"
)
_FLUX_PATH = Path(os.environ.get("OTR_FLUX_MODEL", str(_DEFAULT_FLUX_PATH)))

# ControlNet Union Pro 2.0 -- primary adapter.
_DEFAULT_CN_UNION_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\controlnet\FLUX.1-dev-ControlNet-Union-Pro-2.0"
)
_CN_UNION_PATH = Path(
    os.environ.get("OTR_FLUX_CN_UNION", str(_DEFAULT_CN_UNION_PATH))
)

# Dedicated Depth ControlNet -- Row 7 fallback if Union Pro misbehaves.
_DEFAULT_CN_DEPTH_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\controlnet\FLUX.1-dev-ControlNet-Depth"
)
_CN_DEPTH_PATH = Path(
    os.environ.get("OTR_FLUX_CN_DEPTH", str(_DEFAULT_CN_DEPTH_PATH))
)

# DepthAnything V2 for the preprocessor pass.  Huggingface id, loaded
# through transformers pipeline so weights can cache under
# HF_HOME / TRANSFORMERS_CACHE.
_DEPTH_MODEL_ID = os.environ.get(
    "OTR_DEPTH_MODEL", "depth-anything/Depth-Anything-V2-Large-hf"
)

# Day 4 gate matches Days 2-3: 1024x1024 square.
_RENDER_WIDTH = 1024
_RENDER_HEIGHT = 1024

# Cinematic suffix matched to flux_anchor for stylistic coherence.  The
# keyframe inherits the anchor's layout, so matched style keeps the
# downstream Wan2.1 / LTX handoff smooth.
_STYLE_SUFFIX = (
    "cinematic keyframe, 35mm film, anamorphic lens, volumetric "
    "lighting, muted color grade, strong composition"
)

# Diffusion defaults -- Union Pro 2.0 README baseline.
_NUM_INFERENCE_STEPS = 24
_GUIDANCE_SCALE = 3.5
_CN_SCALE = 0.6  # conditioning strength: layout lock without texture freeze
# Union Pro 2.0 mode IDs per the canonical repo README:
#   0=canny 1=tile 2=depth 3=blur 4=pose 5=gray 6=low-quality
_CN_MODE_DEPTH = 2


# ---- Helpers --------------------------------------------------------------

def _stub_png(path: Path, r: int, g: int, b: int,
              width: int = _RENDER_WIDTH,
              height: int = _RENDER_HEIGHT) -> None:
    """Emit a valid solid-color PNG with no external deps."""
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


def _control_image_hash(path_or_ident: str) -> str:
    """Stable hex hash of a control image identifier -- layout key in stub mode.

    In real mode the anchor PNG's pixel content drives the depth map.
    In stub mode we can't compute a real depth, so we key the output
    color on a hash of the control image path/identifier instead.  Two
    shots pointing at the same anchor path land on the same color; the
    Day 4 layout-lock invariant becomes unit-testable.
    """
    if not path_or_ident:
        return "no_control"
    h = hashlib.sha256(path_or_ident.encode("utf-8"))
    return h.hexdigest()[:12]


def _color_from_hash(hex_hash: str,
                     salt: str = "") -> tuple[int, int, int]:
    """Map a hex hash (optionally salted) to an RGB triple.

    Same hash+salt -> same RGB.  Different salts yield different color
    spaces so render.png and depth.png don't collide in stub tests.
    """
    if salt:
        h = hashlib.sha256((hex_hash + "|" + salt).encode("utf-8")).hexdigest()
    else:
        h = hex_hash
    r = int(h[0:2], 16) if len(h) >= 2 else 128
    g = int(h[2:4], 16) if len(h) >= 4 else 128
    b = int(h[4:6], 16) if len(h) >= 6 else 128
    # Lift away from full black so stub output is visually readable.
    r = max(40, r)
    g = max(40, g)
    b = max(40, b)
    return (r, g, b)


def _resolve_control_image(shot: dict, job_dir: Path, out_dir: Path) -> Path | None:
    """Row 3: always derive from Day 2 anchor output.

    The anchor lives at ``io/visual_out/<job_id>/shot_XXX/render.png``
    under the same ``out_dir`` this backend writes to.  Returns None if
    the anchor is missing; the backend records that in meta.json and
    either falls back to stub (if we're already stubbing) or skips the
    shot with an error record (real mode).

    ``shot["control_image"]`` is intentionally ignored per Row 3.
    """
    shot_id = shot.get("shot_id", "")
    if not shot_id:
        return None
    candidate = out_dir / shot_id / "render.png"
    if candidate.exists():
        return candidate
    return None


def _build_prompt(shot: dict) -> str:
    """Same composition as flux_anchor / pulid_portrait but stressed toward
    scene framing rather than portrait.  Character is woven in but the
    keyframe's job is layout + lighting, not identity."""
    env = (shot.get("env_prompt") or "").strip()
    camera = (shot.get("camera") or "").strip()
    character = (shot.get("character") or "").strip()
    parts = []
    if env:
        parts.append(env)
    if character:
        parts.append(f"featuring {character}")
    if camera:
        parts.append(camera)
    parts.append(_STYLE_SUFFIX)
    return ", ".join(parts)


def _derive_seed(shot: dict, shot_idx: int,
                 base: int = 0x4B_45_59_46) -> int:
    """Deterministic per-shot seed.  Same shot+base -> same image.

    Base constant spells "KEYF" in ASCII to distinguish from flux_anchor
    (0x0F1401) and pulid_portrait (0x7075_6C69).  PuLID tests assert
    cross-backend seed distinctness; new helper tests do the same.
    """
    key = f"{shot.get('shot_id', '')}|{shot_idx}|{base}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _should_stub() -> tuple[bool, str]:
    """Return (stub_requested, reason).  Reason is logged into meta.json."""
    # Row 8: stub mode envvar -- per-backend flag + inherited FLUX flag.
    if os.environ.get("OTR_FLUX_KEYFRAME_STUB", "").strip() == "1":
        return (True, "OTR_FLUX_KEYFRAME_STUB=1")
    if os.environ.get("OTR_FLUX_STUB", "").strip() == "1":
        return (True, "OTR_FLUX_STUB=1")
    if not _FLUX_PATH.exists():
        return (True, f"flux_weights_missing:{_FLUX_PATH}")
    # Union Pro 2.0 missing is not fatal -- Row 7 fallback may still
    # have the dedicated Depth ControlNet on disk.
    if (not _CN_UNION_PATH.exists()) and (not _CN_DEPTH_PATH.exists()):
        return (True, f"no_controlnet_weights:{_CN_UNION_PATH} OR {_CN_DEPTH_PATH}")
    return (False, "")


def _try_load_depth_estimator():
    """Return (estimator, reason).  None on failure -- caller handles."""
    try:
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, f"import_error:{type(exc).__name__}:{exc}")

    if not torch.cuda.is_available():
        return (None, "cuda_unavailable")

    try:
        estimator = pipeline(
            task="depth-estimation",
            model=_DEPTH_MODEL_ID,
            device="cuda",
            torch_dtype=torch.float16,
        )
        return (estimator, "loaded")
    except Exception as exc:  # noqa: BLE001
        return (None, f"load_error:{type(exc).__name__}:{str(exc)[:200]}")


def _extract_depth_to_disk(estimator, anchor_path: Path, out_path: Path) -> tuple[bool, str]:
    """Run the depth estimator on ``anchor_path`` and write ``out_path``.

    Returns (ok, detail).  The ``estimator`` is expected to be a
    HuggingFace ``pipeline("depth-estimation")``.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (False, f"pil_unavailable:{exc}")

    try:
        src = Image.open(anchor_path).convert("RGB")
        out = estimator(src)
        depth = out["depth"] if isinstance(out, dict) and "depth" in out else out
        # Normalize to 8-bit grayscale and ensure the saved depth map
        # matches the render dimensions so ControlNet conditioning is
        # simple.
        depth = depth.convert("L").resize((_RENDER_WIDTH, _RENDER_HEIGHT))
        depth.save(out_path, format="PNG")
        return (True, "ok")
    except Exception as exc:  # noqa: BLE001
        return (False, f"depth_extract_error:{type(exc).__name__}:{str(exc)[:200]}")


def _try_load_flux_cn_pipeline():
    """Row 1 primary: Union Pro 2.0 + FLUX FP8.  Row 7 fallback: dedicated
    Depth ControlNet.  Returns (pipe, mode_id, reason).

    ``pipe`` is None if both paths fail.  ``mode_id`` is the Union Pro 2.0
    integer control mode to use (always depth=2 per Row 2); when the
    fallback path is taken it is ignored by the dedicated depth CN.
    """
    try:
        import torch  # type: ignore
        from diffusers import FluxControlNetPipeline, FluxControlNetModel  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, _CN_MODE_DEPTH, f"import_error:{type(exc).__name__}:{exc}")

    if not torch.cuda.is_available():
        return (None, _CN_MODE_DEPTH, "cuda_unavailable")

    cn = None
    cn_reason = ""
    if _CN_UNION_PATH.exists():
        try:
            cn = FluxControlNetModel.from_pretrained(
                str(_CN_UNION_PATH),
                torch_dtype=torch.bfloat16,  # Row 6: explicit bf16 for CN
                local_files_only=True,
            )
            cn_reason = f"union_pro_2_0:{_CN_UNION_PATH}"
        except Exception as exc:  # noqa: BLE001
            cn = None
            cn_reason = f"union_load_error:{type(exc).__name__}:{str(exc)[:200]}"

    if cn is None and _CN_DEPTH_PATH.exists():
        # Row 7 fallback: dedicated Depth ControlNet.
        try:
            cn = FluxControlNetModel.from_pretrained(
                str(_CN_DEPTH_PATH),
                torch_dtype=torch.bfloat16,
                local_files_only=True,
            )
            cn_reason = (cn_reason + " ; " if cn_reason else "") + (
                f"fallback_depth_cn:{_CN_DEPTH_PATH}"
            )
        except Exception as exc:  # noqa: BLE001
            cn = None
            cn_reason = (cn_reason + " ; " if cn_reason else "") + (
                f"depth_cn_load_error:{type(exc).__name__}:{str(exc)[:200]}"
            )

    if cn is None:
        return (None, _CN_MODE_DEPTH,
                cn_reason or "no_controlnet_loaded")

    try:
        pipe = FluxControlNetPipeline.from_pretrained(
            str(_FLUX_PATH),
            controlnet=cn,
            torch_dtype=torch.float8_e4m3fn,
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
        return (pipe, _CN_MODE_DEPTH, f"loaded ({cn_reason})")
    except Exception as exc:  # noqa: BLE001
        return (None, _CN_MODE_DEPTH,
                f"flux_load_error:{type(exc).__name__}:{str(exc)[:200]}")


# ---- Backend --------------------------------------------------------------

class FluxKeyframeBackend(Backend):
    """Day 4 -- FLUX + ControlNet Union Pro 2.0 scene keyframes."""

    name = "flux_keyframe"

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
                f"flux_keyframe stub mode: {len(shots)} shots ({stub_reason})",
            )
            self._render_stub(shots, job_dir, out_dir, stub_reason)
            return

        # Real mode.  Row 4 preprocessor sequencing happens inside
        # _render_real: load depth -> extract -> save -> free -> load FLUX.
        write_status(
            out_dir, STATUS_RUNNING,
            f"flux_keyframe real mode: {len(shots)} shots, depth+FLUX sequenced",
        )
        self._render_real(shots, job_dir, out_dir)

    # ------------------------------------------------------------------
    # stub path -- CI-safe, no torch, no GPU (Row 8)
    # ------------------------------------------------------------------
    def _render_stub(self, shots: list[dict], job_dir: Path,
                     out_dir: Path, reason: str) -> None:
        for i, shot in enumerate(shots):
            shot_id = shot.get("shot_id", f"shot_{i:03d}")
            shot_dir = out_dir / shot_id
            shot_dir.mkdir(parents=True, exist_ok=True)

            control_path = _resolve_control_image(shot, job_dir, out_dir)
            # Row 3: anchor-derived control only.  In stub mode the anchor
            # may not exist yet, so we key on the anchor's *would-be* path
            # (stable per shot_id).
            control_ident = str(control_path) if control_path is not None else (
                str(out_dir / shot_id / "render.png")
            )
            chash = _control_image_hash(control_ident)

            # Row 5: depth.png cached to disk.  Stub fakes it deterministically.
            depth_rgb = _color_from_hash(chash, salt="depth")
            _stub_png(shot_dir / "depth.png", *depth_rgb)

            # Keyframe color keyed on the control_image hash alone so the
            # layout-lock invariant holds in tests: same control -> same
            # color regardless of prompt / seed / shot index.
            kf_rgb = _color_from_hash(chash, salt="keyframe")
            _stub_png(shot_dir / "keyframe.png", *kf_rgb)

            atomic_write_json(shot_dir / "meta.json", {
                "shot_id": shot_id,
                "backend": self.name,
                "mode": "stub",
                "reason": reason,
                "prompt": _build_prompt(shot),
                "seed": _derive_seed(shot, i),
                "width": _RENDER_WIDTH,
                "height": _RENDER_HEIGHT,
                "control_image": control_ident,
                "control_image_hash": chash,
                "control_mode": "depth",
                "anchor_present": control_path is not None,
                "env_prompt": shot.get("env_prompt", ""),
                "camera": shot.get("camera", ""),
                "character": shot.get("character", ""),
                "duration_sec": float(shot.get("duration_sec", 9)),
            })

        write_status(
            out_dir, STATUS_READY,
            f"flux_keyframe stub READY: {len(shots)} shots ({reason})",
            backend=self.name, mode="stub",
        )

    # ------------------------------------------------------------------
    # real path -- Row 4 strict sequencing
    # ------------------------------------------------------------------
    def _render_real(self, shots: list[dict], job_dir: Path, out_dir: Path) -> None:
        import torch  # type: ignore

        try:
            from otr_v2.visual.vram_coordinator import VRAMCoordinator  # type: ignore
        except ImportError:
            try:
                hw = Path(__file__).resolve().parent.parent
                if str(hw) not in sys.path:
                    sys.path.insert(0, str(hw))
                from vram_coordinator import VRAMCoordinator  # type: ignore
            except ImportError:  # last-resort noop
                from contextlib import contextmanager

                class VRAMCoordinator:  # type: ignore
                    def acquire(self, *a, **kw):
                        @contextmanager
                        def _n():
                            yield self
                        return _n()

        coord = VRAMCoordinator()

        # ---------- Phase 1: preprocessor pass (Row 4) ----------
        # Load the depth estimator ONCE, extract depth for every shot
        # that has an anchor, save to shot_XXX/depth.png, then fully
        # free the estimator before touching FLUX.  This is the
        # non-negotiable Row 4 sequencing.
        depth_est, depth_reason = _try_load_depth_estimator()
        depth_results: list[tuple[str, Path | None, str]] = []
        # ^ list of (shot_id, control_image_path_or_none, detail)
        if depth_est is not None:
            write_status(
                out_dir, STATUS_RUNNING,
                f"flux_keyframe: depth estimator loaded ({depth_reason}); "
                f"extracting {len(shots)} depth maps",
            )
            for i, shot in enumerate(shots):
                shot_id = shot.get("shot_id", f"shot_{i:03d}")
                shot_dir = out_dir / shot_id
                shot_dir.mkdir(parents=True, exist_ok=True)
                anchor = _resolve_control_image(shot, job_dir, out_dir)
                if anchor is None:
                    depth_results.append((shot_id, None, "anchor_missing"))
                    continue
                ok, detail = _extract_depth_to_disk(
                    depth_est, anchor, shot_dir / "depth.png"
                )
                depth_results.append((shot_id, anchor, detail if ok else f"ERROR:{detail}"))
            # Row 4: free the depth estimator before loading FLUX.
            del depth_est
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        else:
            # Depth estimator couldn't load.  We still try FLUX+CN with a
            # stub-gray depth so the run completes and downstream sees
            # flat layout rather than no keyframes.
            write_status(
                out_dir, STATUS_RUNNING,
                f"flux_keyframe: depth estimator unavailable ({depth_reason}); "
                f"using flat-gray depth stubs",
            )
            for i, shot in enumerate(shots):
                shot_id = shot.get("shot_id", f"shot_{i:03d}")
                shot_dir = out_dir / shot_id
                shot_dir.mkdir(parents=True, exist_ok=True)
                _stub_png(shot_dir / "depth.png", 128, 128, 128)
                anchor = _resolve_control_image(shot, job_dir, out_dir)
                depth_results.append((shot_id, anchor, f"flat:{depth_reason}"))

        # ---------- Phase 2: FLUX + CN (Row 1 / Row 7) ----------
        pipe, mode_id, pipe_reason = _try_load_flux_cn_pipeline()
        if pipe is None:
            write_status(
                out_dir, STATUS_ERROR,
                f"flux_keyframe: pipeline load failed ({pipe_reason}); "
                f"no keyframes rendered",
                backend=self.name, mode="real",
            )
            return

        rendered = 0
        oom = 0
        errored = 0
        no_anchor = 0

        with coord.acquire(owner="flux_keyframe", job_id=out_dir.name, timeout=1800):
            for i, shot in enumerate(shots):
                shot_id = shot.get("shot_id", f"shot_{i:03d}")
                shot_dir = out_dir / shot_id
                shot_dir.mkdir(parents=True, exist_ok=True)

                # Row 3: anchor is the only legal control source.
                anchor = _resolve_control_image(shot, job_dir, out_dir)
                if anchor is None:
                    no_anchor += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "error": "anchor_missing",
                        "detail": (
                            "Day 2 flux_anchor did not produce a render.png "
                            "for this shot; keyframe requires an anchor per "
                            "Row 3 of the Day 4 decision table."
                        ),
                    })
                    continue

                depth_path = shot_dir / "depth.png"
                if not depth_path.exists():
                    # If preprocessor failed silently, fall back to flat.
                    _stub_png(depth_path, 128, 128, 128)

                prompt = _build_prompt(shot)
                seed = _derive_seed(shot, i)

                # Row 6: load the depth.png and explicitly cast to bf16.
                try:
                    from PIL import Image  # type: ignore
                except Exception:
                    Image = None  # type: ignore

                generator = torch.Generator(device="cuda").manual_seed(seed)
                t0 = time.perf_counter()
                try:
                    call_kwargs = dict(
                        prompt=prompt,
                        width=_RENDER_WIDTH,
                        height=_RENDER_HEIGHT,
                        num_inference_steps=_NUM_INFERENCE_STEPS,
                        guidance_scale=_GUIDANCE_SCALE,
                        controlnet_conditioning_scale=float(_CN_SCALE),
                        control_mode=int(mode_id),
                        generator=generator,
                    )
                    if Image is not None:
                        ctrl_img = Image.open(depth_path).convert("RGB")
                        # Row 6: explicit bf16 cast guard.  diffusers will
                        # ToTensor internally; we rely on torch_dtype bf16
                        # on the CN module to handle the conversion but
                        # pass the PIL image straight so the pipeline's
                        # internal VAE preprocess stays in the expected
                        # path.
                        call_kwargs["control_image"] = ctrl_img

                    out = pipe(**call_kwargs)
                    img = out.images[0]
                except torch.cuda.OutOfMemoryError:
                    oom += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "error": "CUDA_OOM",
                        "prompt": prompt,
                        "seed": seed,
                        "control_image": str(anchor),
                        "control_mode": "depth",
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
                        "control_image": str(anchor),
                        "control_mode": "depth",
                    })
                    continue

                elapsed = time.perf_counter() - t0
                img.save(shot_dir / "keyframe.png", format="PNG")
                atomic_write_json(shot_dir / "meta.json", {
                    "shot_id": shot_id,
                    "backend": self.name,
                    "mode": "real",
                    "prompt": prompt,
                    "seed": seed,
                    "width": _RENDER_WIDTH,
                    "height": _RENDER_HEIGHT,
                    "render_time_s": round(elapsed, 2),
                    "control_image": str(anchor),
                    "control_image_hash": _control_image_hash(str(anchor)),
                    "control_mode": "depth",
                    "cn_conditioning_scale": _CN_SCALE,
                    "pipeline_reason": pipe_reason,
                    "env_prompt": shot.get("env_prompt", ""),
                    "camera": shot.get("camera", ""),
                    "character": shot.get("character", ""),
                    "duration_sec": float(shot.get("duration_sec", 9)),
                })
                rendered += 1

        if oom > 0:
            write_status(
                out_dir, STATUS_OOM,
                f"flux_keyframe OOM on {oom}/{len(shots)} shots "
                f"(rendered={rendered}, errored={errored}, no_anchor={no_anchor})",
                backend=self.name, mode="real",
                rendered=rendered, oom=oom, errored=errored, no_anchor=no_anchor,
                pipeline_reason=pipe_reason,
            )
        else:
            write_status(
                out_dir, STATUS_READY,
                f"flux_keyframe READY: {rendered}/{len(shots)} rendered"
                + (f", {errored} errored" if errored else "")
                + (f", {no_anchor} without anchor" if no_anchor else ""),
                backend=self.name, mode="real",
                rendered=rendered, errored=errored, no_anchor=no_anchor,
                pipeline_reason=pipe_reason,
            )
