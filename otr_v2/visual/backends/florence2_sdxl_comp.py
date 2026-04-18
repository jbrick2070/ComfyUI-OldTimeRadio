"""
otr_v2.visual.backends.florence2_sdxl_comp  --  Day 7 Florence-2 + SDXL inpaint
================================================================================

Text-prompt mask via Florence-2 -> SDXL inpaint insert.  Day 7 shipment
on the 14-day video stack sprint.  Where Days 4-6 produced stills
(keyframe.png) and motion (motion.mp4 / loop.mp4), Day 7 produces a
*composite* still: takes the Day 4 keyframe, segments a named region
("cockpit window", "billboard", etc.) with Florence-2's referring
expression head, and uses SDXL inpaint to paint a different asset
("CRT overlay", "noir poster", etc.) into that region.

Planner use case is the SIGNAL LOST cockpit prototype:
    Day 4 keyframe  = wide cockpit shot
    Day 7 composite = same cockpit with the window replaced by a CRT
                      overlay showing the pirate signal

Platform pins (ROADMAP.md, CLAUDE.md):

* Gate G7 -- Florence-2 + SDXL inpaint runs in <= 8 GB VRAM.  Florence-2
  is small (~0.8B large variant).  SDXL inpaint at fp16 with
  ``enable_model_cpu_offload`` fits under the ceiling; operators who
  need to co-host with Stage 3-5 state should set fp8 via the
  ``OTR_SDXL_INPAINT_DTYPE`` env (documented below).
* Constraint C4 is non-applicable (no video output); Day 7 writes a
  single composite PNG per shot.
* Constraint C5 -- SDXL inpaint defaults to ``torch.float16`` on
  Blackwell (SDXL has less fp8 operator coverage than FLUX); the
  backend will honour an operator-provided fp8 override but records
  the choice in meta.json for post-hoc VRAM analysis.
* Constraint C7 -- no audio imports on this path; byte-identical audio
  gate is preserved.
* VRAM ceiling: 14.5 GB real-mode audio target / 15.5 GB video target.
  Subprocess spawn pattern ensures upstream FLUX / LTX / Wan VRAM is
  fully released before this sidecar loads Florence-2 and SDXL.

Two execution modes, matching Days 2-6:

1. **Real mode** (default when both weight trees exist + CUDA is
   available):
      a) Load Florence-2 via ``transformers.AutoModelForCausalLM`` +
         processor, run task ``<REFERRING_EXPRESSION_SEGMENTATION>``
         or ``<OPEN_VOCABULARY_DETECTION>`` on the mask prompt,
         build a binary PIL mask.
      b) Release Florence-2 (del + empty_cache) before loading the
         inpaint pipeline -- the Day 4 CN handoff discipline.
      c) Load ``diffusers.StableDiffusionXLInpaintPipeline`` at fp16
         (or env-overridden fp8), ``enable_model_cpu_offload``,
         VRAMCoordinator gate, run inpaint with the insert_prompt.
      d) Save composite.png + mask.png per shot.

2. **Stub mode** (``OTR_FLORENCE_STUB=1`` / either weight tree
   missing / no CUDA): emit a composite PNG whose color is keyed on
   the SHA256 of (input_still_identity + mask_prompt + insert_prompt),
   plus a mask.png keyed on mask_prompt alone.  Three-way determinism
   makes the (still, mask, insert) tuple unit-testable as an
   end-to-end handoff invariant.

Input source priority (inherits Day 5 / Day 6 Row 5-6 rule):

    1. ``<out_dir>/<shot_id>/keyframe.png`` (Day 4 output) -- preferred
    2. ``<out_dir>/<shot_id>/render.png`` (Day 2 output) -- fallback
    3. otherwise: error record in meta.json, skip the shot

Output filenames:
    * ``composite.png`` -- the final still (distinct from Day 4's
      ``keyframe.png`` so the renderer can choose which to mux)
    * ``mask.png`` -- the binary mask used for inpaint, cached for
      audit and re-use by downstream VHS filters

Per-shot seed base ``0x46_32_53_44`` spells "F2SD" in ASCII, distinct
from flux_anchor (0x0F1401), pulid_portrait (0x7075_6C69),
flux_keyframe (0x4B45_5946), ltx_motion (0x4C54_584D), and
wan21_loop (0x5741_4E32) so one shot_id doesn't collide across the
six pipelines.
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

_DEFAULT_FLORENCE_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\florence2\Florence-2-large"
)
_FLORENCE_PATH = Path(
    os.environ.get("OTR_FLORENCE_MODEL", str(_DEFAULT_FLORENCE_PATH)),
)

_DEFAULT_SDXL_INPAINT_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\diffusers"
    r"\stable-diffusion-xl-1.0-inpainting-0.1"
)
_SDXL_INPAINT_PATH = Path(
    os.environ.get(
        "OTR_SDXL_INPAINT_MODEL", str(_DEFAULT_SDXL_INPAINT_PATH),
    ),
)

# Inpaint resolution -- SDXL canonical 1024 so it matches FLUX anchors.
_RENDER_WIDTH = int(os.environ.get("OTR_SDXL_INPAINT_WIDTH", "1024"))
_RENDER_HEIGHT = int(os.environ.get("OTR_SDXL_INPAINT_HEIGHT", "1024"))
_NUM_INFERENCE_STEPS = int(os.environ.get("OTR_SDXL_INPAINT_STEPS", "30"))

# dtype override -- default fp16 (matches SDXL canonical), fp8 opt-in.
_SDXL_DTYPE_NAME = os.environ.get(
    "OTR_SDXL_INPAINT_DTYPE", "float16",
).strip().lower()

# Florence-2 task token -- referring expression segmentation produces
# a polygon mask for a free-text phrase like "cockpit window".
_FLORENCE_TASK = "<REFERRING_EXPRESSION_SEGMENTATION>"


# ---- Helpers --------------------------------------------------------------

def _stub_png(path: Path, r: int, g: int, b: int,
              width: int = _RENDER_WIDTH,
              height: int = _RENDER_HEIGHT) -> None:
    """Emit a valid solid-color PNG with no external deps.

    Matches flux_keyframe._stub_png byte-for-byte so the two Day 4 / 7
    stub renders stay interoperable with the same PIL-less readers.
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


def _stub_mask_png(path: Path, r: int,
                   width: int = _RENDER_WIDTH,
                   height: int = _RENDER_HEIGHT) -> None:
    """Emit a single-channel (grayscale) PNG so mask.png is a true 8-bit
    mask regardless of who reads it downstream."""
    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = zlib.crc32(c) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)

    # IHDR bit-depth=8, colour type=0 (grayscale).
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    row = bytes([0] + [r] * width)
    raw = row * height
    idat = zlib.compress(raw, 1)

    png = b"\x89PNG\r\n\x1a\n"
    png += _chunk(b"IHDR", ihdr)
    png += _chunk(b"IDAT", idat)
    png += _chunk(b"IEND", b"")
    path.write_bytes(png)


def _composite_hash(still_ident: str, mask_prompt: str,
                    insert_prompt: str) -> str:
    """Three-way key for Day 7 composite determinism.

    Two shots that agree on (input still, mask prompt, insert prompt)
    must land on the same stub bytes; change any of the three and the
    composite must shift.  Tested in both directions.
    """
    key = (
        f"{still_ident}|mask={mask_prompt}|insert={insert_prompt}"
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def _mask_hash(mask_prompt: str) -> str:
    """Mask-only key.  mask.png is a function of mask_prompt alone so
    the same mask can be re-used across different insert_prompt runs."""
    if not mask_prompt:
        return "no_mask"
    return hashlib.sha256(
        ("mask:" + mask_prompt).encode("utf-8"),
    ).hexdigest()[:12]


def _derive_seed(shot: dict, shot_idx: int,
                  base: int = 0x46_32_53_44) -> int:
    """Deterministic per-shot seed, distinct base from other backends."""
    key = f"{shot.get('shot_id', '')}|{shot_idx}|{base}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _build_insert_prompt(shot: dict) -> str:
    """SDXL inpaint prompt -- what to paint into the masked region.

    Falls back to ``env_prompt`` so shots that don't carry a dedicated
    insert prompt still produce something coherent rather than noise.
    """
    insert = (shot.get("insert_prompt") or "").strip()
    if insert:
        parts = [insert]
    else:
        env = (shot.get("env_prompt") or "").strip()
        parts = [env] if env else ["cinematic interior"]
    parts.append(
        "cinematic lighting, film grain, 35mm, photorealistic",
    )
    return ", ".join(parts)


def _build_mask_prompt(shot: dict) -> str:
    """Florence-2 referring-expression prompt -- what to segment.

    Raw pass-through.  Florence-2 handles phrases like
    ``"the large cockpit window"`` without needing our suffix.
    """
    return (shot.get("mask_prompt") or "").strip()


def _resolve_input_still(shot: dict, job_dir: Path,
                         out_dir: Path) -> tuple[Path | None, str]:
    """Row 7 handoff priority: keyframe.png > render.png > None.

    Returns ``(path, source_tag)`` with ``source_tag`` in
    ``"keyframe" | "anchor" | "missing"`` -- recorded in meta.json.
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
    """Return (stub_requested, reason).  Reason is logged into meta.json.

    Either Florence-2 or the SDXL inpaint tree missing triggers stub
    mode -- Day 7 is a two-model handoff and we don't silently degrade
    to half a pipeline.
    """
    if os.environ.get("OTR_FLORENCE_STUB", "").strip() == "1":
        return (True, "OTR_FLORENCE_STUB=1")
    if not _FLORENCE_PATH.exists():
        return (True, f"florence2_weights_missing:{_FLORENCE_PATH}")
    if not _SDXL_INPAINT_PATH.exists():
        return (True, f"sdxl_inpaint_weights_missing:{_SDXL_INPAINT_PATH}")
    return (False, "")


def _color_from_hash(h: str) -> tuple[int, int, int]:
    """Map a 12-char hex hash to an (r, g, b) triple for stub PNGs."""
    if len(h) < 6:
        h = (h + "000000")[:6]
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return (r, g, b)


def _mask_value_from_hash(h: str) -> int:
    """Map a hash to a single grayscale value 1-254 for stub masks.

    Avoids 0 (all-black = no mask) and 255 (all-white = whole image)
    so tests can verify the mask PNG actually carries information.
    """
    if len(h) < 2:
        h = (h + "00")[:2]
    v = int(h[0:2], 16)
    # Clamp into (0, 255) exclusive -- never degenerate.
    return max(1, min(254, v))


def _try_load_florence2():
    """Return (model, processor, reason).  Lazy transformers import."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, None,
                f"torch_import_error:{type(exc).__name__}:{exc}")
    if not torch.cuda.is_available():
        return (None, None, "cuda_unavailable")
    try:
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore
        model = AutoModelForCausalLM.from_pretrained(
            str(_FLORENCE_PATH),
            torch_dtype=torch.float16,
            trust_remote_code=True,
            local_files_only=True,
        )
        processor = AutoProcessor.from_pretrained(
            str(_FLORENCE_PATH),
            trust_remote_code=True,
            local_files_only=True,
        )
        model.to("cuda")
        return (model, processor, f"florence2_loaded:{_FLORENCE_PATH}")
    except Exception as exc:  # noqa: BLE001
        return (None, None,
                f"florence2_load_error:{type(exc).__name__}:"
                f"{str(exc)[:200]}")


def _try_load_sdxl_inpaint():
    """Return (pipe, dtype_str, reason).  SDXL inpaint at fp16 by default."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, "",
                f"torch_import_error:{type(exc).__name__}:{exc}")
    if not torch.cuda.is_available():
        return (None, "", "cuda_unavailable")

    # dtype resolution: operator override first, then canonical fp16.
    dtype_candidates = []
    if _SDXL_DTYPE_NAME:
        dtype_candidates.append(_SDXL_DTYPE_NAME)
    for fallback in ("float16", "float8_e4m3fn"):
        if fallback not in dtype_candidates:
            dtype_candidates.append(fallback)

    try:
        from diffusers import StableDiffusionXLInpaintPipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, "",
                f"sdxl_inpaint_import_error:{type(exc).__name__}:"
                f"{str(exc)[:200]}")

    last_exc: Exception | None = None
    for dtype_name in dtype_candidates:
        dtype = getattr(torch, dtype_name, None)
        if dtype is None:
            continue
        try:
            pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
                str(_SDXL_INPAINT_PATH),
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
            return (pipe, dtype_name,
                    f"sdxl_inpaint_loaded:{_SDXL_INPAINT_PATH}:{dtype_name}")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            continue

    reason = (
        f"sdxl_inpaint_load_error:"
        f"{type(last_exc).__name__ if last_exc else 'NoDtypeAvailable'}:"
        f"{str(last_exc)[:200] if last_exc else 'no torch dtype accepted'}"
    )
    return (None, "", reason)


# ---- Backend --------------------------------------------------------------

class Florence2SdxlCompBackend(Backend):
    """Day 7 -- Florence-2 (text-prompt mask) + SDXL inpaint composite."""

    name = "florence2_sdxl_comp"

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
                f"florence2_sdxl_comp stub mode: {len(shots)} shots "
                f"({stub_reason})",
            )
            self._render_stub(shots, job_dir, out_dir, stub_reason)
            return

        write_status(
            out_dir, STATUS_RUNNING,
            f"florence2_sdxl_comp real mode: {len(shots)} shots, "
            f"Florence-2 mask -> SDXL inpaint",
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
            mask_prompt = _build_mask_prompt(shot)
            insert_prompt_full = _build_insert_prompt(shot)
            # Use the raw insert_prompt (pre-suffix) as the stub key so
            # tests can control the hash by controlling the shot field
            # directly.
            raw_insert = (shot.get("insert_prompt") or "").strip()

            chash = _composite_hash(still_ident, mask_prompt, raw_insert)
            mhash = _mask_hash(mask_prompt)

            _stub_png(shot_dir / "composite.png", *_color_from_hash(chash))
            _stub_mask_png(shot_dir / "mask.png", _mask_value_from_hash(mhash))

            atomic_write_json(shot_dir / "meta.json", {
                "shot_id": shot_id,
                "backend": self.name,
                "mode": "stub",
                "reason": reason,
                "insert_prompt": insert_prompt_full,
                "mask_prompt": mask_prompt,
                "seed": _derive_seed(shot, i),
                "input_still": still_ident,
                "input_still_source": source_tag,
                "composite_hash": chash,
                "mask_hash": mhash,
                "still_present": still_path is not None,
                "width": _RENDER_WIDTH,
                "height": _RENDER_HEIGHT,
                "florence_task": _FLORENCE_TASK,
                "env_prompt": shot.get("env_prompt", ""),
                "camera": shot.get("camera", ""),
            })

        write_status(
            out_dir, STATUS_READY,
            f"florence2_sdxl_comp stub READY: {len(shots)} shots ({reason})",
            backend=self.name, mode="stub",
        )

    # ------------------------------------------------------------------
    # real path -- Florence-2 -> SDXL inpaint
    # ------------------------------------------------------------------
    def _render_real(self, shots: list[dict], job_dir: Path,
                      out_dir: Path) -> None:
        import torch  # type: ignore

        try:
            from otr_v2.visual.vram_coordinator import VRAMCoordinator  # type: ignore
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

        try:
            from PIL import Image  # type: ignore
        except Exception as exc:  # noqa: BLE001
            write_status(
                out_dir, STATUS_ERROR,
                f"florence2_sdxl_comp: PIL import failed "
                f"({type(exc).__name__}:{exc})",
                backend=self.name, mode="real",
            )
            return

        rendered = 0
        oom = 0
        errored = 0
        no_still = 0

        # Phase A: Florence-2 pass (all shots, build masks).  We
        # persist the masks to disk per-shot and release Florence-2
        # before loading SDXL inpaint -- Day 4 handoff discipline.
        flo_model, flo_proc, flo_reason = _try_load_florence2()
        if flo_model is None:
            write_status(
                out_dir, STATUS_ERROR,
                f"florence2_sdxl_comp: Florence-2 load failed "
                f"({flo_reason})",
                backend=self.name, mode="real",
            )
            return

        mask_records: list[dict] = []
        with coord.acquire(owner="florence2", job_id=out_dir.name, timeout=1800):
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
                        "phase": "florence2",
                        "error": "still_missing",
                        "detail": (
                            "No Day 2 render.png or Day 4 keyframe.png "
                            "on disk for this shot_id; Day 7 composite "
                            "needs an upstream still."
                        ),
                    })
                    continue

                mask_prompt = _build_mask_prompt(shot)
                if not mask_prompt:
                    errored += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "phase": "florence2",
                        "error": "mask_prompt_missing",
                        "detail": (
                            "Day 7 requires shot['mask_prompt'] to name "
                            "a region for Florence-2 to segment."
                        ),
                    })
                    continue

                try:
                    image = Image.open(still_path).convert("RGB")
                    inputs = flo_proc(
                        text=_FLORENCE_TASK + mask_prompt,
                        images=image,
                        return_tensors="pt",
                    ).to("cuda")
                    with torch.inference_mode():
                        generated_ids = flo_model.generate(
                            input_ids=inputs["input_ids"],
                            pixel_values=inputs["pixel_values"],
                            max_new_tokens=1024,
                            num_beams=3,
                            do_sample=False,
                        )
                    generated_text = flo_proc.batch_decode(
                        generated_ids, skip_special_tokens=False,
                    )[0]
                    parsed = flo_proc.post_process_generation(
                        generated_text,
                        task=_FLORENCE_TASK,
                        image_size=(image.width, image.height),
                    )
                    mask_img = self._rasterize_mask(
                        parsed, image.size, _FLORENCE_TASK,
                    )
                    mask_img.save(shot_dir / "mask.png", format="PNG")
                    mask_records.append({
                        "shot_id": shot_id,
                        "shot_idx": i,
                        "still_path": str(still_path),
                        "source_tag": source_tag,
                        "mask_prompt": mask_prompt,
                    })
                except torch.cuda.OutOfMemoryError:
                    oom += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "phase": "florence2",
                        "error": "CUDA_OOM",
                        "mask_prompt": mask_prompt,
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
                        "phase": "florence2",
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "traceback_tail": traceback.format_exc()[-800:],
                        "mask_prompt": mask_prompt,
                    })
                    continue

        # Release Florence-2 before loading SDXL -- Day 4 discipline.
        try:
            flo_model.to("cpu")  # type: ignore
        except Exception:
            pass
        del flo_model
        del flo_proc
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

        # Phase B: SDXL inpaint pass over shots that produced a mask.
        if not mask_records:
            write_status(
                out_dir, STATUS_READY,
                f"florence2_sdxl_comp: no maskable shots "
                f"(rendered=0, errored={errored}, oom={oom}, "
                f"no_still={no_still})",
                backend=self.name, mode="real",
                rendered=0, errored=errored, oom=oom, no_still=no_still,
            )
            return

        pipe, dtype_str, pipe_reason = _try_load_sdxl_inpaint()
        if pipe is None:
            write_status(
                out_dir, STATUS_ERROR,
                f"florence2_sdxl_comp: SDXL inpaint load failed "
                f"({pipe_reason})",
                backend=self.name, mode="real",
            )
            return

        shot_by_idx = {i: s for i, s in enumerate(shots)}

        with coord.acquire(owner="sdxl_inpaint", job_id=out_dir.name, timeout=1800):
            for rec in mask_records:
                shot_id = rec["shot_id"]
                shot = shot_by_idx[rec["shot_idx"]]
                shot_dir = out_dir / shot_id
                still_path = Path(rec["still_path"])
                mask_prompt = rec["mask_prompt"]

                insert_prompt = _build_insert_prompt(shot)
                raw_insert = (shot.get("insert_prompt") or "").strip()
                seed = _derive_seed(shot, rec["shot_idx"])
                generator = torch.Generator(device="cuda").manual_seed(seed)

                t0 = time.perf_counter()
                try:
                    base_img = Image.open(still_path).convert("RGB")
                    mask_img = Image.open(shot_dir / "mask.png").convert("L")
                    if base_img.size != (_RENDER_WIDTH, _RENDER_HEIGHT):
                        base_img = base_img.resize(
                            (_RENDER_WIDTH, _RENDER_HEIGHT),
                            Image.LANCZOS,
                        )
                    if mask_img.size != (_RENDER_WIDTH, _RENDER_HEIGHT):
                        mask_img = mask_img.resize(
                            (_RENDER_WIDTH, _RENDER_HEIGHT),
                            Image.LANCZOS,
                        )
                    out = pipe(
                        prompt=insert_prompt,
                        image=base_img,
                        mask_image=mask_img,
                        num_inference_steps=_NUM_INFERENCE_STEPS,
                        width=_RENDER_WIDTH,
                        height=_RENDER_HEIGHT,
                        generator=generator,
                    )
                    composite = out.images[0]
                except torch.cuda.OutOfMemoryError:
                    oom += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "phase": "sdxl_inpaint",
                        "error": "CUDA_OOM",
                        "insert_prompt": insert_prompt,
                        "mask_prompt": mask_prompt,
                        "seed": seed,
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
                        "phase": "sdxl_inpaint",
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "traceback_tail": traceback.format_exc()[-800:],
                        "insert_prompt": insert_prompt,
                        "mask_prompt": mask_prompt,
                        "seed": seed,
                        "dtype": dtype_str,
                    })
                    continue

                elapsed = time.perf_counter() - t0
                try:
                    composite.save(shot_dir / "composite.png", format="PNG")
                except Exception as exc:  # noqa: BLE001
                    errored += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "phase": "sdxl_inpaint",
                        "error": f"save_error:{type(exc).__name__}:{exc}",
                        "insert_prompt": insert_prompt,
                        "mask_prompt": mask_prompt,
                        "seed": seed,
                        "dtype": dtype_str,
                    })
                    continue

                atomic_write_json(shot_dir / "meta.json", {
                    "shot_id": shot_id,
                    "backend": self.name,
                    "mode": "real",
                    "phase": "complete",
                    "dtype": dtype_str,
                    "insert_prompt": insert_prompt,
                    "mask_prompt": mask_prompt,
                    "seed": seed,
                    "width": _RENDER_WIDTH,
                    "height": _RENDER_HEIGHT,
                    "num_inference_steps": _NUM_INFERENCE_STEPS,
                    "render_time_s": round(elapsed, 2),
                    "input_still": str(still_path),
                    "input_still_source": rec["source_tag"],
                    "composite_hash": _composite_hash(
                        str(still_path), mask_prompt, raw_insert,
                    ),
                    "mask_hash": _mask_hash(mask_prompt),
                    "florence_reason": flo_reason,
                    "pipeline_reason": pipe_reason,
                    "florence_task": _FLORENCE_TASK,
                    "env_prompt": shot.get("env_prompt", ""),
                    "camera": shot.get("camera", ""),
                })
                rendered += 1

        if oom > 0:
            write_status(
                out_dir, STATUS_OOM,
                f"florence2_sdxl_comp OOM on {oom}/{len(shots)} shots "
                f"(rendered={rendered}, errored={errored}, "
                f"no_still={no_still})",
                backend=self.name, mode="real",
                rendered=rendered, oom=oom, errored=errored,
                no_still=no_still,
                florence_reason=flo_reason,
                pipeline_reason=pipe_reason,
            )
        else:
            write_status(
                out_dir, STATUS_READY,
                f"florence2_sdxl_comp READY: {rendered}/{len(shots)} rendered"
                + (f", {errored} errored" if errored else "")
                + (f", {no_still} without still" if no_still else ""),
                backend=self.name, mode="real",
                rendered=rendered, errored=errored, no_still=no_still,
                florence_reason=flo_reason,
                pipeline_reason=pipe_reason,
            )

    # ------------------------------------------------------------------
    # Florence-2 mask post-processing
    # ------------------------------------------------------------------
    def _rasterize_mask(self, parsed, image_size: tuple[int, int],
                        task: str):
        """Turn Florence-2's parsed polygon/bbox output into a PIL
        grayscale mask.  Handles the three shapes that Florence-2 can
        return depending on the task token: polygon list, bbox list,
        or an already-rasterised dict with a key matching the task."""
        from PIL import Image, ImageDraw  # type: ignore

        w, h = image_size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        payload = parsed.get(task, parsed) if isinstance(parsed, dict) else parsed

        polygons = []
        bboxes = []
        if isinstance(payload, dict):
            polygons = payload.get("polygons", []) or []
            bboxes = payload.get("bboxes", []) or []

        for poly_group in polygons:
            # Florence-2 groups polygons per instance; each instance is
            # a list of flattened [x1,y1,x2,y2,...] floats.
            if not poly_group:
                continue
            for flat in poly_group:
                coords = [(float(flat[i]), float(flat[i + 1]))
                          for i in range(0, len(flat) - 1, 2)]
                if len(coords) >= 3:
                    draw.polygon(coords, fill=255)

        if not polygons and bboxes:
            for bb in bboxes:
                if len(bb) == 4:
                    x1, y1, x2, y2 = (float(v) for v in bb)
                    draw.rectangle([x1, y1, x2, y2], fill=255)

        return mask
