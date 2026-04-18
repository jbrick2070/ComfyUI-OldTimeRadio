"""
otr_v2.visual.backends.pulid_portrait  --  Day 3 identity-locked portraits
============================================================================

PuLID-FLUX portrait renderer.  Takes 1-3 reference face images per shot
and produces a portrait locked to that identity while honoring the
shot's env_prompt and camera directive.

Canonical upstream (checked into ROADMAP.md Day 3):
    https://github.com/ToTheBeginning/PuLID  -- PuLID-FLUX variant

Dependencies (added to ``requirements.video.txt`` on Day 3):
    torch==2.10.0+cu130, diffusers==0.37.0, transformers==5.5.0,
    accelerate==1.13.0, facexlib, insightface, eva-clip (pulled by
    PuLID upstream), Pillow.

Model discovery:
    FLUX weights: see flux_anchor._MODEL_PATH
    PuLID weights: ``C:/Users/jeffr/Documents/ComfyUI/models/pulid/pulid_flux.safetensors``
    (env override: OTR_PULID_MODEL)

Two execution modes (same pattern as flux_anchor.py):

1. **Real mode** (default when weights exist + CUDA is available):
   loads FluxPipeline FP8 + PuLID adapter, encodes reference images
   into identity embedding, renders 1024x1024 with `enable_model_cpu_offload`.
   Target peak VRAM: 14 GB (Day 3 kill criterion).

2. **Stub mode** (``OTR_PULID_STUB=1`` OR weights missing OR no CUDA):
   emits a deterministic 1024x1024 PNG whose color signature is keyed
   on ``(shot_id, refs_hash)`` so identity-lock behavior is unit-testable
   without real weights -- two shots with the same refs produce the
   same color signature regardless of shot-level prompt noise.

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

_DEFAULT_PULID_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\pulid\pulid_flux.safetensors"
)
_PULID_PATH = Path(os.environ.get("OTR_PULID_MODEL", str(_DEFAULT_PULID_PATH)))

# FLUX base -- pulled from the same default as flux_anchor so a single
# weight set powers both Day 2 anchors and Day 3 portraits.
_DEFAULT_FLUX_PATH = Path(
    r"C:\Users\jeffr\Documents\ComfyUI\models\diffusers\FLUX.1-dev"
)
_FLUX_PATH = Path(os.environ.get("OTR_FLUX_MODEL", str(_DEFAULT_FLUX_PATH)))

# Day 3 gate: 1024x1024 square matches Day 2 so downstream nodes never
# branch on dimension.
_RENDER_WIDTH = 1024
_RENDER_HEIGHT = 1024

# Cinematic suffix lifted from flux_anchor for stylistic coherence.
_STYLE_SUFFIX = (
    "cinematic portrait, 35mm film, anamorphic lens, volumetric "
    "lighting, muted color grade, sharp focus on eyes"
)

# PuLID-FLUX guidance / strength defaults.  Day 3 tuning starts at the
# canonical PuLID README defaults; the Day 10 canary may need to nudge
# id_weight once we have real weights and per-episode reference sets
# (characters are emitted fresh by the LLM script process each run,
# so there's no fixed roster to tune against -- tuning has to generalize).
_NUM_INFERENCE_STEPS = 20
_GUIDANCE_SCALE = 3.5
_ID_WEIGHT = 1.0  # how strongly the identity embedding dominates
_TRUE_CFG = 1.0   # PuLID-FLUX default


# ---- Helpers --------------------------------------------------------------

def _stub_png(path: Path, r: int, g: int, b: int,
              width: int = _RENDER_WIDTH,
              height: int = _RENDER_HEIGHT) -> None:
    """Emit a valid 1024x1024 solid PNG with no external deps."""
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


def _extract_refs(shot: dict) -> list[str]:
    """Pull reference image paths / identifiers out of a shotlist entry.

    The shotlist schema carries them under ``refs`` (list[str]) per the
    Director's plan.  Older plans may instead embed a single ``ref`` or
    a ``character`` field -- support both.  Returns a normalized list
    (strings, deduped, order-preserving).
    """
    raw: list = []
    for key in ("refs", "reference_images", "id_refs"):
        v = shot.get(key)
        if isinstance(v, list):
            raw.extend(v)
        elif isinstance(v, str) and v:
            raw.append(v)
    # Single-ref legacy keys
    for key in ("ref", "character_ref", "portrait_ref"):
        v = shot.get(key)
        if isinstance(v, str) and v:
            raw.append(v)
    # Normalize + dedupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for item in raw:
        s = str(item).strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _refs_hash(refs: list[str]) -> str:
    """Stable hex hash of the reference list -- identity key in stub mode."""
    if not refs:
        return "no_refs"
    h = hashlib.sha256()
    for r in refs:
        h.update(r.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:12]


def _build_prompt(shot: dict) -> str:
    env = (shot.get("env_prompt") or "").strip()
    camera = (shot.get("camera") or "").strip()
    character = (shot.get("character") or "").strip()
    parts = []
    if character:
        parts.append(f"portrait of {character}")
    if env:
        parts.append(env)
    if camera:
        parts.append(camera)
    parts.append(_STYLE_SUFFIX)
    return ", ".join(parts)


def _derive_seed(shot: dict, shot_idx: int,
                 base: int = 0x7075_6C69) -> int:
    """Deterministic per-shot seed.  Same shot+base -> same image.

    The ``shot_id`` is preferred so seeds survive re-orderings; fall
    back to shot index so shots without an id still get a stable seed.
    Base constant spells "puli" in ASCII to distinguish from flux_anchor.
    """
    key = f"{shot.get('shot_id', '')}|{shot_idx}|{base}"
    h = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "big") & 0x7FFFFFFF


def _color_from_refs(refs_hash: str) -> tuple[int, int, int]:
    """Map the refs_hash to a deterministic RGB triple.

    Identity-lock simulation: two shots with the same refs land on the
    same color regardless of seed, prompt, or shot index.  Different
    refs -> different color.  Unit tests assert this invariant.
    """
    # Walk the first 6 hex chars as three byte pairs.
    r = int(refs_hash[0:2], 16) if len(refs_hash) >= 2 else 128
    g = int(refs_hash[2:4], 16) if len(refs_hash) >= 4 else 128
    b = int(refs_hash[4:6], 16) if len(refs_hash) >= 6 else 128
    # Lift away from full black so output is visually distinguishable.
    r = max(40, r)
    g = max(40, g)
    b = max(40, b)
    return (r, g, b)


def _should_stub() -> tuple[bool, str]:
    """Return (stub_requested, reason).  Reason is logged into meta.json."""
    if os.environ.get("OTR_PULID_STUB", "").strip() == "1":
        return (True, "OTR_PULID_STUB=1")
    if not _PULID_PATH.exists():
        return (True, f"pulid_weights_missing:{_PULID_PATH}")
    if not _FLUX_PATH.exists():
        return (True, f"flux_weights_missing:{_FLUX_PATH}")
    return (False, "")


def _try_load_pipeline():
    """Load FluxPipeline + PuLID adapter.  Returns (pipe, reason).

    ``pipe`` is None when import/load fails -- caller falls back to
    stub mode and records the reason in meta.json.

    We do NOT raise on any subprocess path: the gate is "still emit PNGs
    so downstream has art to composite", ERR/OOM is tracked via status.
    """
    try:
        import torch  # type: ignore
        from diffusers import FluxPipeline  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return (None, f"import_error:{type(exc).__name__}:{exc}")

    if not torch.cuda.is_available():
        return (None, "cuda_unavailable")

    # PuLID integration is gated behind the canonical upstream.  On Day
    # 3 the ComfyUI-side install lives at the path below; absent that,
    # we surface the missing-adapter reason and let the caller stub.
    try:
        # Import guard: PuLID isn't a pip package -- it ships as a
        # node pack under ComfyUI/custom_nodes/PuLID or as a git-cloned
        # standalone module.  The import path differs across installs,
        # so we try the two most common layouts before giving up.
        pulid_mod = None
        for modname in ("pulid.pipeline_flux", "PuLID.pipeline_flux",
                        "comfyui_pulid_flux.pipeline_flux"):
            try:
                pulid_mod = __import__(modname, fromlist=["*"])
                break
            except Exception:
                continue
        if pulid_mod is None:
            return (None, "pulid_module_unavailable")

        pipe = FluxPipeline.from_pretrained(
            str(_FLUX_PATH),
            torch_dtype=torch.float8_e4m3fn,
            local_files_only=True,
        )
        # Attach PuLID.  The upstream API exposes either
        # ``PuLIDFluxPipeline(pipe, weight_path=...)`` or an
        # ``inject_pulid(pipe, weight_path)`` helper; try both so we're
        # robust against PuLID repo revisions.
        if hasattr(pulid_mod, "PuLIDFluxPipeline"):
            pipe = pulid_mod.PuLIDFluxPipeline(pipe, weight_path=str(_PULID_PATH))
        elif hasattr(pulid_mod, "inject_pulid"):
            pulid_mod.inject_pulid(pipe, weight_path=str(_PULID_PATH))
        else:
            return (None, "pulid_api_unknown")

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
        return (pipe, "loaded")
    except Exception as exc:  # noqa: BLE001
        return (None, f"load_error:{type(exc).__name__}:{str(exc)[:200]}")


# ---- Backend --------------------------------------------------------------

class PulidPortraitBackend(Backend):
    """Day 3 -- PuLID-FLUX identity-locked portrait renderer."""

    name = "pulid_portrait"

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
                f"pulid_portrait stub mode: {len(shots)} shots ({stub_reason})",
            )
            self._render_stub(shots, out_dir, stub_reason)
            return

        # Real mode -- attempt pipeline load.  Any failure falls back to
        # stub rather than surfacing ERROR, so downstream nodes always
        # have PNGs to work with even when weights are mid-download.
        write_status(
            out_dir, STATUS_RUNNING,
            f"pulid_portrait real mode: loading PuLID-FLUX from {_PULID_PATH}",
        )
        pipe, reason = _try_load_pipeline()
        if pipe is None:
            write_status(
                out_dir, STATUS_RUNNING,
                f"pulid_portrait load failed ({reason}); falling back to stub",
            )
            self._render_stub(shots, out_dir, reason)
            return

        self._render_real(pipe, shots, out_dir)

    # ------------------------------------------------------------------
    # stub path -- CI-safe, no torch, no GPU
    # ------------------------------------------------------------------
    def _render_stub(self, shots: list[dict], out_dir: Path, reason: str) -> None:
        for i, shot in enumerate(shots):
            shot_id = shot.get("shot_id", f"shot_{i:03d}")
            shot_dir = out_dir / shot_id
            shot_dir.mkdir(parents=True, exist_ok=True)

            refs = _extract_refs(shot)
            rhash = _refs_hash(refs)
            # Identity-lock invariant: color keyed on refs_hash alone.
            r, g, b = _color_from_refs(rhash)
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
                "refs": refs,
                "refs_hash": rhash,
                "env_prompt": shot.get("env_prompt", ""),
                "camera": shot.get("camera", ""),
                "character": shot.get("character", ""),
                "duration_sec": float(shot.get("duration_sec", 9)),
            })

        write_status(
            out_dir, STATUS_READY,
            f"pulid_portrait stub READY: {len(shots)} shots ({reason})",
            backend=self.name, mode="stub",
        )

    # ------------------------------------------------------------------
    # real path -- loads torch lazily, uses VRAMCoordinator gate
    # ------------------------------------------------------------------
    def _render_real(self, pipe, shots: list[dict], out_dir: Path) -> None:
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
        rendered = 0
        oom = 0
        errored = 0
        no_refs = 0

        with coord.acquire(owner="pulid_portrait", job_id=out_dir.name, timeout=1800):
            for i, shot in enumerate(shots):
                shot_id = shot.get("shot_id", f"shot_{i:03d}")
                shot_dir = out_dir / shot_id
                shot_dir.mkdir(parents=True, exist_ok=True)

                refs = _extract_refs(shot)
                prompt = _build_prompt(shot)
                seed = _derive_seed(shot, i)

                if not refs:
                    # Portrait backend without refs degrades to a straight
                    # FLUX render.  We log the skip and continue so the
                    # shot still gets a PNG -- downstream will compare
                    # against ID-locked shots and notice if identity drifts.
                    no_refs += 1

                generator = torch.Generator(device="cuda").manual_seed(seed)
                t0 = time.perf_counter()
                try:
                    call_kwargs = dict(
                        prompt=prompt,
                        width=_RENDER_WIDTH,
                        height=_RENDER_HEIGHT,
                        num_inference_steps=_NUM_INFERENCE_STEPS,
                        guidance_scale=_GUIDANCE_SCALE,
                        generator=generator,
                    )
                    if refs:
                        # Canonical PuLID-FLUX accepts ``id_images`` (list
                        # of PIL.Image) and ``id_weight``.  Loading the
                        # reference images happens here so a single bad
                        # ref only kills one shot, not the whole run.
                        try:
                            from PIL import Image  # type: ignore
                        except Exception:
                            Image = None  # type: ignore
                        id_images = []
                        if Image is not None:
                            for ref_path in refs:
                                try:
                                    id_images.append(Image.open(ref_path).convert("RGB"))
                                except Exception:
                                    # Missing ref -- skip, stay running
                                    pass
                        if id_images:
                            call_kwargs["id_images"] = id_images
                            call_kwargs["id_weight"] = _ID_WEIGHT
                            call_kwargs["true_cfg"] = _TRUE_CFG

                    out = pipe(**call_kwargs)
                    img = out.images[0]
                except torch.cuda.OutOfMemoryError:
                    oom += 1
                    atomic_write_json(shot_dir / "meta.json", {
                        "shot_id": shot_id,
                        "backend": self.name,
                        "mode": "real",
                        "error": "CUDA_OOM",
                        "refs": refs,
                        "refs_hash": _refs_hash(refs),
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
                        "error": f"{type(exc).__name__}: {str(exc)[:300]}",
                        "traceback_tail": traceback.format_exc()[-800:],
                        "refs": refs,
                        "refs_hash": _refs_hash(refs),
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
                    "prompt": prompt,
                    "seed": seed,
                    "width": _RENDER_WIDTH,
                    "height": _RENDER_HEIGHT,
                    "render_time_s": round(elapsed, 2),
                    "refs": refs,
                    "refs_hash": _refs_hash(refs),
                    "id_weight": _ID_WEIGHT if refs else 0.0,
                    "env_prompt": shot.get("env_prompt", ""),
                    "camera": shot.get("camera", ""),
                    "character": shot.get("character", ""),
                    "duration_sec": float(shot.get("duration_sec", 9)),
                })
                rendered += 1

        if oom > 0:
            write_status(
                out_dir, STATUS_OOM,
                f"pulid_portrait OOM on {oom}/{len(shots)} shots "
                f"(rendered={rendered}, errored={errored}, no_refs={no_refs})",
                backend=self.name, mode="real",
                rendered=rendered, oom=oom, errored=errored, no_refs=no_refs,
            )
        else:
            write_status(
                out_dir, STATUS_READY,
                f"pulid_portrait READY: {rendered}/{len(shots)} rendered"
                + (f", {errored} errored" if errored else "")
                + (f", {no_refs} without refs" if no_refs else ""),
                backend=self.name, mode="real",
                rendered=rendered, errored=errored, no_refs=no_refs,
            )
