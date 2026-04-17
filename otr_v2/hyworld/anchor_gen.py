"""
anchor_gen.py  --  SD 1.5 anchor frame generator (Phase B v0).
==============================================================

Replaces the solid-color placeholder PNG that ``worker.run_stub`` writes per
shot with a real anchor frame from Stable Diffusion 1.5.  The downstream
Ken Burns motion step in ``worker._make_motion_clip`` is unchanged --
it just gets a richer input image to animate.

Why SD 1.5 (decision recorded 2026-04-16):
    - Already on disk at
      ``C:/Users/jeffr/Documents/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.ckpt``
      (4.3 GB).  No download blocking the first render.
    - 4-7 GB VRAM peak vs. SDXL's 9-13.5 GB; large headroom under the
      14.5 GB ceiling.
    - Its "muddy / lower-fidelity" failure mode is plausibly a feature for
      the SIGNAL LOST broadcast-artifact aesthetic.
    - Massive period-style LoRA library if reinforcement is needed later.
    - Upgrade path to SDXL is preserved (this module accepts an injectable
      model loader; swapping in SDXL is one config change).

See ``docs/superpowers/consultations/2026-04-16-2026-04-16-phase-b-anchor-pick/``
for the full round-robin transcript.

Design constraints (binding):
    C2  No CheckpointLoaderSimple in main graph.  This module is intended
        for use only in spawned subprocesses (worker.py, future sidecars).
    C3  Subprocess-safe: no module-level torch import; the inference
        callable is constructed only when actually generating.
    C7  Audio-side files untouched.  This module writes ONLY under
        ``io/hyworld_out/<job_id>/anchors/`` and per-shot anchor PNGs.

Cache key (SHA-256 over a stable JSON encoding):
    (model_id, lora_set_hash, prompt, negative_prompt, seed,
     width, height, sampler, steps, cfg)
Same machine + same key  ->  same PNG bytes (read from cache, no inference).

Public surface:
    SIGNAL_LOST_STYLE / SIGNAL_LOST_NEGATIVE
        Default style / negative prompt suffixes for the SIGNAL LOST look.
    AnchorRequest, AnchorResult
        Lightweight dataclasses describing the per-shot work.
    build_prompt(shot, *, style_suffix=...)
        Compose the full prompt from a shotlist shot dict.
    cache_key(req) / cache_path(req, cache_dir)
        Deterministic key + path resolution.
    generate_for_shotlist(
        shots, out_root, *, infer=None, model_loader=None, ...
    )
        Main entry point.  Returns shot_id -> AnchorResult.

The two injectable seams (``infer`` and ``model_loader``) keep this module
unit-testable without diffusers / torch installed:

    - ``infer``         -- a callable (req: AnchorRequest) -> bytes
                           that returns PNG bytes.  If passed, model_loader
                           is ignored.  Tests pass a deterministic fake.
    - ``model_loader``  -- a zero-arg callable that constructs and returns
                           an ``infer`` callable on first cache miss.  This
                           defers torch / diffusers imports until something
                           actually needs to be generated, so a fully-cached
                           re-run incurs zero model-load cost.

If neither is supplied, ``_default_sd15_loader()`` is used; importing it
requires diffusers + torch and will raise ImportError if either is missing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Optional

# Same path-shim trick worker.py uses so this module is importable both as
# ``otr_v2.hyworld.anchor_gen`` AND from a bare ``python anchor_gen.py``
# invocation in a stripped sidecar env.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
try:
    from _atomic import atomic_write_json, atomic_write_text  # type: ignore
except ImportError:  # pragma: no cover -- worker.py degrades the same way.
    def atomic_write_json(path, data, indent: int = 2) -> None:  # type: ignore
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(data, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )

    def atomic_write_text(path, text: str) -> None:  # type: ignore
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Style scaffolding (SIGNAL LOST aesthetic)
# ---------------------------------------------------------------------------
# These are SAFE FOR WORK and intentionally restrained.  The goal is a
# 1980s-VHS / over-the-air UHF / CRT-glow look that reads as an
# atmospheric still from a late-night radio drama -- never characters
# (C6: IP-Adapter for environments only; SD 1.5 character output is even
# more failure-prone than SDXL's, so we keep characters out of prompts
# entirely and let the audio carry them).

SIGNAL_LOST_STYLE: str = (
    "1980s VHS broadcast still, soft CRT glow, scanline ghosting, "
    "muted desaturated palette, late-night Los Angeles atmosphere, "
    "Miracle Mile dusk light, environmental storytelling, "
    "no people, empty environment, wide cinematic framing"
)

SIGNAL_LOST_NEGATIVE: str = (
    "people, faces, hands, character, text, watermark, signature, "
    "logo, oversharp, hyperreal, 4k, ultra hd, photoreal, modern, "
    "neon anime, cgi render"
)

# Default model identifier.  This is just a string used in the cache key;
# the real model path is resolved by the loader (or by the caller).
DEFAULT_MODEL_ID: str = "sd15-v1-5-pruned-emaonly"
DEFAULT_WIDTH: int = 1024
DEFAULT_HEIGHT: int = 576   # 16:9-ish; SD 1.5 is happiest near 512-768.
DEFAULT_STEPS: int = 28
DEFAULT_CFG: float = 7.0
DEFAULT_SAMPLER: str = "euler_a"
DEFAULT_SEED_BASE: int = 0


# ---------------------------------------------------------------------------
# Request / result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnchorRequest:
    """All inputs that affect the rendered pixels.

    The cache key is derived from this dataclass; any field added here
    becomes part of the determinism contract.  Be deliberate about new
    fields -- adding one invalidates every existing cached PNG.
    """
    shot_id: str
    prompt: str
    negative_prompt: str
    seed: int
    width: int
    height: int
    sampler: str
    steps: int
    cfg: float
    model_id: str
    lora_set_hash: str = ""

    def to_key_dict(self) -> dict:
        """JSON-stable subset that participates in the cache key.

        ``shot_id`` is intentionally EXCLUDED -- the same prompt at the
        same seed should resolve to the same PNG regardless of which
        shot first asked for it.  Two shots with identical env_prompt
        and seed will share the cached file.
        """
        d = asdict(self)
        d.pop("shot_id", None)
        return d


@dataclass
class AnchorResult:
    """Per-shot render outcome."""
    shot_id: str
    png_path: Path
    cache_hit: bool
    prompt: str
    negative_prompt: str
    seed: int
    cache_key: str
    elapsed_sec: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Pure helpers (no torch, no I/O of model weights)
# ---------------------------------------------------------------------------

def build_prompt(shot: dict, *, style_suffix: str = SIGNAL_LOST_STYLE) -> str:
    """Compose the per-shot positive prompt.

    Uses ``env_prompt`` as the spine, layers in ``mood`` and ``camera``
    if present, then appends the style suffix.  Empty / missing fields
    are skipped silently so an under-specified shot still gets a usable
    prompt.
    """
    env = (shot.get("env_prompt") or "").strip() or "empty room, dim light"
    mood = (shot.get("mood") or "").strip()
    camera = (shot.get("camera") or "").strip()

    parts: list[str] = [env]
    if mood:
        parts.append(f"{mood} mood")
    if camera:
        parts.append(f"camera: {camera}")
    if style_suffix:
        parts.append(style_suffix)
    return ", ".join(parts)


def derive_seed(shot: dict, seed_base: int = DEFAULT_SEED_BASE) -> int:
    """Stable per-shot seed.

    Combines a caller-supplied base with a hash of ``shot_id`` so each
    shot has its own deterministic seed but the whole episode can be
    re-rolled by changing ``seed_base``.
    """
    shot_id = shot.get("shot_id", "")
    h = hashlib.sha256(shot_id.encode("utf-8")).digest()
    # Take first 4 bytes -> uint32, fold against seed_base.  Mask to
    # uint32 range so downstream samplers don't choke on negative ints.
    sid_u32 = int.from_bytes(h[:4], "big")
    return (seed_base ^ sid_u32) & 0xFFFFFFFF


def cache_key(req: AnchorRequest) -> str:
    """SHA-256 hex digest over the determinism-relevant fields."""
    payload = json.dumps(req.to_key_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_path(req: AnchorRequest, cache_dir: Path) -> Path:
    """Resolve ``<cache_dir>/<sha256>.png`` for the given request."""
    return Path(cache_dir) / f"{cache_key(req)}.png"


# ---------------------------------------------------------------------------
# Default SD 1.5 loader (lazy: imports torch / diffusers only on first call)
# ---------------------------------------------------------------------------

def _default_sd15_loader(
    *,
    model_path: str = "C:/Users/jeffr/Documents/ComfyUI/models/checkpoints/v1-5-pruned-emaonly.ckpt",
    device: str = "cuda",
) -> Callable[[AnchorRequest], bytes]:  # pragma: no cover -- requires torch.
    """Construct an SD 1.5 inference callable.

    Heavy: imports diffusers + torch and loads ~4 GB of weights.  Call
    once per worker-process lifetime, not per shot.

    Returns a callable ``infer(req: AnchorRequest) -> bytes`` that
    produces PNG bytes for the given request.  Raises ImportError if
    diffusers / torch aren't installed in the current env.
    """
    try:
        import torch  # type: ignore
        from diffusers import StableDiffusionPipeline  # type: ignore
        from io import BytesIO
    except ImportError as e:
        raise ImportError(
            "anchor_gen default loader requires diffusers + torch. "
            "Install with: pip install diffusers transformers accelerate. "
            f"Underlying error: {e}"
        ) from e

    pipe = StableDiffusionPipeline.from_single_file(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,        # SFW prompts only -- no need for the checker
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    # SD 1.5 + diffusers default scheduler is fine for euler_a-like behavior;
    # if a caller wants something else they should pass their own loader.

    def _infer(req: AnchorRequest) -> bytes:
        generator = torch.Generator(device=device).manual_seed(int(req.seed))
        image = pipe(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            width=req.width,
            height=req.height,
            num_inference_steps=req.steps,
            guidance_scale=req.cfg,
            generator=generator,
        ).images[0]
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    return _infer


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_for_shotlist(
    shots: list[dict],
    out_root: Path,
    *,
    infer: Optional[Callable[[AnchorRequest], bytes]] = None,
    model_loader: Optional[Callable[[], Callable[[AnchorRequest], bytes]]] = None,
    cache_dir: Optional[Path] = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    steps: int = DEFAULT_STEPS,
    cfg: float = DEFAULT_CFG,
    sampler: str = DEFAULT_SAMPLER,
    seed_base: int = DEFAULT_SEED_BASE,
    style_suffix: str = SIGNAL_LOST_STYLE,
    style_negative: str = SIGNAL_LOST_NEGATIVE,
    model_id: str = DEFAULT_MODEL_ID,
    lora_set_hash: str = "",
    progress_cb: Optional[Callable[[int, int, AnchorResult], None]] = None,
) -> dict[str, AnchorResult]:
    """Generate (or fetch from cache) one anchor PNG per shot.

    Behavior:
        1. Build the AnchorRequest for each shot.
        2. If the cache PNG exists -> mark cache hit, no inference.
        3. Otherwise:
             a. If ``infer`` was supplied, use it.
             b. Else if ``model_loader`` was supplied, lazily call it
                ONCE on the first cache miss to materialize an ``infer``.
             c. Else use ``_default_sd15_loader()``.
           Then run inference, write PNG to cache + into shot dir.
        4. Per-shot meta and a top-level cache_index.json are written
           atomically.

    The resulting ``AnchorResult.png_path`` is the path the caller
    (worker.py) should treat as the shot's anchor frame.

    Args:
        shots: list of shot dicts (from shotlist.json["shots"]).
        out_root: ``io/hyworld_out/<job_id>/`` (the per-job output root).
            Anchors are written under ``out_root / "anchors"`` by default.
        infer / model_loader: see module docstring.
        cache_dir: override cache location.  Defaults to ``out_root/anchors``.
        progress_cb: optional callback invoked per shot, useful for
            updating STATUS.json from the calling worker.
            Signature: ``(index, total, result) -> None``.

    Returns:
        Dict mapping ``shot_id`` -> ``AnchorResult``.
    """
    out_root = Path(out_root)
    cache_dir = Path(cache_dir) if cache_dir is not None else out_root / "anchors"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, AnchorResult] = {}
    resolved_infer: Optional[Callable[[AnchorRequest], bytes]] = infer

    total = len(shots)
    for idx, shot in enumerate(shots):
        shot_id = shot.get("shot_id", f"shot_{idx:03d}")
        req = AnchorRequest(
            shot_id=shot_id,
            prompt=build_prompt(shot, style_suffix=style_suffix),
            negative_prompt=style_negative,
            seed=derive_seed(shot, seed_base=seed_base),
            width=width,
            height=height,
            sampler=sampler,
            steps=steps,
            cfg=cfg,
            model_id=model_id,
            lora_set_hash=lora_set_hash,
        )
        key = cache_key(req)
        cpath = cache_dir / f"{key}.png"

        t0 = time.time()
        cache_hit = cpath.exists() and cpath.stat().st_size > 0
        error = ""
        if not cache_hit:
            # Lazy-load the inference callable on the first real miss.
            if resolved_infer is None:
                if model_loader is not None:
                    resolved_infer = model_loader()
                else:  # pragma: no cover -- requires torch.
                    resolved_infer = _default_sd15_loader()
            try:
                png_bytes = resolved_infer(req)
                if not png_bytes or not png_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError("inference returned non-PNG bytes")
                # Atomic write: tempfile + os.replace, mirroring _atomic.py.
                tmp = cpath.with_suffix(cpath.suffix + ".tmp")
                tmp.write_bytes(png_bytes)
                # Use os.replace for atomicity; on Windows the worker has
                # already retried the same race in _atomic._replace_with_retry,
                # but the bare PNG write here is single-writer (one shot at a
                # time in the worker loop), so a simple replace is safe.
                import os
                os.replace(tmp, cpath)
            except Exception as e:  # noqa: BLE001 -- record + continue
                error = f"{type(e).__name__}: {e}"
        elapsed = time.time() - t0

        # Mirror the cached PNG into the shot dir so worker.py can hand
        # ffmpeg a stable path without knowing the cache layout.
        shot_dir = out_root / shot_id
        shot_dir.mkdir(parents=True, exist_ok=True)
        shot_png = shot_dir / "render.png"
        if not error and cpath.exists():
            shot_png.write_bytes(cpath.read_bytes())

        result = AnchorResult(
            shot_id=shot_id,
            png_path=shot_png if not error else cpath,
            cache_hit=cache_hit,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
            cache_key=key,
            elapsed_sec=round(elapsed, 3),
            error=error,
        )
        results[shot_id] = result
        if progress_cb is not None:
            try:
                progress_cb(idx, total, result)
            except Exception:  # progress callback errors must never abort the run
                pass

    # Cache index: shot_id -> {key, prompt, seed, cache_hit, error}.
    # Useful for debugging "why did shot X re-render?" without diffing PNGs.
    index = {
        sid: {
            "cache_key": r.cache_key,
            "cache_hit": r.cache_hit,
            "prompt": r.prompt,
            "seed": r.seed,
            "elapsed_sec": r.elapsed_sec,
            "error": r.error,
            "png_path": str(r.png_path),
        }
        for sid, r in results.items()
    }
    atomic_write_json(cache_dir / "cache_index.json", index)
    return results


# ---------------------------------------------------------------------------
# Standalone entry (useful for sidecar split later; not used by worker.py
# in Phase B v0).
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover -- exercised by integration only.
    if len(sys.argv) < 2:
        print("Usage: python anchor_gen.py <job_dir>", file=sys.stderr)
        sys.exit(1)
    job_dir = Path(sys.argv[1])
    shotlist_path = job_dir / "shotlist.json"
    if not shotlist_path.exists():
        print(f"shotlist.json not found in {job_dir}", file=sys.stderr)
        sys.exit(1)
    shots = json.loads(shotlist_path.read_text(encoding="utf-8")).get("shots", [])
    out_root = job_dir.parent.parent.parent / "io" / "hyworld_out" / job_dir.name
    results = generate_for_shotlist(shots, out_root)
    summary = {sid: {"cache_hit": r.cache_hit, "error": r.error} for sid, r in results.items()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
