"""OTR_ImageGenDispatcher -- cache-checked image generation + ledger write-back (C1).

The terminal C node: for each image object it (1) composes the request cache key
``(role, object_id, prompt_hash, seed, engine_id, engine_version)``, (2) reuses
the existing image on a cache hit, (3) otherwise ``assert_usable`` (fail-closed,
NO silent Flux swap) -> takes the SHARED AS-3 GPU-residency lease -> generates
(in-process Flux gen-1 OR a cu128 sidecar; injected ``gen_fn`` in tests) -> writes
the still CONTENT-ADDRESSED at ``output/otr/stills/{portrait_content_hash}.png``
(AS-5, never-overwrite) -> stamps the ledger -> releases the lease + re-probes
NVML -> emits the ``image_done`` STRING token (mirrors ``audio_done``).

Seam guards folded in (PASS-IMG MUST/SHOULD-FIX):
* DISK-PATH handoff -- a sidecar returns a ``.png`` PATH, never an IMAGE tensor;
  the dispatcher reads decoded pixels to content-address. The ``prompt`` is
  asserted NOT to be a path (distinct prompt-STRING vs path-STRING contract).
* CONTENT-ADDRESSED cache so a re-gen (changed prompt/seed/engine) yields a new
  key -> new pixels -> new ``portrait_content_hash`` -> new file, and B's mesh
  cache (keyed on that hash) invalidates correctly.
* FRESH-mode hard cap = ``min(fresh_cap, beat_budget)`` -- never over-generate.
* image generation is the only GPU step; it is injected on the CPU test path, so
  this node's cache/ledger/gate logic is fully unit-tested without a GPU.

Cold-import clean: torch / PIL / numpy are never imported at module scope.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os

log = logging.getLogger("OTR")

from ._otr_shared import gpu_residency as _lease
from ._otr_shared import portrait_ledger as _pl
from ._otr_image_engines import registry as _ireg


def _content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def request_cache_key(role, object_id, prompt_hash, seed, engine_id, engine_version) -> str:
    """The dispatch dedup key (PASS-IMG MUST-FIX #5). A change in ANY field ->
    new key -> regen -> new content hash -> B's mesh cache invalidates."""
    return _content_hash([
        str(role), str(object_id), str(prompt_hash), int(seed),
        str(engine_id), str(engine_version),
    ])


def _assert_not_path(prompt: str) -> None:
    """Fail-closed prompt-STRING vs path-STRING guard (PASS-IMG SHOULD-FIX)."""
    p = str(prompt or "")
    looks_pathy = (
        os.sep in p or (os.altsep and os.altsep in p)
        or p.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    )
    if looks_pathy:
        raise ValueError(
            "OTR_ImageGenDispatcher: image prompt looks like a PATH, not prompt "
            f"text ({p[:60]!r}); the prompt-STRING and path-STRING sockets must "
            "not be crossed"
        )


def _coerce_pixels(result):
    """Decoded uint8 pixel array from a gen_fn result.

    A sidecar returns a ``.png`` PATH (disk-path handoff, never a tensor); an
    in-process gen returns a numpy uint8 array (a comfy IMAGE tensor must be
    ``.clone()``d + converted by the caller BEFORE here). Lazy PIL/numpy import.
    """
    if isinstance(result, str):
        from PIL import Image  # lazy (V-12)
        import numpy as np     # lazy
        return np.asarray(Image.open(result).convert("RGB"))
    if hasattr(result, "tobytes"):           # numpy array (decoded pixels)
        return result
    raise TypeError(
        "gen_fn must return a .png path (sidecar) or a decoded uint8 pixel "
        "array (in-process); got " + type(result).__name__
    )


def apply_fresh_cap(n_requested: int, fresh_cap: int, beat_budget: int) -> int:
    """Hard cap FRESH renders to ``min(fresh_cap, beat_budget)`` (never over-gen)."""
    cap = min(int(fresh_cap), int(beat_budget)) if beat_budget else int(fresh_cap)
    return max(0, min(int(n_requested), cap))


def dispatch_images(ledger: dict, image_policy: dict, image_prompts: dict, *,
                    gen_fn=None, output_dir=None, lockdir=None, lease_timeout_s=120.0):
    """Generate (or cache-reuse) one portrait per character; stamp the ledger.

    ``gen_fn(request: dict) -> (numpy uint8 array | .png path)`` is the only GPU
    step (injected in tests; the real Flux path on the operator smoke). Returns
    ``(patched_ledger, image_done, report, warnings)``. Never raises on a normal
    miss; ``assert_usable`` failures are recorded as warnings + skipped
    (fail-closed: that object simply has no image, never a silent wrong engine).
    """
    warnings: list = []
    report: list = []
    cast = ledger.get("cast") if isinstance(ledger.get("cast"), list) else []
    images_section = ledger.get("images") if isinstance(ledger.get("images"), dict) else {}
    cache_index = dict(images_section.get("cache_index") or {})
    images = list(images_section.get("images") or [])
    seed = int(((image_policy or {}).get("seed") or {}).get("request_seed") or 0)
    other_engine = ((image_policy or {}).get("image_models") or {}).get("other_beats_image_model") or {}
    engine_id = other_engine.get("engine_id") if isinstance(other_engine, dict) else str(other_engine or "")
    role = "character_video"

    rev = int(images_section.get("image_revision") or 0) + 1
    made = 0
    reused = 0
    for cid, pinfo in (image_prompts or {}).items():
        prompt = str((pinfo or {}).get("prompt") or "")
        prompt_hash = str((pinfo or {}).get("prompt_hash") or "")
        if not engine_id:
            warnings.append(f"{cid}: no image engine selected; skipped (fail-closed)")
            continue
        try:
            _assert_not_path(prompt)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        eng_version = str(getattr(_safe_engine(engine_id), "engine_version", "1"))
        key = request_cache_key(role, cid, prompt_hash, seed, engine_id, eng_version)
        if key in cache_index:
            reused += 1
            report.append(f"{cid}: cache HIT ({cache_index[key][:12]})")
            continue
        # cache miss -> assert usable (fail-closed) -> lease -> generate -> stamp
        try:
            _ireg.assert_usable(engine_id, role)
        except Exception as exc:  # noqa: BLE001  (EngineUnusable et al.)
            warnings.append(f"{cid}: engine '{engine_id}' not usable for {role} ({exc}); skipped")
            continue
        if gen_fn is None:
            warnings.append(f"{cid}: no gen_fn (GPU render is the operator smoke); skipped on CPU")
            continue
        request = {
            "request_id": key, "role": role, "object_id": cid,
            "engine_id": engine_id, "engine_version": eng_version,
            "prompt": prompt, "prompt_hash": prompt_hash, "seed": seed,
        }
        lease = None
        try:
            lease = _lease.acquire(timeout_s=lease_timeout_s, lockdir=lockdir)
            pixels = _coerce_pixels(gen_fn(request))
            content_hash = _pl.compute_portrait_hash(pixels)
            path = _pl.stamp_portrait(ledger, cid, pixels, output_dir=output_dir)
        except _lease.LeaseTimeout as exc:
            warnings.append(f"{cid}: GPU lease timeout ({exc}); skipped")
            continue
        finally:
            if lease is not None:
                _lease.release(lease)
        # post-generation residency confirm (best-effort; gates the C->A handoff).
        if not _lease.wait_until_below_mb(15000, attempts=3, sleep_s=0.0):
            log.info("[OTR_ImageGenDispatcher] NVML re-probe inconclusive (CPU/no-NVML or busy)")
        image_id = f"img_{cid}_{content_hash[:12]}"
        images.append({
            "image_id": image_id, "role": role, "object_id": cid,
            "path": str(path), "engine_id": engine_id, "engine_version": eng_version,
            "request_hash": key, "portrait_content_hash": content_hash,
            "prompt_hash": prompt_hash, "provenance": {"source": (pinfo or {}).get("source", "")},
        })
        cache_index[key] = image_id
        made += 1
        report.append(f"{cid}: generated -> {os.path.basename(str(path))}")

    ledger["images"] = {
        "image_revision": rev,
        "granularity_by_role": (image_policy or {}).get("granularity") or {},
        "images": images,
        "cache_index": cache_index,
        "warnings": warnings,
    }
    image_done = f"image:done:rev={rev} made={made} reused={reused}"
    report.insert(0, f"image_dispatch rev={rev}: made={made} reused={reused} total={len(images)}")
    for w in warnings:
        report.append(f"WARN: {w}")
        log.warning("[OTR_ImageGenDispatcher] %s", w)
    return ledger, image_done, "\n".join(report), warnings


def _safe_engine(engine_id):
    try:
        return _ireg.get_engine(engine_id)
    except Exception:  # noqa: BLE001
        return None


class OTRImageGenDispatcher:
    """Registered as ``OTR_ImageGenDispatcher``. Cache-checked image gen + ledger + image_done."""

    CATEGORY = "OldTimeRadio/v2/image"
    FUNCTION = "dispatch"
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("patched_ledger_json", "image_done", "report")
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "Frozen ledger JSON; the dispatcher stamps ledger['images'] into it.",
                }),
                "image_policy_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "OTR_ImageDirector policy (engine per role + granularity + seed).",
                }),
                "image_prompts_json": ("STRING", {
                    "multiline": True, "default": "{}", "forceInput": True,
                    "tooltip": "OTR_MetaBriefImagePromptGen output (char_id -> prompt + hash).",
                }),
            },
            "optional": {
                "gate_in": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (e.g. audio_done); opaque STRING.",
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def dispatch(self, script_json, image_policy_json="{}", image_prompts_json="{}", gate_in=""):
        led = self._loads(script_json, {})
        policy = self._loads(image_policy_json, {})
        prompts = self._loads(image_prompts_json, {})
        # gen_fn is None here -> the in-graph node defers actual pixels to the
        # GPU/operator smoke (the in-process Flux gen-1 wiring lands with the
        # render path); the CPU platform tests call dispatch_images() directly
        # with an injected gen_fn. Cache reuse + ledger shape still resolve.
        led, image_done, report, _warn = dispatch_images(
            led, policy, prompts, gen_fn=None,
        )
        patched = json.dumps(led, ensure_ascii=True, separators=(",", ":"))
        return (patched, image_done, report)

    @staticmethod
    def _loads(raw, default):
        try:
            v = json.loads(raw or "null")
            return v if isinstance(v, type(default)) else default
        except (ValueError, TypeError):
            return default
