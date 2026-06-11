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
import time

log = logging.getLogger("OTR")

from ._otr_shared import gpu_residency as _lease
from ._otr_shared import portrait_ledger as _pl
from ._otr_image_engines import registry as _ireg

#: Smallest plausible real PNG (8-byte signature + IHDR + IDAT + IEND). Anything
#: smaller off the cross-process disk handoff is a 0-byte / truncated write,
#: never a finished render.
_MIN_PNG_BYTES = 67


class ImageHandoffTimeout(RuntimeError):
    """A sidecar-written image never became readable within the bounded retries.

    A cu128 image sidecar writes its ``.png`` to disk and hands back the PATH; if
    the main venv decodes it before the bytes are flushed it sees a 0-byte or
    truncated file (PASS-PM C1 handoff race). The dispatcher treats this as a
    fail-closed miss (warn + skip that object) -- never a silent bad image.
    """


def wait_for_file_ready(path, min_bytes: int = _MIN_PNG_BYTES,
                        attempts: int = 40, sleep_s: float = 0.05) -> str:
    """Block until ``path`` exists, is ``>= min_bytes``, and its size is STABLE
    across two consecutive probes (the writer has stopped), then return it; raise
    :class:`ImageHandoffTimeout` after the bounded retries.

    Guards the cross-process disk-path handoff against a still-flushing or
    0-byte ``.png``. Pure stdlib (os + time) -- cold-import clean (V-12).
    """
    last = -1
    for _ in range(max(2, int(attempts))):
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        if size >= int(min_bytes) and size == last:
            return path
        last = size
        if sleep_s:
            time.sleep(float(sleep_s))
    raise ImageHandoffTimeout(
        f"image handoff not ready: {path!r} (last size {last} bytes; "
        f"need >= {min_bytes} and stable across two probes)"
    )


def _content_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def request_cache_key(role, object_id, prompt_hash, seed, engine_id, engine_version,
                      kind="", w=0, h=0) -> str:
    """The dispatch dedup key (PASS-IMG MUST-FIX #5). A change in ANY field ->
    new key -> regen -> new content hash -> B's mesh cache invalidates.

    Still-spine ST-3 (pass-02 Gem-1): the key gains ``kind`` + ``w`` + ``h``
    so a landscape scene still and a portrait of the same subject can never
    collide, and a dim change regenerates."""
    return _content_hash([
        str(role), str(object_id), str(prompt_hash), int(seed),
        str(engine_id), str(engine_version),
        str(kind), int(w or 0), int(h or 0),
    ])


def resolve_object_seed(seed_cfg, object_id, prompt_hash) -> int:
    """Per-object seed under the V-7 request-hash scheme: ``mode=request_hash``
    (the ImageDirector default) derives a deterministic seed from
    ``request_seed + object_id + prompt_hash`` so every object gets its own
    seed while the whole episode stays reproducible; ``mode=fixed`` returns
    ``request_seed`` verbatim. Pure."""
    cfg = seed_cfg if isinstance(seed_cfg, dict) else {}
    base = int(cfg.get("request_seed") or 0)
    if str(cfg.get("mode") or "request_hash") != "request_hash":
        return base
    digest = hashlib.sha256(
        f"{base}:{object_id}:{prompt_hash}".encode()).hexdigest()
    return int(digest[:8], 16)


#: ImageDirector slot per object ROLE (still-spine ST-3: the slots finally
#: honored -- announcer stills render on the announcer slot, music/open stills
#: on the music slot; characters + scene b-roll + background on other_beats).
_ROLE_TO_IMAGE_SLOT = {
    "announcer_visual": "announcer_image_model",
    "music_visual": "music_image_model",
}


def resolve_engine_for_role(image_policy, role):
    """``(engine_id, slot_name, fallback_used)`` for an object role. An empty
    named slot falls back to ``other_beats_image_model`` (the caller warns
    LOUD); never raises."""
    models = (image_policy or {}).get("image_models") or {}
    slot = _ROLE_TO_IMAGE_SLOT.get(str(role or ""), "other_beats_image_model")

    def _eid(entry):
        if isinstance(entry, dict):
            return str(entry.get("engine_id") or "")
        return str(entry or "")

    engine_id = _eid(models.get(slot))
    if engine_id:
        return engine_id, slot, False
    fb = _eid(models.get("other_beats_image_model"))
    return fb, slot, slot != "other_beats_image_model" and bool(fb)


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


def _coerce_pixels(result, *, min_bytes: int = _MIN_PNG_BYTES,
                   wait_attempts: int = 40, wait_sleep_s: float = 0.05):
    """Decoded uint8 pixel array from a gen_fn result.

    A sidecar returns a ``.png`` PATH (disk-path handoff, never a tensor); an
    in-process gen returns a numpy uint8 array (a comfy IMAGE tensor must be
    ``.clone()``d + converted by the caller BEFORE here). Lazy PIL/numpy import.

    On the PATH branch the file is read only AFTER :func:`wait_for_file_ready`
    confirms it is fully flushed (PASS-PM C1 0-byte/truncated handoff race); a
    never-ready file raises :class:`ImageHandoffTimeout`.
    """
    if isinstance(result, str):
        wait_for_file_ready(result, min_bytes=min_bytes,
                            attempts=wait_attempts, sleep_s=wait_sleep_s)
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


def _reresolve_episode_stills_dir(ep, ep_dir, warnings):
    """Rename-proof the EPISODE stills dir (operator ticket 2026-06-11; the
    OTR_MasterAudioMux ``_reresolve_master_audio`` contract applied to the
    dispatcher).

    The episode dir starts life as ``episodes/pending_<ts>/``; SignalLostVideo
    renames it to the final title slug once the audio title is finalized. The
    dispatcher's wired ``episode_id`` was captured BEFORE that rename, so a
    post-rename dispatch would re-CREATE the stale ``pending_*`` dir and strand
    every still outside the real episode folder. Re-resolve via the newest
    on-disk ledger (the same durable-ledger contract the mux + ShotLock use):

    * non-``pending_*`` id, or the pending dir still exists (rename has not
      happened yet) -> unchanged;
    * pending dir GONE -> the newest ledger's episode dir is the rename target;
      re-key there with a LOUD log + warning. Never silent;
    * any probe failure -> unchanged (the shipped behavior).

    Returns ``(ep_dir, ep)``. ``OTR_TEST_MODE=1`` skips the disk scan (mirrors
    the mux; the helper itself stays directly testable with the env cleared).
    """
    try:
        if os.environ.get("OTR_TEST_MODE") == "1":
            return ep_dir, ep
        if not str(ep).startswith("pending_"):
            return ep_dir, ep
        episode_dir = os.path.dirname(str(ep_dir))          # .../episodes/<ep>
        if os.path.isdir(episode_dir):
            return ep_dir, ep                               # rename not yet happened
        from pathlib import Path
        from ._otr_ledger import find_most_recent_ledger
        episodes_root = os.path.dirname(episode_dir)        # .../episodes
        p = find_most_recent_ledger([Path(episodes_root)])
        if not p:
            return ep_dir, ep
        new_episode_dir = Path(p).parent.parent             # <root>/<ep>/audio/x_ledger.json
        if new_episode_dir.is_dir() and new_episode_dir.parent == Path(episodes_root):
            new_ep = new_episode_dir.name
            log.warning(
                "[OTR_ImageGenDispatcher] LOUD re-resolve: episode_id %r is a "
                "stale pending id (its dir was renamed after capture); stills "
                "re-keyed to the newest ledger's episode dir %r (same episode, "
                "post-rename name).", ep, new_ep)
            warnings.append(
                f"episode stills dir re-resolved: stale {ep!r} -> {new_ep!r} "
                f"(title rename happened before image dispatch; LOUD)")
            return str(new_episode_dir / "stills"), new_ep
    except Exception as exc:  # noqa: BLE001 -- never block dispatch on the probe
        log.warning(
            "[OTR_ImageGenDispatcher] episode stills re-resolve skipped: %s", exc)
    return ep_dir, ep


def _materialize_episode_copy(src_path, ep_dir, object_id, content_hash):
    """Copy a still into the EPISODE stills dir (idempotent; readable name
    ``{object_id}_{hash12}.png``). Returns the episode-local path. Lazy
    shutil; raises on a failed copy (the caller warns + falls back)."""
    import shutil
    os.makedirs(str(ep_dir), exist_ok=True)
    dst = os.path.join(str(ep_dir), f"{object_id}_{str(content_hash)[:12]}.png")
    if not os.path.exists(dst):
        shutil.copyfile(str(src_path), dst)
    return dst


def dispatch_images(ledger: dict, image_policy: dict, image_prompts: dict, *,
                    gen_fn=None, output_dir=None, lockdir=None, lease_timeout_s=120.0,
                    handoff_min_bytes: int = _MIN_PNG_BYTES,
                    handoff_wait_attempts: int = 40, handoff_wait_sleep_s: float = 0.05,
                    episode_id: str = ""):
    """Generate (or cache-reuse) every image OBJECT (portraits + scene
    stills); stamp the ledger.

    ``image_prompts`` is the ONE versioned ``{"version": 1, "objects": [...]}``
    payload from ``derive_image_prompts`` (still-spine ST-2 / pass-02 item 1:
    objects only -- never bare char_id maps; no dual-schema shims).

    ``gen_fn(request: dict) -> (numpy uint8 array | .png path)`` is the only GPU
    step (injected in tests; the real Flux path on the operator smoke). Returns
    ``(patched_ledger, image_done, report, warnings)``. Never raises on a normal
    miss; ``assert_usable`` failures are recorded as warnings + skipped
    (fail-closed: that object simply has no image, never a silent wrong engine).

    EXCEPTION -- the downstream 3D HALT (3D plan section 3): a policy whose
    ``locked_3d_slots`` carry ``granularity == per_beat`` RAISES before any
    object is dispatched. That combination means a requires_mesh_portrait
    video engine would get a fresh portrait (= a mesh REBUILD) per beat; the
    ImageDirector already fails closed on it, so reaching here means the
    policy was hand-crafted or stale -- a malformed POLICY, not a normal miss.
    """
    locked_3d = set((image_policy or {}).get("locked_3d_slots") or [])
    gran_by_slot = (image_policy or {}).get("granularity") or {}
    viol = sorted(s for s in locked_3d if gran_by_slot.get(s) == "per_beat")
    if viol:
        raise ValueError(
            "OTR_ImageGenDispatcher HALT: 3D-locked slot(s) %s carry "
            "granularity=per_beat (mesh-rebuild-per-beat). The image policy "
            "is malformed/stale -- re-run OTR_ImageDirector (it fails closed "
            "on this) instead of hand-editing image_policy_json." % viol)
    warnings: list = []
    report: list = []
    cast = ledger.get("cast") if isinstance(ledger.get("cast"), list) else []
    images_section = ledger.get("images") if isinstance(ledger.get("images"), dict) else {}
    cache_index = dict(images_section.get("cache_index") or {})
    images = list(images_section.get("images") or [])
    seed_cfg = (image_policy or {}).get("seed") or {}

    # Episode keying (still-spine ST-3 / W3): every still materializes into
    # episodes/<ep>/stills/ so the episode folder is self-contained. The id
    # comes from the wired input first, then the ledger; an unkeyed dispatch
    # is LOUD, never silent.
    ep = str(episode_id
             or (ledger or {}).get("episode_id")
             or ((ledger or {}).get("meta") or {}).get("episode_id") or "")
    if not ep:
        ep = "unkeyed_episode"
        warnings.append(
            "episode_id missing (input unwired AND not in the ledger); "
            "stills materialize under episodes/unkeyed_episode/ (LOUD)")
    ep_dir = str(_pl.episode_stills_dir(ep, output_dir=output_dir))
    # Operator ticket 2026-06-11: the title rename (pending_* -> final slug)
    # can land BEFORE image dispatch; re-key the stills dir to the renamed
    # episode dir via the newest on-disk ledger (mux-style re-resolve, LOUD).
    ep_dir, ep = _reresolve_episode_stills_dir(ep, ep_dir, warnings)
    ep_rows: list = []

    rev = int(images_section.get("image_revision") or 0) + 1
    made = 0
    reused = 0
    # Synthetic non-cast subjects (the ANNOUNCER radio-style portrait + every
    # scene still): they are recorded in ledger['images'] below -- the index
    # the render path resolves init_image from -- but there is no cast row to
    # stamp, so stamp_portrait must not fail-closed on them (cast stays
    # CastLock's frozen authority; never added to here).
    cast_ids = {str(c.get("char_id") or "") for c in cast if isinstance(c, dict)}
    objects = (image_prompts or {}).get("objects") or []
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("object_id") or "")
        kind = str(obj.get("kind") or "portrait")
        role = str(obj.get("role") or "character_video")
        char_id = str(obj.get("char_id") or "")
        beat_id = str(obj.get("beat_id") or "")
        prompt = str(obj.get("prompt") or "")
        prompt_hash = str(obj.get("prompt_hash") or "")
        obj_w = int(obj.get("w") or 0)
        obj_h = int(obj.get("h") or 0)
        if not oid:
            continue
        # Slot resolution per OBJECT role (ST-3: the ImageDirector slots
        # finally honored); empty named slot -> other_beats fallback, LOUD.
        engine_id, slot, fell_back = resolve_engine_for_role(image_policy, role)
        if fell_back:
            warnings.append(
                f"{oid}: image slot {slot} empty; fell back to "
                "other_beats_image_model (LOUD)")
        if not engine_id:
            warnings.append(f"{oid}: no image engine selected; skipped (fail-closed)")
            continue
        try:
            _assert_not_path(prompt)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        seed = resolve_object_seed(seed_cfg, oid, prompt_hash)
        eng_version = str(getattr(_safe_engine(engine_id), "engine_version", "1"))
        key = request_cache_key(role, oid, prompt_hash, seed, engine_id,
                                eng_version, kind=kind, w=obj_w, h=obj_h)
        if key in cache_index:
            # Cache HIT (pass-02 Gem-2): the hit must STILL materialize into
            # the CURRENT episode's stills/ + append a fresh ledger row --
            # the old `continue` silently left episode folders missing every
            # reused still. A stale index entry (file gone) degrades to a
            # fresh render below, LOUD.
            hit_id = cache_index[key]
            src = ""
            ref_row = None
            for im in images:
                if isinstance(im, dict) and im.get("image_id") == hit_id:
                    ref_row = im
                    src = str(im.get("pool_path") or im.get("path") or "")
                    break
            if src and os.path.exists(src):
                try:
                    dst = _materialize_episode_copy(
                        src, ep_dir, oid,
                        (ref_row or {}).get("portrait_content_hash") or "x")
                except OSError as exc:
                    warnings.append(
                        f"{oid}: episode materialization of cache hit failed "
                        f"({exc}); row points at the pool copy (LOUD)")
                    dst = src
                fresh = dict(ref_row or {})
                fresh.update({
                    "image_id": hit_id, "object_id": oid, "kind": kind,
                    "role": role, "path": dst, "pool_path": src,
                    "provenance": {"source": "cache_hit"},
                })
                if char_id:
                    fresh["char_id"] = char_id
                if beat_id:
                    fresh["beat_id"] = beat_id
                images.append(fresh)
                ep_rows.append(fresh)
                reused += 1
                report.append(f"{oid}: cache HIT ({hit_id[:12]}) -> "
                              f"{os.path.basename(dst)}")
                continue
            warnings.append(
                f"{oid}: cache index entry {hit_id[:12]} has no on-disk file; "
                "regenerating (LOUD)")
        # cache miss -> assert usable (fail-closed) -> lease -> generate -> stamp
        try:
            _ireg.assert_usable(engine_id, role)
        except Exception as exc:  # noqa: BLE001  (EngineUnusable et al.)
            warnings.append(f"{oid}: engine '{engine_id}' not usable for {role} ({exc}); skipped")
            continue
        if gen_fn is None:
            warnings.append(f"{oid}: no gen_fn (GPU render is the operator smoke); skipped on CPU")
            continue
        request = {
            "request_id": key, "role": role, "object_id": oid,
            "kind": kind, "char_id": char_id, "beat_id": beat_id,
            "engine_id": engine_id, "engine_version": eng_version,
            "prompt": prompt, "prompt_hash": prompt_hash, "seed": seed,
            # w/h end-to-end (pass-02 Gem-1): the engine call reads
            # width/height (flux_gen1._flux_params request precedence), so
            # landscape scene stills are REAL, not env defaults.
            "w": obj_w, "h": obj_h,
            "width": obj_w or None, "height": obj_h or None,
        }
        lease = None
        try:
            lease = _lease.acquire(timeout_s=lease_timeout_s, lockdir=lockdir)
            pixels = _coerce_pixels(
                gen_fn(request), min_bytes=handoff_min_bytes,
                wait_attempts=handoff_wait_attempts, wait_sleep_s=handoff_wait_sleep_s,
            )
            content_hash = _pl.compute_portrait_hash(pixels)
            # portraits stamp the cast row (require_cast_entry for real cast);
            # scene stills only write the content-addressed file (no cast row
            # could ever exist for a beat).
            path = _pl.stamp_portrait(
                ledger, oid, pixels, output_dir=output_dir,
                require_cast_entry=(kind == "portrait" and oid in cast_ids))
        except _lease.LeaseTimeout as exc:
            warnings.append(f"{oid}: GPU lease timeout ({exc}); skipped")
            continue
        except ImageHandoffTimeout as exc:
            warnings.append(f"{oid}: image handoff not ready ({exc}); skipped")
            continue
        except Exception as exc:  # noqa: BLE001 -- any render failure -> floor
            # The image GATE never crashes the episode: a wrapper-node-missing /
            # CUDA-OOM / decode failure means this object simply has no image,
            # so the render path degrades to the radio floor LOUD downstream.
            warnings.append(
                f"{oid}: image render failed ({type(exc).__name__}: {exc}); "
                "skipped (radio floor will be used)")
            continue
        finally:
            if lease is not None:
                _lease.release(lease)
        # post-generation residency confirm (best-effort; gates the C->A handoff).
        if not _lease.wait_until_below_mb(15000, attempts=3, sleep_s=0.0):
            log.info("[OTR_ImageGenDispatcher] NVML re-probe inconclusive (CPU/no-NVML or busy)")
        image_id = f"img_{oid}_{content_hash[:12]}"
        # Materialize the fresh render into the EPISODE stills dir (ST-3/W3);
        # the ledger row's `path` is the EPISODE-LOCAL copy, `pool_path` the
        # content-addressed global cache file.
        try:
            ep_path = _materialize_episode_copy(str(path), ep_dir, oid,
                                                content_hash)
        except OSError as exc:
            warnings.append(
                f"{oid}: episode materialization failed ({exc}); row points "
                "at the pool copy (LOUD)")
            ep_path = str(path)
        row = {
            "image_id": image_id, "role": role, "object_id": oid,
            "kind": kind,
            "path": ep_path, "pool_path": str(path),
            "engine_id": engine_id, "engine_version": eng_version,
            "request_hash": key, "portrait_content_hash": content_hash,
            "content_hash": content_hash, "w": obj_w, "h": obj_h,
            "prompt_hash": prompt_hash, "provenance": {"source": obj.get("source", "")},
        }
        if char_id:
            row["char_id"] = char_id
        if beat_id:
            row["beat_id"] = beat_id
        images.append(row)
        ep_rows.append(row)
        cache_index[key] = image_id
        made += 1
        report.append(f"{oid}: generated -> {os.path.basename(ep_path)}")

    ledger["images"] = {
        "image_revision": rev,
        "episode_id": ep,
        "granularity_by_role": (image_policy or {}).get("granularity") or {},
        "images": images,
        "cache_index": cache_index,
        "warnings": warnings,
    }
    # stills_manifest.json beside the episode stills (ST-3/W3): the durable
    # per-episode index of every still this dispatch materialized. Fail-soft.
    if ep_rows:
        try:
            os.makedirs(ep_dir, exist_ok=True)
            manifest = {
                "episode_id": ep, "image_revision": rev,
                "stills": [{
                    "object_id": r.get("object_id"),
                    "kind": r.get("kind"),
                    "role": r.get("role"),
                    "char_id": r.get("char_id", ""),
                    "beat_id": r.get("beat_id", ""),
                    "path": r.get("path"),
                    "content_hash": (r.get("content_hash")
                                     or r.get("portrait_content_hash")),
                    "prompt_hash": r.get("prompt_hash"),
                    "provenance": r.get("provenance"),
                } for r in ep_rows],
            }
            with open(os.path.join(ep_dir, "stills_manifest.json"), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=True, indent=2)
        except OSError as exc:
            warnings.append(f"stills_manifest.json write failed ({exc}); "
                            "episode stills are on disk but unindexed (LOUD)")
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


def _inprocess_gen_fn(request):
    """The in-graph image gen_fn: resolve the request's image engine from the
    registry and run its in-process ``render_image`` (the real GPU step on the
    live server). Model-agnostic -- any registered image engine plugs in with no
    dispatcher edit. Cold-import clean (V-12): the registry is dep-free and the
    engine lazy-imports torch / comfy only when it actually renders, so importing
    this module never pulls a model framework. Returns a decoded uint8 (H,W,3)
    pixel array (in-process) or a ``.png`` PATH (a future cu128 sidecar); the
    dispatcher content-addresses + stamps it. A render failure RAISES -- the
    dispatcher catches it fail-closed so the episode falls to the radio floor."""
    eng = _ireg.get_engine(request.get("engine_id"))
    eng = eng() if isinstance(eng, type) else eng
    prepared = None
    prep = getattr(eng, "prepare", None)
    if callable(prep):
        prepared = prep(None, None, None)
    try:
        return eng.render_image(request, prepared)
    finally:
        td = getattr(eng, "teardown", None)
        if callable(td):
            try:
                td(prepared)
            except Exception:  # noqa: BLE001 -- teardown is best-effort
                pass


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
                    "tooltip": "OTR_MetaBriefImagePromptGen output: the versioned {\"objects\":[...]} payload (portraits + scene stills).",
                }),
            },
            "optional": {
                "gate_in": ("STRING", {
                    "multiline": True, "default": "", "forceInput": True,
                    "tooltip": "Optional ordering signal (e.g. audio_done); opaque STRING.",
                }),
                "episode_id": ("STRING", {
                    "default": "", "forceInput": True,
                    "tooltip": (
                        "Episode id (still-spine ST-3/DS-3; wired in the saved "
                        "json). Every still materializes into "
                        "episodes/<episode_id>/stills/ + stills_manifest.json. "
                        "Falls back to the ledger's episode_id; an unkeyed "
                        "dispatch is LOUD."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def dispatch(self, script_json, image_policy_json="{}", image_prompts_json="{}",
                 gate_in="", episode_id=""):
        led = self._loads(script_json, {})
        policy = self._loads(image_policy_json, {})
        prompts = self._loads(image_prompts_json, {})
        # The in-graph node mints REAL portraits via the request's image engine
        # (in-process Flux gen-1) under the AS-3 GPU-residency lease. On a box
        # without the GPU / wrapper nodes / checkpoint the render fails closed and
        # the dispatcher degrades that object to the radio floor LOUD (never a
        # crash). The CPU platform tests bypass this by calling dispatch_images()
        # directly with an injected gen_fn.
        led, image_done, report, _warn = dispatch_images(
            led, policy, prompts, gen_fn=_inprocess_gen_fn,
            episode_id=str(episode_id or ""),
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
