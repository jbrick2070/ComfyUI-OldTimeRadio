"""OTR_ImageGenDispatcher -- cache-checked image generation + ledger write-back (C1).

The terminal C node: for each image object it (1) composes the request cache key
``(role, object_id, prompt_hash, seed, engine_id, engine_version)``, (2) reuses
the existing image on a cache hit, (3) otherwise ``assert_usable`` (fail-closed,
NO silent Flux swap) -> takes the SHARED AS-3 GPU-residency lease for local
engines only -> generates (in-process Flux gen-1, Partner/cloud adapter, or a
cu128 sidecar; injected ``gen_fn`` in tests) -> writes
the still CONTENT-ADDRESSED at ``output/otr/stills/{portrait_content_hash}.png``
(AS-5, never-overwrite) -> stamps the ledger -> releases the lease + re-probes
NVML for local engines -> emits the ``image_done`` STRING token (mirrors
``audio_done``).

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
from ._otr_shared import role_slots as _role_slots
from ._otr_story_brief_helpers import (
    append_visual_safety_clause,
    visual_safety_negative,
)
from ._otr_image_engines import registry as _ireg

#: Smallest plausible real PNG (8-byte signature + IHDR + IDAT + IEND). Anything
#: smaller off the cross-process disk handoff is a 0-byte / truncated write,
#: never a finished render.
_MIN_PNG_BYTES = 67


def _prompt_content_hash(prompt: str) -> str:
    return hashlib.sha256(
        json.dumps(
            str(prompt or ""),
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


class ImageHandoffTimeout(RuntimeError):
    """A sidecar-written image never became readable within the bounded retries.

    A cu128 image sidecar writes its ``.png`` to disk and hands back the PATH; if
    the main venv decodes it before the bytes are flushed it sees a 0-byte or
    truncated file (PASS-PM C1 handoff race). The dispatcher treats this as a
    fail-closed miss (warn + skip that object) -- never a silent bad image.
    """


class ImageRenderError(RuntimeError):
    """The selected image engine could not produce the requested still.

    NO FALLBACKS (operator 2026-06-18: "hard fail if it can't use the req
    model"): a selected-but-unusable engine (opt-in flag off / weights absent /
    unbuilt) OR a render failure (OOM / missing node / decode / lease/handoff
    timeout) is TERMINAL -- the episode fails LOUD instead of skipping the object
    or silently substituting flux / degrading to the radio floor. Mirrors the
    video render_driver.RenderError contract.
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


def resolve_object_seed(seed_cfg, object_id, prompt_hash, kind="") -> int:
    """Per-object seed under the V-7 request-hash scheme: ``mode=request_hash``
    (the ImageDirector default) derives a deterministic seed from
    ``request_seed + object_id + prompt_hash`` so every object gets its own
    seed while the whole episode stays reproducible; ``mode=fixed`` returns
    ``request_seed`` verbatim. Pure.

    BUG-411 restore: the radio BOOKEND (``kind == "scene_open"``) renders with a
    FIXED deterministic seed (the 6/5 ``radio_bookend_seed=4242`` widget,
    env-overridable via ``OTR_RADIO_BOOKEND_SEED``) so the opening radio still is
    reproducible run-to-run independent of the request hash -- exactly the 6/5
    behavior the rewrite lost."""
    # The radio BOOKEND still (scene_open) AND the radio-HOST FACE object
    # (2026-07-01 brief-driven radio-host; object_id "radio_host_portrait",
    # matches otr_meta_brief_image_prompt.RADIO_HOST_PORTRAIT_ID) both render
    # with the FIXED deterministic bookend seed so the host face is reproducible
    # run-to-run and open/inter/close share ONE canonical face.
    _oid = str(object_id or "")
    if (str(kind or "") == "scene_open" or _oid == "radio_host_portrait"
            or _oid.endswith("_radio_face_169")):   # ltx talking radio-face
        try:
            return int(os.environ.get("OTR_RADIO_BOOKEND_SEED", 4242))
        except (TypeError, ValueError):
            return 4242
    cfg = seed_cfg if isinstance(seed_cfg, dict) else {}
    base = int(cfg.get("request_seed") or 0)
    if str(cfg.get("mode") or "request_hash") != "request_hash":
        return base
    digest = hashlib.sha256(
        f"{base}:{object_id}:{prompt_hash}".encode()).hexdigest()
    return int(digest[:8], 16)


#: ImageDirector slot per object ROLE (still-spine ST-3: the slots finally
#: honored -- announcer stills render on the announcer slot, music/open stills
#: on the music slot; character portraits + character-associated scene stills on
#: the character slot). Renamed 2026-07-03: the old catch-all image slot
#: is now character_image_model (retired_role_a + background roles were ripped
#: 2026-07-01, so it serves only the character_video lane).
_ROLE_TO_IMAGE_SLOT = {
    "announcer_visual": "announcer_image_model",
    "music_visual": "music_image_model",
}


def resolve_engine_for_role(image_policy, role):
    """``(engine_id, slot_name, fallback_used)`` for an object role.

    NO-FALLBACK (operator 2026-07-03), E8-precise: a NAMED role (announcer_visual /
    music_visual) whose dedicated slot is PRESENT in the policy but explicitly
    EMPTY is an operator config error -> FAIL LOUD; never silently substitute the
    general character model for a slot the operator deliberately exposed and left
    blank. An ABSENT dedicated slot is different -- the role simply has no special
    model configured, so it uses the character slot (a DEFAULT, not a silent model
    swap; still flagged ``fallback_used=True`` for observability)."""
    models = (image_policy or {}).get("image_models") or {}
    slot = _ROLE_TO_IMAGE_SLOT.get(str(role or ""), "character_image_model")

    def _eid(entry):
        if isinstance(entry, dict):
            return str(entry.get("engine_id") or "")
        return str(entry or "")

    if slot != "character_image_model" and slot in models and not _eid(models.get(slot)):
        raise ValueError(
            f"image slot {slot!r} for role {role!r} is PRESENT but EMPTY -- pick a "
            f"model for it (or remove the slot to use the general character_image_"
            f"model). NO silent fallback (no-fallback rip)."
        )

    engine_id = _eid(models.get(slot))
    if engine_id:
        return engine_id, slot, False
    fb = _eid(models.get("character_image_model"))
    return fb, slot, slot != "character_image_model" and bool(fb)


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


def _reresolve_episode_stills_dir(ep, ep_dir, warnings, ledger=None):
    """Rename-proof the EPISODE stills dir (operator ticket 2026-06-11; the
    OTR_MasterAudioMux ``_reresolve_master_audio`` contract applied to the
    dispatcher).

    The episode dir starts life as ``episodes/pending_<ts>/``; SignalLostVideo
    renames it to the final title slug once the audio title is finalized. The
    dispatcher's wired ``episode_id`` was captured BEFORE that rename, so a
    post-rename dispatch would re-CREATE the stale ``pending_*`` dir and strand
    every still outside the real episode folder. Re-resolve via the active
    in-flight ledger (the same durable-ledger contract ShotLock uses), never a
    newest-mtime sibling guess:

    * non-``pending_*`` id, or the pending dir still exists (rename has not
      happened yet) -> unchanged;
    * pending dir GONE -> the active durable ledger's episode dir is accepted
      only after immutable freeze identity matches the wire ledger;
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
        from . import _otr_ledger as _OTRL
        episodes_root = os.path.dirname(episode_dir)        # .../episodes
        p = _OTRL.in_flight_ledger_path()
        if not p:
            return ep_dir, ep
        p = Path(p)
        new_episode_dir = Path(p).parent.parent             # <root>/<ep>/audio/x_ledger.json
        if not (
            new_episode_dir.is_dir()
            and new_episode_dir.parent == Path(episodes_root)
        ):
            return ep_dir, ep

        disk = _OTRL.load_ledger_safe(p)
        wire_meta = (
            ledger.get("meta") if isinstance(ledger, dict)
            and isinstance(ledger.get("meta"), dict) else {}
        )
        disk_meta = (
            disk.get("meta") if isinstance(disk, dict)
            and isinstance(disk.get("meta"), dict) else {}
        )
        wire_freeze = str(wire_meta.get("freeze_timestamp") or "").strip()
        disk_freeze = str(disk_meta.get("freeze_timestamp") or "").strip()
        if not wire_freeze or not disk_freeze or wire_freeze != disk_freeze:
            message = (
                "episode stills re-resolve REJECTED: active durable ledger "
                "does not share the wire freeze receipt"
            )
            log.warning("[OTR_ImageGenDispatcher] %s", message)
            warnings.append(message)
            return ep_dir, ep

        if isinstance(disk, dict):
            new_ep = new_episode_dir.name
            if str(disk.get("episode_id") or "").strip() != new_ep:
                return ep_dir, ep
            log.warning(
                "[OTR_ImageGenDispatcher] LOUD re-resolve: episode_id %r is a "
                "stale pending id (its dir was renamed after capture); stills "
                "re-keyed to the active ledger's episode dir %r (same episode, "
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


#: object role -> the OTR_VideoDirector video slot that owns it. Route-A: the
#: ONE shared per-role map (nodes/_otr_shared/role_slots.py); aliased here so any
#: importer of this name keeps working but the rule lives in one place.
_ROLE_TO_VIDEO_SLOT = _role_slots.ROLE_TO_VIDEO_SLOT


def engine_consumes_still(eng) -> bool:
    """Coverage architecture (2026-06-18) -- the ONE capability the still dispatcher
    keys on, so coverage (which image feeds which video) is decided in a single
    place with NO per-(image,video) whitelist. A video/3D lane consumes the role's
    SELECTED image still iff:
      1. it explicitly declares ``accepts_still`` (the capability wins -- every real
         motion lane inherits ``accepts_still=True`` from MotionEngineBase, so a new
         engine accepts the chosen image automatically; floors / audio-only lanes
         override to False), OR
      2. (dual-read migration for engines that have not declared the flag yet) it
         lists ``init_image`` in ``required_inputs`` -- required_inputs wins for the
         existing names so humo / wan / ltx_av_talk / still_parallax / mesh_stage /
         the 3D talkers keep working unchanged.
    Pure attr read (cold-import clean)."""
    cap = getattr(eng, "accepts_still", None)
    if cap is not None:
        return bool(cap)
    return "init_image" in tuple(getattr(eng, "required_inputs", ()) or ())


def _effective_engine_after_force_map(role: str, eng_id: str) -> str:
    """Resolve the effective VIDEO engine for ``role`` AFTER applying
    ``OTR_FORCE_ENGINE_MAP`` -- the SAME all-one-engine override that
    ``render_driver.apply_engine_override`` applies at render time (operator
    2026-07-01). The still dispatcher runs BEFORE that render-time override, so
    without this it resolves the stale node-87 pick (e.g. humo) and mints a full
    Flux still that the forced no-still visualizer (e.g. ``*=viz_mxc_mandala``)
    then ignores -- a wasted ~10 min image-gen pass per episode. Unset env or any
    parse error -> ``eng_id`` unchanged (byte-identical to a normal run, where the
    env is unset). Pure except for the env read + a lazy import."""
    spec = os.environ.get("OTR_FORCE_ENGINE_MAP", "").strip()
    if not spec:
        return eng_id
    try:
        from ._otr_video_engines.render_driver import parse_engine_override
        mapping = parse_engine_override(spec)
    except Exception:  # noqa: BLE001 - never block dispatch on a bad override
        return eng_id
    forced = mapping.get(role) or mapping.get("*")
    return forced or eng_id


def _effective_video_engine_for_role(role: str, eng_id: str) -> str:
    """Resolve the VIDEO engine the render phase will actually use for ``role``.

    The image phase runs before ShotLock/video dispatch, so it cannot read the
    final mutated shot row. It still has to mint exactly the stills that final
    render will require. Mirror the render order:

    1. ``OTR_FORCE_ENGINE_MAP`` rewrites planned shot engines.
    2. ``render_driver._enforce_radio_is_host`` redirects announcer/music local
       HuMo-family bookends to ``ltx_audio_in`` when ``OTR_ENABLE_HUMO_HOSTS`` is
       off. Partner/cloud engines stay cloud.

    Unknown engines or import failures keep the input id (fail-safe: mint the
    still rather than quietly skipping an asset that render might need)."""
    role_s = str(role or "")
    eff = _effective_engine_after_force_map(role_s, str(eng_id or ""))
    if os.environ.get("OTR_ENABLE_HUMO_HOSTS", "0") == "1":
        return eff
    if role_s not in ("announcer_visual", "music_visual"):
        return eff
    try:
        from ._otr_video_engines.render_driver import _radio_is_host_redirect_applies
        if _radio_is_host_redirect_applies(eff):
            return "ltx_audio_in"
    except Exception:  # noqa: BLE001 -- fail-safe: keep the selected id
        return eff
    return eff


def still_consumer_capability(image_policy: dict, role: str):
    """``True`` / ``False`` / ``None`` for one role's init-image consumer.

    ``True`` and ``False`` are proven only after resolving the selected engine
    through the same force-map and radio-host redirect the render path uses.
    ``None`` means no proof: a missing/partial policy, unknown role, empty slot,
    or unresolvable effective engine.  Keeping that third state prevents an
    unrelated custom slot from erasing a proven character consumer while still
    forbidding generation for an unproven object role.
    """
    if not isinstance(image_policy, dict):
        return None
    vmodels = image_policy.get("video_models")
    role_s = str(role or "")
    if not isinstance(vmodels, dict) or role_s not in _role_slots.ROLE_TO_VIDEO_SLOT:
        return None
    try:
        eng_id = _role_slots.engine_id_for_role(vmodels, role_s)
        if not eng_id:
            return None
        eng_id = _effective_video_engine_for_role(role_s, eng_id)
        # Register built-in lightweight adapters before the query; this remains
        # lazy so importing the dispatcher stays cold-import clean.
        from . import _otr_video_engines  # noqa: F401
        from ._otr_video_engines import registry as _vreg
        if not _vreg.is_registered(eng_id):
            return None
        return engine_consumes_still(_vreg.get_engine(eng_id))
    except Exception:  # noqa: BLE001 -- absence of proof is a real third state
        return None


def _still_needed_for_role(image_policy: dict, role: str) -> bool:
    """Legacy boolean view of :func:`still_consumer_capability`.

    Direct legacy callers historically fail-safe toward keeping a still when
    configuration is unknown.  The terminal dispatcher now consumes the
    tri-state helper directly and refuses to render without proof instead.
    """
    capability = still_consumer_capability(image_policy, role)
    if capability is None:
        log.warning("[OTR_ImageGenDispatcher] _still_needed_for_role: cannot prove "
                    "an effective init-image consumer for role=%s; retaining the "
                    "legacy boolean fail-safe", role)
        return True
    return capability


def still_consumer_capabilities(image_policy: dict):
    """Return ``{role: True|False|None}`` or ``None`` for malformed policy."""
    if not isinstance(image_policy, dict) or not isinstance(
            image_policy.get("video_models"), dict):
        return None
    return {
        role: still_consumer_capability(image_policy, role)
        for role in _role_slots.ROLE_TO_VIDEO_SLOT
    }


def roles_requiring_stills(image_policy: dict):
    """Return conclusively still-consuming roles, or ``None`` if any role is unknown.

    The compact set is ideal for an all-procedural upstream bypass.  Callers
    that need per-object mixed-policy behavior should use
    :func:`still_consumer_capabilities` and preserve ``None`` as uncertainty.
    """
    capabilities = still_consumer_capabilities(image_policy)
    if capabilities is None or any(value is None for value in capabilities.values()):
        return None
    return frozenset(role for role, value in capabilities.items() if value)


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
    step (injected in tests; the wired node passes the in-process engine path).
    Returns ``(patched_ledger, image_done, report, warnings)``. Fail-loud
    contract: an unusable selected engine raises ``ImageRenderError`` (NO
    FALLBACKS, operator 2026-06-18), and a pending target reached with
    ``gen_fn=None`` raises ``ImageRenderError`` too (S0 portability,
    2026-07-10 -- the old "skipped on CPU" warning let an episode "succeed"
    with missing stills).

    EXCEPTION -- the downstream 3D HALT (3D plan section 3): a policy whose
    ``locked_3d_slots`` carry ``granularity == per_beat`` RAISES before any
    object is dispatched. That combination means a requires_mesh_portrait
    video engine would get a fresh portrait (= a mesh REBUILD) per beat; the
    ImageDirector already fails closed on it, so reaching here means the
    policy was hand-crafted or stale -- a malformed POLICY, not a normal miss.
    """
    # S4 platform-portability (2026-07-10): a NON-EMPTY policy must be
    # version 2. A v1 policy means a stale OTR_ImageDirector emitted it --
    # fail LOUD before any LLM/API/render work burns on it.
    if image_policy and int((image_policy or {}).get("policy_version") or 0) != 2:
        raise ValueError(
            "OTR_ImageGenDispatcher: image_policy carries policy_version="
            f"{(image_policy or {}).get('policy_version')!r}; expected 2. "
            "Re-run OTR_ImageDirector (stale/hand-crafted policy).")
    # Real host facts + the episode device/dtype policy for ADAPTER-level
    # usability (S4): the adapter-side enforcement protocol existed since
    # the registry protocol but was never called on the image path.
    from ._otr_shared.host_caps import build_host_caps
    _host_caps = build_host_caps()
    _adapter_profile = {
        "policy_version": 2,
        "device_policy": str((image_policy or {}).get("device_policy") or "cuda"),
        "dtype_policy": str((image_policy or {}).get("dtype_policy") or "fp8_ok"),
    }
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
    objects = (image_prompts or {}).get("objects") or []
    still_capabilities = still_consumer_capabilities(image_policy)
    if objects and still_capabilities is None:
        raise ImageRenderError(
            "OTR_ImageGenDispatcher cannot prove an image consumer: image_policy "
            "must carry a video_models map before any still is rendered. "
            "Refusing image generation rather than assuming a consumer."
        )
    cast = ledger.get("cast") if isinstance(ledger.get("cast"), list) else []
    images_section = ledger.get("images") if isinstance(ledger.get("images"), dict) else {}
    cache_index = dict(images_section.get("cache_index") or {})
    images = list(images_section.get("images") or [])
    seed_cfg = (image_policy or {}).get("seed") or {}
    # Credits: per-role image-engine histogram (role -> {engine_id: count}),
    # stamped into meta after the loop so the dossier can show image model per
    # slot (the ledger['images'] section is dropped before the credits read).
    img_hist: dict = {}

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
    # episode dir via the active in-flight ledger (mux-style re-resolve, LOUD).
    ep_dir, ep = _reresolve_episode_stills_dir(
        ep, ep_dir, warnings, ledger=ledger,
    )
    # A successful same-freeze re-key is durable identity truth, not merely an
    # output-directory hint.  Carry it through the wire so VideoRenderBatch
    # never recreates the retired pending workspace.
    ledger["episode_id"] = ep
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
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("object_id") or "")
        kind = str(obj.get("kind") or "portrait")
        role = str(obj.get("role") or "character_video")
        char_id = str(obj.get("char_id") or "")
        beat_id = str(obj.get("beat_id") or "")
        # RADIO FACE LOGIC (2026-07-04): the announcer portrait's chosen radio-host
        # style rides onto the dispatched ledger row so the render-side guard can
        # fail CLOSED on a faceless still about to feed a HuMo announcer init.
        radio_host_style = str(obj.get("radio_host_style") or "")
        # still_word v2 (2026-07-04): the LOCKED per-episode lettering + backdrop
        # family propagate onto the row for operator QA.
        lettering_style = str(obj.get("lettering_style") or "")
        backdrop_family = str(obj.get("backdrop_family") or "")
        prompt = append_visual_safety_clause(str(obj.get("prompt") or ""))
        prompt_hash = _prompt_content_hash(prompt)
        obj_w = int(obj.get("w") or 0)
        obj_h = int(obj.get("h") or 0)
        if not oid:
            continue
        if role not in _role_slots.ROLE_TO_VIDEO_SLOT:
            raise ImageRenderError(
                f"{oid}: image object declares unknown role {role!r}; cannot "
                "prove an init-image consumer. Refusing image generation."
            )
        capability = still_capabilities.get(role)
        if capability is None:
            raise ImageRenderError(
                f"{oid}: cannot prove an effective init-image consumer for role "
                f"{role!r}; resolve that role's video engine before image "
                "generation."
            )
        # A still is generated ONLY for a proven effective consumer.  The shared
        # tri-state decision accounts for force-map + radio-host redirects, so
        # this cannot drift from the render engine's actual init-image capability.
        if not capability:
            log.info("[OTR_ImageGenDispatcher] skip %s (role=%s): no proven "
                    "effective video consumer for init_image -- no still "
                    "generated", oid, role)
            continue
        # Slot resolution per OBJECT role (ST-3: the ImageDirector slots
        # finally honored); empty named slot -> character fallback, LOUD.
        engine_id, slot, fell_back = resolve_engine_for_role(image_policy, role)
        # BUG-LOCAL-405 (per-role image selection): LOUD trace of how each
        # object's role resolved to a policy slot + engine, so an unexpected
        # engine (e.g. every still minting flux_gen1 because the SAVED policy
        # carried flux_gen1 in all slots) is visible in the server log instead
        # of being inferred from the output. Permanent observability.
        log.info(
            "[OTR_ImageGenDispatcher] resolve: object=%s kind=%s role=%s -> "
            "slot=%s engine=%s", oid, kind, role, slot, engine_id or "<none>")
        if fell_back:
            warnings.append(
                f"{oid}: image slot {slot} empty; fell back to "
                "character_image_model (LOUD)")
        if not engine_id:
            warnings.append(f"{oid}: no image engine selected; skipped (fail-closed)")
            continue
        # Credits (operator 2026-06-21): the per-slot IMAGE model never reached
        # the credits dossier because the ledger['images'] section does not
        # survive to the on-disk ledger the HUD reads (only meta does). Mirror
        # the VIDEO render_engines.by_role histogram into meta so the credits
        # can show the image model used for each role (announcer/music/cast).
        _ih_role = str(role or "?")
        img_hist.setdefault(_ih_role, {})
        img_hist[_ih_role][str(engine_id)] = (
            img_hist[_ih_role].get(str(engine_id), 0) + 1)
        try:
            _assert_not_path(prompt)
        except ValueError as exc:
            warnings.append(str(exc))
            continue
        seed = resolve_object_seed(seed_cfg, oid, prompt_hash, kind=kind)
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
                if radio_host_style:
                    fresh["radio_host_style"] = radio_host_style
                if lettering_style:
                    fresh["lettering_style"] = lettering_style
                if backdrop_family:
                    fresh["backdrop_family"] = backdrop_family
                images.append(fresh)
                ep_rows.append(fresh)
                reused += 1
                report.append(f"{oid}: cache HIT ({hit_id[:12]}) -> "
                              f"{os.path.basename(dst)}")
                continue
            warnings.append(
                f"{oid}: cache index entry {hit_id[:12]} has no on-disk file; "
                "regenerating (LOUD)")
        # cache miss -> assert usable (fail-closed) -> generate -> stamp. Local
        # image engines take the GPU residency lease; Partner/cloud adapters do
        # not touch the local GPU gate.
        try:
            _ireg.assert_usable(engine_id, role)
        except Exception as exc:  # noqa: BLE001  (EngineUnusable et al.)
            # NO FALLBACKS (operator 2026-06-18): a selected-but-unusable engine
            # (opt-in flag off / weights absent / unbuilt) HARD-FAILS the episode
            # LOUD -- never skipped, never silently substituted with flux. Enable
            # the engine + provide its weights, or pick a usable engine.
            raise ImageRenderError(
                f"{oid}: selected image engine '{engine_id}' is not usable for "
                f"role '{role}' ({exc}). NO FALLBACK -- enable the engine "
                f"(its flag) + provide its weights, or select a usable engine "
                f"in the OTR_VideoDirector dropdown."
            ) from exc
        # S4: ADAPTER-level usability with REAL host facts + the episode
        # policy (the registry-level check above is name/role only; every
        # image adapter declares assert_usable(host_caps, profile, ...)
        # but it was never called on this path -- the campaign's most
        # consequential wiring catch, image side). Stubs without the
        # method skip; a legacy 2-arg signature keeps working.
        try:
            _eng = _ireg.get_engine(engine_id)
            _eng = _eng() if isinstance(_eng, type) else _eng
            _adapter_assert = getattr(_eng, "assert_usable", None)
            if callable(_adapter_assert):
                try:
                    _adapter_assert(host_caps=_host_caps,
                                    profile=_adapter_profile,
                                    request_template=None)
                except TypeError:
                    _adapter_assert(host_caps=_host_caps,
                                    profile=_adapter_profile)
        except Exception as exc:  # noqa: BLE001  (EngineUnusable et al.)
            raise ImageRenderError(
                f"{oid}: image engine '{engine_id}' failed ADAPTER-level "
                f"usability for role '{role}' ({exc}). NO FALLBACK -- fix "
                f"the engine's weights/flags/host requirements or select "
                f"a usable engine."
            ) from exc
        if gen_fn is None:
            # S0 portability (2026-07-10): a pending target with no way to
            # render it is a HARD FAIL, never a silent skip. The old branch
            # warned "skipped on CPU" and continued, so an episode could
            # "succeed" with missing stills. Inject a gen_fn (tests) or
            # dispatch via OTRImageGenDispatcher (in-process engine path).
            raise ImageRenderError(
                f"{oid}: image target requires generation but no gen_fn was "
                f"provided (engine '{engine_id}', role '{role}'). NO SILENT "
                "SKIP -- inject a gen_fn or dispatch via the node."
            )
        request = {
            "request_id": key, "role": role, "object_id": oid,
            "kind": kind, "char_id": char_id, "beat_id": beat_id,
            "engine_id": engine_id, "engine_version": eng_version,
            "prompt": prompt, "prompt_hash": prompt_hash, "seed": seed,
            "negative_prompt": visual_safety_negative(
                str(obj.get("negative_prompt") or "")),
            # w/h end-to-end (pass-02 Gem-1): the engine call reads
            # width/height (flux_gen1._flux_params request precedence), so
            # landscape scene stills are REAL, not env defaults.
            "w": obj_w, "h": obj_h,
            "width": obj_w or None, "height": obj_h or None,
        }
        lease = None
        cloud_image_engine = _is_cloud_image_engine(engine_id)
        try:
            if not cloud_image_engine:
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
            # NO FALLBACKS (operator 2026-06-18): hard-fail, never skip.
            raise ImageRenderError(
                f"{oid}: GPU lease timeout rendering '{engine_id}' ({exc}). "
                f"NO FALLBACK.") from exc
        except ImageHandoffTimeout as exc:
            raise ImageRenderError(
                f"{oid}: image handoff from '{engine_id}' not ready ({exc}). "
                f"NO FALLBACK.") from exc
        except ImageRenderError:
            raise
        except Exception as exc:  # noqa: BLE001 -- any render failure -> HARD FAIL
            # NO FALLBACKS (operator 2026-06-18): a wrapper-node-missing /
            # CUDA-OOM / decode failure HARD-FAILS the episode LOUD -- no skip,
            # no radio-floor degrade, no silent flux substitution.
            raise ImageRenderError(
                f"{oid}: image render with '{engine_id}' failed "
                f"({type(exc).__name__}: {exc}). NO FALLBACK -- fix the engine "
                f"or select a usable one.") from exc
        finally:
            if lease is not None:
                _lease.release(lease)
        # post-generation residency confirm (best-effort; gates the C->A handoff).
        if (not cloud_image_engine
                and not _lease.wait_until_below_mb(15000, attempts=3, sleep_s=0.0)):
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
        if radio_host_style:
            row["radio_host_style"] = radio_host_style
        if lettering_style:
            row["lettering_style"] = lettering_style
        if backdrop_family:
            row["backdrop_family"] = backdrop_family
        # 3D image streams (2026-06-21): carry the mesh subject identity onto the
        # row (additive) so the mesh cache keys on a STABLE per-subject id (the
        # mesh_fodder file), not the per-beat still hash. Absent on non-3D rows.
        _msid = str(obj.get("mesh_subject_id") or "")
        if _msid:
            row["mesh_subject_id"] = _msid
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
    # Credits: stamp the per-role image-engine histogram into meta (which DOES
    # persist to the on-disk ledger the credits dossier reads) so the dossier
    # shows the image model used for each slot. Additive; never overwrites.
    _ie_meta = ledger.get("meta")
    if not isinstance(_ie_meta, dict):
        _ie_meta = {}
        ledger["meta"] = _ie_meta
    _ie_meta["image_engines"] = {"by_role": img_hist, "image_revision": rev}
    # S2 durable persistence (credits enrichment 2026-07-03): ``ledger`` here
    # is the LOCAL wire-parsed dict -- the stamp above was wire-only, so the
    # image-engine receipts died with the wire. Copy the images section + the
    # image_engines meta into the production-ledger SINGLETON and save
    # LOUDLY (raises LedgerStampError on save failure; test-mode injects
    # in-memory only). The late OTR_CreditsRoll node reads THIS.
    from .production_ledger import stamp_durable
    stamp_durable(
        sections={"images": ledger["images"]},
        meta_updates={"image_engines": _ie_meta["image_engines"]},
        source="image_dispatcher",
    )
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


def _is_cloud_image_engine(engine_id):
    """True for external image engines that do not need the local GPU lease.

    Includes Partner/cloud image engines and direct BYO API engines that declare
    ``native=False``. Local wrappers omit ``native`` or set it truthy, so they
    keep the lease behavior.
    """
    eid = str(engine_id or "")
    if eid.startswith("cloud_"):
        return True
    eng = _safe_engine(eid)
    if eng is None:
        return False
    if getattr(eng, "native", True) is False:
        return True
    node_key = str(getattr(eng, "node_key", "") or "")
    return node_key.startswith("cloud_")


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
        # S4: real host facts instead of the old (None, None, None).
        from ._otr_shared.host_caps import build_host_caps
        prepared = prep(build_host_caps(), {}, {})
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
