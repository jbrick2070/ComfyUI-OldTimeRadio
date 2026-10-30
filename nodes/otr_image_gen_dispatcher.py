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
* The still count follows the beat budget. ``fresh_cap`` rides along on the
  image policy and is RECORDED here, but nothing in this module applies it as a
  cap (docs review, 2026-10-14: the old "min(fresh_cap, beat_budget)" line
  described a limiter that does not exist; only the cloud cost estimate reads
  the value).
* image generation is the only GPU step; it is injected on the CPU test path, so
  this node's cache/ledger/gate logic is fully unit-tested without a GPU.

Cold-import clean: torch / PIL / numpy are never imported at module scope.
"""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import re as _re
import time
from typing import NamedTuple

log = logging.getLogger("OTR")

from ._otr_shared import character_photos as _photos
from ._otr_shared import gpu_residency as _lease
from ._otr_shared import portrait_ledger as _pl
from ._otr_shared import role_slots as _role_slots
from ._otr_shared import still_receipt as _receipt
# Cold-import clean: node_progress is stdlib-only, and reaches ComfyUI only
# through modules that are already loaded.
from ._otr_shared.node_progress import (NodeProgress as _NodeProgress,
                                        QuietInnerBars as _QuietInnerBars)
# Cold-import clean: route_freeze is stdlib-only at module scope.
from ._otr_shared.route_freeze import RouteFreezeError as _RouteFreezeError
from ._otr_story_brief_helpers import (
    append_visual_safety_clause,
    visual_safety_negative,
)
# The banana route (2026-08-06 build spec): pure,
# stdlib-only house-style transform applied at THIS funnel before the prompt
# content hash, gated by env + the fidelity-bank idiom. Cold-import clean.
from . import _otr_banana_route as _banana
from ._otr_image_engines import registry as _ireg
# The single canonical residue-freer (writer LLM + Bark, then a surgical detach of any
# ComfyUI-tracked patcher, then the allocator flush). Cold-import clean: stdlib only at
# module scope; torch / comfy are imported lazily inside the call.
from . import _otr_vram_levers as _levers

try:
    from ._otr_shared import env as otr_env
    from ._otr_shared import still_seed_settings as _seed_settings
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore
    from _otr_shared import still_seed_settings as _seed_settings  # type: ignore

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
                      kind="", w=0, h=0, *, anchor="", source_context_hash="") -> str:
    """The dispatch dedup key (PASS-IMG MUST-FIX #5). A change in ANY field ->
    new key -> regen -> new content hash -> B's mesh cache invalidates.

    Still-spine ST-3 (pass-02 Gem-1): the key gains ``kind`` + ``w`` + ``h``
    so a landscape scene still and a portrait of the same subject can never
    collide, and a dim change regenerates.

    ``anchor`` is the portrait hash a reference-conditioned mint was actually
    given. It is appended ONLY when non-empty, which is load-bearing twice over:
    an unconditional append would change every existing digest, and a mint that
    consumed no reference must not be rekeyed by one. Without it in the key, a
    REGENERATED portrait would keep serving stills conditioned on the old face."""
    parts = [
        str(role), str(object_id), str(prompt_hash), int(seed),
        str(engine_id), str(engine_version),
        str(kind), int(w or 0), int(h or 0),
    ]
    if anchor:
        parts.append(str(anchor))
    if source_context_hash:
        parts.append({"source_context_hash": str(source_context_hash)})
    return _content_hash(parts)


def resolve_seed_and_mode(
    seed_cfg, object_id, prompt_hash, *, kind="", char_id="",
    portrait_prompt_hash="", identity_take=0, take=0,
) -> "tuple[int, str]":
    """Per-object seed AND how it was chosen, as ``(seed, mode)``.

    A character's face changed on every single beat, by construction: each scene
    still derived its seed from its own ``object_id`` plus its own per-beat
    ``prompt_hash``, so three beats meant three unrelated draws. A
    ``scene_character`` still now derives its seed from the CHARACTER'S OWN
    PORTRAIT draw, which is the one image the whole episode already agrees on
    (the portrait is also the sole init for the video lane). One face per
    character, on every lane -- the invention lanes included, where there is no
    source gender to get wrong in the first place.

    The mode is returned alongside so the ledger can stamp what ACTUALLY
    happened rather than a literal that lies the moment the kill switch is used.

    jump_segment is deliberately excluded: a jump CUT is supposed to be a cut,
    and tests/test_multiclip_jump_stills.py pins three DISTINCT seeds across a
    base plus two segments. Those rows still get the anchor and the reference --
    just not the shared seed.

    Otherwise unchanged: ``mode=request_hash`` (the ImageDirector default)
    derives a deterministic seed from ``request_seed + object_id + prompt_hash``
    so every object gets its own seed while the whole episode stays reproducible;
    ``mode=fixed`` returns ``request_seed`` verbatim. Pure.

    BUG-411 restore: the radio BOOKEND (``kind == "scene_open"``) renders with a
    FIXED deterministic seed (the 6/5 ``radio_bookend_seed=4242`` widget,
    env-overridable via ``OTR_RADIO_BOOKEND_SEED``) so the opening radio still is
    reproducible run-to-run independent of the request hash -- exactly the 6/5
    behavior the rewrite lost.

    SCENE PRODUCER TAKES (chunk 3c, 2026-10-22). A take above zero redraws, in
    every branch above (:func:`_seed_for_take`); take 0 is today's seed, byte
    for byte. ``identity_take`` is the character's take: it enters a portrait's
    seed and the identity seed of every scene still of that character, so the
    face and the stills that follow it move together. ``take`` is the still's
    own take, entered last, on that still alone. Both are 0 on every run that
    is not a producer run.
    """
    # The radio BOOKEND still (scene_open) AND the radio-HOST FACE object
    # (2026-07-01 brief-driven radio-host; object_id "radio_host_portrait",
    # matches otr_meta_brief_image_prompt.RADIO_HOST_PORTRAIT_ID) both render
    # with the FIXED deterministic bookend seed so the host face is reproducible
    # run-to-run and open/inter/close share ONE canonical face.
    _oid = str(object_id or "")
    if str(kind or "") == "scene_open" or _oid == "radio_host_portrait":
        return (_seed_for_take(_seed_settings.bookend_seed(), take), "")
    cfg = seed_cfg if isinstance(seed_cfg, dict) else {}
    base = int(cfg.get("request_seed") or 0)
    if str(cfg.get("mode") or "request_hash") != "request_hash":
        return (_seed_for_take(_seed_for_take(base, identity_take), take), "")
    # AFTER the mode gate, never before it. `base` does not exist until the line
    # above, so a branch placed earlier raises UnboundLocalError inside a call
    # that dispatch_images makes outside any try/except -- the whole dispatch
    # dies. Placed between `base` and the gate, it silently breaks the
    # documented mode='fixed' contract instead, which is worse.
    if (str(kind or "") == "scene_character" and char_id and portrait_prompt_hash
            and _seed_settings.identity_seed_enabled()):
        # EXACTLY the portrait object's own draw: the portrait's object_id IS
        # the char_id (otr_meta_brief_image_prompt.py:1765), so this reproduces
        # the seed that character's portrait already rendered with, leaving the
        # portrait itself byte-identical.
        digest = hashlib.sha256(
            f"{base}:{char_id}:{portrait_prompt_hash}".encode()).hexdigest()
        identity = _seed_for_take(int(digest[:8], 16), identity_take)
        return (_seed_for_take(identity, take), "seed")
    digest = hashlib.sha256(
        f"{base}:{object_id}:{prompt_hash}".encode()).hexdigest()
    # A character's take enters any object tied to that character: its portrait
    # (whose draw IS the identity seed above) and, with the identity seed
    # switched off, its scene stills too.
    seed = _seed_for_take(int(digest[:8], 16), identity_take if char_id else 0)
    return (_seed_for_take(seed, take), "")


def _seed_for_take(seed: int, take) -> int:
    """A Scene Producer take above zero is a different, reproducible draw;
    take 0 is the seed itself, byte for byte."""
    try:
        take = int(take or 0)
    except (TypeError, ValueError):
        take = 0
    if take <= 0:
        return seed
    return int(hashlib.sha256(f"{seed}:take:{take}".encode()).hexdigest()[:8], 16)


def _producer_takes(direction, kind, char_id, beat_id) -> "tuple[int, int]":
    """``(identity take, still take)`` for one image object on a Scene
    Producer run, both 0 on any other. The character's take follows its
    char_id; a still's take follows the shot its beat makes (``shot_<beat>``,
    as Shot Lock names it), so a beat's jump segments are redrawn with it. A
    portrait has no shot."""
    if not isinstance(direction, dict):
        return 0, 0
    takes = direction.get("takes") if isinstance(direction.get("takes"), dict) else {}
    characters = takes.get("characters") if isinstance(takes.get("characters"), dict) else {}
    stills = takes.get("stills") if isinstance(takes.get("stills"), dict) else {}
    identity = int(characters.get(str(char_id), 0) or 0) if char_id else 0
    still = (int(stills.get("shot_%s" % beat_id, 0) or 0)
             if beat_id and str(kind or "") != "portrait" else 0)
    return identity, still


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
    slot = _role_slots.image_slot_for_role(role)

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


def negative_source_label(pack_negative, obj_negative) -> str:
    """Where THIS row's composed negative came from, and the one place that names it.

    Pure, and deliberately blind to the engine. It answers exactly one question --
    which of the two COMPOSITION inputs contributed -- so every value it can return
    is verifiable from its own two arguments. The vocabulary is these four values
    and nothing else: ``pack+request``, ``pack``, ``request``, ``none_contributed``
    (COMPOSITION ONLY). It has already drifted once unnoticed -- the
    one-style-authority PLAN documented ``env_override``, which never shipped.

    ITEM H (2026-08-17): the empty arm is `none_contributed`, never
    `engine_hygiene` (a claim about what the ENGINE adds on top), which would be
    wrong twice over. It is computed before `resolve_engine_for_role` picks an
    engine, so it would assert a property of an engine not yet chosen --
    accurate for `z_image_turbo` (which does end
    `.strip() or _HYGIENE_NEGATIVE`) and false for `lumina_image` (no floor at
    all) purely by coincidence. And it would put two authorities in one value:
    composition belongs to the dispatcher, the hygiene floor belongs to the
    engine.

    Naming the empty case for what is actually known dissolves the ordering
    coupling entirely -- the answer no longer depends on the engine, so it does not
    matter where in the loop this is called. Per-engine hygiene telemetry, if ever
    wanted, is a SEPARATE post-resolution field, and engines must DECLARE a floor
    for it to read (the `engine_consumes_still` dual-read pattern) rather than be
    matched by name, per item A's ruling that name-matching ships false positives.

    `none_contributed` does NOT mean "no negative reached the model" -- an engine
    may still apply its own hygiene floor. It means neither the pack nor the object
    supplied one. Whether the negative was LIVE at all is a different question
    again (at cfg 1.0 it conditions nothing) and wants the resolved cfg, which is
    D-BIS finding 4 and is not recorded yet.
    """
    if pack_negative and obj_negative:
        return "pack+request"
    if pack_negative:
        return "pack"
    if obj_negative:
        return "request"
    return "none_contributed"


#: Excerpt half-width around a path-guard match: a ``prompt[:60]`` report shows
#: nothing useful when the offending character sits late in a long prompt, so
#: the excerpt is centred on the match.
_PATH_GUARD_EXCERPT_RADIUS = 90

_DRIVE_ROOT_RE = _re.compile(r"^[A-Za-z]:[\/]")
_UNC_ROOT_RE = _re.compile(r"^[\/]{2}[^\/]")
_POSIX_ROOT_RE = _re.compile(r"^/[^/\s]")
_FILE_URI_RE = _re.compile(r"^file://", _re.IGNORECASE)
_EXPLICIT_RELATIVE_RE = _re.compile(r"^\.{1,2}[\/]")
#: A bare filename or a relative path ENDING in an image extension, with no
#: whitespace anywhere -- "shot.png", "assets/shot.png", "a\b\shot.jpeg".
_IMAGE_PATH_RE = _re.compile(
    r"^[^\s]+\.(?:png|jpg|jpeg|webp|bmp|gif|tif|tiff)$", _re.IGNORECASE
)


def path_guard_arm(prompt: str) -> "dict | None":
    r"""Which arm of the path guard a prompt trips, or ``None`` if it is clean.

    Returns ``{arm, token, index, excerpt, prompt_len, prompt_hash}``, where
    ``prompt_hash`` is the dispatcher's canonical sha256 (``_prompt_content_hash``)
    so the evidence joins the ledger's field of the same name.

    WHOLE-STRING CLASSIFICATION (2026-08-05). The guard exists because a
    prompt-STRING socket and a path-STRING socket must not be crossed, and a
    socket carries a WHOLE value: a crossed socket delivers a string that IS a
    path, never prose that merely contains one. A bare separator test
    (``os.sep in p or os.altsep in p or endswith(ext)``) is wrong: on Windows
    ``os.altsep`` is ``/``, so "a black/white striped scarf" and "the corner of
    5th/Main" would be refused and their beats render NO still. Laundering
    slashes out of prose is wrong too: it turns a loud refusal of a REAL path
    into a quiet garbled render.

    The arms are ORDERED, most specific first, and every one of them describes
    the ENTIRE string:

      ``drive_root``        C:\... or C:/...
      ``unc_root``          \server\share or //server/share
      ``posix_root``        /var/tmp/x
      ``file_uri``          file:///...
      ``explicit_relative`` ./x or ../x
      ``image_path``        whitespace-free and ending in an image extension,
                            which covers both "shot.png" and "assets/shot.png"

    Prose containing a separator is CLEAN. Prose ending in ".png" is clean too
    if it contains whitespace -- "a portrait of the radio host.png" is a
    sentence, "host.png" is a filename.
    """
    p = str(prompt or "")
    if not p:
        return None
    stripped = p.strip()
    if not stripped:
        return None

    arm = token = None
    for candidate_arm, pattern in (
        ("drive_root", _DRIVE_ROOT_RE),
        ("unc_root", _UNC_ROOT_RE),
        ("posix_root", _POSIX_ROOT_RE),
        ("file_uri", _FILE_URI_RE),
        ("explicit_relative", _EXPLICIT_RELATIVE_RE),
    ):
        match = pattern.match(stripped)
        if match:
            arm, token = candidate_arm, match.group(0)
            break
    if arm is None and _IMAGE_PATH_RE.match(stripped):
        arm, token = "image_path", stripped[stripped.rfind("."):]
    if arm is None:
        return None

    index = p.index(token) if token and token in p else 0
    start = max(0, index - _PATH_GUARD_EXCERPT_RADIUS)
    end = min(len(p), index + _PATH_GUARD_EXCERPT_RADIUS)
    return {
        "arm": arm,
        "token": token,
        "index": index,
        "excerpt": repr(p[start:end]),
        "prompt_len": len(p),
        # The dispatcher's CANONICAL prompt hash (sha256), so this correlates
        # with the ledger's own prompt_hash rather than inventing a second
        # digest nothing else can join on.
        "prompt_hash": _prompt_content_hash(p),
    }


def _assert_not_path(prompt: str) -> None:
    """Fail-closed prompt-STRING vs path-STRING guard (PASS-IMG SHOULD-FIX)."""
    hit = path_guard_arm(prompt)
    if hit is not None:
        exc = ValueError(
            "OTR_ImageGenDispatcher: image prompt looks like a PATH, not prompt "
            "text (arm=%s token=%r at index %d of %d; excerpt %s); the "
            "prompt-STRING and path-STRING sockets must not be crossed"
            % (hit["arm"], hit["token"], hit["index"], hit["prompt_len"],
               hit["excerpt"])
        )
        # Carry the structured verdict on the exception so the caller does not
        # have to re-run the predicate (and its hash) to recover it.
        exc.path_guard_hit = hit
        raise exc


#: Stills whose edge statistics differ STRUCTURALLY rather than stylistically.
#: Word/title cards are typographic by design -- flat plates with hard lettering
#: -- so including them would make a healthy episode look fractured.
_SPREAD_EXCLUDED_SOURCES = frozenset({"still_word"})


def _laplacian_variance(pixels) -> float:
    """Mean 4-neighbour Laplacian variance of one decoded still.

    The identical formula the q-bakeoff uses for its sharpness rank
    (`scripts/run_ltx_av_q_bakeoff.py`), applied to a single 3D `[H,W,C]`
    frame instead of a sampled 4D stack.

    This is a SHARPNESS statistic, not a style classifier, and it is recorded
    on exactly that footing: a dense cross-hatched engraving and a detailed
    photograph can both score high. It is useful only COMPARATIVELY, against
    the episode's own median, and it never decides anything. Returns 0.0 on
    anything it cannot measure -- telemetry must not raise.
    """
    try:
        import numpy as np  # lazy, mirroring _coerce_pixels
        arr = np.asarray(pixels)
        if arr.ndim == 4:            # [N,H,W,C] -> first frame
            arr = arr[0]
        if arr.ndim == 2:            # already luma
            g = arr.astype(np.float64)
        elif arr.ndim == 3 and arr.shape[-1] >= 3:
            f = arr.astype(np.float64)
            g = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        else:
            return 0.0
        if g.shape[0] < 3 or g.shape[1] < 3:
            return 0.0
        lap = (g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
               - 4.0 * g[1:-1, 1:-1])
        return round(float(lap.var()), 2)
    except Exception:  # noqa: BLE001 -- telemetry never breaks a render
        return 0.0


def _style_spread(rows) -> dict:
    """Max pairwise Laplacian spread across an episode's comparable stills.

    RELATIVE to the episode's own median, never an absolute style constant --
    the q-bakeoff shape, where sharpness is ranked against a baseline rather
    than a fixed number.

    It reports; it never gates. THE LAW forbids failing OR rerolling an episode
    for style or visual vocabulary, and this is a style property, so `exceeded`
    is a flag for a human and nothing downstream may branch a render on it.
    `threshold: null` means the metric shipped uncalibrated and no claim about
    spread is being made.
    """
    vals = [float(r.get("laplacian") or 0.0) for r in rows
            if r.get("kind") != "shot"
            and r.get("source") not in _SPREAD_EXCLUDED_SOURCES
            and float(r.get("laplacian") or 0.0) > 0.0]
    out = {
        "metric": "laplacian_variance",
        "comparable_stills": len(vals),
        "excluded_sources": sorted(_SPREAD_EXCLUDED_SOURCES),
        "median": None, "max_pairwise": None, "max_ratio": None,
        "threshold": None, "exceeded": None,
        # Procedural viz_* bookends never mint a still, so they cannot be
        # measured here. This audits a SUBSET of the fracture surface.
        "covers": "minted stills only",
    }
    if len(vals) < 2:
        return out
    ordered = sorted(vals)
    mid = len(ordered) // 2
    median = (ordered[mid] if len(ordered) % 2
              else (ordered[mid - 1] + ordered[mid]) / 2.0)
    out["median"] = round(median, 2)
    out["max_pairwise"] = round(max(vals) - min(vals), 2)
    if median > 0:
        out["max_ratio"] = round(max(vals) / median, 3)
    return out


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
        if otr_env.get("OTR_TEST_MODE") == "1":
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
            log.info(
                "[OTR_ImageGenDispatcher] PATH RECONCILED: episode_id %r is a "
                "stale pending id (its dir was renamed after capture); stills "
                "re-keyed to the active ledger's episode dir %r (same episode, "
                "post-rename name).", ep, new_ep)
            # THE ID AND ITS PATHS MOVE TOGETHER (2026-09-26), as in
            # ShotLock's post-audio overlay. The caller writes the new id onto
            # the wire; every episode-local path the wire already carries
            # still points into the renamed-away pending dir, so move them in
            # the same step, in place, with the walker the rename itself uses.
            if isinstance(ledger, dict):
                from .production_ledger import _rebase_episode_local_paths
                rebased, moved = _rebase_episode_local_paths(
                    ledger, str(Path(episodes_root) / str(ep)),
                    str(new_episode_dir))
                if moved and isinstance(rebased, dict):
                    ledger.clear()
                    ledger.update(rebased)
                    log.info(
                        "[OTR_ImageGenDispatcher] moved %d wire path(s) from "
                        "%s onto %s", moved, ep, new_ep)
            return str(new_episode_dir / "stills"), new_ep
    except Exception as exc:  # noqa: BLE001 -- never block dispatch on the probe
        log.warning(
            "[OTR_ImageGenDispatcher] episode stills re-resolve skipped: %s", exc)
    return ep_dir, ep


#: PNG magic. A cache source must BE a PNG, not merely exist: the whole point of
#: the forged-entry attack is that any readable file passes an existence check.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _portrait_png(portrait_row, portrait_hash, output_dir) -> str:
    """The portrait row's PNG on disk, or "". Confined before the stat (Sonnet
    QA, 2026-10-18): the row is ledger data, so outside the approved folders it
    reads as absent; the content-addressed pool path is the fallback.

    THE EPISODE COPY IS THE LAST RESORT (Sonnet QA, 2026-10-22): a replay
    imported into a new workspace, or run after the shared pool was cleared,
    finds neither pool copy, while the row's own episode copy -- rebased onto
    this workspace by the import -- is right here. Tried last, so every case
    the pool answers reads exactly the file it read before."""
    row = portrait_row or {}
    cand = _confined_or_empty(str(row.get("pool_path") or row.get("path") or ""))
    if not cand or not os.path.isfile(cand):
        try:
            # The pool fallback is confined too: the hash is ledger data, and a
            # linked cache file or folder must read as absent, never as bytes
            # from outside the approved folders (registry policy review,
            # 2026-10-22).
            cand = _confined_or_empty(
                str(_pl.portrait_path_for_hash(portrait_hash, output_dir) or ""))
        except Exception:  # noqa: BLE001 -- resolver is best-effort
            cand = ""
    if not cand or not os.path.isfile(cand):
        cand = _confined_or_empty(str(row.get("path") or ""))
    return cand if cand and os.path.isfile(cand) else ""


def _confined_or_empty(path) -> str:
    try:
        from ._otr_paths import confined_or_empty
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from _otr_paths import confined_or_empty  # type: ignore
    return confined_or_empty(path)


def _trusted_pool_source(src, output_dir) -> str:
    """The confined absolute path of ``src`` when it is a still this pack
    itself wrote, else ``""``. NEVER raises. The caller copies THIS path, not
    ``src``: a relative ``src`` is checked against ``output_dir`` and would
    otherwise be opened against the process CWD (Cursor review, 2026-10-18).

    The cache index and the image rows both arrive in ``script_json``, i.e. from
    an unauthenticated ``/prompt`` body. Before this check the only gate was
    ``os.path.exists(src)``, so a forged ``pool_path`` pointing at any readable
    file was copied into the episode stills directory -- which ComfyUI serves
    over ``/view``. That made it an arbitrary file READ, not merely a stray copy.

    Three conditions, all cheap:
      * not a remote path -- a UNC value would authenticate to the host the
        workflow named, on the stat below (the dispatcher had NO remote guard);
      * inside this run's own ``<output>/otr/`` tree, resolved from the SAME
        ``output_dir`` the caller already threads to ``episode_stills_dir`` --
        not ``comfy_output_dir()``, which would refuse every ``output_dir=tmp_path``
        test;
      * a real PNG of at least ``_MIN_PNG_BYTES``.

    An empty answer is not an error: the caller falls into the existing
    "no on-disk file; regenerating (LOUD)" path, which is exactly right -- an
    untrusted cache entry should simply not be a cache hit.
    """
    try:
        text = str(src or "").strip()
        if not text:
            return ""
        # Inside this run's own otr/ tree LEXICALLY AND AFTER REALPATH
        # (2026-10-18): the abspath check alone let a link inside otr/ lead
        # out of it -- registry review flagged "arbitrary-file-read". The
        # helper refuses a remote path first, and anchors a relative value at
        # the SAME output base the root is built from.
        try:
            from ._otr_paths import confined_or_empty
        except ImportError:  # pragma: no cover -- flat (sys.path) load
            from _otr_paths import confined_or_empty  # type: ignore
        base = _pl._output_base(output_dir)
        real = confined_or_empty(text, roots=[os.path.join(base, "otr")], base=base)
        if not real or not os.path.isfile(real):
            return ""
        if os.path.getsize(real) < _MIN_PNG_BYTES:
            return ""
        with open(real, "rb") as handle:
            return real if handle.read(len(_PNG_MAGIC)) == _PNG_MAGIC else ""
    except Exception:  # noqa: BLE001 -- a trust probe must never raise
        return ""


def _materialize_episode_copy(src_path, ep_dir, object_id, content_hash):
    """Copy a still into the EPISODE stills dir (idempotent; readable name
    ``{object_id}_{hash12}.png``). Returns the episode-local path. Lazy
    shutil; raises on a failed copy (the caller warns + falls back)."""
    import re
    import shutil
    try:
        from ._otr_paths import confine_to_output_tree
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from _otr_paths import confine_to_output_tree  # type: ignore
    # THE DIGEST IS HEX OR IT IS NOT A DIGEST (registry review, 2026-10-30):
    # `content_hash` is a ledger field too, and its first twelve characters
    # were the filename's tail.
    digest = re.sub(r"[^0-9A-Fa-f]", "", str(content_hash or ""))[:12] or "nodigest"
    dst = os.path.join(str(ep_dir), f"{object_id}_{digest}.png")
    # THE DESTINATION MUST STAY IN THE EPISODE DIR (2026-09-05). `object_id`
    # reaches this f-string from the workflow's own ledger, so a value like
    # `../escaped` walks the copy out of stills/. Asserted here rather than at
    # the call sites because BOTH callers want it and neither has a reason to
    # write outside the episode folder.
    base = os.path.normcase(os.path.abspath(str(ep_dir)))
    if os.path.commonpath([os.path.normcase(os.path.abspath(dst)), base]) != base:
        raise ImageRenderError(
            "image object_id %r would place its still outside the episode "
            "directory; ids name a file, never a location." % (object_id,))
    # AND AFTER REALPATH, AGAINST THE OUTPUT TREE (registry review,
    # 2026-10-30): the lexical check above cannot see a junction already
    # sitting at stills/, through which the copy landed outside the tree.
    try:
        dst = confine_to_output_tree(dst, "episode still")
    except ValueError as exc:                 # OtrPathContractError
        raise ImageRenderError(
            "image still for %r cannot be written: %s" % (object_id, exc)) from None
    os.makedirs(str(ep_dir), exist_ok=True)
    if not os.path.exists(dst):
        shutil.copyfile(str(src_path), dst)
    return dst


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


def _effective_video_engine_for_role(role: str, eng_id: str) -> str:
    """Resolve the VIDEO engine the render phase will actually use for ``role``.

    The image phase runs before ShotLock/video dispatch, so it cannot read the
    final mutated shot row. It still has to mint exactly the stills that final
    render will require. The only rewrite is ``OTR_FORCE_ENGINE_MAP``; there is
    no role-based engine swap (the picked lane is the lane, 2026-10-15).

    Unknown engines or import failures keep the input id (fail-safe: mint the
    still rather than quietly skipping an asset that render might need).

    DELEGATES to the ONE route-freeze authority (2026-07-25, chunk 1a), so the
    image phase and the render phase can never disagree about the route."""
    try:
        from ._otr_shared import route_freeze as _rf  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import route_freeze as _rf  # type: ignore
    return _rf.effective_engine_for_role(str(role or ""), str(eng_id or ""))


def still_consumer_capability(image_policy: dict, role: str):
    """``True`` / ``False`` / ``None`` for one role's init-image consumer.

    ``True`` and ``False`` are proven only after resolving the selected engine
    through the same force map the render path uses.
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
    except _RouteFreezeError:
        # A MALFORMED ROUTING ENVIRONMENT IS TERMINAL AND MUST PROPAGATE
        # (2026-07-25 QA). "Absence of proof" is the right third state for an
        # unknown ROLE or an unregistered engine -- it is the wrong answer for
        # a typo'd OTR_FORCE_ENGINE_MAP. Swallowed here, every role resolves to
        # None and the operator is told "cannot prove an effective init-image
        # consumer for role X", pointing them at their engine slots instead of
        # at the env var they actually mistyped. Worse, an episode with no
        # object of an affected role would raise nothing at all here and only
        # die later at render -- defeating fail-closed-at-the-first-reader.
        raise
    except Exception:  # noqa: BLE001 -- absence of proof is a real third state
        return None


def still_consumer_capabilities(image_policy: dict):
    """Return ``{role: True|False|None}`` or ``None`` for malformed policy."""
    if not isinstance(image_policy, dict) or not isinstance(
            image_policy.get("video_models"), dict):
        return None
    return {
        role: still_consumer_capability(image_policy, role)
        for role in _role_slots.ROLE_TO_VIDEO_SLOT
    }


def _coverage_plan_module():
    try:
        from ._otr_video_engines import coverage_plan as _cp  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_video_engines import coverage_plan as _cp  # type: ignore
    return _cp


def merge_jump_still_requests(ledger, objects, required_scene_targets):
    """Merge ShotLock's per-segment jump-still requests into the image phase.

    THE POINT OF CHUNK 4: a jump-cut beat renders N independent clips, but the
    image phase mints exactly ONE still per beat, so segments 1..N-1 would
    reach the render with no init image at all. ShotLock stamped a request row
    per such segment (``shot['jump_still_requests']``); this turns each one
    into a real image OBJECT the dispatcher will render, plus a
    ``required_scene_targets`` row so the existing completion contract fails
    closed when one does not materialize.

    THE SEGMENT STILL IS A CLONE OF THE BEAT'S OWN SCENE STILL -- same prompt,
    same dimensions, same visual style -- with a different ``object_id``. That
    is what makes it a jump CUT rather than a duplicate frame: the per-object
    seed is derived from the object id (``resolve_object_seed`` under the
    default ``request_hash`` mode), so each segment gets a different image of
    the same scene. Under an operator-selected ``fixed`` seed mode they are
    identical by that choice, not by this merge. Prompt DIFFERENTIATION per
    segment is a later chunk's job; the image phase owns prompts, this does not
    invent any.

    FAIL-CLOSED, NEVER GUESSING:
      * a beat whose still IS required but has no scene object to clone,
      * a mesh/plate beat, where the correct source (fodder? plate? both?) is
        genuinely ambiguous and multi-clip mesh lanes are out of scope,
      * more than one candidate scene object for one beat,
      * a malformed request row, or an id that already exists,
    all RAISE. There is no skip branch: whether a lane consumes a still at all
    is decided ONCE at the mint, by the still spine's own predicate, so a
    still-less visualizer lane never reaches here carrying requests.

    Returns ``(objects, required_scene_targets, report_lines)``; the inputs are
    never mutated.
    """
    _cp = _coverage_plan_module()
    shots = [shot for shot in
             (((ledger or {}).get("video") or {}).get("shots") or [])
             if isinstance(shot, dict)]
    requests = []
    for shot in shots:
        rows = shot.get("jump_still_requests")
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise ImageRenderError(
                "shot %s carries a malformed jump_still_requests stamp (%s); "
                "refusing an image phase that cannot be proven complete."
                % (shot.get("shot_id"), type(rows).__name__))
        for row in rows:
            if not isinstance(row, dict) or not str(row.get("object_id") or ""):
                raise ImageRenderError(
                    "shot %s carries a jump-still request without an "
                    "object_id; refusing an unverifiable image phase."
                    % (shot.get("shot_id"),))
            requests.append((shot, row))
    if not requests:
        return objects, required_scene_targets, []

    scene_by_beat = {}
    ambiguous_beats = set()
    mesh_beats = set()
    existing_ids = set()
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        existing_ids.add(str(obj.get("object_id") or ""))
        kind = str(obj.get("kind") or "")
        beat = str(obj.get("beat_id") or "")
        if not beat:
            continue
        if kind in ("mesh_fodder", "scene_background_plate"):
            mesh_beats.add(beat)
        elif kind.startswith("scene_"):
            if beat in scene_by_beat:
                ambiguous_beats.add(beat)
            scene_by_beat[beat] = obj

    required_beats = {
        str(target.get("beat_id") or "")
        for target in (required_scene_targets or [])
        if isinstance(target, dict)
    }
    merged_objects = list(objects)
    merged_targets = list(required_scene_targets or [])
    report = []
    for shot, row in requests:
        beat = str(row.get("beat_id") or "")
        oid = str(row["object_id"])
        segment = int(row.get("segment_index") or 0)
        if beat in mesh_beats:
            raise ImageRenderError(
                "beat %s renders a 3D/mesh lane and also asks for jump-segment "
                "stills; the source still is ambiguous (mesh fodder or "
                "background plate) and multi-clip mesh coverage is not built. "
                "NO GUESS." % beat)
        if beat in ambiguous_beats:
            raise ImageRenderError(
                "beat %s has more than one scene still object, so the image "
                "phase cannot say which one its jump segments should look "
                "like. NO GUESS." % beat)
        base = scene_by_beat.get(beat)
        if base is None:
            # NO SILENT SKIP (2026-07-25 QA fix). This branch must not infer "no
            # scene object and no required target means the lane consumes no
            # still, so its segments need none" and drop the requests: that
            # contradicts the still spine, which demands every STAMPED request
            # back regardless, so the episode would die at the render boundary
            # with a message about a missing still nobody had decided to skip.
            # The inference happens ONCE, at the mint
            # (``otr_shot_lock._lane_consumes_a_still``), using the spine's own
            # predicate, so a still-less lane never carries requests to begin
            # with. Arriving here therefore means the stamp and the image
            # payload genuinely disagree about one beat, and guessing which is
            # right is how a jump cut ends up rendering from nothing.
            raise ImageRenderError(
                "beat %s owes a jump-segment still (%s, segment %d) but the "
                "image phase minted no scene still to derive it from%s. A jump "
                "cut with no still renders from nothing. NO FALLBACK."
                % (beat, oid, segment,
                   "" if beat in required_beats
                   else " -- and the beat is absent from required_scene_targets"
                        ", so the stamp and the image payload disagree about "
                        "whether this lane consumes a still at all"))
        if oid in existing_ids:
            raise ImageRenderError(
                "jump-segment still %s already exists in the image payload; "
                "refusing an ambiguous duplicate object." % oid)
        clone = dict(base)
        clone["object_id"] = oid
        if base.get("kind") == "scene_character":
            clone["source_scene_scope"] = "scene_character"
        # THE FIXED-SEED LANES LOSE THEIR FIXED SEED HERE, DELIBERATELY.
        # ``resolve_object_seed`` pins seed 4242 for ``kind == "scene_open"``
        # and for the radio-face object ids, so that the BOOKEND is one
        # canonical image run to run. Rewriting the kind and the id drops a
        # segment out of both pins and onto the request-hash seed. That is the
        # correct answer for a jump CUT -- segment 1 must not be segment 0's
        # frame again -- and it does NOT cost reproducibility: the request-hash
        # seed is derived from (request_seed, object_id, prompt_hash), all
        # stable, so the same episode re-renders the same segment stills. What
        # a bookend loses is only the shared canonical LOOK across its own
        # segments, which is what cutting means. Stated because it was a side
        # effect of the kind rewrite before it was a decision.
        clone["kind"] = _cp.JUMP_STILL_KIND
        clone["beat_id"] = beat
        clone["segment_index"] = segment
        clone["role"] = str(row.get("role") or base.get("role") or "")
        clone["source"] = "jump_segment_cover"
        clone.pop("mesh_subject_id", None)
        existing_ids.add(oid)
        merged_objects.append(clone)
        merged_targets.append({
            "object_id": oid,
            "kind": _cp.JUMP_STILL_KIND,
            "role": clone["role"],
            "beat_id": beat,
        })
        report.append(
            "jump-still: beat %s segment %d -> %s (from %s)"
            % (beat, segment, oid, base.get("object_id")))
    return merged_objects, merged_targets, report


class _NormalizedPrompt(NamedTuple):
    """One prompt after the dispatcher's canonical mutation chain.

    ``pre_banana_hash`` and ``prompt_hash`` are DIFFERENT ON PURPOSE and both
    are load-bearing: the object write-back records the post-style/pre-banana
    text, while the ledger row and the cache key record the text actually
    rendered. Collapsing them is the obvious refactor and it is wrong.
    """

    text: str
    styled: bool
    pre_banana_hash: str
    banana_result: object
    banana_receipt: object
    prompt_hash: str
    base_prompt_hash: str = ""
    normalization_receipt: object = None


def normalize_prompt_for_render(raw_prompt, *, vstyle, banana_on, banana_key,
                                source="") -> _NormalizedPrompt:
    """THE canonical prompt normalization, factored so it has exactly ONE copy.

    WHY THIS IS A FUNCTION (2026-08-26). The seed that keeps a character's face
    stable across beats is derived from the character's PORTRAIT prompt hash --
    and that hash is the one computed HERE, after the safety clause, the style
    front-anchor and the banana transform, not the raw hash the producer
    stamped upstream (`otr_meta_brief_image_prompt.py`). A design that
    transported the producer's raw hash would have moved every face exactly
    once; it was caught in review before it shipped.

    So when a lane declares it needs identity but mints no portrait pixels, the
    basis must be derived by running the portrait's PROMPT TEXT through this
    same chain. That is only safe if there is ONE chain. A second copy in the
    producer would drift -- silently, and the symptom would be faces moving --
    which is why the dispatch loop below calls this too rather than keeping its
    own inline copy.

    Pure over its inputs. The caller owns the side effects the loop needs (the
    object write-back and the banana substitution counter), because a virtual
    portrait that is never rendered must NOT contribute to a rendered-work
    metric.
    """
    base_prompt = str(raw_prompt or "")
    base_hash = _prompt_content_hash(base_prompt)
    prompt = append_visual_safety_clause(base_prompt)
    _pre_style = prompt
    safety_hash = _prompt_content_hash(prompt)
    if vstyle is not None:
        # Imported HERE, with the same package/flat fallback the dispatch loop
        # uses, because `_otr_visual_styles` is not a module-level import in
        # this file -- the loop imports it inside a try/except so a style that
        # will not resolve cannot kill a render. A module-scope reference here
        # raises NameError on every styled object, which is exactly what it did
        # the first time this helper was written.
        try:
            from ._otr_visual_styles import prefix_style_cue  # type: ignore
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_visual_styles import prefix_style_cue  # type: ignore
        prompt = prefix_style_cue(vstyle, prompt)
    styled = prompt != _pre_style
    # The write-back hash: post-style, PRE-banana. See the NamedTuple docstring.
    pre_banana_hash = _prompt_content_hash(prompt) if styled else ""
    style_hash = _prompt_content_hash(prompt)
    if banana_on:
        bres = _banana.apply(
            prompt, variety_key=banana_key,
            shield_quoted_card_text=(str(source or "") == "still_word"))
        prompt = bres.text
        receipt = _banana.receipt_keys(bres)
    else:
        bres = None
        receipt = _banana.off_receipt(prompt, variety_key=banana_key)
    final_hash = _prompt_content_hash(prompt)
    normalization = {
        "hash_format": "sha256_json_string_ascii", "base_prompt_hash": base_hash,
        "safety_prompt_hash": safety_hash, "style_prompt_hash": style_hash,
        "final_prompt_hash": final_hash, "final_prompt": prompt,
        "safety_changed": safety_hash != base_hash, "style_changed": styled,
        "banana_changed": final_hash != style_hash,
        "final_changed": final_hash != base_hash,
        "source_qualified": False,
    }
    return _NormalizedPrompt(prompt, styled, pre_banana_hash, bres, receipt,
                             final_hash, base_hash, normalization)


def _current_source_context_hash(ledger, obj):
    """Rejoin the current frozen scene; an old payload cannot certify a new source."""
    if (str(obj.get("kind") or "") != "scene_character"
            and obj.get("source_scene_scope") != "scene_character"):
        return ""
    meta = ledger.get("meta") or {}
    if not isinstance(meta.get("my_story"), dict):
        return ""
    from .otr_meta_brief_image_prompt import _iter_beat_lines, _scene_source_context
    from ._otr_story_source import candidate_sha256
    lines = ledger.get("lines") or []
    line = next((row for bid, row in _iter_beat_lines(lines)
                 if bid == str(obj.get("beat_id") or "")), {})
    context = _scene_source_context(meta, ledger.get("cast") or [], lines, obj, line, ledger)
    return candidate_sha256(context) if context is not None else ""


def _source_rewrite_for_render(obj, normalized, source_context_hash):
    """Bind current source evidence to the actual dispatch input, never a cached claim."""
    original = obj.get("source_rewrite")
    if not isinstance(original, dict):
        return None
    receipt = json.loads(json.dumps(original, ensure_ascii=True))
    matched = (bool(source_context_hash)
               and receipt.get("base_prompt_hash") == normalized.base_prompt_hash
               and receipt.get("source_context_hash") == obj.get("source_context_hash")
               == source_context_hash)
    receipt.update({
        "base_matches_rewrite": matched,
        "dispatch_base_prompt_hash": normalized.base_prompt_hash,
        "dispatch_source_context_hash": source_context_hash,
        "final_prompt_hash": normalized.prompt_hash,
        "final_prompt": normalized.text,
        "normalization": normalized.normalization_receipt,
        "qualified": False,
        "dispatch_disposition": (
            "stale_source_receipt" if not matched else
            "transformed_after_source_operation" if normalized.normalization_receipt["final_changed"]
            else "source_operation_retained"),
    })
    return receipt


def _replay_rebase_to_live_episode(ledger: dict) -> dict:
    """Move a pre-rename replay snapshot onto the episode's current directory.

    Returns the ledger unchanged when nothing renamed it, when the singleton is
    unavailable, or on any failure -- a rebase that cannot be proven is not
    worth risking, and the caller's own existence check is the backstop.
    """
    if not isinstance(ledger, dict):
        return ledger
    try:
        from .production_ledger import (
            peek_ledger as _peek,
            _rebase_episode_local_paths as _rebase,
        )
        live = _peek()
        if live is None:
            return ledger
        stale_id = str(ledger.get("episode_id") or "").strip()
        live_id = str((live.data or {}).get("episode_id") or "").strip()
        if not stale_id or not live_id or stale_id == live_id:
            return ledger            # nothing renamed under us
        # PROVE THE SINGLETON IS THIS REPLAY BEFORE REWRITING ANY PATH.
        # Without this the function will happily rebase onto an UNRELATED
        # episode -- a Sonnet QA lane demonstrated exactly that by execution,
        # feeding a live ledger with no lineage to the wire one and watching
        # every image path move to a directory the files are not in. It is
        # not reachable today (one prompt at a time, one new_ledger per graph)
        # but "currently unreachable" is not a guarantee, and the cost of the
        # check is one string compare.
        #
        # ``import_replay_bundle`` stamps meta.replay_workspace_id and
        # rename_episode does not touch meta, so the two must agree.
        stale_ws = str(((ledger.get("meta") or {}) if isinstance(ledger.get("meta"), dict) else {}).get("replay_workspace_id") or "").strip()
        live_ws = str(((live.data or {}).get("meta") or {}).get("replay_workspace_id") or "").strip()
        if not stale_ws or not live_ws or stale_ws != live_ws:
            log.warning("[OTR_ImageGenDispatcher] REPLAY: the live ledger is "
                        "%r, not this replay (%r); refusing to rebase its "
                        "paths", live_id or "?", stale_id)
            return ledger
        # `<episode dir>/audio/<id>_ledger.json` -> the episode dir, then its parent
        new_ep_dir = os.path.dirname(os.path.dirname(str(live.path)))
        old_ep_dir = os.path.join(os.path.dirname(new_ep_dir), stale_id)
        if not os.path.isdir(new_ep_dir):
            return ledger
        rebased, n = _rebase(ledger, old_ep_dir, new_ep_dir)
        if not isinstance(rebased, dict):
            return ledger
        if n:
            log.info("[OTR_ImageGenDispatcher] REPLAY: rebased %d path(s) from "
                     "%s onto the renamed episode %s", n, stale_id, live_id)
        return rebased
    except Exception as exc:             # noqa: BLE001 -- never fatal
        log.warning("[OTR_ImageGenDispatcher] REPLAY: could not rebase the "
                    "imported ledger onto the renamed episode (%r); verifying "
                    "the paths as they arrived", exc)
        return ledger


def verify_replay_images(ledger: dict):
    """REPLAY (campaign item 0): every imported image row's file must exist at
    its rebased ``path`` with non-zero bytes; nothing is minted. Returns
    ``(ledger, image_done, report)`` and stamps the unchanged images section
    durably so the singleton and the wire agree."""
    # THE DICT ON THE WIRE CAN PREDATE ``rename_episode`` (found on the 4060,
    # 2026-09-23). A replay workspace is ALWAYS renamed to a fresh episode id,
    # and the rename moves the whole per-episode directory and rebases the
    # durable ledger -- but the snapshot travelling down the graph was captured
    # before that, so every row still points into the pre-rename directory that
    # no longer exists. The failure reads as "27 imported image row(s) have no
    # file on disk" while all 27 files are present and the saved ledger is
    # correct, which sends the reader hunting for missing assets that were
    # never missing. It would fire on EVERY replay that reaches this node.
    #
    # Rebase against the live singleton with the same walker the rename itself
    # used, so this cannot drift from that behaviour.
    ledger = _replay_rebase_to_live_episode(ledger)
    images = ledger.get("images") if isinstance(ledger.get("images"), dict) else {}
    rows = images.get("images") if isinstance(images.get("images"), list) else []
    missing = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw = str(row.get("path") or "")
        # Statted only inside the approved folders (2026-10-18): a row naming
        # anything else counts as missing rather than probing the disk, and the
        # stat reads the checked spelling, not the raw one.
        path = _confined_or_empty(raw) if raw else ""
        if not (path and os.path.isfile(path) and os.path.getsize(path) > 0):
            missing.append(str(row.get("image_id") or raw or "?"))
    if missing:
        raise RuntimeError(
            "[OTR_ImageGenDispatcher] REPLAY: %d imported image row(s) have no file on "
            "disk: %s" % (len(missing), ", ".join(missing[:6])))
    try:
        from .production_ledger import stamp_durable as _stamp_durable
        _stamp_durable(sections={"images": images}, source="OTR_ImageGenDispatcher:replay")
    except ImportError:  # pragma: no cover
        pass
    report = "replay: %d imported image row(s) verified on disk, nothing minted" % len(rows)
    log.warning("[OTR_ImageGenDispatcher] %s", report)
    return ledger, "image_done:replay", report


def _pending_holds_portrait(pending, char_id) -> bool:
    cid = str(char_id or "")
    if not cid:
        return False
    for job in pending:
        if str(job.get("kind") or "") != "portrait":
            continue
        if str(job.get("oid") or "") == cid or str(job.get("char_id") or "") == cid:
            return True
    return False


def _still_engine_refused(exc, engine_id=None) -> bool:
    """True when the STILL engine declined the card, however it said so.

    TWO DIALECTS, ONE VERDICT. The local and Google engines mark a refusal by
    setting ``is_model_refusal`` on the exception (eng_google_image,
    ideogram4_local). A PARTNER engine cannot: it raises a
    ``CloudMediaError`` stamped ``CONTENT_REFUSED`` at the cloud invoke
    boundary and carries no such attribute, so every cloud still refusal used
    to skip the sanctioned-gap path entirely and re-raise as "NO FALLBACK --
    fix the engine" -- taking the whole fan-out wave, already submitted and
    already paid, down with it. Found 2026-09-16 alongside the video-side
    twin; the two funnels now read the same verdict the same way.

    THE ENGINE GATE IS NOT DECORATION, and the video funnel had it from the
    start while this one did not. ``is_content_refusal`` falls back to prose
    when NOTHING in the chain carries a stamped code -- which is exactly the
    shape of a LOCAL engine crash. So a local Flux or SDXL traceback whose
    text happened to contain a needle would have been recorded as a
    sanctioned model refusal and published as a degraded episode. A local
    engine keeps its explicit ``is_model_refusal`` dialect and nothing else.
    """
    if getattr(exc, "is_model_refusal", False):
        return True
    if engine_id is not None and not _is_cloud_image_engine(engine_id):
        return False
    try:
        from ._otr_shared.cloud_media_backend import is_content_refusal
    except Exception:  # noqa: BLE001 -- classification must never itself raise
        return False
    return is_content_refusal(exc)


def _render_still_pixels(job, ctx):
    """Partner/local gen_fn -> pixels. Model refusal is a result, not a raise."""
    gen_fn = ctx["gen_fn"]
    try:
        pixels = _coerce_pixels(
            gen_fn(job["request"]),
            min_bytes=ctx["handoff_min_bytes"],
            wait_attempts=ctx["handoff_wait_attempts"],
            wait_sleep_s=ctx["handoff_wait_sleep_s"],
        )
    except Exception as exc:  # noqa: BLE001 -- split refusal vs engine fault
        if _still_engine_refused(exc, job.get("engine_id")):
            return {"refusal": exc}
        raise
    return {
        "pixels": pixels,
        "content_hash": _pl.compute_portrait_hash(pixels),
        "laplacian": _laplacian_variance(pixels),
    }


def _cloud_still_job_failure(exc, engine_id=None):
    """The job-scoped cloud failure code for a still, or "".

    The still-funnel twin of ``render_driver._cloud_floor_reason``, and it
    reads the same stamped codes: one object failed on the provider side, and
    the next object is unaffected. Run-scoped codes (AUTH, BUDGET, the config
    four) and unstamped exceptions get nothing -- those still raise.
    """
    if engine_id is not None and not _is_cloud_image_engine(engine_id):
        return ""
    try:
        from ._otr_shared.cloud_media_backend import cloud_job_failure_code
    except Exception:  # noqa: BLE001 -- classification must never itself raise
        return ""
    code = cloud_job_failure_code(exc)
    return code.value if code is not None else ""


def _record_still_refusal(job, exc, ctx, reason=None):
    oid = job["oid"]
    engine_id = job["engine_id"]
    prompt = job["prompt"]
    seed = job["seed"]
    reason = str(reason or _receipt.SANCTIONABLE_SKIP_REASON)
    is_refusal = reason == _receipt.SANCTIONABLE_SKIP_REASON
    headline = "MODEL REFUSAL" if is_refusal else "CLOUD JOB FAILED"
    ctx["skip_evidence_by_oid"][oid] = {
        "reason": reason, "role": job["role"], "engine_id": engine_id,
        "prompt": prompt, "seed": seed, "detail": str(exc),
    }
    ctx["warnings"].append(
        f"{oid}: '{engine_id}' {headline} -- no still for this "
        f"object; the episode continues (operator 2026-08-22). "
        f"prompt={prompt!r} seed={seed} ({exc})")
    log.warning(
        "[OTR_ImageGenDispatcher] %s on %s via '%s': %s\n"
        "  prompt: %s\n  negative: %s\n  seed: %s\n"
        "  The episode CONTINUES with no still for this object. This "
        "prompt is recorded so the outcome can be diagnosed as seed-, "
        "content- or provider-driven -- the previous hard-fail erased it.",
        headline, oid, engine_id, exc, prompt, job["effective_neg"], seed)


def _commit_minted_still(job, pixels, content_hash, lap, ctx):
    oid = job["oid"]
    kind = job["kind"]
    role = job["role"]
    path = _pl.stamp_portrait(
        ctx["ledger"], oid, pixels, output_dir=ctx["output_dir"],
        require_cast_entry=(kind == "portrait" and oid in ctx["cast_ids"]))
    try:
        ep_path = _materialize_episode_copy(
            str(path), ctx["ep_dir"], oid, content_hash)
    except OSError as exc:
        ctx["warnings"].append(
            f"{oid}: episode materialization failed ({exc}); row points "
            "at the pool copy (LOUD)")
        ep_path = str(path)
    image_id = f"img_{oid}_{content_hash[:12]}"
    row = {
        "image_id": image_id, "role": role, "object_id": oid,
        "kind": kind,
        "path": ep_path, "pool_path": str(path),
        "engine_id": job["engine_id"], "engine_version": job["eng_version"],
        "request_hash": job["key"], "portrait_content_hash": content_hash,
        "content_hash": content_hash, "w": job["obj_w"], "h": job["obj_h"],
        "prompt_hash": job["prompt_hash"],
        "provenance": {"source": job["obj"].get("source", "")},
        "derived_from_portrait_hash": job["anchor_hash"],
        "portrait_anchor_mode": job["anchor_mode"],
        "identity_seed_basis": job["seed_basis_used"],
    }
    row.update(job["banana_rcpt"])
    row["source_context_hash"] = job["source_context_hash"]
    row["source_rewrite"] = job["source_rewrite"]
    if job["char_id"]:
        row["char_id"] = job["char_id"]
    if job["beat_id"]:
        row["beat_id"] = job["beat_id"]
    if job["radio_host_style"]:
        row["radio_host_style"] = job["radio_host_style"]
    if job.get("identity_photo"):
        row["identity_photo"] = job["identity_photo"]
    if job.get("identity_photo_more"):
        row["identity_photo_more"] = list(job["identity_photo_more"])
    if job["lettering_style"]:
        row["lettering_style"] = job["lettering_style"]
    if job["backdrop_family"]:
        row["backdrop_family"] = job["backdrop_family"]
    _msid = str(job["obj"].get("mesh_subject_id") or "")
    if _msid:
        row["mesh_subject_id"] = _msid
    _vstyle = ctx["vstyle"]
    if _vstyle is not None:
        row["visual_style"] = str(getattr(_vstyle, "style_id", "") or "")
    if lap:
        row["laplacian"] = lap
    ctx["visual_rows"].append({
        "kind": kind, "object_id": oid, "role": role,
        "source": job["source"], "beat_id": job["beat_id"],
        "styled": bool(job["styled_now"]),
        "already_styled": bool(ctx["style_cue"]) and not job["styled_now"],
        "prompt_sha8": str(job["prompt_hash"] or "")[:8],
        "negative": job["effective_neg"],
        "negative_source": job["neg_source"],
        "laplacian": lap,
    })
    ctx["images"].append(row)
    ctx["ep_rows"].append(row)
    ctx["cache_index"][job["key"]] = image_id
    ctx["made"] += 1
    ctx["report"].append(f"{oid}: generated -> {os.path.basename(ep_path)}")


def _wrap_still_render_error(job, exc):
    oid = job["oid"]
    engine_id = job["engine_id"]
    if isinstance(exc, ImageRenderError):
        return exc
    if isinstance(exc, _lease.LeaseTimeout):
        return ImageRenderError(
            f"{oid}: GPU lease timeout rendering '{engine_id}' ({exc}). "
            f"NO FALLBACK.")
    if isinstance(exc, ImageHandoffTimeout):
        return ImageRenderError(
            f"{oid}: image handoff from '{engine_id}' not ready ({exc}). "
            f"NO FALLBACK.")
    return ImageRenderError(
        f"{oid}: image render with '{engine_id}' failed "
        f"({type(exc).__name__}: {exc}). NO FALLBACK -- fix the "
        f"engine or select a usable one.")


def _mint_still_serial(job, ctx, *, take_lease):
    lease = None
    try:
        if take_lease:
            lease = _lease.acquire(
                timeout_s=ctx["lease_timeout_s"], lockdir=ctx["lockdir"])
        packed = _render_still_pixels(job, ctx)
        if packed.get("refusal"):
            _record_still_refusal(job, packed["refusal"], ctx)
            return
        _commit_minted_still(
            job, packed["pixels"], packed["content_hash"],
            packed["laplacian"], ctx)
    except Exception as exc:  # noqa: BLE001
        if getattr(exc, "is_model_refusal", False):
            _record_still_refusal(job, exc, ctx)
            return
        if _cloud_still_job_failure(exc, job.get("engine_id")):
            _record_still_refusal(
                job, exc, ctx, reason=_receipt.CLOUD_JOB_SKIP_REASON)
            return
        wrapped = _wrap_still_render_error(job, exc)
        if wrapped is exc:
            raise
        raise wrapped from exc
    finally:
        if lease is not None:
            _lease.release(lease)
    if take_lease and not _lease.wait_until_below_mb(
            15000, attempts=3, sleep_s=0.0):
        log.info("[OTR_ImageGenDispatcher] NVML re-probe inconclusive (CPU/no-NVML or busy)")


def _flush_pending_cloud_stills(pending, ctx):
    if not pending:
        return
    from ._otr_shared.cloud_fanout import (
        cloud_fanout_workers, run_cloud_fanout, snapshot_prompt_id,
    )
    jobs = list(pending)
    pending.clear()
    workers = min(cloud_fanout_workers(), len(jobs))
    log.warning(
        "[OTR image] cloud fan-out: %d still(s), workers=%d; "
        "assemble stays object order",
        len(jobs), workers)
    if workers <= 1:
        for job in jobs:
            _mint_still_serial(job, ctx, take_lease=False)
        return
    outcome = run_cloud_fanout(
        jobs,
        item_id=lambda j: str(j.get("oid") or ""),
        execute=lambda j: _render_still_pixels(j, ctx),
        workers=workers,
        prompt_id=snapshot_prompt_id())
    for job in jobs:
        oid = job["oid"]
        if oid in outcome.errors:
            # A PROVIDER-SIDE FAILURE IS ONE OBJECT, NOT THE WAVE. Every still
            # in this fan-out was submitted and paid before the walk begins,
            # so raising here discarded all of them for one bad job. Only
            # job-scoped codes floor; an unstamped crash still fails LOUD.
            _why = _cloud_still_job_failure(
                outcome.errors[oid], job.get("engine_id"))
            if _why:
                _record_still_refusal(
                    job, outcome.errors[oid], ctx,
                    reason=_receipt.CLOUD_JOB_SKIP_REASON)
                continue
            raise _wrap_still_render_error(job, outcome.errors[oid]) from outcome.errors[oid]
        packed = outcome.results.get(oid)
        if packed is None:
            raise ImageRenderError(
                "cloud still fan-out never rendered %s" % oid)
        if packed.get("refusal"):
            _record_still_refusal(job, packed["refusal"], ctx)
            continue
        _commit_minted_still(
            job, packed["pixels"], packed["content_hash"],
            packed["laplacian"], ctx)


def _kept_row(images, oid):
    """The newest row for ``oid`` whose still is on disk, for a KEPT card: its
    verified bytes are reused as they are. Keep never vouches for a missing
    file -- the newest row with no file on disk means there is nothing to keep."""
    for row in reversed(images):
        if isinstance(row, dict) and str(row.get("object_id") or "") == oid:
            path = _confined_or_empty(str(row.get("path") or ""))
            if path and os.path.isfile(path) and os.path.getsize(path) > 0:
                return row
            return None
    return None


def _one_row_per_object(images):
    """A Scene Producer run keeps ONE row per object, the newest replacing the
    older ones, so the render driver's first-row-wins index needs no change.

    Returns ``(rows, superseded)``. The superseded rows are not thrown away:
    the cache index still names their stills, and a picture edit that is undone
    -- a note cleared, a text restored -- is a cache hit on exactly those rows,
    so the run keeps them beside the images for the cache to find."""
    newest = {}
    for index, row in enumerate(images):
        if isinstance(row, dict) and row.get("object_id"):
            newest[str(row["object_id"])] = index
    keep = set(newest.values())
    rows, superseded = [], []
    for index, row in enumerate(images):
        if isinstance(row, dict) and row.get("object_id") and index not in keep:
            superseded.append(row)
        else:
            rows.append(row)
    return rows, superseded


def _draws_partner_heartbeat(engine_id) -> bool:
    """True when this image engine renders through a Comfy partner node, whose
    wait loop redraws the executing node's bar with its own heartbeat
    (``cloud_media_invoke._emit_heartbeat``).

    The id alone does not say so (Codex QA, 2026-10-22): ``ideo`` is a partner
    engine whose node key is ``cloud_ideogram_v4``. So the registered engine's
    node key decides too -- the partner half of :func:`_is_cloud_image_engine`.
    A direct API engine (``native=False`` with no partner node) draws nothing."""
    eid = str(engine_id or "")
    if eid.startswith("cloud_"):
        return True
    eng = _safe_engine(eid)
    return eng is not None and str(getattr(eng, "node_key", "") or "").startswith("cloud_")


def _stage_bar_items(objects, image_policy) -> int:
    """How many items the Scene Producer's stage bar counts: every object, or
    none when any of them renders on a partner engine
    (:func:`_draws_partner_heartbeat`). A partner call's wait loop redraws this
    node's bar with its own heartbeat, from fan-out worker threads the quiet
    does not reach, and two bars on one node fight -- the same rule the voice
    loop keeps (``cloud_fanout.adapter_draws_partner_heartbeat``). A role whose
    slot cannot resolve is left to the loop below, which refuses it in its
    place."""
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        try:
            engine_id, _slot, _fell_back = resolve_engine_for_role(
                image_policy, str(obj.get("role") or "character_video"))
        except ValueError:
            continue
        if _draws_partner_heartbeat(engine_id):
            return 0
    return len(objects)


def dispatch_images(ledger: dict, image_policy: dict, image_prompts: dict, *,
                    gen_fn=None, output_dir=None, lockdir=None, lease_timeout_s=120.0,
                    handoff_min_bytes: int = _MIN_PNG_BYTES,
                    handoff_wait_attempts: int = 40, handoff_wait_sleep_s: float = 0.05,
                    episode_id: str = "", quiet_inner_bars=None):
    """Generate (or cache-reuse) every image OBJECT (portraits + scene
    stills); stamp the ledger.

    ``quiet_inner_bars`` is given only on a Scene Producer run: an UNOPENED
    ``QuietInnerBars``. This node then draws one stage bar over its objects and
    opens the quiet once that bar exists, so the samplers inside each still
    cannot restart it. The CALLER closes it, on every path.

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

    (A STALE policy that still carries the retired ``locked_3d_slots`` field is
    simply ignored: the capability it guarded can no longer be declared.)
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
    warnings: list = []
    report: list = []
    # THE SCENE PRODUCER'S PAYLOAD (plan, "Stills"). On a producer run the
    # payload node 89 handed the FIRST stills run is kept as it arrived, BEFORE
    # this node's own merge adds jump stills, as ``images.producer_prompt_payload``,
    # and every later stills run keeps that same one: node 89 rebuilds each
    # run's prompts from it and from the raw texts kept beside it, which this
    # run updates with the direction's edits. On every other run nothing is
    # kept and nothing changes.
    from . import _otr_producer_direction as _PD
    _producer = _PD.runtime_direction(ledger)
    _producer_payload = _raw_texts = None
    _kept_stills: set = set()
    if _producer is not None:
        _arrived = ledger.get("images") if isinstance(ledger.get("images"), dict) else {}
        _saved = _arrived.get(_PD.PAYLOAD_KEY)
        if isinstance(_saved, dict):                  # a stills REPLAY
            # The policy this node actually receives must be the one the
            # project's stills were painted with: node 63 checked the
            # directors' widgets, this checks whatever is wired here.
            _drift = _PD.check_policy_matches_stamp(image_policy, ledger.get("meta"))
            if _drift:
                raise ImageRenderError(
                    "Scene Producer stills replay refused: the image policy wired to this "
                    "node differs from the one the project's stills were painted with: %s"
                    % _drift)
            _producer_payload = copy.deepcopy(_saved)
            _raw_texts = _PD.merged_raw_texts(_arrived.get(_PD.RAW_TEXTS_KEY), _producer)
        else:                                         # the project's first stills run
            _producer_payload = copy.deepcopy(image_prompts)
            _raw_texts = _PD.initial_raw_texts(image_prompts)
        _kept_stills = set(_producer["kept"]["stills"])
    objects = (image_prompts or {}).get("objects") or []
    required_scene_targets = (image_prompts or {}).get(
        "required_scene_targets")
    if required_scene_targets is None:
        required_scene_targets = []
    if not isinstance(required_scene_targets, list):
        raise ImageRenderError(
            "OTR_ImageGenDispatcher received a malformed required_scene_targets "
            "receipt; refusing video dispatch without a target contract.")
    # THE JUMP-STILL MERGE (2026-07-25, multi-clip coverage chunk 4). ShotLock
    # ran BEFORE this node and stamped one still request per jump-cut segment
    # on its shot rows; fold them into the object list and the required-target
    # receipt HERE, before either is validated below, so the merged rows are
    # held to exactly the same id/duplicate/completion contract as the
    # producer's own. Empty and free on every single-clip episode, which is
    # every episode until an adapter opts in to multi-clip coverage.
    objects, required_scene_targets, _jump_report = merge_jump_still_requests(
        ledger, objects, required_scene_targets)
    report.extend(_jump_report)
    # The coverage-plan module was previously bound only inside
    # merge_jump_still_requests. The object loop below needs JUMP_STILL_KIND to
    # decide which rows carry a portrait anchor, so bind it here too -- the same
    # lazy, guarded import, no new dependency.
    _cp = _coverage_plan_module()
    _required_ids = []
    for _target in required_scene_targets:
        if not isinstance(_target, dict) or not str(
                _target.get("object_id") or ""):
            raise ImageRenderError(
                "OTR_ImageGenDispatcher received a required scene target "
                "without an object_id; refusing an unverifiable image phase.")
        _required_ids.append(str(_target["object_id"]))
    if len(_required_ids) != len(set(_required_ids)):
        raise ImageRenderError(
            "OTR_ImageGenDispatcher received duplicate required scene target "
            "ids; refusing an ambiguous image receipt.")
    still_capabilities = still_consumer_capabilities(image_policy)
    if objects and still_capabilities is None:
        raise ImageRenderError(
            "OTR_ImageGenDispatcher cannot prove an image consumer: image_policy "
            "must carry a video_models map before any still is rendered. "
            "Refusing image generation rather than assuming a consumer."
        )
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
    # Bound AFTER the re-key, which may rebuild the wire's sections to move
    # their paths; a binding taken earlier would hold the pre-move rows.
    cast = ledger.get("cast") if isinstance(ledger.get("cast"), list) else []
    images_section = ledger.get("images") if isinstance(ledger.get("images"), dict) else {}
    cache_index = dict(images_section.get("cache_index") or {})
    images = list(images_section.get("images") or [])
    # A producer run's earlier stills of each object (``_one_row_per_object``):
    # empty on every other run.
    _superseded = (list(images_section.get(_PD.SUPERSEDED_KEY) or [])
                   if _producer is not None else [])
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
    # THE BANANA ROUTE (2026-08-06 build spec). Gate and
    # variety key are per-EPISODE facts, resolved once: env switch + the
    # fidelity-bank idiom on meta.source_bank; variety keyed on the immutable
    # freeze_timestamp so re-rendering a frozen ledger reproduces the same
    # fruits (coherence is structural, never per-re-render).
    _banana_meta = (ledger.get("meta") or {}) if isinstance(ledger, dict) else {}
    _banana_on = _banana.banana_gate(_banana_meta, lane="stills")
    _banana_key = str(_banana_meta.get("freeze_timestamp") or "")
    #: CANDIDATE substitutions: incremented as each prompt is transformed, which
    #: happens BEFORE the consumer/engine skip decisions below, so this counts
    #: transformed candidates that may never mint a row. It is a summary log
    #: metric only -- the durable per-row receipts are exact -- and the name
    #: says so rather than the number quietly meaning something else.
    _banana_candidate_subs = 0
    #: object_id -> why this object was SKIPPED without a ledger row. Keyed, not
    #: substring-matched against free-text warnings: "still_b1" is a prefix of
    #: "still_b12", so matching on text would cross-associate their evidence.
    #: Read by the completion contract below, which raises BEFORE the images
    #: section (and its warnings) is stamped to the ledger.
    skip_evidence_by_oid: dict = {}
    # ONE STYLE AUTHORITY (PBUG-20260817-01). This node is the only one holding
    # BOTH prompt families' authors upstream of it -- ShotLock (video) and
    # MetaBrief (stills) -- and it is upstream of every mint, so it is where the
    # still family gets its style FRONT-anchored and where the pack's own
    # negative is composed. Video prompts keep their existing prepend in
    # render_driver, which already front-anchors the identical token.
    try:
        try:
            from ._otr_visual_styles import (  # type: ignore
                get_visual_style, prefix_style_cue, compact_style_cue,
                effective_negative)
        except ImportError:  # pragma: no cover -- flat test imports
            from _otr_visual_styles import (  # type: ignore
                get_visual_style, prefix_style_cue, compact_style_cue,
                effective_negative)
        _vstyle = get_visual_style(ledger.get("meta") or {})
    except Exception as exc:  # noqa: BLE001
        # A style that will not resolve must not kill a render: the mint simply
        # keeps today's behaviour (no prefix, engine hygiene negative).
        _vstyle = None
        warnings.append(
            f"style authority: visual style unresolved ({exc}); stills mint "
            f"without a style prefix or pack negative"
        )
    _style_cue = compact_style_cue(_vstyle) if _vstyle is not None else ""
    # The pack's negative with any phrase the pack's OWN positive asks for
    # removed (operator ruling 2026-08-17: a negative may never conflict with a
    # visual style). `_pack_negative_authored` is kept beside it so the ledger
    # can show what the pack declared AND what actually conditioned the mint.
    _pack_negative_authored = str(getattr(_vstyle, "negative_tail", "") or "")
    _pack_negative = (effective_negative(_vstyle) if _vstyle is not None else "")
    if _pack_negative != _pack_negative_authored:
        log.info("[OTR.image.style_authority] pack %r self-veto resolved: "
                 "authored=%r effective=%r",
                 str(getattr(_vstyle, "style_id", "") or ""),
                 _pack_negative_authored, _pack_negative)
    #: The VISUAL LEDGER (operator-authorized 2026-08-17). One durable row per
    #: visual prompt: what style was in force, whether this pass had to add it,
    #: and the measured scalar. Before this, the final still prompts existed
    #: only on the wire between two nodes and were never persisted, so nothing
    #: on disk recorded what was actually rendered.
    _visual_rows: list = []
    # Leg C4b (4060 clean room, 2026-09-02): the writer LLM that composed the still
    # prompts moments earlier was STILL on the card when the first local still rendered
    # (48 MB free, the encoder's VBAR at 0 resident pages), so the image engine had no
    # VRAM to load into. The ghost lane already releases the writer before its image
    # phase; the general path did not. Freed ONCE per dispatch, right before the first
    # LOCAL render (cloud adapters never touch the local GPU), through the same
    # canonical call the LTX 2.5 engine makes in its preflight.
    # Nothing after the image stage requests an LLM slot, so there is no reload cost.
    _residue_freed = False
    pending_cloud = []
    ctx = {
        "ledger": ledger,
        "output_dir": output_dir,
        "ep_dir": ep_dir,
        "images": images,
        "ep_rows": ep_rows,
        "cache_index": cache_index,
        "warnings": warnings,
        "report": report,
        "skip_evidence_by_oid": skip_evidence_by_oid,
        "visual_rows": _visual_rows,
        "gen_fn": gen_fn,
        "handoff_min_bytes": handoff_min_bytes,
        "handoff_wait_attempts": handoff_wait_attempts,
        "handoff_wait_sleep_s": handoff_wait_sleep_s,
        "lockdir": lockdir,
        "lease_timeout_s": lease_timeout_s,
        "cast_ids": cast_ids,
        "style_cue": _style_cue,
        "vstyle": _vstyle,
        "made": 0,
    }
    # THE SCENE PRODUCER'S STAGE BAR (plan, "Progress"). On a producer run this
    # node draws ONE bar over its objects -- portraits and scene stills alike --
    # and the samplers inside each still are quieted, so they cannot restart it
    # once a still; Radio Play Studio reads this node's step and total as the
    # stage bar. The node's own bar is built FIRST, so it keeps the real hook
    # (a bar reads the hook when it is built). No bar when a partner engine
    # draws its heartbeat here; on an ordinary run, no bar and no quiet at all.
    _stage_bar = _NodeProgress(
        _stage_bar_items(objects, image_policy) if quiet_inner_bars is not None else 0,
        "Scene Producer stills")
    if quiet_inner_bars is not None and _stage_bar.total:
        quiet_inner_bars.open()
    for _object_index, obj in enumerate(objects):
        _stage_bar.at(_object_index)
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("object_id") or "")
        kind = str(obj.get("kind") or "portrait")
        role = str(obj.get("role") or "character_video")
        char_id = str(obj.get("char_id") or "")
        beat_id = str(obj.get("beat_id") or "")
        # A KEPT CARD keeps the still it has (plan, "Stills"): its verified row
        # is carried as it is, whatever bulk scene or character edit has changed
        # its prompt since. Every picture of the card's beat is kept with it.
        if (_kept_stills and beat_id and kind != "portrait"
                and "shot_%s" % beat_id in _kept_stills):
            _kept = _kept_row(images, oid)
            if _kept is None:
                raise ImageRenderError(
                    "Scene Producer keeps the still of card shot_%s, but %s has no "
                    "verified still on disk to keep" % (beat_id, oid))
            # Marked kept, so its old digest is not stamped over below, and
            # still saying where its pixels came from (Sonnet QA, 2026-10-22):
            # the provenance of the paint it keeps, through any earlier keep.
            _was = _kept.get("provenance") if isinstance(_kept.get("provenance"), dict) else {}
            _kept = dict(_kept, provenance={
                "source": "kept",
                "painted": _was.get("painted", {}) if _was.get("source") == "kept" else _was})
            images.append(_kept)
            ep_rows.append(_kept)
            reused += 1
            # A kept still was painted by its engine: the histogram (credits,
            # and a replay's readiness, image_engines_used) must still name it,
            # or un-keeping it later would call an engine nothing planned.
            _kept_engine = str(_kept.get("engine_id") or "")
            if _kept_engine:
                _kept_role = str(_kept.get("role") or role or "?")
                img_hist.setdefault(_kept_role, {})
                img_hist[_kept_role][_kept_engine] = (
                    img_hist[_kept_role].get(_kept_engine, 0) + 1)
            report.append(f"{oid}: kept -> {os.path.basename(str(_kept.get('path') or ''))}")
            continue
        # Composition provenance -- the banana quote-shield discriminator below.
        source = str(obj.get("source") or "")
        # RADIO FACE LOGIC (2026-07-04): the announcer portrait's chosen radio-host
        # style rides onto the dispatched ledger row so the render-side guard can
        # fail CLOSED on a faceless still about to feed a HuMo announcer init.
        radio_host_style = str(obj.get("radio_host_style") or "")
        # still_word v2 (2026-07-04): the LOCKED per-episode lettering + backdrop
        # family propagate onto the row for operator QA.
        lettering_style = str(obj.get("lettering_style") or "")
        backdrop_family = str(obj.get("backdrop_family") or "")
        # THE canonical mutation chain (safety clause -> style front-anchor ->
        # banana -> hash) now lives in ONE place, `normalize_prompt_for_render`
        # above, because the identity-seed basis has to be derived through the
        # exact same chain for a lane that mints no portrait pixels. Read that
        # function's docstring before changing anything here: a second copy of
        # this chain drifts, and the symptom of drift is faces moving.
        #
        # STYLE, FRONT-ANCHORED (ONE STYLE AUTHORITY). Additive only, and
        # positional rather than membership-based on purpose: finish_visual_prompt
        # already appends the style TAIL, so an "is it present" test would find
        # it and skip exactly the prompts that fractured. Runs BEFORE the banana
        # transform and the content hash, so the stored hash describes the
        # text actually rendered. Portraits are deliberately included -- the
        # measured fracture propagated FROM a portrait via reference_latent.
        _norm = normalize_prompt_for_render(
            obj.get("prompt"), vstyle=_vstyle, banana_on=_banana_on,
            banana_key=_banana_key, source=source)
        prompt = _norm.text
        source_context_hash = _current_source_context_hash(ledger, obj)
        source_rewrite = _source_rewrite_for_render(obj, _norm, source_context_hash)
        _styled_now = _norm.styled
        # THE NEGATIVE FOR THIS ROW, resolved once and recorded (operator
        # 2026-08-17: "lock them in the ledger"). Composed, never precedence --
        # the pack half is about STYLE and the object half (radio_host_negative)
        # is about keeping the announcer radio-object FACELESS, so letting
        # either win silently drops the other.
        _obj_negative = str(obj.get("negative_prompt") or "")
        _effective_neg = visual_safety_negative(
            ", ".join(t for t in (_pack_negative, _obj_negative) if t))
        # Named from what ACTUALLY contributed, per row. An episode-level label
        # read off the pack alone reported the empty case on announcer stills of a
        # dynamic-pack episode, where the pack is empty by design but the object
        # negative is real and is what conditioned the mint.
        #
        # COMPOSITION ONLY, and the one place that names it is
        # `negative_source_label` -- read its docstring before changing a value.
        _neg_source = negative_source_label(_pack_negative, _obj_negative)
        if _styled_now:
            # Provenance: MetaBrief stamped its own prompt_hash upstream for its
            # report. The dispatch cache key is recomputed from prompt TEXT
            # below and never trusts this field, but leaving it describing
            # pre-style text would make the ledger contradict itself.
            # POST-STYLE, PRE-BANANA -- a different hash from `prompt_hash`.
            obj["prompt_hash"] = _norm.pre_banana_hash
        # Banana ran BEFORE the content hash inside the helper, so flipping the
        # switch re-mints every cached still instead of serving a stale gun.
        #
        # Quote shielding is SCOPED to card prompts, and `source` is the
        # discriminator the helper is handed. `source` is stamped at COMPOSITION
        # (derive_image_prompts writes "still_word" on the card objects), which
        # is why it -- and not `kind` or `role` -- is used: a card's `kind` is
        # inherited from the scene target it replaced, and its `role` can drift
        # under OTR_FORCE_ENGINE_MAP between derive and dispatch. On a card the
        # quoted span is script (rendered as picture text in word mode; spoken
        # and shown in the credits in music mode); everywhere else a quote is
        # decoration, and shielding it let a writer-styled
        # `a man carrying a "revolver"` survive untransformed.
        #
        # The substitution COUNTER stays here, in the loop, and is fed only by
        # really-dispatched objects: a virtual identity-basis normalization must
        # never inflate a rendered-work metric.
        if _norm.banana_result is not None:
            _banana_candidate_subs += _norm.banana_result.substitutions
        banana_rcpt = _norm.banana_receipt
        prompt_hash = _norm.prompt_hash
        obj_w = int(obj.get("w") or 0)
        obj_h = int(obj.get("h") or 0)
        if not oid:
            continue
        # `oid` becomes a FILENAME (_materialize_episode_copy) and a ledger key.
        # It arrives from the workflow, so a separator or traversal token in it
        # is a location, not a name (2026-09-05). Blacklist, not whitelist: live
        # ids include still_b001, still_music_opening_001, c01 and announcer,
        # and a portrait id can inherit an LLM-authored char_id.
        if any(tok in oid for tok in ("/", "\\", "..", "\x00", ":")):
            raise ImageRenderError(
                "image object_id %r contains a path separator or traversal "
                "token; ids name a still, never a location." % (oid,))
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
        # tri-state decision accounts for the force map, so
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
            # The wire-only warning above is discarded by the completion gate's
            # own raise (the gate runs BEFORE the images section is stamped), so
            # the reason must reach the server log at skip time and the
            # structured map at the same moment.
            skip_evidence_by_oid[oid] = {
                "reason": "no_engine", "role": role, "slot": slot,
            }
            log.warning(
                "[OTR_ImageGenDispatcher] SKIP %s (role=%s slot=%s): no image "
                "engine selected; fail-closed, no still generated",
                oid, role, slot)
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
            # THE BRANCH THAT COST AN EPISODE AND COULD NOT SAY SO (2026-08-04).
            # This skip was invisible three ways: the warning carried no object
            # id (so nothing could correlate it), it was wire-only (and the
            # completion gate raises before the wire is stamped), and nothing
            # reached the server log. A 320-word leg lost two stills here and
            # the evidence was gone by the time anyone looked.
            warnings.append(f"{oid}: {exc}")
            # `exc` already carries the guard's verdict; reuse it rather than
            # re-running the (pure) predicate and its hash a second time.
            hit = getattr(exc, "path_guard_hit", None) or {}
            skip_evidence_by_oid[oid] = dict(hit, reason="prompt_path_guard",
                                             role=role, engine_id=engine_id)
            log.warning(
                "[OTR_ImageGenDispatcher] SKIP %s (role=%s engine=%s): prompt "
                "tripped the path guard -- arm=%s token=%r at index %s of %s, "
                "prompt_hash=%s, excerpt %s. NO still generated.",
                oid, role, engine_id, hit.get("arm"), hit.get("token"),
                hit.get("index"), hit.get("prompt_len"),
                hit.get("prompt_hash"), hit.get("excerpt"))
            continue
        # Resolve the character's PORTRAIT row once, before the seed and the
        # cache key. Read off the IMAGES list, never the cast row: stamp_portrait
        # writes portrait_content_hash only on a fresh render, so on a cache HIT
        # the cast lookup returns None -- exactly where the anchor matters most.
        # reversed() so a stale row from an earlier image_revision cannot win.
        if _pending_holds_portrait(pending_cloud, char_id):
            _flush_pending_cloud_stills(pending_cloud, ctx)
        portrait_row = None
        if kind in ("scene_character", _cp.JUMP_STILL_KIND) and char_id:
            portrait_row = next(
                (r for r in reversed(images)
                 if isinstance(r, dict) and r.get("kind") == "portrait"
                 and str(r.get("object_id") or "") == char_id),
                None,
            )
        anchor_hash = str((portrait_row or {}).get("portrait_content_hash") or "")

        # THE IDENTITY BASIS -- the hash the seed derives a character's face
        # from. Prefer the real portrait row; otherwise derive it from the
        # portrait PROMPT TEXT the producer transported.
        #
        # It MUST go through `normalize_prompt_for_render`, the same chain the
        # dispatch loop runs, because the hash on a portrait row is computed
        # AFTER the safety clause, the style front-anchor and the banana
        # transform -- not from the raw text. An earlier cut of this fix
        # transported the producer's RAW hash and would have moved every face
        # exactly once; that is why there is one shared normalizer and why the
        # keystone test asserts virtual == rendered.
        #
        # Costs no render: portrait PIXELS stay suppressed by the lane's
        # `required="never"`. Only the text is hashed.
        _identity = str(obj.get("identity") or "")
        identity_basis = str((portrait_row or {}).get("prompt_hash") or "")
        if (not identity_basis and _identity != "none"
                and obj.get("identity_prompt")):
            identity_basis = normalize_prompt_for_render(
                obj.get("identity_prompt"), vstyle=_vstyle,
                banana_on=_banana_on, banana_key=_banana_key,
                source=str(obj.get("identity_prompt_source") or ""),
            ).prompt_hash
        # RE-KEYED 2026-08-26. This must not fire on "no portrait row": after
        # the portrait-free lanes shipped that is the NORMAL state on every
        # still lane, and it would cry wolf ~5x per healthy episode while the
        # seed was correctly anchored. The real defect is an empty BASIS on an
        # object whose lane asked for a face. jump_segment keeps its own
        # missing-portrait signal below; it deliberately does NOT share the
        # scene_character seed.
        if (kind == "scene_character" and char_id
                and _identity != "none" and not identity_basis):
            report.append(
                f"{oid}: no identity basis for char_id {char_id!r} -- face "
                f"will drift across this character's beats (LOUD)")
            log.warning(
                "[OTR_ImageGenDispatcher] %s: no identity basis for char_id "
                "%r; this character's face is not anchored", oid, char_id)
        elif portrait_row is None and kind == _cp.JUMP_STILL_KIND and char_id:
            report.append(
                f"{oid}: no portrait row for char_id {char_id!r} -- jump still "
                f"has no reference (LOUD)")

        # Resolve the reference PNG HERE -- before the cache key, not after it.
        # The cache-HIT branch below `continue`s, so a reference resolved later
        # could never enter the key, and a REGENERATED portrait would go on
        # serving stills conditioned on the old face. Silently.
        reference_image = ""
        if (portrait_row is not None and anchor_hash
                and getattr(_safe_engine(engine_id),
                            "accepts_reference_image", False)
                and otr_env.get("OTR_PORTRAIT_REFERENCE", "1") != "0"):
            _cand = _portrait_png(portrait_row, anchor_hash, output_dir)
            if _cand:
                reference_image = _cand
            else:
                # Fail SOFT and LOUD. An unanchored mint is exactly as good as
                # yesterday's build; refusing the episode over a missing file
                # would invent a hard stop for a degradation.
                report.append(
                    f"{oid}: portrait reference file missing for char_id "
                    f"{char_id!r}; minting WITHOUT an identity reference (LOUD)")
                log.warning(
                    "[OTR_ImageGenDispatcher] %s: portrait reference not on "
                    "disk for char_id %r; mint proceeds unanchored", oid, char_id)

        # A PERSON'S OWN PHOTO (character_photos, 2026-10-20). MetaBrief matched
        # it to this character and tagged the portrait; the photo is the
        # portrait's REFERENCE, so the portrait is painted in the episode's
        # style and every still of the character then follows the portrait.
        # The photo's own hash is the anchor, so a different photo is a
        # different portrait and never a cache hit on the old one.
        identity_photo = str(obj.get("identity_photo") or "")
        photo_more: list = []
        photo_more_names: list = []   # the extras actually used, for the row
        if kind == "portrait" and identity_photo:
            _photo_path = _photos.resolve_photo(identity_photo)
            if not _photo_path:
                report.append(
                    f"{oid}: photo {identity_photo!r} is not in ComfyUI's input "
                    f"folder; the portrait is painted from the text only (LOUD)")
                log.warning("[OTR_ImageGenDispatcher] %s: photo %r not found in "
                            "the input folder", oid, identity_photo)
            elif not getattr(_safe_engine(engine_id), "accepts_reference_image", False):
                report.append(
                    f"{oid}: photo {identity_photo!r} NOT USED -- still engine "
                    f"{engine_id!r} cannot take a reference image; pick "
                    f"flux3_klein_ref for the character stills (LOUD)")
                log.warning("[OTR_ImageGenDispatcher] %s: engine %r takes no "
                            "reference; photo %r unused", oid, engine_id,
                            identity_photo)
            else:
                reference_image = _photo_path
                anchor_hash = _photos.photo_hash(_photo_path)
                # Further photos of the same person ride along as references
                # of their own, and each joins the cache anchor.
                for _more in obj.get("identity_photo_more") or ():
                    _more_path = _photos.resolve_photo(str(_more))
                    if _more_path:
                        photo_more.append(_more_path)
                        photo_more_names.append(str(_more))
                        anchor_hash = f"{anchor_hash}+{_photos.photo_hash(_more_path)}"
                    else:
                        report.append(
                            f"{oid}: photo {_more!r} is not in ComfyUI's input "
                            f"folder; the portrait uses the others (LOUD)")
                log.info("[OTR_ImageGenDispatcher] %s: portrait painted from the "
                         "photo(s) %s", oid,
                         ", ".join([identity_photo] + [str(m) for m in
                                                       obj.get("identity_photo_more") or ()]))

        # THE OTHER PEOPLE IN THE FRAME (2026-10-20). A character still follows
        # its speaker's portrait, but a two-shot drew the OTHER person from
        # text alone, so they changed face beat to beat (live proof, 00:30: the
        # man beside MagnoliaRex was a stranger in every one of her stills).
        # MetaBrief lists the other cast members the scene prompt names; each
        # one's portrait rides along as a further reference, and every hash
        # joins the cache anchor so a new portrait of anyone in the frame is a
        # new still.
        co_references: list = list(photo_more)
        if reference_image and kind == "scene_character":
            for _co_id in obj.get("co_char_ids") or ():
                _co_row = next(
                    (r for r in reversed(images)
                     if isinstance(r, dict) and r.get("kind") == "portrait"
                     and str(r.get("object_id") or "") == str(_co_id)), None)
                _co_hash = str((_co_row or {}).get("portrait_content_hash") or "")
                _co_png = _portrait_png(_co_row, _co_hash, output_dir) if _co_row else ""
                if _co_png:
                    co_references.append(_co_png)
                    anchor_hash = f"{anchor_hash}+{_co_hash}"
                else:
                    # As LOUD as a missing speaker portrait: this person in the
                    # frame is drawn from text alone and may change face.
                    report.append(
                        f"{oid}: no portrait for {_co_id!r}, who is in this "
                        f"frame; they are drawn without a reference (LOUD)")

        _identity_take, _still_take = _producer_takes(_producer, kind, char_id, beat_id)
        seed, anchor_mode = resolve_seed_and_mode(
            seed_cfg, oid, prompt_hash, kind=kind, char_id=char_id,
            portrait_prompt_hash=identity_basis,
            identity_take=_identity_take, take=_still_take,
        )
        # Captured BEFORE any later override: `mode="fixed"` exits the resolver
        # early, the kill switch bypasses the identity branch, and a real
        # reference rewrites anchor_mode to "reference_latent" a few lines down.
        # The receipt must record whether the basis actually PARTICIPATED in the
        # seed, not merely whether one was available.
        seed_basis_used = identity_basis if anchor_mode == "seed" else ""
        if reference_image:
            # A reference was truly consumed, so the face it carries is part of
            # this request's identity. Only then -- an unanchored mint keeps its
            # existing key.
            anchor_mode = "reference_latent"
        eng_version = str(getattr(_safe_engine(engine_id), "engine_version", "1"))
        key = request_cache_key(role, oid, prompt_hash, seed, engine_id,
                                eng_version, kind=kind, w=obj_w, h=obj_h,
                                anchor=anchor_hash if reference_image else "",
                                source_context_hash=source_context_hash)
        if key in cache_index:
            # Cache HIT (pass-02 Gem-2): the hit must STILL materialize into the
            # CURRENT episode's stills/ + append a fresh ledger row (a bare
            # `continue` would silently leave episode folders missing every
            # reused still). A stale index entry (file gone) degrades to a fresh
            # render below, LOUD.
            hit_id = cache_index[key]
            src = ""
            ref_row = None
            for im in images + _superseded:
                if isinstance(im, dict) and im.get("image_id") == hit_id:
                    ref_row = im
                    src = str(im.get("pool_path") or im.get("path") or "")
                    break
            # An untrusted cache source is simply NOT A HIT: fall through to
            # the LOUD regenerate below rather than copying it (2026-09-05).
            trusted = _trusted_pool_source(src, output_dir) if src else ""
            if not trusted and ref_row is not None:
                # THE EPISODE COPY WHEN THE POOL COPY IS GONE (Sonnet QA,
                # 2026-10-22). A replay imported into a new workspace, or run
                # after the shared pool was cleared, finds the row's pool copy
                # missing while its episode copy -- rebased onto this
                # workspace by the import -- is right here. Without it every
                # unchanged still was painted again: the same pixels, and on
                # a cloud engine the same spend, once more per replay.
                src = str(ref_row.get("path") or "")
                trusted = _trusted_pool_source(src, output_dir) if src else ""
            if trusted:
                src = trusted   # the checked spelling is the one copied
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
                    # Stamped UNCONDITIONALLY, including the empty case. This
                    # row starts as a copy of an older one, so a conditional
                    # stamp would inherit a stale anchor from a previous
                    # character or a previous revision.
                    "derived_from_portrait_hash": anchor_hash,
                    "portrait_anchor_mode": anchor_mode,
                    # The identity-seed BASIS, unconditional for the
                    # same stale-inheritance reason. Distinct from
                    # derived_from_portrait_hash, which is the rendered
                    # PIXEL hash -- a lane can be seed-anchored with no
                    # portrait pixels at all, and that pair is how an
                    # audit tells the two apart.
                    "identity_seed_basis": seed_basis_used,
                })
                # Banana receipt, UNCONDITIONALLY, for the same reason as the
                # anchor stamp above: this row is a copy of an older one, and a
                # conditional stamp would inherit a stale receipt.
                fresh.update(banana_rcpt)
                # Copies of historical rows must never inherit an older source
                # operation, including when the current object carries none.
                fresh["source_context_hash"] = source_context_hash
                fresh["source_rewrite"] = source_rewrite
                if char_id:
                    fresh["char_id"] = char_id
                if beat_id:
                    fresh["beat_id"] = beat_id
                if radio_host_style:
                    fresh["radio_host_style"] = radio_host_style
                if reference_image and identity_photo and kind == "portrait":
                    fresh["identity_photo"] = identity_photo
                if lettering_style:
                    fresh["lettering_style"] = lettering_style
                if backdrop_family:
                    fresh["backdrop_family"] = backdrop_family
                if _vstyle is not None:
                    fresh["visual_style"] = str(
                        getattr(_vstyle, "style_id", "") or "")
                # Recorded so the visual ledger covers the whole episode, with
                # laplacian 0.0 meaning UNMEASURED -- a cache hit decodes no
                # fresh pixels. _style_spread skips these rather than treating
                # a missing measurement as a real value.
                _visual_rows.append({
                    "kind": kind, "object_id": oid, "role": role,
                    "source": source, "beat_id": beat_id,
                    "styled": bool(_styled_now),
                    "already_styled": bool(_style_cue) and not _styled_now,
                    "prompt_sha8": str(prompt_hash or "")[:8],
            "negative": _effective_neg,
            "negative_source": _neg_source,
                    "laplacian": 0.0, "reused": True,
                })
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
            # render it is a HARD FAIL, never a silent skip (a warn-and-continue
            # would let an episode "succeed" with missing stills). Inject a
            # gen_fn (tests) or dispatch via OTRImageGenDispatcher (in-process
            # engine path).
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
            # THE EPISODE'S VISUAL PACK, BY ID (2026-09-12). The request has
            # always carried the pack's WORDS (prompt, negative) and never
            # which pack they came from, so an engine could not act on the
            # pack itself -- which is what a per-style checkpoint needs
            # (`_otr_image_engines/sd15.py::_resolve_ckpt_name`). Engines that
            # do not read it are unaffected; the value is already stamped on
            # the visual rows and the ledger, so this adds no new authority.
            "visual_style": (str(getattr(_vstyle, "style_id", "") or "")
                             if _vstyle is not None else ""),
            # COMPOSED, never precedence (ONE STYLE AUTHORITY). The two
            # negatives are orthogonal: the pack's is about STYLE, the object's
            # (radio_host_negative) is about keeping the announcer radio-object
            # FACELESS. Letting either win outright silently drops the other --
            # and dropping the pack half on an announcer still is precisely the
            # fracture this build fixes. visual_safety_negative de-duplicates.
            "negative_prompt": _effective_neg,
            # The character's canonical portrait, when this engine declared it
            # can condition on one. Empty string on every other path, so an
            # adapter that never opts in sees exactly the request it always saw.
            "reference_image": reference_image,
            # The portraits of the other characters the prompt names; empty
            # unless the engine took the speaker's reference above.
            "reference_images": co_references,
            # w/h end-to-end (pass-02 Gem-1): the engine call reads
            # width/height (flux_gen1._flux_params request precedence), so
            # landscape scene stills are REAL, not env defaults.
            "w": obj_w, "h": obj_h,
            "width": obj_w or None, "height": obj_h or None,
        }
        cloud_image_engine = _is_cloud_image_engine(engine_id)
        if not cloud_image_engine and not _residue_freed:
            _residue_freed = True
            _residue = _levers.free_otr_pipeline_residue(
                reason="image engine load preflight (%s)" % engine_id)
            log.info(
                "[OTR_ImageGenDispatcher] pipeline residue freed before the first local "
                "still: free %.1f GB after; ran=%s failed=%s",
                _residue.get("free_gb_after", float("nan")),
                ",".join(_residue.get("steps_run", []) or []) or "-",
                ",".join(_residue.get("steps_failed", []) or []) or "-")
        still_job = {
            "oid": oid, "kind": kind, "role": role, "char_id": char_id,
            "beat_id": beat_id, "engine_id": engine_id,
            "eng_version": eng_version, "request": request, "key": key,
            "anchor_hash": anchor_hash, "anchor_mode": anchor_mode,
            "seed_basis_used": seed_basis_used, "prompt": prompt,
            "seed": seed, "effective_neg": _effective_neg,
            "neg_source": _neg_source, "styled_now": _styled_now,
            "source": source, "banana_rcpt": banana_rcpt,
            "source_context_hash": source_context_hash,
            "source_rewrite": source_rewrite,
            "radio_host_style": radio_host_style,
            # Recorded on the row only when the photo truly conditioned it.
            "identity_photo": (identity_photo if reference_image and kind == "portrait"
                               else ""),
            "identity_photo_more": list(photo_more_names),
            "lettering_style": lettering_style,
            "backdrop_family": backdrop_family,
            "obj": obj, "obj_w": obj_w, "obj_h": obj_h,
            "prompt_hash": prompt_hash,
        }
        if cloud_image_engine:
            pending_cloud.append(still_job)
            continue
        _flush_pending_cloud_stills(pending_cloud, ctx)
        _mint_still_serial(still_job, ctx, take_lease=True)

    _flush_pending_cloud_stills(pending_cloud, ctx)
    _stage_bar.finish()
    made = ctx["made"]

    # ST-3 completion contract: the producer-owned target manifest is the
    # boundary between image generation and video. A generated object that was
    # skipped, failed to materialize, or points at a missing file cannot become
    # a text-only/dark-floor video by accident.
    required_target_receipt = []
    if required_scene_targets:
        rows_by_object = {
            str(row.get("object_id") or ""): row
            for row in images if isinstance(row, dict)
        }
        # A row appended THIS dispatch, not one inherited from a prior image
        # revision: ``images`` is seeded from the incoming ledger section above,
        # so a stale row would let a genuinely missing still read as present.
        this_dispatch_ids = {
            str(row.get("object_id") or "")
            for row in ep_rows if isinstance(row, dict)
        }
        # Identity keys for a refused target: the gap row must carry the same
        # kind/role/beat_id an ``ok`` row would, and the missing-target record
        # does not keep them.
        targets_by_object = {
            str(tgt["object_id"]): tgt for tgt in required_scene_targets
            if isinstance(tgt, dict) and tgt.get("object_id")}
        missing_targets = []
        for target in required_scene_targets:
            oid = str(target["object_id"])
            row = rows_by_object.get(oid)
            path = str((row or {}).get("path") or "")
            # THE FRESHNESS TEST BELONGS ON BOTH BRANCHES (2026-08-28). It used
            # to run only inside the missing branch, to label a status -- so an
            # INHERITED row whose file still happened to exist on disk took the
            # success path and was stamped a CURRENT receipt, even when this
            # dispatch's request for that very target had been refused. The
            # stale row masked the fresh refusal, and every reader downstream
            # believed a still existed for a beat that never got one. Cache
            # reuse is unaffected: a cache HIT appends its row to ``ep_rows``
            # (see the reuse branch above), so a legitimately reused image is
            # in ``this_dispatch_ids`` like any freshly rendered one.
            stale = oid not in this_dispatch_ids
            # `stale` before the stat, and the stat only on the confined
            # spelling (Sonnet QA, 2026-10-18): an inherited row's path is
            # ledger data and must not be probed outside the approved folders.
            if (not row or not path or stale
                    or not os.path.isfile(_confined_or_empty(path) or "")):
                # ONE deterministic status per target, so logs and tests can
                # assert on it instead of parsing a sentence.
                if not row:
                    status = "no_row"
                elif stale:
                    # Covers both the file-absent and the file-PRESENT stale
                    # row; the latter is the mask described above.
                    status = "historical_row_only"
                else:
                    # Row was appended this dispatch but its file is empty or
                    # absent -- one status, whichever of the two it is.
                    status = "dead_path"
                missing_targets.append({
                    "object_id": oid, "status": status, "path": path,
                    "evidence": skip_evidence_by_oid.get(oid),
                })
                continue
            required_target_receipt.append({
                "object_id": oid,
                # DECLARED, never inferred (2026-08-28). Every row says what it
                # is, so no reader downstream has to conclude "no path, so it
                # must have been refused" -- an inference that cannot tell a
                # sanctioned refusal from a crashed render.
                "status": _receipt.STATUS_OK,
                "kind": str(target.get("kind") or ""),
                "role": str(target.get("role") or ""),
                "beat_id": str(target.get("beat_id") or ""),
                "path": path,
                "content_hash": row.get("content_hash") or row.get(
                    "portrait_content_hash"),
            })
        # A MODEL REFUSAL IS A SANCTIONED GAP, NOT A MISSING TARGET (operator
        # 2026-08-22). The engine ran, the model declined one card, and the
        # refusal was recorded with its prompt and seed above. Failing the
        # episode here would reinstate exactly the behaviour the operator ruled
        # against -- one blemish destroying every finished beat around it.
        #
        # STILL NARROW ON PURPOSE, and now exactly two reasons wide:
        # `model_refusal` (the model declined the card, 2026-08-22) and
        # `cloud_job_failed` (a partner job produced no image for this object,
        # 2026-09-16 -- a timeout, a corrupt file, a lost job). A dead path, a
        # historical-row-only target, a no-engine skip and every other absence
        # still raise, so the gate keeps its whole job for real gaps. This does
        # not soften the 2026-06-18 NO FALLBACKS rule: nothing is substituted
        # and nothing is silent. `is_sanctionable_skip` owns the list so this
        # site cannot drift from the vocabulary again.
        refused_targets = [
            m for m in missing_targets
            if _receipt.is_sanctionable_skip(
                (m.get("evidence") or {}).get("reason"))]
        if refused_targets:
            missing_targets = [m for m in missing_targets
                               if m not in refused_targets]
            for miss in refused_targets:
                ev = miss.get("evidence") or {}
                log.warning(
                    "[OTR_ImageGenDispatcher] TOLERATED GAP %s [%s]: no still "
                    "was produced for this object and the episode continues "
                    "WITHOUT it (engine=%s seed=%s). prompt=%r",
                    miss.get("object_id"), ev.get("reason"),
                    ev.get("engine_id"), ev.get("seed"), ev.get("prompt"))
                # THE GAP ROW IS THE EVIDENCE (2026-08-28). A tolerated refusal
                # must be recorded in the receipt: ``skip_evidence_by_oid`` is
                # function-local and dies with this call, so without this row
                # the ledger records a refusal NOWHERE, the still-spine
                # validator later finds no materialized still and kills the
                # episode -- a 30-minute render dying for a blemish the operator
                # had already ruled survivable. The row below is the evidence
                # every reader downstream needs, and it is stamped at the one
                # place that actually knows the refusal happened.
                #
                # NO PATH AND NO CONTENT HASH, deliberately: nothing was
                # produced, and a gap row carrying a path would let a reader
                # believe an image exists. The identity keys match an ``ok``
                # row exactly so the two join on the same fields.
                gap_target = targets_by_object.get(miss.get("object_id")) or {}
                required_target_receipt.append({
                    "object_id": str(miss.get("object_id") or ""),
                    "status": _receipt.STATUS_SANCTIONED_GAP,
                    "kind": str(gap_target.get("kind") or ""),
                    "role": str(gap_target.get("role") or ev.get("role") or ""),
                    "beat_id": str(gap_target.get("beat_id") or ""),
                    "reason": str(ev.get("reason") or ""),
                    "engine_id": str(ev.get("engine_id") or ""),
                    "seed": ev.get("seed"),
                    # The FULL prompt is kept, not just its hash: it is the
                    # 2026-08-22 refusal diagnostic and the existing refusal
                    # tests assert on it. The hash rides along for cheap
                    # equality checks against the request that was sent.
                    "prompt": ev.get("prompt"),
                    "prompt_hash": ev.get("prompt_hash"),
                    "detail": str(ev.get("detail") or ev.get("excerpt") or ""),
                    "image_revision": rev,
                })
        if missing_targets:
            # THE RAISE CARRIES ITS OWN EVIDENCE (2026-08-04): the reason sits
            # in ``warnings`` -- which is stamped into the ledger BELOW this
            # line and therefore dies with the exception -- so the message must
            # name more than the object ids.
            detail = []
            for miss in missing_targets:
                part = "%s (%s" % (miss["object_id"], miss["status"])
                if miss["status"] in ("dead_path", "historical_row_only"):
                    part += ", path=%r" % (miss["path"],)
                ev = miss.get("evidence")
                if ev:
                    part += ", skipped: reason=%s" % (ev.get("reason"),)
                    if ev.get("arm"):
                        part += (" arm=%s token=%r index=%s of %s "
                                 "prompt_hash=%s excerpt=%s"
                                 % (ev.get("arm"), ev.get("token"),
                                    ev.get("index"), ev.get("prompt_len"),
                                    ev.get("prompt_hash"),
                                    ev.get("excerpt")))
                else:
                    part += ", no skip evidence recorded"
                detail.append(part + ")")
            # THE SERVER LOG IS THE PRODUCTION CHANNEL, NOT THE EXCEPTION.
            # The canonical runner renders a failure as
            # `str(status["messages"])[:500]` (scripts/otr_api.py), so a rich
            # message is TRUNCATED before an operator ever reads it. Emit one
            # compact JSON record per missing target, in target order, BEFORE
            # raising -- that survives independently of the exception and of the
            # ledger stamp further below.
            for miss in missing_targets:
                log.error(
                    "[OTR_ImageGenDispatcher] MISSING_TARGET %s",
                    json.dumps(miss, ensure_ascii=True, sort_keys=True,
                               default=str))
            err = ImageRenderError(
                "required scene image targets missing or unmaterialized before "
                "video dispatch: " + "; ".join(detail))
            # Structured attribute consumed by the D1 regression tests, which
            # assert the evidence SCHEMA rather than substring-matching a
            # sentence. Not a production transport -- see the log record above.
            err.missing_targets = missing_targets
            raise err

    if _producer is not None:
        # EACH CARD STILL RECORDS WHAT IT WAS PAINTED FROM (chunk 3e): the
        # digest node 63 recomputes before a Video run, so an edit the still
        # does not show yet is refused there. A kept row keeps the digest of
        # the paint it is, never of the texts written since.
        _payload_objects = {str(o.get("object_id")): o
                            for o in (_producer_payload or {}).get("objects") or []
                            if isinstance(o, dict)}
        _scene_of_beat = {str(b.get("beat_id")): str(b.get("scene_id") or "")
                          for b in (ledger.get("beats") or []) if isinstance(b, dict)}
        _painted = _PD.painted_takes(_producer)
        for _row in ep_rows:
            if not isinstance(_row, dict) or (_row.get("provenance") or {}).get("source") == "kept":
                continue
            if str(_row.get("kind") or "") == "portrait":
                # The character take this portrait was painted with. A refused
                # repaint adds no row, so the live portrait keeps the take it
                # actually shows (Codex QA, 2026-10-22).
                _who = str(_row.get("char_id") or _row.get("object_id") or "")
                _row[_PD.ROW_IDENTITY_TAKE] = int(_painted["characters"].get(_who, 0) or 0)
                continue
            _obj = _payload_objects.get(str(_row.get("object_id") or ""))
            if _obj is not None and _PD.object_shot_id(_obj):
                _row[_PD.ROW_INPUT_HASH] = _PD.card_still_digest(
                    _obj, str(_row.get("engine_id") or ""), _raw_texts, _painted,
                    _scene_of_beat)
        images, _newly_superseded = _one_row_per_object(images)
        # ONLY STILLS NO LONGER SHOWN, each once (Sonnet QA, 2026-10-22). A
        # cache hit or a kept still is a new row under its old image_id, and
        # every run used to file a copy of it here beside the live row: a
        # ledger, and every bundle frozen from it, a row longer per still per
        # run. A still shown again leaves the list.
        _filed, _seen = [], {str(r.get("image_id")) for r in images if isinstance(r, dict)}
        for _row in _superseded + _newly_superseded:
            if isinstance(_row, dict) and str(_row.get("image_id")) not in _seen:
                _seen.add(str(_row.get("image_id")))
                _filed.append(_row)
        _superseded = _filed
    ledger["images"] = {
        "image_revision": rev,
        "episode_id": ep,
        "granularity_by_role": (image_policy or {}).get("granularity") or {},
        "images": images,
        "cache_index": cache_index,
        "warnings": warnings,
    }
    if required_scene_targets:
        ledger["images"]["required_scene_targets"] = required_target_receipt
    if _producer is not None:
        ledger["images"][_PD.PAYLOAD_KEY] = _producer_payload
        ledger["images"][_PD.RAW_TEXTS_KEY] = _raw_texts
        ledger["images"][_PD.TAKES_KEY] = _PD.painted_takes(_producer)
        ledger["images"][_PD.SUPERSEDED_KEY] = _superseded
    # Credits: stamp the per-role image-engine histogram into meta (which DOES
    # persist to the on-disk ledger the credits dossier reads) so the dossier
    # shows the image model used for each slot. Additive; never overwrites.
    _ie_meta = ledger.get("meta")
    if not isinstance(_ie_meta, dict):
        _ie_meta = {}
        ledger["meta"] = _ie_meta
    _ie_meta["image_engines"] = {"by_role": img_hist, "image_revision": rev}
    _meta_updates = {"image_engines": _ie_meta["image_engines"]}
    if _producer is not None:
        # THE IMAGE-ENGINE STAMP: the engines this policy paints every slot
        # with, which every producer freeze copies into its manifest and a
        # later run must match (``_otr_producer_direction``).
        _ie_meta[_PD.META_IMAGE_ENGINES] = _PD.policy_image_engines(image_policy)
        _meta_updates[_PD.META_IMAGE_ENGINES] = _ie_meta[_PD.META_IMAGE_ENGINES]
        _ie_meta[_PD.META_IMAGE_SEED] = _PD.policy_image_seed(image_policy)
        _meta_updates[_PD.META_IMAGE_SEED] = _ie_meta[_PD.META_IMAGE_SEED]
    # S2 durable persistence (credits enrichment 2026-07-03): ``ledger`` here
    # is the LOCAL wire-parsed dict -- the stamp above was wire-only, so the
    # image-engine receipts died with the wire. Copy the images section + the
    # image_engines meta into the production-ledger SINGLETON and save
    # LOUDLY (raises LedgerStampError on save failure; test-mode injects
    # in-memory only). The late OTR_CreditsRoll node reads THIS.
    # THE VISUAL LEDGER. Assembled here and stamped on the SAME durable call as
    # the images section below -- one owner, no second LedgerStampError site.
    _spread = _style_spread(_visual_rows)
    ledger["visual"] = {
        "authority_version": 1,
        "style_id": (str(getattr(_vstyle, "style_id", "") or "")
                     if _vstyle is not None else ""),
        "style_token": _style_cue,
        # What the pack DECLARED and what survived the self-veto resolution.
        # Equal on a cleanly authored pack; different means the pack asked to
        # suppress something it also asks for, and the mint kept the request.
        "pack_negative_authored": _pack_negative_authored,
        "pack_negative": _pack_negative,
        "self_veto_resolved": _pack_negative != _pack_negative_authored,
        # Per-ROW `negative` / `negative_source` live on each prompts[] entry:
        # one episode-level label cannot describe a per-row composition.
        "prompts": _visual_rows,
        "spread": _spread,
    }
    _styled_n = sum(1 for r in _visual_rows if r.get("styled"))
    log.info("[OTR.image.style_authority] style=%r token=%r styled=%d/%d "
             "pack_negative=%s spread=%s",
             ledger["visual"]["style_id"], _style_cue, _styled_n,
             len(_visual_rows), bool(_pack_negative),
             _spread.get("max_pairwise"))

    from .production_ledger import stamp_durable
    stamp_durable(
        sections={"images": ledger["images"], "visual": ledger["visual"]},
        meta_updates=_meta_updates,
        source="image_dispatcher",
    )
    # stills_manifest.json beside the episode stills (ST-3/W3): the durable
    # per-episode index of every still this dispatch materialized. Fail-soft.
    if ep_rows:
        try:
            os.makedirs(ep_dir, exist_ok=True)
            manifest = {
                "episode_id": ep, "image_revision": rev,
                # Style telemetry beside the stills it describes, so the
                # measurement is auditable without opening the ledger.
                "visual_style": (str(getattr(_vstyle, "style_id", "") or "")
                                 if _vstyle is not None else ""),
                "style_spread": _spread,
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
                    "source_context_hash": r.get("source_context_hash", ""),
                    "source_rewrite": r.get("source_rewrite"),
                    "provenance": r.get("provenance"),
                    # The banana receipt rides the manifest so an operator can
                    # audit the route without opening the ledger.
                    "banana_route": r.get("banana_route"),
                    "banana_table_version": r.get("banana_table_version"),
                    "banana_substitutions": r.get("banana_substitutions"),
                    "banana_sha256_before": r.get("banana_sha256_before"),
                    "banana_sha256_after": r.get("banana_sha256_after"),
                    "banana_varieties": r.get("banana_varieties"),
                    "visual_style": r.get("visual_style"),
                    "laplacian": r.get("laplacian"),
                } for r in ep_rows],
            }
            try:
                from ._otr_paths import confine_to_output_tree as _confine_manifest
            except ImportError:  # pragma: no cover -- flat (sys.path) load
                from _otr_paths import confine_to_output_tree as _confine_manifest  # type: ignore
            # Its own leaf (registry review, 2026-10-30): the directory was
            # proven, a link planted at the manifest's name was not.
            with open(_confine_manifest(os.path.join(ep_dir, "stills_manifest.json"),
                                        "stills manifest"), "w",
                      encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=True, indent=2)
        except OSError as exc:
            warnings.append(f"stills_manifest.json write failed ({exc}); "
                            "episode stills are on disk but unindexed (LOUD)")
    # ONE aggregate banana line per dispatch, never per object. A split env
    # state (stills vs video) is valid operator input on t2v lanes and a
    # footgun on i2v lanes (the anchor still carries the look) -- INFO, never
    # LOUD, per the build spec's cut list.
    log.info(
        "[OTR_ImageGenDispatcher] banana_route=%s candidate_substitutions=%d "
        "varieties=%s%s",
        "on" if _banana_on else "off", _banana_candidate_subs,
        _banana.varieties_receipt(_banana.select_varieties(_banana_key)),
        ("" if (_banana_on
                == _banana.banana_gate(_banana_meta, lane="video"))
         else " (NOTE: stills/video EFFECTIVE gates disagree for this episode "
              "-- coherent on t2v lanes only; i2v anchors carry the still's "
              "look)"))
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
        # S4: real host facts.
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

    DESCRIPTION = (
        "Coordinates still image generation for character portraits and scene backgrounds "
        "against a content-addressed disk cache. Reuses existing stills on cache hits, "
        "renders missing images using the selected image model, records image paths in the "
        "ledger, and emits the image-done signal. Connect script, image policy, and prompt "
        "wires or supply ordering gates."
    )

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
                    "tooltip": "OTR_ImageDirector policy (engine per role + fresh cap + seed).",
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
            # The cloud_* image rows spend with the key OTR_ComfyCredential
            # stashed for this prompt (plan 0k: no V1 hidden key here).
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True

    def dispatch(self, script_json, image_policy_json="{}", image_prompts_json="{}",
                 gate_in="", episode_id=""):
        led = self._loads(script_json, {})
        policy = self._loads(image_policy_json, {})
        prompts = self._loads(image_prompts_json, {})
        # THE SCENE PRODUCER'S DIRECTION, as Shot Lock put it on the wire. None
        # on every ordinary run, and then nothing below changes.
        from . import _otr_producer_direction as _PD
        direction = _PD.runtime_direction(led)
        # CANONICAL REPLAY (campaign item 0): the imported ``images`` rows are
        # verified on disk and re-stamped as they are; gen_fn is never called.
        # The one replay that PAINTS is a Scene Producer stills stage: it runs
        # the cached path below on node 89's rebuilt payload, so only what
        # changed is minted (plan, "Only the Stills stage paints").
        from .production_ledger import replay_descriptor as _replay_descriptor
        _replay = bool(_replay_descriptor((led.get("meta") or {})))
        _paints = direction is not None and direction["stage"] == "stills"
        if _replay and not _paints:
            led, image_done, report = verify_replay_images(led)
        else:
            if _replay:
                # The wire snapshot can predate rename_episode; the same rebase
                # the verify path runs, before any imported row is read.
                led = _replay_rebase_to_live_episode(led)
            # The in-graph node mints REAL portraits via the request's image
            # engine (in-process Flux gen-1) under the AS-3 GPU-residency lease.
            # On a box without the GPU / wrapper nodes / checkpoint the render
            # fails closed and the dispatcher degrades that object to the radio
            # floor LOUD (never a crash). The CPU platform tests bypass this by
            # calling dispatch_images() directly with an injected gen_fn.
            # A Scene Producer run draws the stage bar; its quiet is closed
            # here on every path, so a refused still never leaves this thread's
            # later bars dark.
            quiet = _QuietInnerBars() if direction is not None else None
            try:
                led, image_done, report, _warn = dispatch_images(
                    led, policy, prompts, gen_fn=_inprocess_gen_fn,
                    episode_id=str(episode_id or ""), quiet_inner_bars=quiet,
                )
            finally:
                if quiet is not None:
                    quiet.close()
        if direction is not None:
            if direction["stage"] == "stills":
                return self._finish_stills_stage(led, direction, episode_id, report)
            # Any later stage: hand the direction on to Batch, whatever the
            # helpers above did to the ledger dict.
            _PD.attach_runtime(led, direction)
        patched = json.dumps(led, ensure_ascii=True, separators=(",", ":"))
        return (patched, image_done, report)

    @staticmethod
    def _finish_stills_stage(led, direction, episode_id, report):
        """END OF A SCENE PRODUCER STILLS STAGE: freeze this workspace into the
        bundle the direction names, then stop the graph here. Every node
        downstream -- the video, composite, captions, credits and mux -- is
        skipped by ComfyUI's silent blocker, and the prompt still ends as a
        success. The stills are already stamped durably, so the freeze reads
        them from disk.

        THE LEDGER'S ID FIRST (Sonnet QA, 2026-10-22): ``dispatch_images``
        re-keys a stale ``pending_*`` id to the folder the stills actually went
        to and stamps it on the ledger as durable identity truth, so that, not
        the wired id, names the workspace to freeze. And the blocker is built
        BEFORE the freeze: a core without it must fail with no bundle left
        behind, or the retry would meet "bundle already exists"."""
        from . import _otr_producer_clips as _PC
        from . import _otr_producer_direction as _PD
        blocked = _PD.execution_blocker()
        ep = str(led.get("episode_id") or episode_id
                 or (led.get("meta") or {}).get("episode_id") or "")
        # A STILLS BUNDLE AFTER A VIDEO ONE CARRIES ITS CLIPS (plan, "Freezer"):
        # the video section's reuse map and floor payloads name files in clips/
        # and audio/foley/, and the next Video run reuses them -- or a kept
        # floor -- from this bundle even when the project's clip store is gone.
        # Before any video run there are none, and a stills bundle carries
        # stills only, as it always has.
        video = led.get("video") if isinstance(led.get("video"), dict) else {}
        bundle = _PD.freeze_stage(direction, ep, include_render_outputs=bool(
            video.get(_PC.MAP_KEY) or video.get(_PC.FLOOR_KEY)))
        log.warning("[OTR_ImageGenDispatcher] Scene Producer stills stage done: froze "
                    "episode %s into %s; the graph stops here (%s)", ep, bundle,
                    str(report or "").splitlines()[0] if report else "no report")
        return (blocked, blocked, blocked)

    @staticmethod
    def _loads(raw, default):
        try:
            v = json.loads(raw or "null")
            return v if isinstance(v, type(default)) else default
        except (ValueError, TypeError):
            return default
