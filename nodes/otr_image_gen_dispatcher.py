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
import re as _re
import time
from typing import NamedTuple

log = logging.getLogger("OTR")

from ._otr_shared import gpu_residency as _lease
from ._otr_shared import portrait_ledger as _pl
from ._otr_shared import role_slots as _role_slots
from ._otr_shared import still_receipt as _receipt
# Cold-import clean: route_freeze is stdlib-only at module scope.
from ._otr_shared.route_freeze import RouteFreezeError as _RouteFreezeError
from ._otr_story_brief_helpers import (
    append_visual_safety_clause,
    visual_safety_negative,
)
# The banana route (docs/2026-08-06-BUILD-SPEC-banana-route.md): pure,
# stdlib-only house-style transform applied at THIS funnel before the prompt
# content hash, gated by env + the fidelity-bank idiom. Cold-import clean.
from . import _otr_banana_route as _banana
from ._otr_image_engines import registry as _ireg
# The single canonical residue-freer (writer LLM + Bark, then a surgical detach of any
# ComfyUI-tracked patcher, then the allocator flush). Cold-import clean: stdlib only at
# module scope; torch / comfy are imported lazily inside the call.
from . import _otr_vram_levers as _levers

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
                      kind="", w=0, h=0, *, anchor="") -> str:
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
    return _content_hash(parts)


def resolve_object_seed(seed_cfg, object_id, prompt_hash, kind="") -> int:
    """The seed alone. See ``resolve_seed_and_mode`` for the full contract.

    Kept as a thin wrapper so every existing call site stays byte-identical --
    none of them pass a char_id, so none of them can reach the identity branch.
    """
    return resolve_seed_and_mode(seed_cfg, object_id, prompt_hash, kind=kind)[0]


def resolve_seed_and_mode(
    seed_cfg, object_id, prompt_hash, *, kind="", char_id="",
    portrait_prompt_hash="",
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
    """
    # The radio BOOKEND still (scene_open) AND the radio-HOST FACE object
    # (2026-07-01 brief-driven radio-host; object_id "radio_host_portrait",
    # matches otr_meta_brief_image_prompt.RADIO_HOST_PORTRAIT_ID) both render
    # with the FIXED deterministic bookend seed so the host face is reproducible
    # run-to-run and open/inter/close share ONE canonical face.
    _oid = str(object_id or "")
    if (str(kind or "") == "scene_open" or _oid == "radio_host_portrait"
            or _oid.endswith("_radio_face_169")):   # ltx talking radio-face
        try:
            return (int(os.environ.get("OTR_RADIO_BOOKEND_SEED", 4242)), "")
        except (TypeError, ValueError):
            return (4242, "")
    cfg = seed_cfg if isinstance(seed_cfg, dict) else {}
    base = int(cfg.get("request_seed") or 0)
    if str(cfg.get("mode") or "request_hash") != "request_hash":
        return (base, "")
    # AFTER the mode gate, never before it. `base` does not exist until the line
    # above, so a branch placed earlier raises UnboundLocalError inside a call
    # that dispatch_images makes outside any try/except -- the whole dispatch
    # dies. Placed between `base` and the gate, it silently breaks the
    # documented mode='fixed' contract instead, which is worse.
    if (str(kind or "") == "scene_character" and char_id and portrait_prompt_hash
            and os.environ.get("OTR_PORTRAIT_IDENTITY_SEED", "1") != "0"):
        # EXACTLY the portrait object's own draw: the portrait's object_id IS
        # the char_id (otr_meta_brief_image_prompt.py:1765), so this reproduces
        # the seed that character's portrait already rendered with, leaving the
        # portrait itself byte-identical.
        digest = hashlib.sha256(
            f"{base}:{char_id}:{portrait_prompt_hash}".encode()).hexdigest()
        return (int(digest[:8], 16), "seed")
    digest = hashlib.sha256(
        f"{base}:{object_id}:{prompt_hash}".encode()).hexdigest()
    return (int(digest[:8], 16), "")


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


#: The four values `negative_source` may take. COMPOSITION ONLY -- see
#: `negative_source_label`. Exported so a test can pin the vocabulary rather than
#: re-typing it, because this enum has already drifted once unnoticed: the
#: one-style-authority PLAN documented `env_override`, which never shipped.
NEGATIVE_SOURCE_LABELS = ("pack+request", "pack", "request", "none_contributed")


def negative_source_label(pack_negative, obj_negative) -> str:
    """Where THIS row's composed negative came from, and the one place that names it.

    Pure, and deliberately blind to the engine. It answers exactly one question --
    which of the two COMPOSITION inputs contributed -- so every value it can return
    is verifiable from its own two arguments.

    ITEM H (2026-08-17): the empty arm used to read `engine_hygiene`, a claim about
    what the ENGINE adds on top. That was wrong twice over. It was computed before
    `resolve_engine_for_role` picks an engine, so it asserted a property of an
    engine not yet chosen -- accurate for `z_image_turbo` (which does end
    `.strip() or _HYGIENE_NEGATIVE`) and false for `lumina_image` (no floor at all)
    purely by coincidence. And it put two authorities in one value: composition
    belongs to the dispatcher, the hygiene floor belongs to the engine.

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


#: Excerpt half-width around a path-guard match. The old report was
#: ``prompt[:60]``, which shows nothing useful when the offending character sits
#: late in a long prompt -- the excerpt is centred on the match instead.
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
    path, never prose that merely contains one. The old predicate was
    ``os.sep in p or os.altsep in p or endswith(ext)``, and on Windows
    ``os.altsep`` is ``/`` -- so "a black/white striped scarf" and "the corner
    of 5th/Main" were refused and their beats rendered NO still. Two producers
    grew local sanitizers to launder slashes out of prose, and the helper's own
    comment conceded that laundering a REAL path turns a loud refusal into a
    quiet garbled render. Both are removed with this change.

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
    then ignores -- a wasted ~10 min image-gen pass per episode.

    DELEGATES to the ONE route-freeze authority (2026-07-25, chunk 1a):
    ``_otr_shared.route_freeze``. This function used to re-parse the env itself
    and SWALLOW a malformed map (``except Exception: return eng_id``), which is
    the silent-unforced defect ``render_driver.apply_engine_override`` was made
    terminal for at ``57f4983a`` -- the image phase would spend ~10 minutes
    minting stills for a plan that was going to die at render anyway. It is now
    FAIL-CLOSED at the first reader. Unset env -> ``eng_id`` unchanged."""
    try:
        from ._otr_shared import route_freeze as _rf  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import route_freeze as _rf  # type: ignore
    snap = _rf.routing_env_snapshot()
    mapping = _rf.parse_force_map(snap.get("force_engine_map", ""))
    return _rf.forced_engine_for_role(role, mapping) or eng_id


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
    still rather than quietly skipping an asset that render might need).

    DELEGATES to the ONE route-freeze authority (2026-07-25, chunk 1a). It used
    to hard-code the redirect target as the bare literal ``"ltx_audio_in"``
    rather than reading ``render_driver._NEVER_HUMO_REDIRECT_ENGINE``, so a
    rename of that constant would have silently desynced the image phase from
    the render phase. The target now comes from the constant itself."""
    try:
        from ._otr_shared import route_freeze as _rf  # type: ignore
    except ImportError:  # pragma: no cover -- flat test imports
        from _otr_shared import route_freeze as _rf  # type: ignore
    return _rf.effective_engine_for_role(str(role or ""), str(eng_id or ""))


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
            # NO SILENT SKIP (2026-07-25 QA fix). This branch used to infer
            # "no scene object and no required target means the lane consumes
            # no still, so its segments need none" and drop the requests. That
            # inference contradicted the still spine, which demands every
            # STAMPED request back regardless -- so the episode died at the
            # render boundary instead, with a message about a missing still
            # nobody had decided to skip. The inference now happens ONCE, at
            # the mint (``otr_shot_lock._lane_consumes_a_still``), using the
            # spine's own predicate, so a still-less lane never carries
            # requests to begin with. Arriving here therefore means the stamp
            # and the image payload genuinely disagree about one beat, and
            # guessing which is right is how a jump cut ends up rendering from
            # nothing.
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
    prompt = append_visual_safety_clause(str(raw_prompt or ""))
    _pre_style = prompt
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
    if banana_on:
        bres = _banana.apply(
            prompt, variety_key=banana_key,
            shield_quoted_card_text=(str(source or "") == "still_word"))
        prompt = bres.text
        receipt = _banana.receipt_keys(bres)
    else:
        bres = None
        receipt = _banana.off_receipt(prompt, variety_key=banana_key)
    return _NormalizedPrompt(prompt, styled, pre_banana_hash, bres, receipt,
                             _prompt_content_hash(prompt))


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

    (The old downstream 3D HALT -- locked_3d_slots x per_beat -- was removed
    with the dormant 3D family, lean-mean order 4, 2026-08-23. The director no
    longer emits the field; a STALE policy that still carries it is simply
    ignored, because the capability it guarded can no longer be declared.)
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
    # THE BANANA ROUTE (docs/2026-08-06-BUILD-SPEC-banana-route.md). Gate and
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
    # canonical call the LTX 2.5 engine and the GGUF backend make in their preflight.
    # Nothing after the image stage requests an LLM slot, so there is no reload cost.
    _residue_freed = False
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("object_id") or "")
        kind = str(obj.get("kind") or "portrait")
        role = str(obj.get("role") or "character_video")
        char_id = str(obj.get("char_id") or "")
        beat_id = str(obj.get("beat_id") or "")
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
        # Item H (2026-08-17) renamed the empty arm off "engine_hygiene", which was
        # a claim about the ENGINE made before `resolve_engine_for_role` below has
        # chosen one.
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
        # RE-KEYED 2026-08-26. This used to fire on "no portrait row", which
        # after the portrait-free lanes shipped is the NORMAL state on every
        # still lane -- it would cry wolf ~5x per healthy episode while the
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
                and os.environ.get("OTR_PORTRAIT_REFERENCE", "1") != "0"):
            _cand = str(portrait_row.get("pool_path")
                        or portrait_row.get("path") or "")
            if not _cand or not os.path.isfile(_cand):
                try:
                    _cand = str(_pl.portrait_path_for_hash(
                        anchor_hash, output_dir) or "")
                except Exception:  # noqa: BLE001 -- resolver is best-effort
                    _cand = ""
            if _cand and os.path.isfile(_cand):
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

        seed, anchor_mode = resolve_seed_and_mode(
            seed_cfg, oid, prompt_hash, kind=kind, char_id=char_id,
            portrait_prompt_hash=identity_basis,
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
                                anchor=anchor_hash if reference_image else "")
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
            # w/h end-to-end (pass-02 Gem-1): the engine call reads
            # width/height (flux_gen1._flux_params request precedence), so
            # landscape scene stills are REAL, not env defaults.
            "w": obj_w, "h": obj_h,
            "width": obj_w or None, "height": obj_h or None,
        }
        lease = None
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
        try:
            if not cloud_image_engine:
                lease = _lease.acquire(timeout_s=lease_timeout_s, lockdir=lockdir)
            pixels = _coerce_pixels(
                gen_fn(request), min_bytes=handoff_min_bytes,
                wait_attempts=handoff_wait_attempts, wait_sleep_s=handoff_wait_sleep_s,
            )
            content_hash = _pl.compute_portrait_hash(pixels)
            # Style-spread telemetry: measured here because the pixels are
            # already a decoded CPU uint8 array on this line (the hash above
            # consumes the same object), so it costs one numpy pass and adds no
            # device sync. Aggregated once AFTER the loop -- a spread computed
            # mid-loop would warn before the episode's stills exist.
            _lap = _laplacian_variance(pixels)
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
        except Exception as exc:  # noqa: BLE001 -- split by KIND, just below
            if not getattr(exc, "is_model_refusal", False):
                # NO FALLBACKS (operator 2026-06-18): a wrapper-node-missing /
                # CUDA-OOM / decode failure HARD-FAILS the episode LOUD -- no
                # skip, no radio-floor degrade, no silent flux substitution.
                raise ImageRenderError(
                    f"{oid}: image render with '{engine_id}' failed "
                    f"({type(exc).__name__}: {exc}). NO FALLBACK -- fix the "
                    f"engine or select a usable one.") from exc
            # A MODEL REFUSAL IS NOT AN ENGINE FAILURE (operator 2026-08-22:
            # "why is refusing card killing the episode, i dont think thats
            # good feature" / "i didnt want any fail on this or that").
            #
            # The engine worked: it returned valid decoded pixels at the exact
            # requested dimensions and the graph completed. The MODEL declined
            # this one card. Hard-failing here destroyed an entire episode over
            # one blemish -- and destroyed the evidence with it, because the
            # refused card's prompt was never persisted anywhere, which is why
            # the 2026-08-21 refusal still cannot be diagnosed as seed-vs-
            # content. So this degrades LOUD and RECORDS THE PROMPT.
            #
            # This is NOT a fallback and does not weaken the 2026-06-18 rule:
            # no engine is substituted, nothing is silent, and every real
            # engine fault still hard-fails through the branch below. The beat
            # simply has no still, which is a state the pipeline already has.
            # RECORDED, not just logged. The completeness gate below reads this
            # map: without an entry the target reports "no_row, no skip evidence
            # recorded" and hard-fails the episode anyway -- which is how the
            # first version of this fix was only HALF a fix.
            skip_evidence_by_oid[oid] = {
                "reason": "model_refusal", "role": role, "engine_id": engine_id,
                "prompt": prompt, "seed": seed, "detail": str(exc),
            }
            warnings.append(
                f"{oid}: '{engine_id}' MODEL REFUSAL -- no still for this "
                f"object; the episode continues (operator 2026-08-22). "
                f"prompt={prompt!r} seed={seed} ({exc})")
            log.warning(
                "[OTR_ImageGenDispatcher] MODEL REFUSAL on %s via '%s': %s\n"
                "  prompt: %s\n  negative: %s\n  seed: %s\n"
                "  The episode CONTINUES with no still for this object. This "
                "prompt is recorded so the refusal can be diagnosed as seed- "
                "or content-driven -- the previous hard-fail erased it.",
                oid, engine_id, exc, prompt, _effective_neg, seed)
            continue
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
            # portrait_content_hash is THIS row's own decoded-pixel hash on
            # every kind, portrait or scene -- render_driver and the mesh cache
            # read it under that name, so it is not renamed.
            # derived_from_portrait_hash is the different thing: which
            # character's portrait this scene still was anchored to.
            "request_hash": key, "portrait_content_hash": content_hash,
            "content_hash": content_hash, "w": obj_w, "h": obj_h,
            "prompt_hash": prompt_hash, "provenance": {"source": obj.get("source", "")},
            "derived_from_portrait_hash": anchor_hash,
            "portrait_anchor_mode": anchor_mode,
            "identity_seed_basis": seed_basis_used,
        }
        # Banana receipt (six keys), on the fresh-generation row exactly as on
        # the cache-hit row -- both are the durable ledger record.
        row.update(banana_rcpt)
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
        # Style attribution travels WITH the still: portrait rows carried no
        # visual_style at all before this, so a still could not be traced back
        # to the pack that shaped it.
        if _vstyle is not None:
            row["visual_style"] = str(getattr(_vstyle, "style_id", "") or "")
        if _lap:
            row["laplacian"] = _lap
        _visual_rows.append({
            "kind": kind, "object_id": oid, "role": role,
            "source": source, "beat_id": beat_id,
            "styled": bool(_styled_now),
            "already_styled": bool(_style_cue) and not _styled_now,
            "prompt_sha8": str(prompt_hash or "")[:8],
            "negative": _effective_neg,
            "negative_source": _neg_source,
            "laplacian": _lap,
        })
        images.append(row)
        ep_rows.append(row)
        cache_index[key] = image_id
        made += 1
        report.append(f"{oid}: generated -> {os.path.basename(ep_path)}")

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
            if not row or not path or not os.path.isfile(path) or stale:
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
        # NARROW ON PURPOSE: only `reason == "model_refusal"` is tolerated. A
        # dead path, a historical-row-only target, a no-engine skip and every
        # other absence still raise, so the gate keeps its whole job for real
        # gaps. This does not soften the 2026-06-18 NO FALLBACKS rule: nothing
        # is substituted and nothing is silent.
        refused_targets = [
            m for m in missing_targets
            if str(((m.get("evidence") or {}).get("reason") or ""))
            == _receipt.SANCTIONABLE_SKIP_REASON]
        if refused_targets:
            missing_targets = [m for m in missing_targets
                               if m not in refused_targets]
            for miss in refused_targets:
                ev = miss.get("evidence") or {}
                log.warning(
                    "[OTR_ImageGenDispatcher] TOLERATED REFUSAL %s: the model "
                    "declined this card and the episode continues WITHOUT it "
                    "(engine=%s seed=%s). prompt=%r",
                    miss.get("object_id"), ev.get("engine_id"), ev.get("seed"),
                    ev.get("prompt"))
                # THE ROW THE WHOLE CONTROL PATH WAS MISSING (2026-08-28).
                # Until now a tolerated refusal was logged and then dropped:
                # ``skip_evidence_by_oid`` is function-local and dies with this
                # call, and the target never entered the receipt. So the ledger
                # recorded a refusal NOWHERE, the still-spine validator later
                # found no materialized still and killed the episode, and a
                # 30-minute render died for a blemish the operator had already
                # ruled survivable. The row below is the evidence every reader
                # downstream needs, and it is stamped at the one place that
                # actually knows the refusal happened.
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
            # THE RAISE CARRIES ITS OWN EVIDENCE (2026-08-04). It used to report
            # only the object ids, while the reason sat in ``warnings`` -- which
            # is stamped into the ledger BELOW this line and therefore died with
            # the exception. A 320-word leg failed here and took its own
            # explanation with it.
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
            with open(os.path.join(ep_dir, "stills_manifest.json"), "w",
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
