"""
nodes/production_ledger.py -- OTR Production Ledger (L2, write-only)

Single-source-of-truth JSON record of everything an episode produced:
cast, scenes, shots, beats, lines, music, and their positions on
the final audio timeline.

Hierarchy:
    Scene > Shot > Beat > Clip
        - Scene: high-level narrative location.
        - Shot: continuous visual unit (same framing/lighting).
        - Beat: single-speaker continuous turn within a shot. NEW in L2.
        - Clip: one HuMo render call (length 4n+1 frames @ 25 fps).

The beat layer was added 2026-04-25 PM (schema bump l1 -> l2) so the
HuMo clip-fill rule applies per-beat (not per-shot). This guarantees
no clip's audio window crosses a speaker boundary, which preserves
identity in Goal 3 daisy-chain mode (see ROADMAP Goal 3 +
docs/2026-04-25-humo-continuity-brief.md).

L2 scope:
  * Write-only on the producing side (FULL pipeline + script-side
    builders such as build_silent_test_episode.py).
  * Read by the HuMo orchestrator (render_humo_batch.py) and downstream
    visualisation (artifacts).
  * Incremental saves after every stage -- a crash leaves a partial JSON
    showing exactly where the pipeline died.
  * Writes NEVER raise. A ledger failure is logged and the pipeline continues.

Stages that populate the ledger (in pipeline order):
  LedgerScriptWriter DONE -> cast (voice_presets) + lines + meta.visual_plan
                             (replaces legacy ScriptWriter -> LLMDirector chain
                             retired in voice-path-cleanbreak S2)
  SceneSequencer DONE  -> lines/music start_s + dur_s + beats (speaker turns)
  SignalLostVideo DONE -> episode_id (real title), final_audio_path,
                          final_video_path, total_episode_dur_s

Backward compatibility:
  Older callers that pre-date the beat hierarchy can omit beat_id and
  boundary on lines; the orchestrator must treat missing values as
  equivalent to "shot_start" for safety.

Usage inside any OTR node:

    from nodes.production_ledger import get_ledger
    led = get_ledger()                 # re-use current episode
    led.set_cast([{"char_id":"c01", "name":"EDNA", ...}])
    led.set_beats([{"beat_id":"shot_001_b1", "speaker":"EDNA", ...}])
    led.set_lines([{"line_id":"l001", "beat_id":"shot_001_b1", "boundary":"shot_start", ...}])
    led.save()
"""
from __future__ import annotations

import hashlib
import json
import uuid
import shutil
import copy
import logging
import ntpath
import os
import posixpath
import re
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

try:
    from ._otr_shared import proc as otr_proc
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import proc as otr_proc  # type: ignore

try:
    from ._otr_shared import env as otr_env
except ImportError:  # pragma: no cover -- flat test imports
    from _otr_shared import env as otr_env  # type: ignore

log = logging.getLogger("OTR.production_ledger")

# Canonical spoken-role test (S1). _otr_ledger_scrub is a leaf module
# (stdlib-only, import-safe), so this never forms an import cycle. Dual
# import keeps the module loadable both as a package node and standalone.
try:  # pragma: no cover - exercised by both import styles
    from ._otr_ledger_scrub import is_spoken_role, append_compose_flag
    from ._otr_text_metrics import (
        canonical_char_count,
        canonical_word_count,
        set_line_text_metrics,
        stamp_line_text_metrics,
    )
except ImportError:  # pragma: no cover
    from _otr_ledger_scrub import is_spoken_role, append_compose_flag  # type: ignore
    from _otr_text_metrics import (  # type: ignore
        canonical_char_count,
        canonical_word_count,
        set_line_text_metrics,
        stamp_line_text_metrics,
    )

_LEDGER_LOCK = threading.Lock()
_CURRENT: Optional["Ledger"] = None
_GIT_HEAD_CACHE: Optional[str] = None


# ---------------------------------------------------------------------------
# D3 (2026-06-22, story-quality lift) -- speaker_role <-> char_id coercion.
#
# A character line whose char_id is a real CAST id must carry
# speaker_role="character". The b011 "Chandra's Echo" defect: the reviewer's
# role_mismatch repair honoured an LLM expected="announcer" on a c02 (cast) row,
# so a cast character was routed as the announcer. These helpers force the
# correct role at the repair guard, in set_lines, and (mandatory) in a pre-freeze
# sweep -- COERCE-NEVER-CRASH; non-cast / announcer / music-sentinel rows are
# left untouched so the legitimate announcer re-stamp is never fought.
# ---------------------------------------------------------------------------

#: char_id values that are NOT cast characters (the announcer + music render
#: contracts). A row carrying one of these is never coerced to "character".
#: ("sfx" removed 2026-07-01, rip-sfx-broll -- old sfx ledgers fail loud at
#: the freeze gate / role resolvers instead of riding a sentinel.)
_NON_CHARACTER_CHAR_ID_SENTINELS: frozenset = frozenset({
    "announcer", "music", "music_open", "music_close", "music_inter",
})


def cast_ids_from_ledger(ledger_data: Any) -> "set[str]":
    """Return the set of real CAST char_ids from a ledger dict's `cast` table,
    MINUS the announcer / music sentinels. Defensive; never raises.

    The announcer is NOT always keyed by the literal "announcer" char_id -- a
    real episode often stores it as an ordinary cast slot (e.g. char_id="c01",
    name="ANNOUNCER", tts_model="kokoro"). Mirror the reviewer's existing roster
    convention (`name != "ANNOUNCER"`) and exclude any row whose NAME is the
    announcer marker, so D3 never coerces a legitimate announcer-intro line to
    "character" (live-smoke 2026-06-22: c01=ANNOUNCER intro was wrongly
    re-roled, then routed to the character TTS engine and crashed)."""
    ids: "set[str]" = set()
    try:
        for row in (ledger_data or {}).get("cast", []) or []:
            if not isinstance(row, dict):
                continue
            cid = str(row.get("char_id") or "").strip()
            if not cid or cid.lower() in _NON_CHARACTER_CHAR_ID_SENTINELS:
                continue
            name = str(row.get("name") or "").strip().upper()
            if name == "ANNOUNCER":
                continue  # the announcer occupying a cast slot is NOT a character
            ids.add(cid)
    except Exception:  # noqa: BLE001 -- a malformed ledger is never fatal here
        return set()
    return ids


def coerce_speaker_role_for_char_id(
    line: Any, cast_ids: "set[str]", source: str = "",
) -> "tuple[Any, bool]":
    """If ``line.char_id`` is a real cast id, force ``speaker_role="character"``.

    Returns ``(line, changed)``. Mutates ``line`` IN PLACE when it coerces and
    stamps a per-line ``compose_flags`` breadcrumb (the ledger ROW schema is
    FIXED -- there is NO per-line meta dict). No-op (changed=False) when: the row
    is not a dict; char_id is empty / None / an announcer / music
    sentinel; char_id is not in ``cast_ids``; or the role is already
    "character". COERCE-NEVER-CRASH: any error -> (line, False)."""
    try:
        if not isinstance(line, dict):
            return line, False
        cid = str(line.get("char_id") or "").strip()
        if not cid or cid.lower() in _NON_CHARACTER_CHAR_ID_SENTINELS:
            return line, False
        if cid not in (cast_ids or set()):
            return line, False
        prev = str(line.get("speaker_role") or "")
        if prev == "character":
            return line, False
        line["speaker_role"] = "character"
        append_compose_flag(
            line,
            f"role_coerce:prev={prev or 'none'},new=character,"
            f"reason=cast_char_id,src={source or 'unknown'}",
        )
        return line, True
    except Exception:  # noqa: BLE001
        return line, False


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------

def _default_out_dir(episode_id: Optional[str] = None) -> str:
    """Per-episode audio dir for the ledger + Bark/MusicGen/AudioGen caches.

    Path: ``<output>/otr/episodes/<episode_id>/audio/``.

    History:
      - 2026-04-26 PM BUG-LOCAL-067: moved from ``output/old_time_radio/``
        to ``output/otr/audio/``.
      - 2026-05-02 EVENING (Jeffrey directive: one-stop-shop): moved into
        per-episode workspace ``output/otr/episodes/<ep>/audio/``. SignalLostVideo
        finalizes the canonical episode_id; until then the dir is named
        ``output/otr/episodes/pending_<ts>/audio/`` and gets renamed by
        ``Ledger.rename_episode`` once the title is finalized.
    """
    ep = episode_id or ("pending_" + time.strftime("%Y%m%d_%H%M%S"))
    # BUG-LOCAL-292: route through the shared resolver instead of a hardcoded
    # ~/Documents path. comfy_output_dir() resolves OTR_OUTPUT_DIR env, else
    # ComfyUI folder_paths, else a node-relative walk-up -- so the ledger lands
    # under the SAME output/otr/episodes/<ep>/audio tree as portraits, the
    # latest_ledger route, and every other _otr_paths consumer (no more
    # Documents-vs-AppData split). Lazy import: _otr_paths is pure (no back-dep
    # on this module), and importing at call time avoids any load-order risk.
    from ._otr_paths import otr_audio_dir
    return str(otr_audio_dir(ep))


def _slugify(s: str, limit: int = 60) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return (s[:limit] or "episode")


def _git_head_short() -> str:
    """Best-effort short HEAD hash. Cached per process. Never raises."""
    global _GIT_HEAD_CACHE
    if _GIT_HEAD_CACHE is not None:
        return _GIT_HEAD_CACHE
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.dirname(here)
        out = otr_proc.run(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stdout=otr_proc.PIPE,
            stderr=otr_proc.DEVNULL,
            timeout=5,
            check=True,
        ).stdout.decode("utf-8", errors="ignore").strip()
        _GIT_HEAD_CACHE = out or "unknown"
    except Exception:  # noqa: BLE001 -- any failure means we just skip
        _GIT_HEAD_CACHE = "unknown"
    return _GIT_HEAD_CACHE


def _word_count(text: str) -> int:
    return canonical_word_count(text)


def _char_count(text: str) -> int:
    return canonical_char_count(text)


def _safe_float(v: Any, default: Optional[float] = None) -> Optional[float]:
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def _safe_int(v: Any, default: int = 0) -> int:
    if v is None:
        return default
    try:
        return int(v)
    except (TypeError, ValueError):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return default


# ---------------------------------------------------------------------------
# Durable-field identity (S2 P1.1 / 720-bakeoff C1)
# ---------------------------------------------------------------------------
#
# The disk merge (BUG-LOCAL-108) carries out-of-band DURABLE renderer
# fields (wav/cache/timing/render stamps) forward from an earlier on-disk
# ledger so an incremental re-save does not destroy them. Before this fix
# that copy-forward was BLIND: an edited line, a re-authored music cue, or
# a changed clip render request would keep the STALE render bytes from the
# previous pass (wrong audio glued to new text). C1 gates the copy-forward
# on a per-row content IDENTITY: durable fields survive ONLY when identity
# is unchanged; a changed identity invalidates them (dropped, never
# resurrected). Identity is recomputed from CONTENT on both the in-memory
# and on-disk rows -- never read from a stored hash -- so legacy on-disk
# ledgers (written before the hash fields existed) migrate cleanly when
# their content is unchanged.

#: Authored music cue-spec fields hashed into ``cue_spec_sha256``. Ordered
#: for readability only; the hash is over a sorted-key JSON so order is
#: irrelevant to the digest.
_MUSIC_CUE_SPEC_FIELDS = (
    "generation_prompt", "target_duration_s", "placement", "anchor_line_id",
)

#: Clip render OUTPUT / timing fields excluded from the clip render-spec
#: identity -- they are products of the render, not the request that
#: produced it (l3 schema: clips[].warmup_pad_ms + humo_render_ms +
#: mp4_dur_s + mp4_frames + audio_fed_to_humo_dur_s, plus shared per-row
#: render stamps).
_CLIP_OUTPUT_FIELDS = frozenset({
    "warmup_pad_ms", "humo_render_ms", "mp4_dur_s", "mp4_frames",
    "audio_fed_to_humo_dur_s", "start_s", "dur_s", "start_s_space",
    "render_ms", "generated_dur_s", "audio_sample_hash",
    "render_spec_sha256",
})


def _canonical_sha256(payload: Any) -> str:
    """SHA256 hex of a canonical (sorted-key, compact) JSON encoding of
    ``payload``. Deterministic across runs and machines. Falls back to
    ``repr`` for non-JSON-serialisable payloads (best-effort; never
    raises)."""
    try:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
    except (TypeError, ValueError):
        blob = repr(payload)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _row_identity(arr_name: str, row: Dict[str, Any]) -> Optional[str]:
    """Content identity for a ROW_KEYED row (lines / music / clips),
    recomputed from the row's authored/source content. Returns ``None``
    when identity is undefined OR the identity SOURCE IS EMPTY -- the
    caller then falls back to the legacy (ungated) copy-forward.

    The empty-source -> None rule is deliberate: a line whose text was
    CLEARED to "" (a Script Doctor skip, or an authoritative empty
    rebuild) has no new content to render, so there is nothing stale to
    invalidate -- the out-of-band durable field (bark_wav_path, timings)
    is preserved per the ownership/skip contract (BUG-108 +
    test_ledger_merge_ownership). The identity gate only bites when the
    in-memory content is MEANINGFUL and DIFFERS from disk -- i.e. an edit
    that would otherwise glue a stale render onto changed text.

    - lines: sha256 of the canonical line text (the source for
      ``text_for_tts``); None when the text is empty/whitespace.
    - music: sha256 of the authored cue spec
      (``_MUSIC_CUE_SPEC_FIELDS``) == ``cue_spec_sha256``; None when the
      whole spec is empty.
    - clips: sha256 of the render-request fields, EXCLUDING render
      outputs/timing (``_CLIP_OUTPUT_FIELDS``) and the ``line_id`` merge
      key; None when no source fields remain.
    """
    if not isinstance(row, dict):
        return None
    if arr_name == "lines":
        text = _safe_str(row.get("text"))
        if not text.strip():
            return None
        return _canonical_sha256(text)
    if arr_name == "music":
        spec = {k: row.get(k) for k in _MUSIC_CUE_SPEC_FIELDS}
        if not any(v not in (None, "", [], {}) for v in spec.values()):
            return None
        return _canonical_sha256(spec)
    if arr_name == "clips":
        spec = {k: v for k, v in row.items()
                if k != "line_id" and k not in _CLIP_OUTPUT_FIELDS}
        if not spec:
            return None
        return _canonical_sha256(spec)
    return None


def music_cue_spec_sha256(row: Dict[str, Any]) -> Optional[str]:
    """Return the canonical authored identity for one ``music[]`` row.

    Music producers, the cue manifest, and post-audio consumers must use the
    exact same identity function.  Keeping this public wrapper beside the
    ledger merge gate prevents a second implementation from silently hashing
    a different field set or JSON encoding.
    """
    return _row_identity("music", row)


def _rebase_episode_local_paths(
    value: Any,
    old_episode_root: str,
    new_episode_root: str,
) -> "tuple[Any, int]":
    """Purely rebase absolute JSON paths inside a moved episode directory.

    ``Ledger.rename_episode`` moves the whole per-episode directory.  Every
    absolute ledger pointer below that directory must move with it, including
    nested renderer receipts that this module does not otherwise understand.
    A field-name allow-list is therefore the wrong ownership boundary.  This
    walker changes only absolute string *values* whose normalized path is
    contained by ``old_episode_root`` on a component boundary; external model,
    source, shared-cache, and OBS paths remain byte-identical.

    Both Windows and POSIX path spellings are handled deterministically even
    when tests run on the other platform.  The returned count makes the rename
    receipt auditable.  Input containers are not mutated.
    """
    old_root = str(old_episode_root or "")
    new_root = str(new_episode_root or "")
    if not old_root or not new_root:
        return value, 0

    # ``ntpath`` recognizes both C:\\x and C:/x on every host.  Use POSIX
    # semantics only when the root is genuinely POSIX-absolute.
    pathmod = ntpath if ntpath.isabs(old_root) else posixpath
    old_norm = pathmod.normcase(pathmod.normpath(old_root))
    new_norm = pathmod.normpath(new_root)

    def _walk(current: Any) -> "tuple[Any, int]":
        if isinstance(current, str):
            if not pathmod.isabs(current):
                return current, 0
            candidate_norm = pathmod.normcase(pathmod.normpath(current))
            try:
                inside = pathmod.commonpath((old_norm, candidate_norm)) == old_norm
            except (ValueError, OSError):
                inside = False
            if not inside:
                return current, 0
            relative = pathmod.relpath(pathmod.normpath(current), old_root)
            rebased = pathmod.normpath(pathmod.join(new_norm, relative))
            return rebased, int(rebased != current)
        if isinstance(current, dict):
            changed = 0
            rebuilt: Dict[Any, Any] = {}
            for key, item in current.items():
                rebuilt_item, item_count = _walk(item)
                rebuilt[key] = rebuilt_item
                changed += item_count
            return rebuilt, changed
        if isinstance(current, list):
            changed = 0
            rebuilt_list = []
            for item in current:
                rebuilt_item, item_count = _walk(item)
                rebuilt_list.append(rebuilt_item)
                changed += item_count
            return rebuilt_list, changed
        if isinstance(current, tuple):
            changed = 0
            rebuilt_tuple = []
            for item in current:
                rebuilt_item, item_count = _walk(item)
                rebuilt_tuple.append(rebuilt_item)
                changed += item_count
            return tuple(rebuilt_tuple), changed
        return current, 0

    return _walk(value)


def _same_durable_run(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    """Identity proof for a disk-to-singleton merge.

    The immutable freeze receipt survives the legitimate pending-to-title
    rename. If either side has one, both sides must have the same value.
    Older/pre-freeze records may fall back to an exact non-empty episode id
    only when neither side has a receipt. A stale or partially stripped record
    is never allowed to exchange producer-owned rows merely because a filename
    or episode id collides.
    """
    left_meta = left.get("meta") if isinstance(left.get("meta"), dict) else {}
    right_meta = right.get("meta") if isinstance(right.get("meta"), dict) else {}
    # A REPLAY WORKSPACE keeps the source's freeze receipt byte-identical (the
    # freeze consumers and the banana variety need it) and carries its own
    # workspace id (campaign item 0, 2026-09-02). When either side carries the
    # id, both must match -- otherwise a replay and its source would be one
    # durable run and a replay's post-audio join would write the ORIGINAL.
    left_ws = str(left_meta.get("replay_workspace_id") or "").strip()
    right_ws = str(right_meta.get("replay_workspace_id") or "").strip()
    if left_ws or right_ws:
        if not (left_ws and right_ws and left_ws == right_ws):
            return False
    left_freeze = str(left_meta.get("freeze_timestamp") or "").strip()
    right_freeze = str(right_meta.get("freeze_timestamp") or "").strip()
    if left_freeze or right_freeze:
        return bool(
            left_freeze
            and right_freeze
            and left_freeze == right_freeze
        )
    left_episode = str(left.get("episode_id") or "").strip()
    right_episode = str(right.get("episode_id") or "").strip()
    return bool(left_episode and right_episode and left_episode == right_episode)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def new_ledger(episode_id: Optional[str] = None,
               out_dir: Optional[str] = None) -> "Ledger":
    """Start a fresh ledger. Call at the beginning of a pipeline run.

    If episode_id is None, a placeholder ``pending_<timestamp>`` id is used;
    rename later via ledger.rename_episode(new_id) once the real filename
    is known (usually at SignalLostVideo stage).

    The ledger lives at ``<output>/otr/episodes/<episode_id>/audio/<episode_id>_ledger.json``;
    Bark / MusicGen / AudioGen caches sit alongside it in the same dir.
    """
    global _CURRENT
    with _LEDGER_LOCK:
        ep = episode_id or ("pending_" + time.strftime("%Y%m%d_%H%M%S"))
        _CURRENT = Ledger(ep, out_dir or _default_out_dir(ep))
        return _CURRENT


def get_ledger() -> "Ledger":
    """Return the current in-progress ledger, creating a placeholder if
    none exists. Safe to call from any stage."""
    global _CURRENT
    with _LEDGER_LOCK:
        if _CURRENT is None:
            ep = "pending_" + time.strftime("%Y%m%d_%H%M%S")
            _CURRENT = Ledger(ep, _default_out_dir(ep))
        return _CURRENT


# Wiring-review #3 (2026-05-11): non-creating accessors for downstream
# nodes that should fail loud when called without a writer upstream.
# The cascade (OTR_LedgerFreezeCascade) uses these instead of
# get_ledger() so it never operates on an empty placeholder ledger.


# --------------------------------------------------------------------------- #
# CANONICAL REPLAY (campaign item 0, 2026-09-02)
# --------------------------------------------------------------------------- #
REPLAY_MANIFEST_NAME = "manifest.json"
REPLAY_MANIFEST_SCHEMA = "otr_replay_bundle_v1"
#: Sections and meta keys the replay must not carry over from the source run.
#: Meta a replay REMAKES for itself, so carrying the source's copy would
#: publish the previous run's telemetry as if it described this one.
#:
#: EVERY MEMBER MUST BE SOMETHING THE REPLAY ACTUALLY RE-STAMPS, and that is the
#: whole rule (learned on the first live replay, 2026-09-03). `image_engines`
#: was in this list and had to come out: the credits roll REQUIRES it
#: (`otr_credits_roll._require(meta, "image_engines", "meta")`), and the only
#: thing that stamps it is `OTR_ImageGenDispatcher` -- which a replay does not
#: run, because a replay IMPORTS the source's stills instead of minting new
#: ones. So clearing it guaranteed a `CreditsDataError` at mux time on every
#: replay: eight clips rendered, sixteen minutes spent, and nothing published.
#:
#: Carrying it forward is not a stale value, it is the CORRECT one. The replay
#: shows the imported stills, so the engines that made them are exactly the
#: source's engines. (Verified on that live leg: the other six members --
#: `render_engines`, `render_trace`, `render_trace_version`, `phase_ms`,
#: `audio_motion_profile`, `paths` -- were all rebuilt by the replay, and
#: `video_readiness` is a freeze-cascade diagnostic that nothing requires.)
_REPLAY_RUN_VOLATILE_META = (
    "render_engines", "render_trace", "render_trace_version", "phase_ms",
    "video_readiness", "audio_motion_profile", "paths",
)


class ReplayBundleError(RuntimeError):
    """A replay bundle that cannot be trusted: manifest, digest or path fault."""


def replay_descriptor(meta: Any) -> Dict[str, str]:
    """The tiny replay descriptor a node reads off any ledger meta: empty when
    this is not a replay. Nodes branch on ``bool(replay_descriptor(meta))``."""
    if not isinstance(meta, dict):
        return {}
    src = str(meta.get("replay_from") or "").strip()
    if not src:
        return {}
    return {
        "replay_from": src,
        "replay_of_episode": str(meta.get("replay_of_episode") or ""),
        "replay_workspace_id": str(meta.get("replay_workspace_id") or ""),
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _safe_relative(rel: str) -> str:
    """A manifest path must be relative, normalized, and inside the bundle."""
    raw = str(rel or "")
    if not raw or os.path.isabs(raw) or raw.startswith(("/", "\\")):
        raise ReplayBundleError("manifest path is not relative: %r" % (raw,))
    norm = os.path.normpath(raw).replace("\\", "/")
    if norm.startswith("../") or norm == ".." or "/../" in norm or norm.startswith("/"):
        raise ReplayBundleError("manifest path escapes the bundle: %r" % (raw,))
    if ":" in norm:
        raise ReplayBundleError("manifest path carries a drive or stream: %r" % (raw,))
    return norm


def load_replay_manifest(bundle_dir: str) -> Dict[str, Any]:
    """Read and validate a bundle's manifest: schema, relative paths, no
    duplicates (case-folded), every file present with the recorded size and
    SHA-256. Raises ReplayBundleError on any fault; never repairs."""
    # `bundle_dir` reaches here from OTR_LedgerScriptWriter's `replay_from`
    # WIDGET, i.e. unauthenticated /prompt input. A UNC value would make
    # this machine open an SMB session and authenticate to the host the
    # workflow named, on the very first isfile() below (2026-09-04).
    try:
        from ._otr_paths import reject_remote_path as _reject_remote
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from _otr_paths import reject_remote_path as _reject_remote  # type: ignore
    _reject_remote(bundle_dir, "replay_from")
    bundle = os.path.abspath(bundle_dir)
    mpath = os.path.join(bundle, REPLAY_MANIFEST_NAME)
    if not os.path.isfile(mpath):
        raise ReplayBundleError("no %s under %s" % (REPLAY_MANIFEST_NAME, bundle))
    with open(mpath, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != REPLAY_MANIFEST_SCHEMA:
        raise ReplayBundleError("manifest schema is not %s" % REPLAY_MANIFEST_SCHEMA)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReplayBundleError("manifest lists no files")
    seen = set()
    for row in files:
        rel = _safe_relative(row.get("path"))
        key = rel.lower()
        if key in seen:
            raise ReplayBundleError("manifest lists %r twice (case-folded)" % (rel,))
        seen.add(key)
        full = os.path.join(bundle, *rel.split("/"))
        if os.path.islink(full):
            raise ReplayBundleError("manifest entry is a link: %r" % (rel,))
        if not os.path.isfile(full):
            raise ReplayBundleError("manifest entry missing on disk: %r" % (rel,))
        size = os.path.getsize(full)
        if int(row.get("bytes") or -1) != size or size <= 0:
            raise ReplayBundleError("manifest size mismatch for %r: %s vs %s"
                                    % (rel, row.get("bytes"), size))
        if str(row.get("sha256") or "") != _sha256_file(full):
            raise ReplayBundleError("manifest digest mismatch for %r" % (rel,))
    for key in ("ledger", "master_audio", "source_episode_id"):
        if not manifest.get(key):
            raise ReplayBundleError("manifest lacks %r" % (key,))
    _safe_relative(manifest["ledger"])
    _safe_relative(manifest["master_audio"])
    return manifest


def replay_episode_id(source_episode_id: str) -> str:
    """``<source>_replay_<stamp with microseconds>`` -- distinct per replay so
    the output dir and the obs file never collide (rename_episode hard-fails on
    an existing directory)."""
    import datetime as _dt
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    base = str(source_episode_id or "episode").strip() or "episode"
    return "%s_replay_%s" % (base, stamp)


def import_replay_bundle(bundle_dir: str, new_episode_id: Optional[str] = None) -> "Ledger":
    """ONE validated clone of a frozen episode into a NEW workspace, and the
    process singleton rebound to it (campaign item 0, 2026-09-02).

    What it does, in order: validate the manifest (schema, relative paths,
    digests); deep-copy the bundle's ledger; give it the new root episode id
    and remember the source under ``meta.replay_of_episode``; stamp
    ``meta.replay_from`` and a workspace-unique ``meta.replay_workspace_id``
    (the freeze receipt stays byte-identical); materialize the bundle's stills,
    portraits and canon into the new episode dir; rebase every episode-local
    path (``path`` AND ``pool_path``, and every other absolute pointer below
    the old root) onto the new dir; re-file the publication receipt under the
    new id; clear the source's terminal / obs pointers and its run-volatile
    telemetry (render engines, render trace, phase timings); ``new_ledger``
    so ``in_flight_ledger_path()`` binds the new workspace; save atomically.
    The master WAV is NOT copied here: node 7 (EpisodeAssembler) owns that
    file and copies it onto its canonical name, verified, before it emits
    ``audio_done``."""
    # `bundle_dir` reaches here from OTR_LedgerScriptWriter's `replay_from`
    # WIDGET, i.e. unauthenticated /prompt input. A UNC value would make
    # this machine open an SMB session and authenticate to the host the
    # workflow named, on the very first isfile() below (2026-09-04).
    try:
        from ._otr_paths import reject_remote_path as _reject_remote
    except ImportError:  # pragma: no cover -- flat (sys.path) load
        from _otr_paths import reject_remote_path as _reject_remote  # type: ignore
    _reject_remote(bundle_dir, "replay_from")
    bundle = os.path.abspath(bundle_dir)
    manifest = load_replay_manifest(bundle)
    source_id = str(manifest["source_episode_id"])
    new_id = str(new_episode_id or "").strip() or replay_episode_id(source_id)
    ledger_path = os.path.join(bundle, *manifest["ledger"].split("/"))
    with open(ledger_path, "r", encoding="utf-8") as fh:
        source = json.load(fh)
    if not isinstance(source, dict) or not source.get("lines"):
        raise ReplayBundleError("bundle ledger carries no lines")
    data = copy.deepcopy(source)
    meta = data.get("meta")
    if not isinstance(meta, dict):
        meta = {}
        data["meta"] = meta
    old_root = str(manifest.get("source_episode_root") or "")

    led = new_ledger(episode_id=new_id)          # binds _CURRENT + makes the dir
    new_audio_dir = led.out_dir
    new_root = os.path.dirname(new_audio_dir)
    # materialize the bundle's assets under the new episode dir, by relative path
    materialized = {}
    for row in manifest["files"]:
        rel = _safe_relative(row["path"])
        if rel in (manifest["ledger"], manifest["master_audio"]):
            continue
        src = os.path.join(bundle, *rel.split("/"))
        dst = os.path.join(new_root, *rel.split("/"))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)
        materialized[rel] = dst
    # identity + provenance
    data["episode_id"] = new_id
    meta["episode_id"] = new_id
    meta["replay_of_episode"] = source_id
    meta["replay_from"] = bundle
    meta["replay_workspace_id"] = uuid.uuid4().hex
    meta["replay_source_commit"] = str(manifest.get("source_commit") or "")
    meta["replay_master_audio"] = manifest["master_audio"]
    # A DERIVED bundle (scripts/otr_freeze_replay_bundle.py --derive-engine)
    # names the engine the replay must render on. Stamped RAW here -- this
    # module imports no engine registry; OTR_ShotLock's replay branch validates
    # the id (a registered Ghost sibling of the frozen plan's engine) and
    # rewrites the whole plan atomically before its durable stamp.
    meta["replay_engine_override"] = str(manifest.get("engine_override") or "")
    meta["replay_derived_from"] = str(manifest.get("derived_from") or "")
    meta["replay_master_sha256"] = next(
        (str(r.get("sha256") or "") for r in manifest["files"]
         if _safe_relative(r.get("path")) == manifest["master_audio"]), "")
    # the source's run-volatile telemetry and terminal pointers do not carry over
    for key in _REPLAY_RUN_VOLATILE_META:
        meta.pop(key, None)
    for key in ("final_audio_path", "final_video_path"):
        data[key] = None
    for key in ("obs_publish", "obs_path", "published_path"):
        meta.pop(key, None)
    # rebase every episode-local absolute path onto the new root
    if old_root:
        data, _n = _rebase_episode_local_paths(data, old_root, new_root)
    led.data = data
    led._rebase_publication_eligibility(new_id)
    path = led.save()
    if path is None:
        raise ReplayBundleError("import_replay_bundle: the imported ledger did not save")
    log.info("[Ledger] replay workspace %s imported from %s (source %s, %d asset(s))",
             new_id, bundle, source_id, len(materialized))
    return led


def peek_ledger() -> Optional["Ledger"]:
    """Return the current ledger or None. Never creates a placeholder.

    Use this from downstream nodes (reviewer, sequencer, audit
    artifacts) where an absent ledger is an error condition rather
    than a fresh-run signal.
    """
    with _LEDGER_LOCK:
        return _CURRENT


class LedgerStampError(RuntimeError):
    """S2 durable-persistence contract (credits enrichment 2026-07-03):
    a required credits stamp could not be persisted to the production
    ledger singleton. LOUD by design -- the late OTR_CreditsRoll node can
    only read what is on disk, so a silent save miss here becomes a
    missing-receipt failure at mux time. Never catch-and-continue."""


def stamp_durable(*, sections: Optional[Dict[str, Any]] = None,
                  meta_updates: Optional[Dict[str, Any]] = None,
                  source: str = "") -> Optional[str]:
    """Copy credit-bearing sections from a LOCAL wire-parsed ledger dict
    into the process SINGLETON and persist, LOUDLY.

    S2 gotcha this exists for: CastLock and the image dispatcher stamp a
    LOCAL dict parsed off the wire (``load_ledger(script_json)``), never
    the singleton -- so a bare ``get_ledger().save()`` would persist the
    singleton's EMPTY/stale state and silently drop the stamps. Callers
    hand the updated sections here and this helper does the copy + save.

    - ``sections``: top-level keys replaced wholesale on the singleton
      (e.g. ``{"cast": [...]}``, ``{"images": {...}}``).
    - ``meta_updates``: merged into ``data["meta"]`` (additive).
    - Production: ``Ledger.save()`` returns ``None`` on failure and never
      raises, so the return IS checked here and a ``None`` RAISES
      ``LedgerStampError`` (Fable BUILD-BREAKER #3).
    - Test mode (``OTR_TEST_MODE=1``): explicit injection path -- the
      in-memory singleton copy still happens (tests can read it back) but
      the disk save is skipped, so no placeholder ledgers leak into the
      real output tree. Returns ``None`` in that case.
    """
    led = get_ledger()
    for key, value in (sections or {}).items():
        led.data[key] = value
    if meta_updates:
        meta = led.data.setdefault("meta", {})
        if not isinstance(meta, dict):
            meta = {}
            led.data["meta"] = meta
        meta.update(meta_updates)
    if otr_env.get("OTR_TEST_MODE") == "1":
        log.info("[Ledger] stamp_durable(%s): test-mode injection "
                 "(in-memory only, disk save skipped)", source or "?")
        return None
    path = led.save()
    if path is None:
        raise LedgerStampError(
            f"stamp_durable({source or '?'}): Ledger.save() returned None -- "
            f"durable credits stamp NOT persisted (sections="
            f"{sorted((sections or {}).keys())}, meta_keys="
            f"{sorted((meta_updates or {}).keys())})"
        )
    return path


def has_current_ledger() -> bool:
    """True iff a writer-produced ledger with at least one line row
    exists in the current process. Use this as the gate before any
    downstream LLM work that would be wasteful on an empty ledger.
    """
    with _LEDGER_LOCK:
        if _CURRENT is None:
            return False
        lines = _CURRENT.data.get("lines", []) or []
        return len(lines) > 0


# Post-Phase-3 review (Fix 3, 2026-05-11) — §6.G word-count stamping.
# The existing `_recompute_totals` maintains the ROOT-level
# `total_word_count` which conflates character + announcer (used by
# the save-log message). Per §6.G the BUDGET-enforcement reader needs
# a character-only `meta.character_word_count` plus separate
# announcer + total fields for forensic visibility. Stamped at end
# of writer's run() and again at end of reviewer's review_ledger()
# after every commit/restore path. Reversing my D3 critique on
# 2026-05-11 — the spec field is load-bearing, not "lean docs"
# decoration.


def stamp_word_counts(ledger: "Ledger") -> None:
    """Stamp the three §6.G word-count totals onto `ledger.data.meta`.

    Skipped lines (lines[k].skip=True per Step 2.5 / Script Doctor
    skip action) contribute zero to all three counts -- a phantom-
    muted line is gone from the budget enforcement view.

    Pure Python. Never raises. Stamps:
      meta.character_word_count -- authoritative for "did we hit
                                   the target_words character
                                   dialogue budget?" reads
      meta.announcer_word_count -- forensic only
      meta.total_word_count     -- character + announcer; for
                                   user-facing display

    Existing root-level `data["total_word_count"]` maintained by
    `_recompute_totals` is unchanged (used by the save log line).
    """
    if ledger is None:
        return
    data = ledger if isinstance(ledger, dict) else getattr(ledger, "data", None)
    if not isinstance(data, dict):
        return
    char_n = 0
    ann_n = 0
    for line in data.get("lines", []) or []:
        if line.get("skip"):
            continue
        text = line.get("text") or ""
        n = canonical_word_count(text)
        role = line.get("speaker_role", "")
        if role == "character":
            char_n += n
        elif role == "announcer":
            ann_n += n
    meta = data.setdefault("meta", {})
    meta["character_word_count"] = char_n
    meta["announcer_word_count"] = ann_n
    meta["total_word_count"] = char_n + ann_n


def refresh_ledger_text_metrics(ledger: Any) -> None:
    """Refresh row, root, and budget-facing derived text metrics.

    Production ``Ledger`` instances use their full aggregate roll-up.  The
    lightweight fallback keeps tests and adapter-shaped ledger objects honest
    without requiring a concrete Ledger class.
    """
    if ledger is None:
        return
    data = ledger if isinstance(ledger, dict) else getattr(ledger, "data", None)
    if not isinstance(data, dict):
        return
    recompute = getattr(ledger, "_recompute_totals", None)
    if callable(recompute):
        recompute()
    else:
        total_chars = 0
        total_words = 0
        total_lines = 0
        for row in data.get("lines", []) or []:
            if not isinstance(row, dict):
                continue
            stamp_line_text_metrics(row)
            total_chars += _safe_int(row.get("char_count"))
            total_words += _safe_int(row.get("word_count"))
            total_lines += 1
        data["total_char_count"] = total_chars
        data["total_word_count"] = total_words
        data["total_dialogue_lines"] = total_lines
    stamp_word_counts(ledger)


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

class Ledger:
    """One episode's production ledger. Thread-safe for single-graph
    ComfyUI runs (sequential node execution)."""

    # Schema version bumped 2026-04-25 PM:
    #   l1-2026-04-24  -> baseline (cast, scenes, shots, lines, sfx, music, clips)
    #   l2-2026-04-25  -> adds beats[] hierarchy
    #   l3-2026-04-28  -> diagnostic expansion (meta.phase_ms,
    #                     audio_gates[] sha256, lines[].text_for_tts +
    #                     bark_wav_dur_s + bark_render_ms,
    #                     clips[].warmup_pad_ms + humo_render_ms +
    #                     mp4_dur_s + mp4_frames + audio_fed_to_humo_dur_s,
    #                     start_s_space tag, transitions[] crossfades,
    #                     radio_bookend_path) -- see _otr_ledger.py
    #                     and BUG-LOCAL-100..107 entries.
    #
    # Beat = single-speaker continuous turn within a shot. Hierarchy:
    #     Scene > Shot > Beat > Clip
    # See ROADMAP Goal 3 + docs/2026-04-25-humo-continuity-brief.md.
    #
    # SCHEMA_VERSION pulled live from _otr_ledger so the two ledger
    # write paths stay in lockstep. Falls back to a hardcoded l3
    # string if _otr_ledger import fails (defensive).
    try:
        from . import _otr_ledger as _OTRL_FOR_SCHEMA  # type: ignore
        SCHEMA_VERSION = _OTRL_FOR_SCHEMA.CURRENT_SCHEMA_VERSION
        del _OTRL_FOR_SCHEMA
    except Exception:  # pragma: no cover -- defensive fallback
        SCHEMA_VERSION = "l4-2026-08-07"

    def __init__(self, episode_id: str, out_dir: str):
        self.episode_id = episode_id
        self.out_dir = out_dir
        try:
            os.makedirs(out_dir, exist_ok=True)
        except Exception as e:  # noqa: BLE001
            log.warning("[Ledger] out_dir make failed: %s", e)
        self.data: Dict[str, Any] = {
            "schema_version": self.SCHEMA_VERSION,
            "episode_id": episode_id,
            "commit": _git_head_short(),
            "total_episode_dur_s": None,
            "total_char_count": 0,
            "total_word_count": 0,
            "total_dialogue_lines": 0,
            "total_beats": 0,
            "cast": [],
            "scenes": [],
            "shots": [],
            "beats": [],
            "lines": [],
            "music": [],
            "clips": [],
            "audio": {},
            "final_audio_path": None,
            "final_video_path": None,
        }

    # -- identity ------------------------------------------------------


    def _rebase_publication_eligibility(self, new_id: str) -> None:
        """Carry the publication receipt's episode id across a rename.

        THE RECEIPT IS AN EPISODE-LOCAL DURABLE POINTER, and this method
        already owns rebasing every one of those onto the new identity -- it
        just predates this particular field.

        WHY IT MATTERS (live, 2026-08-15). The freeze stamps the publication
        verdict while the episode is still `pending_<ts>`; this rename then
        gives it its real slug. The terminal mux compares the receipt's episode
        id against the live ledger's, found `pending_...` versus
        `signal_lost_...`, read that as a STALE SINGLETON and withheld the OBS
        copy -- on every episode, because every episode is renamed. Two
        finished, correct episodes stayed unpublished.

        This does NOT re-decide anything. The verdict and its reasons are
        untouched; only the name the receipt files itself under moves, exactly
        as the durable paths above it do. Rights stay owned by the one producer
        at the freeze.
        """
        meta = self.data.get("meta")
        if not isinstance(meta, dict):
            return
        try:
            from ._otr_publication_eligibility import (
                PUBLICATION_ELIGIBILITY_META_KEY,
            )
        except ImportError:  # pragma: no cover -- flat/standalone load
            try:
                from _otr_publication_eligibility import (  # type: ignore
                    PUBLICATION_ELIGIBILITY_META_KEY,
                )
            except ImportError:  # pragma: no cover
                return
        receipt = meta.get(PUBLICATION_ELIGIBILITY_META_KEY)
        if isinstance(receipt, dict) and receipt.get("episode_id") != new_id:
            receipt["episode_id"] = new_id

    def rename_episode(self, new_id: str) -> None:
        """Rename the episode id and atomically move BOTH the parent
        per-episode dir AND the ledger + treatment files inside.

        Invariant (BUG-LOCAL-015, Phase B 2026-05-02): rename_episode
        either completes with canonical episode dir + canonical ledger
        + canonical treatment, OR raises BEFORE mutating in-memory
        episode state. No silent split state. No "fallback to file-only
        rename" — that path was the root cause of confusing downstream
        crashes when the dir-move silently failed and the in-memory
        episode_id was advanced anyway.

        State matrix:
          old exists, new missing  -> retry os.replace 3x; hard-fail after
          old exists, new exists   -> hard-fail conflict (NEVER merge)
          old missing, new exists  -> accept as already moved (idempotent)
          old missing, new missing -> hard-fail (nothing to rename)
          old == new (case-insensitive normcase) -> no-op

        Order of operations after dir is in final position:
          1) Move dir (with retry)
          2) Rename ledger file to canonical name
          3) Rebase every episode-local durable path and atomically save
          4) Rebase/update in-memory state (episode_id, out_dir)
          5) Rename treatment + sidecar (<old>_*.txt -> <new>_*.txt)

        The durable path rewrite is required and fails loudly: continuing with
        pointers into the deleted pending directory creates a mechanically
        successful but false ledger. Sidecar renames remain best-effort.

        BUG-LOCAL-108 history (2026-04-29 morning): prior implementation
        only changed in-memory ``self.episode_id``. On-disk
        ``pending_<ts>_ledger.json`` was left orphaned, next ``save()``
        wrote a fresh file at the canonical path. Fields written by
        audio nodes' schema-l3 helpers between LLMScriptWriter and
        SignalLostVideo landed on the orphan and were silently lost.
        Phase B (BUG-LOCAL-015) replaces that file-only-rename fallback
        with a hard-fail + retry; treatment files are also renamed.
        """
        import time as _time

        old = self.episode_id
        if old == new_id:
            return

        old_audio_dir = self.out_dir   # otr/episodes/pending_<ts>/audio
        old_ep_dir = os.path.dirname(old_audio_dir)  # otr/episodes/pending_<ts>
        new_ep_dir = os.path.join(os.path.dirname(old_ep_dir), new_id)
        new_audio_dir = os.path.join(new_ep_dir, os.path.basename(old_audio_dir))

        # Case-insensitive same-path check (Windows). If old and new
        # ep_dirs resolve to the same on-disk path, treat as no-op.
        if (os.path.normcase(os.path.abspath(old_ep_dir))
                == os.path.normcase(os.path.abspath(new_ep_dir))):
            log.info("[Ledger] rename_episode: %s -> %s same dir, no-op",
                     old, new_id)
            self.episode_id = new_id
            self.data["episode_id"] = new_id
            self._rebase_publication_eligibility(new_id)
            return

        old_exists = os.path.isdir(old_ep_dir)
        new_exists = os.path.isdir(new_ep_dir)
        moved_dir = False

        if old_exists and new_exists:
            # CRITICAL: on Windows, os.replace ALWAYS fails when dest
            # dir exists, even if empty (Gemini consult correction
            # 2026-05-02). Refuse to merge or overwrite -- this is a
            # partial-state crash recovery case that needs human eyes.
            raise RuntimeError(
                f"[Ledger] rename_episode: both source and destination "
                f"episode directories exist ({old_ep_dir} -> {new_ep_dir}). "
                f"This is a partial-state from a previous crash or manual "
                f"copy. In-memory state NOT updated. Resolve by manually "
                f"merging or deleting one side, then re-queue."
            )
        if not old_exists and not new_exists:
            raise RuntimeError(
                f"[Ledger] rename_episode: neither source nor destination "
                f"episode directory exists ({old_ep_dir} -> {new_ep_dir}). "
                f"In-memory state NOT updated. Did the LLMScriptWriter "
                f"node fail to create the pending workspace?"
            )
        if old_exists and not new_exists:
            # Step 1: move the per-episode parent dir, with retry.
            # Defender / Search Indexer / human-held locks (Notepad,
            # VLC) all manifest as OSError here. 3 attempts at 0.5s
            # spacing (~1.5s total wall) catches the common transient
            # cases without making the failure feel hung.
            last_exc: Optional[BaseException] = None
            for attempt in range(3):
                try:
                    os.replace(old_ep_dir, new_ep_dir)
                    moved_dir = True
                    last_exc = None
                    log.info(
                        "[Ledger] per-episode dir moved %s -> %s "
                        "(attempt %d)",
                        os.path.basename(old_ep_dir),
                        os.path.basename(new_ep_dir),
                        attempt + 1,
                    )
                    break
                except OSError as exc:
                    last_exc = exc
                    log.warning(
                        "[Ledger] per-episode dir move attempt %d/3 "
                        "failed (%s -> %s): %s",
                        attempt + 1, old_ep_dir, new_ep_dir, exc,
                    )
                    if attempt < 2:
                        _time.sleep(0.5)
            if not moved_dir and last_exc is not None:
                raise RuntimeError(
                    f"[Ledger] rename_episode: per-episode dir move "
                    f"failed after 3 attempts ({old_ep_dir} -> "
                    f"{new_ep_dir}): {last_exc}. In-memory state NOT "
                    f"updated. Common causes on Windows: a file inside "
                    f"the source dir is open in Notepad / VLC / Explorer "
                    f"preview / your editor; close it and re-queue. If "
                    f"the lock persists, check Defender / Search Indexer "
                    f"or wait a few seconds before re-queueing."
                )
        elif not old_exists and new_exists:
            # Idempotent recovery: a previous run already moved the dir
            # but crashed before updating in-memory state. Accept the
            # final-dir layout and continue.
            log.info(
                "[Ledger] rename_episode: source missing, destination "
                "exists -- accepting as already-moved (%s)",
                new_ep_dir,
            )
            moved_dir = True

        # Step 2: rename the ledger file inside the (now-moved) audio dir to
        # the canonical filename.  This used to be best-effort, but a failure
        # here would make the required durable path rewrite impossible and the
        # next singleton save would silently drop out-of-band producer rows.
        old_ledger_path = os.path.join(
            new_audio_dir, f"{_slugify(old, limit=120)}_ledger.json"
        )
        new_ledger_path = os.path.join(
            new_audio_dir, f"{_slugify(new_id, limit=120)}_ledger.json"
        )
        if (old_ledger_path != new_ledger_path
                and os.path.exists(old_ledger_path)):
            try:
                os.replace(old_ledger_path, new_ledger_path)
                log.info(
                    "[Ledger] ledger file moved %s -> %s",
                    os.path.basename(old_ledger_path),
                    os.path.basename(new_ledger_path),
                )
            except OSError as exc:
                raise RuntimeError(
                    f"[Ledger] ledger file rename failed "
                    f"({old_ledger_path} -> {new_ledger_path}): {exc}. "
                    "Durable episode-local paths were NOT rewritten; retry "
                    "the idempotent rename after the file lock clears."
                ) from exc

        # Step 3: rebase the durable JSON itself before any later singleton
        # merge can copy stale pending-root paths back into memory.  This is a
        # recursive path-ownership rule, not a field allow-list: future nested
        # media receipts move safely while external refs remain untouched.
        disk_rebased = 0
        if os.path.exists(new_ledger_path):
            try:
                from pathlib import Path as _Path
                from . import _otr_ledger as _OTRL  # type: ignore

                with open(new_ledger_path, "r", encoding="utf-8") as _fh:
                    _disk_ledger = json.load(_fh)
                if not isinstance(_disk_ledger, dict):
                    raise TypeError(
                        "durable ledger root is "
                        f"{type(_disk_ledger).__name__}, expected object"
                    )
                _disk_ledger, disk_rebased = _rebase_episode_local_paths(
                    _disk_ledger, old_ep_dir, new_ep_dir,
                )
                _disk_ledger["episode_id"] = new_id
                if not _OTRL.save_ledger_safe(
                    _Path(new_ledger_path), _disk_ledger,
                ):
                    raise OSError("atomic save_ledger_safe returned False")
            except Exception as exc:  # noqa: BLE001 - required boundary
                raise RuntimeError(
                    "[Ledger] rename_episode: durable episode-local path "
                    f"rewrite failed for {new_ledger_path}: {exc}. "
                    "Pipeline stopped before consumers could observe stale "
                    "pending-root pointers; retry is idempotent."
                ) from exc

        # Step 4: only advance the singleton after the directory, canonical
        # ledger name, and required durable rewrite are all established.  If a
        # failure above occurs, the singleton keeps the old pending identity;
        # a retry sees old-missing/new-present and resumes safely.
        rebased_data, memory_rebased = _rebase_episode_local_paths(
            self.data, old_ep_dir, new_ep_dir,
        )
        if not isinstance(rebased_data, dict):  # defensive; ledger is a dict
            raise RuntimeError(
                "[Ledger] rename_episode: in-memory path rebase returned "
                f"{type(rebased_data).__name__}, expected object"
            )
        self.data = rebased_data
        self.episode_id = new_id
        self.data["episode_id"] = new_id
        self._rebase_publication_eligibility(new_id)
        self.out_dir = new_audio_dir

        # Step 5: rename treatment + any other owned txt sidecars from
        # <old>_*.txt to <new>_*.txt. Uses the SAME _slugify(..., limit=120)
        # as the ledger filename to keep prefixes consistent. Per consult
        # 2026-05-02 (all 3 reviewers): use the precise old-id prefix
        # glob, NOT a broad pending_* glob, to avoid catching unrelated
        # files. Best-effort: warn on individual failures.
        old_prefix = f"{_slugify(old, limit=120)}_"
        new_prefix = f"{_slugify(new_id, limit=120)}_"
        try:
            from pathlib import Path as _Path
            sidecar_dir = _Path(new_audio_dir)
            if sidecar_dir.exists():
                for tx in sidecar_dir.glob(f"{old_prefix}*.txt"):
                    suffix = tx.name[len(old_prefix):]
                    canon = sidecar_dir / f"{new_prefix}{suffix}"
                    if tx == canon:
                        continue
                    try:
                        os.replace(str(tx), str(canon))
                        log.info(
                            "[Ledger] sidecar moved %s -> %s",
                            tx.name, canon.name,
                        )
                    except OSError as exc:
                        log.warning(
                            "[Ledger] sidecar rename failed (%s -> %s): %s",
                            tx, canon, exc,
                        )
        except Exception as exc:  # noqa: BLE001
            # Sidecar walk itself failed -- log but don't raise.
            log.warning(
                "[Ledger] sidecar rename walk failed in %s: %s",
                new_audio_dir, exc,
            )

        log.info(
            "[Ledger] renamed %s -> %s (dir_moved=%s, "
            "durable_paths_rebased=%d, memory_paths_rebased=%d)",
            old, new_id, moved_dir, disk_rebased, memory_rebased,
        )

    @property
    def path(self) -> str:
        return os.path.join(
            self.out_dir,
            f"{_slugify(self.episode_id, limit=120)}_ledger.json",
        )

    # -- writers -------------------------------------------------------
    # All writers are tolerant: bad input gets logged but doesn't raise.
    # Each writer returns self so calls can be chained if useful.

    def set_cast(self, cast_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        rows: List[Dict[str, Any]] = []
        for r in cast_rows or []:
            # Cast field renamed 2026-05-10: description -> character_description.
            # S26-B3 cleanbreak: legacy `description` input key dropped;
            # callers MUST supply `character_description`.
            cdesc = _safe_str(r.get("character_description")) or None
            # Cast field added 2026-05-10: tts_model. Routing column
            # that says which TTS family the voice_preset belongs to
            # ("bark", "kokoro", future: "fish_speech", "cosyvoice", ...).
            # Lets downstream consumers route by reading the field
            # directly instead of pattern-matching the voice_preset
            # prefix.
            # S26-B3 cleanbreak: legacy voice_preset->tts_model derivation
            # dropped; callers MUST supply tts_model explicitly.
            tts_model = _safe_str(r.get("tts_model")) or None
            voice_preset = _safe_str(r.get("voice_preset")) or None
            # Cast field added 2026-05-10: voice_params. Model-
            # dependent dict (or None) where the casting LLM stores
            # per-character knobs it chose -- Bark might use
            # "temperature", Kokoro might use "speed", etc.
            # Allowed param shape per model lives in
            # config.cast_pools.VOICE_REGISTRY[<model>]["params_spec"].
            # Today the LLM does not populate this yet (Phase 2);
            # consumers fall back to their defaults when None.
            #
            # Validation: must be a dict or None. A non-dict, non-None
            # value (e.g. a stray string) gets coerced to None so the
            # ledger schema stays clean.
            vparams_in = r.get("voice_params")
            voice_params = vparams_in if isinstance(vparams_in, dict) else None
            row = {
                "char_id":               _safe_str(r.get("char_id")),
                "name":                  _safe_str(r.get("name")),
                "character_description": cdesc,
                "gender":                _safe_str(r.get("gender")) or None,
                "tts_model":             tts_model,
                "voice_preset":          voice_preset,
                "voice_params":          voice_params,
                "line_count":            _safe_int(r.get("line_count")),
                "word_count":            _safe_int(r.get("word_count")),
            }
            # presentation_gender (item 8 chunk 4, 2026-08-06): the gender the
            # DELIVERED voice presents as, stamped by CastLock from the reference
            # it actually chose. It has to be carried HERE or it evaporates --
            # this method rebuilds a FIXED row and silently drops every key it
            # does not name.
            #
            # Carried CONDITIONALLY, on the same reasoning as provider_voice_id
            # below: a writer-stage row that nobody has cast yet stays
            # byte-identical to the legacy contract, so the cast-row drift guard
            # keeps its teeth. Absence is read through by
            # _otr_roster_gender.get_presentation_gender, which falls back to the
            # row's normalized label -- which is also what makes the 1,595
            # pre-field ledgers readable rather than violations.
            pg = _safe_str(r.get("presentation_gender"))
            if pg:
                row["presentation_gender"] = pg
            acc = _safe_str(r.get("accent"))
            if acc:
                row["accent"] = acc
            dorth = _safe_str(r.get("dialogue_orthography"))
            if dorth:
                row["dialogue_orthography"] = dorth
            ssig = _safe_str(r.get("speech_signature"))
            if ssig:
                row["speech_signature"] = ssig
            rows.append(row)
        self.data["cast"] = rows
        return self

    def set_scenes(self, scene_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        rows: List[Dict[str, Any]] = []
        for r in scene_rows or []:
            rows.append({
                "scene_id":    _safe_str(r.get("scene_id")),
                "description": _safe_str(r.get("description")) or None,
                "env":         _safe_str(r.get("env")) or None,
                "line_count":  _safe_int(r.get("line_count")),
                "word_count":  _safe_int(r.get("word_count")),
            })
        self.data["scenes"] = rows
        return self

    def set_shots(self, shot_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        rows: List[Dict[str, Any]] = []
        for r in shot_rows or []:
            rows.append({
                "shot_id":        _safe_str(r.get("shot_id")),
                "scene_id":       _safe_str(r.get("scene_id")) or None,
                "description":    _safe_str(r.get("description")) or None,
                "visual_prompt":  _safe_str(r.get("visual_prompt")),
                "png_path":       _safe_str(r.get("png_path")) or None,
                "start_s":        _safe_float(r.get("start_s")),
                "dur_s":          _safe_float(r.get("dur_s")),
            })
        self.data["shots"] = rows
        return self

    def set_beats(self, beat_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        """Set the beats[] section.

        A beat is a single-speaker continuous turn within a shot. It is
        the unit at which the HuMo clip-fill rule applies (beats never
        cross speakers, so HuMo audio windows align cleanly).

        Hierarchy: Scene > Shot > Beat > Clip.

        Each input row is normalised to:
            beat_id, shot_id, scene_id, speaker, char_id,
            line_ids[], start_s, dur_s
        """
        rows: List[Dict[str, Any]] = []
        for r in beat_rows or []:
            line_ids_in = r.get("line_ids") or []
            line_ids = [_safe_str(x) for x in line_ids_in if _safe_str(x)]
            rows.append({
                "beat_id":   _safe_str(r.get("beat_id")),
                "shot_id":   _safe_str(r.get("shot_id")) or None,
                "scene_id":  _safe_str(r.get("scene_id")) or None,
                "speaker":   _safe_str(r.get("speaker")) or None,
                "char_id":   _safe_str(r.get("char_id")) or None,
                "line_ids":  line_ids,
                "start_s":   _safe_float(r.get("start_s")),
                "dur_s":     _safe_float(r.get("dur_s")),
            })
        self.data["beats"] = rows
        self.data["total_beats"] = len(rows)
        return self

    # -- Phase 2B (2026-05-11): progressive ledger writes ------------
    #
    # Pre-Phase-2B callers (build_test_ledger_from_director, scene
    # builders, etc.) still use `set_lines()` to stamp the full lines
    # list at the end of their pipeline. The v2.0-alpha writer
    # (OTR_LedgerScriptWriter) instead pre-stamps a skeleton via
    # `init_lines_from_outline()` immediately after outline validates
    # and then updates each row's text in place via `update_line_text()`
    # after every `compose_line` call -- saving the ledger between
    # every line so a mid-loop crash leaves a partial-but-coherent
    # ledger on disk (`text == ""` signals "row composed pending").
    #
    # NOTE on save frequency. The writer calls `led.save()` once per
    # composed line (~15 times for a 12-beat episode, ~25 times for
    # a 7-act 19-beat episode plus reviewer rewrites). Every save
    # goes through write-tmp + atomic-rename, which on a fast NVMe
    # SSD is microseconds and unremarkable. On slow / network /
    # antivirus-scanned storage this pattern would degrade. A future
    # storage migration should reconsider the per-line save cadence
    # and may want to coalesce writes (e.g. save every K lines).

    def init_lines_from_outline(
        self,
        outline,
        char_id_by_name: Optional[Dict[str, str]] = None,
    ) -> "Ledger":
        """Pre-stamp one canonical beat and line row per outline beat."""
        char_id_by_name = char_id_by_name or {}
        beats = list(getattr(outline, "beats", []) or [])
        rows: List[Dict[str, Any]] = []
        beat_rows: List[Dict[str, Any]] = []
        for beat in beats:
            def _g(key: str, default=""):
                if isinstance(beat, dict):
                    return beat.get(key, default)
                return getattr(beat, key, default)

            beat_id = _safe_str(_g("beat_id"))
            speaker = _safe_str(_g("speaker"))
            role = _safe_str(_g("speaker_role")) or "character"
            mood = _safe_str(_g("mood"))
            intent = _safe_str(_g("intent"))
            arc_phase = _safe_str(_g("arc_phase"))

            target_words_raw = _g("target_words", None)
            target_words = (
                _safe_int(target_words_raw)
                if target_words_raw is not None
                else None
            )
            dialogue_slot_raw = _g("dialogue_slot_id", None)
            dialogue_slot_id = (
                _safe_str(dialogue_slot_raw) or None
                if dialogue_slot_raw is not None
                else None
            )
            if role == "character":
                char_id = char_id_by_name.get(speaker, "")
            elif role == "announcer":
                char_id = "announcer"
            else:
                char_id = role

            # Spoken rows are filled by update_line_text. Music rows remain
            # render contracts and never carry transcript text.
            text = ""
            rows.append({
                "line_id":          beat_id,
                "shot_id":          None,
                "beat_id":          beat_id,
                # The owning beat's speaker, carried onto the line row so
                # the sealed ledger names who says each line (PBUG-20260814-01).
                "speaker":          speaker or None,
                "char_id":          char_id,
                "text":             text,
                "traits":           mood or None,
                "boundary":         None,
                "char_count":       _char_count(text),
                "word_count":       _word_count(text),
                "bark_wav_path":    None,
                "start_s":          None,
                "dur_s":            None,
                "speaker_role":     role,
                "arc_phase":        arc_phase or None,
                "compose_flags":    [],
                "beat_intent":      intent or None,
                "target_words":     target_words,
                "dialogue_slot_id": dialogue_slot_id,
            })
            beat_rows.append({
                "beat_id":  beat_id,
                "shot_id":  _g("shot_id", None),
                "scene_id": _g("scene_id", None),
                "speaker":  speaker or None,
                "char_id":  char_id or None,
                "line_ids": [beat_id] if beat_id else [],
                "start_s":  _g("start_s", None),
                "dur_s":    _g("dur_s", None),
            })

        self.data["lines"] = rows
        self.set_beats(beat_rows)
        self._recompute_totals()
        return self

    def update_line_text(
        self,
        beat_id: str,
        text: str,
        compose_flags_append: Optional[Iterable[str]] = None,
    ) -> bool:
        """Update one existing line and its derived text metrics atomically."""
        safe_text = text or ""
        target = _safe_str(beat_id)
        for row in self.data["lines"]:
            if row.get("beat_id") == target or row.get("line_id") == target:
                set_line_text_metrics(row, safe_text)
                if safe_text.strip():
                    row.pop("skip", None)
                    row.pop("tts_skip_reason", None)
                if compose_flags_append:
                    for flag in compose_flags_append:
                        append_compose_flag(row, flag)
                self._recompute_totals()
                return True
        return False

    def set_lines(self, line_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        """Replace lines[] while preserving the canonical row schema."""
        rows: List[Dict[str, Any]] = []
        for raw in line_rows or []:
            text = _safe_str(raw.get("text"))
            beat_intent_raw = raw.get("beat_intent")
            beat_intent = (
                _safe_str(beat_intent_raw) or None
                if beat_intent_raw is not None
                else None
            )
            target_words_raw = raw.get("target_words")
            target_words = (
                _safe_int(target_words_raw)
                if target_words_raw is not None
                else None
            )
            dialogue_slot_raw = raw.get("dialogue_slot_id")
            dialogue_slot_id = (
                _safe_str(dialogue_slot_raw) or None
                if dialogue_slot_raw is not None
                else None
            )
            compose_flags_raw = raw.get("compose_flags")
            compose_flags = (
                list(compose_flags_raw)
                if isinstance(compose_flags_raw, list)
                else []
            )
            rows.append({
                "line_id":          _safe_str(raw.get("line_id")),
                "shot_id":          _safe_str(raw.get("shot_id")) or None,
                "beat_id":          _safe_str(raw.get("beat_id")) or None,
                # PBUG-20260814-01: `speaker` was absent from this
                # enumerated schema while the sibling normalizer
                # `set_beats()` carried it, so every caller that supplied a
                # speaker name on a line row had it silently discarded and
                # every published ledger shipped `speaker: None` on every
                # spoken row. A key-enumerating normalizer drops everything
                # it does not name; the asymmetry against its sibling was
                # the tell. Keep these two schemas in step.
                "speaker":          _safe_str(raw.get("speaker")) or None,
                "char_id":          _safe_str(raw.get("char_id")) or None,
                "text":             text,
                "traits":           _safe_str(raw.get("traits")) or None,
                "boundary":         _safe_str(raw.get("boundary")) or None,
                "char_count":       _char_count(text),
                "word_count":       _word_count(text),
                "bark_wav_path":    _safe_str(raw.get("bark_wav_path")) or None,
                "start_s":          _safe_float(raw.get("start_s")),
                "dur_s":            _safe_float(raw.get("dur_s")),
                "speaker_role":     _safe_str(raw.get("speaker_role")) or "character",
                "arc_phase":        _safe_str(raw.get("arc_phase")) or None,
                "compose_flags":    compose_flags,
                "beat_intent":      beat_intent,
                "target_words":     target_words,
                "dialogue_slot_id": dialogue_slot_id,
            })

        cast_ids = cast_ids_from_ledger(self.data)
        if cast_ids:
            for row in rows:
                coerce_speaker_role_for_char_id(
                    row, cast_ids, source="set_lines")
        self.data["lines"] = rows
        self._recompute_totals()
        return self

    # set_sfx + apply_sfx_timings deleted S27 (cleanbreak-tail Item 2).
    # The S25/CD-3 + S26-A3 sweep had already removed the sfx[] schema
    # scaffold from the canonical ledger; these two methods were kept
    # alive only to forward stale sfx rows from old on-disk ledger
    # files back into memory at save time. Per S27 directive: old
    # on-disk ledgers rebuild on next run. Zero current producers was
    # the green light to delete, not the excuse to preserve. Audio
    # writes flow through nodes/_otr_ledger.save_ledger_safe directly.

    def set_music(self, music_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        rows: List[Dict[str, Any]] = []
        for r in music_rows or []:
            row = {
                "cue_id":            _safe_str(r.get("cue_id")),
                "description":       _safe_str(r.get("description")) or None,
                "generation_prompt": _safe_str(r.get("generation_prompt")) or None,
                # Authored cue-spec fields (S2 P1.1 / C1): CARRIED so the
                # durable-field identity (cue_spec_sha256) survives an
                # incremental re-save. Dropped before this fix -- set_music
                # silently discarded placement / anchor_line_id /
                # target_duration_s, so a re-authored cue could not be told
                # apart from its predecessor and kept a stale rendered wav.
                "anchor_line_id":    _safe_str(r.get("anchor_line_id")) or None,
                "placement":         _safe_str(r.get("placement")) or None,
                "target_duration_s": _safe_float(r.get("target_duration_s")),
                # Durable render fields -- identity-gated on cue_spec_sha256
                # by _merge_with_disk (copied forward only on identity match).
                "wav_path":          _safe_str(r.get("wav_path")) or None,
                "start_s":           _safe_float(r.get("start_s")),
                "dur_s":             _safe_float(r.get("dur_s")),
            }
            # Stamp the authored-spec identity with the SAME function the
            # merge gate uses, so the persisted value and the gate agree
            # by construction.
            row["cue_spec_sha256"] = music_cue_spec_sha256(row)
            rows.append(row)
        self.data["music"] = rows
        return self

    def set_final_paths(self,
                        audio_path: Optional[str] = None,
                        video_path: Optional[str] = None,
                        total_episode_dur_s: Optional[float] = None) -> "Ledger":
        if audio_path is not None:
            self.data["final_audio_path"] = str(audio_path)
        if video_path is not None:
            self.data["final_video_path"] = str(video_path)
        if total_episode_dur_s is not None:
            self.data["total_episode_dur_s"] = _safe_float(total_episode_dur_s)
        return self

    # -- back-fill timings --------------------------------------------
    # Called by SceneSequencer / SignalLostVideo once it knows the timeline
    # positions. The argument is a mapping id -> (start_s, dur_s).

    def apply_line_timings(self, timing: Dict[str, Dict[str, float]]) -> "Ledger":
        for row in self.data["lines"]:
            t = timing.get(row.get("line_id"))
            if t:
                row["start_s"] = _safe_float(t.get("start_s"))
                row["dur_s"]   = _safe_float(t.get("dur_s"))
                if t.get("bark_wav_path"):
                    row["bark_wav_path"] = str(t["bark_wav_path"])
        return self

    def apply_music_timings(self, timing: Dict[str, Dict[str, float]]) -> "Ledger":
        for row in self.data["music"]:
            t = timing.get(row.get("cue_id"))
            if t:
                row["start_s"] = _safe_float(t.get("start_s"))
                row["dur_s"]   = _safe_float(t.get("dur_s"))
                if t.get("wav_path"):
                    row["wav_path"] = str(t["wav_path"])
        return self

    # -- totals --------------------------------------------------------

    def _recompute_totals(self) -> None:
        total_chars = 0
        total_words = 0
        total_lines = 0
        per_char_count: Dict[str, int] = {}
        per_char_words: Dict[str, int] = {}
        per_scene_count: Dict[str, int] = {}
        per_scene_words: Dict[str, int] = {}
        for ln in self.data["lines"]:
            stamp_line_text_metrics(ln)
            total_lines += 1
            total_chars += _safe_int(ln.get("char_count"))
            total_words += _safe_int(ln.get("word_count"))
            char_id = ln.get("char_id")
            if char_id:
                per_char_count[char_id] = per_char_count.get(char_id, 0) + 1
                per_char_words[char_id] = per_char_words.get(char_id, 0) + _safe_int(ln.get("word_count"))
            # derive scene_id via shot_id -> shot -> scene_id
            shot_id = ln.get("shot_id")
            if shot_id:
                sc = next((s.get("scene_id") for s in self.data["shots"]
                           if s.get("shot_id") == shot_id), None)
                if sc:
                    per_scene_count[sc] = per_scene_count.get(sc, 0) + 1
                    per_scene_words[sc] = per_scene_words.get(sc, 0) + int(ln.get("word_count") or 0)
        self.data["total_char_count"] = total_chars
        self.data["total_word_count"] = total_words
        self.data["total_dialogue_lines"] = total_lines
        for row in self.data["cast"]:
            cid = row.get("char_id")
            row["line_count"] = per_char_count.get(cid, 0)
            row["word_count"] = per_char_words.get(cid, 0)
        for row in self.data["scenes"]:
            sid = row.get("scene_id")
            row["line_count"] = per_scene_count.get(sid, 0)
            row["word_count"] = per_scene_words.get(sid, 0)

    def refresh_text_metrics(self) -> "Ledger":
        """Re-derive every persisted count from canonical line text."""
        refresh_ledger_text_metrics(self)
        return self

    # -- save ----------------------------------------------------------

    def save(self) -> Optional[str]:
        """Write the ledger to disk. Returns the path on success, None on
        failure. Never raises.

        BUG-LOCAL-108 (2026-04-29 morning): merge with on-disk content
        BEFORE writing so that schema-l3 fields written by audio
        nodes (via _otr_ledger.save_ledger_safe) are not silently
        clobbered. The Ledger class doesn't know about
        ``meta``/``audio_gates``/``transitions``/``radio_bookend_path``
        or the per-row ``start_s_space``/``text_for_tts``/
        ``warmup_pad_ms``/``bark_wav_dur_s`` fields, but it MUST NOT
        destroy them when its in-memory state is flushed.
        """
        try:
            self.refresh_text_metrics()
            path = self.path

            # Build the payload to write: start from in-memory data,
            # then merge any schema-l3 extras from the existing
            # on-disk version. Per-row merge is keyed by line_id /
            # cue_id so audio-node row updates survive.
            merged = self._merge_with_disk(dict(self.data), path)

            # BUG-LOCAL-018 (Phase E, 2026-05-02): stamp meta.paths block
            # so downstream nodes can look up canonical episode dirs
            # without reconstructing them from episode_id. Resolved
            # fresh on every save from the actual on-disk path -- so if
            # rename_episode (Phase B) moved the per-episode dir between
            # saves, this picks up the new location automatically.
            try:
                from pathlib import Path as _Path
                from . import _otr_ledger as _OTRL_PATHS  # type: ignore
                _meta = merged.setdefault("meta", {})
                _meta["paths"] = _OTRL_PATHS._build_meta_paths(
                    _Path(path),
                    str(self.episode_id),
                    published_obs_path=_meta.get("obs_final_path"),
                )
                _meta["schema_version"] = _OTRL_PATHS.CURRENT_SCHEMA_VERSION
                merged["schema_version"] = _OTRL_PATHS.CURRENT_SCHEMA_VERSION
            except Exception as exc:  # noqa: BLE001
                # Best-effort: a meta-stamping failure must NEVER break
                # the actual ledger write.
                log.warning("[Ledger] meta.paths stamp failed: %s", exc)

            # Atomic write: temp file + replace
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(merged, f, indent=2, ensure_ascii=False)
                f.flush()
                try:
                    os.fsync(f.fileno())
                except OSError:
                    pass
            os.replace(tmp, path)
            # Wiring-review #5 (2026-05-11): assign merged payload
            # back to self.data so the in-memory ledger matches the
            # on-disk JSON byte-for-byte. Without this, the next read
            # of `led.data` returns pre-merge state and
            # `json.dumps(led.data)` for downstream consumers (writer
            # slot, reviewer slot) diverges from disk.
            self.data = merged
            log.info("[Ledger] saved %s (%d lines, %d words)",
                     os.path.basename(path),
                     self.data["total_dialogue_lines"],
                     self.data["total_word_count"])
            return path
        except Exception as e:  # noqa: BLE001
            log.warning("[Ledger] save failed: %s", e)
            return None

    @staticmethod
    def _merge_with_disk(in_mem: Dict[str, Any], path: str) -> Dict[str, Any]:
        """BUG-LOCAL-108 helper. Read the on-disk JSON at ``path`` (if
        any) and merge any schema-l3 fields the Ledger class doesn't
        know about into ``in_mem``. Returns the merged dict. On any
        error returns ``in_mem`` unchanged (best-effort).

        Top-level fields preserved from disk:
          schema_version, meta, audio, audio_gates, transitions,
          radio_bookend_path, final_audio_path

        Per-row fields preserved (keyed by line_id / cue_id):
          For each lines[i] / clips[i] / music[i]: copy forward any
          key present on disk that is missing or empty/null in the
          in-mem row. (The ``sfx[]`` array was ripped 2026-07-01
          rip-sfx-broll / S27; it is no longer a copied row shape --
          see the ROW_KEYED map below.)
        """
        try:
            if not os.path.exists(path):
                return in_mem
            with open(path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("[Ledger] BUG-108 merge: on-disk read failed: %s", exc)
            return in_mem

        # Capture identity BEFORE any missing metadata is copied from disk.
        # Otherwise a stale sibling could donate its own freeze receipt into an
        # empty wire object and then appear to match itself.
        same_durable_run = _same_durable_run(in_mem, on_disk)

        # Top-level fields the Ledger class doesn't manage but must
        # not destroy. final_audio_path is on the kept-list because
        # SignalLostVideo overwrites it via set_final_paths -- that's
        # an explicit overwrite, not a merge concern.
        TOP_PRESERVE = (
            "schema_version", "audio", "audio_gates", "transitions",
            "radio_bookend_path",
            # THE PLANNED VIDEO SECTION (campaign item 0, 2026-09-02): ShotLock
            # stamps it durably through stamp_durable; every later save keeps it.
            "video",
        )
        for k in TOP_PRESERVE:
            if k in on_disk and (k not in in_mem or in_mem.get(k) in (None, "", [], {})):
                in_mem[k] = on_disk[k]

        # `meta` is recursive: BUG-LOCAL-018 (7c84ee8) added meta.paths
        # which made in_mem["meta"] always non-empty at save time, so
        # the bulk-replace rule above stopped copying disk meta. That
        # silently dropped schema-l3 phase telemetry written by audio
        # nodes (meta.phase_ms.bark, meta.git_commit, etc.) on every
        # SignalLostVideo dual-ledger save. Fix: per-key merge -- disk
        # wins where in-mem doesn't have a key or has an empty value;
        # in-mem wins where it has a real value.
        if "meta" in on_disk:
            disk_meta = on_disk.get("meta") or {}
            if not isinstance(disk_meta, dict):
                disk_meta = {}
            in_mem_meta = in_mem.get("meta")
            if not isinstance(in_mem_meta, dict):
                in_mem_meta = {}
            for mk, mv in disk_meta.items():
                if mk not in in_mem_meta or in_mem_meta.get(mk) in (None, "", [], {}):
                    in_mem_meta[mk] = mv
            in_mem["meta"] = in_mem_meta

        # Per-row merge. Keyed by line_id (lines, clips) or cue_id
        # (music). "sfx": "cue_id" entry deleted S27 (cleanbreak-tail
        # Item 2) -- the sfx[] schema scaffold was deleted in S26-A3
        # and the ROW_KEYED entry was kept only to forward stale rows
        # from old on-disk ledger files. Old ledgers rebuild on next
        # run; the merge no longer tries to preserve a deleted shape.
        ROW_KEYED = {
            "lines": "line_id",
            "clips": "line_id",
            "music": "cue_id",
        }
        # Canonical row fields the in-memory Ledger OWNS outright: the
        # disk merge must never resurrect stale values into them --
        # neither when the fresh in-memory value is deliberately falsy
        # (external QA round 1, 2026-07-10) NOR when a fresh
        # authoritative rebuild simply OMITS the key (external QA r2
        # P0.2: `set_lines()` builds rows without `skip`/
        # skip state, and the old
        # key-presence test read that omission as "please restore old
        # disk state" -- resurrecting a stale skip flag onto a newly
        # non-empty line -> Phase 10 freeze error). Ownership is at the
        # ROW level: a row present in memory is the authoritative
        # composition; disk may only contribute out-of-band DURABLE
        # renderer fields (bark_wav_path, timings, render stamps --
        # BUG-LOCAL-108). Per-field source/revision identity for the
        # durable set is the S2 P1.1 follow-up.
        _MERGE_OWNED_ROW_FIELDS = frozenset({
            # composition / content
            "text", "char_count", "word_count", "traits", "boundary",
            "char_id", "speaker_role", "arc_phase", "compose_flags",
            "beat_intent", "target_words", "dialogue_slot_id",
            "shot_id", "beat_id",
            # skip / editorial state
            "skip", "tts_skip_reason", "needs_render_realign",
            # authored music cue spec (S2 P1.1 / C1): OWNED like line
            # composition -- a re-authored cue must never have its spec
            # resurrected from a stale disk row (that would make the
            # identity falsely match and keep the wrong wav). cue_id is
            # the merge key (never copied). The DURABLE music render
            # fields (wav_path / start_s / dur_s) are deliberately NOT
            # here: they copy forward on an identity MATCH.
            "description", "generation_prompt", "anchor_line_id",
            "placement", "target_duration_s", "cue_spec_sha256",
        })
        for arr_name, key_field in ROW_KEYED.items():
            on_disk_rows = on_disk.get(arr_name) or []
            in_mem_rows = in_mem.get(arr_name) or []
            if not on_disk_rows or not in_mem_rows:
                # If disk has rows and memory doesn't, prefer disk.
                if on_disk_rows and not in_mem_rows:
                    in_mem[arr_name] = on_disk_rows
                continue
            on_disk_map = {
                r.get(key_field): r for r in on_disk_rows
                if r.get(key_field)
            }
            for row in in_mem_rows:
                key = row.get(key_field)
                if not key or key not in on_disk_map:
                    continue
                disk_row = on_disk_map[key]
                # DURABLE-FIELD IDENTITY (S2 P1.1 / C1): the copy-forward
                # below exists only to preserve out-of-band DURABLE render
                # fields (wav/cache/timing/render stamps). Those are valid
                # ONLY while the row's authored/source content is unchanged.
                # Recompute the content identity on BOTH sides (never from a
                # stored hash, so legacy disk rows migrate) -- if they
                # differ, the on-disk render is STALE: skip the whole
                # copy-forward for this row so the stale wav/timing is
                # dropped rather than glued onto the edited content. Owned
                # composition fields are never copied regardless (below).
                _id_mem = _row_identity(arr_name, row)
                _id_disk = _row_identity(arr_name, disk_row)
                if (_id_mem is not None and _id_disk is not None
                        and _id_mem != _id_disk):
                    continue
                # Copy forward keys present on disk but absent or
                # null in memory. Never overwrite a present in-mem
                # value with a disk value -- in-memory is fresher
                # for rows the Ledger class actually manages.
                #
                # Canonical composition/state fields are never restored from
                # disk. Only out-of-band durable render fields copy forward.
                for k, v in disk_row.items():
                    # ``shot_id`` is authored ownership for lines/clips, but
                    # renderer ownership for music: SceneSequencer derives it
                    # from the cue's resolved anchor.  Preserve that durable
                    # music receipt on an identity match without weakening the
                    # stale-row protection for any spoken or clip row.
                    if (
                        k in _MERGE_OWNED_ROW_FIELDS
                        and not (arr_name == "music" and k == "shot_id")
                    ):
                        continue
                    if k not in row or row.get(k) in (None, "", [], {}):
                        row[k] = v

        # EpisodeAssembler is the sole producer of ``mirrored_from=music``
        # timeline rows.  It persists them out-of-band through
        # ``save_ledger_safe`` while the singleton still owns only authored
        # dialogue.  A later singleton save must preserve those disk-only rows,
        # but must never generalize into resurrecting arbitrary disk content.
        disk_lines = on_disk.get("lines") or []
        disk_mirrors = [
            row for row in disk_lines
            if isinstance(row, dict) and row.get("mirrored_from") == "music"
        ] if isinstance(disk_lines, list) else []
        if disk_mirrors and not same_durable_run:
            log.warning(
                "[Ledger] disk-only music mirrors rejected: durable run "
                "identity mismatch (count=%d)", len(disk_mirrors),
            )
        elif disk_mirrors:
            mem_music_rows = in_mem.get("music") or []
            disk_music_rows = on_disk.get("music") or []
            if not isinstance(mem_music_rows, list):
                mem_music_rows = []
            if not isinstance(disk_music_rows, list):
                disk_music_rows = []

            def _unique_cue_map(rows: List[Any]) -> "tuple[Dict[str, Dict[str, Any]], set[str]]":
                mapped: Dict[str, Dict[str, Any]] = {}
                duplicates: "set[str]" = set()
                for candidate in rows:
                    if not isinstance(candidate, dict):
                        continue
                    cue_id = str(candidate.get("cue_id") or "").strip()
                    if not cue_id:
                        continue
                    if cue_id in mapped:
                        duplicates.add(cue_id)
                    else:
                        mapped[cue_id] = candidate
                return mapped, duplicates

            mem_music_by_id, mem_duplicate_cues = _unique_cue_map(mem_music_rows)
            disk_music_by_id, disk_duplicate_cues = _unique_cue_map(disk_music_rows)
            invalid_cues = mem_duplicate_cues | disk_duplicate_cues
            mem_lines = in_mem.get("lines") or []
            if not isinstance(mem_lines, list):
                mem_lines = []
            existing_line_ids = {
                str(row.get("line_id")) for row in mem_lines
                if isinstance(row, dict) and row.get("line_id")
            }
            preserved = 0
            rejected = 0
            for mirror in disk_mirrors:
                line_id = str(mirror.get("line_id") or "").strip()
                cue_id = str(mirror.get("music_cue_id") or "").strip()
                role = str(mirror.get("speaker_role") or "").strip()
                start_s = mirror.get("start_s")
                dur_s = mirror.get("dur_s")
                if (
                    not line_id
                    or not cue_id
                    or cue_id in invalid_cues
                    or role not in {"music_open", "music_inter", "music_close"}
                    or mirror.get("start_s_space") != "master_mix"
                    or not isinstance(start_s, (int, float))
                    or not isinstance(dur_s, (int, float))
                    or float(start_s) < 0.0
                    or float(dur_s) <= 0.0
                ):
                    rejected += 1
                    continue
                mem_cue = mem_music_by_id.get(cue_id)
                disk_cue = disk_music_by_id.get(cue_id)
                mem_hash = music_cue_spec_sha256(mem_cue or {})
                disk_hash = music_cue_spec_sha256(disk_cue or {})
                mem_stored = (mem_cue or {}).get("cue_spec_sha256")
                disk_stored = (disk_cue or {}).get("cue_spec_sha256")
                if (
                    mem_hash is None
                    or disk_hash is None
                    or mem_hash != disk_hash
                    or disk_stored != disk_hash
                    or (mem_stored not in (None, "") and mem_stored != mem_hash)
                ):
                    rejected += 1
                    continue
                if line_id in existing_line_ids:
                    # Idempotent: an existing row is already the singleton's
                    # copy of this producer receipt; never append twice.
                    continue
                mem_lines.append(dict(mirror))
                existing_line_ids.add(line_id)
                preserved += 1
            if preserved:
                mem_lines.sort(key=lambda row: (
                    float(row.get("start_s"))
                    if isinstance(row, dict)
                    and isinstance(row.get("start_s"), (int, float))
                    else 1e18
                ))
                in_mem["lines"] = mem_lines
            log.info(
                "[Ledger] disk-only assembler mirrors: preserved=%d "
                "rejected=%d existing=%d",
                preserved, rejected, len(disk_mirrors) - preserved - rejected,
            )
        return in_mem


def assemble_script_text_from_ledger(led_data: dict) -> str:
    """Rebuild the writer's slot-0 `script_text` string from the
    canonical ledger lines.

    Token format matches what `OTR_LedgerScriptWriter.run()` emitted per beat
    before that in-loop list was removed (2026-08-28); this function is now
    the only producer of the slot-0 transcript:
      - character beat:  ``[VOICE: NAME, traits] <text>``
      - announcer beat:  ``[VOICE: ANNOUNCER, traits] <text>``
      - music beat:      skipped (render contract, no transcript text;
        the legacy ``[SFX: ...]`` token was removed 2026-07-01,
        rip-sfx-broll)

    Used as the post-loop authoritative source for slot-0 in BOTH the
    writer (after the news-wiring overlay patches `led.data['lines']`
    in place) and the reviewer (after the 3-pass review may have
    rewritten line text). Pre-Tier-1, both callers shipped a stale
    string that did not reflect those mutations.

    `led_data` is the in-memory dict that `Ledger.data` points at.
    Reads `led_data['cast']` for char_id -> name lookup. Empty / falsy
    text rows are skipped. Stable across calls (no RNG, no time).
    Never raises.
    """
    if not isinstance(led_data, dict):
        return ""
    cast = led_data.get("cast", []) or []
    name_by_cid: dict[str, str] = {}
    for row in cast:
        if not isinstance(row, dict):
            continue
        cid = _safe_str(row.get("char_id"))
        name = _safe_str(row.get("name"))
        if cid and name:
            name_by_cid[cid] = name
    parts: list[str] = []
    for line in led_data.get("lines", []) or []:
        if not isinstance(line, dict):
            continue
        text = _safe_str(line.get("text")).strip()
        if not text:
            continue
        role = _safe_str(line.get("speaker_role"))
        traits = _safe_str(line.get("traits")).strip() or "neutral"
        if role == "character":
            cid = _safe_str(line.get("char_id"))
            name = name_by_cid.get(cid) or _safe_str(line.get("speaker"))
            if not name:
                # Last-resort fallback: render the char_id literally so
                # the downstream director / parser still sees a
                # speaker tag rather than silently dropping the line.
                name = cid or "UNKNOWN"
            parts.append(f"[VOICE: {name}, {traits}] {text}")
        elif role == "announcer":
            parts.append(f"[VOICE: ANNOUNCER, {traits}] {text}")
        elif role in {"music_open", "music_close", "music_inter"}:
            # Music rows are render contracts, never transcript rows.
            # By construction they carry text=="" (init stamps "" and
            # the composer loop writes "") so this branch is
            # unreachable for a fresh ledger; it exists to keep a
            # stale text-bearing music row OUT of the transcript
            # rather than emitting the retired [SFX:] token.
            continue
        else:
            # Unknown role: keep the text visible to downstream
            # consumers tagged by role so a debug pass can spot it.
            tag = role.upper() if role else "BEAT"
            parts.append(f"[{tag}: {text}]")
    return "\n\n".join(parts)


__all__ = [
    "Ledger",
    "get_ledger",
    "new_ledger",
    "assemble_script_text_from_ledger",
]
