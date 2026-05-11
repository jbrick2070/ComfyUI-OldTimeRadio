"""
nodes/production_ledger.py -- OTR Production Ledger (L2, write-only)

Single-source-of-truth JSON record of everything an episode produced:
cast, scenes, shots, beats, lines, sfx, music, and their positions on
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
  ScriptWriter DONE    -> cast + lines (text + word/char counts)
  LLMDirector DONE     -> cast (voice_presets) + scenes (env) + shots (visual prompts)
  SceneSequencer DONE  -> lines/sfx/music start_s + dur_s + beats (speaker turns)
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

import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, List, Optional

log = logging.getLogger("OTR.production_ledger")

_LEDGER_LOCK = threading.Lock()
_CURRENT: Optional["Ledger"] = None
_GIT_HEAD_CACHE: Optional[str] = None


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
    return os.path.join(
        os.path.expanduser("~"),
        "Documents", "ComfyUI", "output", "otr", "episodes", ep, "audio",
    )


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
        out = subprocess.check_output(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8", errors="ignore").strip()
        _GIT_HEAD_CACHE = out or "unknown"
    except Exception:  # noqa: BLE001 -- any failure means we just skip
        _GIT_HEAD_CACHE = "unknown"
    return _GIT_HEAD_CACHE


def _word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", text))


def _char_count(text: str) -> int:
    if not text:
        return 0
    return len(text)


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


def _derive_tts_model_from_voice_preset(voice_preset: str) -> Optional[str]:
    """Map a cast row's voice_preset to its TTS family.

    Bark presets follow the Suno convention `v2/en_speaker_*` (and
    legacy foreign-accent presets like `de_speaker_*` / `fr_speaker_*`
    if they're ever re-enabled). Kokoro presets follow the BBC/American
    convention `bm_*` / `bf_*` / `am_*` / `af_*`.

    Used as a back-compat shim in `Ledger.set_cast` when a caller
    passes a pre-tts_model cast row. New code MUST supply `tts_model`
    explicitly; this helper only catches in-flight rows that haven't
    been updated yet. Returns None for unknown prefixes.
    """
    if not voice_preset:
        return None
    s = voice_preset.strip()
    if s.startswith("v2/") and "speaker" in s:
        return "bark"
    if s.startswith(("bm_", "bf_", "am_", "af_")):
        return "kokoro"
    # Legacy Bark foreign-accent presets (currently disabled per
    # cast_pools.py commentary; kept here in case they're re-enabled).
    if any(s.startswith(p) for p in (
        "de_speaker_", "es_speaker_", "fr_speaker_", "hi_speaker_",
        "it_speaker_", "ja_speaker_", "ko_speaker_", "ru_speaker_",
        "pt_speaker_", "pl_speaker_",
    )):
        return "bark"
    return None


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
        SCHEMA_VERSION = "l3-2026-05-02"

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
            "sfx": [],
            "music": [],
            "clips": [],
            "final_audio_path": None,
            "final_video_path": None,
        }

    # -- identity ------------------------------------------------------

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
          2) Update in-memory state (episode_id, out_dir)
          3) Rename ledger file to canonical name
          4) Rename treatment + sidecar (<old>_*.txt -> <new>_*.txt)

        Steps 3 + 4 log warnings on individual file failures but do not
        raise; the dir-move success has already established the
        invariant that downstream nodes need.

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

        # Step 2: update in-memory state. Only past this point if dir
        # is in its final on-disk position (or was already there).
        self.episode_id = new_id
        self.data["episode_id"] = new_id
        self.out_dir = new_audio_dir

        # Step 3: rename the ledger file inside the (now-moved) audio dir
        # to the new canonical filename. Best-effort: warn on failure
        # but do not raise (the dir invariant is already satisfied; the
        # next save() will write to self.path which uses new_id).
        old_ledger_path = os.path.join(
            new_audio_dir, f"{_slugify(old, limit=120)}_ledger.json"
        )
        new_ledger_path = self.path
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
                log.warning(
                    "[Ledger] ledger file rename failed (%s -> %s): %s; "
                    "next save() will reconcile",
                    old_ledger_path, new_ledger_path, exc,
                )

        # Step 4: rename treatment + any other owned txt sidecars from
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

        log.info("[Ledger] renamed %s -> %s (dir_moved=%s)",
                 old, new_id, moved_dir)

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
            # Back-compat input shim: still accept the old key on the way IN
            # so any in-flight ledger or cached cast row from a prior session
            # normalizes cleanly. Output is always the new key.
            cdesc = (
                _safe_str(r.get("character_description"))
                or _safe_str(r.get("description"))
                or None
            )
            # Cast field added 2026-05-10: tts_model. Routing column
            # that says which TTS family the voice_preset belongs to
            # ("bark", "kokoro", future: "fish_speech", "cosyvoice", ...).
            # Lets downstream consumers route by reading the field
            # directly instead of pattern-matching the voice_preset
            # prefix.
            #
            # Back-compat input shim: if a caller doesn't supply
            # tts_model (e.g. an in-flight cast row from before the
            # field landed), derive it from the voice_preset prefix.
            # Bark presets are "v2/en_speaker_*"; Kokoro presets are
            # "bm_*", "bf_*", "am_*", "af_*". Anything else falls back
            # to None and the consumer must error-check.
            tts_model = _safe_str(r.get("tts_model")) or None
            voice_preset = _safe_str(r.get("voice_preset")) or None
            if tts_model is None and voice_preset:
                tts_model = _derive_tts_model_from_voice_preset(voice_preset)
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
            rows.append({
                "char_id":               _safe_str(r.get("char_id")),
                "name":                  _safe_str(r.get("name")),
                "character_description": cdesc,
                "gender":                _safe_str(r.get("gender")) or None,
                "tts_model":             tts_model,
                "voice_preset":          voice_preset,
                "voice_params":          voice_params,
                "line_count":            _safe_int(r.get("line_count")),
                "word_count":            _safe_int(r.get("word_count")),
            })
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

    def set_lines(self, line_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        """Set the lines[] section.

        ``beat_id`` and ``boundary`` are optional but recommended once
        the upstream pipeline (SceneSequencer or build_silent_test_episode)
        has populated the beat hierarchy. ``boundary`` is one of:
            - "shot_start"  first clip of a new shot (visual reset)
            - "beat_start"  first clip of a new speaker turn within a shot
            - "continue"    same shot, same speaker (Goal 3 daisy chain)
        Older callers that pre-date the beat hierarchy can omit both
        fields; downstream consumers must treat missing values as
        equivalent to "shot_start" for safety.
        """
        rows: List[Dict[str, Any]] = []
        for r in line_rows or []:
            text = _safe_str(r.get("text"))
            rows.append({
                "line_id":        _safe_str(r.get("line_id")),
                "shot_id":        _safe_str(r.get("shot_id")) or None,
                "beat_id":        _safe_str(r.get("beat_id")) or None,
                "char_id":        _safe_str(r.get("char_id")) or None,
                "text":           text,
                "traits":         _safe_str(r.get("traits")) or None,
                "boundary":       _safe_str(r.get("boundary")) or None,
                "char_count":     _char_count(text),
                "word_count":     _word_count(text),
                "bark_wav_path":  _safe_str(r.get("bark_wav_path")) or None,
                "start_s":        _safe_float(r.get("start_s")),
                "dur_s":          _safe_float(r.get("dur_s")),
            })
        self.data["lines"] = rows
        self._recompute_totals()
        return self

    def set_sfx(self, sfx_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        rows: List[Dict[str, Any]] = []
        for r in sfx_rows or []:
            rows.append({
                "cue_id":            _safe_str(r.get("cue_id")),
                "shot_id":           _safe_str(r.get("shot_id")) or None,
                "description":       _safe_str(r.get("description")),
                "generation_prompt": _safe_str(r.get("generation_prompt")) or None,
                "wav_path":          _safe_str(r.get("wav_path")) or None,
                "start_s":           _safe_float(r.get("start_s")),
                "dur_s":             _safe_float(r.get("dur_s")),
            })
        self.data["sfx"] = rows
        return self

    def set_music(self, music_rows: Iterable[Dict[str, Any]]) -> "Ledger":
        rows: List[Dict[str, Any]] = []
        for r in music_rows or []:
            rows.append({
                "cue_id":            _safe_str(r.get("cue_id")),
                "description":       _safe_str(r.get("description")) or None,
                "generation_prompt": _safe_str(r.get("generation_prompt")) or None,
                "wav_path":          _safe_str(r.get("wav_path")) or None,
                "start_s":           _safe_float(r.get("start_s")),
                "dur_s":             _safe_float(r.get("dur_s")),
            })
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

    def apply_sfx_timings(self, timing: Dict[str, Dict[str, float]]) -> "Ledger":
        for row in self.data["sfx"]:
            t = timing.get(row.get("cue_id"))
            if t:
                row["start_s"] = _safe_float(t.get("start_s"))
                row["dur_s"]   = _safe_float(t.get("dur_s"))
                if t.get("wav_path"):
                    row["wav_path"] = str(t["wav_path"])
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
            if cid in per_char_count:
                row["line_count"] = per_char_count[cid]
                row["word_count"] = per_char_words[cid]
        for row in self.data["scenes"]:
            sid = row.get("scene_id")
            if sid in per_scene_count:
                row["line_count"] = per_scene_count[sid]
                row["word_count"] = per_scene_words[sid]

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
            self._recompute_totals()
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
                    _Path(path), str(self.episode_id)
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
          schema_version, meta, audio_gates, transitions,
          radio_bookend_path, final_audio_path

        Per-row fields preserved (keyed by line_id / cue_id):
          For each lines[i] / clips[i] / sfx[i] / music[i]: copy
          forward any key present on disk that is missing or
          empty/null in the in-mem row.
        """
        try:
            if not os.path.exists(path):
                return in_mem
            with open(path, "r", encoding="utf-8") as f:
                on_disk = json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("[Ledger] BUG-108 merge: on-disk read failed: %s", exc)
            return in_mem

        # Top-level fields the Ledger class doesn't manage but must
        # not destroy. final_audio_path is on the kept-list because
        # SignalLostVideo overwrites it via set_final_paths -- that's
        # an explicit overwrite, not a merge concern.
        TOP_PRESERVE = (
            "schema_version", "meta", "audio_gates", "transitions",
            "radio_bookend_path",
        )
        for k in TOP_PRESERVE:
            if k in on_disk and (k not in in_mem or in_mem.get(k) in (None, "", [], {})):
                in_mem[k] = on_disk[k]

        # Per-row merge. Keyed by line_id (lines, clips) or cue_id
        # (sfx, music).
        ROW_KEYED = {
            "lines": "line_id",
            "clips": "line_id",
            "sfx": "cue_id",
            "music": "cue_id",
        }
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
                # Copy forward keys present on disk but absent or
                # null in memory. Never overwrite a present in-mem
                # value with a disk value -- in-memory is fresher
                # for rows the Ledger class actually manages.
                for k, v in disk_row.items():
                    if k not in row or row.get(k) in (None, "", [], {}):
                        row[k] = v
        return in_mem


__all__ = [
    "Ledger",
    "get_ledger",
    "new_ledger",
]
