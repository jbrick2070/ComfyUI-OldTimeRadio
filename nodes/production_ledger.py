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

def _default_out_dir() -> str:
    # 2026-04-26 PM BUG-LOCAL-067: moved from output/old_time_radio/ to
    # output/otr/audio/ so every OTR artifact (audio episodes, ledgers,
    # treatments, FLUX stills, HuMo videos, 1080p deliveries) nests under
    # the single output/otr/ super-folder. Legacy output/old_time_radio/
    # is left in place for already-rendered episodes; scripts fall back
    # to reading both locations.
    return os.path.join(
        os.path.expanduser("~"),
        "Documents", "ComfyUI", "output", "otr", "audio",
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def new_ledger(episode_id: Optional[str] = None,
               out_dir: Optional[str] = None) -> "Ledger":
    """Start a fresh ledger. Call at the beginning of a pipeline run.

    If episode_id is None, a placeholder `pending_<timestamp>` id is used;
    rename later via ledger.rename_episode(new_id) once the real filename
    is known (usually at SignalLostVideo stage).
    """
    global _CURRENT
    with _LEDGER_LOCK:
        ep = episode_id or ("pending_" + time.strftime("%Y%m%d_%H%M%S"))
        _CURRENT = Ledger(ep, out_dir or _default_out_dir())
        return _CURRENT


def get_ledger() -> "Ledger":
    """Return the current in-progress ledger, creating a placeholder if
    none exists. Safe to call from any stage."""
    global _CURRENT
    with _LEDGER_LOCK:
        if _CURRENT is None:
            _CURRENT = Ledger(
                "pending_" + time.strftime("%Y%m%d_%H%M%S"),
                _default_out_dir(),
            )
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
    #
    # Beat = single-speaker continuous turn within a shot. Hierarchy:
    #     Scene > Shot > Beat > Clip
    # See ROADMAP Goal 3 + docs/2026-04-25-humo-continuity-brief.md.
    SCHEMA_VERSION = "l2-2026-04-25"

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
        """Rename the episode id. The on-disk file path follows."""
        old = self.episode_id
        self.episode_id = new_id
        self.data["episode_id"] = new_id
        log.info("[Ledger] renamed %s -> %s", old, new_id)

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
            rows.append({
                "char_id":       _safe_str(r.get("char_id")),
                "name":          _safe_str(r.get("name")),
                "description":   _safe_str(r.get("description")) or None,
                "gender":        _safe_str(r.get("gender")) or None,
                "voice_preset":  _safe_str(r.get("voice_preset")) or None,
                "line_count":    _safe_int(r.get("line_count")),
                "word_count":    _safe_int(r.get("word_count")),
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
        failure. Never raises."""
        try:
            self._recompute_totals()
            path = self.path
            # Atomic write: temp file + replace
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
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


__all__ = [
    "Ledger",
    "get_ledger",
    "new_ledger",
]
