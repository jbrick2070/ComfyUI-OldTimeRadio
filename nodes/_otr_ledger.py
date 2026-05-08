"""
_otr_ledger.py  --  shared ledger I/O helpers for OTR nodes
============================================================

Every OTR node that writes diagnostic fields back to the per-episode
``*_ledger.json`` re-implements roughly the same load-mutate-save
dance (BUG-090 director_raw, BUG-094 line timings, BUG-095 music
wav_paths, BUG-096 line dur_s, BUG-098 cast dedup, BUG-102 clips
warmup_pad_ms, ...). This module consolidates the boilerplate so:

  * Schema version bumps land in exactly one place.
  * sha256-of-first-1KB gate hashing is consistent across nodes.
  * git-commit lookup is identical regardless of caller.
  * Failures degrade silently with a warning rather than crashing
    a 4-hour render -- a missing ledger field is never worth
    aborting the run.

All helpers are best-effort: any I/O exception is caught and logged
at WARNING level. Callers should NEVER let a ledger write-back
failure abort their main work.

Schema version
--------------
Current canonical: ``l3-2026-04-28`` (post-BUG-100/101/102 diagnostic
expansion). Previous: ``l2-2026-04-25``. The version string is
written to ``ledger["schema_version"]`` AND ``ledger["meta"]["schema_version"]``.
Consumers should accept either location for back-compat with l2 ledgers.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Optional

log = logging.getLogger("OTR")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CURRENT_SCHEMA_VERSION = "l3-2026-05-08"
"""Set on every ledger write performed via this module. Bump when
adding any field that downstream consumers must check for. Keep the
date suffix so the lineage is greppable.

Lineage:
  l1-2026-04-24 -- baseline (cast, scenes, shots, lines, sfx, music, clips)
  l2-2026-04-25 -- adds beats[] hierarchy
  l3-2026-04-28 -- diagnostic expansion (meta.phase_ms, audio_gates,
                   text_for_tts, bark_render_ms, warmup_pad_ms,
                   transitions, radio_bookend_path)
  l3-2026-05-02 -- ADDITIVE: meta.paths block resolved at write time so
                   downstream nodes can look up canonical episode dirs
                   without reconstructing them from episode_id (Phase E,
                   BUG-LOCAL-018). All readers must continue to use
                   meta.get(...) with default-None to stay back-compat
                   with l3-2026-04-28 ledgers.
  l3-2026-05-08 -- ADDITIVE: BUG-126 telemetry + Cast Contract pre-wiring
                   per round-robin synthesis (ChatGPT gpt-5.5 +
                   Gemini gemini-3.1-pro-preview-customtools, transcripts
                   at docs/2026-05-08-ledger-schema-additions__*.md).
                   New fields:
                     lines[].oom_recovery_count       (int, default 0)
                     lines[].render_method            (str, default "unknown")
                                                       -- one of: humo_native,
                                                          ltx_native,
                                                          static_radio_fill,
                                                          kokoro_announcer_only,
                                                          unknown
                     lines[].cast_contract_version    (str | None)
                     clips[].humo_oom_recovered       (bool, default False)
                     clips[].humo_oom_recovered_at_chunk (int | None)
                     cast[].voice_spec                (str | None,
                                                       canonical "engine:preset")
                     cast_contract_version            (top-level, str | None)
                     cast_contract                    (top-level, dict | None,
                                                       mirrors CastContract.to_dict)
                     meta.cuda_hard_reset_count       (int, default 0)
                     meta.soak_cap                    (dict)
                     meta.process_runs[]              (append-only list of
                                                       {pid, started_at})
                     meta.audit_verdict               (dict, set by
                                                       audit_otr_full_run.py)
                   Plus: save_ledger_safe is now ATOMIC via tempfile +
                   os.replace (BUG-LOCAL-127 catch from same round-robin).
                   Soft-deprecated (still written by older producers, do
                   not retire until l4): beats[], total_beats,
                   lines[].boundary, lines[].shot_id, lines[].beat_id,
                   shots[].png_path/start_s/dur_s, lines[].bark_wav_path
                   (use lines[].tts_wav_path going forward, both written
                   during the transition).

Reader pattern for all l3-2026-05-08 fields: ALWAYS use
``dict.get(field, default)`` with the documented default. Older
ledgers will return the default, newer ledgers return the stamped
value. No KeyError on either side.
"""

GATE_HASH_BYTES = 1024
"""How many leading bytes of a master audio waveform to feed into
sha256 for the per-gate integrity check. 1 KB is enough to detect
sample-rate / channel / encoding drift but cheap enough to compute
inside a render's hot path. The full audio is NOT hashed -- this is
a tripwire, not a verification suite."""


# ---------------------------------------------------------------------------
# Load / save (best-effort)
# ---------------------------------------------------------------------------

def load_ledger_safe(path: Path) -> Optional[dict]:
    """Read a ledger JSON file. Returns None on any error.

    Callers should treat None as "the ledger isn't usable; skip
    write-back". Errors are logged at WARNING.
    """
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        log.warning("[OTR_Ledger] ledger not found: %s", path)
        return None
    except json.JSONDecodeError as exc:
        log.warning("[OTR_Ledger] ledger JSON parse failed (%s): %s", path, exc)
        return None
    except Exception as exc:  # OSError, permissions, etc.
        log.warning("[OTR_Ledger] ledger load failed (%s): %s", path, exc)
        return None


def _build_meta_paths(ledger_path: Path, episode_id: str) -> dict:
    """Resolve the canonical absolute paths for this episode's per-dir
    workspace, derived from the on-disk ledger location.

    Layout (post 2026-05-02 EVENING reorg):
        output/otr/
          obs/
            <episode_id>.mp4
          episodes/
            <episode_id>/
              audio/<episode_id>_ledger.json
              stills/
              portraits/
              videos/
              composited/

    Detects layout by checking if the ledger sits two levels under a
    directory literally named "episodes". When detected, returns a fully
    populated paths block with absolute paths for every per-episode
    subdir + the OBS final mp4. When not detected (legacy flat layout
    or test fixtures with arbitrary parents), returns a minimal block
    with just episode_root and audio_dir resolved from the ledger
    location -- no fabricated subdirs.

    BUG-LOCAL-018 (Phase E, 2026-05-02): adding this block lets
    downstream nodes look up canonical paths via
    ``ledger["meta"]["paths"]["audio_dir"]`` instead of reconstructing
    them from episode_id (which would re-introduce the slug-mismatch
    risk Phase C closed). All readers must use ``.get(...)`` with
    default-None for back-compat with older ledgers that lack this
    block.
    """
    ledger_path = Path(ledger_path).resolve()
    audio_dir = ledger_path.parent
    ep_dir = audio_dir.parent

    paths: dict = {
        "ledger_path": str(ledger_path),
        "episode_root": str(ep_dir),
        "audio_dir": str(audio_dir),
    }

    # Per-episode workspace layout detection: ep_dir's parent must be
    # named "episodes" (case-insensitive on Windows but stored as-is).
    parent_of_ep = ep_dir.parent
    if parent_of_ep.name.lower() == "episodes":
        paths["stills_dir"] = str(ep_dir / "stills")
        paths["portraits_dir"] = str(ep_dir / "portraits")
        paths["videos_dir"] = str(ep_dir / "videos")
        paths["composited_dir"] = str(ep_dir / "composited")
        # OBS final lives at output/otr/obs/<ep>.mp4 -- a sibling of
        # episodes/, NOT a child. Only stamp it if the grandparent of
        # ep_dir has a sibling named "obs"; otherwise omit (we don't
        # fabricate a path that may not exist).
        otr_root = parent_of_ep.parent
        obs_root = otr_root / "obs"
        if otr_root.name.lower() == "otr":
            paths["obs_final"] = str(obs_root / f"{episode_id}.mp4")
            paths["obs_dir"] = str(obs_root)
        paths["layout"] = "per-episode-workspace"
    else:
        paths["layout"] = "legacy-flat"

    return paths


def save_ledger_safe(path: Path, ledger: dict) -> bool:
    """Write a ledger dict back to disk with schema_version + meta.paths
    stamped. ATOMIC via tempfile + ``os.replace`` (BUG-LOCAL-127).

    Always sets:
      - ``ledger["schema_version"]``
      - ``ledger["meta"]["schema_version"]``
      - ``ledger["meta"]["paths"]``  (BUG-LOCAL-018, Phase E)

    The ``meta.paths`` block is resolved fresh on every save from the
    actual on-disk ``path`` argument. This makes it self-correcting --
    if the per-episode dir was renamed mid-pipeline (Phase B
    rename_episode), the next ledger save reflects the new location
    without any caller having to update it explicitly.

    Atomic write rationale (BUG-LOCAL-127, caught by 2026-05-08
    round-robin): the ledger is the single source of truth for
    every downstream node + the multi-batch resume path. A hard
    CUDA crash during an in-progress write would leave a 0-byte /
    truncated JSON, destroying the production state for that
    episode. Mitigates by writing to ``<path>.tmp.<random>`` in
    the same directory then ``os.replace`` -- atomic on Windows
    NTFS and POSIX.

    Returns True on success, False on failure (always logs WARNING).
    Cleanup: any partial temp file is unlinked on failure.
    """
    try:
        ledger["schema_version"] = CURRENT_SCHEMA_VERSION
        meta = ledger.setdefault("meta", {})
        meta["schema_version"] = CURRENT_SCHEMA_VERSION
        episode_id = ledger.get("episode_id") or ""
        meta["paths"] = _build_meta_paths(Path(path), str(episode_id))
        payload = json.dumps(ledger, indent=2, ensure_ascii=False)
        target = Path(path)
        # tempfile.mkstemp in the SAME dir as the target so os.replace
        # is a same-filesystem atomic rename (cross-fs replace can
        # fail with EXDEV on POSIX; on NTFS it's still atomic but
        # slower). Prefix is hidden + descriptive for forensic value.
        tmp_dir = str(target.parent) if target.parent.exists() else None
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=".ledger.save.",
            suffix=".tmp.json",
            dir=tmp_dir,
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    # fsync can fail on some Windows file systems
                    # (network drives, filesystems without flush
                    # support). Non-fatal; the write itself flushed.
                    pass
            os.replace(tmp_name, target)
        except Exception:
            # Clean up the partial temp so we don't leave debris.
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return True
    except Exception as exc:
        log.warning("[OTR_Ledger] ledger save failed (%s): %s", path, exc)
        return False


# ---------------------------------------------------------------------------
# Auto-discover most recent ledger in the canonical audio dir
# ---------------------------------------------------------------------------

def in_flight_ledger_path() -> Optional[Path]:
    """Return the in-flight Ledger singleton's on-disk path.

    BUG-LOCAL-021 (Phase G, 2026-05-03): replaces ``find_most_recent_ledger``
    for in-flight write-back paths. The mtime walker can return a stale
    leftover ledger across queue boundaries (proven by the FLUX radio
    bookend stamping to a 6-day-old episode on the 2026-05-02 soak). The
    in-flight singleton's ``path`` property advances correctly through
    ``Ledger.rename_episode`` (Phase B), so it tracks the active episode
    by construction.

    ComfyUI sequential queue + ``LLMScriptWriter.IS_CHANGED = time.time()``
    prevent the singleton from going stale across queued runs. Falls back
    to ``find_most_recent_ledger`` for headless/standalone scenarios where
    no LLMScriptWriter has run in this process.

    Returns the path on success, None on miss. Never raises.
    """
    try:
        # Late import: production_ledger imports _otr_ledger at class-init
        # for the SCHEMA_VERSION; doing the reverse import at module load
        # would cycle. Late-binding inside the function is safe.
        try:
            from . import production_ledger as _PL  # type: ignore
        except ImportError:
            import production_ledger as _PL  # type: ignore
        led = _PL.get_ledger()
        p = Path(led.path)
        if p.exists():
            return p
        # Singleton's path doesn't exist on disk (rare: rename failed
        # silently, or singleton initialized but never .save()'d). Fall
        # through to legacy walker.
        log.debug(
            "[OTR_Ledger] in_flight singleton path %s not on disk; "
            "falling back to mtime walker", p,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] in_flight singleton lookup failed (%s); "
            "falling back to mtime walker", exc,
        )
    # Last-resort fallback: legacy mtime walker. Standalone/test paths
    # without an active singleton can still find a ledger this way.
    try:
        try:
            from . import _otr_paths as _P
        except ImportError:
            import _otr_paths as _P  # type: ignore
        return find_most_recent_ledger(
            [_P.otr_episodes_root(), _P.otr_legacy_audio_dir()]
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("[OTR_Ledger] fallback mtime walker failed: %s", exc)
        return None


def find_most_recent_ledger(audio_dirs: Iterable[Path]) -> Optional[Path]:
    """Search the supplied dirs for ``*_ledger.json`` and return the
    newest by mtime. Returns None if no candidates found.

    Walks each given dir at TWO levels:
      1. ``<dir>/*_ledger.json``                   (legacy flat layout)
      2. ``<dir>/<episode_id>/audio/*_ledger.json``  (per-episode workspace
         layout, post 2026-05-02 EVENING reorg -- see
         ``otr_episodes_root()`` in ``_otr_paths``)

    BUG-LOCAL-103 (2026-04-29 morning): we used to filter out
    ``pending_*_ledger.json`` here, which silently broke every
    audio-node ledger write. BatchBark / SceneSequencer /
    AudioEnhance / EpisodeAssembler all run while the ledger is
    still ``pending_<title>_ledger.json`` (LLMScriptWriter creates
    the pending file; SignalLostVideo renames it once the audio
    title is finalized). Filtering pending_* meant those four
    nodes' write-backs no-op'd because they couldn't find the
    in-flight ledger. Fix: include pending_* in the glob; mtime
    sort still prefers the newest, so a renamed canonical ledger
    naturally wins after rename.

    Use ``otr_episodes_root()`` (per-episode workspace) +
    ``otr_legacy_audio_dir()`` (pre-cutover legacy) from
    ``_otr_paths`` as the canonical search list.
    """
    candidates: list[Path] = []
    for d in audio_dirs:
        try:
            d = Path(d)
            if not d.exists():
                continue
            # Legacy flat layout: ledgers directly in the dir.
            candidates.extend(d.glob("*_ledger.json"))
            # Per-episode workspace layout (post 2026-05-02 EVENING):
            # ledgers under <dir>/<episode_id>/audio/*_ledger.json.
            candidates.extend(d.glob("*/audio/*_ledger.json"))
        except Exception as exc:
            log.warning("[OTR_Ledger] ledger glob failed in %s: %s", d, exc)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Per-line / per-clip patching
# ---------------------------------------------------------------------------

def patch_line_fields(
    ledger: dict,
    line_id: str,
    fields: dict,
) -> bool:
    """Merge ``fields`` into ``ledger["lines"][i]`` for the line whose
    ``line_id`` matches. Returns True if a row was updated, False if
    no matching line was found.
    """
    lines = ledger.get("lines") or []
    for ln in lines:
        if ln.get("line_id") == line_id:
            ln.update(fields)
            return True
    return False


def patch_clip_fields(
    ledger: dict,
    line_id: str,
    fields: dict,
) -> bool:
    """Merge ``fields`` into ``ledger["clips"][i]`` for the clip whose
    ``line_id`` matches. Returns True if a row was updated, False if
    no matching clip was found.
    """
    clips = ledger.get("clips") or []
    for cl in clips:
        if cl.get("line_id") == line_id:
            cl.update(fields)
            return True
    return False


# ---------------------------------------------------------------------------
# Per-line audio render metadata (BUG-LOCAL-030 audit-completion, 2026-05-03 EVENING)
# ---------------------------------------------------------------------------
#
# Adds explicit forensic provenance for every per-line audio render
# (Bark / Kokoro / MusicGen / AudioGen) to the ledger. Closes the audit
# gap surfaced by the artifacts-grid review: previously only Bark
# stamped per-line metadata (text_for_tts + bark_render_ms); the other
# three TTS/audio engines either stamped nothing per-line (Kokoro) or
# stamped only the render-result wav path (MusicGen, AudioGen).
#
# Five new per-line fields any audio node can stamp:
#   - tts_engine       : "bark" | "kokoro" | "musicgen" | "audiogen"
#   - voice_preset     : the voice/instrument slot used
#   - render_ms        : wall-clock generation time
#   - generated_dur_s  : actual generated audio duration (vs scheduled dur_s)
#   - audio_sample_hash: SHA8 of leading audio bytes — forensic byte-identity
#
# audio_sample_hash is the killer field: lets a future C7 audit confirm
# "same input → same audio bytes" without keeping the bytes themselves.

def compute_audio_sample_hash(
    waveform_bytes_or_array,
    n_bytes: int = 1024,
) -> str:
    """Compute an 8-char SHA256 hex hash of the leading ``n_bytes`` of
    audio bytes. Tripwires any change to the audio payload, sample rate
    handling, or padding behavior.

    Accepts:
      - ``bytes`` directly (already-extracted leading slice)
      - numpy ndarray (calls ``.tobytes()`` and slices)
      - torch.Tensor (calls ``.detach().cpu().numpy().tobytes()`` and slices)

    Returns a lowercase 8-char hex string. Returns ``""`` on any
    extraction failure (best-effort; never raises).
    """
    import hashlib
    try:
        if isinstance(waveform_bytes_or_array, (bytes, bytearray)):
            buf = bytes(waveform_bytes_or_array)
        else:
            # numpy or torch — duck-type via .tobytes() / .detach().cpu().numpy()
            arr = waveform_bytes_or_array
            if hasattr(arr, "detach"):
                arr = arr.detach().cpu().numpy()
            if hasattr(arr, "tobytes"):
                buf = arr.tobytes()
            else:
                return ""
        head = buf[:int(n_bytes)]
        return hashlib.sha256(head).hexdigest()[:8]
    except Exception:  # noqa: BLE001
        return ""


def stamp_per_line_audio_meta(
    ledger: dict,
    line_id: str,
    *,
    tts_engine: str,
    voice_preset: str = "",
    render_ms: int = 0,
    generated_dur_s: float = 0.0,
    audio_sample_hash: str = "",
) -> bool:
    """Stamp per-line audio render metadata. Wraps ``patch_line_fields``.

    Skips empty/zero values so callers can pass a partial bundle without
    overwriting fields they didn't compute (e.g. MusicGen has no
    voice_preset; just leave it ``""`` and that field stays unset).

    Returns True if a row was updated, False if no matching line_id.
    Never raises.
    """
    fields: dict = {"tts_engine": str(tts_engine)}
    if voice_preset:
        fields["voice_preset"] = str(voice_preset)
    if int(render_ms) > 0:
        fields["render_ms"] = int(render_ms)
    if float(generated_dur_s) > 0:
        fields["generated_dur_s"] = float(generated_dur_s)
    if audio_sample_hash:
        fields["audio_sample_hash"] = str(audio_sample_hash)
    try:
        return patch_line_fields(ledger, line_id, fields)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# Audio gate integrity hashing (CLAUDE.md C7)
# ---------------------------------------------------------------------------

def audio_gate_record(
    gate_name: str,
    waveform_bytes: bytes,
    dur_s: float,
    sample_count: int,
    sample_rate: int,
) -> dict:
    """Build a single ``audio_gates[]`` entry. Caller passes the
    already-extracted leading bytes (typically waveform.cpu().numpy()
    .tobytes()[:GATE_HASH_BYTES]) so this helper does no torch I/O.
    """
    h = hashlib.sha256(waveform_bytes).hexdigest()[:32]
    return {
        "gate": str(gate_name),
        "dur_s": float(dur_s),
        "sample_count": int(sample_count),
        "sample_rate": int(sample_rate),
        "sha256_first_kb": h,
    }


def append_audio_gate(
    ledger: dict,
    gate_record: dict,
) -> None:
    """Append a gate record to ``ledger["audio_gates"]``, creating
    the array if absent. No-op-safe (never raises).
    """
    try:
        gates = ledger.setdefault("audio_gates", [])
        gates.append(gate_record)
    except Exception as exc:
        log.warning("[OTR_Ledger] append_audio_gate failed: %s", exc)


# ---------------------------------------------------------------------------
# Meta block (git commit, phase timing, schema version)
# ---------------------------------------------------------------------------

_GIT_COMMIT_CACHE: Optional[str] = None
"""Resolve once per process. The OTR repo's HEAD doesn't change mid-run;
re-shelling out to git for every ledger save wastes a few ms."""


def lookup_git_commit(repo_root: Path) -> Optional[str]:
    """Return the short SHA of HEAD for the repo containing this
    module. Caches the result for the lifetime of the process. Returns
    None if git lookup fails (uninstalled, not a repo, etc.).
    """
    global _GIT_COMMIT_CACHE
    if _GIT_COMMIT_CACHE is not None:
        return _GIT_COMMIT_CACHE
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(repo_root)),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0:
            _GIT_COMMIT_CACHE = out.stdout.strip() or None
            return _GIT_COMMIT_CACHE
    except Exception as exc:
        log.warning("[OTR_Ledger] git rev-parse failed: %s", exc)
    return None


def set_meta(ledger: dict, key: str, value: Any) -> None:
    """Set ``ledger["meta"][key] = value``, creating ``meta`` if
    absent. No-op-safe.
    """
    try:
        meta = ledger.setdefault("meta", {})
        meta[key] = value
    except Exception as exc:
        log.warning("[OTR_Ledger] set_meta failed (%s): %s", key, exc)


def record_phase_ms(ledger: dict, phase_name: str, ms: int) -> None:
    """Record a node's wall-clock duration into
    ``ledger["meta"]["phase_ms"][phase_name]``. Multiple writes for
    the same phase overwrite (last write wins). Used to surface the
    bottleneck-phase per episode.
    """
    try:
        meta = ledger.setdefault("meta", {})
        phase_ms = meta.setdefault("phase_ms", {})
        phase_ms[str(phase_name)] = int(ms)
    except Exception as exc:
        log.warning("[OTR_Ledger] record_phase_ms failed (%s): %s", phase_name, exc)


# ---------------------------------------------------------------------------
# Transitions (consecutive line boundaries with crossfade)
# ---------------------------------------------------------------------------

def append_transition(
    ledger: dict,
    from_line_id: str,
    to_line_id: str,
    crossfade_ms: int,
    boundary_s: float,
) -> None:
    """Append a transition record to ``ledger["transitions"]``. Used
    by SceneSequencer / EpisodeAssembler to capture the exact
    crossfade overlap between consecutive lines so post-mortem can
    diagnose audio-tail-eating-next-clip-start regressions.
    """
    try:
        transitions = ledger.setdefault("transitions", [])
        transitions.append({
            "from_line_id": str(from_line_id) if from_line_id else None,
            "to_line_id": str(to_line_id) if to_line_id else None,
            "crossfade_ms": int(crossfade_ms),
            "boundary_s": float(boundary_s),
        })
    except Exception as exc:
        log.warning("[OTR_Ledger] append_transition failed: %s", exc)


# ---------------------------------------------------------------------------
# l3-2026-05-08: BUG-126 telemetry helpers
# ---------------------------------------------------------------------------
#
# All helpers below are best-effort and never raise. Callers (HuMo
# loop, audit script, etc.) MUST be able to call these from
# recovery paths without escalating the fault.


VALID_RENDER_METHODS = (
    "humo_native",
    "ltx_native",
    "static_radio_fill",
    "kokoro_announcer_only",
    "unknown",
)


def _find_line(ledger: dict, line_id: str) -> Optional[dict]:
    """Locate a line dict by ``line_id`` inside ``ledger["lines"]``.

    Returns the dict (mutating-friendly reference) or None. Used
    internally by the per-line stamp helpers so callers don't have
    to thread the index.
    """
    if not isinstance(ledger, dict) or not line_id:
        return None
    lines = ledger.get("lines") or []
    for line in lines:
        if isinstance(line, dict) and str(line.get("line_id") or "") == str(line_id):
            return line
    return None


def bump_line_oom_recovery_count(ledger: dict, line_id: str) -> int:
    """Increment ``lines[i].oom_recovery_count`` for the named line.

    Returns the new count, or 0 on miss. Caller logs at WARNING if
    the count is climbing -- BUG-126 telemetry; high count means
    the cleanup chain isn't actually relieving allocator pressure.
    """
    try:
        line = _find_line(ledger, line_id)
        if line is None:
            return 0
        cur = int(line.get("oom_recovery_count") or 0)
        new = cur + 1
        line["oom_recovery_count"] = new
        return new
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] bump_line_oom_recovery_count(%s) failed: %s",
            line_id, exc,
        )
        return 0


def stamp_line_render_method(ledger: dict, line_id: str, method: str) -> None:
    """Stamp ``lines[i].render_method``. Warns (does NOT raise) if
    ``method`` is not in ``VALID_RENDER_METHODS``; unknown values
    still pass through so a future producer adding a new method
    (e.g. ``cosyvoice_native``) doesn't crash older readers.
    """
    try:
        if method not in VALID_RENDER_METHODS:
            log.warning(
                "[OTR_Ledger] stamp_line_render_method(%s, %r): "
                "value not in VALID_RENDER_METHODS=%s; stamping anyway",
                line_id, method, VALID_RENDER_METHODS,
            )
        line = _find_line(ledger, line_id)
        if line is None:
            return
        line["render_method"] = str(method)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_line_render_method(%s) failed: %s",
            line_id, exc,
        )


def stamp_clip_humo_oom_recovered(
    clip_record: dict,
    recovered_at_chunk: Optional[int] = None,
) -> None:
    """Mark a clip record as having survived a caught HuMo OOM.

    Mutates ``clip_record`` in place (caller appends it to
    ``ledger["clips"]`` afterwards). Idempotent -- calling twice
    keeps the same flag, only updates ``humo_oom_recovered_at_chunk``
    if previously None.
    """
    try:
        if not isinstance(clip_record, dict):
            return
        clip_record["humo_oom_recovered"] = True
        if recovered_at_chunk is not None and clip_record.get(
            "humo_oom_recovered_at_chunk"
        ) is None:
            clip_record["humo_oom_recovered_at_chunk"] = int(recovered_at_chunk)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_clip_humo_oom_recovered failed: %s", exc,
        )


def bump_meta_cuda_hard_reset_count(ledger: dict) -> int:
    """Increment ``meta.cuda_hard_reset_count``. Direct telemetry for
    whether the BUG-LOCAL-126 cleanup chain has fired. Returns the
    new count, or 0 on miss.
    """
    try:
        meta = ledger.setdefault("meta", {})
        cur = int(meta.get("cuda_hard_reset_count") or 0)
        new = cur + 1
        meta["cuda_hard_reset_count"] = new
        return new
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] bump_meta_cuda_hard_reset_count failed: %s", exc,
        )
        return 0


def stamp_meta_soak_cap(
    ledger: dict,
    cap: int,
    lines_completed_when_hit: int,
    hit: bool = True,
) -> None:
    """Stamp ``meta.soak_cap`` on the HumoSoakCapReached path.

    Schema: ``{cap: int, lines_completed_when_hit: int, hit: bool}``.
    Default for absent field per round-robin: ``{hit: false,
    cap: null, lines_completed_when_hit: null}`` so a reader can
    distinguish "no cap configured" from "cap configured but not
    yet hit".
    """
    try:
        meta = ledger.setdefault("meta", {})
        meta["soak_cap"] = {
            "cap": int(cap) if cap is not None else None,
            "lines_completed_when_hit": (
                int(lines_completed_when_hit)
                if lines_completed_when_hit is not None
                else None
            ),
            "hit": bool(hit),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_meta_soak_cap failed: %s", exc,
        )


def append_meta_process_run(
    ledger: dict,
    pid: Optional[int] = None,
    started_at: Optional[str] = None,
) -> None:
    """Append a process-run entry to ``meta.process_runs[]``.

    Round-robin Element 10 catch (Gemini): a scalar ``meta.process_id``
    would be overwritten on every resume, destroying the chain. The
    append-only list preserves the multi-batch sequence so any single
    ledger proves the order of restarts.

    Defaults: ``pid`` from ``os.getpid()``, ``started_at`` from
    ``datetime.utcnow().isoformat()``. Caller can pass values
    explicitly when stamping a known cross-process state.
    """
    try:
        if pid is None:
            pid = os.getpid()
        if started_at is None:
            started_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        meta = ledger.setdefault("meta", {})
        runs = meta.setdefault("process_runs", [])
        # Avoid duplicate stamps when a node calls this twice in the
        # same run; latest entry wins on the timestamp tiebreak.
        if runs and runs[-1].get("pid") == int(pid):
            runs[-1]["started_at"] = str(started_at)
            return
        runs.append({"pid": int(pid), "started_at": str(started_at)})
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] append_meta_process_run failed: %s", exc,
        )


def stamp_meta_audit_verdict(
    ledger: dict,
    pass_: bool,
    fail_patterns_hit: Optional[list[str]] = None,
    ts: Optional[str] = None,
) -> None:
    """Stamp ``meta.audit_verdict`` after ``audit_otr_full_run.py``
    finishes. Schema: ``{pass: bool, ts: str, fail_patterns_hit: list[str]}``.

    Writers should run this LAST, after every other ledger field
    has been stamped, so the audit reflects the final state of the
    run. Re-running the audit on a previously-audited ledger
    overwrites this block (idempotent).
    """
    try:
        if ts is None:
            ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        meta = ledger.setdefault("meta", {})
        meta["audit_verdict"] = {
            "pass": bool(pass_),
            "ts": str(ts),
            "fail_patterns_hit": list(fail_patterns_hit or []),
        }
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_meta_audit_verdict failed: %s", exc,
        )


# ---------------------------------------------------------------------------
# l3-2026-05-08: Cast Contract pre-wiring helpers (Phase 0+ §1+§2+§3)
# ---------------------------------------------------------------------------
#
# These helpers EXIST now so the ledger schema is aligned with the
# cast contract module. The orchestrator hooks at L6423/L640/L920
# in story_orchestrator.py call them once those wires land.
#
# IMPORTANT (round-robin C7 caution): the cast_contract version sha
# is computed by `nodes/_otr_cast_contract.CastContract.stamp_version`
# from the canonical roster ONLY. It MUST NOT include any of the
# l3-2026-05-08 telemetry fields (process_runs, audit_verdict,
# cuda_hard_reset_count, soak_cap, etc.). Otherwise content addressing
# breaks across otherwise-identical resumes.


def stamp_cast_contract_version(ledger: dict, version: Optional[str]) -> None:
    """Stamp top-level ``cast_contract_version``. Pass ``None`` to
    keep the field absent (back-compat with l3-2026-05-02 readers).
    """
    try:
        if version is None:
            return
        ledger["cast_contract_version"] = str(version)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_cast_contract_version failed: %s", exc,
        )


def stamp_cast_contract(ledger: dict, contract_dict: Optional[dict]) -> None:
    """Stamp top-level ``cast_contract`` (the locked-roster JSON dump
    from ``CastContract.to_dict``). Mirrors the contract's exact
    shape: ``{version: "sha:...", characters: [...]}``.
    """
    try:
        if contract_dict is None:
            return
        if not isinstance(contract_dict, dict):
            log.warning(
                "[OTR_Ledger] stamp_cast_contract: contract_dict not a dict (%r)",
                type(contract_dict).__name__,
            )
            return
        ledger["cast_contract"] = dict(contract_dict)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_cast_contract failed: %s", exc,
        )


def stamp_line_cast_contract_version(
    ledger: dict,
    line_id: str,
    version: Optional[str],
) -> None:
    """Stamp ``lines[i].cast_contract_version`` for the named line.
    Lets ``production_ledger`` reject mixed-version merges in O(1).
    Pass ``None`` to keep absent.
    """
    try:
        if version is None:
            return
        line = _find_line(ledger, line_id)
        if line is None:
            return
        line["cast_contract_version"] = str(version)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_line_cast_contract_version(%s) failed: %s",
            line_id, exc,
        )


def stamp_cast_voice_spec(
    ledger: dict,
    char_id: str,
    voice_spec: Optional[str],
) -> None:
    """Stamp ``cast[i].voice_spec`` (canonical ``"engine:preset"``)
    for the named character. Adds the new field WITHOUT removing
    the legacy ``cast[i].voice_preset`` -- both written during the
    cast-contract transition window per round-robin synthesis.
    """
    try:
        if voice_spec is None:
            return
        cast = ledger.get("cast") or []
        for member in cast:
            if not isinstance(member, dict):
                continue
            if str(member.get("char_id") or "") == str(char_id):
                member["voice_spec"] = str(voice_spec)
                return
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "[OTR_Ledger] stamp_cast_voice_spec(%s) failed: %s", char_id, exc,
        )


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "GATE_HASH_BYTES",
    "VALID_RENDER_METHODS",
    "load_ledger_safe",
    "save_ledger_safe",
    "find_most_recent_ledger",
    "in_flight_ledger_path",
    "patch_line_fields",
    "patch_clip_fields",
    "audio_gate_record",
    "append_audio_gate",
    "lookup_git_commit",
    "set_meta",
    "record_phase_ms",
    "append_transition",
    # l3-2026-05-08 BUG-126 telemetry
    "bump_line_oom_recovery_count",
    "stamp_line_render_method",
    "stamp_clip_humo_oom_recovered",
    "bump_meta_cuda_hard_reset_count",
    "stamp_meta_soak_cap",
    "append_meta_process_run",
    "stamp_meta_audit_verdict",
    # l3-2026-05-08 Cast Contract pre-wiring
    "stamp_cast_contract_version",
    "stamp_cast_contract",
    "stamp_line_cast_contract_version",
    "stamp_cast_voice_spec",
]
