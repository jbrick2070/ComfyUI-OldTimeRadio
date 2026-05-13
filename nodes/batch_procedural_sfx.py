"""
Batch Procedural SFX - lightweight, zero-VRAM Foley for "Signal Lost".
=====================================================================

Designed for the Obsidian Edition (4GB VRAM). Mirroring the BatchAudioGen
logic, this node parses the script and generates SFX cues, but uses the
existing procedural synthesis engine (math-based) instead of a generative
model.

v1.5 AudioGen Integration - Jeffrey Brick
v2 ledger consumer (2026-05-09): Reads the v2 ledger from the
script_json wire input, walks sfx lines, persists per-cue wavs to
``<episode>/audio/sfx/proc_<sfx_type>_<line_id>_<perm>.wav`` (best-effort,
where ``<perm>`` is an 8-char SHA-256 over (dur_s, type, line_id) per S12.1),
and stamps ``sfx_wav_path`` + ``sfx_engine="procedural"`` + ``sfx_type``
+ ``dur_s`` per line_id on ``ledger.lines[]``. No cache layer
(procedural is cheap + deterministic). Disk-write failure falls
through to ``sfx_wav_path=None`` on the row; AUDIO batch return is
unaffected.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Optional

import numpy as np
import torch

from ._otr_sfx_lib import SFX_GENERATORS
from ._otr_ledger_freeze import SFX_DUR_MIN_S, SFX_DUR_MAX_S

log = logging.getLogger("OTR")

SAMPLE_RATE = 48000 # Procedural default

PROC_SFX_SUBDIR = "sfx"
"""Per-episode SFX output dir name. Lives alongside the ledger at
``<episode>/audio/sfx/`` -- sibling of AudioGen's cache files (which
land in the same episode/audio/ dir under canonical names) and of the
ledger JSON itself."""


def _safe_filename_segment(s: str, fallback: str = "unknown") -> str:
    """Sanitize a string for use as a filename segment: alphanumeric +
    underscore only. Mirrors AudioGen's ``re.sub(r'[^a-zA-Z0-9]', '_', ...)``
    canonicalization idiom (see ``batch_audiogen_generator._cache_prefix``)
    so the two paths feel symmetric in the on-disk inventory."""
    out = re.sub(r"[^a-zA-Z0-9]+", "_", str(s)).strip("_")
    return out or fallback


def _resolve_proc_sfx_dir() -> Optional[str]:
    """Resolve ``<episode>/audio/sfx/`` via the in-flight ledger path.

    Returns the absolute dir (created on demand) or None if the
    in-flight ledger isn't on disk yet (test fixtures, headless runs
    where SignalLostVideo never finalized the episode_id, etc.). The
    caller treats None as "skip disk write, continue rendering."
    """
    try:
        from . import _otr_ledger as _OTRL
        led_path = _OTRL.in_flight_ledger_path()
        if led_path is None:
            return None
        sfx_dir = os.path.join(str(led_path.parent), PROC_SFX_SUBDIR)
        os.makedirs(sfx_dir, exist_ok=True)
        return sfx_dir
    except Exception as exc:
        log.warning(
            "[BatchProceduralSFX] sfx output dir resolution failed: %s", exc,
        )
        return None


def _save_proc_sfx_wav(path: str, audio_np: np.ndarray, sample_rate: int) -> bool:
    """Atomic best-effort wav write. Returns True on success, False
    otherwise. Mirrors AudioGen's ``_save_wav`` pattern (write to
    ``<path>.tmp`` then ``os.replace``) so a crash mid-write doesn't
    leave a corrupted cache hit on a subsequent run."""
    tmp = path + ".tmp"
    try:
        import soundfile as sf
        sf.write(tmp, audio_np, sample_rate, subtype="FLOAT", format="WAV")
        os.replace(tmp, path)
        return True
    except Exception as exc:
        log.warning(
            "[BatchProceduralSFX] wav write failed (%s): %s", path, exc,
        )
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except Exception:
            pass
        return False


class BatchProceduralSFX:
    """Generates a batch of SFX cues from a script using procedural synthesis."""

    CATEGORY = "OldTimeRadio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("sfx_audio_clips", "batch_log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {"multiline": True, "default": "[]"}),
            },
            "optional": {
                "default_duration": ("FLOAT", {
                    "default": 2.0,
                    "min": SFX_DUR_MIN_S,   # G7 lower bound -- imported, not magic
                    "max": SFX_DUR_MAX_S,   # G7 upper bound -- imported, not magic
                    "step": 0.1,
                }),
                "volume_db": ("FLOAT", {"default": 0.0, "min": -30.0, "max": 6.0, "step": 1.0}),
            }
        }

    def generate(self, script_json, default_duration=2.0, volume_db=0.0):
        batch_log = ["=== Batch Procedural SFX (Obsidian) ==="]

        # Read-side: parse the wire input as a v2 ledger dict.
        # load_ledger raises ValueError on the legacy parser-list shape;
        # ProcSFX is in the loud-fail group (Pattern 1) -- bad wiring
        # halts the run early. The legacy Director production_plan_json
        # secondary input was deleted in voice-path-cleanbreak 2026-05-12 --
        # ProcSFX never read its content (dead-weight socket).
        from . import _otr_ledger_consumers as _OTRLC
        led = _OTRLC.load_ledger(script_json)

        # 1. Walk ledger sfx lines (Pattern 2). Cue text = line["text"]
        # directly -- no regex on inline [SFX: ...] markers (the legacy
        # parser-list emitted those embedded in dialogue content; the
        # v2 ledger has dedicated sfx lines with text = cue). Lowercase
        # and trim is preserved so the existing fuzzy matcher behaves
        # identically. line_id is carried through render_results so
        # the write-back below stamps by line_id (Pattern 4).
        sfx_items: list[dict] = []
        for line in _OTRLC.iter_lines(led, roles={"sfx"}):
            cue = (line.get("text") or "").strip().lower()
            if not cue:
                continue
            sfx_items.append({
                "line_id": line.get("line_id"),
                "tag":     cue,
                # Voice-path-cleanbreak Sprint 3 (2026-05-12): per-cue
                # dur_s from the writer's outline. None when the outline
                # doesn't emit a per-cue override; falls back to
                # default_duration in the generator call below. G7
                # invariant in the FreezeCascade has already validated
                # bounds [0.25, 12.0].
                "dur_s":   line.get("dur_s"),
            })

        if not sfx_items:
            batch_log.append("No SFX cues found in ledger. Returning silence.")
            return ({"waveform": torch.zeros(1, 1, 10), "sample_rate": SAMPLE_RATE}, "\n".join(batch_log))

        batch_log.append(f"Found {len(sfx_items)} SFX cues.")

        # 2. Resolve per-episode sfx output dir (best-effort; None on
        # headless / test paths -> skip disk write, sfx_wav_path stays
        # None on the ledger row).
        sfx_dir = _resolve_proc_sfx_dir()

        # 3. Match tags to available procedural generators
        # We use fuzzy matching to try to find a generator like 'door_knock' if tag is 'knock'
        available_types = list(SFX_GENERATORS.keys())

        final_clips = []
        # Track render results for the line_id-keyed write-back below.
        render_results: list[dict] = []
        for i, sfx_item in enumerate(sfx_items):
            tag = sfx_item["tag"]
            line_id = sfx_item.get("line_id")
            chosen_type = "radio_tuning" # Default fallback

            # Simple keyword matching
            for t in available_types:
                if t in tag or tag in t:
                    chosen_type = t
                    break

            # Additional semantic aliases
            if "knock" in tag or "door" in tag: chosen_type = "door_knock"
            elif "beep" in tag or "computer" in tag or "digital" in tag: chosen_type = "sci_fi_beep"
            elif "static" in tag or "noise" in tag: chosen_type = "white_noise"
            elif "boom" in tag or "thud" in tag: chosen_type = "explosion"
            elif "theremin" in tag or "eerie" in tag: chosen_type = "theremin"

            # Voice-path-cleanbreak Sprint 10.1 (S10.1): per-cue dur_s
            # takes precedence over default_duration. Defensive clamp
            # imports G7 constants -- no magic numbers in the consumer.
            # If G7's window changes, this clamp moves with it.
            _cue_dur_s = sfx_item.get("dur_s")
            if isinstance(_cue_dur_s, (int, float)) and _cue_dur_s > 0:
                cue_duration = max(SFX_DUR_MIN_S, min(SFX_DUR_MAX_S, float(_cue_dur_s)))
            else:
                cue_duration = float(default_duration)

            batch_log.append(
                f"  [{i}] Tag: '{tag[:20]}' -> Procedural Model: "
                f"'{chosen_type}' ({cue_duration:.2f}s)"
            )

            generator = SFX_GENERATORS[chosen_type]
            audio_np = generator(cue_duration, SAMPLE_RATE)

            # Apply volume
            if volume_db != 0.0:
                gain = 10.0 ** (volume_db / 20.0)
                audio_np *= gain

            # Peak normalize
            peak = np.abs(audio_np).max()
            if peak > 0.95:
                audio_np *= (0.95 / peak)

            # 3a. Persist per-cue wav to <episode>/audio/sfx/ (best-effort).
            # Filename: proc_<sfx_type>_<line_id>_<perm>.wav, all
            # segments sanitized. <perm> is an 8-char content-addressed
            # SHA-256 over (cue_duration, chosen_type, line_id) so
            # iteration on scene timing doesn't overwrite the previous
            # render -- the same line_id at two different durations
            # produces two distinct on-disk wavs (S12.1 / F-6 fix).
            # Procedural wavs are deterministic so collisions within
            # fixed inputs are impossible. Disk usage grows with
            # iteration count but procedural wavs are kB-scale.
            # On any disk error the function returns False and we
            # stamp sfx_wav_path=None on the ledger row (the AUDIO
            # batch return still ships).
            wav_path: Optional[str] = None
            if sfx_dir is not None and line_id:
                perm = hashlib.sha256(
                    f"{cue_duration:.3f}|{chosen_type}|{line_id}".encode("utf-8")
                ).hexdigest()[:8]
                fname = (
                    f"proc_{_safe_filename_segment(chosen_type)}_"
                    f"{_safe_filename_segment(line_id)}_{perm}.wav"
                )
                fpath = os.path.join(sfx_dir, fname)
                if _save_proc_sfx_wav(fpath, audio_np, SAMPLE_RATE):
                    wav_path = fpath
                    batch_log.append(f"    [{i}] wrote {fname}")
                else:
                    batch_log.append(f"    [{i}] disk write failed; sfx_wav_path=None")

            final_clips.append(torch.from_numpy(audio_np).float().unsqueeze(0).unsqueeze(0))
            render_results.append({
                "line_id":     line_id,
                "chosen_type": chosen_type,
                "dur_s":       float(audio_np.shape[-1]) / float(SAMPLE_RATE),
                "wav_path":    wav_path,
            })

        # 4. Build batched AUDIO output
        max_samples = max(clip.shape[2] for clip in final_clips)
        batched_waveform = torch.zeros(len(final_clips), 1, max_samples)

        for i, clip in enumerate(final_clips):
            samples = clip.shape[2]
            batched_waveform[i, 0, :samples] = clip[0, 0, :samples]

        # 5. Per-line ledger write-back (Pattern 4): patch by line_id
        # on ledger.lines[]. Fields per architect spec: sfx_wav_path
        # (may be None on disk-write failure), sfx_engine="procedural",
        # sfx_type, dur_s. ProcSFX never had a legacy ledger.sfx[]
        # write-back; this is purely the new lines[]-side stamping.
        # Single save_ledger_safe at the end. Best-effort -- any I/O
        # failure logs into batch_log but never raises.
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
            ledger_path = _OTRL.in_flight_ledger_path()
            if ledger_path is not None:
                led_disk = _OTRL.load_ledger_safe(ledger_path)
                if led_disk is None:
                    log.warning(
                        "[BatchProceduralSFX] in-flight ledger load failed at "
                        "%s; skipping ledger write-back",
                        ledger_path,
                    )
                else:
                    updated_lines = 0
                    for r in render_results:
                        line_id = r.get("line_id")
                        if not line_id:
                            continue
                        line_fields = {
                            "sfx_engine":   "procedural",
                            "sfx_type":     str(r["chosen_type"]),
                            "dur_s":        float(r["dur_s"]),
                            "sfx_wav_path": r.get("wav_path"),
                        }
                        if _OTRL.patch_line_fields(
                            led_disk, line_id, line_fields,
                        ):
                            updated_lines += 1
                    if updated_lines:
                        _OTRL.save_ledger_safe(ledger_path, led_disk)
                        batch_log.append(
                            f"ledger updated (line_id stamping): "
                            f"{updated_lines}/{len(render_results)} sfx "
                            f"line(s) -> {ledger_path.name}"
                        )
        except Exception as _exc:
            batch_log.append(f"ledger write-back failed (non-fatal): {_exc}")

        return ({"waveform": batched_waveform, "sample_rate": SAMPLE_RATE}, "\n".join(batch_log))

NODE_CLASS_MAPPINGS = {"OTR_BatchProceduralSFX": BatchProceduralSFX}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_BatchProceduralSFX": "[FAST] Batch Procedural SFX (Obsidian)"}
