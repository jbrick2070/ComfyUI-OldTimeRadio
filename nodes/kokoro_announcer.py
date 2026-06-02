"""
Kokoro Announcer - dedicated non-Bark narrator bus.

Routes every ANNOUNCER dialogue line in the script to Kokoro v1.0 instead of
Bark, eliminating Bark's "ums" and "ahs" and restroom-reverb throat clears
from the "Voice of God" bookends. Kokoro is clean, broadcast-ready, and costs
about 1.5 GB VRAM - well inside our 14.5 GB real-world ceiling.

Design notes (see ROADMAP v1.4 Theme A):
  - Picks ONE British voice per episode from a curated grab bag of 4
    (2 male + 2 female, seeded from episode_seed). Gender is balanced 50/50
    across episodes, matching the Bark announcer pool behavior we replaced.
  - Lazy-imports `kokoro` and `KPipeline` so a missing install does not
    break the rest of the OTR node load.
  - Voice .pt files are pulled on demand from 1038lab/KokoroTTS via
    huggingface_hub - the grab-bag only needs 4 files total (~12 MB).
  - Output is a batched AUDIO tensor in script order (ANNOUNCER lines only),
    which SceneSequencer consumes via a separate announcer_clip_idx counter.
    Non-announcer dialogue still flows through BatchBark as before.

Jeffrey Brick - v1.4 Theme A
"""

import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger("OTR")


# British grab bag, 2 male + 2 female, BBC authoritative + documentary relaxed.
# Keep this list small and intentional - the whole point is a clean, curated
# announcer pool instead of sharing Bark's 10-preset crowd.
ANNOUNCER_VOICE_POOL = [
    "bm_george",   # BBC authoritative male
    "bm_fable",    # documentary relaxed male
    "bf_emma",     # BBC authoritative female
    "bf_lily",     # documentary relaxed female
]

KOKORO_SAMPLE_RATE = 24000
KOKORO_MODEL_SUBDIR = os.path.join("TTS", "KokoroTTS")


def _kokoro_model_dir() -> str:
    """Return absolute path to ComfyUI models/TTS/KokoroTTS."""
    try:
        import folder_paths
        return os.path.join(folder_paths.models_dir, KOKORO_MODEL_SUBDIR)
    except Exception:
        # Fallback for non-Comfy contexts (tests, CLI)
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.normpath(os.path.join(here, "..", "..", "..", "models", KOKORO_MODEL_SUBDIR))


def _ensure_voice_file(voice_id: str) -> str:
    """Make sure the .pt voice file for voice_id is on disk.

    Downloads from 1038lab/KokoroTTS on Hugging Face if missing. Returns the
    absolute path to the .pt file. Raises on download failure so the caller
    can fall back to another voice.
    """
    base = _kokoro_model_dir()
    voice_dir = os.path.join(base, "voices")
    os.makedirs(voice_dir, exist_ok=True)
    target = os.path.join(voice_dir, f"{voice_id}.pt")
    if os.path.exists(target):
        return target

    log.info("[KokoroAnnouncer] Downloading voice %s", voice_id)
    from huggingface_hub import hf_hub_download
    hf_hub_download(
        repo_id="1038lab/KokoroTTS",
        filename=f"voices/{voice_id}.pt",
        local_dir=base,
        local_dir_use_symlinks=False,
    )
    if not os.path.exists(target):
        raise RuntimeError(f"Kokoro voice download succeeded but file missing: {target}")
    return target


def _pick_announcer_voice(episode_seed: str, voice_override: str) -> str:
    """Select one voice from the grab bag.

    If voice_override is a specific voice id, return it. Otherwise seed the
    RNG from the episode seed so the same episode always picks the same
    announcer voice (deterministic, reproducible).
    """
    if voice_override and voice_override != "random":
        return voice_override
    rng = random.Random(f"{episode_seed}_kokoro_announcer")
    return rng.choice(ANNOUNCER_VOICE_POOL)


def _extract_announcer_lines(led: dict) -> list:
    """Pull every announcer line out of the v2 ledger.

    Walks ``ledger["lines"]`` filtered to ``speaker_role == "announcer"``
    via ``_otr_ledger_consumers.iter_lines``. Returns a list of dicts:
    ``{script_idx, line_id, line, traits}``. Order matches ledger line
    order so SceneSequencer can consume them sequentially.

    The legacy parser-list shape (``[{type: "dialogue", character_name:
    "ANNOUNCER", line: "..."}]``) is no longer accepted here -- the
    caller in ``render()`` calls ``load_ledger`` first, which raises
    ValueError on the legacy shape (see Pattern 1: loud-fail at the
    consumer boundary).
    """
    from . import _otr_ledger_consumers as _OTRLC
    out = []
    for i, line in enumerate(_OTRLC.iter_lines(led, roles={"announcer"})):
        text = (line.get("text") or "").strip()
        if not text:
            continue
        out.append({
            "script_idx": i,
            "line_id":    line.get("line_id"),
            "line":       text,
            "traits":     (line.get("traits") or ""),
        })
    return out


class KokoroAnnouncer:
    """OTR v1.4 - dedicated Kokoro-based ANNOUNCER bus.

    Reads the script JSON, extracts ANNOUNCER lines only, renders them with
    Kokoro v1.0 (British voice, seeded grab bag), and emits a batched AUDIO
    tensor for SceneSequencer to splice in.
    """

    CATEGORY = "OldTimeRadio"
    FUNCTION = "render"
    RETURN_TYPES = ("AUDIO", "STRING", "STRING")
    RETURN_NAMES = ("announcer_audio_clips", "render_log", "chosen_voice")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "Parsed script JSON from LLMScriptWriter",
                }),
            },
            "optional": {
                "episode_seed": ("STRING", {
                    "default": "",
                    "tooltip": "Seed string (from ScriptWriter). Same seed -> same announcer voice.",
                }),
                "voice_override": (["random"] + ANNOUNCER_VOICE_POOL, {
                    "default": "random",
                    "tooltip": "Force a specific British voice, or 'random' for seeded pick.",
                }),
                "speed": ("FLOAT", {
                    "default": 0.95, "min": 0.7, "max": 1.3, "step": 0.05,
                    "tooltip": "Kokoro speech rate. 0.95 gives a calmer authoritative cadence.",
                }),
            },
        }

    def render(self, script_json, episode_seed="", voice_override="random", speed=0.95):
        import time as _kk_time

        # Read-side: parse the wire input as a v2 ledger dict.
        # load_ledger raises ValueError on the legacy parser-list shape;
        # Kokoro is in the "fail loud" group (Pattern 1) -- bad wiring
        # halts the run early instead of silently producing no announcer
        # audio.
        from . import _otr_ledger_consumers as _OTRLC
        led = _OTRLC.load_ledger(script_json)
        # R0a: seed all RNGs from the FROZEN script_json so legacy Kokoro audio
        # is reproducible (I-2). No parse -- hash the raw wire string. (The voice
        # pick uses its own seeded random.Random, unaffected.) Operator render-
        # twice baseline (R0a step f) locks this in.
        from ._otr_resolved_request import _seed_to_int64
        from ._otr_determinism import seed_all_rngs
        seed_all_rngs(_seed_to_int64("kokoro_legacy_v1", script_json))
        announcer_items = _extract_announcer_lines(led)

        if not announcer_items:
            log.info("[KokoroAnnouncer] No ANNOUNCER lines in script")
            empty = {
                "waveform": torch.zeros(1, 1, int(KOKORO_SAMPLE_RATE * 0.1)),
                "sample_rate": KOKORO_SAMPLE_RATE,
            }
            return (empty, "No ANNOUNCER lines found", "none")

        voice_id = _pick_announcer_voice(episode_seed, voice_override)
        log.info("[KokoroAnnouncer] Chosen announcer voice: %s (%d lines)",
                 voice_id, len(announcer_items))

        # Lazy import so a missing kokoro install doesn't break OTR startup.
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            log.error("[KokoroAnnouncer] kokoro package not installed: %s", exc)
            empty = {
                "waveform": torch.zeros(1, 1, int(KOKORO_SAMPLE_RATE * 0.1)),
                "sample_rate": KOKORO_SAMPLE_RATE,
            }
            return (empty,
                    "kokoro package not installed - run: pip install kokoro",
                    voice_id)

        try:
            _ensure_voice_file(voice_id)
        except Exception as exc:
            log.error("[KokoroAnnouncer] Voice file fetch failed for %s: %s",
                      voice_id, exc)
            empty = {
                "waveform": torch.zeros(1, 1, int(KOKORO_SAMPLE_RATE * 0.1)),
                "sample_rate": KOKORO_SAMPLE_RATE,
            }
            return (empty, f"Kokoro voice fetch failed: {exc}", voice_id)

        # lang_code 'b' = British English in Kokoro v1.0
        # v1.4.10: Explicitly specify repo_id to suppress migration warnings and ensure local-only stability.
        pipeline = KPipeline(lang_code="b", device="cuda", repo_id='hexgrad/Kokoro-82M')

        clips = []
        render_log = [f"=== Kokoro Announcer ({voice_id}, speed={speed}) ==="]

        # BUG-LOCAL-030 audit-completion (2026-05-03 EVENING): track
        # per-line render metadata so we can stamp the new forensic
        # ledger fields after the loop. line_id is carried from
        # _extract_announcer_lines so the write-back below can patch
        # by line_id (Pattern 4) instead of fragile text-match.
        per_line_meta: list[dict] = []

        for item in announcer_items:
            idx = item["script_idx"]
            line = item["line"]
            line_id = item.get("line_id")
            _kk_t0 = _kk_time.time()
            try:
                generator = pipeline(
                    line,
                    voice=voice_id,
                    speed=speed,
                    split_pattern=r"\n+",
                )
                segments = []
                for _, _, audio_data in generator:
                    if torch.is_tensor(audio_data):
                        audio_np = audio_data.detach().cpu().numpy()
                    else:
                        audio_np = np.asarray(audio_data, dtype=np.float32)
                    segments.append(audio_np.astype(np.float32).squeeze())

                if not segments:
                    raise RuntimeError("pipeline produced no audio")

                clip_np = np.concatenate(segments) if len(segments) > 1 else segments[0]
                peak = float(np.max(np.abs(clip_np))) or 1.0
                clip_np = clip_np / peak * 0.9  # peak-normalize to -1 dBFS
                clips.append(clip_np)
                dur = len(clip_np) / KOKORO_SAMPLE_RATE
                _kk_render_ms = int((_kk_time.time() - _kk_t0) * 1000)
                # Compute hash + stash render_ms for ledger stamping.
                _kk_hash = ""
                try:
                    from . import _otr_ledger as _OTRL_HASH  # type: ignore
                    _kk_hash = _OTRL_HASH.compute_audio_sample_hash(clip_np)
                except Exception:
                    _kk_hash = ""
                per_line_meta.append({
                    "line_id": line_id,
                    "render_ms": int(_kk_render_ms),
                    "generated_dur_s": float(dur),
                    "audio_sample_hash": _kk_hash,
                })
                render_log.append(
                    f"  [{idx}] ANNOUNCER ({dur:.1f}s, render_ms={_kk_render_ms}): {line[:55]}"
                )
            except Exception as exc:
                log.warning("[KokoroAnnouncer] Line %d failed: %s", idx, exc)
                render_log.append(f"  [{idx}] ANNOUNCER FAILED: {exc}")
                # Silence placeholder estimated from word count at 2.5 wps
                word_count = max(1, len(line.split()))
                est_samples = int(KOKORO_SAMPLE_RATE * word_count / 2.5)
                clips.append(np.zeros(est_samples, dtype=np.float32))
                per_line_meta.append({
                    "line_id": line_id,
                    "render_ms": 0,
                    "generated_dur_s": float(est_samples) / KOKORO_SAMPLE_RATE,
                    "audio_sample_hash": "",
                })

        # Assemble into batched AUDIO tensor (B, C, T) with zero-padding.
        max_len = max(len(c) for c in clips)
        batch = np.zeros((len(clips), 1, max_len), dtype=np.float32)
        for b, clip in enumerate(clips):
            batch[b, 0, : len(clip)] = clip
        waveform = torch.from_numpy(batch)

        audio_out = {"waveform": waveform, "sample_rate": KOKORO_SAMPLE_RATE}
        render_log.append(f"--- {len(clips)} announcer clips rendered ---")
        
        # Bug Bible 12.19: explicitly drop model refs to return VRAM.
        try:
            if hasattr(pipeline, "model"):
                pipeline.model.to("cpu")
            del pipeline
        except Exception:
            pass
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ---- BUG-LOCAL-030 audit-completion (2026-05-03 EVENING) ----
        # Stamp per-line announcer audio render metadata into the
        # in-flight ledger. Closes the audit gap surfaced by the
        # artifacts-grid review: KokoroAnnouncer previously had ZERO
        # ledger writes. Now stamps tts_engine="kokoro" + voice_preset
        # + render_ms + generated_dur_s + audio_sample_hash + dur_s +
        # start_s on every announcer ledger.lines[] row.
        #
        # Stamp strategy (v2 ledger): patch by line["line_id"] via
        # _otr_ledger.patch_line_fields(). Replaces the old
        # text-match-then-stamp block (fragile when two announcer
        # lines shared identical short text). per_line_meta carries
        # line_id from the iter_lines walk in _extract_announcer_lines.
        # start_s is cumulative across the announcer set only --
        # SceneSequencer overwrites with the assembled-timeline value
        # downstream; this is a forensic stamp.
        # Best-effort: any I/O failure is logged but never raised.
        try:
            from . import _otr_ledger as _OTRL  # type: ignore
            ledger_path = _OTRL.in_flight_ledger_path()
            if ledger_path is not None:
                led_disk = _OTRL.load_ledger_safe(ledger_path)
                if led_disk is None:
                    log.warning(
                        "[KokoroAnnouncer] in-flight ledger load failed at "
                        "%s; skipping ledger write-back",
                        ledger_path,
                    )
                else:
                    cumulative_start = 0.0
                    updated = 0
                    for meta in per_line_meta:
                        line_id = meta.get("line_id")
                        if not line_id:
                            continue
                        dur = float(meta.get("generated_dur_s") or 0.0)
                        fields: dict = {
                            "tts_engine":  "kokoro",
                            "voice_preset": str(voice_id),
                            "dur_s":        dur,
                            "start_s":      cumulative_start,
                        }
                        if int(meta.get("render_ms", 0)) > 0:
                            fields["render_ms"] = int(meta["render_ms"])
                        if dur > 0:
                            fields["generated_dur_s"] = dur
                        if meta.get("audio_sample_hash"):
                            fields["audio_sample_hash"] = str(
                                meta["audio_sample_hash"]
                            )
                        if _OTRL.patch_line_fields(led_disk, line_id, fields):
                            cumulative_start += dur
                            updated += 1
                    if updated:
                        _OTRL.save_ledger_safe(ledger_path, led_disk)
                        render_log.append(
                            f"ledger updated (line_id stamping): "
                            f"{updated} announcer line(s) -> "
                            f"{ledger_path.name}"
                        )
                        log.info(
                            "[KokoroAnnouncer] BUG-030 ledger updated "
                            "(line_id stamping): %d announcer line(s) "
                            "stamped in %s",
                            updated, ledger_path.name,
                        )
        except Exception as _kk_exc:
            log.warning(
                "[KokoroAnnouncer] BUG-030 ledger write-back failed: %s",
                _kk_exc,
            )
            render_log.append(f"ledger write-back failed: {_kk_exc}")

        log_text = "\n".join(render_log)
        return (audio_out, log_text, voice_id)


NODE_CLASS_MAPPINGS = {"KokoroAnnouncer": KokoroAnnouncer}
NODE_DISPLAY_NAME_MAPPINGS = {"KokoroAnnouncer": "[EMOJI]- Kokoro Announcer"}
