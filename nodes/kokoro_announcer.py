"""
Kokoro Announcer — dedicated non-Bark narrator bus.

Routes every ANNOUNCER dialogue line in the script to Kokoro v1.0 instead of
Bark, eliminating Bark's "ums" and "ahs" and restroom-reverb throat clears
from the "Voice of God" bookends. Kokoro is clean, broadcast-ready, and costs
about 1.5 GB VRAM — well inside our 14.5 GB real-world ceiling.

Design notes (see ROADMAP v1.4 Theme A):
  - Picks ONE British voice per episode from a curated grab bag of 4
    (2 male + 2 female, seeded from episode_seed). Gender is balanced 50/50
    across episodes, matching the Bark announcer pool behavior we replaced.
  - Lazy-imports `kokoro` and `KPipeline` so a missing install does not
    break the rest of the OTR node load.
  - Voice .pt files are pulled on demand from 1038lab/KokoroTTS via
    huggingface_hub — the grab-bag only needs 4 files total (~12 MB).
  - Output is a batched AUDIO tensor in script order (ANNOUNCER lines only),
    which SceneSequencer consumes via a separate announcer_clip_idx counter.
    Non-announcer dialogue still flows through BatchBark as before.

Jeffrey Brick — v1.4 Theme A
"""

import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger("OTR")


# British grab bag, 2 male + 2 female, BBC authoritative + documentary relaxed.
# Keep this list small and intentional — the whole point is a clean, curated
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


def _extract_announcer_lines(script) -> list:
    """Pull every ANNOUNCER dialogue line out of the Canonical 1.0 script.

    Returns a list of dicts: {script_idx, line}. Order matches script order
    so SceneSequencer can consume them sequentially.
    """
    out = []
    for i, item in enumerate(script):
        if item.get("type") != "dialogue":
            continue
        name = (item.get("character_name") or "").strip().upper()
        line = (item.get("line") or "").strip()
        if name == "ANNOUNCER" and line:
            out.append({"script_idx": i, "line": line})
    return out


class KokoroAnnouncer:
    """OTR v1.4 — dedicated Kokoro-based ANNOUNCER bus.

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
                    "tooltip": "Parsed script JSON from Gemma4ScriptWriter",
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
        import json

        script = json.loads(script_json) if isinstance(script_json, str) else script_json
        announcer_items = _extract_announcer_lines(script)

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
                    "kokoro package not installed — run: pip install kokoro",
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
        pipeline = KPipeline(lang_code="b")

        clips = []
        render_log = [f"=== Kokoro Announcer ({voice_id}, speed={speed}) ==="]

        for item in announcer_items:
            idx = item["script_idx"]
            line = item["line"]
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
                render_log.append(f"  [{idx}] ANNOUNCER ({dur:.1f}s): {line[:55]}")
            except Exception as exc:
                log.warning("[KokoroAnnouncer] Line %d failed: %s", idx, exc)
                render_log.append(f"  [{idx}] ANNOUNCER FAILED: {exc}")
                # Silence placeholder estimated from word count at 2.5 wps
                word_count = max(1, len(line.split()))
                est_samples = int(KOKORO_SAMPLE_RATE * word_count / 2.5)
                clips.append(np.zeros(est_samples, dtype=np.float32))

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
            
        log_text = "\n".join(render_log)
        return (audio_out, log_text, voice_id)


NODE_CLASS_MAPPINGS = {"KokoroAnnouncer": KokoroAnnouncer}
NODE_DISPLAY_NAME_MAPPINGS = {"KokoroAnnouncer": "🎙️ Kokoro Announcer"}
