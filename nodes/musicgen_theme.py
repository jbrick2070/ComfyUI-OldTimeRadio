"""
MusicGen Theme — dedicated instrumental music bus for opening, closing, and
act-break interstitial cues.

Replaces the previous "no music at all" hole in the OTR pipeline. Reads the
three fixed music cues out of production_plan_json (written by Gemma4Director,
one tailored prompt per episode), generates them via transformers' native
MusicGen-medium (facebook/musicgen-medium), and emits three AUDIO tensors that
feed straight into EpisodeAssembler's opening_theme_audio and
closing_theme_audio inputs, plus an interstitial clip for future act-break use.

Design notes (see ROADMAP v1.4 Theme A):
  - NO audiocraft dependency. Uses transformers.MusicgenForConditionalGeneration
    and AutoProcessor — both already installed in the OTR venv via the main
    transformers package. Clean install, no MSVC, no spacy pin, no av conflict.
  - Per-episode caching. Each (prompt, duration) pair is SHA-256 hashed to a
    .wav filename under models/musicgen_cache/. If the cache file exists the
    model is never loaded. Same episode -> same music, deterministic.
  - Sequential VRAM discipline. Model loads only if at least one cue is
    uncached. After generation it is explicitly unloaded and cuda cache is
    flushed, so Bark has its full VRAM window when BatchBark runs next.
  - musicgen-medium is ~6 GB VRAM — fits cleanly inside the 14.5 GB ceiling
    once Gemma4 has been unloaded (which happens automatically at the
    Gemma4Director exit, before this node runs).
  - 32 kHz native sample rate, mono. SceneSequencer output is 48 kHz — the
    EpisodeAssembler downstream already handles rate matching, so we leave
    the 32 kHz rate intact in the returned AUDIO dict.

Jeffrey Brick — v1.4 Theme A
"""

import gc
import hashlib
import json
import logging
import os

import numpy as np
import torch

log = logging.getLogger("OTR")


# Fixed cue ids the Director is instructed to emit. If any are missing from
# the production plan we fall back to sensible defaults so the pipeline never
# breaks on a malformed plan.
CUE_IDS = ["opening", "closing", "interstitial"]
CUE_DEFAULTS = {
    "opening": {
        "duration_sec": 12,
        "generation_prompt": (
            "1940s old time radio opening theme, warm brass fanfare, upright bass, "
            "snare brushes, mono AM radio character, tube saturation, confident and "
            "mysterious, ends on a held chord"
        ),
    },
    "closing": {
        "duration_sec": 8,
        "generation_prompt": (
            "1940s old time radio closing sting, brass and strings, resolving cadence, "
            "warm tube saturation, fades to silence"
        ),
    },
    "interstitial": {
        "duration_sec": 4,
        "generation_prompt": (
            "short old time radio act-break stinger, single brass hit with cymbal "
            "swell, mono, tube warmth"
        ),
    },
}

MUSICGEN_MODEL_ID = "facebook/musicgen-medium"
MUSICGEN_SAMPLE_RATE = 32000  # native rate for musicgen-medium
CACHE_SUBDIR = "musicgen_cache"


def _cache_dir() -> str:
    """Return models/musicgen_cache, creating it if needed."""
    try:
        import folder_paths
        base = os.path.join(folder_paths.models_dir, CACHE_SUBDIR)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        base = os.path.normpath(os.path.join(here, "..", "..", "..", "models", CACHE_SUBDIR))
    os.makedirs(base, exist_ok=True)
    return base


def _cache_key(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    """Deterministic cache filename. Episode seed is part of the key so two
    episodes with identical prompts still get their own cached files if the
    user explicitly scoped them by seed."""
    payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{cue_id}_{digest}.wav"


def _load_cached_wav(path: str) -> torch.Tensor | None:
    """Load a cached .wav as a (1, 1, T) float tensor, or None if missing."""
    if not os.path.exists(path):
        return None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)  # force mono
        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
        return tensor.unsqueeze(0).unsqueeze(0), sr  # (1, 1, T), sr
    except Exception as exc:
        log.warning("[MusicGenTheme] Failed to read cache %s: %s", path, exc)
        return None


def _save_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    try:
        import soundfile as sf
        sf.write(path, waveform, sample_rate, subtype="FLOAT")
    except Exception as exc:
        log.warning("[MusicGenTheme] Failed to write cache %s: %s", path, exc)


def _resolve_cue(cue_id: str, music_plan: list) -> tuple[str, int]:
    """Pull the matching cue dict out of the plan, falling back to defaults
    for any missing field."""
    defaults = CUE_DEFAULTS[cue_id]
    for entry in music_plan or []:
        if (entry.get("cue_id") or "").strip().lower() == cue_id:
            prompt = (entry.get("generation_prompt") or "").strip() or defaults["generation_prompt"]
            try:
                duration = int(entry.get("duration_sec") or defaults["duration_sec"])
            except (TypeError, ValueError):
                duration = defaults["duration_sec"]
            return prompt, duration
    return defaults["generation_prompt"], defaults["duration_sec"]


def _silent_audio_dict(sample_rate: int = MUSICGEN_SAMPLE_RATE) -> dict:
    return {
        "waveform": torch.zeros(1, 1, int(sample_rate * 0.1)),
        "sample_rate": sample_rate,
    }


class MusicGenTheme:
    """OTR v1.4 — instrumental music generator for opening, closing, and
    act-break interstitial cues.

    Reads the three music cues written by Gemma4Director into
    production_plan_json, generates any cue that isn't already in the
    per-episode cache, and returns three AUDIO tensors ready to wire into
    EpisodeAssembler.
    """

    CATEGORY = "OldTimeRadio"
    FUNCTION = "render"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "STRING")
    RETURN_NAMES = ("opening_audio", "closing_audio", "interstitial_audio", "render_log")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Production plan JSON from Gemma4Director. music_plan key is read.",
                }),
            },
            "optional": {
                "episode_seed": ("STRING", {
                    "default": "",
                    "tooltip": "Episode seed string. Becomes part of the cache key so re-runs of the same episode reuse the same music.",
                }),
                "model_id": ("STRING", {
                    "default": MUSICGEN_MODEL_ID,
                    "tooltip": "Hugging Face model id. Default is facebook/musicgen-medium (~6 GB VRAM).",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Classifier-free guidance. 3.0 is the MusicGen default.",
                }),
            },
        }

    def render(self, production_plan_json, episode_seed="",
               model_id=MUSICGEN_MODEL_ID, guidance_scale=3.0):

        try:
            plan = json.loads(production_plan_json) if isinstance(production_plan_json, str) else production_plan_json
        except Exception as exc:
            log.error("[MusicGenTheme] production_plan_json parse failed: %s", exc)
            plan = {}

        music_plan = plan.get("music_plan", [])

        # Resolve all three cues from the plan (with fallback defaults).
        cues = {}
        for cue_id in CUE_IDS:
            prompt, duration = _resolve_cue(cue_id, music_plan)
            cues[cue_id] = {"prompt": prompt, "duration_sec": duration}

        cache_dir = _cache_dir()
        render_log = [
            "=== MusicGen Theme (medium) ===",
            f"cache dir: {cache_dir}",
            f"episode seed: {episode_seed or '<none>'}",
        ]

        # First pass: try to load all three from cache. Only load the model if
        # at least one cue is missing. This keeps re-runs of the same episode
        # instant and VRAM-free.
        results: dict[str, dict] = {}
        to_generate: list[str] = []
        for cue_id, cue in cues.items():
            cache_path = os.path.join(
                cache_dir,
                _cache_key(cue_id, cue["prompt"], cue["duration_sec"], episode_seed),
            )
            cue["cache_path"] = cache_path
            cached = _load_cached_wav(cache_path)
            if cached is not None:
                tensor, sr = cached
                results[cue_id] = {"waveform": tensor, "sample_rate": sr}
                render_log.append(f"  [{cue_id}] CACHE HIT ({os.path.basename(cache_path)})")
            else:
                to_generate.append(cue_id)
                render_log.append(f"  [{cue_id}] MISS — will generate ({cue['duration_sec']}s)")

        if to_generate:
            try:
                from transformers import MusicgenForConditionalGeneration, AutoProcessor
            except ImportError as exc:
                log.error("[MusicGenTheme] transformers MusicGen not available: %s", exc)
                # Return silence for anything we could not generate.
                for cue_id in to_generate:
                    results[cue_id] = _silent_audio_dict()
                render_log.append(f"  ERROR: transformers MusicGen import failed: {exc}")
                return (
                    results["opening"], results["closing"], results["interstitial"],
                    "\n".join(render_log),
                )

            log.info("[MusicGenTheme] Loading %s for %d uncached cue(s)",
                     model_id, len(to_generate))
            render_log.append(f"loading {model_id} for {len(to_generate)} cue(s)...")

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32
            processor = AutoProcessor.from_pretrained(model_id)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype,
            ).to(device)
            model.eval()

            # MusicGen produces ~50 tokens per second of audio at 32 kHz.
            tokens_per_sec = 50

            try:
                for cue_id in to_generate:
                    cue = cues[cue_id]
                    prompt = cue["prompt"]
                    duration = cue["duration_sec"]
                    max_new_tokens = int(duration * tokens_per_sec) + 8

                    log.info("[MusicGenTheme] Generating %s (%ds): %s",
                             cue_id, duration, prompt[:60])

                    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to(device)

                    with torch.no_grad():
                        audio_values = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            guidance_scale=guidance_scale,
                        )

                    # audio_values shape: (batch=1, channels=1, samples)
                    audio_np = audio_values[0, 0].detach().cpu().float().numpy()
                    # Peak normalize to -1 dBFS so cues sit at consistent level.
                    peak = float(np.max(np.abs(audio_np))) or 1.0
                    audio_np = (audio_np / peak * 0.89).astype(np.float32)

                    _save_wav(cue["cache_path"], audio_np, MUSICGEN_SAMPLE_RATE)

                    tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                    results[cue_id] = {
                        "waveform": tensor,
                        "sample_rate": MUSICGEN_SAMPLE_RATE,
                    }
                    render_log.append(
                        f"  [{cue_id}] GENERATED {len(audio_np) / MUSICGEN_SAMPLE_RATE:.1f}s "
                        f"-> {os.path.basename(cue['cache_path'])}"
                    )
            finally:
                # Always unload to return VRAM to Bark, even if generation failed.
                try:
                    del model
                    del processor
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                render_log.append("model unloaded, cuda cache cleared")
        else:
            render_log.append("all cues cached — MusicGen model not loaded")

        render_log.append(
            f"--- 3 music cues ready (opening, closing, interstitial) ---"
        )
        log_text = "\n".join(render_log)
        return (
            results["opening"],
            results["closing"],
            results["interstitial"],
            log_text,
        )


NODE_CLASS_MAPPINGS = {"MusicGenTheme": MusicGenTheme}
NODE_DISPLAY_NAME_MAPPINGS = {"MusicGenTheme": "🎺 MusicGen Theme"}
