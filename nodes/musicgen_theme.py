"""
MusicGen Theme - dedicated instrumental music bus for opening, closing, and
act-break interstitial cues.

Replaces the previous "no music at all" hole in the OTR pipeline. Reads the
three fixed music cues out of production_plan_json (written by LLMDirector,
one tailored prompt per episode), generates them via transformers' native
MusicGen-medium (facebook/musicgen-medium), and emits three AUDIO tensors that
feed straight into EpisodeAssembler's opening_theme_audio and
closing_theme_audio inputs, plus an interstitial clip for future act-break use.

Design notes (see ROADMAP v1.4 Theme A):
  - NO audiocraft dependency. Uses transformers.MusicgenForConditionalGeneration
    and AutoProcessor - both already installed in the OTR venv via the main
    transformers package. Clean install, no MSVC, no spacy pin, no av conflict.
  - Per-episode caching. Each (prompt, duration) pair is SHA-256 hashed to a
    .wav filename under models/musicgen_cache/. If the cache file exists the
    model is never loaded. Same episode -> same music, deterministic.
  - Sequential VRAM discipline. Model loads only if at least one cue is
    uncached. After generation it is explicitly unloaded and cuda cache is
    flushed, so Bark has its full VRAM window when BatchBark runs next.
  - musicgen-medium is ~6 GB VRAM - fits cleanly inside the 14.5 GB ceiling
    once the LLM has been unloaded (which happens automatically at the
    LLMDirector exit, before this node runs).
  - 32 kHz native sample rate, mono. SceneSequencer output is 48 kHz - the
    EpisodeAssembler downstream already handles rate matching, so we leave
    the 32 kHz rate intact in the returned AUDIO dict.

Jeffrey Brick - v1.4 Theme A
"""

import gc
import hashlib
import json
import logging
import os

import numpy as np
import torch

from ._otr_paths import otr_episodes_root, otr_legacy_audio_dir
from . import _otr_ledger as _OTRL_PATHS
from ._vram_log import force_vram_offload

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
    """Per-episode MusicGen output dir.

    Walks the in-flight ledger via ``find_most_recent_ledger`` to find the
    current episode's ``otr/episodes/<ep>/audio/`` dir, then writes
    MusicGen wavs alongside the ledger. Falls back to legacy
    ``models/musicgen_cache/`` when no in-flight ledger is found (testing
    or first-of-pipeline contexts).

    Per Jeffrey directive 2026-05-02 EVENING: every per-episode asset
    including MusicGen output lives in the per-episode workspace under
    ``otr/episodes/<episode_id>/audio/``. SignalLostVideo's rename pass
    moves the entire per-episode dir from ``pending_<ts>/`` to
    ``<canonical_episode_id>/`` so the wavs travel with the rename.
    """
    try:
        # BUG-LOCAL-021 (Phase G): use in-flight singleton, not mtime
        # walker. Singleton tracks the active episode by construction;
        # walker can return a stale leftover across queue boundaries.
        ledger_path = _OTRL_PATHS.in_flight_ledger_path()
        if ledger_path is not None:
            base = str(ledger_path.parent)
            os.makedirs(base, exist_ok=True)
            return base
    except Exception as exc:
        log.warning("[MusicGenTheme] per-episode cache_dir lookup failed: %s", exc)

    # Legacy fallback: shared models/musicgen_cache/.
    try:
        import folder_paths
        base = os.path.join(folder_paths.models_dir, CACHE_SUBDIR)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        base = os.path.normpath(os.path.join(here, "..", "..", "..", "models", CACHE_SUBDIR))
    os.makedirs(base, exist_ok=True)
    return base


def _cache_prefix(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    """Deterministic cache identity prefix.

    Format: ``<cue_id>_<sha8>``
      - ``cue_id`` -- human-readable cue (opening / closing / interstitial)
      - ``sha8`` -- 8 hex chars of SHA256(cue_id|duration|prompt|seed)

    BUG-LOCAL-017 (Phase D, 2026-05-02): the prior implementation appended
    ``_<timestamp_ms>`` to this prefix and returned a full filename. That
    forced a fresh filename on every call, which guaranteed a cache MISS
    every run (the lookup checked exactly the just-computed path). Net
    cost: ~22s wasted MusicGen renders per episode AND a Rule C7 violation
    because FFmpeg embeds input WAV filenames in MP4 metadata streams,
    so the final mp4 bytes drifted between identical-input runs.

    Fix: split lookup identity (this function, deterministic prefix) from
    write filename (``_cache_filename_for_write``, canonical
    ``<prefix>.wav``). Lookup uses ``_find_cached`` which checks the
    canonical filename first, then falls back to legacy timestamped
    files for back-compat.
    """
    payload = f"{cue_id}|{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{cue_id}_{digest}"


def _cache_filename_for_write(cue_id: str, prompt: str, duration_sec: int,
                              episode_seed: str) -> str:
    """Canonical filename for a fresh cache write. Deterministic -- no
    timestamp. C7-safe: same inputs always land at the same filename so
    downstream FFmpeg metadata stays byte-identical run-to-run."""
    return f"{_cache_prefix(cue_id, prompt, duration_sec, episode_seed)}.wav"


def _cache_key(cue_id: str, prompt: str, duration_sec: int, episode_seed: str) -> str:
    """Backward-compatible wrapper -- returns the canonical write filename.
    Kept so any external import of ``_cache_key`` from this module still
    resolves. New code should call ``_cache_filename_for_write`` (write)
    or ``_cache_prefix`` + ``_find_cached`` (lookup) directly."""
    return _cache_filename_for_write(cue_id, prompt, duration_sec, episode_seed)


def _find_cached(cache_dir: str, prefix: str) -> str | None:
    """Find a cached WAV for ``prefix``. Two-level lookup:
      1. Canonical: ``<prefix>.wav`` (preferred, written by current code)
      2. Legacy:    ``<prefix>_<ts>.wav`` (back-compat with files written
         by the pre-Phase-D timestamped implementation)

    Returns the absolute path on hit, None on miss. Never raises.

    Per Phase D consult (Gemini): use ``iterdir() + startswith`` rather
    than ``Path.glob()`` because prompts can contain glob metacharacters
    like ``[`` and ``*`` which break glob patterns. Sort legacy matches
    by parsed filename timestamp (not mtime) so selection is stable
    across copy/restore/touch operations.
    """
    from pathlib import Path

    base = Path(cache_dir)
    canonical = base / f"{prefix}.wav"
    if canonical.is_file():
        return str(canonical)

    if not base.exists():
        return None

    legacy_prefix = prefix + "_"
    matches: list[Path] = []
    try:
        for path in base.iterdir():
            name = path.name
            if (path.is_file()
                    and name.startswith(legacy_prefix)
                    and name.lower().endswith(".wav")):
                matches.append(path)
    except OSError as exc:
        log.warning("[MusicGenTheme] cache_dir iterdir failed: %s", exc)
        return None

    if not matches:
        return None
    if len(matches) > 1:
        log.warning(
            "[MusicGenTheme] multiple legacy cache files for prefix %s; "
            "using newest filename timestamp",
            prefix,
        )

    def _legacy_sort_key(path: Path):
        # name = "<prefix>_<ts_ms>.wav" -- strip prefix and .wav, parse int
        suffix = path.name[len(legacy_prefix):-4]
        try:
            return (1, int(suffix), path.name)
        except ValueError:
            return (0, 0, path.name)

    matches.sort(key=_legacy_sort_key, reverse=True)
    return str(matches[0])


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
    """Atomic WAV write: write to a sibling ``.tmp`` then ``os.replace``
    into the canonical path. Prevents corrupted cache hits if the
    process is killed mid-write (Phase D consult, Gemini's catch).

    Note: explicit ``format='WAV'`` is required because soundfile cannot
    infer the audio format from the ``.tmp`` extension.
    """
    tmp_path = path + ".tmp"
    try:
        import soundfile as sf
        sf.write(tmp_path, waveform, sample_rate,
                 subtype="FLOAT", format="WAV")
        os.replace(tmp_path, path)
    except Exception as exc:
        log.warning("[MusicGenTheme] Failed to write cache %s: %s", path, exc)
        # Best-effort cleanup of tmp on failure
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


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
    """OTR v1.4 - instrumental music generator for opening, closing, and
    act-break interstitial cues.

    Reads the three music cues written by LLMDirector into
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
                    "tooltip": "Production plan JSON from LLMDirector. music_plan key is read.",
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

        # [EMOJI] MANDATORY VRAM POWER WASH (Clean slate before start)
        force_vram_offload()

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
        #
        # BUG-LOCAL-017 (Phase D): lookup uses the deterministic prefix via
        # ``_find_cached`` which checks canonical ``<prefix>.wav`` first
        # then falls back to legacy ``<prefix>_<ts>.wav`` files. Write
        # path uses the canonical (no-timestamp) filename so re-runs
        # produce byte-identical mp4 metadata (Rule C7).
        results: dict[str, dict] = {}
        to_generate: list[str] = []
        for cue_id, cue in cues.items():
            prefix = _cache_prefix(
                cue_id, cue["prompt"], cue["duration_sec"], episode_seed
            )
            hit_path = _find_cached(cache_dir, prefix)
            if hit_path is not None:
                cached = _load_cached_wav(hit_path)
                if cached is not None:
                    cue["cache_path"] = hit_path
                    tensor, sr = cached
                    results[cue_id] = {"waveform": tensor, "sample_rate": sr}
                    render_log.append(
                        f"  [{cue_id}] CACHE HIT ({os.path.basename(hit_path)})"
                    )
                    continue
                # Load failed (corrupt/unreadable) -- fall through to
                # generate, write to canonical path which will overwrite
                # the bad file via atomic _save_wav.
                render_log.append(
                    f"  [{cue_id}] CACHE FOUND BUT UNREADABLE ({os.path.basename(hit_path)}); regenerating"
                )
            cue["cache_path"] = os.path.join(
                cache_dir,
                _cache_filename_for_write(
                    cue_id, cue["prompt"], cue["duration_sec"], episode_seed
                ),
            )
            to_generate.append(cue_id)
            render_log.append(
                f"  [{cue_id}] MISS - will generate ({cue['duration_sec']}s)"
            )

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

            # v1.4.10 Hardening: Force cache_dir to our local Hub directory
            cache_dir_path = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.float16 if device == "cuda" else torch.float32

            processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_dir_path)
            model = MusicgenForConditionalGeneration.from_pretrained(
                model_id, torch_dtype=dtype, cache_dir=cache_dir_path
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

                    # BUG-LOCAL-030 audit-completion (2026-05-03 EVENING):
                    # track per-cue render wall-clock so the ledger can
                    # stamp ledger.music[].render_ms.
                    import time as _mg_time
                    _mg_t0 = _mg_time.time()
                    with torch.no_grad():
                        audio_values = model.generate(
                            **inputs,
                            max_new_tokens=max_new_tokens,
                            do_sample=True,
                            guidance_scale=guidance_scale,
                        )
                    _mg_render_ms = int((_mg_time.time() - _mg_t0) * 1000)

                    # audio_values shape: (batch=1, channels=1, samples)
                    audio_np = audio_values[0, 0].detach().cpu().float().numpy()
                    # Peak normalize to -1 dBFS so cues sit at consistent level.
                    peak = float(np.max(np.abs(audio_np))) or 1.0
                    audio_np = (audio_np / peak * 0.89).astype(np.float32)

                    _save_wav(cue["cache_path"], audio_np, MUSICGEN_SAMPLE_RATE)

                    tensor = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                    # BUG-LOCAL-030 audit-completion: stash render_ms +
                    # audio_sample_hash on the cue dict so the ledger
                    # write-back loop below can pick them up alongside
                    # the existing wav_path + dur_s stamps.
                    cue["_render_ms"] = int(_mg_render_ms)
                    try:
                        from . import _otr_ledger as _OTRL_HASH  # type: ignore
                        cue["_audio_sample_hash"] = (
                            _OTRL_HASH.compute_audio_sample_hash(audio_np)
                        )
                    except Exception:
                        cue["_audio_sample_hash"] = ""
                    results[cue_id] = {
                        "waveform": tensor,
                        "sample_rate": MUSICGEN_SAMPLE_RATE,
                    }
                    render_log.append(
                        f"  [{cue_id}] GENERATED {len(audio_np) / MUSICGEN_SAMPLE_RATE:.1f}s "
                        f"-> {os.path.basename(cue['cache_path'])} "
                        f"(render_ms={_mg_render_ms})"
                    )
            finally:
                # Always unload to return VRAM to Bark, even if generation failed.
                # Bug Bible 12.19 VRAM leak fix - explicit .cpu() before dropping references
                try:
                    if 'model' in locals():
                        model.cpu()
                        del model
                    if 'processor' in locals():
                        del processor
                except Exception:
                    pass
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                render_log.append("model unloaded, cuda cache cleared")
        else:
            render_log.append("all cues cached - MusicGen model not loaded")

        render_log.append(
            f"--- 3 music cues ready (opening, closing, interstitial) ---"
        )

        # ---- BUG-LOCAL-095: write music wav_paths back to ledger ----
        # Find the most recent pending or final ledger and stamp each
        # cue's wav_path + dur_s into ledger.music[]. Silent skip when
        # no ledger exists yet (rare; usually ScriptWriter has run by
        # the time MusicGen does). Errors logged but never crash.
        try:
            import json as _json
            import time as _time
            # Per-episode workspace: walk otr/episodes/<ep>/audio/*_ledger.json
            # via the centralized helper so both layouts (legacy flat +
            # per-episode tree) are searched.
            ledger_path = _OTRL_PATHS.find_most_recent_ledger(
                [otr_episodes_root(), otr_legacy_audio_dir()]
            )
            if ledger_path is not None:
                led = _json.loads(ledger_path.read_text(encoding="utf-8"))
                music_rows = led.get("music") or []
                # Build a lookup by cue_id; if a row is missing for
                # one of our cues, skip that one (don't synthesize
                # ledger schema -- keep node side-effect bounded).
                music_by_id = {(r.get("cue_id") or ""): r for r in music_rows}
                updated = 0
                for cue_id in CUE_IDS:
                    cue_dict = cues.get(cue_id) or {}
                    cache_path = cue_dict.get("cache_path")
                    result_dict = results.get(cue_id) or {}
                    wf = result_dict.get("waveform")
                    sr = int(result_dict.get("sample_rate", 0)) or MUSICGEN_SAMPLE_RATE
                    dur = None
                    if wf is not None and sr > 0:
                        try:
                            dur = float(wf.shape[-1]) / float(sr)
                        except Exception:
                            dur = None
                    row = music_by_id.get(cue_id)
                    if row is not None:
                        if cache_path:
                            row["wav_path"] = str(cache_path)
                        if dur is not None:
                            row["dur_s"] = dur
                        # BUG-LOCAL-030 audit-completion (2026-05-03 EVENING):
                        # stamp tts_engine + render_ms + generated_dur_s +
                        # audio_sample_hash on the music row. The
                        # generation_prompt is already populated by
                        # LLMDirector; this closes the loop on the
                        # render-result side.
                        row["tts_engine"] = "musicgen"
                        if cue_dict.get("_render_ms"):
                            row["render_ms"] = int(cue_dict["_render_ms"])
                        if dur is not None:
                            row["generated_dur_s"] = float(dur)
                        if cue_dict.get("_audio_sample_hash"):
                            row["audio_sample_hash"] = str(
                                cue_dict["_audio_sample_hash"]
                            )
                        updated += 1
                if updated:
                    ledger_path.write_text(
                        _json.dumps(led, indent=2, ensure_ascii=False),
                        encoding="utf-8",
                    )
                    log.info(
                        "[MusicGenTheme] BUG-095 ledger updated: "
                        "%d music cue path(s) written to %s",
                        updated, ledger_path.name,
                    )
                    render_log.append(
                        f"ledger updated: {updated} music wav_path(s) -> "
                        f"{ledger_path.name}"
                    )
        except Exception as _exc:
            log.warning("[MusicGenTheme] BUG-095 ledger write-back failed: %s", _exc)

        log_text = "\n".join(render_log)
        return (
            results["opening"],
            results["closing"],
            results["interstitial"],
            log_text,
        )


NODE_CLASS_MAPPINGS = {"MusicGenTheme": MusicGenTheme}
NODE_DISPLAY_NAME_MAPPINGS = {"MusicGenTheme": "[EMOJI] MusicGen Theme"}
