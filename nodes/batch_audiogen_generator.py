"""
Batch AudioGen Generator - high-fidelity generative Foley for "Signal Lost".
==========================================================================

Replaces the previous "silence" or procedural-only SFX with high-quality sound
effects generated via facebook/audiogen-medium. 

Architectural Highlights:
  - Contextual Matching: Parses the script_json for [SFX: ...] tags and matches 
    them in order to the sfx_plan dictionary from LLMDirector.
  - Per-Prompt Caching: SHA-256 hashed filenames under models/sfx_cache/ ensure
    we never waste VRAM or time generating the same sound twice for the same episode.
  - VRAM Discipline: Loads the model only if uncached cues exist. unloads it
    immediately after generation, returning the memory window for Bark TTS or 
    Video rendering.
  - Native transformers implementation: Low friction, no complex audiocraft 
    dependency issues.

v1.5 AudioGen Integration - Jeffrey Brick
"""

import gc
import hashlib
import json
import logging
import os
import re

import numpy as np
import torch

from ._otr_paths import otr_episodes_root, otr_legacy_audio_dir
from . import _otr_ledger as _OTRL_PATHS
from ._vram_log import force_vram_offload

log = logging.getLogger("OTR")

AUDIOGEN_MODEL_ID = "facebook/audiogen-medium"
AUDIOGEN_SAMPLE_RATE = 32000 # Native rate for AudioGen
CACHE_SUBDIR = "sfx_cache"


def _cache_dir() -> str:
    """Per-episode AudioGen / SFX output dir.

    Walks the in-flight ledger via ``find_most_recent_ledger`` to find the
    current episode's ``otr/episodes/<ep>/audio/`` dir, then writes SFX
    wavs alongside the ledger. Falls back to legacy
    ``models/sfx_cache/`` when no in-flight ledger is found.

    Per Jeffrey directive 2026-05-02 EVENING: every per-episode asset
    including SFX output lives in the per-episode workspace under
    ``otr/episodes/<episode_id>/audio/``. SignalLostVideo's rename pass
    moves the entire per-episode dir from ``pending_<ts>/`` to
    ``<canonical_episode_id>/`` so the wavs travel with the rename.
    """
    try:
        # BUG-LOCAL-021 (Phase G): use in-flight singleton, not mtime
        # walker. See full rationale in _otr_ledger.in_flight_ledger_path.
        ledger_path = _OTRL_PATHS.in_flight_ledger_path()
        if ledger_path is not None:
            base = str(ledger_path.parent)
            os.makedirs(base, exist_ok=True)
            return base
    except Exception as exc:
        print(f"[BatchAudioGen] per-episode cache_dir lookup failed: {exc}")

    # Legacy fallback: shared models/sfx_cache/.
    try:
        import folder_paths
        base = os.path.join(folder_paths.models_dir, CACHE_SUBDIR)
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        base = os.path.normpath(os.path.join(here, "..", "..", "..", "models", CACHE_SUBDIR))
    os.makedirs(base, exist_ok=True)
    return base


def _cache_prefix(prompt: str, duration_sec: float, episode_seed: str) -> str:
    """Deterministic SFX cache identity prefix.

    Format: ``sfx_<safe_prompt_prefix>_<sha8>``
      - ``safe_prompt_prefix`` -- first 20 chars of prompt, sanitized
        for filenames (human-readable; cosmetic, not identity)
      - ``sha8`` -- 8 hex chars of SHA256(duration|prompt|episode_seed)

    BUG-LOCAL-017 (Phase D, 2026-05-02): the prior implementation appended
    ``_<timestamp_ms>`` to this prefix and returned a full filename. That
    forced a fresh filename on every call, which guaranteed a cache MISS
    every run AND violated Rule C7 because FFmpeg embeds input WAV
    filenames in MP4 metadata. Same fix as MusicGen: split lookup
    identity (this function) from write filename
    (``_cache_filename_for_write``, canonical ``<prefix>.wav``).
    """
    payload = f"{duration_sec}|{prompt}|{episode_seed}".encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    safe_name = re.sub(r'[^a-zA-Z0-9]', '_', prompt[:20]).lower()
    return f"sfx_{safe_name}_{digest}"


def _cache_filename_for_write(prompt: str, duration_sec: float,
                              episode_seed: str) -> str:
    """Canonical filename for a fresh cache write. Deterministic -- no
    timestamp. C7-safe."""
    return f"{_cache_prefix(prompt, duration_sec, episode_seed)}.wav"


def _cache_key(prompt: str, duration_sec: float, episode_seed: str) -> str:
    """Backward-compatible wrapper -- returns the canonical write filename.
    Kept so any external import of ``_cache_key`` from this module still
    resolves."""
    return _cache_filename_for_write(prompt, duration_sec, episode_seed)


def _find_cached(cache_dir: str, prefix: str) -> str | None:
    """Find a cached SFX WAV for ``prefix``. Two-level lookup:
      1. Canonical: ``<prefix>.wav`` (preferred, written by current code)
      2. Legacy:    ``<prefix>_<ts>.wav`` (back-compat with files written
         by the pre-Phase-D timestamped implementation)

    Per Phase D consult (Gemini): use ``iterdir() + startswith`` rather
    than ``Path.glob()`` because prompts can contain glob metacharacters
    like ``[`` and ``*``. Sort legacy matches by parsed filename
    timestamp (not mtime) for stability across copy/restore.
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
        log.warning("[BatchAudioGen] cache_dir iterdir failed: %s", exc)
        return None

    if not matches:
        return None
    if len(matches) > 1:
        log.warning(
            "[BatchAudioGen] multiple legacy cache files for prefix %s; "
            "using newest filename timestamp",
            prefix,
        )

    def _legacy_sort_key(path: Path):
        suffix = path.name[len(legacy_prefix):-4]
        try:
            return (1, int(suffix), path.name)
        except ValueError:
            return (0, 0, path.name)

    matches.sort(key=_legacy_sort_key, reverse=True)
    return str(matches[0])


def _save_wav(path: str, waveform: np.ndarray, sample_rate: int) -> None:
    """Atomic WAV write: write to ``.tmp`` then ``os.replace``. Prevents
    corrupted cache hits if the process is killed mid-write.

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
        log.warning("[BatchAudioGen] Failed to write cache %s: %s", path, exc)
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass


def _load_cached_wav(path: str) -> torch.Tensor | None:
    if not os.path.exists(path):
        return None
    try:
        import soundfile as sf
        data, sr = sf.read(path, dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1) # force mono
        tensor = torch.from_numpy(np.asarray(data, dtype=np.float32))
        return tensor.unsqueeze(0).unsqueeze(0), sr # (1, 1, T), sr
    except Exception as exc:
        log.warning("[BatchAudioGen] Failed to read cache %s: %s", path, exc)
        return None


class BatchAudioGenGenerator:
    """Generates a batch of SFX cues from a script using AudioGen."""

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
                "production_plan_json": ("STRING", {
                    "multiline": True, "default": "{}",
                    "tooltip": "Production plan JSON from LLMDirector (optional under v2 ledger flow; empty {} degrades gracefully -- sfx_plan falls back to default_duration)",
                }),
                "episode_seed": ("STRING", {"default": ""}),
                # BUG-LOCAL-027: the "3"/"3.0"/3/3.0 entries were scar tissue
                # from widget-drift hitting this node. With the mapper fix in
                # _workflow_to_api_prompt, socket-only inputs no longer leak
                # into widget slots, so the hack is no longer needed. Fail
                # loudly on bad input instead of silently accepting garbage.
                "model_id": (["facebook/audiogen-medium", "facebook/audiogen-small"], {"default": "facebook/audiogen-medium"}),
                "guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "default_duration": ("FLOAT", {"default": 3.0, "min": 0.5, "max": 10.0, "step": 0.5}),
            }
        }

    def generate(self, script_json, production_plan_json="{}", episode_seed="",
                 model_id="facebook/audiogen-medium", guidance_scale=3.0, default_duration=3.0):
        
        # [EMOJI] MANDATORY VRAM POWER WASH (Clean slate before start)
        force_vram_offload()
        
        # UI JSON back-compat fix
        if str(model_id) in ["3", "3.0"]:
            model_id = "facebook/audiogen-medium"
            
        batch_log = ["=== Batch AudioGen Generator ==="]

        # Read-side: parse the wire input as a v2 ledger dict.
        # load_ledger raises ValueError on the legacy parser-list shape;
        # AudioGen is in the loud-fail group (Pattern 1) -- bad wiring
        # halts the run early. production_plan_json is now optional;
        # an empty / unwired "{}" degrades to default_duration via the
        # graceful production_plan_or_empty helper.
        from . import _otr_ledger_consumers as _OTRLC
        led = _OTRLC.load_ledger(script_json)
        plan = _OTRLC.production_plan_or_empty(production_plan_json)

        sfx_plan = plan.get("sfx_plan", [])

        # Walk ledger sfx lines (Pattern 2: roles={"sfx"}). The cue
        # text comes DIRECTLY from line["text"] -- no [SFX:] regex,
        # no parser-list "description" field. line_id is carried
        # through render_queue so the write-back below can stamp by
        # line_id (Pattern 4) on ledger.lines[].
        sfx_items = []
        for line in _OTRLC.iter_lines(led, roles={"sfx"}):
            cue = (line.get("text") or "").strip()
            if not cue:
                continue
            sfx_items.append({
                "line_id": line.get("line_id"),
                "text":    cue,
            })
        sfx_tags = [item["text"] for item in sfx_items]

        if not sfx_tags:
            batch_log.append("No SFX cues found in ledger. Returning silence.")
            return ({"waveform": torch.zeros(1, 1, 10), "sample_rate": 32000}, "\n".join(batch_log))

        batch_log.append(f"Found {len(sfx_tags)} SFX cues in ledger.")

        # 2. Match tags to plan prompts
        # BUG-LOCAL-116 fix 2026-04-30: honor Director's sfx_plan[i].dur_s
        # (or .duration_sec / .duration) instead of always using default.
        # The Director plans per-cue durations to fit narrative beats;
        # ignoring them was producing a 3.0s render of every SFX even
        # when the cue was a short sting. Combined with the output-shape
        # bug, the cache_key collision meant short/long stings shared
        # cache files. Both fixed here.
        render_queue = []
        for i, tag in enumerate(sfx_tags):
            prompt = tag
            duration = float(default_duration)

            # Try to match to sfx_plan by order or description
            if i < len(sfx_plan):
                plan_entry = sfx_plan[i] if isinstance(sfx_plan[i], dict) else {}
                prompt = (
                    plan_entry.get("generation_prompt")
                    or plan_entry.get("description")
                    or tag
                )
                # Director-specified duration takes precedence over default.
                # Several schema variants exist across versions of the
                # Director output -- accept any of them.
                for _key in ("dur_s", "duration_sec", "duration"):
                    _v = plan_entry.get(_key)
                    if isinstance(_v, (int, float)) and _v > 0:
                        # Clamp to AudioGen's reasonable bounds so a
                        # hallucinated 60-second cue doesn't OOM.
                        duration = max(0.5, min(10.0, float(_v)))
                        break

            # BUG-LOCAL-017 (Phase D): split lookup from write path. Lookup
            # via _find_cached checks canonical <prefix>.wav first then
            # falls back to legacy <prefix>_<ts>.wav. Write path uses the
            # canonical (no-timestamp) filename so re-runs produce
            # byte-identical mp4 metadata (Rule C7).
            cache_dir_now = _cache_dir()
            prefix = _cache_prefix(prompt, duration, episode_seed)
            hit_path = _find_cached(cache_dir_now, prefix)
            cache_path = hit_path if hit_path is not None else os.path.join(
                cache_dir_now,
                _cache_filename_for_write(prompt, duration, episode_seed),
            )
            render_queue.append({
                "index":   i,
                "line_id": sfx_items[i]["line_id"],
                "tag":     tag,
                "prompt":  prompt,
                "duration": duration,
                "cache_path": cache_path,
                "had_cache_hit_at_resolve": hit_path is not None,
            })

        # 3. Check cache (loads bytes for verified hits; on read-failure
        # of a "hit" path, fall through to regenerate to canonical path).
        final_clips = [None] * len(render_queue)
        to_generate_indices = []

        for i, item in enumerate(render_queue):
            cached = (_load_cached_wav(item["cache_path"])
                      if item["had_cache_hit_at_resolve"] else None)
            if cached:
                item["audio"], sr = cached
                final_clips[i] = item["audio"]
                batch_log.append(
                    f"  [{i}] CACHE HIT: {item['tag'][:30]} "
                    f"({os.path.basename(item['cache_path'])})"
                )
            else:
                if item["had_cache_hit_at_resolve"]:
                    batch_log.append(
                        f"  [{i}] CACHE FOUND BUT UNREADABLE: {item['tag'][:30]}; regenerating"
                    )
                    # Redirect write to canonical path so we overwrite
                    # the bad legacy/canonical file via atomic _save_wav.
                    item["cache_path"] = os.path.join(
                        cache_dir_now,
                        _cache_filename_for_write(
                            item["prompt"], item["duration"], episode_seed
                        ),
                    )
                else:
                    batch_log.append(f"  [{i}] MISS: {item['tag'][:30]}")
                to_generate_indices.append(i)

        # 4. Generate missing cues
        if to_generate_indices:
            try:
                from transformers import AutoProcessor, AudiogenForConditionalGeneration
            except ImportError as exc:
                batch_log.append(f"Error: transformers AudioGen not available: {exc}")
                # Fallback to silence for missing
                for idx in to_generate_indices:
                    final_clips[idx] = torch.zeros(1, 1, int(AUDIOGEN_SAMPLE_RATE * default_duration))
            else:
                batch_log.append(f"Loading {model_id}...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                dtype = torch.float16 if device == "cuda" else torch.float32
                
                processor = AutoProcessor.from_pretrained(model_id)
                model = AudiogenForConditionalGeneration.from_pretrained(model_id, torch_dtype=dtype).to(device)
                model.eval()
                
                tokens_per_sec = 50 # AudioGen specific approx
                
                try:
                    for idx in to_generate_indices:
                        item = render_queue[idx]
                        prompt = item["prompt"]
                        duration = item["duration"]
                        max_new_tokens = int(duration * tokens_per_sec)

                        batch_log.append(
                            f"  Generating [{idx}] dur={duration:.2f}s "
                            f"max_new_tokens={max_new_tokens}: {prompt[:50]}..."
                        )
                        inputs = processor(
                            text=[prompt], padding=True, return_tensors="pt"
                        ).to(device)

                        # BUG-LOCAL-030 audit-completion (2026-05-03 EVENING):
                        # track per-sfx render wall-clock so the ledger
                        # write-back can stamp ledger.sfx[].render_ms.
                        import time as _ag_time
                        _ag_t0 = _ag_time.time()
                        with torch.no_grad():
                            audio_values = model.generate(
                                **inputs,
                                max_new_tokens=max_new_tokens,
                                do_sample=True,
                                guidance_scale=guidance_scale,
                            )
                        item["_render_ms"] = int(
                            (_ag_time.time() - _ag_t0) * 1000
                        )

                        # BUG-LOCAL-116 fix 2026-04-30: robust shape
                        # extraction. AudioGen's transformers output shape
                        # has varied across versions:
                        #   - [batch, channels, samples]  (older, decoded)
                        #   - [batch, samples]            (newer, decoded mono)
                        #   - [batch, num_codebooks, seq] (token IDs - bug)
                        # The OLD code did audio_values[0, 0] which on a
                        # 2D tensor returns a SCALAR, and on a token-ID
                        # tensor returns the codebook 0 token sequence
                        # (~150 ints). Result: 3 ms WAVs.
                        # New logic: find the longest float-ish 1D slice
                        # and validate length is plausibly audio.
                        _av = audio_values
                        if hasattr(_av, "audio_values"):
                            _av = _av.audio_values   # named-tuple wrapper
                        _av = _av.detach().cpu().float()
                        if _av.dim() == 3:
                            audio_np = _av[0, 0].numpy()
                        elif _av.dim() == 2:
                            audio_np = _av[0].numpy()
                        elif _av.dim() == 1:
                            audio_np = _av.numpy()
                        else:
                            audio_np = _av.flatten().numpy()

                        # Sanity check: minimum 0.25 sec of real audio
                        # (= 8000 samples @ 32kHz). Anything shorter is
                        # an output-shape bug or generation failure;
                        # fall back to silence at the requested duration
                        # so the timeline still has a plausible slot
                        # rather than a 3 ms blip.
                        _min_samples = int(AUDIOGEN_SAMPLE_RATE * 0.25)
                        if audio_np.size < _min_samples:
                            log.warning(
                                "[BatchAudioGen] [%d] generated audio too "
                                "short (%d samples = %.4fs) -- expected "
                                ">= %.2fs. Output shape was %s. "
                                "Falling back to silence at %.2fs. "
                                "(BUG-LOCAL-116 / transformers AudioGen "
                                "output-shape regression.)",
                                idx, audio_np.size,
                                audio_np.size / AUDIOGEN_SAMPLE_RATE,
                                _min_samples / AUDIOGEN_SAMPLE_RATE,
                                tuple(audio_values.shape) if hasattr(audio_values, "shape") else "?",
                                duration,
                            )
                            audio_np = np.zeros(
                                int(AUDIOGEN_SAMPLE_RATE * duration),
                                dtype=np.float32,
                            )

                        # Peak normalize
                        peak = np.abs(audio_np).max() or 1.0
                        audio_np = (audio_np / peak * 0.9).astype(np.float32)

                        _save_wav(item["cache_path"], audio_np, AUDIOGEN_SAMPLE_RATE)
                        final_clips[idx] = torch.from_numpy(audio_np).unsqueeze(0).unsqueeze(0)
                        # BUG-LOCAL-030 audit-completion: stash hash for
                        # ledger write-back. compute_audio_sample_hash
                        # is best-effort (returns "" on extraction
                        # failure) so this never raises.
                        try:
                            from . import _otr_ledger as _OTRL_HASH  # type: ignore
                            item["_audio_sample_hash"] = (
                                _OTRL_HASH.compute_audio_sample_hash(audio_np)
                            )
                        except Exception:
                            item["_audio_sample_hash"] = ""
                        batch_log.append(
                            f"    [{idx}] saved {audio_np.size} samples "
                            f"({audio_np.size / AUDIOGEN_SAMPLE_RATE:.2f}s) "
                            f"render_ms={item.get('_render_ms', 0)}"
                        )
                        
                finally:
                    # 2026-04-26 PM BUG-LOCAL-073 GUARD: synchronize before
                    # cpu() so a pending CUDA kernel fault surfaces as a
                    # clean Python exception instead of zombifying the
                    # process during model walk. Mirror of the guard in
                    # story_orchestrator._unload_llm.
                    sync_ok = True
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.synchronize()
                        except Exception as sync_err:
                            sync_ok = False
                            batch_log.append(
                                f"AudioGen unload: cuda.synchronize() failed ({sync_err}); skipping cpu() walk"
                            )
                    if 'model' in locals():
                        if sync_ok:
                            try:
                                model.cpu()
                            except Exception as cpu_err:
                                batch_log.append(
                                    f"AudioGen unload: model.cpu() failed ({cpu_err}); proceeding with empty_cache only"
                                )
                        del model
                    if 'processor' in locals():
                        del processor
                    gc.collect()
                    if torch.cuda.is_available():
                        try:
                            torch.cuda.empty_cache()
                        except Exception as ec_err:
                            batch_log.append(f"AudioGen unload: empty_cache failed ({ec_err})")
                    batch_log.append("Model unloaded, VRAM cleared.")

        # 5. Build batched AUDIO output
        # ComfyUI AUDIO is a list or a dict with 'waveform' tensor.
        # Batching requires padding to max length.
        max_samples = max(clip.shape[2] for clip in final_clips)
        batched_waveform = torch.zeros(len(final_clips), 1, max_samples)

        for i, clip in enumerate(final_clips):
            samples = clip.shape[2]
            batched_waveform[i, 0, :samples] = clip[0, 0, :samples]

        # ---- BUG-LOCAL-095: write SFX wav_paths back to ledger ----
        # Two parallel write-back paths:
        #   1. Legacy ledger.sfx[] parallel-index walk (preserved
        #      for back-compat producers that still emit a separate
        #      ledger.sfx[] array; v2 producers leave sfx[] empty
        #      and this loop no-ops).
        #   2. NEW v2: per-line stamp on ledger.lines[] for sfx
        #      rows via patch_line_fields(led, line_id, ...). sfx
        #      are first-class lines in the v2 ledger -- this
        #      mirrors Sequencer's new sfx_line_positions pattern
        #      from consumer #4.
        # Both paths mutate the same led_disk dict; one save_ledger_safe
        # at the end (Pattern 4 contract).
        try:
            ledger_path = _OTRL_PATHS.in_flight_ledger_path()
            if ledger_path is not None:
                led_disk = _OTRL_PATHS.load_ledger_safe(ledger_path)
                if led_disk is None:
                    log.warning(
                        "[BatchAudioGen] in-flight ledger load failed at "
                        "%s; skipping ledger write-back",
                        ledger_path,
                    )
                else:
                    # Path 1: legacy ledger.sfx[] (preserved as-is).
                    sfx_rows = led_disk.get("sfx") or []
                    updated_sfx_array = 0
                    for i, item in enumerate(render_queue):
                        if i >= len(sfx_rows):
                            break
                        row = sfx_rows[i]
                        cache_path = item.get("cache_path")
                        clip_t = final_clips[i] if i < len(final_clips) else None
                        dur = None
                        if clip_t is not None:
                            try:
                                dur = float(clip_t.shape[2]) / float(AUDIOGEN_SAMPLE_RATE)
                            except Exception:
                                dur = None
                        if cache_path:
                            row["wav_path"] = str(cache_path)
                        if dur is not None:
                            row["dur_s"] = dur
                        row["tts_engine"] = "audiogen"
                        if item.get("_render_ms"):
                            row["render_ms"] = int(item["_render_ms"])
                        if dur is not None:
                            row["generated_dur_s"] = float(dur)
                        if item.get("_audio_sample_hash"):
                            row["audio_sample_hash"] = str(
                                item["_audio_sample_hash"]
                            )
                        updated_sfx_array += 1

                    # Path 2: NEW v2 ledger.lines[] sfx rows via line_id.
                    # Field names: sfx_wav_path, dur_s, sfx_engine="audiogen"
                    # (sfx-specific names disambiguate from dialogue's
                    # tts_engine/voice_preset/bark_wav_path on the same
                    # ledger.lines[] array). Plus the per-line audiogen
                    # meta fields (render_ms, generated_dur_s,
                    # audio_sample_hash) under their existing names.
                    updated_lines = 0
                    for i, item in enumerate(render_queue):
                        line_id = item.get("line_id")
                        if not line_id:
                            continue
                        cache_path = item.get("cache_path")
                        clip_t = final_clips[i] if i < len(final_clips) else None
                        dur = None
                        if clip_t is not None:
                            try:
                                dur = float(clip_t.shape[2]) / float(AUDIOGEN_SAMPLE_RATE)
                            except Exception:
                                dur = None
                        line_fields: dict = {
                            "sfx_engine": "audiogen",
                        }
                        if cache_path:
                            line_fields["sfx_wav_path"] = str(cache_path)
                        if dur is not None:
                            line_fields["dur_s"] = float(dur)
                            line_fields["generated_dur_s"] = float(dur)
                        if item.get("_render_ms"):
                            line_fields["render_ms"] = int(item["_render_ms"])
                        if item.get("_audio_sample_hash"):
                            line_fields["audio_sample_hash"] = str(
                                item["_audio_sample_hash"]
                            )
                        if _OTRL_PATHS.patch_line_fields(
                            led_disk, line_id, line_fields,
                        ):
                            updated_lines += 1

                    if updated_sfx_array or updated_lines:
                        # Single atomic save (Pattern 4 contract).
                        _OTRL_PATHS.save_ledger_safe(ledger_path, led_disk)
                        batch_log.append(
                            f"BUG-095 ledger updated (line_id stamping): "
                            f"sfx_array={updated_sfx_array}/{len(render_queue)}, "
                            f"lines={updated_lines}/{len(render_queue)} -> "
                            f"{ledger_path.name}"
                        )
        except Exception as _exc:
            batch_log.append(f"BUG-095 ledger write-back failed: {_exc}")

        return ({"waveform": batched_waveform, "sample_rate": AUDIOGEN_SAMPLE_RATE}, "\n".join(batch_log))

NODE_CLASS_MAPPINGS = {"OTR_BatchAudioGenGenerator": BatchAudioGenGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_BatchAudioGenGenerator": "[FAST] Batch AudioGen (Foley)"}
