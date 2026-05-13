"""
MusicGen Theme - dedicated instrumental music bus for opening, closing, and
act-break interstitial cues.

Reads style + mood signal directly from the L3 ledger (the post-freeze
script_json emitted by OTR_LedgerFreezeCascade) and synthesizes three
era-neutral cue prompts via a deterministic palette + mood overlay. No
LLM call, no legacy Director-derived plan, no period defaults. Generates the
audio via transformers' native MusicGen-medium and emits three AUDIO
tensors that feed straight into EpisodeAssembler's opening_theme_audio
and closing_theme_audio inputs plus an interstitial clip for act-break
use.

Data source contract:
  - script_json -> L3 ledger dict (parsed via _otr_ledger_consumers.load_ledger)
  - style       -> ledger.meta.gen_params_initial.style (snake_case slug)
  - mood signal -> ledger.meta.news.script_brief (when present)
  - episode_seed widget overrides ledger.meta.gen_params_initial.seed
    for cache key construction

Design notes:
  - NO audiocraft dependency. Uses transformers.MusicgenForConditionalGeneration
    and AutoProcessor - both already installed in the OTR venv via the main
    transformers package.
  - Per-episode caching. Each (prompt, duration, seed) tuple is SHA-256
    hashed to a .wav filename under the per-episode workspace dir. If the
    cache file exists the model is never loaded. Same episode -> same
    music, deterministic.
  - Sequential VRAM discipline. Model loads only if at least one cue is
    uncached. After generation it is explicitly unloaded and cuda cache is
    flushed, so Bark has its full VRAM window when BatchBark runs next.
  - musicgen-medium is ~6 GB VRAM - fits cleanly inside the 14.5 GB ceiling
    once the LLM has been unloaded.
  - 32 kHz native sample rate, mono. SceneSequencer output is 48 kHz - the
    EpisodeAssembler downstream already handles rate matching, so we leave
    the 32 kHz rate intact in the returned AUDIO dict.

Style palette and mood tags are deliberately era-neutral. No "1940s",
"vintage", "old time radio", or other period anchors. The writer's
style slug owns the visual / aural register; this module renders music
to match.

Jeffrey Brick - voice-path-cleanbreak 2026-05-12
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


# Three fixed cues. Durations are part of the cache identity, so changing
# them invalidates every cached wav on disk. Keep stable.
CUE_IDS = ["opening", "closing", "interstitial"]
CUE_DURATIONS = {"opening": 12, "closing": 8, "interstitial": 4}

# Universal suffix appended to every MusicGen prompt. MusicGen-medium can
# drift into vocal-like artefacts; this steers it back to instrumental.
# Applied AFTER mood overlay so it always lands last.
_PROMPT_TAIL = ", instrumental only, no dialogue, no vocals"

# Single source of truth for music cue prompts. One entry per active
# writer style slug (matches OTR_LedgerScriptWriter._STYLE_PICKER_SEED_POOL).
# Era-neutral by directive - no "1940s", "vintage", "old time radio",
# "warm brass", "upright bass" anchors. Unknown slug is a hard fail; no
# default palette survives.
_STYLE_PALETTE: dict[str, dict[str, str]] = {
    "closed_room_suspense": {
        "opening":      "slow muted strings, low piano cluster, sparse texture, "
                        "rising tension on a single chord",
        "closing":      "diminished chord fade, slow exhale of low woodwind, "
                        "resolves to silence",
        "interstitial": "single sustained bass note, room tone bed, "
                        "minimal percussion, brief and unresolved",
    },
    "detective_case_file": {
        "opening":      "methodical minor piano motif, walking double bass, "
                        "soft brushed snare, procedural and unhurried",
        "closing":      "piano cadence resolving down a minor third, "
                        "single cymbal bell tap, settles cleanly",
        "interstitial": "short walking bass figure, two-bar minor piano "
                        "phrase, professional and brief",
    },
    "pulp_serial_cliffhanger": {
        "opening":      "bold orchestral brass theme, full strings, "
                        "rolling timpani, theatrical and dramatic, "
                        "ends on a held suspended chord",
        "closing":      "orchestral hit, dramatic crescendo into snare "
                        "roll, resolves on a held major chord",
        "interstitial": "brass stab, quick timpani roll, single "
                        "high cymbal accent",
    },
    "mission_control_procedural": {
        "opening":      "clean synth pulse, instrument-panel beep accents, "
                        "tight arpeggiated bass, calm institutional energy",
        "closing":      "descending synth bleeps, soft pad release, "
                        "console-shutdown texture",
        "interstitial": "single synth bleep sequence, quiet hum bed, "
                        "very short",
    },
    "deep_space_distress_call": {
        "opening":      "modal synthesizer pad, distant pulsing beacon, "
                        "sub-bass swell, isolated and vast, slow build",
        "closing":      "isolated low pad, signal dropout, granular "
                        "static decay into silence",
        "interstitial": "single sustained tone, faint radio interference, "
                        "brief and dimensional",
    },
    "noir_interrogation": {
        "opening":      "muted solo trumpet, low double bass walk, "
                        "smoky tenor saxophone, dim and atmospheric",
        "closing":      "muted trumpet fade, low bass note hold, "
                        "single piano chord trailing off",
        "interstitial": "single muted trumpet phrase, sparse bass, "
                        "smoky and short",
    },
    "small_town_uncanny": {
        "opening":      "diffuse string pad, muted bell tone, slow harmonic "
                        "drift, quiet wrongness in the mid register",
        "closing":      "soft string fade, single high bell strike, "
                        "decay into ambient hush",
        "interstitial": "single muted bell, faint string pad, brief and "
                        "off-balance",
    },
    "radio_newsroom_emergency": {
        "opening":      "urgent ticker percussion, telegraph rhythm pattern, "
                        "tight low brass pulse, motion and momentum",
        "closing":      "decisive low brass figure, snare hit, fast cadence "
                        "to a resolving chord",
        "interstitial": "short telegraph rhythm, single brass accent, "
                        "newsroom urgency",
    },
    "haunted_broadcast_signal": {
        "opening":      "degraded signal artefacts, ghostly choir-like synth pad, "
                        "granular static texture, distant and unstable",
        "closing":      "pad decay through layered static, fading into "
                        "noise floor and silence",
        "interstitial": "short granular swell, signal flutter, brief "
                        "ghosted texture",
    },
    "laboratory_containment": {
        "opening":      "sterile electronic tone, precise arpeggio, "
                        "high pure sine layer, clean and isolated",
        "closing":      "tone descends to a held low frequency, single "
                        "click, controlled fade",
        "interstitial": "short electronic pulse sequence, sterile and "
                        "precise",
    },
}

# Light mood overlay mined from meta.news.script_brief. Keyword scan,
# no LLM. Tags concatenate to the cue prompt as a comma-prefixed suffix.
# Keywords are checked case-insensitively against the brief.
_MOOD_TAGS: dict[str, str] = {
    "betrayal":   "minor mode, unresolved tension",
    "discovery":  "rising figure, slight upward motion",
    "loss":       "subdued, slow decay",
    "urgent":     "tighter rhythm, percussive accents",
    "isolation":  "sparse texture, wide stereo field",
    "danger":     "building tension, dissonant cluster",
    "mystery":    "harmonic ambiguity, slow modulation",
    "triumph":    "resolving cadence, brighter register",
    "conflict":   "rhythmic accents, opposing voices",
    "silence":    "minimal density, long pauses",
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


def _cache_prefix(*, cue_id: str, prompt: str, duration_sec: float,
                  episode_seed: str, model_id: str,
                  guidance_scale: float) -> str:
    """Deterministic cache identity prefix.

    Format: ``<cue_id>_<sha12>``
      - ``cue_id`` -- human-readable cue (opening / closing / interstitial)
      - ``sha12`` -- 12 hex chars of SHA256 over a JSON-canonical
        payload of every output-determining input

    S17.1 (IMP-11): ported AudioGen's S12.3 cache key uplift verbatim
    so MusicGen has the same invariant. Three changes from the prior
    8-char positional implementation:

      1. 12 hex chars (was 8): collision risk at MusicGen scale drops
         to negligible.
      2. JSON-canonical payload: floats serialized via f"{x:.3f}" /
         f"{x:.2f}" so IEEE-754 float-string drift no longer collapses
         or splits keys.
      3. ``model_id`` + ``guidance_scale`` included: switching MusicGen
         models or CFG no longer silently returns the prior cue.

    BUG-LOCAL-017 (Phase D, 2026-05-02): the pre-Phase-D implementation
    appended ``_<timestamp_ms>`` and returned a full filename, forcing
    a cache MISS every run AND violating Rule C7 because FFmpeg
    embeds input WAV filenames in MP4 metadata streams. Lookup
    identity (this function) split from write filename
    (``_cache_filename_for_write``, canonical ``<prefix>.wav``) at
    that point; S17.1 only changes the identity payload, not the
    split.

    Keyword-only: every output-determining input is spelled at the
    call site so a future knob added to MusicGen has to extend the
    signature deliberately.
    """
    payload = json.dumps({
        "cue_id":         str(cue_id),
        "duration_sec":   f"{float(duration_sec):.3f}",
        "prompt":         prompt,
        "episode_seed":   str(episode_seed),
        "model_id":       str(model_id),
        "guidance_scale": f"{float(guidance_scale):.2f}",
    }, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:12]
    return f"{cue_id}_{digest}"


def _cache_filename_for_write(*, cue_id: str, prompt: str,
                              duration_sec: float, episode_seed: str,
                              model_id: str,
                              guidance_scale: float) -> str:
    """Canonical filename for a fresh cache write. Deterministic -- no
    timestamp. C7-safe: same inputs always land at the same filename
    so downstream FFmpeg metadata stays byte-identical run-to-run.

    Keyword-only signature mirrors ``_cache_prefix`` -- positional
    callers raise TypeError, intentional after S17.1.
    """
    prefix = _cache_prefix(
        cue_id=cue_id,
        prompt=prompt,
        duration_sec=duration_sec,
        episode_seed=episode_seed,
        model_id=model_id,
        guidance_scale=guidance_scale,
    )
    return f"{prefix}.wav"


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


def _mood_suffix(script_brief: str) -> str:
    """Mine mood tags from the news script_brief. Returns a comma-prefixed
    suffix (e.g. ``", minor mode, unresolved tension"``) or an empty
    string if no mood keyword matches. Case-insensitive substring match.
    """
    if not script_brief:
        return ""
    low = script_brief.lower()
    tags: list[str] = []
    seen: set[str] = set()
    for keyword, tag in _MOOD_TAGS.items():
        if keyword in low and tag not in seen:
            tags.append(tag)
            seen.add(tag)
    return (", " + ", ".join(tags)) if tags else ""


def _resolve_cue_from_style(cue_id: str, style: str,
                            mood_suffix: str) -> tuple[str, int]:
    """Resolve a single cue from the style palette.

    Returns (prompt, duration_sec). Loud-fails on unknown style slug -
    no default palette, no fallback. The writer's gen_params_initial.style
    is the canonical source; an unknown value indicates a writer-side or
    palette-coverage bug that must be fixed, not papered over.
    """
    palette = _STYLE_PALETTE.get(style)
    if palette is None:
        known = ", ".join(sorted(_STYLE_PALETTE.keys()))
        raise ValueError(
            f"MusicGenTheme: unknown style slug {style!r}. "
            f"Add an entry to _STYLE_PALETTE. Known slugs: {known}"
        )
    base_prompt = palette[cue_id]
    return (
        f"{base_prompt}{mood_suffix}{_PROMPT_TAIL}",
        CUE_DURATIONS[cue_id],
    )


def _silent_audio_dict(sample_rate: int = MUSICGEN_SAMPLE_RATE) -> dict:
    return {
        "waveform": torch.zeros(1, 1, int(sample_rate * 0.1)),
        "sample_rate": sample_rate,
    }


class MusicGenTheme:
    """Instrumental music generator for opening, closing, and act-break
    interstitial cues.

    Reads style + mood signal from the L3 ledger
    (OTR_LedgerFreezeCascade.script_json) and synthesizes three
    deterministic cue prompts via _STYLE_PALETTE + _MOOD_TAGS. No LLM
    call, no legacy Director plan. Generates any cue that isn't already in the
    per-episode cache and returns three AUDIO tensors ready to wire into
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
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": (
                        "L3 ledger JSON from OTR_LedgerFreezeCascade. "
                        "Reads meta.gen_params_initial.style for the "
                        "music palette and meta.news.script_brief for "
                        "mood overlay."
                    ),
                }),
            },
            "optional": {
                "episode_seed": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Cache key component. Leave empty to derive from "
                        "ledger meta.gen_params_initial.seed."
                    ),
                }),
                "model_id": ("STRING", {
                    "default": MUSICGEN_MODEL_ID,
                    "tooltip": "Hugging Face model id. Default is facebook/musicgen-medium (~6 GB VRAM).",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Classifier-free guidance. 3.0 is the MusicGen default.",
                }),
                # C3 (S24, 2026-05-13): default False -- transformers
                # MusicGen ImportError raises RuntimeError so production
                # never silently substitutes silence on the music bus
                # (Directive 1). Opt in for smoke tests where the
                # optional dep isn't installed; silence renders stamp
                # music_render_status="fallback_silence" on each cue's
                # ledger row.
                "allow_silence_fallback": ("BOOLEAN", {"default": False}),
            },
        }

    def render(self, script_json, episode_seed="",
               model_id=MUSICGEN_MODEL_ID, guidance_scale=3.0,
               allow_silence_fallback=False):

        # MANDATORY VRAM POWER WASH (clean slate before start).
        force_vram_offload()

        # S17.4 (IMP-17): defensive coercion at the node boundary.
        # The cache_prefix path calls str(episode_seed); a future
        # caller passing a dict would str() it to a Py-version-stable
        # but fragile representation. Coerce here, once, at the
        # public surface.
        episode_seed = str(episode_seed) if episode_seed is not None else ""

        # ---- L3 ledger reads (single source of truth) ----
        from . import _otr_ledger_consumers as _OTRLC
        led = _OTRLC.load_ledger(script_json)
        meta = led.get("meta", {}) or {}
        gen_params = meta.get("gen_params_initial", {}) or {}

        style = (gen_params.get("style") or "").strip()
        if not style:
            raise ValueError(
                "MusicGenTheme: meta.gen_params_initial.style missing "
                "from ledger. Writer cast-lock contract violation - "
                "every L3 ledger must stamp a style slug."
            )

        news_meta = meta.get("news", {}) or {}
        script_brief = (news_meta.get("script_brief") or "")
        mood_suffix = _mood_suffix(script_brief)

        if not episode_seed:
            seed_from_ledger = gen_params.get("seed")
            if seed_from_ledger is not None:
                episode_seed = str(seed_from_ledger)

        # ---- Resolve all three cues from the style palette ----
        cues: dict[str, dict] = {}
        for cue_id in CUE_IDS:
            prompt, duration = _resolve_cue_from_style(
                cue_id, style, mood_suffix
            )
            cues[cue_id] = {"prompt": prompt, "duration_sec": duration}

        cache_dir = _cache_dir()
        render_log = [
            "=== MusicGen Theme (medium) ===",
            f"cache dir: {cache_dir}",
            f"style: {style}",
            f"mood suffix: {mood_suffix or '<none>'}",
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
                cue_id=cue_id,
                prompt=cue["prompt"],
                duration_sec=cue["duration_sec"],
                episode_seed=episode_seed,
                model_id=model_id,
                guidance_scale=guidance_scale,
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
                    cue_id=cue_id,
                    prompt=cue["prompt"],
                    duration_sec=cue["duration_sec"],
                    episode_seed=episode_seed,
                    model_id=model_id,
                    guidance_scale=guidance_scale,
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
                # C3 (S24, 2026-05-13): strict failure by default,
                # matching AudioGen S17.2 (IMP-19). Silent silence on
                # a production music bus is a Directive 1 breach
                # (audio is king). The allow_silence_fallback widget
                # opts into the prior silence-fill path for smoke
                # tests where transformers/MusicGen isn't installed.
                msg = (
                    f"MusicGen ImportError: transformers MusicGen not "
                    f"available: {exc}. This is a production surface; "
                    f"silent silence is a Directive 1 breach. Install "
                    f"the MusicGen optional deps or set "
                    f"allow_silence_fallback=True for smoke tests only."
                )
                if not allow_silence_fallback:
                    log.error(f"[MusicGenTheme] {msg}")
                    raise RuntimeError(msg) from exc
                log.warning(f"[MusicGenTheme] FALLBACK SILENCE: {msg}")
                render_log.append(
                    f"  WARNING: transformers MusicGen import failed; "
                    f"allow_silence_fallback=True -> silence."
                )
                for cue_id in to_generate:
                    results[cue_id] = _silent_audio_dict()
                    # Tag cue dict so the writeback below stamps
                    # music_render_status="fallback_silence" on the
                    # ledger row (handled in the per-cue post-render
                    # block; this just carries the marker forward).
                    if cue_id in cues:
                        cues[cue_id]["_render_status"] = "fallback_silence"
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
                        # audio_sample_hash on the music row. Historically
                        # the generation_prompt was already populated by
                        # the legacy LLMDirector; post-cleanbreak it comes
                        # from the ledger meta. Either way this closes
                        # the loop on the render-result side.
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
