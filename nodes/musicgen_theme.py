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

from . import _otr_ledger as _OTRL_PATHS
# Path F (2026-05-18): _STYLE_PALETTE retired from this module.
# Cue prompts are now composed from meta brief fields directly
# (see _compose_music_prompt). KNOWN_STYLE_SLUGS still imported by
# the freeze cascade for writer-slug drift validation; that import
# lives there, not here. Per Jeffrey no-legacy-back-compat:
# delete the dead import rather than keep as a shim.
from ._otr_story_brief_helpers import (
    get_story_brief_music_mood,
    get_story_brief_status,
)
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

# Music cue prompts now live in nodes/_otr_style_palette.py (S25/MG-6,
# BUG-LOCAL-216). Imported above as _STYLE_PALETTE + KNOWN_STYLE_SLUGS;
# both names preserved here so existing call sites at _resolve_cue_from_style
# need no edit. tests/test_style_palette_drift.py pins the writer pool +
# palette + freeze validator to set-equality with the single source of truth.

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
    """Find a cached WAV for ``prefix``. Single-tier canonical lookup:
    ``<prefix>.wav``. Returns the absolute path on hit, None on miss.
    Never raises.
    """
    from pathlib import Path

    canonical = Path(cache_dir) / f"{prefix}.wav"
    if canonical.is_file():
        return str(canonical)
    return None


def _load_cached_wav(path: str) -> tuple[torch.Tensor, int] | None:
    """Load a cached .wav as ((1, 1, T) float tensor, sr), or None if missing."""
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


def _save_wav(path: str, waveform: np.ndarray, sample_rate: int) -> bool:
    """Atomic WAV write. Returns True on confirmed write, False on any
    exception. Callers gate ledger wav_path stamping on this return.

    Write to a sibling ``.tmp`` then ``os.replace`` into the canonical
    path. Prevents corrupted cache hits if the process is killed
    mid-write (Phase D consult, Gemini's catch). Explicit
    ``format='WAV'`` is required because soundfile cannot infer the
    audio format from the ``.tmp`` extension.

    S25/MG-1 (BUG-LOCAL-211) -- parity with AudioGen S24/C2: prior
    contract was ``-> None``, so callers couldn't distinguish a
    confirmed write from a swallowed exception. The writeback path
    keyed on the (truthy) cache_path string instead of the save's
    outcome, which could leave a ledger row pointing at a path that
    was never written.
    """
    tmp_path = path + ".tmp"
    try:
        import soundfile as sf
        sf.write(tmp_path, waveform, sample_rate,
                 subtype="FLOAT", format="WAV")
        os.replace(tmp_path, path)
        return True
    except Exception as exc:
        log.warning("[MusicGenTheme] Failed to write cache %s: %s", path, exc)
        # Best-effort cleanup of tmp on failure
        try:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        except Exception:
            pass
        return False


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


# _apply_story_brief_mood_prefix retired 2026-05-18 (Path F).
# Mood prefix is now integrated into the prompt body by
# _compose_music_prompt directly (it pulls mood terms from
# atmosphere FIRST, with a script_brief keyword-mining fallback).
# Per Jeffrey no-legacy-back-compat: dead function deleted, not
# kept as a shim.


# Path F (2026-05-18): cue-specific musical character templates.
# Each cue gets a fixed template describing its musical shape
# (intro / outro / bridge). The narrative content -- mood, setting,
# theme -- is supplied by the meta brief at render time.
#
# These templates replace the per-slug entries in _STYLE_PALETTE.
# The old palette had 10 slugs x 3 cues = 30 prompts that needed to
# stay in lockstep with the style_picker's invent path; the invent
# path explicitly produces NEW slugs that have no palette entry
# ("let the story decide" design intent per memory entry
# reference_style_auto_sentinel_label). Pulling music prompts
# directly from the meta brief makes that collision structurally
# impossible.
_CUE_CHARACTER: dict[str, str] = {
    "opening":      "slow atmospheric build, introduces the scene, "
                    "ends on a sustained chord, instrumental intro",
    "closing":      "resolving cadence, gentle decay into silence, "
                    "instrumental outro",
    "interstitial": "brief textural bridge, unresolved, short "
                    "instrumental transition",
}


def _compose_music_prompt(
    meta: dict, cue_id: str,
) -> tuple[str, int]:
    """Synthesize a MusicGen cue prompt from the meta brief.

    Pulls from L3 ledger meta (already passed via script_json
    linked input to node 14):
      - meta.story_brief_terms.atmosphere  -> mood adjectives
      - meta.story_brief_terms.setting     -> scene context
      - meta.news.script_brief             -> keyword-mined mood
        tags (existing _MOOD_TAGS helper)
      - meta.gen_params_initial            -> period_voice (if
        present)

    Combines with the cue-specific musical character template
    (opening / closing / interstitial) and the prompt tail
    ("instrumental only, no vocals").

    Returns (prompt, duration_sec). Never raises on unknown slug
    -- there is no slug lookup. Falls back to a neutral
    atmospheric prompt when meta is absent or sparse.
    """
    terms = (meta.get("story_brief_terms") or {}) if isinstance(meta, dict) else {}
    if not isinstance(terms, dict):
        terms = {}

    atmosphere_raw = terms.get("atmosphere") or []
    setting_raw = terms.get("setting") or []
    if not isinstance(atmosphere_raw, list):
        atmosphere_raw = []
    if not isinstance(setting_raw, list):
        setting_raw = []

    atmosphere = [str(t).strip() for t in atmosphere_raw if str(t).strip()]
    setting_terms = [str(t).strip() for t in setting_raw if str(t).strip()]

    # Mood: prefer atmosphere terms from the reflection pass. Top 3
    # to keep the prompt under MusicGen's effective context window.
    # If the reflection pass produced nothing, fall back to keyword
    # mining of news.script_brief via the legacy _MOOD_TAGS helper
    # (we already had that signal path; reusing it here keeps the
    # fallback graceful when story_brief_status='absent' on legacy
    # ledgers).
    if atmosphere:
        mood_terms = atmosphere[:3]
    else:
        news_meta = (meta.get("news") or {}) if isinstance(meta, dict) else {}
        if not isinstance(news_meta, dict):
            news_meta = {}
        keyword_suffix = _mood_suffix(
            news_meta.get("script_brief") or ""
        )
        # _mood_suffix returns ", tag1, tag2" or ""; strip and split.
        kw_str = keyword_suffix.lstrip(", ").strip()
        mood_terms = [t.strip() for t in kw_str.split(",") if t.strip()] if kw_str else []

    # Setting: top 2 terms keeps the prompt readable.
    setting_str = ", ".join(setting_terms[:2]) if setting_terms else ""

    # Period overlay: the writer's gen_params_initial may carry a
    # period_voice descriptor when the talkie 1940s row is active.
    # Layer it into the prompt as a register cue so MusicGen leans
    # toward period instrumentation (brass / strings / etc.) rather
    # than a modern synth bed. Modern profile = no descriptor = no
    # overlay.
    gen_params = meta.get("gen_params_initial") if isinstance(meta, dict) else None
    period_descriptor = ""
    if isinstance(gen_params, dict):
        pv = gen_params.get("period_voice")
        if isinstance(pv, dict):
            descriptor = pv.get("descriptor") or pv.get("music_descriptor")
            if isinstance(descriptor, str) and descriptor.strip():
                period_descriptor = descriptor.strip()

    # Compose: mood -> setting -> period -> cue character -> tail.
    parts: list[str] = []
    if mood_terms:
        parts.append(", ".join(mood_terms))
    else:
        # Graceful default when both atmosphere AND script_brief are
        # empty. "atmospheric" is the safest one-word seed for
        # MusicGen and matches the closing fallback in the legacy
        # path.
        parts.append("atmospheric")
    if setting_str:
        parts.append(f"evokes {setting_str}")
    if period_descriptor:
        parts.append(period_descriptor)
    parts.append(_CUE_CHARACTER[cue_id])
    prompt = ", ".join(parts) + _PROMPT_TAIL
    return prompt, CUE_DURATIONS[cue_id]


def _silent_audio_dict(duration_sec: float,
                       sample_rate: int = MUSICGEN_SAMPLE_RATE) -> dict:
    """S25/MG-4 (BUG-LOCAL-214): honor caller-supplied duration so
    silence fallbacks don't break the EpisodeAssembler timeline.
    Prior implementation always emitted 0.1s regardless of cue, so the
    12s opening + 8s closing cues both got a 100 ms silence on the
    ImportError fallback path."""
    samples = int(sample_rate * float(duration_sec))
    return {
        "waveform": torch.zeros(1, 1, samples),
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
    # S30 B6 opt-in: this `model_id` widget picks a MusicGen
    # checkpoint (facebook/musicgen-small etc.), not an LLM. The
    # two-slot LLM rule does not apply.
    NON_LLM_MODEL_WIDGET_OK = True
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

        # Path F (2026-05-18): style slug is no longer the music
        # prompt source. Read it ONLY for diagnostic logging; the
        # cue prompts are composed from the meta brief directly via
        # _compose_music_prompt. Missing style is no longer a fatal
        # error -- MusicGen synthesizes from atmosphere + setting
        # terms regardless of what the writer stamped.
        style_for_log = (gen_params.get("style") or "").strip() or "<absent>"

        if not episode_seed:
            seed_from_ledger = gen_params.get("seed")
            if seed_from_ledger is not None:
                episode_seed = str(seed_from_ledger)

        # ---- Compose all three cue prompts from the meta brief ----
        # Sprint H 3.7 Path F (2026-05-18): _compose_music_prompt
        # replaces _resolve_cue_from_style + _apply_story_brief_mood_prefix
        # as a single meta-brief-driven synthesis. Per Jeffrey:
        # "MusicGen is downstream of story gen. The meta brief / story
        # script is the canonical source. Style slug was a thin
        # abstraction." Removes the structural collision with the
        # style_picker's let-the-story-decide invent path.
        cues: dict[str, dict] = {}
        _brief_status_logged = False
        for cue_id in CUE_IDS:
            prompt, duration = _compose_music_prompt(meta, cue_id)
            if not _brief_status_logged:
                _brief_status = get_story_brief_status(meta)
                _brief_mood_terms = get_story_brief_music_mood(meta)
                log.info(
                    "[OTR_MusicGenTheme] story_brief_status=%s "
                    "mood_terms=%s style_slug_diag=%s",
                    _brief_status, _brief_mood_terms, style_for_log,
                )
                _brief_status_logged = True
            cues[cue_id] = {"prompt": prompt, "duration_sec": duration}

        cache_dir = _cache_dir()
        render_log = [
            "=== MusicGen Theme (medium) ===",
            f"cache dir: {cache_dir}",
            f"style (diagnostic only): {style_for_log}",
            f"episode seed: {episode_seed or '<none>'}",
        ]

        # S25/AG-1+MG-7 (BUG-LOCAL-220): _fallback/ is ephemeral by
        # design; wipe stale entries from prior runs of the same
        # episode. C2's _fallback/ redirect (short-output cache
        # poisoning guard) shipped without a cleanup hook so the dir
        # accumulated orphan silence wavs across iterations.
        _fb_dir = os.path.join(cache_dir, "_fallback")
        if os.path.isdir(_fb_dir):
            _removed = 0
            for _f in os.listdir(_fb_dir):
                if _f.lower().endswith(".wav"):
                    try:
                        os.unlink(os.path.join(_fb_dir, _f))
                        _removed += 1
                    except OSError:
                        pass
            if _removed:
                render_log.append(
                    f"_fallback/ cleanup: removed {_removed} stale wav(s)"
                )

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
                    # S25/MG-2 (BUG-LOCAL-212): mark this cue as a
                    # confirmed cache hit so the writeback block can
                    # gate wav_path on either fresh-save success OR
                    # pre-existing cache presence (matching AudioGen
                    # S24/C2 pattern).
                    cue["_had_cache_hit"] = True
                    cue["_render_status"] = "ok_cache"
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
                # Sprint E E5 / M9: name the install command explicitly.
                # Pre-E5 the message said "install the MusicGen optional
                # deps" which gave the operator no actionable handle on
                # WHICH deps. The transformers package handles MusicGen
                # natively; the audiocraft extra is required for some
                # alternate generation paths and pinned weights.
                msg = (
                    f"MusicGen ImportError: transformers MusicGen not "
                    f"available: {exc}. This is a production surface; "
                    f"silent silence is a Directive 1 breach. Install "
                    f"with: pip install transformers audiocraft "
                    f"(or set allow_silence_fallback=True for smoke "
                    f"tests only)."
                )
                if not allow_silence_fallback:
                    log.error(f"[MusicGenTheme] {msg}")
                    raise RuntimeError(msg) from exc
                log.warning(f"[MusicGenTheme] FALLBACK SILENCE: {msg}")
                render_log.append(
                    f"  WARNING: transformers MusicGen import failed; "
                    f"allow_silence_fallback=True -> silence."
                )
                # S25/MG-3 (BUG-LOCAL-213) + MG-4: stamp render_status
                # + save_ok on each silenced cue and emit a
                # correct-duration silence wav so the writeback block
                # below sees a consistent shape and the EpisodeAssembler
                # timeline doesn't break. Prior implementation emitted a
                # 100 ms silence regardless of cue and returned early,
                # bypassing the writeback block entirely so
                # music_render_status was never written.
                for cue_id in to_generate:
                    cue_dur = CUE_DURATIONS.get(cue_id, 0)
                    results[cue_id] = _silent_audio_dict(duration_sec=cue_dur)
                    if cue_id in cues:
                        cues[cue_id]["_render_status"] = "fallback_silence"
                        cues[cue_id]["_save_ok"] = False
                # Intentional: NO early return here. The writeback
                # block at the end of render() now sees the cues + the
                # status stamps and routes wav_path / music_render_status
                # accordingly. Same exit path as the success branch.

            else:
                # S25/MG-3 (BUG-LOCAL-213): the original control flow
                # let the fallback-silence path fall through into the
                # model-loading code (NameError on undefined classes).
                # Wrapping the load + generate block in `else:` ensures
                # model loading only runs when the transformers import
                # succeeded; the fallback branch above falls through
                # straight to the writeback block at the bottom.
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

                        # S25/MG-7 (BUG-LOCAL-220): robust shape extraction
                        # ported from AudioGen S24/C2. transformers' MusicGen
                        # output shape varies across versions:
                        #   - [batch, channels, samples]   (older, decoded)
                        #   - [batch, samples]             (newer, decoded mono)
                        #   - [batch, num_codebooks, seq]  (token IDs - bug)
                        # The legacy audio_values[0, 0] on a token-ID tensor
                        # returned the codebook 0 token sequence (~150 ints),
                        # producing 3 ms WAVs. New logic: pick the float
                        # slice that matches the expected shape; fall back
                        # to flatten if all else fails.
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

                        # S25/MG-7: short-output sanity. Anything under
                        # 0.25s is an output-shape bug or generation
                        # failure; fall back to canonical-duration
                        # silence so the timeline still has a plausible
                        # slot, AND route the save to _fallback/ so the
                        # canonical cache_path isn't poisoned with the
                        # silence stand-in (a subsequent transformers
                        # fix can re-generate cleanly to the canonical
                        # path on the next run).
                        _min_samples = int(MUSICGEN_SAMPLE_RATE * 0.25)
                        short_output_fallback = audio_np.size < _min_samples
                        if short_output_fallback:
                            log.warning(
                                "[MusicGenTheme] [%s] generated audio too "
                                "short (%d samples = %.4fs) -- expected "
                                ">= %.2fs. Output shape was %s. "
                                "Falling back to silence at %.2fs. "
                                "(BUG-LOCAL-220 / transformers MusicGen "
                                "output-shape regression.)",
                                cue_id, audio_np.size,
                                audio_np.size / MUSICGEN_SAMPLE_RATE,
                                _min_samples / MUSICGEN_SAMPLE_RATE,
                                tuple(audio_values.shape) if hasattr(audio_values, "shape") else "?",
                                duration,
                            )
                            audio_np = np.zeros(
                                int(MUSICGEN_SAMPLE_RATE * duration),
                                dtype=np.float32,
                            )

                        # Peak normalize to -1 dBFS so cues sit at consistent level.
                        peak = float(np.max(np.abs(audio_np))) or 1.0
                        audio_np = (audio_np / peak * 0.89).astype(np.float32)

                        if short_output_fallback:
                            fb_dir = os.path.join(
                                os.path.dirname(cue["cache_path"]),
                                "_fallback",
                            )
                            try:
                                os.makedirs(fb_dir, exist_ok=True)
                            except Exception:
                                fb_dir = None
                            if fb_dir is not None:
                                fb_path = os.path.join(
                                    fb_dir,
                                    os.path.basename(cue["cache_path"]),
                                )
                                _save_wav(
                                    fb_path, audio_np, MUSICGEN_SAMPLE_RATE,
                                )
                                cue["_fallback_path"] = fb_path
                            cue["_render_status"] = "fallback_output_shape"
                            cue["_save_ok"] = False  # canonical not written
                        else:
                            # S25/MG-2 (BUG-LOCAL-212): capture the save
                            # outcome so the writeback block can gate
                            # wav_path on confirmed disk presence.
                            save_ok = _save_wav(
                                cue["cache_path"], audio_np, MUSICGEN_SAMPLE_RATE
                            )
                            cue["_save_ok"] = bool(save_ok)
                            cue["_render_status"] = "ok" if save_ok else "error"

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
                            f"(render_ms={_mg_render_ms}, "
                            f"status={cue.get('_render_status', '?')})"
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
        # Find the in-flight Ledger singleton's on-disk path and stamp
        # each cue's wav_path + dur_s + music_render_status into
        # ledger.music[]. Silent skip when no in-flight ledger exists
        # (rare; usually ScriptWriter has run by the time MusicGen
        # does). Errors logged but never crash.
        #
        # S25/MG-8 (BUG-LOCAL-211..213 sibling): swap the legacy
        # find_most_recent_ledger / write_text path for the safe
        # helpers in _otr_ledger.py:
        #   - in_flight_ledger_path()  -- BUG-LOCAL-021 contract
        #   - load_ledger_safe()       -- skip-on-error contract
        #   - save_ledger_safe()       -- atomic + path-stamping
        try:
            # S25/MG-2 + MG-3: writeback gates and music_render_status
            # ledger_path is the in-flight singleton (None when no
            # ledger has been registered yet -- usually means the
            # writer hasn't run, in which case there's nothing to
            # stamp).
            ledger_path = _OTRL_PATHS.in_flight_ledger_path()
            if ledger_path is not None:
                led = _OTRL_PATHS.load_ledger_safe(ledger_path)
                if led is None:
                    log.warning(
                        "[MusicGenTheme] BUG-095 ledger load returned "
                        "None for %s; skipping wav_path writeback",
                        ledger_path.name,
                    )
                else:
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
                            # S25/MG-2 (BUG-LOCAL-212): gate wav_path
                            # stamping on confirmed disk presence.
                            # Prior code stamped str(cache_path)
                            # unconditionally, leaving ghost-path rows
                            # on save failure / silence fallback.
                            save_ok = bool(cue_dict.get("_save_ok"))
                            had_cache_hit = bool(cue_dict.get("_had_cache_hit"))
                            if (cache_path
                                    and (save_ok or had_cache_hit)
                                    and os.path.isfile(cache_path)):
                                row["wav_path"] = str(cache_path)
                            else:
                                row["wav_path"] = ""
                            # S25/MG-3 (BUG-LOCAL-213): always stamp the
                            # render status -- "ok" / "ok_cache" /
                            # "error" / "fallback_silence" /
                            # "fallback_output_shape". Previously the
                            # docstring promised stamping but no path
                            # ever wrote it (the ImportError branch
                            # returned early; the success branch never
                            # stamped). Default to "ok" only when
                            # nothing else set it (defensive).
                            row["music_render_status"] = str(
                                cue_dict.get("_render_status") or "ok"
                            )
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
                        # S25/AG-7 (BUG-LOCAL-219): soft-mode audit
                        # walker. Surfaces §6.16 violations to
                        # render_log; the consumer stays non-halting
                        # until the per-consumer strict flip lands
                        # after the walker holds clean for two full
                        # pipeline runs (post-S25 soak).
                        violations = _OTRLC.audit_post_freeze_writeback(
                            led, strict=False,
                        )
                        if violations:
                            render_log.append(
                                f"§6.16 audit: {len(violations)} violation(s)"
                            )
                            for v in violations[:5]:
                                render_log.append(f"  {v}")
                            if len(violations) > 5:
                                render_log.append(
                                    f"  ... +{len(violations) - 5} more "
                                    "(see audit_post_freeze_writeback)"
                                )
                        _OTRL_PATHS.save_ledger_safe(ledger_path, led)
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


# S25/MG-5 (BUG-LOCAL-215): in-module mapping aligned to the canonical
# OTR_ prefix that the top-level __init__.py registers under. The
# repo's actual ComfyUI registration is __init__.py:_NODE_MODULES,
# which already keys this class as "OTR_MusicGenTheme"; this in-module
# dict was inconsistent and would break any test harness that imported
# from nodes.musicgen_theme directly. Also dropped the literal
# "[EMOJI]" placeholder string from the display name -- minimal-emoji
# convention.
NODE_CLASS_MAPPINGS = {"OTR_MusicGenTheme": MusicGenTheme}
NODE_DISPLAY_NAME_MAPPINGS = {"OTR_MusicGenTheme": "MusicGen Theme"}
