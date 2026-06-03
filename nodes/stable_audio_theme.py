"""OTR_StableAudioTheme -- model-agnostic music-theme node (Wave 1 / 1c).

The generic theme-music surface for the opt-in v2 audio lane. It emits the three
fixed cues (opening / closing / interstitial) the EpisodeAssembler consumes, and
picks its engine from the shared audio-engine registry, dispatching FAIL-CLOSED:

  * ``musicgen`` (byte-identical legacy default, ``batch`` interface) -> RAW
    verbatim delegation to OTR_MusicGenTheme with the exact upstream
    ``script_json`` string plus the frozen widget tuple from
    ``config/legacy_invocation_manifest.json`` (zero transform; I-1, I-3). The
    legacy node returns the three cue AUDIO tensors, passed through unchanged.
  * ``stable_audio_music`` (opt-in, flag-gated + HF-token-gated, ``clip``
    interface) -> a music-only prompt per cue, a per-cue external seed
    (``_seed_to_int64(music_rng_seed, slot)``, G1), the adapter ``generate_clip``
    call, and ``pack_audio_batch`` into the AUDIO-batch contract (C-4).

Unlike the voice nodes this node has THREE AUDIO outputs and a ``clip`` path, so
it is self-contained rather than built on the voice-node base; it reuses the
shared ``frozen_batch_widgets`` / ``coerce_int_seed`` / ``build_engine_combo``
helpers. Engine libraries are lazy-imported inside ``generate``. Teardown runs in
``finally`` BEFORE the ``done`` signal (I-7). UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import gc
import logging

from ._otr_voice_node_common import build_engine_combo, coerce_int_seed, frozen_batch_widgets

log = logging.getLogger("OTR")

ROLE = "music"
_LEGACY_FIRST_FALLBACK = ("musicgen", "stable_audio_music")

# The three fixed theme cues + their durations (seconds). Durations are part of
# cue identity; keep stable (mirrors the legacy MusicGen cue durations).
_CUE_SLOTS = ("opening", "closing", "interstitial")
_CUE_DURATIONS = {"opening": 12.0, "closing": 8.0, "interstitial": 4.0}
_CUE_CHARACTER = {
    "opening": "slow atmospheric build, sustained instrumental intro",
    "closing": "resolving cadence, gentle decay into silence, instrumental outro",
    "interstitial": "brief textural bridge, short instrumental transition",
}
_MUSIC_TAIL = ", instrumental only, no vocals, no dialogue"


def _music_prompt(meta, slot) -> str:
    """A deterministic music-only cue prompt (<500 chars, no dialogue/names).

    Mines optional mood terms from the ledger meta (``music_mood_terms`` first,
    then ``story_brief_terms.atmosphere``); falls back to a neutral seed.
    """
    mood = []
    if isinstance(meta, dict):
        mm = meta.get("music_mood_terms")
        if isinstance(mm, list):
            mood = [str(t).strip() for t in mm if str(t).strip()][:3]
        if not mood:
            terms = meta.get("story_brief_terms")
            atmo = terms.get("atmosphere") if isinstance(terms, dict) else None
            if isinstance(atmo, list):
                mood = [str(t).strip() for t in atmo if str(t).strip()][:3]
    head = ", ".join(mood) if mood else "atmospheric"
    return f"{head}, {_CUE_CHARACTER[slot]}{_MUSIC_TAIL}"[:480]


def _load_meta(*sources) -> dict:
    """Best-effort ledger meta from the first parseable source. Empty on miss."""
    from . import _otr_ledger_consumers as _OTRLC

    for src in sources:
        if not (src or "").strip():
            continue
        try:
            return _OTRLC.load_ledger(src).get("meta") or {}
        except Exception:  # noqa: BLE001 -- prompts fall back to neutral defaults
            continue
    return {}


class StableAudioTheme:
    """Generic theme-music node. Registered as ``OTR_StableAudioTheme``.

    Engine order: musicgen (legacy byte-identical default) > stable_audio_music.
    """

    CATEGORY = "OldTimeRadio/v2/audio"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO", "AUDIO", "AUDIO", "STRING", "STRING")
    RETURN_NAMES = (
        "opening_theme_audio", "closing_theme_audio", "interstitial_theme_audio",
        "render_log", "done",
    )
    OUTPUT_NODE = False

    @classmethod
    def INPUT_TYPES(cls):
        engines = build_engine_combo(ROLE, _LEGACY_FIRST_FALLBACK)
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "forceInput": True,
                    "tooltip": (
                        "Frozen v2 ledger JSON from OTR_LedgerFreezeCascade. "
                        "Passed VERBATIM to the legacy engine on the batch "
                        "path; read for cue mood on the clip path."
                    ),
                }),
                "engine": (engines, {
                    "default": engines[0],
                    "tooltip": (
                        "Theme-music engine. Legacy MusicGen is the "
                        "byte-identical default; stable_audio_music is opt-in "
                        "(flag + HF token) until the GPU pilot promotes it. "
                        "Unusable selections fail closed with a named error."
                    ),
                }),
            },
            "optional": {
                "ledger_json": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Cast-locked ledger from OTR_CastLock (carries "
                        "episode_seed). Read on the clip path for the per-cue "
                        "seed; the batch path delegates the raw script_json."
                    ),
                }),
                "gate_in": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "forceInput": True,
                    "tooltip": (
                        "Optional ordering signal. Wire an upstream 'done' "
                        "here to force this node to run after it."
                    ),
                }),
                "stereo_policy": (["mono_safe"], {
                    "default": "mono_safe",
                    "tooltip": (
                        "Channel policy for the clip path. 'mono_safe' "
                        "downmixes Stable Audio's native stereo to mono so the "
                        "mono assembly chain stays byte-identical."
                    ),
                }),
            },
        }

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        # D: local-disk only; engine/token/model checks are fail-closed on the
        # generate() path, not here, so a box-fresh graph validates clean.
        return True

    # ------------------------------------------------------------------ #
    def generate(self, script_json, engine, ledger_json="", gate_in="",
                 stereo_policy="mono_safe"):
        from ._otr_audio_engines import (
            EngineUnusable, EngineUsabilityReason, assert_usable, get_engine,
        )

        render_log: list = []
        cues = None
        adapter = None
        n_cues = 0
        try:
            engine = assert_usable(engine, ROLE)  # FAIL CLOSED (6-class)
            adapter = get_engine(engine)
            interface = getattr(adapter, "interface", "clip")

            if interface == "batch":
                cues, line = self._delegate_batch(adapter, engine, script_json)
                render_log.append(line)
            elif interface == "clip":
                cues, lines = self._render_clips(
                    adapter, engine, script_json, ledger_json, stereo_policy,
                )
                render_log.extend(lines)
            else:
                raise EngineUnusable(
                    engine, ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                    f"engine {engine!r} declares unsupported interface "
                    f"{interface!r} for role {ROLE!r}",
                )
            n_cues = len(_CUE_SLOTS)
        finally:
            # I-7: free single-engine residency in finally, BEFORE 'done'.
            self._teardown(adapter)

        if cues is None:
            from ._otr_resolved_request import empty_audio_batch

            sr = int(getattr(adapter, "sample_rate", 44100) or 44100)
            cues = {slot: empty_audio_batch(sr) for slot in _CUE_SLOTS}
        done = f"music:done:engine={engine}:cues={int(n_cues)}"
        return (
            cues["opening"], cues["closing"], cues["interstitial"],
            "\n".join(render_log), done,
        )

    # ------------------------------------------------------------------ #
    def _delegate_batch(self, adapter, engine, script_json):
        """Raw, byte-identical delegation to the legacy MusicGen theme (I-3).

        The legacy node is handed the exact ``script_json`` string + the frozen
        manifest widget tuple. It returns the three cue AUDIO tensors (slots
        0/1/2), passed through unchanged after a contract assert.
        """
        from ._otr_audio_engines import EngineUnusable, EngineUsabilityReason
        from ._otr_resolved_request import assert_audio_batch_contract

        node = adapter.make_batch_node()
        func_name = getattr(node, "FUNCTION", None)
        fn = getattr(node, func_name, None) if func_name else None
        if not callable(fn):
            raise EngineUnusable(
                engine, ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                f"legacy theme node {type(node).__name__!r} has no callable "
                f"FUNCTION {func_name!r}",
            )
        widgets = frozen_batch_widgets(node, ROLE)
        result = fn(script_json, *widgets)
        if not isinstance(result, tuple) or len(result) < 3:
            raise EngineUnusable(
                engine, ROLE, EngineUsabilityReason.MALFORMED_CONFIG,
                f"legacy theme node returned {type(result).__name__} with "
                f"fewer than 3 AUDIO outputs",
            )
        cues = {}
        for i, slot in enumerate(_CUE_SLOTS):
            cues[slot] = assert_audio_batch_contract(
                result[i], where=f"StableAudioTheme.delegate.{slot}",
            )
        line = (
            f"music: delegated to legacy '{engine}' (batch, verbatim "
            f"script_json + {len(widgets)} frozen widgets, 3 cues)"
        )
        return cues, line

    # ------------------------------------------------------------------ #
    def _render_clips(self, adapter, engine, script_json, ledger_json,
                      stereo_policy):
        """Per-cue clip generation for stable_audio_music (clip interface)."""
        from ._otr_audio_engines import pack_audio_batch
        from ._otr_engine_profiles import (
            assert_model_available, assert_token_for_profile, require_resolver,
        )
        from ._otr_determinism import deterministic_inference
        from ._otr_resolved_request import _seed_to_int64

        resolver = require_resolver()
        profile = resolver.resolve_casting_plan(role=ROLE, engine=engine)
        assert_token_for_profile(profile)   # MISSING_HF_TOKEN if HF-gated
        assert_model_available(profile)
        sr = int(profile.sample_rate or getattr(adapter, "sample_rate", 44100))
        mono = (stereo_policy == "mono_safe")

        meta = _load_meta(ledger_json, script_json)
        music_seed_base = _seed_to_int64(
            "music_rng_seed_v1", coerce_int_seed(meta.get("episode_seed")),
        )

        cues = {}
        log_lines = [
            f"music: rendering 3 cues on '{engine}' (profile {profile.profile_id})"
        ]
        for slot in _CUE_SLOTS:
            prompt = _music_prompt(meta, slot)
            duration_s = _CUE_DURATIONS[slot]
            engine_seed = _seed_to_int64(music_seed_base, slot)  # G1 theme seed
            # G1: scope determinism + seed/restore around the single forward
            # (non-strict; bit_exact is gated on the F pilot -- see voice path).
            with deterministic_inference(engine_seed, warn_only=True):
                clip = adapter.generate_clip(prompt, duration_s, engine_seed)
            cues[slot] = pack_audio_batch([clip], sample_rate=sr, mono=mono)
            log_lines.append(f"  [{slot}] {duration_s:.0f}s at {sr} Hz")
        return cues, log_lines

    # ------------------------------------------------------------------ #
    @staticmethod
    def _teardown(adapter):
        """I-7 teardown: unload the engine + free VRAM. Best-effort."""
        try:
            if adapter is not None and hasattr(adapter, "unload"):
                adapter.unload()
        except Exception:  # noqa: BLE001
            pass
        try:
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
