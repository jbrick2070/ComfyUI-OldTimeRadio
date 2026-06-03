"""OTR_StableAudioTheme -- model-agnostic music-theme node (Wave 1 / 1c).

The generic theme-music surface for the opt-in v2 audio lane. It emits the three
fixed cues (opening / closing / interstitial) the EpisodeAssembler consumes, and
picks its engine from the shared audio-engine registry, dispatching FAIL-CLOSED:

Every music engine is a self-contained ``clip`` engine (the legacy
batch-delegation path was retired in the audio clean-break, 1c):

  * ``musicgen`` (default, ``clip``) -> a per-cue prompt from the Meta brief
    (``_otr_music_prompt.compose_music_prompt``), a per-cue external seed
    (``_seed_to_int64(music_rng_seed, slot)``, G1), the adapter ``generate_clip``
    call, and ``pack_audio_batch`` into the AUDIO-batch contract (C-4).
  * ``stable_audio_music`` (opt-in, flag-gated + HF-token-gated, ``clip``) -> the
    same clip path; native stereo downmixed to mono while the assembly chain is
    mono.

Unlike the voice nodes this node has THREE AUDIO outputs, so it is self-contained
rather than built on the voice-node base; it reuses the shared
``coerce_int_seed`` / ``build_engine_combo`` helpers plus the
``compose_music_prompt`` composer. Engine libraries are lazy-imported inside
``generate``. Teardown runs in ``finally`` BEFORE the ``done`` signal (I-7).
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import gc
import logging

from ._otr_voice_node_common import build_engine_combo, coerce_int_seed
from ._otr_music_prompt import compose_music_prompt

log = logging.getLogger("OTR")

ROLE = "music"
_LEGACY_FIRST_FALLBACK = ("musicgen", "stable_audio_music")

# The three fixed theme cues. Durations + the per-cue prompt are composed by the
# single-source Meta-brief composer (nodes/_otr_music_prompt.compose_music_prompt),
# so music pulls period/setting/mood from the same propagating creative brief as
# every other downstream creative call -- not a local template.
_CUE_SLOTS = ("opening", "closing", "interstitial")


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

            if interface == "clip":
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
            prompt, duration_s = compose_music_prompt(meta, slot)
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
