"""Bark voice adapter -- self-contained per_line (clean-break 1a).

Serves both char_voice and announcer_voice (2026-08-24) -- the same v2/*
preset mechanism either way; the caller's ``role`` (threaded onto the
adapter instance by the dispatch core) only selects which curated profile
(``char_bark_v1`` / ``announcer_bark_v1``) supplies the per-stage temps.

Sources inference from _otr_bark_lib (relocated, delegation-free); no
construction of the heavy batch node. interface == "per_line". Library imports
are lazy so importing the registry package stays light (C-5).

Gate 3 (voice-path-cleanbreak): an empty / non-v2/* voice_preset is a writer
cast-lock contract violation -- generate_voice fails closed with a named
EngineUnusable(MALFORMED_CONFIG), the same renderability net the legacy batch
node enforced before it was retired.

text_temp is the char_bark_v1 profile default (config/audio_engine_profiles.yaml,
the curated-params SSOT, plan D5). tests/test_bark_legacy_node_retired.py pins
text_temp == that profile value so the hardcoded constant cannot silently drift
from the profile. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

from .registry import register


@register
class BarkEngine:
    name = "bark"
    roles = ("char_voice", "announcer_voice")
    default_roles = ()                   # DEMOTED 2026-06-04: indextts2 is now the char_voice default; bark stays selectable
    commercial_clean = False             # Suno Bark terms not confirmed commercial
    requires_flag = None
    interface = "per_line"
    sample_rate = 24000
    supports_external_generator = False  # Bark.generate binds no external Generator
    voice_ref_field = "voice_preset"     # dispatch routes cast.voice_preset to the ref slot
    text_temp = 0.7                      # == char_bark_v1 profile default_params.text_temp (legacy alias)
    # Per-stage temperatures (2026-06-17 whiny-voice fix). These class attrs are
    # the FALLBACK baseline when the char_bark_v1 profile is missing/malformed;
    # the live values come from the profile via _resolve_stage_temps (which honors
    # an explicit 0.0 and accepts the text_temp/waveform_temp aliases). semantic
    # stays warm (0.7) for content commitment; coarse/fine drop to 0.5 to firm up
    # the acoustic stages (the thin/whiny timbre came from over-hot acoustics).
    semantic_temp = 0.7
    coarse_temp = 0.5
    fine_temp = 0.5

    def __init__(self):
        self._loaded = False
        self._presets_started = set()    # first-line anti-hallucination guard tracking

    def load(self):
        if self._loaded:
            return
        from .._otr_bark_lib import _load_bark

        _load_bark("suno/bark")
        self._loaded = True

    def unload(self):
        self._loaded = False
        self._presets_started = set()    # reset so the next episode re-guards first lines
        try:
            from .._otr_bark_lib import _unload_bark

            _unload_bark()
        except Exception:  # noqa: BLE001 -- teardown must never raise
            pass

    def prepare_text(self, text, delivery_vector=None):
        from .._otr_bark_lib import _clean_text_for_bark

        return _clean_text_for_bark(text)

    def _resolve_stage_temps(self):
        """(semantic, coarse, fine) temps from the char_bark_v1 /
        announcer_bark_v1 profile (role-dependent -- ``self.role`` is set by
        the dispatch core before ``generate_voice`` runs), with the
        BarkEngine class attrs as the fail-soft fallback.

        Alias ladder (precise -- a key is used only if PRESENT and NOT None, so an
        explicit ``0.0`` is honored, unlike ``or``):
          semantic <- semantic_temp, then text_temp, then class attr (0.7)
          coarse   <- coarse_temp,   then waveform_temp, then class attr (0.5)
          fine     <- fine_temp,     then waveform_temp, then class attr (0.5)
        A missing/malformed profile yields ``{}`` -> the class attrs stand in.
        """
        params: dict = {}
        try:
            from .._otr_engine_profiles import load_resolver

            resolver = load_resolver()
            if resolver is not None:
                prof = resolver.profile_for(
                    getattr(self, "role", "char_voice"), self.name)
                if prof is not None:
                    params = dict(prof.default_params or {})
        except Exception:  # noqa: BLE001 -- profile read must never break a render
            params = {}

        def pick(*keys, default):
            for k in keys:
                if k in params and params[k] is not None:
                    return float(params[k])
            return float(default)

        semantic = pick("semantic_temp", "text_temp", default=self.semantic_temp)
        coarse = pick("coarse_temp", "waveform_temp", default=self.coarse_temp)
        fine = pick("fine_temp", "waveform_temp", default=self.fine_temp)
        return semantic, coarse, fine

    def generate_voice(self, text, voice_preset, delivery_vector, seed):
        """One character line -> mono AUDIO {"waveform":[1,1,T], "sample_rate"}.

        voice_preset (e.g. "v2/en_speaker_3") arrives via voice_ref_field (the
        dispatch routes cast.voice_preset into the positional ref slot). Runs
        inside the caller's deterministic_inference wrap; Bark binds no external
        Generator. Preserves the [clears throat] first-line guard per preset so
        the per-line path keeps the same anti-hallucination behavior the grouped
        batch path had (guard fires on the first occurrence of each preset).
        """
        import numpy as np
        import torch

        from .._otr_bark_lib import (
            _generate_single_line,
            _load_bark,
            _resolve_bark_inject_anchor,
            _resolve_bark_speech_only,
        )
        from .registry import EngineUnusable, EngineUsabilityReason

        if not voice_preset or not str(voice_preset).startswith("v2/"):
            raise EngineUnusable(
                self.name, getattr(self, "role", "char_voice"),
                EngineUsabilityReason.MALFORMED_CONFIG,
                f"bark requires a v2/* voice_preset; got {voice_preset!r}",
            )
        model, processor = _load_bark("suno/bark")
        self._loaded = True
        is_first = voice_preset not in self._presets_started
        self._presets_started.add(voice_preset)
        semantic_temp, coarse_temp, fine_temp = self._resolve_stage_temps()
        # B1 (2026-06-22): bark is the char_voice engine -> every line is
        # DIALOGUE, so render in speech-only mode (drop the squeal tokens) and
        # skip the first-line [clears throat] anchor by default. Both are
        # explicit kwargs resolved from env (OTR_BARK_SPEECH_ONLY=1 default,
        # OTR_BARK_DISABLE_THROAT_CLEAR=1 default) -- no implicit defaults.
        speech_only = _resolve_bark_speech_only()
        inject_first_line_anchor = _resolve_bark_inject_anchor()
        audio_np, sr = _generate_single_line(
            text, voice_preset, model, processor, is_first_line=is_first,
            semantic_temp=semantic_temp, coarse_temp=coarse_temp,
            fine_temp=fine_temp,
            inject_first_line_anchor=inject_first_line_anchor,
            speech_only=speech_only,
            # B2: thread the EXISTING per-line seed (was dropped before) so the
            # clip is reproducible (Bark.generate is unseeded otherwise).
            seed=seed,
        )
        wav = torch.from_numpy(
            np.asarray(audio_np, dtype=np.float32)
        ).reshape(1, 1, -1)
        return {"waveform": wav, "sample_rate": int(sr)}
