"""Bark character-voice adapter -- self-contained per_line (clean-break 1a).

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
    roles = ("char_voice",)
    default_roles = ("char_voice",)      # internal default until promotion (I)
    commercial_clean = False             # Suno Bark terms not confirmed commercial
    requires_flag = None
    interface = "per_line"
    sample_rate = 24000
    supports_external_generator = False  # Bark.generate binds no external Generator
    voice_ref_field = "voice_preset"     # dispatch routes cast.voice_preset to the ref slot
    text_temp = 0.7                      # == char_bark_v1 profile default_params.text_temp

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

        from .._otr_bark_lib import _generate_single_line, _load_bark
        from .registry import EngineUnusable, EngineUsabilityReason

        if not voice_preset or not str(voice_preset).startswith("v2/"):
            raise EngineUnusable(
                self.name, "char_voice", EngineUsabilityReason.MALFORMED_CONFIG,
                f"bark requires a v2/* voice_preset; got {voice_preset!r}",
            )
        model, processor = _load_bark("suno/bark")
        self._loaded = True
        is_first = voice_preset not in self._presets_started
        self._presets_started.add(voice_preset)
        audio_np, sr = _generate_single_line(
            text, voice_preset, model, processor,
            temperature=self.text_temp, is_first_line=is_first,
        )
        wav = torch.from_numpy(
            np.asarray(audio_np, dtype=np.float32)
        ).reshape(1, 1, -1)
        return {"waveform": wav, "sample_rate": int(sr)}
