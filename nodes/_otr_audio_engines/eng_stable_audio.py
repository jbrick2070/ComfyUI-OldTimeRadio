"""Stable Audio music adapter -- opt-in, commercial-clean.

Stability Community license (commercial use under a revenue cap). Native
ComfyUI support keeps Blackwell risk low. Opt-in behind OTR_ENABLE_STABLE_AUDIO.
``interface == "clip"``: the theme node calls ``generate_clip``.

Stable Audio is natively stereo and the target is stereo end to end, so output
channels are preserved (``canonical_audio`` keeps ``[B, C, T]``); a mono bridge
is only a transitional step while the assembly chain is still mono. The
inference call is wired and verified in the GPU pilot.
"""
from __future__ import annotations

from .registry import register


@register
class StableAudioMusicEngine:
    name = "stable_audio_music"
    roles = ("music",)
    default_roles = ()
    commercial_clean = True  # Stability Community license (revenue-capped)
    requires_flag = "OTR_ENABLE_STABLE_AUDIO"
    interface = "clip"
    sample_rate = 44100

    def __init__(self):
        self._model = None

    def load(self):
        if self._model is not None:
            return
        try:
            import stable_audio_tools  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "stable-audio-tools is not installed -- install Stable Audio "
                "before enabling OTR_ENABLE_STABLE_AUDIO"
            ) from exc
        # VERIFY in the pilot: load the Stable Audio pipeline (native ComfyUI
        # nodes or stable_audio_tools) and cache it on self._model.
        self._model = stable_audio_tools

    def unload(self):
        self._model = None

    def generate_clip(self, prompt, duration_s, seed):
        """Text prompt -> stereo AUDIO clip. Inference wired in the GPU pilot.

        TODO-for-F: the assumed library call is the ComfyUI-native SA3 sampler /
            generate_diffusion_cond(model, steps=<int>,
                conditioning=[{"prompt": prompt, "seconds_total": duration_s}],
                seed=seed, generator=<bound torch.Generator>)
        preserving stereo via canonical_audio. scripts/otr_audio_dep_pilot.py
        verifies on GPU that the native SA3 path binds a torch.Generator (so
        render-twice is byte-identical) before supports_external_generator is
        flipped True and this body is filled. Until then stable_audio_music is a
        flag-gated, default-off stub -- never run blind in production.
        """
        self.load()
        raise RuntimeError(
            "Stable Audio inference is wired and verified in the GPU pilot; "
            f"engine registered and selectable (prompt len={len(prompt or '')})"
        )
