"""Stable Audio music adapter -- opt-in, commercial-clean.

Stability Community license (commercial use under a revenue cap). Native
ComfyUI support keeps Blackwell risk low. Opt-in by INSTALL (the registry is
the menu -- no enable flag exists); loading fails with a NAMED error until
stable-audio-tools is installed.
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
    requires_flag = None  # vestigial (registry IS the menu; no flag gate)
    interface = "clip"
    sample_rate = 44100

    def __init__(self):
        self._model = None

    def load(self):
        if self._model is not None:
            return
        try:
            from stable_audio_tools import get_pretrained_model
        except ImportError as exc:
            raise RuntimeError(
                "stable-audio-tools is not installed -- install Stable Audio "
                "before rendering with stable_audio_music"
            ) from exc
        import os

        # GPU-VALIDATE (F): the plan's target is the ComfyUI-native SA3 loader;
        # this loads the documented stable_audio_tools pretrained model. The env
        # var points at the SA3 checkpoint / model id.
        model_id = os.environ.get(
            "OTR_STABLE_AUDIO_MODEL", "stabilityai/stable-audio-open-1.0"
        )
        model, _config = get_pretrained_model(model_id)
        self._model = model

    def unload(self):
        self._model = None

    def generate_clip(self, prompt, duration_s, seed):
        """Text prompt -> stereo AUDIO clip ``{"waveform", "sample_rate"}``.

        Implemented to the documented assumed_call:
            generate_diffusion_cond(model, steps=<int>,
                conditioning=[{"prompt": prompt, "seconds_total": duration_s}],
                seed=seed[, generator=<bound torch.Generator>])
        Stereo is preserved (pack_audio_batch downmixes only while the assembly
        chain is still mono). GPU-VALIDATE (F): the plan's target is the
        ComfyUI-native SA3 sampler; scripts/otr_audio_dep_pilot pins the real
        entry point + that it binds a ``torch.Generator`` on the box, then flips
        ``supports_external_generator`` True. Default-off until then.
        """
        import torch

        from .base import supported_kwargs

        self.load()
        seed = int(seed)
        torch.manual_seed(seed)

        from stable_audio_tools.inference.generation import generate_diffusion_cond

        # Post-ship audit fix (2026-07-10): EXPLICIT device -- the theme
        # node stamps the CastLock ledger device onto every music adapter
        # as requested_device; this legacy engine was the one adapter the
        # S4 waterfall sweep missed (the spec listed kokoro/musicgen/
        # chatterbox by name). No probe, no fallback: an unavailable
        # device fails loud downstream.
        dev = getattr(self, "requested_device", None) or "cuda"
        if dev == "cuda":
            torch.cuda.manual_seed_all(seed)
        kwargs = supported_kwargs(
            generate_diffusion_cond,
            steps=100,
            conditioning=[{"prompt": prompt, "seconds_total": float(duration_s)}],
            seed=seed,
            device=dev,
            generator=torch.Generator(device=dev).manual_seed(seed),
        )
        audio = generate_diffusion_cond(self._model, **kwargs)
        return {"waveform": audio, "sample_rate": self.sample_rate}
