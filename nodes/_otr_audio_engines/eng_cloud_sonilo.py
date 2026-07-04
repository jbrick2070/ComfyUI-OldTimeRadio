"""Sonilo cloud MUSIC adapter (cloud-audio S5, 2026-07-03).

A truly-cloud music engine (theme cues) that runs on Comfy Cloud via the built
cloud backend (``invoke_partner_node``), NOT a local model. Fail-loud, no
fallback, dropdown-is-enable.

Contract parity with the local music engines (``eng_stable_audio_3`` /
``eng_musicgen``): ``interface == "clip"``, ``roles == ("music",)``, and one
``generate_clip(prompt, duration_s, seed)`` per theme cue returning the Bark
AUDIO clip contract ``{"waveform"[1,C,T], "sample_rate"}``. The prompt is composed
UPSTREAM by the single-source Meta-brief composer (``_otr_music_prompt``) exactly
like every other music engine -- this adapter only consumes it.

Sonilo exposes provider-native ``duration`` (INT seconds), ``prompt`` and
``seed`` (the pinned ``cloud_sonilo_music`` row). No local model, no sidecar, no
VRAM: the cost is credits + auth, enforced at invoke.

C7 budget: a PER-CUE cost estimate (duration * $0.15/60s) is passed as
``estimated_usd`` so the backend budget cap is not inert.

Import-time is side-effect-free (C-5): the cloud/canonicalize imports are LAZY
inside ``generate_clip``; module scope is the adapter contract only.
"""
from __future__ import annotations

import logging

from .registry import register

log = logging.getLogger("OTR")

#: Sonilo music price (PRICING.md): $0.15 / 60 seconds of rendered audio.
_USD_PER_60S = 0.15
#: invoke watchdog ceiling for one cue (generous; the backend heartbeats).
_MUSIC_TIMEOUT_S = 240.0
#: the pinned partner row (partner_nodes.yaml) this adapter drives.
_PARTNER_ROW = "cloud_sonilo_music"


def estimate_music_usd(duration_s) -> float:
    """C7 per-CUE cost estimate. Per-cue scale (a per-cue estimate of the whole
    episode's music seconds would over-reserve the budget)."""
    try:
        dur = max(1.0, float(duration_s))
    except (TypeError, ValueError):
        dur = 1.0
    return dur * _USD_PER_60S / 60.0


@register
class SoniloCloudMusic:
    """Registered as ``sonilo``. Cloud music (theme cues) via Comfy Cloud."""

    name = "sonilo"
    roles = ("music",)
    default_roles = ()                    # dropdown-opt-in; NEVER a default (C1/C2)
    commercial_clean = True               # Sonilo library service, ToS commercial-clean
    requires_flag = None                  # C6: registry IS the menu (no flag gate)
    native = False                        # runs on Comfy Cloud, not ComfyUI's own nodes
    interface = "clip"
    sample_rate = 44100

    # The conformance kwargs test resolves adapters by node_key and calls
    # _partner_inputs(request) to check the emitted kwargs against the pinned
    # row's declared inputs. The names are static (same for every cue).
    node_key = _PARTNER_ROW

    def _partner_inputs(self, request=None):
        return ("prompt", "duration", "seed")

    # Cloud engine has no local residency -- load/unload are no-ops (the Protocol
    # requires them; teardown in the theme node calls unload() best-effort).
    def load(self) -> None:
        return None

    def unload(self) -> None:
        return None

    def generate_clip(self, prompt, duration_s, seed):
        """One theme cue -> a canonicalized stereo AUDIO clip, via Comfy Cloud.

        FAIL-LOUD, NO FALLBACK: a blank prompt or any provider/auth/budget failure
        RAISES (never a silent local swap to musicgen / stable_audio)."""
        if not prompt or not str(prompt).strip():
            raise ValueError("sonilo.generate_clip: blank music prompt (no-fallback)")
        try:
            dur = max(1, int(round(float(duration_s))))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "sonilo.generate_clip: non-numeric duration_s %r (no-fallback)"
                % (duration_s,)) from exc

        from .._otr_shared.cloud_media_invoke import invoke_partner_node
        from .._otr_shared.cloud_media_canonical import canonicalize_audio

        inputs = {
            "prompt": str(prompt),
            "duration": dur,                        # provider-native seconds (INT)
            "seed": int(seed or 0),
        }
        result = invoke_partner_node(
            _PARTNER_ROW, inputs,
            timeout_s=_MUSIC_TIMEOUT_S,
            estimated_usd=estimate_music_usd(dur))

        # Stereo music matched to the LOCAL music lane (RMS -16 dBFS) + 44.1k WAV.
        canon = canonicalize_audio(
            result, {"sample_rate": self.sample_rate, "stereo_policy": "stereo"},
            session=None)

        return self._wav_to_audio(str(canon.path))

    @staticmethod
    def _wav_to_audio(path: str) -> dict:
        """Load a WAV into the Bark AUDIO contract {waveform[1,C,T], sample_rate}."""
        import numpy as np
        import torch
        try:
            import soundfile as sf
            data, sr = sf.read(path, dtype="float32", always_2d=True)  # (T, C)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "sonilo: failed to read canonical WAV %r: %s" % (path, exc)
            ) from exc
        wav = torch.from_numpy(np.ascontiguousarray(data.T)).unsqueeze(0)  # (1,C,T)
        return {"waveform": wav, "sample_rate": int(sr)}


__all__ = ["SoniloCloudMusic", "estimate_music_usd"]
