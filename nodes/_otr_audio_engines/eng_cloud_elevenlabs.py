"""ElevenLabs cloud TTS adapter (cloud-audio S1, 2026-07-03).

A casting-integrated CHARACTER + ANNOUNCER voice engine that runs on Comfy Cloud
via the built cloud backend (`invoke_partner_node`), NOT a local model. Fail-loud,
no fallback, dropdown-is-enable.

C3 voice identity = ADAPTER METADATA, not a disk sentinel: this engine sets
``requires_voice_ref=False`` (never sent down the local ref-resolution /
missing_ref_fallback path) and ``voice_ref_field="provider_voice_id"`` so the
dispatch feeds ``cast["provider_voice_id"]`` straight to ``generate_voice``.

C4 (grounded against the live core): the ``ELEVENLABS_VOICE`` socket is just a
STRING = the stable ElevenLabs voice_id. ``ElevenLabsVoiceSelector`` merely maps a
display label -> voice_id; ``ElevenLabsTextToSpeech.execute`` uses ``voice``
directly in its request path (``/text-to-speech/{voice}``). So the adapter builds
the payload IN-PROCESS by passing ``provider_voice_id`` (the id) as ``voice`` -- no
selector node in the graph.

C7 budget: a PER-LINE cost estimate (chars * $0.24/1K) is passed as
``estimated_usd`` so the backend budget cap is not inert.

Import-time is side-effect-free (C-5): heavy work + the cloud/canonicalize imports
are LAZY inside ``generate_voice``; module scope is the adapter contract only.
"""
from __future__ import annotations

import logging
import os

from .base import AudioEngineAdapter
from .registry import register

log = logging.getLogger("OTR")

#: ElevenLabs TTS price (PRICING.md): $0.24 / 1000 characters, flat across tiers.
_USD_PER_1K_CHARS = 0.24
#: invoke watchdog ceiling for one line (generous; the backend heartbeats).
_TTS_TIMEOUT_S = 180.0
#: the pinned partner row (partner_nodes.yaml) this adapter drives.
_PARTNER_ROW = "cloud_elevenlabs_tts"
#: Grounded against the installed Comfy Partner node:
#: ElevenLabsTextToSpeech exposes exactly these TTS models/formats today.
_SUPPORTED_MODELS = ("eleven_multilingual_v2", "eleven_v3")
_DEFAULT_MODEL_ID = "eleven_multilingual_v2"
_SUPPORTED_OUTPUT_FORMATS = ("mp3_44100_192", "opus_48000_192")
_DEFAULT_OUTPUT_FORMAT = "mp3_44100_192"
_SUPPORTED_TEXT_NORM = ("auto", "on", "off")
_DEFAULT_APPLY_TEXT_NORM = "auto"
_SEED_MAX = 2147483647


def estimate_tts_usd(text: str) -> float:
    """C7 per-LINE cost estimate. MUST be per-line scale (an episode-total-chars
    estimate per line would trip the $10 cap ~line 9)."""
    return max(len(text or ""), 1) * _USD_PER_1K_CHARS / 1000.0


def _stability_from_delivery(delivery_vector) -> float:
    """Delivery vector -> stability ONLY (ElevenLabs exposes stability + seed as
    the only per-line knobs). More expressive delivery => LOWER stability. Safe
    default 0.5; clamped to the provider range [0, 1]."""
    dv = delivery_vector or {}
    try:
        if isinstance(dv, dict):
            if dv.get("stability") is not None:
                s = float(dv["stability"])
            elif dv.get("expressiveness") is not None:
                s = 1.0 - float(dv["expressiveness"])   # expressive -> low stability
            else:
                s = 0.5
        else:
            s = 0.5
    except (TypeError, ValueError):
        s = 0.5
    return max(0.0, min(1.0, s))


def _partner_model_payload() -> dict:
    model_id = str(os.environ.get("OTR_ELEVENLABS_MODEL_ID") or _DEFAULT_MODEL_ID).strip()
    if model_id not in _SUPPORTED_MODELS:
        raise ValueError(
            "elevenlabs.generate_voice: unsupported model %r; expected one of %r"
            % (model_id, _SUPPORTED_MODELS)
        )
    payload = {
        "model": model_id,
        "similarity_boost": 0.75,
        "speed": 1.0,
    }
    if model_id == "eleven_multilingual_v2":
        payload.update({
            "use_speaker_boost": False,
            "style": 0.0,
        })
    return payload


def _partner_output_format() -> str:
    fmt = str(os.environ.get("OTR_ELEVENLABS_OUTPUT_FORMAT") or _DEFAULT_OUTPUT_FORMAT).strip()
    if fmt not in _SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            "elevenlabs.generate_voice: unsupported output_format %r; expected one of %r"
            % (fmt, _SUPPORTED_OUTPUT_FORMATS)
        )
    return fmt


def _apply_text_normalization() -> str:
    mode = str(os.environ.get("OTR_ELEVENLABS_TEXT_NORMALIZATION")
               or _DEFAULT_APPLY_TEXT_NORM).strip()
    if mode not in _SUPPORTED_TEXT_NORM:
        raise ValueError(
            "elevenlabs.generate_voice: unsupported apply_text_normalization %r; "
            "expected one of %r" % (mode, _SUPPORTED_TEXT_NORM)
        )
    return mode


def _partner_seed(seed) -> int:
    try:
        val = int(seed or 0)
    except (TypeError, ValueError):
        val = 0
    return val & _SEED_MAX


@register
class ElevenLabsCloudVoice(AudioEngineAdapter):
    """Registered as ``elevenlabs``. Cloud TTS char + announcer voice."""

    name = "elevenlabs"
    roles = ("char_voice", "announcer_voice")
    default_roles = ()                    # dropdown-opt-in; NEVER a default (C1/C2)
    commercial_clean = True               # library voices, ToS-clean (S2 pool)
    interface = "per_line"
    sample_rate = 44100                   # matches the pinned yaml + canonicalize
    native = False                        # runs on Comfy Cloud, not local residency
    # C3: adapter-metadata voice identity -- no local ref clip, no local fallback.
    requires_voice_ref = False
    voice_ref_field = "provider_voice_id"
    voice_ref_kind = "provider_voice_id"
    missing_ref_fallback = None           # fail-loud: a missing id RAISES at the S3 gate

    # FIX-1 (Fable): the conformance kwargs test resolves adapters by node_key and
    # calls _partner_inputs(request) to check the emitted kwargs against the pinned
    # row's declared inputs. The names are static (same for every line).
    node_key = _PARTNER_ROW

    def _partner_inputs(self, request=None):
        return ("voice", "text", "stability", "apply_text_normalization",
                "model", "language_code", "seed", "output_format")

    def generate_voice(self, text, provider_voice_id, delivery_vector, seed):
        """One dialogue line -> a canonicalized AUDIO dict, via Comfy Cloud.

        FAIL-LOUD, NO FALLBACK: a blank line, a missing provider_voice_id, or any
        provider/auth/budget failure RAISES (never a silent local swap)."""
        if not text or not str(text).strip():
            raise ValueError("elevenlabs.generate_voice: blank line (no-fallback)")
        vid = str(provider_voice_id or "").strip()
        if not vid:
            raise ValueError(
                "elevenlabs.generate_voice: no provider_voice_id on this cast "
                "entry -- the S2 pool / CastLock stamp must supply the ElevenLabs "
                "voice_id. NO FALLBACK (never inherits another voice).")

        from .._otr_shared.cloud_media_invoke import invoke_partner_node
        from .._otr_shared.cloud_media_canonical import canonicalize_audio

        inputs = {
            "voice": vid,                                   # ELEVENLABS_VOICE = the id
            "text": str(text),
            "stability": _stability_from_delivery(delivery_vector),
            "apply_text_normalization": _apply_text_normalization(),
            "model": _partner_model_payload(),
            "language_code": "",
            "seed": _partner_seed(seed),
            "output_format": _partner_output_format(),
        }
        result = invoke_partner_node(
            _PARTNER_ROW, inputs,
            timeout_s=_TTS_TIMEOUT_S,
            estimated_usd=estimate_tts_usd(str(text)))

        # loudness matched to the LOCAL lane (RMS -16 dBFS) + 44.1k WAV (C8).
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
                "elevenlabs: failed to read canonical WAV %r: %s" % (path, exc)
            ) from exc
        wav = torch.from_numpy(np.ascontiguousarray(data.T)).unsqueeze(0)  # (1,C,T)
        return {"waveform": wav, "sample_rate": int(sr)}


__all__ = ["ElevenLabsCloudVoice", "estimate_tts_usd"]
