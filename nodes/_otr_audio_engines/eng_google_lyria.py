"""Google Lyria music BYO-key adapter.

Direct Google API lane for theme music cues. This is not a Comfy Cloud Partner
node and it never invokes a local music model. The current shipped surface uses
the official Lyria 3 Clip model: requests contain only ``model`` and ``input``;
the provider returns a fixed 30-second MP3 clip, which OTR trims to the cue
duration after decode.

Import-time stays light: no Google SDK, NumPy, Torch, soundfile, ffmpeg probe,
network, or CUDA imports happen at module scope.
"""
from __future__ import annotations

import base64
import os
import subprocess

from .base import AudioEngineAdapter
from .registry import register
from .._otr_google_api.client import (
    GoogleAPIError,
    GoogleAPIRequestShapeError,
    create_interaction,
    resolve_api_key,
)

DEFAULT_MODEL = "lyria-3-clip-preview"
SUPPORTED_MODELS = (DEFAULT_MODEL,)
FIXED_CLIP_DURATION_S = 30
SAMPLE_RATE = 44100
_TIMEOUT_S = 300


class GoogleLyriaError(GoogleAPIError):
    """Raised for sanitized Google Lyria invoke/decode failures."""


def _selected_model() -> str:
    model = str(
        os.environ.get("OTR_GOOGLE_LYRIA_MODEL_ID")
        or os.environ.get("OTR_GOOGLE_LYRIA_MODEL")
        or DEFAULT_MODEL
    ).strip()
    if model not in SUPPORTED_MODELS:
        raise GoogleAPIRequestShapeError(
            "google_lyria.generate_clip: unsupported model %r; expected one of %r"
            % (model, SUPPORTED_MODELS)
        )
    return model


def _cue_duration_s(value) -> int:
    try:
        raw = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise GoogleAPIRequestShapeError(
            "google_lyria.generate_clip: non-numeric duration_s %r "
            "(no request sent)" % (value,)
        ) from exc
    return max(1, min(FIXED_CLIP_DURATION_S, raw))


def _interaction_payload(model: str, prompt: str) -> dict:
    return {
        "model": model,
        "input": prompt,
    }


def _audio_block(data: str, block: dict) -> dict:
    mime_type = str(block.get("mime_type") or block.get("mimeType") or "").lower()
    if mime_type and mime_type not in ("audio/mpeg", "audio/mp3", "audio/mpeg3"):
        raise GoogleAPIRequestShapeError(
            "Google Lyria response audio MIME %r is unsupported; expected MP3"
            % mime_type
        )
    return {"data": data, "mime_type": mime_type or "audio/mpeg"}


def _extract_audio_data(response: dict) -> dict:
    if not isinstance(response, dict):
        raise GoogleAPIRequestShapeError("Google Lyria response was not a JSON object")

    for key in ("output_audio", "outputAudio"):
        block = response.get(key)
        if isinstance(block, dict):
            data = block.get("data")
            if isinstance(data, str) and data.strip():
                return _audio_block(data, block)

    steps = response.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, dict):
                continue
            for block in step.get("content") or []:
                if not isinstance(block, dict):
                    continue
                data = block.get("data")
                kind = str(block.get("type") or "").lower()
                mime_type = str(
                    block.get("mime_type") or block.get("mimeType") or ""
                ).lower()
                if (
                    isinstance(data, str)
                    and data.strip()
                    and (kind == "audio" or mime_type.startswith("audio/"))
                ):
                    return _audio_block(data, block)

    raise GoogleAPIRequestShapeError(
        "Google Lyria response did not include audio data in output_audio, "
        "outputAudio, or steps[].content[]"
    )


def _mp3_to_audio(mp3_bytes: bytes, *, sample_rate: int = SAMPLE_RATE) -> dict:
    if not mp3_bytes:
        raise GoogleLyriaError("Google Lyria response audio was empty")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "mp3",
        "-i",
        "pipe:0",
        "-f",
        "f32le",
        "-acodec",
        "pcm_f32le",
        "-ac",
        "2",
        "-ar",
        str(int(sample_rate)),
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=mp3_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except FileNotFoundError as exc:
        raise GoogleLyriaError(
            "google_lyria.generate_clip: ffmpeg is required to decode MP3 output"
        ) from exc
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[:500]
        raise GoogleLyriaError(
            "google_lyria.generate_clip: ffmpeg failed to decode MP3 output: %s"
            % err
        )

    import numpy as np
    import torch

    arr = np.frombuffer(proc.stdout, dtype="<f4").copy()
    if arr.size == 0 or arr.size % 2:
        raise GoogleLyriaError(
            "google_lyria.generate_clip: decoded MP3 sample data was invalid"
        )
    wav_np = arr.reshape(-1, 2).T.reshape(1, 2, -1)
    return {"waveform": torch.from_numpy(wav_np), "sample_rate": int(sample_rate)}


def _audio_from_response(response: dict) -> dict:
    block = _extract_audio_data(response)
    try:
        mp3 = base64.b64decode(block["data"], validate=True)
    except Exception as exc:  # noqa: BLE001
        raise GoogleAPIRequestShapeError(
            "Google Lyria response audio was not valid base64 MP3: %s" % exc
        ) from exc
    return _mp3_to_audio(mp3, sample_rate=SAMPLE_RATE)


def _trim_audio(audio: dict, duration_s: int) -> dict:
    sr = int(audio.get("sample_rate") or SAMPLE_RATE)
    n = int(round(float(duration_s) * sr))
    wav = audio["waveform"]
    if n > 0 and wav.shape[-1] > n:
        return {"waveform": wav[..., :n].contiguous(), "sample_rate": sr}
    return audio


@register
class GoogleLyriaMusic(AudioEngineAdapter):
    """Registered as ``google_lyria``. Direct Google Lyria music, BYO key."""

    name = "google_lyria"
    roles = ("music",)
    default_roles = ()
    commercial_clean = True
    interface = "clip"
    sample_rate = SAMPLE_RATE
    native = False
    fixed_provider_duration_s = FIXED_CLIP_DURATION_S

    def generate_clip(self, prompt, duration_s, seed):  # noqa: ARG002
        prompt_text = str(prompt or "").strip()
        if not prompt_text:
            raise GoogleAPIRequestShapeError(
                "google_lyria.generate_clip: blank music prompt (no request sent)"
            )
        duration = _cue_duration_s(duration_s)
        model = _selected_model()
        api_key = resolve_api_key()
        payload = _interaction_payload(model, prompt_text)
        response = create_interaction(
            payload,
            timeout_s=_TIMEOUT_S,
            max_retries=0,
            _api_key=api_key,
        )
        return _trim_audio(_audio_from_response(response), duration)


__all__ = [
    "DEFAULT_MODEL",
    "FIXED_CLIP_DURATION_S",
    "GoogleLyriaError",
    "GoogleLyriaMusic",
    "SUPPORTED_MODELS",
    "_audio_from_response",
    "_interaction_payload",
]
