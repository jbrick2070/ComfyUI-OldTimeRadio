"""Google Gemini TTS BYO-key adapter.

Direct Google API lane for character and announcer voice. This is not a Comfy
Cloud Partner node and it never routes through the Partner backend. The adapter
is explicit-selection-only, fail-loud, and uses the current Gemini Interactions
REST shape.

Import-time stays light: no Google SDK, NumPy, Torch, or network imports happen
at module scope.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import urllib.error
import urllib.request

from .base import AudioEngineAdapter
from .registry import register

log = logging.getLogger("OTR")

_INTERACTIONS_URL = "https://generativelanguage.googleapis.com/v1beta/interactions"
_DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
_LOW_COST_RETRY_MODEL = "gemini-3.1-flash-tts-preview"
_TTS_TIMEOUT_S = 180.0
_SAMPLE_RATE = 24000

_SUPPORTED_MODELS = (
    "gemini-3.1-flash-tts-preview",
    "gemini-2.5-flash-preview-tts",
    "gemini-2.5-pro-preview-tts",
)

_SUPPORTED_VOICES = (
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus",
    "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus",
    "Umbriel", "Algieba", "Despina", "Erinome", "Algenib",
    "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar",
    "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
)

_ANNOUNCER_STYLE_BY_VOICE = {
    "Charon": "a polished British radio announcer style",
    "Sulafat": "a warm British radio announcer style",
}

_EMOTION_PREFIX_THRESHOLD = 0.15
_EMOTION_PREFIX = {
    "angry": "with controlled fury, clipped and tense",
    "afraid": "with audible fear, slightly breathless",
    "happy": "warmly and brightly",
    "sad": "quietly, with restrained grief",
    "disgusted": "with barely concealed contempt",
    "melancholic": "wistful and reflective",
    "surprised": "with a startled intake, then urgency",
    "calm": "",
}
_EMOTION_KEYS = tuple(_EMOTION_PREFIX)

_STAGE_TAGS = {
    "sighs": "[sighs]",
    "whispers": "[whispers]",
    "gasps": "[gasp]",
    "laughs": "[laughs]",
}
_PAREN_STAGE = re.compile(r"\((sighs|whispers|gasps|laughs)\)", re.IGNORECASE)
_BRACKET_STAGE = re.compile(r"\[(sighs|whispers|gasps|laughs|gasp)\]", re.IGNORECASE)


class GoogleTTSError(RuntimeError):
    """Raised for sanitized Google TTS invoke/response failures."""


def _api_key_env_names() -> tuple[str, ...]:
    return ("OTR_GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _known_secret_values(extra=()) -> tuple[str, ...]:
    vals = []
    for name in _api_key_env_names():
        val = os.environ.get(name)
        if val:
            vals.append(str(val))
    vals.extend(str(v) for v in (extra or ()) if v)
    return tuple(v for v in vals if v)


def _redact(text, extra=()) -> str:
    out = str(text)
    for secret in _known_secret_values(extra):
        out = out.replace(secret, "<REDACTED>")
    return out


def _resolve_api_key() -> str:
    for name in _api_key_env_names():
        val = os.environ.get(name)
        if val and str(val).strip():
            return str(val).strip()
    raise GoogleTTSError(
        "google_tts.generate_voice: missing Google API key; set one of "
        "OTR_GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY"
    )


def _selected_model() -> str:
    model = (
        os.environ.get("OTR_GOOGLE_TTS_MODEL_ID")
        or os.environ.get("OTR_GOOGLE_TTS_MODEL")
        or _DEFAULT_MODEL
    )
    model = str(model).strip()
    if model not in _SUPPORTED_MODELS:
        raise ValueError(
            "google_tts.generate_voice: unsupported model %r; expected one of %r"
            % (model, _SUPPORTED_MODELS)
        )
    return model


def _models_to_try(model: str) -> tuple[str, ...]:
    configured = str(os.environ.get("OTR_GOOGLE_TTS_RETRY_MODELS") or "").strip()
    if configured:
        raw = [m.strip() for m in configured.split(",") if m.strip()]
        unknown = [m for m in raw if m not in _SUPPORTED_MODELS]
        if unknown:
            raise ValueError(
                "google_tts.generate_voice: unsupported retry model(s) %r; "
                "expected one of %r" % (unknown, _SUPPORTED_MODELS)
            )
        return tuple(dict.fromkeys([model] + raw))

    enabled = os.environ.get("OTR_GOOGLE_TTS_ENABLE_MODEL_RETRY", "0") == "1"
    if not enabled:
        return (model,)
    return tuple(dict.fromkeys([model, _DEFAULT_MODEL, _LOW_COST_RETRY_MODEL]))


def _validate_voice(voice: str) -> str:
    voice = str(voice or "").strip()
    if not voice:
        raise ValueError(
            "google_tts.generate_voice: missing provider_voice_id; CastLock must "
            "stamp a Google voice id. NO FALLBACK."
        )
    if voice not in _SUPPORTED_VOICES:
        raise ValueError(
            "google_tts.generate_voice: unsupported voice %r; expected one of %r"
            % (voice, _SUPPORTED_VOICES)
        )
    return voice


def _interaction_payload(model: str, text: str, voice: str) -> dict:
    return {
        "model": model,
        "input": text,
        "response_format": {"type": "audio"},
        "generation_config": {
            "speech_config": [
                {"voice": voice},
            ],
        },
    }


def _post_interaction(api_key: str, payload: dict) -> dict:
    data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    req = urllib.request.Request(
        _INTERACTIONS_URL,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
    )
    with urllib.request.urlopen(req, timeout=_TTS_TIMEOUT_S) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _sample_rate_from(block: dict) -> int:
    raw = block.get("sample_rate", block.get("sampleRate", _SAMPLE_RATE))
    try:
        sample_rate = int(raw)
    except (TypeError, ValueError) as exc:
        raise GoogleTTSError(
            "Google TTS response audio sample_rate was not an integer"
        ) from exc
    if sample_rate <= 0:
        raise GoogleTTSError("Google TTS response audio sample_rate was invalid")
    return sample_rate


def _audio_block(data: str, block: dict) -> dict:
    mime_type = str(block.get("mime_type") or block.get("mimeType") or "").lower()
    if mime_type and mime_type not in ("audio/l16", "audio/pcm"):
        raise GoogleTTSError(
            "Google TTS response audio MIME %r is unsupported; expected audio/l16"
            % mime_type
        )
    return {
        "data": data,
        "sample_rate": _sample_rate_from(block),
        "mime_type": mime_type or "audio/l16",
    }


def _extract_audio_data(response: dict) -> dict:
    if not isinstance(response, dict):
        raise GoogleTTSError("Google TTS response was not a JSON object")

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
            content = step.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
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

    raise GoogleTTSError(
        "Google TTS response did not include audio data in output_audio, "
        "outputAudio, or steps[].content[]"
    )


def _pcm16le_to_audio(pcm: bytes, sample_rate: int = _SAMPLE_RATE) -> dict:
    if not pcm:
        raise GoogleTTSError("Google TTS response audio was empty")
    if len(pcm) % 2:
        raise GoogleTTSError("Google TTS PCM byte length is not 16-bit aligned")
    import numpy as np
    import torch

    arr = np.frombuffer(pcm, dtype="<i2").copy()
    wav_np = (arr.astype(np.float32) / 32768.0).reshape(1, 1, -1)
    return {"waveform": torch.from_numpy(wav_np), "sample_rate": sample_rate}


def _audio_from_response(response: dict) -> dict:
    block = _extract_audio_data(response)
    try:
        pcm = base64.b64decode(block["data"], validate=True)
    except Exception as exc:  # noqa: BLE001
        raise GoogleTTSError(
            "Google TTS response audio was not valid base64 PCM: %s" % exc
        ) from exc
    return _pcm16le_to_audio(pcm, int(block["sample_rate"]))


def _preserve_stage_tags(text: str) -> tuple[str, dict[str, str]]:
    saved: dict[str, str] = {}

    def _save(tag: str) -> str:
        marker = "OTRGOOGLETAG%d" % len(saved)
        saved[marker] = tag
        return " %s " % marker

    def _paren(match):
        key = match.group(1).lower()
        return _save(_STAGE_TAGS[key])

    def _bracket(match):
        key = match.group(1).lower()
        tag = "[gasp]" if key == "gasp" else _STAGE_TAGS[key]
        return _save(tag)

    text = _PAREN_STAGE.sub(_paren, text or "")
    text = _BRACKET_STAGE.sub(_bracket, text)
    return text, saved


def _restore_stage_tags(text: str, saved: dict[str, str]) -> str:
    out = text or ""
    for marker, tag in saved.items():
        out = out.replace(marker, tag)
    return re.sub(r"\s+", " ", out).strip()


def _delivery_prefix(delivery_vector) -> str:
    if not isinstance(delivery_vector, dict):
        return ""
    scores = {}
    for key in _EMOTION_KEYS:
        if key == "calm":
            continue
        try:
            scores[key] = float(delivery_vector.get(key) or 0.0)
        except (TypeError, ValueError):
            scores[key] = 0.0
    if not scores:
        return ""
    emotion, score = max(scores.items(), key=lambda item: (item[1], item[0]))
    if score < _EMOTION_PREFIX_THRESHOLD:
        return ""
    return _EMOTION_PREFIX.get(emotion, "")


def _apply_delivery_prompt(text: str, delivery_vector) -> str:
    prefix = _delivery_prefix(delivery_vector)
    if not prefix:
        return text
    if str(text).lower().startswith("say "):
        return text
    return "Say %s: %s" % (prefix, text)


def _apply_voice_prompt(text: str, voice: str) -> str:
    style = _ANNOUNCER_STYLE_BY_VOICE.get(voice)
    if not style or str(text).lower().startswith("say "):
        return text
    return "Say in %s: %s" % (style, text)


@register
class GoogleTTSVoice(AudioEngineAdapter):
    """Registered as ``google_tts``. Direct Google Gemini TTS, BYO key."""

    name = "google_tts"
    roles = ("char_voice", "announcer_voice")
    default_roles = ()
    commercial_clean = True
    interface = "per_line"
    sample_rate = _SAMPLE_RATE
    native = False
    requires_voice_ref = False
    voice_ref_field = "provider_voice_id"
    voice_ref_kind = "provider_voice_id"
    missing_ref_fallback = None

    def prepare_text(self, text, delivery_vector=None):
        if not text or not str(text).strip():
            raise ValueError("google_tts.prepare_text: blank line")
        protected, saved = _preserve_stage_tags(str(text))
        from .._otr_script_prep import prepare_text as _neutral_prepare_text

        cleaned = _restore_stage_tags(_neutral_prepare_text(protected), saved)
        return _apply_delivery_prompt(cleaned, delivery_vector)

    def generate_voice(self, text, provider_voice_id, delivery_vector, seed):
        if not text or not str(text).strip():
            raise ValueError("google_tts.generate_voice: blank line")
        voice = _validate_voice(provider_voice_id)
        model = _selected_model()
        api_key = _resolve_api_key()
        spoken = _apply_voice_prompt(str(text).strip(), voice)

        last_error = None
        for attempt, model_id in enumerate(_models_to_try(model), start=1):
            payload = _interaction_payload(model_id, spoken, voice)
            try:
                response = _post_interaction(api_key, payload)
                return _audio_from_response(response)
            except (urllib.error.HTTPError, urllib.error.URLError,
                    json.JSONDecodeError, UnicodeDecodeError, GoogleTTSError,
                    ValueError) as exc:
                last_error = GoogleTTSError(
                    "Google TTS request failed for model %s: %s"
                    % (model_id, _redact(exc, extra=(api_key,)))
                )
                remaining = _models_to_try(model)[attempt:]
                if remaining:
                    next_model = remaining[0]
                    log.warning(
                        "[google_tts] model %s failed; retrying configured "
                        "Google TTS model %s",
                        model_id,
                        next_model,
                    )
                    continue
                break
        raise last_error or GoogleTTSError("Google TTS request failed")


__all__ = [
    "GoogleTTSError",
    "GoogleTTSVoice",
    "_EMOTION_PREFIX_THRESHOLD",
    "_SUPPORTED_MODELS",
    "_SUPPORTED_VOICES",
    "_interaction_payload",
]
