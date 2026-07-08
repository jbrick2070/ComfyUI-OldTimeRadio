"""Audio-engine registry package.

Exposes the registry primitives and imports every engine adapter so it
self-registers. Adding a new model = add an adapter module + one import line
here, and it appears in every dropdown for its role.
"""
from __future__ import annotations

from .base import (
    AudioEngineAdapter,
    engine_supports_external_generator,
    pack_audio_batch,
)
from .registry import (
    AudioEngine,
    EngineUnusable,
    EngineUsabilityReason,
    assert_usable,
    default_engine_for_role,
    engines_for_role,
    get_engine,
    is_registered,
    register,
)

# Legacy defaults (byte-identical for their role).
from . import eng_bark, eng_kokoro, eng_musicgen  # noqa: E402,F401

# Opt-in engines (flag-gated; never a default).
from . import (  # noqa: E402,F401
    eng_chatterbox,
    eng_dia,
    eng_indextts2,
    eng_stable_audio,
    eng_stable_audio_3,
)

# Cloud/direct-API engines (dropdown-opt-in; never a default). ElevenLabs/Sonilo
# route through Comfy Partner nodes; Google TTS/Lyria are direct Gemini BYO API.
from . import (  # noqa: E402,F401
    eng_cloud_elevenlabs,
    eng_cloud_sonilo,
    eng_google_lyria,
    eng_google_tts,
)

__all__ = [
    "AudioEngine",
    "AudioEngineAdapter",
    "EngineUnusable",
    "EngineUsabilityReason",
    "assert_usable",
    "default_engine_for_role",
    "engine_supports_external_generator",
    "engines_for_role",
    "get_engine",
    "is_registered",
    "pack_audio_batch",
    "register",
]
