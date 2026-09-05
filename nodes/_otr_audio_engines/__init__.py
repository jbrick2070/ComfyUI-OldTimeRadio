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
    resolve_voice_ref_path,
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
    eng_stable_audio,
    eng_stable_audio_3,
)

# eng_indextts2 is EXCLUDED FROM THE COMFY REGISTRY BUNDLE (.comfyignore). Its
# adapter is byte-hashed by _otr_voice_route.RUNTIME_FINGERPRINT_SOURCES and
# carries the one subprocess spawn plus the env reads the registry YARA scan
# flags; it is also a voice-CLONING engine that needs a sidecar venv and
# reference WAVs a registry install does not have. It ships in the GITHUB tree
# with full capability (the Lemmy route and its fingerprint intact), and on a
# registry install the file is simply absent, so the engine does not register --
# exactly the partial-install resilience the pack already relies on. kokoro (the
# shipped default on both voice slots), bark, chatterbox, dia and the cloud
# engines are unaffected.
try:
    from . import eng_indextts2  # noqa: E402,F401
except ImportError:
    pass

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
    "resolve_voice_ref_path",
]
