"""Audio-engine registry package.

Exposes the registry primitives and imports every engine adapter so it
self-registers. Adding a new model = add an adapter module + one import line
here, and it appears in every dropdown for its role.
"""
from __future__ import annotations

from .registry import (
    AudioEngine,
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
from . import eng_chatterbox, eng_indextts2, eng_stable_audio  # noqa: E402,F401

__all__ = [
    "AudioEngine",
    "assert_usable",
    "default_engine_for_role",
    "engines_for_role",
    "get_engine",
    "is_registered",
    "register",
]
