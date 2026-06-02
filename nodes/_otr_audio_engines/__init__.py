"""Audio-engine registry package.

Exposes the registry primitives. Engine adapters self-register via explicit
imports added in a later sprint (each adapter import line makes that engine
appear in every dropdown for its role).
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

__all__ = [
    "AudioEngine",
    "assert_usable",
    "default_engine_for_role",
    "engines_for_role",
    "get_engine",
    "is_registered",
    "register",
]
