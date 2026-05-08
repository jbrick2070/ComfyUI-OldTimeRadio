"""nodes/_voice_backends/_protocol.py

Abstract interface every voice backend driver must conform to.

A backend is a small object with three methods:
    load(preset)       -- prepare the model for ``preset`` (idempotent)
    generate(text, **) -> wav   -- render speech, return raw audio
    unload()           -- release VRAM / file handles (idempotent)

Returning the WAV bytes (or numpy array, depending on backend) keeps
the contract narrow -- the caller decides what to do with the audio
(write to disk, mux into a ledger entry, splice with sfx, etc.). The
``**kwargs`` slot on ``generate`` is for backend-specific knobs
(temperature, hallucination guard, voice tone) -- pluggable per
engine, NOT in the abstract surface.

This module is stdlib-only and does NOT import torch. Drivers that
DO use torch import it inside their own module bodies, lazily, so a
caller that only needs the registry shape pays no torch cost.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class VoiceBackend(Protocol):
    """Structural typing protocol for a voice backend driver.

    Implementers do NOT need to subclass; duck typing is enough.
    ``runtime_checkable`` lets tests use ``isinstance(obj, VoiceBackend)``
    to verify shape conformance without forcing inheritance.
    """

    engine_name: str  # canonical engine name, e.g. "bark"

    def load(self, preset: str) -> None:
        """Prepare the model for ``preset``. Must be idempotent --
        calling load twice with the same preset is a no-op. Calling
        load with a DIFFERENT preset on an already-loaded backend
        should swap presets without raising.
        """
        ...

    def generate(self, text: str, **kwargs: Any) -> bytes:
        """Render ``text`` to audio.

        Returns raw WAV bytes by default (backend may document a
        different return shape -- e.g. numpy float32 array -- and
        callers can branch on the backend's known contract).

        Backend-specific kwargs:
            bark    : temperature (float), text_temp, waveform_temp
            kokoro  : speed, pitch
            cosyvoice : style, language
        Unknown kwargs MUST be ignored, not raise -- forward-compat
        with future backend additions.
        """
        ...

    def unload(self) -> None:
        """Release VRAM / file handles. Must be idempotent. Calling
        unload on a backend that was never loaded is a no-op.
        """
        ...


__all__ = ["VoiceBackend"]
