"""nodes/voice_render.py

Single-canonical voice-render ComfyUI node.

Phase 0+ Voice Backend Abstraction §1: ``OTR_VoiceRender``. Widgets
expose ``voice_model`` (enum across registered engines) and
``voice_preset`` (string). The node parses ``"engine:preset"`` voice
specs via ``_otr_voice_resolver.parse_voice_spec`` and routes through
the ``_voice_backends`` registry to dispatch the actual generation.

Status: NODE CLASS DEFINED, **NOT YET REGISTERED** in
``__init__.py``. Registration is deferred until after the in-flight
FULL acceptance soak completes (the orchestrator-touching step that
also flips ``OTR_BatchBark`` and ``OTR_KokoroAnnouncer`` into thin
shims). Until then this module is import-safe but not surfaced to
ComfyUI's node menu -- which is the whole point of landing it tonight
without violating the "don't touch __init__.py while soak runs" rule.

Once registration lands the existing workflow JSONs continue to validate
unchanged because ``OTR_BatchBark`` and ``OTR_KokoroAnnouncer`` stay in
the registry as back-compat shims.
"""
from __future__ import annotations

from typing import Any

from nodes._otr_voice_resolver import KNOWN_ENGINES, parse_voice_spec
from nodes._voice_backends import (
    available_engines,
    get_factory,
)


class OTR_VoiceRender:
    """ComfyUI node: render a string of dialogue to audio via a
    pluggable backend.

    The ``voice_spec`` widget accepts ``"engine:preset"`` form
    (matching ``_otr_voice_resolver.parse_voice_spec``). The class
    looks up the engine in the backend registry and delegates to that
    backend's ``generate(text, **kwargs)``.

    Backend-specific kwargs are forwarded as-is via
    ``**generation_kwargs``; an unknown engine doesn't validate kwargs
    here -- per the ``VoiceBackend`` protocol contract, drivers ignore
    unknown kwargs rather than raising.
    """

    CATEGORY = "OTR/Voice"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "render"

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        # Engine list pulled from KNOWN_ENGINES so the dropdown stays
        # stable even if a driver hasn't yet self-registered (the
        # registry is built lazily via _register_default_drivers).
        engines = sorted(KNOWN_ENGINES)
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "voice_model": (engines, {"default": "bark"}),
                "voice_preset": ("STRING", {"default": "v2/en_speaker_3"}),
            },
            "optional": {
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
        }

    def render(
        self,
        text: str,
        voice_model: str,
        voice_preset: str,
        temperature: float = 0.7,
    ) -> tuple[bytes]:
        """Render ``text`` into a single AUDIO output.

        Combines ``voice_model`` + ``voice_preset`` into the canonical
        ``engine:preset`` spec, validates via ``parse_voice_spec``,
        looks up the backend factory, loads and calls. Caller handles
        the return shape (downstream OTR audio nodes consume ``bytes``
        and demux as needed).
        """
        spec_string = f"{voice_model}:{voice_preset}"
        spec = parse_voice_spec(spec_string)  # raises ValueError on malformed spec
        try:
            factory = get_factory(spec.engine)
        except KeyError:
            registered = available_engines()
            raise RuntimeError(
                f"voice_model={voice_model!r} not registered "
                f"(available: {registered}). The bundled drivers (bark, "
                f"kokoro) self-register on first import; if you see this "
                f"error in production, the driver module failed to "
                f"import -- check the OTR_VoiceRender side of the dispatch."
            )
        backend = factory()
        backend.load(spec.preset)
        try:
            audio = backend.generate(text, temperature=temperature)
        finally:
            backend.unload()
        return (audio,)


# This module deliberately does NOT register OTR_VoiceRender in
# __init__.py. See the module-docstring "deferred until post-soak"
# comment. Surface registration step:
#   from nodes.voice_render import OTR_VoiceRender
#   NODE_CLASS_MAPPINGS["OTR_VoiceRender"] = OTR_VoiceRender
#   NODE_DISPLAY_NAME_MAPPINGS["OTR_VoiceRender"] = "OTR Voice Render"
__all__ = ["OTR_VoiceRender"]
