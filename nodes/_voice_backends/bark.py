"""nodes/_voice_backends/bark.py

Bark voice backend driver -- THIN STUB.

This module exists so the registry has a placeholder ``bark`` entry to
hand back, and so the migration step (relocating the production Bark
implementation out of ``nodes/batch_bark_generator.py`` and into this
file) lands as a *delete-and-paste* rather than a full rewrite. Until
the migration ships (deferred until after the in-flight FULL acceptance
soak completes), every method here raises ``BarkBackendNotMigrated``.

Migration plan:
1. After the soak, copy the Bark load/generate/unload logic from
   ``batch_bark_generator.py`` into this driver.
2. Update ``batch_bark_generator.py`` to delegate to the driver via
   ``get_factory("bark")()``.
3. ``OTR_BatchBark`` becomes a thin shim that always pre-pins
   ``voice_model="bark"``.

Until then this stub keeps the registry intact and the tests honest
(they verify the registry surface, not the model output -- which would
require Bark's heavy weights and is wrong for unit tests).
"""
from __future__ import annotations

from typing import Any

from nodes._voice_backends import register
from nodes._voice_backends._protocol import VoiceBackend


class BarkBackendNotMigrated(NotImplementedError):
    """Raised by every BarkBackend method until the migration step lands."""


class BarkBackend:
    """Stub Bark driver.

    Conforms to ``VoiceBackend`` structurally (has the right method
    signatures and ``engine_name`` attribute) so registry lookups
    succeed; methods raise ``BarkBackendNotMigrated`` because the
    actual Bark code still lives in ``batch_bark_generator.py``.
    """

    engine_name = "bark"

    def __init__(self) -> None:
        self._loaded_preset: str = ""

    def load(self, preset: str) -> None:
        raise BarkBackendNotMigrated(
            "BarkBackend.load is a stub; production Bark implementation "
            "still in nodes/batch_bark_generator.py. Migration step "
            "deferred until post-soak per the in-flight FULL acceptance "
            "discipline (see ROADMAP.md Voice Backend Abstraction)."
        )

    def generate(self, text: str, **kwargs: Any) -> bytes:
        raise BarkBackendNotMigrated(
            "BarkBackend.generate is a stub; route Bark calls through "
            "OTR_BatchBark / batch_bark_generator.py for now."
        )

    def unload(self) -> None:
        # Stub unload is a no-op so a caller's defensive try/finally
        # cleanup doesn't itself raise.
        self._loaded_preset = ""


# Self-register on module import.
register("bark", BarkBackend)


__all__ = ["BarkBackend", "BarkBackendNotMigrated"]
