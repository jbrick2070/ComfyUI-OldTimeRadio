"""nodes/_voice_backends/kokoro.py

Kokoro voice backend driver -- THIN STUB.

Mirror of ``bark.py`` in shape: registers a Kokoro-shaped driver in the
registry so the abstraction is end-to-end addressable, but every method
raises ``KokoroBackendNotMigrated`` until the production Kokoro code is
relocated out of ``nodes/kokoro_announcer.py``.

Migration plan: same as Bark -- copy logic, swap announcer to delegate,
rebuild as thin shim. Deferred until post-soak.
"""
from __future__ import annotations

from typing import Any

from nodes._voice_backends import register
from nodes._voice_backends._protocol import VoiceBackend


class KokoroBackendNotMigrated(NotImplementedError):
    """Raised by every KokoroBackend method until the migration step lands."""


class KokoroBackend:
    """Stub Kokoro driver."""

    engine_name = "kokoro"

    def __init__(self) -> None:
        self._loaded_preset: str = ""

    def load(self, preset: str) -> None:
        raise KokoroBackendNotMigrated(
            "KokoroBackend.load is a stub; production Kokoro "
            "implementation still in nodes/kokoro_announcer.py."
        )

    def generate(self, text: str, **kwargs: Any) -> bytes:
        raise KokoroBackendNotMigrated(
            "KokoroBackend.generate is a stub; route Kokoro calls "
            "through OTR_KokoroAnnouncer / kokoro_announcer.py for now."
        )

    def unload(self) -> None:
        self._loaded_preset = ""


# Self-register on module import.
register("kokoro", KokoroBackend)


__all__ = ["KokoroBackend", "KokoroBackendNotMigrated"]
