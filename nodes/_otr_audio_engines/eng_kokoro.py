"""Kokoro announcer-voice adapter -- the byte-identical default for
announcer_voice (Apache-2.0, commercial-clean).

Registration over the existing OTR_KokoroAnnouncer. ``interface == "batch"``:
the announcer node delegates to the existing node so the default path stays
byte-identical. Heavy import is deferred to ``make_batch_node()``.
"""
from __future__ import annotations

from .registry import register


@register
class KokoroEngine:
    name = "kokoro"
    roles = ("announcer_voice",)
    default_roles = ("announcer_voice",)
    commercial_clean = True  # Apache-2.0
    requires_flag = None
    interface = "batch"

    def __init__(self):
        self._node = None

    def load(self):
        if self._node is None:
            from ..kokoro_announcer import KokoroAnnouncer

            self._node = KokoroAnnouncer()

    def unload(self):
        self._node = None

    def make_batch_node(self):
        """Return the existing Kokoro announcer node instance (lazy)."""
        self.load()
        return self._node
