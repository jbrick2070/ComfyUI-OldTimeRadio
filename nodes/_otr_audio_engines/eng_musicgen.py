"""MusicGen theme adapter -- the byte-identical default for the music role.

Registration over the existing OTR_MusicGenTheme. ``interface == "batch"``: the
theme node delegates to the existing node so the default path stays
byte-identical. Heavy import is deferred to ``make_batch_node()``.
"""
from __future__ import annotations

from .registry import register


@register
class MusicGenEngine:
    name = "musicgen"
    roles = ("music",)
    default_roles = ("music",)
    commercial_clean = False  # Meta MusicGen terms not confirmed for commercial use
    requires_flag = None
    interface = "batch"

    def __init__(self):
        self._node = None

    def load(self):
        if self._node is None:
            from ..musicgen_theme import MusicGenTheme

            self._node = MusicGenTheme()

    def unload(self):
        self._node = None

    def make_batch_node(self):
        """Return the existing MusicGen theme node instance (lazy)."""
        self.load()
        return self._node
