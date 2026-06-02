"""Bark character-voice adapter -- the byte-identical default for char_voice.

Registration over the existing OTR_BatchBarkGenerator. ``interface == "batch"``:
the generic voice node delegates the whole ledger to the existing node, so the
default path stays byte-identical by construction. The heavy node import is
deferred to ``make_batch_node()`` so importing the registry package stays light.
"""
from __future__ import annotations

from .registry import register


@register
class BarkEngine:
    name = "bark"
    roles = ("char_voice",)
    default_roles = ("char_voice",)
    commercial_clean = False  # Suno Bark terms not confirmed for commercial use
    requires_flag = None
    interface = "batch"

    def __init__(self):
        self._node = None

    def load(self):
        if self._node is None:
            from ..batch_bark_generator import BatchBarkGenerator

            self._node = BatchBarkGenerator()

    def unload(self):
        self._node = None

    def make_batch_node(self):
        """Return the existing Bark node instance (lazy)."""
        self.load()
        return self._node
