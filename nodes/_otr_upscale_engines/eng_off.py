"""``off`` upscale engine -- the mandatory pass-through / HONEST-SWITCH default.

Never actually invoked in the frame loop: ``OTR_SilentComposite._encode_segment``
takes the byte-identical ffmpeg fast path whenever ``engine is None or
engine.name == "off"``. This class exists so:

* the registry has an entry to return for ``get_engine("off")`` (HONEST-SWITCH
  LAW: ``off`` is a real registered engine, not an implicit absence);
* ``assert_usable("off", "upscale_stage")`` succeeds cleanly (never raises);
* the dropdown default and the profile default agree on ``"off"`` as a real
  legal token.

Cold-import clean: NO heavy imports here.
"""
from __future__ import annotations

from .registry import register


@register
class OffUpscale:
    """The pass-through upscale engine. Never touches raw frames."""

    name = "off"
    roles = ("upscale_stage",)
    default_roles = ("upscale_stage",)   # dropdown default; sorts first
    commercial_clean = True
    requires_flag = None

    # Upscale-namespace identity
    device_backends = ("cuda", "cpu")
    requires_vendor = None
    intrinsic_scale = 1

    def __init__(self) -> None:
        self.device = None

    def load(self, device) -> None:
        """Stores the device for Protocol conformance; does nothing else."""
        self.device = device

    def unload(self) -> None:
        self.device = None

    def upscale_frames(self, frames):
        """Pass-through. Never invoked in production -- the composite's fast
        path bypasses the model pipeline entirely when engine.name == "off"."""
        return frames
