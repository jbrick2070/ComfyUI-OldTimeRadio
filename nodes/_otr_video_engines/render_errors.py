"""THE RENDER ERROR LEAF: render failure types, and nothing else.

WHY THIS MODULE EXISTS, and why it is not in ``retry_taxonomy``. ShotLock
needs to catch ONE specific render failure -- a cast-time image gap -- and let
every other render failure through. That needs a named exception type, and the
type has to live somewhere both ``render_driver`` and ``otr_shot_lock`` can
import without a cycle. ``retry_taxonomy`` is the obvious-looking home and is
the wrong one: ``render_driver`` already imports it, so defining a subclass of
``RenderError`` there would import ``render_driver`` straight back.

So this is a LEAF. It imports nothing from this package -- nothing from the
registry, the engines, the driver or the shared helpers -- and nothing here
may ever grow an import that changes that. ``render_driver`` re-exports both
names, so every existing ``from .render_driver import RenderError`` keeps
working and the class objects are identical, which is what makes ``isinstance``
and ``except`` behave the same on both import paths.

Cold-import clean: stdlib only.
"""

from __future__ import annotations


class RenderError(RuntimeError):
    """A shot's selected engine failed to render. Fallbacks are DISABLED
    (operator 2026-06-16, 'this is art, not a space shuttle'): a proven model
    path must prove itself, so this is terminal -- the episode fails LOUD instead
    of swapping engines or degrading to a still floor."""


class DeferredImageGapError(RenderError):
    """The image this request needs DOES NOT EXIST YET, and that is expected.

    Raised only where a still is genuinely not minted at the time the request
    is built -- CAST TIME, before ``OTR_ImageGenDispatcher`` has run. ShotLock
    catches exactly this type, substitutes a placeholder init image, and lets
    the image phase fill the gap; the still spine then proves the real still
    exists before any render starts, so nothing renders on the placeholder.

    IT IS A SUBCLASS ON PURPOSE. Anything that is not ShotLock's narrow catch
    -- the render path, the batch node, a test -- still sees a ``RenderError``
    and still fails LOUD, so introducing this type cannot quietly convert a
    real render failure into a soft one.

    WHAT THIS REPLACED, and why the replacement is the whole point: ShotLock
    used to decide deferrability by SUBSTRING-MATCHING the exception message
    against four needles ("Missing still", "no radio_host_portrait FACE still",
    "NO portrait in the ledger", "NO scene still in the ledger"). The LTX-I2V
    gap says "LTX-I2V requires a minted scene still for beat %s; the image
    phase produced no usable path" -- which matches none of them -- so ShotLock
    re-raised, plan-build died, and node 91 never ran. That is the `ltx_video`
    NO_RENDER of the 2026-07-28 engine-coverage campaign: an engine taken off
    the air by a wording mismatch. A raise site now DECLARES what it is instead
    of hoping a reader's substring list still covers its prose.

    NEVER raise this AFTER the image phase. A gap discovered at still-spine
    validation time means image generation actually failed, and deferring there
    would defer to nobody -- it stays a plain ``RenderError`` and stays fatal.
    """
