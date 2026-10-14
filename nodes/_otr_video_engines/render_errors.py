"""THE RENDER ERROR LEAF: render failure types, and nothing else.

WHY THIS MODULE EXISTS. ShotLock needs to catch ONE specific render failure --
a cast-time image gap -- and let every other render failure through. That
needs a named exception type, and the type has to live somewhere both
``render_driver`` and ``otr_shot_lock`` can import without a cycle.
``render_driver`` itself is the obvious-looking home and is the wrong one:
ShotLock would then import the whole driver just to name one exception.

So this is a LEAF. It imports nothing from this package -- nothing from the
registry, the engines, the driver or the shared helpers -- and nothing here
may ever grow an import that changes that. ``render_driver`` re-exports these
names, so every existing ``from .render_driver import RenderError`` keeps
working and the class objects are identical, which is what makes ``isinstance``
and ``except`` behave the same on both import paths.

Cold-import clean: stdlib only.
"""

from __future__ import annotations

import enum


class FailureKind(str, enum.Enum):
    """The label ``render_driver.classify_failure`` puts on a failed render.

    It names the failure in the :class:`RenderError` text and nothing more:
    there is no retry and no fallback on the render path (operator
    2026-07-02), so a kind changes what the operator reads, never what runs.
    Every member is one ``classify_failure`` can return -- the four cloud ones
    through ``render_driver._JOB_CODE_TO_KIND``.
    """

    DEPENDENCY_MISSING = "dependency_missing"
    INVALID_DAG = "invalid_dag"
    OOM = "oom"
    TIMEOUT = "timeout"
    CRASH_BEFORE_LOAD = "crash_before_load"
    CONTENT_REFUSED = "content_refused"
    CORRUPT_OUTPUT = "corrupt_output"
    TRANSIENT_IO = "transient_io"


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

    A raise site DECLARES what it is; ShotLock never infers deferrability by
    matching the wording of an exception message, because a gap whose message
    matches no needle dies at plan-build instead of deferring.

    NEVER raise this AFTER the image phase. A gap discovered at still-spine
    validation time means image generation actually failed, and deferring there
    would defer to nobody -- it stays a plain ``RenderError`` and stays fatal.
    """
