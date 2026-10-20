"""Item progress for a long node's own loop, drawn by ComfyUI's progress bar.

WHY (plan 0f item 2, 2026-09-25). The writer's per-beat loop, the local
voice render and the scene sequencer each run for minutes with nothing on
the node moving, which looks the same as a hang. ComfyUI draws a bar for
whatever node is executing when ``comfy.utils.ProgressBar`` is updated from
the executing thread; this wraps that for an item loop of known length.

WHERE IT IS NOT USED, ON PURPOSE. A node whose work already runs through a
ComfyUI sampler (local stills, Stable Audio 3) shows the sampler's own
per-step bar, and a cloud call shows the partner heartbeat
(``cloud_media_invoke``). A second bar on those nodes would fight the first
for the same slot, so each call site here is a loop that shows nothing
today. The local VIDEO walk is the exception since 2026-10-19: it quiets its
inner samplers with ``QuietInnerBars`` and shows ``WeightedProgress``
instead, because the sampler bar there restarted several times a clip and
the walk is most of an episode's time.

A progress bar is display only: every failure to build or update one is
logged at debug and the render carries on exactly as it would without it.
Outside a running ComfyUI (unit tests, headless scripts) it is a counter.

CANCEL STILL STOPS THE RUN. ComfyUI's hook (main.py ``hijack_progress``)
calls ``throw_exception_if_processing_interrupted`` on every update, and
``InterruptProcessingException`` is a BaseException, so the ``except
Exception`` below never swallows it -- a Cancel pressed mid-loop now lands
at the next item boundary.
"""
from __future__ import annotations

import logging
import sys
import threading

log = logging.getLogger("OTR.progress")


def _comfy(name: str):
    """A ComfyUI module ONLY if it is already loaded, never an import.

    Inside a running ComfyUI these are always loaded, so nothing changes there.
    Outside one (unit tests, headless scripts, a test that fences every
    non-stdlib import) a progress bar must not be the thing that pulls
    ComfyUI in -- the video walk's stdlib-only telemetry test caught exactly
    that on 2026-10-19.
    """
    module = sys.modules.get(name)
    if module is None:
        raise ImportError("%s is not loaded" % name)
    return module


class NodeProgress:
    """A bar over ``total`` items. ``at(n)`` says n items are finished."""

    def __init__(self, total, label: str = ""):
        try:
            self.total = max(int(total or 0), 0)
        except (TypeError, ValueError):
            self.total = 0
        self.done = 0
        self.label = label
        self._bar = None
        if self.total <= 0:
            return
        try:
            self._bar = _comfy("comfy.utils").ProgressBar(self.total)
        except Exception as exc:  # noqa: BLE001 -- headless, tests, or an old Comfy
            log.debug("[OTR progress] %s: no ComfyUI bar (%s)", label, exc)
            self._bar = None

    def at(self, done) -> None:
        """Show ``done`` of ``total`` items finished (clamped to the range)."""
        try:
            value = int(done)
        except (TypeError, ValueError):
            return
        self.done = min(max(value, 0), self.total)
        if self._bar is None:
            return
        try:
            self._bar.update_absolute(self.done, self.total)
        except Exception as exc:  # noqa: BLE001 -- display only, never the render
            log.debug("[OTR progress] %s: bar update failed (%s); hiding it",
                      self.label, exc)
            self._bar = None

    def step(self, n: int = 1) -> None:
        """One more item (or ``n``) finished."""
        self.at(self.done + n)

    def finish(self) -> None:
        self.at(self.total)


class WeightedProgress(NodeProgress):
    """A bar over items that weigh different amounts -- a video beat's frames.

    ``begin(i)`` moves the bar to where item ``i`` starts, whatever happened to
    the items before it (rendered, floored, skipped), so a beat that fails part
    way and is floored is never counted twice. ``part(units)`` adds work done
    inside the current item, held under that item's end.
    """

    def __init__(self, weights, label: str = ""):
        self.weights = []
        for w in weights or ():
            try:
                self.weights.append(max(int(w), 0))
            except (TypeError, ValueError):
                self.weights.append(0)
        super().__init__(sum(self.weights), label)
        self._start = 0
        self._end = 0
        self._inside = 0

    def begin(self, index: int) -> None:
        i = max(0, min(int(index), len(self.weights)))
        self._start = sum(self.weights[:i])
        self._end = self._start + (self.weights[i] if i < len(self.weights) else 0)
        self._inside = 0
        self.at(self._start)

    def part(self, units) -> None:
        try:
            self._inside += max(int(units), 0)
        except (TypeError, ValueError):
            return
        self.at(min(self._start + self._inside, self._end))


class QuietInnerBars:
    """While open, a ComfyUI progress bar BUILT ON THIS THREAD shows nothing.

    WHY (2026-10-19). The local video walk runs ComfyUI's own samplers and
    decoders in process (``wrapper_bridge.run_graph``), and each of them builds
    a ``comfy.utils.ProgressBar`` that reports to the EXECUTING node -- node 92.
    One LTX clip runs three of them in a row (refine sampler, latent upscale,
    tiled decode), so node 92's bar restarted at 0 several times a clip and
    said nothing about how far through the episode the render was. With them
    quiet, the node's own ``WeightedProgress`` (frames rendered of the
    episode's total) is the only thing on that bar.

    HOW. ``ProgressBar`` reads ``comfy.utils.PROGRESS_BAR_HOOK`` when it is
    BUILT (comfy/utils.py), so bars made while this is open get a hook that:
    on this thread, only checks for Cancel (the work ComfyUI's own hook does
    first, so a Cancel still lands at the next step); on any other thread,
    passes straight to the real hook, so nothing else in the process goes
    dark. Bars built BEFORE ``open()`` -- the node's own -- keep the real hook.

    ``close()`` puts the real hook back only if this one is still installed,
    and is safe to call twice. Open it on the thread that renders, once, around
    the whole walk -- never inside code that also runs on worker threads.
    """

    def __init__(self):
        self._real = None
        self._hook = None
        self._utils = None

    def open(self) -> "QuietInnerBars":
        try:
            utils = _comfy("comfy.utils")
            mm = _comfy("comfy.model_management")
        except Exception as exc:  # noqa: BLE001 -- headless, tests, or an old Comfy
            log.debug("[OTR progress] no ComfyUI hook to quiet (%s)", exc)
            return self
        current = getattr(utils, "PROGRESS_BAR_HOOK", None)
        if current is None or self._hook is not None:
            return self
        # Opened over another quiet: keep the REAL hook, never the other quiet,
        # so whichever closes last cannot leave a quiet installed with no owner.
        real = getattr(current, "_otr_real_hook", current)
        owner = threading.get_ident()

        def hook(value, total, preview_image=None, *args, **kwargs):
            # A bar keeps the hook it was built with, so one built while this
            # was open and used after it closed must show normally again.
            if self._hook is not hook or threading.get_ident() != owner:
                return real(value, total, preview_image, *args, **kwargs)
            mm.throw_exception_if_processing_interrupted()
            return None

        hook._otr_real_hook = real
        self._utils, self._real, self._hook = utils, real, hook
        utils.set_progress_bar_global_hook(hook)
        return self

    def close(self) -> None:
        utils, hook = self._utils, self._hook
        self._hook = None
        if utils is not None and hook is not None and getattr(utils, "PROGRESS_BAR_HOOK", None) is hook:
            utils.set_progress_bar_global_hook(self._real)

    def __enter__(self) -> "QuietInnerBars":
        return self.open()

    def __exit__(self, *exc) -> None:
        self.close()


__all__ = ["NodeProgress", "QuietInnerBars", "WeightedProgress"]
