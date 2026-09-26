"""Item progress for a long node's own loop, drawn by ComfyUI's progress bar.

WHY (plan 0f item 2, 2026-09-25). The writer's per-beat loop, the local
voice render and the scene sequencer each run for minutes with nothing on
the node moving, which looks the same as a hang. ComfyUI draws a bar for
whatever node is executing when ``comfy.utils.ProgressBar`` is updated from
the executing thread; this wraps that for an item loop of known length.

WHERE IT IS NOT USED, ON PURPOSE. A node whose work already runs through a
ComfyUI sampler (local stills, local video, Stable Audio 3) shows the
sampler's own per-step bar, and a cloud call shows the partner heartbeat
(``cloud_media_invoke``). A second bar on those nodes would fight the first
for the same slot, so each call site here is a loop that shows nothing
today.

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

log = logging.getLogger("OTR.progress")


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
            from comfy.utils import ProgressBar
            self._bar = ProgressBar(self.total)
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


__all__ = ["NodeProgress"]
