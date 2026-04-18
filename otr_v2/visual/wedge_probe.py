"""
otr_v2.visual.wedge_probe  --  diagnostic event probe for live graph runs
===========================================================================

Purpose
-------
Trace which node in the live ComfyUI graph is dropping frames, silent-lipping,
or drifting audio sync during an episode run.  Writes an NDJSON event log
(``logs/wedge_probe.ndjson`` by default) that the post-run analyser can read
to pinpoint the stage where the pathology was introduced.

Design rules (see round-robin consult 2026-04-17, Gemini)
---------------------------------------------------------
1. **Zero-cost when disabled.**  The probe is a singleton; if
   ``OTR_WEDGE_PROBE`` is absent or not truthy, every public method is a
   no-op returning immediately (O(1)).  No thread, no queue, no file.

2. **Never blocks the main thread.**  When enabled, events are enqueued
   on an in-memory bounded queue and drained by a single background
   writer thread.  An overflow drops the oldest event instead of waiting
   on a full queue, so the probe cannot stall the graph.

3. **Never modifies the audio path (C7).**  The probe records *metadata
   about* audio events (sample counts, durations, ffmpeg exit codes) but
   never touches the audio tensor, never invokes ffmpeg, never rewrites
   mux flags.  Import from the renderer is observer-only.

4. **Pure stdlib.**  No torch, no numpy, no ComfyUI imports.  Safe for
   CI and unit tests.

Public API
----------
::

    from otr_v2.visual import wedge_probe

    probe = wedge_probe.get_probe()          # singleton, enable via env

    probe.event("backend_start", backend="flux_anchor", shot_id="s001")
    with probe.span("render_mux", shot_id="s001") as s:
        s.note(frames=240, audio_samples=264600)
        ...  # ffmpeg mux here
    probe.counter("frames_written", 240, shot_id="s001")

    probe.flush()                            # optional explicit drain
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import threading
import time
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---- Env gating ------------------------------------------------------------

_ENV_ENABLE = "OTR_WEDGE_PROBE"
_ENV_LOG_PATH = "OTR_WEDGE_PROBE_LOG"
_ENV_QUEUE_MAX = "OTR_WEDGE_PROBE_QUEUE_MAX"

# Default log target.  Lives under logs/ (gitignored) so probe output never
# pollutes commits.  The actual path resolved at probe-construction time.
_DEFAULT_LOG_RELPATH = "logs/wedge_probe.ndjson"

# Bounded queue so a runaway producer cannot eat RAM.  Overflow drops the
# oldest event so the probe degrades gracefully under load.
_DEFAULT_QUEUE_MAX = 10000


def _env_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


# ---- Event record ----------------------------------------------------------


@dataclass
class ProbeEvent:
    """One probe event.  ``kind`` is the free-form event tag; ``fields``
    carries arbitrary JSON-safe payload."""

    t_monotonic: float
    t_unix: float
    kind: str
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "t_monotonic": round(self.t_monotonic, 6),
            "t_unix": round(self.t_unix, 6),
            "kind": self.kind,
        }
        # Merge fields at top-level for easy jq querying, but keep core
        # timing/kind keys reserved so a payload field cannot shadow them.
        for k, v in self.fields.items():
            if k in {"t_monotonic", "t_unix", "kind"}:
                out[f"_{k}"] = v
            else:
                out[k] = v
        return out


# ---- Span context manager --------------------------------------------------


class ProbeSpan(AbstractContextManager["ProbeSpan"]):
    """A timed span.  On exit, emits a single event with ``elapsed_ms``
    plus any notes added inside the ``with`` block."""

    def __init__(self, probe: "WedgeProbe", kind: str, base_fields: dict[str, Any]) -> None:
        self._probe = probe
        self._kind = kind
        self._base = dict(base_fields)
        self._notes: dict[str, Any] = {}
        self._t_start_mono: float = 0.0
        self._t_start_unix: float = 0.0

    def __enter__(self) -> "ProbeSpan":
        self._t_start_mono = time.monotonic()
        self._t_start_unix = time.time()
        return self

    def note(self, **fields: Any) -> None:
        """Attach payload to the span; emitted on exit."""
        self._notes.update(fields)

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        t_end = time.monotonic()
        elapsed_ms = round((t_end - self._t_start_mono) * 1000.0, 3)
        fields: dict[str, Any] = dict(self._base)
        fields.update(self._notes)
        fields["elapsed_ms"] = elapsed_ms
        if exc_type is not None:
            fields["exc_type"] = exc_type.__name__
        self._probe._emit_raw(
            ProbeEvent(
                t_monotonic=self._t_start_mono,
                t_unix=self._t_start_unix,
                kind=f"{self._kind}.span",
                fields=fields,
            )
        )


# ---- No-op span (returned when probe disabled) ------------------------------


class _NoopSpan(AbstractContextManager["_NoopSpan"]):
    def __enter__(self) -> "_NoopSpan":
        return self

    def note(self, **fields: Any) -> None:
        pass

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        return None


_NOOP_SPAN = _NoopSpan()


# ---- The probe -------------------------------------------------------------


class WedgeProbe:
    """Thread-safe event probe.  Use :func:`get_probe` to retrieve the
    process-wide singleton; construct directly only in tests."""

    def __init__(
        self,
        *,
        enabled: bool,
        log_path: Path | None,
        queue_max: int,
    ) -> None:
        self._enabled = bool(enabled)
        self._log_path = log_path
        self._queue_max = int(queue_max)
        self._queue: "queue.Queue[ProbeEvent | None]" = queue.Queue(maxsize=self._queue_max)
        self._dropped_count = 0
        self._dropped_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()

        if self._enabled:
            if self._log_path is None:
                raise ValueError("WedgeProbe enabled but log_path is None")
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._worker = threading.Thread(
                target=self._drain_loop,
                name="wedge-probe-drain",
                daemon=True,
            )
            self._worker.start()

    # -- Public API ----------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def log_path(self) -> Path | None:
        return self._log_path

    @property
    def dropped_count(self) -> int:
        with self._dropped_lock:
            return self._dropped_count

    def event(self, kind: str, **fields: Any) -> None:
        """Record a single event.  Non-blocking; drops oldest on overflow."""
        if not self._enabled:
            return
        evt = ProbeEvent(
            t_monotonic=time.monotonic(),
            t_unix=time.time(),
            kind=kind,
            fields=fields,
        )
        self._emit_raw(evt)

    def span(self, kind: str, **fields: Any) -> AbstractContextManager[Any]:
        """Start a timed span.  Returns a context manager; use ``.note()``
        inside the ``with`` to attach payload."""
        if not self._enabled:
            return _NOOP_SPAN
        return ProbeSpan(self, kind, fields)

    def counter(self, kind: str, value: int, **fields: Any) -> None:
        """Record a counter event (integer value in ``n``)."""
        if not self._enabled:
            return
        evt = ProbeEvent(
            t_monotonic=time.monotonic(),
            t_unix=time.time(),
            kind=f"{kind}.counter",
            fields={"n": int(value), **fields},
        )
        self._emit_raw(evt)

    def flush(self, timeout_s: float = 5.0) -> None:
        """Force-drain the queue.  Returns when queue is empty or timeout."""
        if not self._enabled:
            return
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._queue.empty():
                return
            time.sleep(0.02)

    def shutdown(self, timeout_s: float = 2.0) -> None:
        """Stop the drain thread and flush.  Idempotent."""
        if not self._enabled:
            return
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)  # sentinel
        except queue.Full:
            pass
        if self._worker is not None:
            self._worker.join(timeout=timeout_s)

    # -- Internals -----------------------------------------------------------

    def _emit_raw(self, evt: ProbeEvent) -> None:
        """Enqueue an event; drop oldest on overflow."""
        try:
            self._queue.put_nowait(evt)
        except queue.Full:
            # Drop the oldest to make room -- we never block the producer.
            try:
                self._queue.get_nowait()
                with self._dropped_lock:
                    self._dropped_count += 1
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(evt)
            except queue.Full:
                with self._dropped_lock:
                    self._dropped_count += 1

    def _drain_loop(self) -> None:
        """Background writer.  Appends NDJSON lines until stop + drained."""
        assert self._log_path is not None
        # Line-buffered append for crash-safety.  utf-8 always.
        with open(self._log_path, "a", encoding="utf-8", buffering=1) as fh:
            while True:
                try:
                    evt = self._queue.get(timeout=0.25)
                except queue.Empty:
                    if self._stop_event.is_set():
                        return
                    continue
                if evt is None:
                    return
                try:
                    line = json.dumps(evt.to_dict(), ensure_ascii=False)
                    fh.write(line)
                    fh.write("\n")
                except (TypeError, ValueError):
                    # Payload not JSON-serialisable -- record a marker and
                    # move on; never raise from the drain thread.
                    try:
                        fh.write(
                            json.dumps(
                                {
                                    "t_monotonic": round(evt.t_monotonic, 6),
                                    "t_unix": round(evt.t_unix, 6),
                                    "kind": f"{evt.kind}.serialize_error",
                                    "fields_keys": sorted(list(evt.fields.keys())),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    except (OSError, ValueError):
                        pass


# ---- Singleton accessor ----------------------------------------------------

_singleton_lock = threading.Lock()
_singleton: WedgeProbe | None = None


def _resolve_log_path() -> Path:
    """Resolve the NDJSON log path.  Env override wins; otherwise relative
    to this module's package (``otr_v2/../logs/wedge_probe.ndjson``)."""
    override = os.environ.get(_ENV_LOG_PATH, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    pkg_root = Path(__file__).resolve().parent.parent.parent  # repo root
    return pkg_root / _DEFAULT_LOG_RELPATH


def _resolve_queue_max() -> int:
    raw = os.environ.get(_ENV_QUEUE_MAX, "").strip()
    if not raw:
        return _DEFAULT_QUEUE_MAX
    try:
        value = int(raw)
        return max(100, value)
    except ValueError:
        return _DEFAULT_QUEUE_MAX


def get_probe() -> WedgeProbe:
    """Return the process-wide probe singleton.  Safe to call many times;
    constructs lazily on first call based on ``OTR_WEDGE_PROBE``."""
    global _singleton
    if _singleton is not None:
        return _singleton
    with _singleton_lock:
        if _singleton is not None:
            return _singleton
        enabled = _env_truthy(os.environ.get(_ENV_ENABLE))
        if enabled:
            probe = WedgeProbe(
                enabled=True,
                log_path=_resolve_log_path(),
                queue_max=_resolve_queue_max(),
            )
            atexit.register(probe.shutdown)
        else:
            probe = WedgeProbe(
                enabled=False,
                log_path=None,
                queue_max=_DEFAULT_QUEUE_MAX,
            )
        _singleton = probe
        return probe


def reset_singleton_for_tests() -> None:
    """Test-only: drop the singleton so the next ``get_probe()`` re-reads
    env vars.  Shuts down the existing worker thread if any."""
    global _singleton
    with _singleton_lock:
        if _singleton is not None:
            try:
                _singleton.shutdown(timeout_s=1.0)
            except Exception:
                pass
            _singleton = None
