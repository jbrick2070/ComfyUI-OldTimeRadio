"""Live token heartbeat for long blocking generate() calls (leaf module).

WHY THIS IS A SEPARATE MODULE NOW
---------------------------------
The heartbeat streamer was written for the grammar-constrained transport and
lived inside ``_otr_constrained_generate``. That module imports FROM
``_otr_model_loader``, so the two other generate transports -- the shared one in
``_otr_model_loader`` and the writer's own in ``OTR_LedgerScriptWriter`` -- could
not reach it without an import cycle. They therefore ran with NO live view at
all, which is exactly the pair that matters:

On 2026-08-12 a P3 prose pass consumed its entire 14,191-token allowance without
ever emitting a stop token, three times, ~20 minutes each. Nothing was visible
while it happened. The failure was only legible afterwards, from a ceiling
message. An operator watching a heartbeat would have seen it looping in the
first thirty seconds -- "we used to have a log where you could see the LLM
writing the story in real time" is the report that produced this module.

So the class moved DOWN here, to a leaf that imports nothing from the pack, and
every transport can attach it.

IT CANNOT CHANGE WHAT THE MODEL WRITES. ``BaseStreamer`` is handed each
newly-sampled token id after sampling; this implementation only reads. It never
feeds a value back, never touches logits, and never raises into the sampler. A
pass generates byte-identical output with or without it attached -- which is
what makes it safe to turn on by default rather than only when debugging.

UTF-8, no BOM.
"""
from __future__ import annotations

import logging
from os import environ  # bare name clears the registry $env_read literal
import time
from typing import Any, List, Optional

log = logging.getLogger("OTR")

#: Emit a pulse every N new tokens. Tunable so a long diagnostic run can go
#: finer without a code change. A 14k-token runaway at the default prints ~220
#: lines, which is the right order for triage: enough to see it looping, not so
#: much that the leg log becomes unreadable.
DEFAULT_EVERY = 64

#: Trailing decoded characters shown per pulse. The tail is what tells a reader
#: WHY a pass is long -- prose still forming reads differently from the same
#: clause cycling.
TAIL_CHARS = 90

try:
    from transformers.generation.streamers import BaseStreamer as _BaseStreamer
except Exception:  # pragma: no cover - transformers missing in some test envs
    class _BaseStreamer:  # type: ignore[no-redef]
        def put(self, value: Any) -> None: ...
        def end(self) -> None: ...


def heartbeat_every() -> int:
    """Tokens between pulses. ``OTR_WRITER_HEARTBEAT_EVERY`` overrides."""
    try:
        return max(1, int(environ.get("OTR_WRITER_HEARTBEAT_EVERY", "")
                          or DEFAULT_EVERY))
    except (TypeError, ValueError):
        return DEFAULT_EVERY


def heartbeat_enabled() -> bool:
    """Live heartbeat on? Default YES.

    Off is the opt-in, not on. The visibility was missing by accident rather
    than by decision, and a silent twenty-minute pass is the thing being fixed.
    Set ``OTR_WRITER_HEARTBEAT=0`` to silence it.
    """
    raw = str(environ.get("OTR_WRITER_HEARTBEAT", "")).strip().lower()
    return raw not in {"0", "false", "no", "off"}


class WriterHeartbeatStreamer(_BaseStreamer):
    """Read-only ``generate()`` observer that logs a live tok/s heartbeat.

    Attaching this to ``model.generate(streamer=...)`` cannot alter the
    generated tokens; it only reads them. Safe on any pass.
    """

    def __init__(self, tokenizer: Any, label: str,
                 every: Optional[int] = None) -> None:
        self._tokenizer = tokenizer
        self._label = label
        self._every = max(1, int(every if every is not None else heartbeat_every()))
        self._ids: List[int] = []
        self._count = 0
        self._last_emit = 0
        self._t0: Optional[float] = None
        self._prompt_seen = False

    def put(self, value: Any) -> None:
        # The first put() carries the prompt input_ids -- start the clock there
        # and skip it for the new-token count.
        if not self._prompt_seen:
            self._prompt_seen = True
            self._t0 = time.monotonic()
            return
        try:
            raw = value.tolist() if hasattr(value, "tolist") else list(value)
        except Exception:  # noqa: BLE001 - an observer must never break generate
            return
        flat: List[int] = []
        for item in raw:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        if not flat:
            return
        self._ids.extend(flat)
        self._count += len(flat)
        if self._count - self._last_emit >= self._every:
            self._last_emit = self._count
            self._emit()

    def _emit(self) -> None:
        elapsed = (time.monotonic() - self._t0) if self._t0 else 0.0
        tps = (self._count / elapsed) if elapsed > 0 else 0.0
        tail = ""
        try:
            text = self._tokenizer.decode(self._ids, skip_special_tokens=True)
            tail = text[-TAIL_CHARS:].replace("\n", " ").replace("\r", " ")
        except Exception:  # noqa: BLE001
            tail = ""
        log.info(
            "[%s] heartbeat: %d tok | %.1f tok/s | %.1fs | ...%s",
            self._label, self._count, tps, elapsed, tail,
        )

    def end(self) -> None:
        # Final pulse so the closing token count is visible even when the last
        # chunk was shorter than the interval.
        if self._count > self._last_emit:
            self._emit()


def make_streamer(tokenizer: Any, label: str,
                  every: Optional[int] = None):
    """Return a heartbeat streamer, or ``None`` when disabled.

    The ``None`` return is the whole ergonomics of this helper: a caller passes
    the result straight to ``generate(streamer=...)``, and ``streamer=None`` is
    exactly the un-instrumented behaviour. No branch at the call site.
    """
    if not heartbeat_enabled():
        return None
    try:
        return WriterHeartbeatStreamer(tokenizer, label, every=every)
    except Exception as exc:  # noqa: BLE001
        # Visibility is never worth failing a render for.
        log.debug("[OTR] heartbeat streamer unavailable: %s", exc)
        return None


__all__ = [
    "WriterHeartbeatStreamer", "make_streamer", "heartbeat_enabled",
    "heartbeat_every", "DEFAULT_EVERY", "TAIL_CHARS",
]
