"""A long writer pass must be VISIBLE while it runs, not only after it fails.

WHY THIS EXISTS. On 2026-08-12 a P3 prose pass consumed its entire 14,191-token
allowance without ever emitting a stop token -- three times, roughly twenty
minutes each -- and nothing was observable while it happened. The only signal
arrived at the ceiling, as a truncation message. The operator's report was
blunt: "we used to have a log where you could see the LLM writing the story in
real time ... we could probably better triage what it's actually doing."

The heartbeat existed, but only on the grammar-constrained transport. The two
that matter -- the writer's own `model.generate` in `OTR_LedgerScriptWriter` and
the shared one in `_otr_model_loader` -- had NO streamer, because the class lived
in a module that imports FROM `_otr_model_loader` and so could not be reached
from it. It moved to a leaf and all three transports now attach it.

The load-bearing property is that attaching it CANNOT change what the model
writes: `generate()` hands a `BaseStreamer` each sampled token id, and this one
only reads. That is what makes default-on defensible, so it is asserted here
rather than assumed.
"""
from __future__ import annotations

import logging

import pytest

from nodes import _otr_writer_heartbeat as HB


class _FakeTokenizer:
    """Decodes ids to letters. Enough for the streamer's tail rendering."""

    def decode(self, ids, skip_special_tokens=True):  # noqa: ARG002
        return "".join(chr(97 + (int(i) % 26)) for i in ids)


def _feed(streamer, n, start=0):
    """Drive the streamer like generate() does: prompt first, then tokens."""
    streamer.put([list(range(5))])           # the prompt put()
    for i in range(n):
        streamer.put([start + i])


# ---------------------------------------------------------------------------
# It emits, and it emits something a reader can act on
# ---------------------------------------------------------------------------

def test_it_pulses_while_generating(caplog):
    caplog.set_level(logging.INFO, logger="OTR")
    s = HB.WriterHeartbeatStreamer(_FakeTokenizer(), "P3", every=10)
    _feed(s, 35)
    pulses = [r for r in caplog.records if "heartbeat" in r.getMessage()]
    assert len(pulses) == 3, f"expected a pulse every 10 tokens, got {len(pulses)}"
    msg = pulses[-1].getMessage()
    # The four things that make a pulse useful for triage.
    assert "P3" in msg, "the pulse does not say WHICH pass"
    assert "tok" in msg and "tok/s" in msg, "no rate -- cannot tell slow from stuck"
    assert "..." in msg, "no decoded tail -- cannot tell prose from a loop"


def test_the_final_pulse_is_not_lost(caplog):
    """A pass that ends mid-interval must still report its closing count."""
    caplog.set_level(logging.INFO, logger="OTR")
    s = HB.WriterHeartbeatStreamer(_FakeTokenizer(), "P5", every=100)
    _feed(s, 30)                     # below the interval: nothing emitted yet
    assert not [r for r in caplog.records if "heartbeat" in r.getMessage()]
    s.end()
    tail = [r for r in caplog.records if "heartbeat" in r.getMessage()]
    assert len(tail) == 1 and "30 tok" in tail[0].getMessage()


def test_the_tail_shows_a_LOOP_differently_from_prose(caplog):
    """The tail is the point: a runaway looks different from a long pass.

    This is the signal that was missing. A pass cycling one clause and a pass
    genuinely writing produce different tails, and a reader can tell them apart
    in seconds instead of waiting for a ceiling twenty minutes later.
    """
    caplog.set_level(logging.INFO, logger="OTR")
    s = HB.WriterHeartbeatStreamer(_FakeTokenizer(), "P3", every=20)
    s.put([list(range(5))])
    for _ in range(20):
        s.put([0])                   # the same token, over and over
    looped = [r for r in caplog.records if "heartbeat" in r.getMessage()][-1]
    assert "aaaa" in looped.getMessage(), "a repeating stream is not visible in the tail"


# ---------------------------------------------------------------------------
# It must never change or break the generation
# ---------------------------------------------------------------------------

def test_it_is_read_only_and_returns_nothing_to_the_sampler():
    """put() must have no return value that generate() could act on."""
    s = HB.WriterHeartbeatStreamer(_FakeTokenizer(), "probe", every=5)
    s.put([list(range(3))])
    assert s.put([1]) is None
    assert s.end() is None


def test_a_broken_tokenizer_cannot_kill_a_render(caplog):
    """An observer that raises would fail a render for the sake of a log line."""
    caplog.set_level(logging.INFO, logger="OTR")

    class Exploding:
        def decode(self, *a, **k):
            raise RuntimeError("tokenizer is unhappy")

    s = HB.WriterHeartbeatStreamer(Exploding(), "P3", every=2)
    _feed(s, 6)                      # must not raise
    assert [r for r in caplog.records if "heartbeat" in r.getMessage()]


def test_malformed_token_payloads_are_ignored_not_raised():
    s = HB.WriterHeartbeatStreamer(_FakeTokenizer(), "probe", every=2)
    s.put([list(range(3))])
    s.put(object())                  # not list-able
    s.put([])                        # empty
    s.put([[1, 2], 3])               # nested + flat mixed


# ---------------------------------------------------------------------------
# The switch
# ---------------------------------------------------------------------------

def test_it_is_ON_by_default(monkeypatch):
    """Default-on is the fix. The visibility was lost by accident, not chosen."""
    monkeypatch.delenv("OTR_WRITER_HEARTBEAT", raising=False)
    assert HB.heartbeat_enabled() is True
    assert HB.make_streamer(_FakeTokenizer(), "P3") is not None


@pytest.mark.parametrize("off", ["0", "false", "no", "off", "OFF"])
def test_it_can_be_silenced(monkeypatch, off):
    monkeypatch.setenv("OTR_WRITER_HEARTBEAT", off)
    assert HB.heartbeat_enabled() is False
    # None is what the call sites pass straight to generate(streamer=...),
    # which is exactly the un-instrumented behaviour.
    assert HB.make_streamer(_FakeTokenizer(), "P3") is None


def test_the_interval_is_tunable(monkeypatch):
    monkeypatch.setenv("OTR_WRITER_HEARTBEAT_EVERY", "8")
    assert HB.heartbeat_every() == 8
    monkeypatch.setenv("OTR_WRITER_HEARTBEAT_EVERY", "nonsense")
    assert HB.heartbeat_every() == HB.DEFAULT_EVERY


# ---------------------------------------------------------------------------
# The transports that were blind
# ---------------------------------------------------------------------------

def test_every_local_generate_transport_now_attaches_one():
    """The regression that produced this file: two of three ran blind.

    Asserted by reading the sources, because the alternative is a live GPU pass.
    A future edit that drops the streamer from one transport puts that pass back
    in the dark, and this is what says so.
    """
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1] / "nodes"
    for name in ("OTR_LedgerScriptWriter.py", "_otr_model_loader.py",
                 "_otr_constrained_generate.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "streamer" in src, f"{name} has no streamer -- it generates blind"

    # And the leaf must import nothing from the pack, or it cannot be reached
    # from _otr_model_loader (which is how this was broken in the first place).
    leaf = (root / "_otr_writer_heartbeat.py").read_text(encoding="utf-8")
    assert "from ._otr" not in leaf and "from . import _otr" not in leaf, (
        "the heartbeat leaf grew a pack import -- that reintroduces the cycle "
        "that kept two transports blind")
