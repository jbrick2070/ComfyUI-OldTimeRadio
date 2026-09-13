"""A dead ComfyUI server must not read as a slow render.

Measured three times on 2026-09-13, once per machine: `poll_history` folded a
refused connection into an empty status dict, and the runner's heartbeat
prints an empty status as `pending`, so a server that had died mid-render
looked identical to one still working. The 4060 sat idle for over an hour
looking healthy.

The FIRST cut of the fix was refuted by two contrarians on the same day, and
these tests are written against what they found rather than against what the
author believed:

  * `requests.exceptions.JSONDecodeError` IS a `RequestException` in 2.32.5,
    so the fixture here raises THAT class, not a stand-in ValueError, and a
    non-JSON body must not be a strike;
  * a read timeout is a slow port, not a dead one -- not a strike;
  * an answering server resets the count, whatever it answered with;
  * twelve refused connections are not a verdict until /queue is asked; a
    living queue means the server is alive and the harness must not be handed
    a FAIL it would follow with /interrupt;
  * after a restart /history answers {} -- so does a normal render -- and the
    only discriminator is an empty /queue, checked and then re-checked against
    history once, because a render can complete between the two GETs;
  * the strike count is printed by the runner or it exists nowhere;
  * the bank gate maps the new failure to HARNESS, not DOWNSTREAM.

No sockets: `requests.get` and `queue_snapshot` are replaced, `time.sleep` is
a no-op.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import requests  # noqa: E402

import otr_api  # noqa: E402

STRIKES = otr_api.UNREACHABLE_STRIKES
EVERY = otr_api.VANISHED_CHECK_EVERY


def _json_error():
    """What requests 2.32 actually raises on a non-JSON body."""
    return requests.exceptions.JSONDecodeError("not json", "<html>502</html>", 0)


class _Resp:
    def __init__(self, payload=None, exc=None):
        self._payload = {} if payload is None else payload
        self._exc = exc

    def json(self):
        if self._exc is not None:
            raise self._exc
        return self._payload


def _refused(*_a, **_k):
    raise requests.ConnectionError("refused")


COMPLETED = {"p1": {"status": {"completed": True}}}
IN_FLIGHT = {}  # a normal render has NO history record until it completes


class _Base(unittest.TestCase):
    def setUp(self):
        self._get = otr_api.requests.get
        self._sleep = otr_api.time.sleep
        self._queue = otr_api.queue_snapshot
        otr_api.time.sleep = lambda *_a, **_k: None
        # Default: a live, busy server -- one prompt running.
        otr_api.queue_snapshot = lambda: (1, 0)
        self.ticks = []

    def tearDown(self):
        otr_api.requests.get = self._get
        otr_api.time.sleep = self._sleep
        otr_api.queue_snapshot = self._queue

    def _tick(self, elapsed, status):
        self.ticks.append(dict(status))

    def _script(self, *responses):
        """requests.get answers each entry in order; an Exception instance is
        raised, anything else is returned as the body; the last repeats."""
        state = {"i": 0}

        def get(*_a, **_k):
            i = min(state["i"], len(responses) - 1)
            state["i"] += 1
            item = responses[i]
            if isinstance(item, BaseException):
                raise item
            return _Resp(item)
        otr_api.requests.get = get
        return state


class NothingListensOnThePort(_Base):
    def test_refused_connections_end_as_fail_only_after_queue_is_unreachable_too(self):
        otr_api.requests.get = _refused
        otr_api.queue_snapshot = lambda: (-1, -1)
        status, err = otr_api.poll_history("p1", timeout_s=0, poll_s=1,
                                           on_tick=self._tick)
        self.assertEqual(status, "FAIL", "a dead server is a failure, not a timeout")
        self.assertTrue(err.startswith(otr_api.SERVER_GONE_MARKER))
        self.assertIn("stopped answering", err)
        self.assertIn(otr_api.COMFYUI_URL, err)
        self.assertIn(str(STRIKES), err)
        self.assertIn("/queue is unreachable", err)
        self.assertNotIn("killed", err, "the message must not guess a cause")
        self.assertEqual(len(self.ticks), STRIKES)

    def test_a_living_queue_overrules_the_strikes(self):
        # Refused twelve times, but /queue answers: the server is alive and
        # /history was the thing failing. Reset and carry on -- never hand the
        # harness a FAIL it will follow with /interrupt.
        calls = {"n": 0}

        def refused_then_fine(*_a, **_k):
            calls["n"] += 1
            if calls["n"] <= STRIKES:
                raise requests.ConnectionError("refused")
            return _Resp(COMPLETED)
        otr_api.requests.get = refused_then_fine
        otr_api.queue_snapshot = lambda: (1, 0)
        status, _ = otr_api.poll_history("p1", timeout_s=0, poll_s=1,
                                         on_tick=self._tick)
        self.assertEqual(status, "SUCCESS")
        self.assertEqual(self.ticks[STRIKES - 1]["unreachable_strikes"], STRIKES)

    def test_the_heartbeat_is_told_unreachable_with_count_and_limit(self):
        otr_api.requests.get = _refused
        otr_api.queue_snapshot = lambda: (-1, -1)
        otr_api.poll_history("p1", timeout_s=0, poll_s=1, on_tick=self._tick)
        self.assertTrue(all(t.get("status_str") == "unreachable" for t in self.ticks))
        self.assertEqual([t["unreachable_strikes"] for t in self.ticks],
                         list(range(1, STRIKES + 1)))
        self.assertTrue(all(t["unreachable_limit"] == STRIKES for t in self.ticks))


class TheServerAnsweredSoItIsNotDead(_Base):
    def test_a_non_json_body_is_not_a_strike(self):
        # The real class. In requests 2.32 it subclasses RequestException, which
        # is exactly why the first cut counted it.
        self.assertTrue(issubclass(requests.exceptions.JSONDecodeError,
                                   requests.RequestException))
        self._script(*([_json_error()] * (STRIKES + 3) + [COMPLETED]))
        status, _ = otr_api.poll_history("p1", timeout_s=0, poll_s=1,
                                         on_tick=self._tick)
        self.assertEqual(status, "SUCCESS")
        self.assertFalse(any(t.get("status_str") == "unreachable" for t in self.ticks))

    def test_a_read_timeout_is_a_slow_port_not_a_dead_one(self):
        self._script(*([requests.ReadTimeout("slow")] * (STRIKES + 3) + [COMPLETED]))
        status, _ = otr_api.poll_history("p1", timeout_s=0, poll_s=1,
                                         on_tick=self._tick)
        self.assertEqual(status, "SUCCESS")
        self.assertFalse(any(t.get("status_str") == "unreachable" for t in self.ticks))

    def test_any_answer_resets_the_count_even_a_garbage_one(self):
        # Reviewer C: with the first cut, a non-transport tick after some
        # strikes froze the counter and kept reporting `unreachable` against a
        # server that had just answered.
        self._script(*([requests.ConnectionError("refused")] * 5
                       + [_json_error()]
                       + [IN_FLIGHT] * 2 + [COMPLETED]))
        status, _ = otr_api.poll_history("p1", timeout_s=0, poll_s=1,
                                         on_tick=self._tick)
        self.assertEqual(status, "SUCCESS")
        after_answer = self.ticks[5:]
        self.assertFalse(any(t.get("status_str") == "unreachable" for t in after_answer),
                         "the tick on which the server answered must not say unreachable")

    def test_a_recovered_server_resets_the_count(self):
        # Two outages of STRIKES-1 each, one good poll between them. Without a
        # reset the second outage would cross the limit. Written as STRIKES-1
        # rather than a half so it keeps discriminating if the constant changes
        # parity (reviewer E).
        self._script(*([requests.ConnectionError("blip")] * (STRIKES - 1)
                       + [IN_FLIGHT]
                       + [requests.ConnectionError("blip")] * (STRIKES - 1)
                       + [COMPLETED]))
        otr_api.queue_snapshot = lambda: (-1, -1)  # would be fatal if reached
        status, _ = otr_api.poll_history("p1", timeout_s=0, poll_s=1,
                                         on_tick=self._tick)
        self.assertEqual(status, "SUCCESS")
        self.assertLess(max(t.get("unreachable_strikes", 0) for t in self.ticks), STRIKES)


class TheServerRestartedAndLostTheRender(_Base):
    def test_empty_history_plus_empty_queue_is_named_as_vanished(self):
        self._script(IN_FLIGHT)
        otr_api.queue_snapshot = lambda: (0, 0)
        status, err = otr_api.poll_history("p1", timeout_s=0, poll_s=1,
                                           on_tick=self._tick)
        self.assertEqual(status, "FAIL")
        self.assertTrue(err.startswith(otr_api.SERVER_GONE_MARKER))
        self.assertIn("no record of prompt p1", err)
        self.assertIn("nothing is running or queued", err)
        # Declared at the first check, not before: EVERY ticks in, plus the
        # confirming second look at history.
        self.assertEqual(len(self.ticks), EVERY)

    def test_an_in_flight_render_is_never_called_vanished(self):
        # A normal render has no history record either. The queue says busy.
        self._script(*([IN_FLIGHT] * (3 * EVERY) + [COMPLETED]))
        otr_api.queue_snapshot = lambda: (1, 0)
        status, _ = otr_api.poll_history("p1", timeout_s=0, poll_s=1)
        self.assertEqual(status, "SUCCESS")

    def test_a_render_that_completes_between_the_two_gets_is_not_vanished(self):
        # history GET -> {} ; queue GET -> (0,0) because it JUST finished ;
        # the confirming history GET -> completed. Must not be declared gone.
        self._script(*([IN_FLIGHT] * EVERY + [COMPLETED]))
        otr_api.queue_snapshot = lambda: (0, 0)
        status, _ = otr_api.poll_history("p1", timeout_s=0, poll_s=1)
        self.assertEqual(status, "SUCCESS")


class TheContractsAroundIt(_Base):
    def test_timeout_is_still_timeout_when_the_server_is_merely_slow(self):
        otr_api.requests.get = lambda *_a, **_k: _Resp(IN_FLIGHT)
        clock = {"t": 0.0}
        real_time = otr_api.time.time

        def fake_time():
            clock["t"] += 3
            return clock["t"]
        otr_api.time.time = fake_time
        try:
            status, err = otr_api.poll_history("p1", timeout_s=10, poll_s=1)
        finally:
            otr_api.time.time = real_time
        self.assertEqual((status, err), ("TIMEOUT", ""))

    def test_the_runner_prints_the_strike_count(self):
        src = (Path(__file__).resolve().parents[1] / "scripts"
               / "otr_canonical_api_run.py").read_text(encoding="utf-8")
        self.assertIn("unreachable_strikes", src,
                      "a count nobody prints exists nowhere a person can read it")
        self.assertIn("unreachable_limit", src)

    def test_the_bank_gate_calls_a_gone_server_harness_not_downstream(self):
        import otr_writer_bank_gate as gate
        log = ("[canonical-api] RESULT FAIL prompt_id=x\n"
               f"{otr_api.SERVER_GONE_MARKER}: the ComfyUI server at http://h "
               "stopped answering -- 12 consecutive refused connections\n")
        self.assertEqual(gate.classify_failure(log), "HARNESS")
        # and the spelling is pinned in both files
        gate_src = (Path(__file__).resolve().parents[1] / "scripts"
                    / "otr_writer_bank_gate.py").read_text(encoding="utf-8")
        self.assertIn('"%s"' % otr_api.SERVER_GONE_MARKER, gate_src)


if __name__ == "__main__":
    unittest.main()
