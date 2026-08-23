"""A poll TIMEOUT is about the watcher, not about the render.

FOUND LIVE, 2026-08-23, running item F's `otr_g4_wan_ti2v` leg. The canonical
runner printed

    [canonical-api] RESULT TIMEOUT prompt_id=60d23b04-...

and exited 1 at t=5396s. The render was fine. The server reported one prompt
RUNNING, the GPU sat at 98%, and the wan clip count went on climbing from 21 to
33 while the runner was declaring failure. `--timeout` defaults to 5400s and a
full wan_ti2v episode on the 16 GB box takes longer than that, so this is the
NORMAL outcome for the slowest shipped lane -- not an exception.

The defect was never the timeout itself; `--timeout 0` is documented as the
operator mode for long lanes. The defect is that ONE message covered two
opposite situations -- "your render is still going" and "your render is gone" --
and the terminal-sounding one is what a reader believes. That is how a healthy
90-minute render gets killed and restarted by someone being careful.

`classify_timeout` is pure so this can be pinned without a server.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


def _runner():
    import otr_canonical_api_run as runner  # noqa: PLC0415
    return runner


@pytest.mark.parametrize("running,pending", [
    (1, 0),      # the live case observed on 2026-08-23
    (0, 1),      # queued but not started
    (1, 3),      # busy server
    (2, 0),
])
def test_a_busy_queue_means_the_render_survived(running, pending):
    assert _runner().classify_timeout(running, pending) == "still_running", (
        "a TIMEOUT with work still on the queue must NOT read as a dead "
        "render -- that misreading is what gets a healthy 90-minute wan leg "
        "killed and restarted.")


def test_an_empty_queue_means_it_really_ended():
    assert _runner().classify_timeout(0, 0) == "not_running"


@pytest.mark.parametrize("running,pending", [(-1, -1), (-1, 0), (0, -1)])
def test_an_unreadable_queue_is_reported_as_unknown_not_guessed(running, pending):
    """queue_snapshot is best-effort and returns -1/-1 when it cannot read.

    Absence of evidence must not be rendered as either verdict. Guessing
    "still running" invents a live render; guessing "not running" invents a
    dead one.
    """
    assert _runner().classify_timeout(running, pending) == "unknown"


def test_the_three_outcomes_are_distinct():
    """If two collapse to one string the caller cannot tell them apart again."""
    r = _runner()
    got = {r.classify_timeout(1, 0), r.classify_timeout(0, 0),
           r.classify_timeout(-1, -1)}
    assert len(got) == 3, f"outcomes collapsed: {got}"


def test_the_runner_can_actually_reach_the_queue_helper():
    """The branch is useless if the import is missing -- it would NameError
    at the exact moment a long render times out, which is the worst time."""
    r = _runner()
    assert callable(getattr(r, "queue_snapshot", None)), (
        "otr_canonical_api_run must import queue_snapshot from otr_api; "
        "without it the TIMEOUT branch raises instead of explaining itself.")
