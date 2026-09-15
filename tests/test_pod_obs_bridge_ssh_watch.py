"""SSH --watch must wait until work has been seen, then idle-exit."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "scripts" / "otr_pod_obs_bridge.py"


def _load():
    spec = importlib.util.spec_from_file_location("otr_pod_obs_bridge_under_test", SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _args(**kwargs):
    ns = argparse.Namespace(
        dest="obs",
        host="1.2.3.4",
        port=22,
        max_wait_s=3600,
        poll_s=60,
    )
    for key, value in kwargs.items():
        setattr(ns, key, value)
    return ns


def test_watch_over_ssh_is_the_default_watch_route():
    text = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "def watch_over_ssh(" in text
    assert "if args.watch:\n            return watch_over_ssh(args)" in text
    assert "if not args.http:\n        return sync_over_ssh(args)" not in text


def test_http_watch_uses_the_same_idle_helper():
    text = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text.count("_watch_stop_on_idle(") >= 3
    assert "if counts == (0, 0):" not in text
    assert "if run == 0 and pend == 0:" not in text
    assert "except (subprocess.TimeoutExpired, OSError):" in text


def test_first_empty_queue_is_not_idle_exit():
    assert _load()._watch_stop_on_idle((0, 0), False) == (False, False)


def test_failed_poll_is_not_idle():
    assert _load()._watch_stop_on_idle(None, False) == (False, False)
    assert _load()._watch_stop_on_idle(None, True) == (False, True)


def test_idle_exit_only_after_work_was_seen():
    mod = _load()
    assert mod._watch_stop_on_idle((1, 0), False) == (False, True)
    assert mod._watch_stop_on_idle((0, 3), False) == (False, True)
    assert mod._watch_stop_on_idle((0, 0), True) == (True, True)


def test_ssh_queue_counts_timeout_returns_none(monkeypatch):
    import subprocess
    mod = _load()

    def boom(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=30)

    monkeypatch.setattr(subprocess, "run", boom)
    assert mod.ssh_queue_counts(_args(key="k")) is None


def test_watch_over_ssh_ignores_leading_idle_then_exits_after_busy(monkeypatch, capsys):
    mod = _load()
    polls = iter([(0, 0), (0, 0), (1, 0), (0, 0)])
    syncs = []
    clock = {"t": 0.0}

    monkeypatch.setattr(mod.time, "time", lambda: clock["t"])
    monkeypatch.setattr(mod.time, "sleep", lambda s: clock.__setitem__("t", clock["t"] + s))
    monkeypatch.setattr(mod, "ssh_queue_counts", lambda _args: next(polls))
    monkeypatch.setattr(mod, "sync_over_ssh", lambda _args: syncs.append(clock["t"]) or 0)

    rc = mod.watch_over_ssh(_args())
    assert rc == 0
    # start sync + two idle polls + busy poll + idle-exit sync
    assert syncs[0] == 0.0
    assert len(syncs) >= 4
    out = capsys.readouterr().out
    assert "queue idle after 0 min" not in out
    assert "queue idle after" in out
    # A one-shot `return sync_over_ssh(args)` would sync once and print no idle.
    assert len(syncs) > 1
