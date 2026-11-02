"""The command line: the safety rails and the doctor."""
from __future__ import annotations

import pytest

from radio_play_studio import cli


def test_serve_refuses_to_listen_beyond_this_machine(capsys):
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 2
    assert "no login" in capsys.readouterr().err


def test_start_refuses_a_wide_host_before_starting_anything(monkeypatch, capsys):
    monkeypatch.setattr(cli, "start_bridge", lambda s: pytest.fail("a bridge was started"))
    assert cli.main(["start", "--host", "0.0.0.0"]) == 2
    assert "no login" in capsys.readouterr().err


def test_doctor_reports_each_thing_the_app_needs(monkeypatch, settings, schemas):
    monkeypatch.setattr(cli, "fetch_object_info", lambda url: schemas)
    rows = {what: ok for ok, what, _ in cli.doctor_report(settings)}
    assert rows["workflow file"] and rows["ComfyUI answers"]
    assert rows["Old-Time Radio pack installed"] and rows["form fields resolve"]
    assert rows["comfy-api-proxy answers"] is False        # nothing listens at bridge.invalid


def test_doctor_names_a_missing_pack(monkeypatch, settings):
    monkeypatch.setattr(cli, "fetch_object_info", lambda url: {})
    detail = {what: d for ok, what, d in cli.doctor_report(settings)}
    assert "OTR_LedgerScriptWriter" in detail["Old-Time Radio pack installed"]


class _Result:
    def __init__(self, code=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = code, out, err


def _local(settings):
    import dataclasses
    return dataclasses.replace(settings, comfyui_url="http://127.0.0.1:8188",
                               api_url="http://127.0.0.1:8189")


def test_start_leaves_a_running_bridge_alone(settings):
    ran = []
    said = cli.start_bridge(_local(settings), run=lambda *a, **k: ran.append(a),
                            reachable=lambda url, timeout: True, command=lambda: ["proxy"])
    assert "already answers" in said and ran == []


def test_start_starts_the_bridge_for_this_comfyui_and_waits_for_it(settings):
    ran, looks = [], iter([False, False, True])
    said = cli.start_bridge(_local(settings),
                            run=lambda cmd, **k: ran.append(cmd) or _Result(),
                            reachable=lambda url, timeout: next(looks),
                            sleep=lambda s: None, command=lambda: ["proxy"])
    assert ran == [["proxy", "start", "--comfyui", "http://127.0.0.1:8188", "--port", "8189"]]
    assert said.startswith("started comfy-api-proxy")


def test_start_never_starts_a_bridge_on_another_machine(settings):
    ran = []
    said = cli.start_bridge(settings, run=lambda *a, **k: ran.append(a),   # bridge.invalid
                            reachable=lambda url, timeout: False, command=lambda: ["proxy"])
    assert ran == [] and "not on this machine" in said


def test_start_says_when_the_bridge_is_missing_or_fails(settings):
    local = _local(settings)
    down = lambda url, timeout: False  # noqa: E731
    assert "pip install" in cli.start_bridge(local, reachable=down, command=lambda: None)
    said = cli.start_bridge(local, reachable=down, command=lambda: ["proxy"],
                            run=lambda cmd, **k: _Result(1, err="boom\nport 8189 in use"))
    assert said == "comfy-api-proxy did not start: port 8189 in use"
    said = cli.start_bridge(local, reachable=down, command=lambda: ["proxy"],
                            run=lambda cmd, **k: _Result(), sleep=lambda s: None)
    assert "not answering" in said


def test_start_serves_and_opens_the_browser_even_when_a_check_fails(monkeypatch, settings):
    calls = {}
    monkeypatch.setattr(cli, "_settings", lambda args: settings)
    monkeypatch.setattr(cli, "start_bridge", lambda s: "bridge said")
    monkeypatch.setattr(cli, "doctor_report", lambda s: [(False, "ComfyUI answers", "down")])
    monkeypatch.setattr(cli, "cmd_serve",
                        lambda args, s, open_browser: calls.update(s=s, open=open_browser) or 0)
    assert cli.main(["start"]) == 0
    assert calls == {"s": settings, "open": True}


def test_the_launchers_run_the_start_command():
    from pathlib import Path
    root = Path(cli.__file__).resolve().parents[1]
    bat = (root / "Start Radio Play Studio.bat").read_text(encoding="utf-8")
    sh = (root / "start-radio-play-studio.sh").read_text(encoding="utf-8")
    assert "-m radio_play_studio start" in bat and "-m radio_play_studio start" in sh
    # The repo stores it with LF endings, where cmd can miss a goto label: it has none.
    import re
    assert not re.search(r"^\s*:\w", bat, re.MULTILINE)
