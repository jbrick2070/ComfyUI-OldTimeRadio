"""The command line: the safety rails and the doctor."""
from __future__ import annotations

import pytest

from radio_play_studio import cli


def test_serve_refuses_to_listen_beyond_this_machine(capsys):
    assert cli.main(["serve", "--host", "0.0.0.0"]) == 2
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
