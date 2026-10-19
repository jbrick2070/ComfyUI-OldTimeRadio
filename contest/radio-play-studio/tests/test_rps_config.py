"""Where the app finds things, and the addresses it refuses to guess."""
from __future__ import annotations

import pytest

from radio_play_studio import config
from radio_play_studio.config import WorkflowNotFound, find_workflow, load_settings, validate_url


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("OTR_APP_WORKFLOW", "OTR_PACK_DIR", "COMFYUI_DIR", "COMFYUI_URL", "COMFY_BASE_URL", "RPS_OUT_DIR"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("bad", ["", " ", "/", "cloud.comfy.org", "ftp://host", "http://"])
def test_validate_url_refuses_what_the_sdk_would_read_as_the_cloud(bad):
    with pytest.raises(ValueError):
        validate_url("bridge", bad)


def test_validate_url_cleans_a_good_address():
    assert validate_url("bridge", " http://127.0.0.1:8189/ ") == "http://127.0.0.1:8189"


def test_blank_environment_values_mean_not_given_but_a_blank_flag_value_does_not_pass_as_a_host(monkeypatch, tmp_path):
    wf = tmp_path / "otr_app.json"
    wf.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("COMFY_BASE_URL", " ")
    assert load_settings(workflow=str(wf)).api_url == config.DEFAULT_API_URL
    with pytest.raises(ValueError, match="bridge|COMFY_BASE_URL"):
        load_settings(api_url="/", workflow=str(wf))


def test_a_workflow_named_by_the_person_must_exist(monkeypatch, tmp_path):
    with pytest.raises(WorkflowNotFound, match="--workflow"):
        find_workflow(str(tmp_path / "nope.json"))
    for var, value in (("OTR_APP_WORKFLOW", tmp_path / "nope.json"), ("OTR_PACK_DIR", tmp_path),
                       ("COMFYUI_DIR", tmp_path)):
        monkeypatch.setenv(var, str(value))
        with pytest.raises(WorkflowNotFound, match=var):
            find_workflow()
        monkeypatch.delenv(var)


def test_a_configured_pack_dir_wins_over_the_checkout(monkeypatch, tmp_path):
    (tmp_path / "workflows").mkdir()
    target = tmp_path / "workflows" / "otr_app.json"
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OTR_PACK_DIR", str(tmp_path))
    assert find_workflow() == target
