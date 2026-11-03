"""Where the app finds things, and the addresses it refuses to guess."""
from __future__ import annotations

import pytest

from radio_play_studio import config
from radio_play_studio.config import (DEFAULT_COMFYUI_URL, DESKTOP_COMFYUI_URL, WorkflowNotFound,
                                      default_comfyui_url, find_workflow, load_settings, validate_url)


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


@pytest.mark.parametrize("folder", ["ComfyUI-OldTimeRadio", "comfyui-old-time-radio"])
def test_comfyui_dir_finds_a_git_clone_and_a_registry_install(monkeypatch, tmp_path, folder):
    """A registry install lands in custom_nodes/<node id>, a git clone under the repository's name."""
    target = tmp_path / "custom_nodes" / folder / "workflows" / config.WORKFLOW_NAME
    target.parent.mkdir(parents=True)
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("COMFYUI_DIR", str(tmp_path))
    assert find_workflow() == target


def test_a_missing_pack_under_comfyui_dir_names_both_folders_it_tried(monkeypatch, tmp_path):
    monkeypatch.setenv("COMFYUI_DIR", str(tmp_path))
    with pytest.raises(WorkflowNotFound) as err:
        find_workflow()
    assert all(name in str(err.value) for name in config.PACK_DIR_NAMES)


def test_a_configured_pack_dir_wins_over_the_checkout(monkeypatch, tmp_path):
    (tmp_path / "workflows").mkdir()
    target = tmp_path / "workflows" / config.WORKFLOW_NAME
    target.write_text("{}", encoding="utf-8")
    monkeypatch.setenv("OTR_PACK_DIR", str(tmp_path))
    assert find_workflow() == target


def test_with_no_comfyui_named_the_app_finds_desktop_on_8000():
    """Desktop serves on 8000, a portable install on 8188: the one that answers
    is used, 8188 first, and 8188 when neither does."""
    assert default_comfyui_url(lambda url: url == DESKTOP_COMFYUI_URL) == DESKTOP_COMFYUI_URL
    assert default_comfyui_url(lambda url: True) == DEFAULT_COMFYUI_URL
    assert default_comfyui_url(lambda url: False) == DEFAULT_COMFYUI_URL


def test_a_named_comfyui_is_never_second_guessed(monkeypatch, tmp_path):
    from radio_play_studio import config
    monkeypatch.setattr(config, "default_comfyui_url", lambda: pytest.fail("probed a named ComfyUI"))
    wf = tmp_path / "w.json"
    wf.write_text("{}", encoding="utf-8")
    assert config.load_settings(comfyui_url="http://127.0.0.1:9999", workflow=str(wf)).comfyui_url \
        == "http://127.0.0.1:9999"


def test_only_a_real_comfyui_counts_as_answering(monkeypatch):
    """Something else on port 8000 is not taken for ComfyUI."""
    import io
    from radio_play_studio import config

    class Reply(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    monkeypatch.setattr(config.urllib.request, "urlopen", lambda url, timeout: Reply(b'{"system": {"os": "win32"}}'))
    assert config.comfyui_answers("http://127.0.0.1:8000")
    monkeypatch.setattr(config.urllib.request, "urlopen", lambda url, timeout: Reply(b"<html>not comfy</html>"))
    assert not config.comfyui_answers("http://127.0.0.1:8000")
    monkeypatch.setattr(config.urllib.request, "urlopen", lambda url, timeout: Reply(b'["a list"]'))
    assert not config.comfyui_answers("http://127.0.0.1:8000")


def test_a_port_that_does_not_speak_http_is_not_a_comfyui(monkeypatch):
    """Something on 8000 that answers with bytes that are not HTTP raises
    http.client.HTTPException, which is not an OSError: it must read as "no"."""
    import http.client
    from radio_play_studio import config

    def bad_status(url, timeout):
        raise http.client.BadStatusLine("not http")
    monkeypatch.setattr(config.urllib.request, "urlopen", bad_status)
    assert config.comfyui_answers("http://127.0.0.1:8000") is False


def test_a_found_comfyui_is_looked_for_again_when_it_stops_answering(settings, schemas, monkeypatch):
    """The app settled on 8188 before Desktop came up on 8000: the next look at
    the node definitions finds Desktop. A named ComfyUI is never replaced."""
    import dataclasses
    from radio_play_studio import runner
    from radio_play_studio.graph import ServerUnreachable
    from radio_play_studio.runner import Studio

    def fetch(url):
        if url != DESKTOP_COMFYUI_URL:
            raise ServerUnreachable(f"could not read {url}")
        return schemas
    monkeypatch.setattr(runner, "default_comfyui_url", lambda answers=None: DESKTOP_COMFYUI_URL)
    found = Studio(dataclasses.replace(settings, comfyui_url=DEFAULT_COMFYUI_URL, comfyui_found=True), fetch=fetch)
    assert found._schemas() is schemas and found.settings.comfyui_url == DESKTOP_COMFYUI_URL
    named = Studio(dataclasses.replace(settings, comfyui_url=DEFAULT_COMFYUI_URL), fetch=fetch)
    with pytest.raises(ServerUnreachable):
        named._schemas()
    assert named.settings.comfyui_url == DEFAULT_COMFYUI_URL
