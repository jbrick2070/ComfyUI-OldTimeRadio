"""Runtime artifacts must not land inside the installed pack directory."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_paths as P
from nodes._otr_shared import cloud_media_backend as cmb


@pytest.fixture()
def pinned_out(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.delenv("OTR_CLOUD_MEDIA_CACHE_DIR", raising=False)
    monkeypatch.delenv("OTR_RUNTIME_LOG_PATH", raising=False)
    monkeypatch.delenv("OTR_SIDECAR_STDERR_DIR", raising=False)
    return tmp_path


def _pack_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_cloud_media_cache_default_is_under_output_not_pack(pinned_out):
    root = cmb.resolve_cache_root()
    pack = _pack_root()
    assert pack not in root.parents, "cache root still inside pack: %s" % root
    assert root == P.otr_shared_cache_dir() / "cloud_media"


def test_pack_billing_ledger_is_copied_forward_from_legacy_pack_cache(pinned_out, tmp_path):
    pack_cache = cmb._legacy_pack_cloud_media_cache_root()
    pack_cache.mkdir(parents=True, exist_ok=True)
    legacy = pack_cache / "billing_ledger.jsonl"
    legacy.write_text(json.dumps({"ts": 1, "usd": 1.23}) + "\n", encoding="utf-8")

    session = cmb.CloudMediaSession("prompt-test", cmb.CloudAuth("api_key_hidden", "k"))
    path = session.ledger_path()

    assert path.is_file()
    assert json.loads(path.read_text(encoding="utf-8").strip())["usd"] == 1.23
    assert legacy.is_file(), "legacy ledger must not be moved"
    assert _pack_root() not in path.parents


def test_runtime_log_write_lands_in_state_dir(pinned_out, monkeypatch):
    from nodes.story_orchestrator import _runtime_log

    _runtime_log("pack-runtime-write probe")
    log_path = P.otr_runtime_log_path()
    assert log_path.is_file()
    assert _pack_root() not in log_path.parents
    assert "pack-runtime-write probe" in log_path.read_text(encoding="utf-8")


def test_vram_log_write_lands_in_state_dir(pinned_out):
    from nodes import _vram_log as vl

    vl.vram_snapshot("pack_runtime_test_phase")
    log_path = P.otr_runtime_log_path()
    assert log_path.is_file()
    assert _pack_root() not in log_path.parents
    assert "VRAM_SNAPSHOT" in log_path.read_text(encoding="utf-8")


def test_chatterbox_stderr_lands_in_state_dir(pinned_out, monkeypatch, tmp_path):
    from nodes._otr_audio_engines import eng_chatterbox as mod

    eng = mod.ChatterboxEngine()
    (tmp_path / "py.exe").write_text("", encoding="utf-8")
    (tmp_path / "worker.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(eng, "_venv_python", lambda: str(tmp_path / "py.exe"))
    monkeypatch.setattr(eng, "_worker_script", lambda: str(tmp_path / "worker.py"))

    reached = {"popen": False}

    def _fake_popen(*_a, **_k):
        reached["popen"] = True
        raise RuntimeError("stop after stderr setup")

    monkeypatch.setattr(mod.otr_proc, "popen", _fake_popen)

    with pytest.raises(RuntimeError, match="stop after stderr"):
        eng.load()

    err_path = P.otr_sidecar_stderr_path("_otr_chatterbox_worker.err")
    assert reached["popen"]
    assert _pack_root() not in err_path.parents


def test_chatterbox_unwritable_stderr_does_not_crash_before_worker(pinned_out, monkeypatch, tmp_path):
    from nodes._otr_audio_engines import eng_chatterbox as mod

    eng = mod.ChatterboxEngine()
    (tmp_path / "py.exe").write_text("", encoding="utf-8")
    (tmp_path / "worker.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(eng, "_venv_python", lambda: str(tmp_path / "py.exe"))
    monkeypatch.setattr(eng, "_worker_script", lambda: str(tmp_path / "worker.py"))

    real_open = open

    def guarded_open(path, *args, **kwargs):
        if "_otr_chatterbox_worker.err" in str(path):
            raise OSError(13, "Permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(
        mod.otr_proc,
        "popen",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("popen reached")),
    )

    with pytest.raises(RuntimeError, match="popen reached"):
        eng.load()


def test_dia_unwritable_stderr_does_not_crash_before_worker(pinned_out, monkeypatch, tmp_path):
    from nodes._otr_audio_engines import eng_dia as mod

    eng = mod.DiaEngine()
    (tmp_path / "py.exe").write_text("", encoding="utf-8")
    (tmp_path / "worker.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(eng, "_venv_python", lambda: str(tmp_path / "py.exe"))
    monkeypatch.setattr(eng, "_worker_script", lambda: str(tmp_path / "worker.py"))

    real_open = open

    def guarded_open(path, *args, **kwargs):
        if "_otr_dia_worker.err" in str(path):
            raise OSError(13, "Permission denied")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", guarded_open)
    monkeypatch.setattr(
        mod.otr_proc,
        "popen",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("popen reached")),
    )

    with pytest.raises(RuntimeError, match="popen reached"):
        eng.load()


def test_the_episode_telemetry_never_reads_the_frozen_legacy_log(
        pinned_out, monkeypatch, tmp_path):
    """A field the live log has not written yet keeps its default; it never
    falls through to a stale run in the legacy pack-root log."""
    from nodes import video_engine

    live = tmp_path / "state" / "otr_runtime.log"
    live.parent.mkdir(parents=True)
    live.write_text("[heartbeat] still rendering\n"
                    "VRAM_SNAPSHOT stage=x peak_gb=9.87\n", encoding="utf-8")
    monkeypatch.setenv("OTR_RUNTIME_LOG_PATH", str(live))
    legacy = tmp_path / "legacy" / "otr_runtime.log"
    legacy.parent.mkdir(parents=True)
    legacy.write_text("VRAM_SNAPSHOT stage=x peak_gb=3.33\n"
                      "LLM loaded: org/OLD-STALE-MODEL\n"
                      "DONE: 900 tokens 12.5 tok/s\n", encoding="utf-8")
    monkeypatch.setattr(P, "legacy_pack_runtime_log_path", lambda: legacy)

    assert video_engine._get_latest_telemetry() == ("9.87", "???", "UNKNOWN CORE")
