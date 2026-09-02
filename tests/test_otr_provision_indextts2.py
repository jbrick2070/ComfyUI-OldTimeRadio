"""IndexTTS2 provisioning is exact, isolated, and profile-aware."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "otr_provision.py"


def _load_provision():
    spec = importlib.util.spec_from_file_location(
        "_otr_provision_indextts2_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args], check=True,
        capture_output=True, text=True)
    return result.stdout.strip()


def _make_source(path: Path) -> str:
    (path / "checkpoints").mkdir(parents=True)
    (path / "checkpoints" / "config.yaml").write_text(
        "fixture: true\n", encoding="utf-8")
    (path / "pyproject.toml").write_text(
        "[project]\nname='index-tts-fixture'\nversion='0'\n",
        encoding="utf-8")
    (path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "tests@example.invalid")
    _git(path, "config", "user.name", "OTR tests")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "fixture")
    return _git(path, "rev-parse", "HEAD")


def _write_wav(path: Path, *, sample: int) -> bytes:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22_050)
        handle.writeframes(
            int(sample).to_bytes(2, "little", signed=True) * 22_050)
    return path.read_bytes()


def _bank_row(voice_id: str, filename: str, payload: bytes, gender: str) -> dict:
    return {
        "voice_ref_id": voice_id,
        "engine": "indextts2",
        "gender": gender,
        "timbre": [],
        "roles": ["char_voice"],
        "age_band": "adult",
        "ref_path": "models/TTS/refs/indextts2/%s" % filename,
        "ref_sha256": hashlib.sha256(payload).hexdigest(),
        "commercial_clean": False,
    }


def test_source_gate_requires_exact_pin_and_allows_only_runtime_drift(
        tmp_path, monkeypatch):
    provision = _load_provision()
    source = tmp_path / "index-tts"
    pin = _make_source(source)
    monkeypatch.setattr(provision, "INDEXTTS2_PIN", pin)

    assert provision._indextts2_source_problems(str(source)) == []

    (source / "checkpoints" / "config.yaml").write_text(
        "runtime: changed\n", encoding="utf-8")
    (source / ".venv").mkdir()
    (source / ".venv" / "marker").write_text("ok\n", encoding="utf-8")
    (source / ".uv-python").mkdir()
    (source / ".uv-python" / "managed-python").write_text(
        "persistent\n", encoding="utf-8")
    assert provision._indextts2_source_problems(str(source)) == []

    (source / "pyproject.toml").write_text("drift\n", encoding="utf-8")
    problems = provision._indextts2_source_problems(str(source))
    assert len(problems) == 1
    assert "pyproject.toml" in problems[0]

    monkeypatch.setattr(provision, "INDEXTTS2_PIN", "0" * 40)
    assert any("HEAD" in item for item in
               provision._indextts2_source_problems(str(source)))


def test_reference_gate_uses_registered_rows_not_arbitrary_wavs(tmp_path):
    provision = _load_provision()
    root = tmp_path / "models"
    refs = root / "TTS" / "refs" / "indextts2"
    refs.mkdir(parents=True)
    first = _write_wav(refs / "first.wav", sample=100)
    second = _write_wav(refs / "second.wav", sample=-100)
    (refs / "orphan.wav").write_bytes(b"not in the bank")
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"voices": [
        _bank_row("first", "first.wav", first, "male"),
        _bank_row("second", "second.wav", second, "female"),
        _bank_row(
            "idx_lemmy_algenib_cockney_v1", "private-reserved.wav",
            b"not distributed", "male"),
        {
            "voice_ref_id": "announcer-only", "engine": "kokoro",
            "gender": "male", "timbre": [], "roles": ["announcer_voice"],
            "age_band": "adult", "ref_path": "cloud:k",
            "ref_sha256": "cloud", "commercial_clean": True,
        },
    ]}), encoding="utf-8")

    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == ["first", "second"]
    assert problems == []
    assert registered == 2

    _write_wav(refs / "second.wav", sample=200)
    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == ["first"]
    assert registered == 2
    assert any("second SHA-256" in item for item in problems)
    assert any("lacks female coverage" in item for item in problems)


def test_reference_gate_rejects_invalid_wav_schema_and_duplicate_ids(tmp_path):
    provision = _load_provision()
    root = tmp_path / "models"
    refs = root / "TTS" / "refs" / "indextts2"
    refs.mkdir(parents=True)
    male = _write_wav(refs / "male.wav", sample=1)
    bad = b"this is not wave audio"
    (refs / "female.wav").write_bytes(bad)
    bank = tmp_path / "bank.json"
    male_row = _bank_row("male", "male.wav", male, "male")
    female_row = _bank_row("female", "female.wav", bad, "female")
    bank.write_text(json.dumps({"voices": [male_row, female_row]}),
                    encoding="utf-8")

    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == ["male"] and registered == 2
    assert any("not a readable WAV" in item for item in problems)

    malformed = dict(male_row)
    malformed.pop("age_band")
    bank.write_text(json.dumps({"voices": [malformed, female_row]}),
                    encoding="utf-8")
    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == [] and registered == 0
    assert "voice bank invalid" in problems[0]

    duplicate = dict(female_row, voice_ref_id="male")
    bank.write_text(json.dumps({"voices": [male_row, duplicate]}),
                    encoding="utf-8")
    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == [] and registered == 0
    assert "duplicate voice_ref_id" in problems[0]


def test_reference_gate_rejects_truncated_pcm_payload(tmp_path):
    provision = _load_provision()
    root = tmp_path / "models"
    refs = root / "TTS" / "refs" / "indextts2"
    refs.mkdir(parents=True)
    male_path = refs / "male.wav"
    _write_wav(male_path, sample=5)
    male_path.write_bytes(male_path.read_bytes()[:-4])
    male = male_path.read_bytes()
    female = _write_wav(refs / "female.wav", sample=-5)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"voices": [
        _bank_row("male", "male.wav", male, "male"),
        _bank_row("female", "female.wav", female, "female"),
    ]}), encoding="utf-8")

    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == ["female"] and registered == 2
    assert any("truncated PCM" in item for item in problems)


def test_reference_gate_rejects_gender_spelling_runtime_cannot_match(tmp_path):
    provision = _load_provision()
    root = tmp_path / "models"
    refs = root / "TTS" / "refs" / "indextts2"
    refs.mkdir(parents=True)
    male = _write_wav(refs / "male.wav", sample=7)
    female = _write_wav(refs / "female.wav", sample=-7)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"voices": [
        _bank_row("male", "male.wav", male, "Male"),
        _bank_row("female", "female.wav", female, "Female"),
    ]}), encoding="utf-8")

    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == [] and registered == 2
    assert any("unsupported gender 'Male'" in item for item in problems)
    assert any("unsupported gender 'Female'" in item for item in problems)
    assert any("lacks female, male coverage" in item for item in problems)


def test_reference_gate_rejects_cross_gender_duplicate_recording(tmp_path):
    provision = _load_provision()
    root = tmp_path / "models"
    refs = root / "TTS" / "refs" / "indextts2"
    refs.mkdir(parents=True)
    payload = _write_wav(refs / "male.wav", sample=17)
    (refs / "female.wav").write_bytes(payload)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"voices": [
        _bank_row("male", "male.wav", payload, "male"),
        _bank_row("female", "female.wav", payload, "female"),
    ]}), encoding="utf-8")

    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == [] and registered == 2
    assert any("registered across genders" in item for item in problems)
    assert any("male/female references must be distinct" in item
               for item in problems)
    assert any("lacks female, male coverage" in item for item in problems)


def test_reference_gate_rejects_vendor_incompatible_short_wav(tmp_path):
    provision = _load_provision()
    root = tmp_path / "models"
    refs = root / "TTS" / "refs" / "indextts2"
    refs.mkdir(parents=True)
    short_path = refs / "male.wav"
    with wave.open(str(short_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22_050)
        handle.writeframes(b"\x01\x00" * 64)
    short = short_path.read_bytes()
    female = _write_wav(refs / "female.wav", sample=-17)
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps({"voices": [
        _bank_row("male", "male.wav", short, "male"),
        _bank_row("female", "female.wav", female, "female"),
    ]}), encoding="utf-8")

    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root), str(bank))
    assert usable == ["female"] and registered == 2
    assert any("too short for IndexTTS2" in item for item in problems)
    assert any("lacks male coverage" in item for item in problems)


def test_portable_bank_override_is_shared_by_runtime_and_provisioner(
        tmp_path, monkeypatch):
    provision = _load_provision()
    from nodes import _otr_voice_bank as runtime_bank

    root = tmp_path / "models"
    refs = root / "TTS" / "refs" / "indextts2"
    refs.mkdir(parents=True)
    male = _write_wav(refs / "portable-male.wav", sample=11)
    female = _write_wav(refs / "portable-female.wav", sample=-11)
    shipped_path = ROOT / "config" / "voice_reference_bank.json"
    shipped = json.loads(shipped_path.read_text(encoding="utf-8"))
    non_index = [row for row in shipped["voices"]
                 if row.get("engine") != "indextts2"]
    bank = tmp_path / "portable-bank.json"
    portable_rows = [
        _bank_row("portable-male", "portable-male.wav", male, "male"),
        _bank_row("portable-female", "portable-female.wav", female, "female"),
    ]
    portable_doc = {
        "voice_bank_id": "portable-index-bank",
        "schema_version": shipped.get("schema_version", "1"),
        "voices": non_index + portable_rows,
    }
    bank.write_text(json.dumps(portable_doc), encoding="utf-8")
    monkeypatch.setenv("OTR_VOICE_REFERENCE_BANK", str(bank))
    runtime_bank._BANK_CACHE.clear()

    entries, _sha = runtime_bank.load_voice_bank()
    assert {entry.voice_ref_id for entry in entries
            if entry.engine != "indextts2"} == {
        row["voice_ref_id"] for row in non_index
    }
    assert [entry.voice_ref_id for entry in entries
            if entry.engine == "indextts2"] == [
        "portable-male", "portable-female"]
    usable, problems, registered = provision.verify_registered_indextts2_refs(
        str(root))
    assert usable == ["portable-male", "portable-female"]
    assert problems == [] and registered == 2

    male_pick = runtime_bank.assign_voice_for_slot(
        role="char_voice", engine="indextts2", char_id="portable-male-char",
        gender="male", bank=entries)
    female_pick = runtime_bank.assign_voice_for_slot(
        role="char_voice", engine="indextts2", char_id="portable-female-char",
        gender="female", bank=entries)
    announcer = runtime_bank.assign_voice_for_slot(
        role="announcer_voice", engine="kokoro", char_id="ANNOUNCER",
        gender="male", bank=entries)
    assert male_pick.voice_ref_id == "portable-male"
    assert female_pick.voice_ref_id == "portable-female"
    assert announcer.engine == "kokoro"


def test_runtime_cache_requires_nonempty_snapshot_and_pinned_main_ref(tmp_path):
    provision = _load_provision()
    cache = tmp_path / "hf_cache"
    repo = "Example/runtime"
    revision = "a" * 40
    snapshot = Path(provision._runtime_cache_revision_path(
        str(cache), repo, revision))
    snapshot.mkdir(parents=True)

    expected = {"artifact.bin": 6}
    problems = provision._runtime_cache_problems(
        str(cache), repo, revision, expected)
    assert "missing artifact.bin" in problems
    assert "refs/main is not the exact 40-byte pinned commit" in problems

    (snapshot / "artifact.bin").write_bytes(b"pinned")
    ref = cache / "models--Example--runtime" / "refs" / "main"
    ref.parent.mkdir()
    ref.write_text(revision + "\n", encoding="ascii")
    assert provision._runtime_cache_problems(
        str(cache), repo, revision, expected) == [
            "refs/main is not the exact 40-byte pinned commit"]
    ref.write_bytes(revision.encode("ascii"))
    assert provision._runtime_cache_problems(
        str(cache), repo, revision, expected) == []


def test_real_worker_probe_requires_ready_protocol_line(tmp_path, monkeypatch):
    provision = _load_provision()
    source = tmp_path / "index-tts"
    for name in ("OTR_INDEXTTS2_VENV", "OTR_INDEXTTS2_DIR",
                 "OTR_INDEXTTS2_WORKER"):
        monkeypatch.delenv(name, raising=False)
    responses = iter([
        subprocess.CompletedProcess([], 0, stdout='{"ready": true}\n', stderr=""),
        subprocess.CompletedProcess([], 0, stdout='{"ready": false}\n', stderr=""),
        subprocess.CompletedProcess([], 3, stdout="", stderr="broken"),
    ])
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return next(responses)

    monkeypatch.setattr(provision, "run", fake_run)
    assert provision._probe_indextts2_worker(str(source))[0] is True
    assert provision._probe_indextts2_worker(str(source))[0] is False
    assert provision._probe_indextts2_worker(str(source))[0] is False
    assert calls[0][1]["input"] == '{"stop": true}\n'
    assert calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert calls[0][1]["timeout"] == 600
    assert calls[0][1]["cwd"] == str(source)
    assert "--fp16" not in calls[0][0]


def test_real_worker_probe_matches_runtime_fp16_switch(tmp_path, monkeypatch):
    provision = _load_provision()
    source = tmp_path / "index-tts"
    calls = []
    monkeypatch.setattr(
        provision, "run",
        lambda command, **kwargs: (
            calls.append((list(command), kwargs))
            or subprocess.CompletedProcess(
                command, 0, stdout='{"ready": true}\n', stderr="")))

    monkeypatch.delenv("OTR_INDEXTTS2_FP16", raising=False)
    assert provision._probe_indextts2_worker(str(source))[0] is True
    monkeypatch.setenv("OTR_INDEXTTS2_FP16", "1")
    assert provision._probe_indextts2_worker(str(source))[0] is True
    monkeypatch.setenv("OTR_INDEXTTS2_FP16", "true")
    assert provision._probe_indextts2_worker(str(source))[0] is True

    assert "--fp16" not in calls[0][0]
    assert calls[1][0][-1] == "--fp16"
    assert "--fp16" not in calls[2][0]


def test_windows_installer_honors_locked_and_runtime_overrides():
    script = (ROOT / "scripts" / "_otr_indextts2_install.ps1").read_text(
        encoding="utf-8")
    assert "uv sync --frozen --python 3.10" in script
    assert "$env:OTR_INDEXTTS2_VENV" in script
    assert "$env:OTR_INDEXTTS2_DIR" in script
    assert "$env:OTR_INDEXTTS2_WORKER" in script
    assert '$env:OTR_INDEXTTS2_FP16 -eq "1"' in script
    assert '$WorkerArgs += "--fp16"' in script
    assert "Invoke-NativeChecked" in script
    assert '"rev-parse", "HEAD"' in script
    assert '"status", "--porcelain=v1"' in script
    assert "$env:UV_PYTHON_INSTALL_DIR" in script
    assert '$env:UV_PYTHON_PREFERENCE = "only-managed"' in script
    assert '"python", "install", "3.10"' in script
    assert '"sync", "--frozen", "--python", "3.10"' in script
    assert '$env:HF_HUB_OFFLINE = "1"' in script
    assert '$env:TRANSFORMERS_OFFLINE = "1"' in script
    assert "Push-Location (Split-Path -Parent $Ckpt)" in script

    runpod = (ROOT / "docs" / "RUNPOD_INSTALL.md").read_text(encoding="utf-8")
    assert "UV_PYTHON_INSTALL_DIR" in runpod
    assert "uv python install 3.10        # IndexTTS2" in runpod


def test_posix_default_index_root_is_a_persistent_sibling_not_core_drift(
        tmp_path, monkeypatch):
    provision = _load_provision()
    monkeypatch.delenv("OTR_INDEXTTS2_ROOT", raising=False)
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    expected = os.path.abspath(os.path.join(str(comfy.parent), "index-tts"))
    monkeypatch.setattr(provision.os, "name", "posix")

    source = provision._indextts2_source_root(str(comfy))

    assert os.path.normcase(source) == os.path.normcase(expected)

    installer = (ROOT / "scripts" / "_otr_indextts2_install.ps1").read_text(
        encoding="utf-8")
    engine = (ROOT / "nodes" / "_otr_audio_engines" /
              "eng_indextts2.py").read_text(encoding="utf-8")
    downloader = (ROOT / "scripts" /
                  "_otr_idx_download_weights.py").read_text(encoding="utf-8")
    playbook = (ROOT / "docs" / "RUNPOD_INSTALL.md").read_text(
        encoding="utf-8")
    pod_owner = (ROOT / "scripts" / "otr_pod_provision.sh").read_text(
        encoding="utf-8")

    assert 'Join-Path $ComfyRoot "index-tts"' in installer
    assert 'os.path.join(_COMFY_ROOT, "index-tts", *parts)' in engine
    assert 'base = comfy_root if os.name == "nt"' in downloader
    assert '$(dirname "$COMFY_ROOT")/index-tts' in pod_owner
    assert "/workspace/otr-config/otr-runtime.env" in playbook
    assert "Do not export the Linux offline wrapper as `OTR_INDEXTTS2_VENV`" \
        in playbook
    assert '$OTR_COMFY_ROOT/index-tts' not in playbook


def test_index_root_local_exclusion_makes_second_core_check_clean(tmp_path):
    provision = _load_provision()
    comfy = tmp_path / "ComfyUI"
    source = comfy / "index-tts"
    source.mkdir(parents=True)
    (source / "owned.txt").write_text("managed runtime\n", encoding="utf-8")
    _git(comfy, "init", "-q")

    assert "index-tts/owned.txt" in _git(
        comfy, "status", "--porcelain", "--untracked-files=all")

    provision._exclude_and_link_indextts2_root(str(comfy), str(source))
    provision._exclude_and_link_indextts2_root(str(comfy), str(source))

    assert _git(comfy, "status", "--porcelain", "--untracked-files=all") == ""
    exclude = _git(comfy, "rev-parse", "--git-path", "info/exclude")
    exclude_path = Path(exclude if os.path.isabs(exclude) else comfy / exclude)
    assert exclude_path.read_text(encoding="utf-8").splitlines().count(
        "/index-tts") == 1


def test_linux_index_launcher_owns_offline_cache_and_vendor_cwd(
        tmp_path, monkeypatch):
    provision = _load_provision()
    root = tmp_path / "index-tts"
    real = root / ".venv" / "bin" / "python"
    real.parent.mkdir(parents=True)
    real.write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setattr(provision.os, "name", "posix")

    launcher = provision.link_indextts2_runtime_python(str(root))

    with open(launcher, "r", encoding="utf-8") as handle:
        text = handle.read()
    assert "export HF_HUB_OFFLINE=1" in text
    assert "export TRANSFORMERS_OFFLINE=1" in text
    assert 'cd "$engine_root"' in text
    assert 'exec "$engine_root/.venv/bin/python" "$@"' in text


@pytest.mark.skipif(os.name != "nt", reason="PowerShell 5.1 native-exit contract")
def test_windows_installer_stops_on_native_git_failure_despite_stale_venv(
        tmp_path):
    powershell = shutil.which("powershell.exe") or shutil.which("powershell")
    if not powershell:
        pytest.skip("PowerShell is unavailable")
    root = tmp_path / "index-tts"
    (root / ".git").mkdir(parents=True)
    stale = root / ".venv" / "Scripts" / "python.exe"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale executable fixture")
    shims = tmp_path / "shims"
    shims.mkdir()
    (shims / "git.cmd").write_text("@echo off\r\nexit /b 7\r\n", encoding="ascii")
    (shims / "uv.cmd").write_text("@echo off\r\nexit /b 0\r\n", encoding="ascii")
    env = dict(os.environ)
    env["PATH"] = str(shims) + os.pathsep + env.get("PATH", "")
    env["OTR_INDEXTTS2_ROOT"] = str(root)
    env["OTR_INDEXTTS2_VENV"] = str(stale)
    env["OTR_INDEXTTS2_DIR"] = str(root / "checkpoints")

    result = subprocess.run(
        [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
         str(ROOT / "scripts" / "_otr_indextts2_install.ps1")],
        capture_output=True, text=True, timeout=30, env=env)
    combined = result.stdout + result.stderr

    assert result.returncode != 0
    assert "IndexTTS2 fetch failed (exit 7)" in combined
    assert "Done. indextts2" not in combined


def test_provisioner_refuses_case_mismatched_vendor_checkpoint_dir(
        tmp_path, monkeypatch):
    provision = _load_provision()
    monkeypatch.setenv("OTR_INDEXTTS2_DIR", str(tmp_path / "Checkpoints"))

    with pytest.raises(provision.ProvisionFailure, match="literal lower-case"):
        provision._indextts2_model_dir(str(tmp_path / "source"))

def test_path_overrides_match_provisioner_and_runtime_adapter(
        tmp_path, monkeypatch):
    provision = _load_provision()
    from nodes._otr_audio_engines import eng_indextts2

    source = tmp_path / "volume" / "index-source"
    venv = tmp_path / "volume" / "python"
    checkpoints = tmp_path / "volume" / "checkpoints"
    worker = tmp_path / "volume" / "worker.py"
    monkeypatch.setenv("OTR_INDEXTTS2_ROOT", str(source))
    monkeypatch.setenv("OTR_INDEXTTS2_VENV", str(venv))
    monkeypatch.setenv("OTR_INDEXTTS2_DIR", str(checkpoints))
    monkeypatch.setenv("OTR_INDEXTTS2_WORKER", str(worker))

    assert provision._indextts2_source_root("C:/ComfyUI") == str(source)
    assert provision._indextts2_venv_python(str(source)) == str(venv)
    assert provision._indextts2_model_dir(str(source)) == str(checkpoints)
    engine = eng_indextts2.IndexTTS2Engine()
    assert engine._venv_python() == str(venv)
    assert engine._model_dir() == str(checkpoints)
    assert engine._worker_script() == str(worker)


def test_windows_isolated_voice_runs_required_installer_not_skip(monkeypatch):
    provision = _load_provision()
    calls = []
    monkeypatch.setattr(provision.os, "name", "nt")
    monkeypatch.setattr(
        provision.shutil, "which",
        lambda name: "C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
        if name == "powershell.exe" else None)
    monkeypatch.setattr(
        provision, "run",
        lambda command, **_kwargs: (
            calls.append(list(command))
            or subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")))

    provision.install_isolated_voice("C:/ComfyUI", "chatterbox", [])

    assert len(calls) == 1
    assert calls[0][1:5] == [
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"]
    assert calls[0][-1].endswith("_otr_chatterbox_install.ps1")
    assert any(row[0:2] == ("OK", "chatterbox") for row in provision._LOG)


def test_install_uses_frozen_python310_and_isolated_downloader(
        tmp_path, monkeypatch):
    provision = _load_provision()
    for name in (
        "OTR_INDEXTTS2_ROOT", "OTR_INDEXTTS2_VENV", "OTR_INDEXTTS2_DIR",
        "OTR_INDEXTTS2_WORKER", "UV_PYTHON_INSTALL_DIR",
        "UV_PYTHON_PREFERENCE",
    ):
        monkeypatch.delenv(name, raising=False)
    source = tmp_path / "ComfyUI" / "index-tts"
    source.mkdir(parents=True)
    venv_py = Path(provision._indextts2_venv_python(str(source)))
    calls = []

    monkeypatch.setattr(
        provision, "ensure_indextts2_source", lambda _comfy: str(source))
    monkeypatch.setattr(provision, "_ensure_uv", lambda: "uv-test")
    monkeypatch.setattr(provision, "link_indextts2_runtime_python", lambda _root: "")
    monkeypatch.setattr(provision, "verify_indextts2_install", lambda *_args: True)

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        if command[:2] == ["uv-test", "sync"]:
            venv_py.parent.mkdir(parents=True)
            venv_py.write_bytes(b"fixture")
        return subprocess.CompletedProcess(command, 0, stdout="2.8.0\n", stderr="")

    monkeypatch.setattr(provision, "run", fake_run)
    models = tmp_path / "models"
    provision.install_indextts2(str(tmp_path / "ComfyUI"), str(models))

    assert calls[0][0] == ["uv-test", "python", "install", "3.10"]
    assert calls[0][1]["cwd"] == str(source)
    assert calls[0][1]["env"]["UV_PYTHON_INSTALL_DIR"] == str(
        source / ".uv-python")
    assert calls[0][1]["env"]["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert calls[1][0] == [
        "uv-test", "sync", "--frozen", "--python", "3.10"]
    assert calls[1][1]["cwd"] == str(source)
    assert calls[1][1]["env"]["UV_PYTHON_INSTALL_DIR"] == str(
        source / ".uv-python")
    assert calls[1][1]["env"]["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert calls[2][0][0] == str(venv_py)
    assert calls[2][0][1:3] == ["-c", "import torch,huggingface_hub;print(torch.__version__)"]
    assert calls[3][0] == [
        str(venv_py), str(ROOT / "scripts" / "_otr_idx_download_weights.py")]
    assert calls[3][1]["cwd"] == str(ROOT)
    assert calls[3][1]["env"]["OTR_INDEXTTS2_DIR"] == str(source / "checkpoints")


def _wire_main_stubs(provision, monkeypatch, profile, *, verify_result=True):
    calls = {"install": 0, "verify": 0, "other": []}
    # main() intentionally exports the models root for the helpers it invokes.
    # Own that CLI-process state through monkeypatch so an in-process unit test
    # cannot redirect unrelated resolver tests after this fixture returns.
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", "C:/models")
    monkeypatch.setattr(provision, "load_profile", lambda _pid: profile)
    monkeypatch.setattr(
        provision, "profile_lanes",
        lambda _profile: {"automatic": [], "manual": []})
    monkeypatch.setattr(provision, "comfy_root", lambda: "C:/ComfyUI")
    monkeypatch.setattr(provision, "models_root", lambda _comfy: "C:/models")
    monkeypatch.setattr(provision, "ensure_hf_home", lambda _root: None)
    monkeypatch.setattr(provision, "install_node_packs", lambda _root: None)
    monkeypatch.setattr(provision, "install_requirements", lambda: None)
    monkeypatch.setattr(provision, "fetch_lane_weights", lambda _lanes: None)
    monkeypatch.setattr(provision, "warm_profile_writer_models", lambda _profile: None)

    def install(*_args):
        calls["install"] += 1

    def verify(*_args):
        calls["verify"] += 1
        return verify_result

    monkeypatch.setattr(provision, "install_indextts2", install)
    monkeypatch.setattr(provision, "verify_indextts2_install", verify)
    monkeypatch.setattr(
        provision, "install_isolated_voice",
        lambda _comfy, name, _args: calls["other"].append(name))
    monkeypatch.setattr(provision, "ISOLATED_VOICES", {"one": [], "two": []})
    return calls


def test_selected_profile_verifies_index_without_install_flag(monkeypatch):
    provision = _load_provision()
    profile = {
        "id": "needs-index",
        "slot_overrides": {"char_voice_engine": "indextts2"},
    }
    calls = _wire_main_stubs(
        provision, monkeypatch, profile, verify_result=False)

    assert provision.main(["--profile", "needs-index"]) == 1
    assert calls == {"install": 0, "verify": 1, "other": []}
    assert any(row[1] == "selected profile requires IndexTTS2"
               for row in provision._LOG)


def test_index_flags_install_once_and_all_voices_adds_only_other_engines(
        monkeypatch):
    provision = _load_provision()
    profile = {
        "id": "needs-index",
        "slot_overrides": {"char_voice_engine": "indextts2"},
    }
    calls = _wire_main_stubs(provision, monkeypatch, profile)

    assert provision.main([
        "--profile", "needs-index", "--with-indextts2", "--with-all-voices",
    ]) == 0
    assert calls == {"install": 1, "verify": 0, "other": ["one", "two"]}


def test_profile_without_index_skips_both_install_and_verification(monkeypatch):
    provision = _load_provision()
    profile = {
        "id": "kokoro-only",
        "slot_overrides": {"char_voice_engine": "kokoro"},
    }
    calls = _wire_main_stubs(provision, monkeypatch, profile)

    assert provision.main(["--profile", "kokoro-only"]) == 0
    assert calls == {"install": 0, "verify": 0, "other": []}
