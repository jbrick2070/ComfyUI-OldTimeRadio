from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILE_HELPER = ROOT / "scripts" / "otr_profile_launch_args.py"
RUNTIME = ROOT / "scripts" / "otr_pod_runtime.sh"
SWEEP = ROOT / "scripts" / "otr_pod_overnight_sweep.sh"
SOAK = ROOT / "scripts" / "otr_pod_lane_soak.sh"
PROVISION = ROOT / "scripts" / "otr_provision.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bash():
    found = shutil.which("bash")
    if found:
        return found
    git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
    return str(git_bash) if git_bash.is_file() else None


def test_all_pod_shell_owners_have_valid_bash_syntax():
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    subprocess.run(
        [bash, "-n", str(RUNTIME), str(SWEEP), str(SOAK),
         str(ROOT / "scripts" / "otr_pod_provision.sh")],
        check=True,
    )


@pytest.mark.parametrize(
    ("profile_id", "expected"),
    [
        ("otr_w45_ltx25_video", []),
        ("otr_w45_humo", ["--reserve-vram", "2.921", "--disable-pinned-memory"]),
        ("otr_w45_minimax_h3_video", ["--reserve-vram", "12", "--disable-pinned-memory"]),
        ("otr_w45_ltx_audio_in", ["--disable-pinned-memory"]),
    ],
)
def test_profile_helper_resolves_canonical_boot_argv(profile_id, expected):
    helper = _load(PROFILE_HELPER, "otr_profile_launch_args_test")
    resolved = helper.resolve_launch(helper.load_profile(profile_id))
    assert resolved["argv"] == expected


def test_profile_helper_rejects_contract_environment_drift(tmp_path):
    helper = _load(PROFILE_HELPER, "otr_profile_launch_args_drift_test")
    helper.PROFILES = tmp_path
    (tmp_path / "bad.json").write_text(
        json.dumps({
            "id": "bad",
            "launch": {
                "boot_contract": "humo_diet",
                "sage_attention": True,
                "env": {},
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(helper.LaunchConfigError, match="launch.env disagrees"):
        helper.resolve_launch(helper.load_profile("bad"))


def test_launch_fingerprint_groups_profiles_with_identical_process_state():
    helper = _load(PROFILE_HELPER, "otr_profile_launch_args_fingerprint_test")
    video = helper.resolve_launch(helper.load_profile("otr_w45_ltx25_video"))
    mime = helper.resolve_launch(helper.load_profile("otr_w45_ltx25_mime"))
    humo = helper.resolve_launch(helper.load_profile("otr_w45_humo"))
    assert video["fingerprint"] == mime["fingerprint"]
    assert video["fingerprint"] != humo["fingerprint"]


def _parse_listener_fixture(port: int, fixture: str) -> list[str]:
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    result = subprocess.run(
        [bash, "-c",
         'source "$1"; otr_listener_pids_from_stream "$2"',
         "otr-runtime-test", RUNTIME.as_posix(), str(port)],
        input=fixture,
        text=True,
        capture_output=True,
        check=True,
    )
    return result.stdout.splitlines()


def test_listener_parser_handles_ss_ipv4_ipv6_without_partial_port_matches():
    fixture = "\n".join([
        'LISTEN 0 128 0.0.0.0:8188 0.0.0.0:* users:(("python",pid=123,fd=3))',
        'LISTEN 0 128 [::]:8188 [::]:* users:(("python",pid=456,fd=5))',
        'LISTEN 0 128 0.0.0.0:18188 0.0.0.0:* users:(("python",pid=999,fd=7))',
    ])
    assert _parse_listener_fixture(8188, fixture) == ["123", "456"]


def test_listener_parser_handles_netstat_pid_program_rows():
    fixture = "\n".join([
        "tcp 0 0 0.0.0.0:8188 0.0.0.0:* LISTEN 321/python",
        "tcp6 0 0 :::8188 :::* LISTEN 654/python3",
        "tcp 0 0 0.0.0.0:18188 0.0.0.0:* LISTEN 777/python",
    ])
    assert _parse_listener_fixture(8188, fixture) == ["321", "654"]


def test_runtime_owner_uses_checked_files_and_excludes_cloud_h3():
    text = RUNTIME.read_text(encoding="utf-8")
    assert "otr_profile_output" in text
    assert 'if ! "$COMFY_PY" "$helper"' in text
    assert "mapfile -t launch_args < \"$args_file\"" in text
    assert "mapfile -t launch_args < <(" not in text
    assert '== "h3"' in text
    assert "operator-local and cannot run in the cloud roster" in text
    assert "--listen 0.0.0.0" in text
    assert "OTR_SERVER_FINGERPRINT" in text


def test_writer_warm_deduplicates_and_uses_the_catalog_download_contract(
        tmp_path, monkeypatch):
    provision = _load(PROVISION, "otr_provision_writer_warm_test")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return tmp_path / "snapshot"

    profile = {
        "llm": {
            "creative_model": "mistralai/Mistral-Nemo-Instruct-2407",
            "technical_model": "mistralai/Mistral-Nemo-Instruct-2407",
        }
    }
    provision._LOG.clear()
    provision.warm_profile_writer_models(
        profile, _snapshot_download=snapshot_download)
    catalog = provision._load_writer_catalog()

    assert len(calls) == 1
    assert calls[0] == {
        "repo_id": "mistralai/Mistral-Nemo-Instruct-2407",
        "allow_patterns": list(catalog.ALLOW_PATTERNS),
        "cache_dir": str(tmp_path / "hf" / "hub"),
        "token": "fixture-token",
    }
    assert any(row[:2] == ("OK", "writer: mistralai/Mistral-Nemo-Instruct-2407")
               for row in provision._LOG)


def test_writer_warm_skips_remote_and_gguf_rows_without_download(
        tmp_path, monkeypatch):
    provision = _load(PROVISION, "otr_provision_writer_skip_test")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    catalog = SimpleNamespace(
        CURATED_LLM_MODELS=(
            SimpleNamespace(repo_id="remote:slot", provider="openrouter",
                            loader_backend="openrouter_http"),
            SimpleNamespace(repo_id="local:model.gguf", provider="gguf_native",
                            loader_backend="gguf_native"),
        ),
        ALLOW_PATTERNS=("*.json",),
    )
    monkeypatch.setattr(provision, "_load_writer_catalog", lambda: catalog)
    calls = []
    profile = {"llm": {
        "creative_model": "remote:slot",
        "technical_model": "local:model.gguf",
    }}
    provision._LOG.clear()
    provision.warm_profile_writer_models(
        profile, _snapshot_download=lambda **kwargs: calls.append(kwargs))

    assert calls == []
    assert [row[0] for row in provision._LOG] == ["SKIP", "SKIP"]
