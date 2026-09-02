from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
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
        ("cpu_floor", ["--cpu"]),
        ("otr_cloud_low", ["--cpu"]),
        ("otr_cloud_hq", ["--cpu"]),
        ("otr_cloud_lanes", ["--cpu"]),
        ("otr_w45_ltx25_video", []),
        ("otr_w45_humo", ["--reserve-vram", "2.921", "--disable-pinned-memory"]),
        ("otr_w45_minimax_h3_video", ["--reserve-vram", "12", "--disable-pinned-memory"]),
        ("otr_4060_h3_nano", ["--disable-pinned-memory"]),
        ("otr_nvidia_8gb_h3", ["--disable-pinned-memory"]),
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


def test_launch_fingerprint_includes_runtime_generation_paths_and_secret_hashes(
        monkeypatch):
    helper = _load(PROFILE_HELPER, "otr_profile_runtime_identity_test")
    profile = helper.load_profile("otr_w45_ltx25_video")
    for key in helper.SECRET_IDENTITY_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("OTR_PROVISION_GENERATION", "generation-a")
    monkeypatch.setenv("OTR_OUTPUT_ROOT", "/workspace/output-a")
    monkeypatch.setenv("OTR_COMFY_API_KEY", "fixture-secret-a")
    first = helper.resolve_launch(profile)

    monkeypatch.setenv("OTR_PROVISION_GENERATION", "generation-b")
    generation_changed = helper.resolve_launch(profile)
    monkeypatch.setenv("OTR_PROVISION_GENERATION", "generation-a")
    monkeypatch.setenv("OTR_OUTPUT_ROOT", "/workspace/output-b")
    path_changed = helper.resolve_launch(profile)
    monkeypatch.setenv("OTR_OUTPUT_ROOT", "/workspace/output-a")
    monkeypatch.setenv("OTR_COMFY_API_KEY", "fixture-secret-b")
    secret_changed = helper.resolve_launch(profile)

    assert len({first["fingerprint"], generation_changed["fingerprint"],
                path_changed["fingerprint"], secret_changed["fingerprint"]}) == 4
    assert "fixture-secret-a" not in json.dumps(first)
    identity = helper._runtime_identity()
    assert "fixture-secret-b" not in json.dumps(identity)
    assert identity["secret_hashes"]["OTR_COMFY_API_KEY"]


def test_profile_helper_accepts_machine_matrix_selectors():
    helper = _load(PROFILE_HELPER, "otr_machine_launch_args_test")
    profile = helper.load_profile("machine:8gb")
    resolved = helper.resolve_launch(profile)

    assert profile["id"] == "machine:8gb"
    assert profile["slot_overrides"]["video_render_engine"] == \
        "animatediff15_v3_haunted_video"
    assert resolved["argv"] == []
    assert resolved["boot_contract"] == "default"


def test_default_launch_args_cli_emits_zero_bytes_not_an_empty_argument():
    result = subprocess.run(
        [sys.executable, str(PROFILE_HELPER), "machine:8gb", "--mode", "args"],
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout == b""


def _runtime_receipt(tmp_path, *, profile="", machine="8gb",
                     selector="machine:8gb") -> Path:
    comfy = tmp_path / "ComfyUI"
    models = comfy / "models"
    (comfy / "output").mkdir(parents=True)
    models.mkdir()
    (comfy / "folder_paths.py").write_text("# fixture\n", encoding="utf-8")
    receipt = tmp_path / "otr-runtime.env"
    secrets = tmp_path / "otr-secrets.env"
    secrets.write_text(
        "export OTR_COMFY_API_KEY='fixture-secret-file'\n",
        encoding="utf-8", newline="\n",
    )
    secrets.chmod(0o600)
    values = {
        "OTR_COMFY_ROOT": comfy.as_posix(),
        "OTR_REPO_ROOT": ROOT.as_posix(),
        "COMFY_PY": Path(sys.executable).as_posix(),
        "OTR_COMFYUI_MODELS_ROOT": models.as_posix(),
        "HF_HOME": (models / "huggingface").as_posix(),
        "OTR_INDEXTTS2_ROOT": (tmp_path / "index-tts").as_posix(),
        "OTR_INDEXTTS2_DIR": (tmp_path / "index-tts/checkpoints").as_posix(),
        "OTR_INDEXTTS2_WORKER": (ROOT / "scripts/_otr_indextts2_worker.py").as_posix(),
        "OTR_VOICE_REFERENCE_BANK": (tmp_path / "voice-bank.json").as_posix(),
        "OTR_PROVISION_PROFILE": profile,
        "OTR_PROVISION_MACHINE": machine,
        "OTR_PROVISION_SELECTOR": selector,
        "OTR_PROVISION_GENERATION": "fixture-generation-1",
        "OTR_HEADLESS_PORT": "8188",
        "OTR_RUNTIME_SECRETS_FILE": secrets.as_posix(),
    }
    receipt.write_bytes((
        "\n".join("export %s='%s'" % (key, value.replace("'", "'\\''"))
                  for key, value in values.items()) + "\n"
    ).encode("utf-8"))
    return receipt


def _load_runtime_receipt(receipt: Path, caller_selection: dict[str, str]):
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    env = os.environ.copy()
    for key in ("HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
                "OPENROUTER_API_KEY"):
        env.pop(key, None)
    env.update(caller_selection)
    env["OTR_RUNTIME_ENV"] = receipt.as_posix()
    env["OTR_OUTPUT_ROOT"] = (receipt.parent / "output").as_posix()
    env["OTR_SERVER_LOG"] = (receipt.parent / "runtime/server.log").as_posix()
    env["OTR_SERVER_FINGERPRINT"] = (
        receipt.parent / "runtime/server.receipt").as_posix()
    return subprocess.run(
        [bash, "-c", (
            'source "$1"; otr_secret_file_mode() { printf 600; }; '
            'otr_load_runtime || exit $?; '
            'printf "%s|%s|%s" "$OTR_PROVISION_PROFILE" '
            '"$OTR_PROVISION_MACHINE" "$OTR_PROVISION_SELECTOR"'
        ), "otr-runtime-test", RUNTIME.as_posix()],
        text=True, capture_output=True, env=env,
    )


def test_runtime_selection_trio_is_atomic_and_receipt_authoritative(tmp_path):
    receipt = _runtime_receipt(tmp_path)
    result = _load_runtime_receipt(receipt, {
        "OTR_PROVISION_PROFILE": "stale-profile",
        "OTR_PROVISION_MACHINE": "stale-machine",
        "OTR_PROVISION_SELECTOR": "stale-selector",
    })

    assert result.returncode == 0, result.stderr
    assert result.stdout == "|8gb|machine:8gb"


def test_runtime_receipt_replaces_a_stale_ambient_hf_cache(tmp_path):
    receipt = _runtime_receipt(tmp_path)
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    env = os.environ.copy()
    for key in ("HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
                "OPENROUTER_API_KEY"):
        env.pop(key, None)
    env["OTR_RUNTIME_ENV"] = receipt.as_posix()
    env["HF_HOME"] = "/stale/duplicate-cache"
    env["OTR_OUTPUT_ROOT"] = (receipt.parent / "output").as_posix()
    env["OTR_SERVER_LOG"] = (receipt.parent / "runtime/server.log").as_posix()
    env["OTR_SERVER_FINGERPRINT"] = (
        receipt.parent / "runtime/server.receipt").as_posix()
    result = subprocess.run(
        [bash, "-c", (
            'source "$1"; otr_secret_file_mode() { printf 600; }; '
            'otr_load_runtime || exit $?; printf "%s" "$HF_HOME"'
        ), "otr-runtime-test", RUNTIME.as_posix()],
        text=True, capture_output=True, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == (tmp_path / "ComfyUI/models/huggingface").as_posix()


def test_runtime_rejects_an_internally_inconsistent_selection_receipt(tmp_path):
    receipt = _runtime_receipt(
        tmp_path, profile="also-set", machine="8gb", selector="machine:8gb")
    result = _load_runtime_receipt(receipt, {})

    assert result.returncode != 0
    assert "runtime selection receipt is inconsistent" in result.stderr


def test_runtime_rejects_a_receipt_without_provision_generation(tmp_path):
    receipt = _runtime_receipt(tmp_path)
    receipt.write_bytes(receipt.read_bytes().replace(
        b"export OTR_PROVISION_GENERATION='fixture-generation-1'\n", b""))
    result = _load_runtime_receipt(receipt, {})

    assert result.returncode != 0
    assert "runtime receipt has no provision generation" in result.stderr


def test_runtime_loads_protected_secrets_but_preserves_explicit_overrides(
        tmp_path):
    receipt = _runtime_receipt(tmp_path)
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    env = os.environ.copy()
    for key in ("HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
                "OPENROUTER_API_KEY"):
        env.pop(key, None)
    env["OTR_RUNTIME_ENV"] = receipt.as_posix()
    env["OTR_COMFY_API_KEY"] = "caller-secret"
    env["OTR_OUTPUT_ROOT"] = (receipt.parent / "output").as_posix()
    env["OTR_SERVER_LOG"] = (receipt.parent / "runtime/server.log").as_posix()
    env["OTR_SERVER_FINGERPRINT"] = (
        receipt.parent / "runtime/server.receipt").as_posix()
    result = subprocess.run(
        [bash, "-c", (
            'source "$1"; otr_secret_file_mode() { printf 600; }; '
            'otr_load_runtime || exit $?; '
            'printf "%s|%s" "$OTR_COMFY_API_KEY" '
            '"${OPENROUTER_API_KEY:-missing}"'
        ), "otr-runtime-test", RUNTIME.as_posix()],
        text=True, capture_output=True, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "caller-secret|missing"


def test_runtime_xtrace_never_prints_secret_values(tmp_path):
    receipt = _runtime_receipt(tmp_path)
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    env = os.environ.copy()
    for key in ("HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
                "OPENROUTER_API_KEY"):
        env.pop(key, None)
    env["OTR_RUNTIME_ENV"] = receipt.as_posix()
    env["OTR_COMFY_API_KEY"] = "caller-sentinel-do-not-print"
    env["OTR_OUTPUT_ROOT"] = (receipt.parent / "output").as_posix()
    env["OTR_SERVER_LOG"] = (receipt.parent / "runtime/server.log").as_posix()
    env["OTR_SERVER_FINGERPRINT"] = (
        receipt.parent / "runtime/server.receipt").as_posix()
    result = subprocess.run(
        [bash, "-x", "-c", (
            'source "$1"; otr_secret_file_mode() { printf 600; }; '
            'otr_load_runtime || exit $?; printf ready'
        ), "otr-runtime-test", RUNTIME.as_posix()],
        text=True, capture_output=True, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ready"
    assert "caller-sentinel-do-not-print" not in result.stderr
    assert "fixture-secret-file" not in result.stderr


def test_runtime_xtrace_never_prints_fallback_hf_token(tmp_path):
    receipt = _runtime_receipt(tmp_path)
    token_file = tmp_path / "hf-token"
    token_file.write_text("fallback-sentinel-do-not-print\n", encoding="utf-8")
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    env = os.environ.copy()
    for key in ("HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
                "OPENROUTER_API_KEY"):
        env.pop(key, None)
    env["OTR_RUNTIME_ENV"] = receipt.as_posix()
    env["OTR_HF_TOKEN_FILE"] = token_file.as_posix()
    env["OTR_OUTPUT_ROOT"] = (receipt.parent / "output").as_posix()
    env["OTR_SERVER_LOG"] = (receipt.parent / "runtime/server.log").as_posix()
    env["OTR_SERVER_FINGERPRINT"] = (
        receipt.parent / "runtime/server.receipt").as_posix()
    result = subprocess.run(
        [bash, "-x", "-c", (
            'source "$1"; otr_secret_file_mode() { printf 600; }; '
            'otr_load_runtime || exit $?; printf ready'
        ), "otr-runtime-test", RUNTIME.as_posix()],
        text=True, capture_output=True, env=env,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "ready"
    assert "fallback-sentinel-do-not-print" not in result.stderr


def test_runtime_restores_xtrace_after_rejecting_bad_secret_permissions(
        tmp_path):
    receipt = _runtime_receipt(tmp_path)
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    env = os.environ.copy()
    for key in ("HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
                "OPENROUTER_API_KEY"):
        env.pop(key, None)
    env["OTR_RUNTIME_ENV"] = receipt.as_posix()
    result = subprocess.run(
        [bash, "-x", "-c", (
            'source "$1"; otr_secret_file_mode() { printf 644; }; '
            'otr_load_runtime || printf "|after-error|"; printf done'
        ), "otr-runtime-test", RUNTIME.as_posix()],
        text=True, capture_output=True, env=env,
    )

    assert result.returncode == 0
    assert result.stdout == "|after-error|done"
    assert "protected runtime secret file must be mode" in result.stderr
    assert "+ printf done" in result.stderr


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


def test_server_receipt_rejects_a_stale_boot_and_accepts_exact_listener(
        tmp_path):
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    digest = "a" * 64
    wanted = tmp_path / "wanted.txt"
    stale = tmp_path / "stale.receipt"
    current = tmp_path / "current.receipt"
    wanted.write_text(digest + "\n", encoding="utf-8", newline="\n")
    stale.write_text(
        "fingerprint=%s\nboot_id=old-boot\npid=222\nstart_ticks=900\n"
        % digest,
        encoding="utf-8", newline="\n",
    )
    current.write_text(
        "fingerprint=%s\nboot_id=current-boot\npid=222\nstart_ticks=900\n"
        % digest,
        encoding="utf-8", newline="\n",
    )
    script = (
        'source "$1"; '
        'OTR_HEADLESS_PORT=8188; '
        'otr_current_boot_id() { printf current-boot; }; '
        'otr_process_start_ticks() { [[ "$1" == 222 ]] && printf 900; }; '
        'otr_listener_pid_matches() { [[ "$1" == 222 ]]; }; '
        'OTR_SERVER_FINGERPRINT="$2"; '
        'if otr_server_receipt_matches "$4"; then exit 9; fi; '
        'OTR_SERVER_FINGERPRINT="$3"; '
        'otr_server_receipt_matches "$4"'
    )
    result = subprocess.run(
        [bash, "-c", script, "otr-runtime-test", RUNTIME.as_posix(),
         stale.as_posix(), current.as_posix(), wanted.as_posix()],
        text=True, capture_output=True,
    )

    assert result.returncode == 0, result.stderr


def test_runtime_owner_uses_checked_files_and_excludes_cloud_h3():
    text = RUNTIME.read_text(encoding="utf-8")
    assert "otr_profile_output" in text
    assert 'if ! "$COMFY_PY" "$helper"' in text
    assert "mapfile -t launch_args < \"$args_file\"" in text
    assert "mapfile -t launch_args < <(" not in text
    assert '== "h3" || "$contract_name" == h3_*' in text
    assert "operator-local and cannot run in the cloud roster" in text
    assert "--listen 0.0.0.0" in text
    assert "OTR_SERVER_FINGERPRINT" in text
    assert "otr_server_receipt_matches" in text
    assert "boot_id=" in text
    assert "start_ticks=" in text
    assert "otr_acquire_campaign_lock" in text
    assert "otr_stop_campaign" in text
    assert "otr_release_campaign_lock" in text
    assert "/proc/$pid/cmdline" in text
    boot = text[text.index("otr_boot_profile()"):
                text.index("otr_ensure_profile_server()")]
    assert "exec {OTR_CAMPAIGN_LOCK_FD}>&-" in boot

    sweep = SWEEP.read_text(encoding="utf-8")
    soak = SOAK.read_text(encoding="utf-8")
    assert 'otr_acquire_campaign_lock "overnight sweep"' in sweep
    assert 'otr_acquire_campaign_lock "continuous soak"' in soak
    assert "/root/leg_" not in sweep
    assert "/root/soak_" not in soak
    assert "$OTR_POD_LOG_DIR" in sweep
    assert "$OTR_POD_LOG_DIR" in soak

    playbook = (ROOT / "docs" / "RUNPOD_INSTALL.md").read_text(
        encoding="utf-8")
    assert 'otr_load_runtime || exit $?' in playbook
    assert 'otr_acquire_campaign_lock "manual qualification" || exit $?' in \
        playbook
    assert "trap 'otr_release_campaign_lock' EXIT" in playbook
    assert 'otr_boot_profile "$OTR_PROVISION_SELECTOR" || exit $?' in playbook
    assert "trap - EXIT" in playbook


def test_campaign_lock_refuses_a_second_driver(tmp_path):
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    if subprocess.run(
            [bash, "-lc", "command -v flock"], capture_output=True).returncode:
        pytest.skip("flock is not installed")
    log_dir = tmp_path.as_posix()
    holder = subprocess.Popen(
        [bash, "-c", (
            'source "$1"; OTR_POD_LOG_DIR="$2"; '
            'otr_acquire_campaign_lock holder || exit $?; '
            'printf ready; read -r _'
        ), "otr-lock-holder", RUNTIME.as_posix(), log_dir],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.read(5) == "ready"
        contender = subprocess.run(
            [bash, "-c", (
                'source "$1"; OTR_POD_LOG_DIR="$2"; '
                'otr_acquire_campaign_lock contender'
            ), "otr-lock-contender", RUNTIME.as_posix(), log_dir],
            text=True, capture_output=True,
        )
        assert contender.returncode != 0
        assert "another OTR pod campaign is active" in contender.stderr
        assert "kind=holder" in contender.stderr
    finally:
        if holder.stdin is not None:
            holder.stdin.write("\n")
            holder.stdin.flush()
        holder.communicate(timeout=10)


def test_discovered_roster_fails_when_every_plan_is_excluded(tmp_path):
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    fake_repo = tmp_path / "repo"
    profiles = fake_repo / "config/profiles"
    scripts = fake_repo / "scripts"
    profiles.mkdir(parents=True)
    scripts.mkdir()
    (profiles / "otr_w45_only.json").write_text("{}\n", encoding="utf-8")
    (scripts / "otr_provision.py").write_text("# fixture\n", encoding="utf-8")
    rejector = tmp_path / "reject-plan.sh"
    rejector.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    rejector.chmod(0o755)
    result = subprocess.run(
        [bash, "-c", (
            'source "$1"; OTR_REPO_ROOT="$2"; COMFY_PY="$3"; '
            'unset OTR_POD_PROFILES; otr_profile_roster'
        ), "otr-roster-test", RUNTIME.as_posix(), fake_repo.as_posix(),
         rejector.as_posix()],
        text=True, capture_output=True,
    )

    assert result.returncode != 0
    assert "no pod profiles have a complete runnable provision plan" in \
        result.stderr


def test_explicit_8gb_h3_lab_profile_cannot_bypass_cloud_exclusion(tmp_path):
    bash = _bash()
    if not bash:
        pytest.skip("bash is not installed")
    fake_repo = tmp_path / "repo"
    (fake_repo / "scripts").mkdir(parents=True)
    script = (
        'source "$1"; OTR_REPO_ROOT="$2"; COMFY_PY="$(command -v true)"; '
        'OTR_POD_PROFILES=otr_4060_h3_nano; '
        'otr_profile_output() { printf h3_8gb_lab > "$3"; }; '
        'otr_profile_roster'
    )
    result = subprocess.run(
        [bash, "-c", script, "otr-h3-roster-test", RUNTIME.as_posix(),
         fake_repo.as_posix()],
        text=True, capture_output=True,
    )

    assert result.returncode != 0
    assert "operator-local and cannot run in the cloud roster" in result.stderr


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
