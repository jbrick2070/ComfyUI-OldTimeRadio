from pathlib import Path
import shutil
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
POD = REPO / "scripts" / "otr_pod_provision.sh"


def test_pod_script_has_valid_bash_syntax():
    bash = shutil.which("bash")
    if not bash:
        windows_git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        bash = str(windows_git_bash) if windows_git_bash.is_file() else None
    if not bash:
        pytest.skip("bash is not installed")
    subprocess.run([bash, "-n", str(POD)], check=True)


def test_pod_script_has_one_pack_and_weight_owner():
    text = POD.read_text(encoding="utf-8")
    pin_at = text.index("checkout -q --detach FETCH_HEAD")
    plan_at = text.index('"${VOICE_ARGS[@]}" --check-plan')
    packs_at = text.index("scripts/otr_provision.py --packs-only")
    selection_at = text.index(
        'scripts/otr_provision.py "${PROVISION_ARGS[@]}"')

    assert plan_at < pin_at < packs_at < selection_at
    assert "ComfyUI-AnimateDiff-Evolved|" not in text
    assert 'git clone -q -b v2.0-alpha "$OTR_REPO_URL" "$OTR_ROOT"' in text
    assert "otr_fetch_lane_weights.py \"$L\"" not in text
    assert "ComfyUI-GGUF.git" not in text
    assert "ComfyUI-LTXVideo.git" not in text
    assert "OTR_COMFY_CORE_PIN" in text
    assert "OTR_COMFY_ROOT" in text


def test_pod_script_allows_template_owned_untracked_entries(tmp_path):
    text = POD.read_text(encoding="utf-8")
    assert 'status --porcelain --untracked-files=no' in text
    assert 'status --porcelain --untracked-files=all' not in text

    repo = tmp_path / "comfy-core"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "config", "user.name", "OTR test"],
        check=True,
    )
    tracked = repo / "folder_paths.py"
    tracked.write_text("CORE = True\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "folder_paths.py"], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "fixture"],
        check=True,
    )

    (repo / ".venv-cu128").mkdir()
    (repo / ".venv-cu128" / "pyvenv.cfg").write_text(
        "home = /usr/bin\n", encoding="utf-8"
    )
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert status.stdout == ""

    tracked.write_text("CORE = False\n", encoding="utf-8")
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "folder_paths.py" in status.stdout

    subprocess.run(["git", "-C", str(repo), "add", "folder_paths.py"], check=True)
    status = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "status",
            "--porcelain",
            "--untracked-files=no",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "folder_paths.py" in status.stdout


def test_pod_script_repairs_the_runpod_cu130_driver_mismatch():
    text = POD.read_text(encoding="utf-8")

    assert 'COMFY_TORCH_CU128="2.10.0+cu128"' in text
    assert 'COMFY_TORCHVISION_CU128="0.25.0+cu128"' in text
    assert 'COMFY_TORCHAUDIO_CU128="2.10.0+cu128"' in text
    assert 'COMFY_TORCH_CU128_INDEX="https://download.pytorch.org/whl/cu128"' in text
    assert 'env -u PIP_CONSTRAINT "$COMFY_PY" -m pip install' in text
    assert 'unset PIP_CONSTRAINT' in text
    assert 'torch.cuda.is_available()' in text
    assert 'sample = torch.ones((64, 64)' in text
    assert 'result = sample @ sample' in text
    assert text.index("ensure_compatible_comfy_torch") < text.index(
        '"$COMFY_PY" -m pip install -q -r "$COMFY_ROOT/requirements.txt"'
    )
    final_probe = "FINAL_CUDA_PROBE=$(probe_comfy_cuda 2>&1)"
    assert final_probe in text
    assert text.index(final_probe) > text.index(
        'scripts/otr_provision.py "${PROVISION_ARGS[@]}"'
    )
    assert text.index(final_probe) < text.index("=== provision complete")


def test_pod_script_publishes_one_atomic_non_secret_runtime_receipt():
    text = POD.read_text(encoding="utf-8")
    keys_start = text.index("RUNTIME_KEYS=(")
    keys_end = text.index(")", keys_start)
    keys = text[keys_start:keys_end]
    selection_call = text.index(
        'scripts/otr_provision.py "${PROVISION_ARGS[@]}"')
    plan_check = text.index('"${VOICE_ARGS[@]}" --check-plan')
    receipt_publish = text.index('mv -f "$RUNTIME_TMP" "$OTR_RUNTIME_ENV"')

    # A bad selector/Python tuple must fail before either the expensive install
    # work or replacement of a previous good receipt.
    assert plan_check < text.index("scripts/otr_provision.py --packs-only")
    assert plan_check < receipt_publish < selection_call

    expected = [
        "OTR_COMFY_ROOT", "OTR_REPO_ROOT", "COMFY_PY",
        "OTR_COMFYUI_MODELS_ROOT", "HF_HOME", "OTR_INDEXTTS2_ROOT",
        "OTR_INDEXTTS2_DIR", "OTR_INDEXTTS2_WORKER",
        "OTR_VOICE_REFERENCE_BANK", "OTR_PROVISION_PROFILE",
        "OTR_PROVISION_MACHINE", "OTR_PROVISION_SELECTOR",
        "OTR_PROVISION_GENERATION",
        "OTR_HEADLESS_PORT", "OTR_RUNTIME_SECRETS_FILE",
    ]
    positions = [keys.index(name) for name in expected]
    assert positions == sorted(positions)
    assert "HF_TOKEN" not in keys
    assert "COMFYUI_URL" not in keys
    assert "OTR_INDEXTTS2_VENV" not in keys
    assert "printf 'export %s=%q\\n'" in text
    assert 'chmod 600 "$RUNTIME_TMP"' in text
    assert 'mv -f "$RUNTIME_TMP" "$OTR_RUNTIME_ENV"' in text
    assert keys_start < selection_call

    for secret in ("HF_TOKEN", "OTR_COMFY_API_KEY", "OTR_GOOGLE_API_KEY",
                   "OPENROUTER_API_KEY"):
        assert secret not in keys
    assert 'SECRETS_TMP=$(mktemp' in text
    assert 'chmod 600 "$SECRETS_TMP"' in text
    assert 'mv -f "$SECRETS_TMP" "$OTR_RUNTIME_SECRETS_FILE"' in text
    assert "SECRET_KEYS=(HF_TOKEN OTR_COMFY_API_KEY " \
           "OTR_GOOGLE_API_KEY OPENROUTER_API_KEY)" in text
    assert 'STORED_SECRETS["$SECRET_KEY"]' in text
    assert 'SECRET_VALUE="${STORED_SECRETS[$SECRET_KEY]:-}"' in text
    secret_block = text[text.index("SECRETS_XTRACE_WAS_ON=0"):
                        text.index("RUNTIME_TMP=$(mktemp")]
    assert secret_block.index("set +x") < secret_block.index("SECRET_VALUE=")
    assert secret_block.rindex("set -x") > secret_block.index("unset SECRET_KEY")

    token_block = text[text.index('HF_TOKEN_FILE="${OTR_HF_TOKEN_FILE:-/root/.hf_token}"'):
                       text.index("# Bash owns only the OTR checkout")]
    assert token_block.index('[ -n "${HF_TOKEN:-}" ]') < \
        token_block.index('[ -s "$HF_TOKEN_FILE" ]')

    assert 'OTR_REPO_ROOT="$OTR_ROOT"' in text
    assert 'OTR_REPO_ROOT="${OTR_REPO_ROOT:-$OTR_ROOT}"' not in text


def test_pod_default_uses_machine_matrix_and_profiles_are_explicit_overrides():
    text = POD.read_text(encoding="utf-8")

    assert 'PROFILE="${OTR_PROVISION_PROFILE:-}"' in text
    assert 'MACHINE="${OTR_PROVISION_MACHINE:-}"' in text
    assert 'MACHINE="16gb"' in text
    assert 'MACHINE="12gb"' in text
    assert 'MACHINE="8gb"' in text
    assert 'PROVISION_ARGS=(--machine "$MACHINE")' in text
    assert 'PROVISION_ARGS=(--profile "$PROFILE")' in text
    assert 'SELECTOR="machine:$MACHINE"' in text
    assert 'PROFILE="otr_runpod_starter"' not in text
    assert '[ -n "${VRAM_MIB:-}" ]' in text
    assert '[ "$VRAM_MIB" -lt 8000 ]' in text
    assert "below the supported 8 GB machine floor" in text


def test_legacy_cloud_scripts_cannot_bypass_the_owner():
    setup = (REPO / "scripts" / "setup_cloud.sh").read_text(encoding="utf-8")
    download = (REPO / "scripts" / "download_models.sh").read_text(encoding="utf-8")

    assert "otr_pod_provision.sh" in setup
    assert "ComfyUI-GGUF" not in setup
    assert "ComfyUI-LTXVideo" not in setup
    assert "huggingface-cli download" not in download
    assert "otr_fetch_lane_weights.py --list" in download
    assert "exit 2" in download
