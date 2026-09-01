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
    packs_at = text.index("scripts/otr_provision.py --packs-only")
    profile_at = text.index("scripts/otr_provision.py --profile")

    assert pin_at < packs_at < profile_at
    assert "ComfyUI-AnimateDiff-Evolved|" not in text
    assert 'git clone -q -b v2.0-alpha "$OTR_REPO_URL" "$OTR_ROOT"' in text
    assert "otr_fetch_lane_weights.py \"$L\"" not in text
    assert "ComfyUI-GGUF.git" not in text
    assert "ComfyUI-LTXVideo.git" not in text
    assert "OTR_COMFY_CORE_PIN" in text
    assert "OTR_COMFY_ROOT" in text


def test_legacy_cloud_scripts_cannot_bypass_the_owner():
    setup = (REPO / "scripts" / "setup_cloud.sh").read_text(encoding="utf-8")
    download = (REPO / "scripts" / "download_models.sh").read_text(encoding="utf-8")

    assert "otr_pod_provision.sh" in setup
    assert "ComfyUI-GGUF" not in setup
    assert "ComfyUI-LTXVideo" not in setup
    assert "huggingface-cli download" not in download
    assert "otr_fetch_lane_weights.py --list" in download
    assert "exit 2" in download
