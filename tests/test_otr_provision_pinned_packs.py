from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "otr_provision.py"


def _load_provision():
    spec = importlib.util.spec_from_file_location("otr_provision_pinned_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_repo(path: Path, files: dict[str, bytes]) -> str:
    path.mkdir(parents=True)
    for name, data in files.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    _git(path, "init", "-q")
    _git(path, "config", "user.email", "tests@example.invalid")
    _git(path, "config", "user.name", "OTR tests")
    _git(path, "add", ".")
    _git(path, "commit", "-q", "-m", "fixture")
    return _git(path, "rev-parse", "HEAD")


def _comfy(tmp_path: Path) -> Path:
    root = tmp_path / "ComfyUI"
    (root / "custom_nodes").mkdir(parents=True)
    (root / "folder_paths.py").write_text("# fixture\n", encoding="utf-8")
    return root


def test_runpod_manual_recipes_carry_every_authoritative_manifest():
    """No manual receipt may point at a playbook that omits its exact files."""
    provision = _load_provision()
    playbook = (REPO / "apple" / "RUNPOD_INSTALL.md").read_text(
        encoding="utf-8"
    )
    assert set(provision.MANUAL_DOWNLOADS) == {
        "humo_1_7b",
    }
    for download_id, artifacts in provision.MANUAL_DOWNLOADS.items():
        assert artifacts, f"manual download {download_id} has no artifacts"
        for artifact in artifacts:
            for field in ("repo", "revision", "path", "destination", "sha256"):
                assert artifact[field] in playbook, (
                    f"{download_id} manual recipe is missing {field}="
                    f"{artifact[field]!r}"
                )
            assert str(artifact["bytes"]) in playbook
    assert 'mv -f "$part" "$dest" || {' in playbook
    assert 'rm -f "$part"' in playbook


def test_animatediff_pin_is_a_full_sha():
    provision = _load_provision()
    assert len(provision.ANIMATEDIFF_PIN) == 40
    assert set(provision.ANIMATEDIFF_PIN) <= set("0123456789abcdef")


def test_animatediff_fresh_exact_checkout_and_wrong_commit_refusal(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    upstream = tmp_path / "ade-upstream"
    pin = _make_repo(upstream, {"requirements.txt": b"torch\n", "node.py": b"VALUE = 1\n"})
    installed = []
    monkeypatch.setattr(provision, "ANIMATEDIFF_URL", str(upstream))
    monkeypatch.setattr(provision, "ANIMATEDIFF_PIN", pin)
    monkeypatch.setattr(
        provision,
        "install_pack_requirements",
        lambda name, root: installed.append(name),
    )

    provision.ensure_animatediff_pack(str(comfy))
    dest = comfy / "custom_nodes" / provision.ANIMATEDIFF_PACK_NAME
    assert _git(dest, "rev-parse", "HEAD") == pin           # exact detached checkout
    assert installed == [provision.ANIMATEDIFF_PACK_NAME]

    provision.ensure_animatediff_pack(str(comfy))          # PRESENT at the pin: idempotent
    assert len(installed) == 2

    monkeypatch.setattr(provision, "ANIMATEDIFF_PIN", "0" * 40)
    with pytest.raises(provision.ProvisionFailure, match="required 0000"):
        provision.ensure_animatediff_pack(str(comfy))      # a different commit is refused


def test_animatediff_manager_install_is_present_but_unverifiable(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    dest = comfy / "custom_nodes" / provision.ANIMATEDIFF_PACK_NAME
    dest.mkdir(parents=True)
    (dest / "node.py").write_text("VALUE = 1\n", encoding="utf-8")   # no .git: a Manager install
    installed = []
    monkeypatch.setattr(
        provision,
        "install_pack_requirements",
        lambda name, root: installed.append(name),
    )
    provision.ensure_animatediff_pack(str(comfy))
    assert installed == [provision.ANIMATEDIFF_PACK_NAME]


def test_packs_only_never_resolves_models_or_fetches_weights(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    calls = []
    monkeypatch.setattr(provision, "comfy_root", lambda: str(comfy))
    monkeypatch.setattr(provision, "install_node_packs", lambda root: calls.append("packs"))
    monkeypatch.setattr(provision, "install_requirements", lambda: calls.append("otr-deps"))
    for name in ("models_root", "profile_lanes", "ensure_hf_home", "fetch_lane_weights"):
        monkeypatch.setattr(
            provision,
            name,
            lambda *args, _name=name, **kwargs: pytest.fail("packs-only called %s" % _name),
        )

    assert provision.main(["--packs-only"]) == 0
    assert calls == ["packs", "otr-deps"]


def test_packs_only_failure_is_nonzero_and_clears_old_receipt(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    monkeypatch.setattr(provision, "comfy_root", lambda: str(comfy))
    provision._LOG.append(("FAILED", "stale", "old call"))

    def fail(_root):
        raise provision.ProvisionFailure("pinned pack mismatch")

    monkeypatch.setattr(provision, "install_node_packs", fail)
    monkeypatch.setattr(provision, "install_requirements", lambda: None)

    assert provision.main(["--packs-only"]) == 1
    assert all(row[1] != "stale" for row in provision._LOG)


def test_ltxvideo_is_no_longer_installed():
    """TEST_WAVE B4 (2026-09-26): otr_8gb_video published on a wiped 4060
    WITHOUT ComfyUI-LTXVideo; every LTX class is ComfyUI core. The provisioner
    installs AnimateDiff-Evolved and nothing else, and the patch is gone."""
    provision = _load_provision()
    assert not hasattr(provision, "ensure_ltxvideo_pack")
    assert not (REPO / "patches" / "ComfyUI-LTXVideo-kornia-pad.patch").exists()
    import inspect
    body = inspect.getsource(provision.install_node_packs)
    assert "ensure_animatediff_pack(comfy)" in body and "ensure_ltxvideo_pack(" not in body
