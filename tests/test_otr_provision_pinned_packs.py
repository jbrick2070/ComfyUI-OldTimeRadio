from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
import types

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "otr_provision.py"
FIXTURE_LOADER = REPO / "tests" / "fixtures" / "comfyui_gguf_6ea2651" / "loader.py"


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


def test_runpod_ltx_manual_recipe_carries_authoritative_manifest():
    """The copy/paste recipe may not drift from the executable manifest."""
    provision = _load_provision()
    lab = (REPO / "docs" / "RUNPOD_PORTABILITY_LAB.md").read_text(
        encoding="utf-8"
    )
    artifacts = provision.MANUAL_TIERS["ltx25"]

    assert len(artifacts) == 5
    for artifact in artifacts:
        for field in ("repo", "revision", "path", "destination", "sha256"):
            assert artifact[field] in lab, (
                f"LTX 2.5 manual recipe is missing {field}="
                f"{artifact[field]!r}"
            )
        assert str(artifact["bytes"]) in lab


@pytest.mark.parametrize("line_ending", [b"\n", b"\r\n"])
def test_gguf_patch_applies_to_exact_normalized_preimage(tmp_path, monkeypatch, line_ending):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    dest = comfy / "custom_nodes" / provision.GGUF_PACK_NAME
    clean = FIXTURE_LOADER.read_bytes().replace(b"\n", line_ending)
    pin = _make_repo(dest, {"loader.py": clean, "requirements.txt": b"gguf\n"})
    installed = []
    monkeypatch.setattr(provision, "GGUF_PIN", pin)
    monkeypatch.setattr(
        provision,
        "install_pack_requirements",
        lambda name, root, required=False: installed.append((name, required)),
    )

    provision.ensure_gguf_pack(str(comfy))

    assert provision._normalized_sha256(str(dest / "loader.py")) == provision.GGUF_PATCHED_SHA256
    assert provision._git_changed_paths(str(dest)) == ["loader.py"]
    assert provision._git_untracked_paths(str(dest)) == []
    assert installed == [(provision.GGUF_PACK_NAME, True)]

    # The runtime gate must recognize the exact source the provisioner just
    # produced, not a hand-built approximation or guessed install path.
    from nodes._otr_video_engines import eng_ltx25
    module_name = "_otr_provisioned_gguf_fixture"
    registered_module = types.ModuleType(module_name)
    registered_module.__file__ = str(dest / "nodes.py")
    monkeypatch.setitem(sys.modules, module_name, registered_module)
    registered_cls = type(
        "CLIPLoaderGGUF", (), {"__module__": module_name})
    loader_path, gaps = eng_ltx25._inspect_ltx25_gguf_patch(registered_cls)
    assert loader_path == str((dest / "loader.py").resolve())
    assert gaps == ()


def test_gguf_already_patched_is_idempotent(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    dest = comfy / "custom_nodes" / provision.GGUF_PACK_NAME
    pin = _make_repo(
        dest,
        {"loader.py": FIXTURE_LOADER.read_bytes(), "requirements.txt": b"gguf\n"},
    )
    monkeypatch.setattr(provision, "GGUF_PIN", pin)
    monkeypatch.setattr(provision, "install_pack_requirements", lambda *args, **kwargs: None)

    provision.ensure_gguf_pack(str(comfy))
    first = (dest / "loader.py").read_bytes()
    provision.ensure_gguf_pack(str(comfy))

    assert (dest / "loader.py").read_bytes() == first
    assert provision._git_changed_paths(str(dest)) == ["loader.py"]


def test_manager_install_accepts_only_exact_patched_loader(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    source = comfy / "custom_nodes" / provision.GGUF_PACK_NAME
    pin = _make_repo(
        source,
        {"loader.py": FIXTURE_LOADER.read_bytes(), "requirements.txt": b"gguf\n"},
    )
    monkeypatch.setattr(provision, "GGUF_PIN", pin)
    monkeypatch.setattr(provision, "install_pack_requirements", lambda *args, **kwargs: None)
    provision.ensure_gguf_pack(str(comfy))

    manager_comfy = _comfy(tmp_path / "manager")
    manager_pack = manager_comfy / "custom_nodes" / provision.GGUF_PACK_NAME
    manager_pack.mkdir()
    (manager_pack / "loader.py").write_bytes((source / "loader.py").read_bytes())
    (manager_pack / "requirements.txt").write_text("gguf\n", encoding="utf-8")
    provision.ensure_gguf_pack(str(manager_comfy))

    (manager_pack / "loader.py").write_bytes(FIXTURE_LOADER.read_bytes())
    with pytest.raises(provision.ProvisionFailure, match="clean base"):
        provision.ensure_gguf_pack(str(manager_comfy))


def test_gguf_refuses_wrong_commit_and_dirty_drift(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    dest = comfy / "custom_nodes" / provision.GGUF_PACK_NAME
    pin = _make_repo(
        dest,
        {"loader.py": FIXTURE_LOADER.read_bytes(), "requirements.txt": b"gguf\n"},
    )
    monkeypatch.setattr(provision, "install_pack_requirements", lambda *args, **kwargs: None)

    monkeypatch.setattr(provision, "GGUF_PIN", "0" * 40)
    with pytest.raises(provision.ProvisionFailure, match="required"):
        provision.ensure_gguf_pack(str(comfy))

    monkeypatch.setattr(provision, "GGUF_PIN", pin)
    (dest / "requirements.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(provision.ProvisionFailure, match="dirty checkout"):
        provision.ensure_gguf_pack(str(comfy))


def test_ltxvideo_fresh_exact_checkout_and_drift_refusal(tmp_path, monkeypatch):
    provision = _load_provision()
    comfy = _comfy(tmp_path)
    upstream = tmp_path / "ltx-upstream"
    pin = _make_repo(upstream, {"requirements.txt": b"torch\n", "node.py": b"VALUE = 1\n"})
    installed = []
    monkeypatch.setattr(provision, "LTXVIDEO_URL", str(upstream))
    monkeypatch.setattr(provision, "LTXVIDEO_PIN", pin)
    monkeypatch.setattr(
        provision,
        "install_pack_requirements",
        lambda name, root, required=False: installed.append((name, required)),
    )

    provision.ensure_ltxvideo_pack(str(comfy))
    dest = comfy / "custom_nodes" / provision.LTXVIDEO_PACK_NAME
    assert _git(dest, "rev-parse", "HEAD") == pin
    assert installed == [(provision.LTXVIDEO_PACK_NAME, True)]

    (dest / "node.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(provision.ProvisionFailure, match="must be clean"):
        provision.ensure_ltxvideo_pack(str(comfy))


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
