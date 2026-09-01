"""IndexTTS2 downloads are revision-pinned and validated before readiness."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "_otr_idx_download_weights.py"
PROVISION = ROOT / "scripts" / "otr_provision.py"


def _load_downloader():
    spec = importlib.util.spec_from_file_location("_otr_idx_weights_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_provision():
    spec = importlib.util.spec_from_file_location(
        "_otr_idx_provision_test", PROVISION)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_weight_and_runtime_sources_are_exact_commits():
    downloader = _load_downloader()

    assert downloader._REPO_ID == "IndexTeam/IndexTTS-2"
    assert downloader._REPO_REVISION == (
        "740dcaff396282ffb241903d150ac011cd4b1ede")
    assert downloader._RUNTIME_REPOS == (
        ("facebook/w2v-bert-2.0",
         "da985ba0987f70aaeb84a80f2851cfac8c697a7b"),
        ("amphion/MaskGCT",
         "265c6cef07625665d0c28d2faafb1415562379dc"),
        ("funasr/campplus",
         "e4b6ede7ce16997aff4ae69fbca1f0175e2afede"),
        ("nvidia/bigvgan_v2_22khz_80band_256x",
         "633ff708ed5b74903e86ff1298cf4a98e921c513"),
    )
    assert downloader._EXPECTED["qwen0.6bemo4-merge/model.safetensors"] \
        == 1_192_135_096
    assert all(len(revision) == 40
               for _repo, revision in downloader._RUNTIME_REPOS)


def test_root_override_owns_checkpoints_when_dir_is_unset(tmp_path, monkeypatch):
    downloader = _load_downloader()
    source = tmp_path / "persistent" / "index-tts"
    monkeypatch.delenv("OTR_INDEXTTS2_DIR", raising=False)
    monkeypatch.setenv("OTR_INDEXTTS2_ROOT", str(source))

    assert Path(downloader._default_model_dir()) == source / "checkpoints"


def test_main_pins_every_snapshot_and_validates_nested_files(
        tmp_path, monkeypatch):
    downloader = _load_downloader()
    model_dir = tmp_path / "checkpoints"
    monkeypatch.setenv("OTR_INDEXTTS2_DIR", str(model_dir))
    monkeypatch.setattr(
        downloader, "_EXPECTED", {"one.bin": 3, "nested/two.bin": 2})
    monkeypatch.setattr(
        downloader, "_RUNTIME_EXPECTED",
        {repo: {"artifact.bin": 6} for repo, _revision in downloader._RUNTIME_REPOS})
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, dict(kwargs)))
        if repo_id == downloader._REPO_ID:
            model_dir.mkdir(parents=True, exist_ok=True)
            (model_dir / "one.bin").write_bytes(b"one")
            (model_dir / "nested").mkdir()
            (model_dir / "nested" / "two.bin").write_bytes(b"22")
        else:
            revision = kwargs["revision"]
            snapshot = (
                Path(kwargs["cache_dir"])
                / ("models--" + repo_id.replace("/", "--"))
                / "snapshots" / revision
            )
            snapshot.mkdir(parents=True)
            (snapshot / "artifact.bin").write_bytes(b"pinned")

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

    assert downloader.main() == 0
    assert calls[0] == (
        downloader._REPO_ID,
        {"revision": downloader._REPO_REVISION, "local_dir": str(model_dir)},
    )
    assert calls[1:] == [
        (repo, {"revision": revision,
                "cache_dir": str(model_dir / "hf_cache")})
        for repo, revision in downloader._RUNTIME_REPOS
    ]
    for repo, revision in downloader._RUNTIME_REPOS:
        ref = (model_dir / "hf_cache"
               / ("models--" + repo.replace("/", "--"))
               / "refs" / "main")
        assert ref.read_bytes() == revision.encode("ascii")


def test_exact_ref_is_resolved_by_real_huggingface_cache_api(tmp_path):
    from huggingface_hub import try_to_load_from_cache

    downloader = _load_downloader()
    repo, revision = downloader._RUNTIME_REPOS[0]
    repo_dir = tmp_path / ("models--" + repo.replace("/", "--"))
    snapshot = repo_dir / "snapshots" / revision
    snapshot.mkdir(parents=True)
    artifact = snapshot / "config.json"
    artifact.write_text("{}", encoding="utf-8")
    (repo_dir / "refs").mkdir()
    (repo_dir / "refs" / "main").write_bytes(revision.encode("ascii"))

    assert try_to_load_from_cache(
        repo, "config.json", cache_dir=tmp_path) == str(artifact)


def test_runtime_cache_verifier_rejects_newline_after_ref(tmp_path):
    downloader = _load_downloader()
    provision = _load_provision()
    repo, revision = downloader._RUNTIME_REPOS[0]
    snapshot = (tmp_path / ("models--" + repo.replace("/", "--"))
                / "snapshots" / revision)
    snapshot.mkdir(parents=True)
    (snapshot / "artifact.bin").write_bytes(b"ok")
    ref = snapshot.parent.parent / "refs" / "main"
    ref.parent.mkdir()
    ref.write_bytes(revision.encode("ascii") + b"\n")

    assert provision._runtime_cache_problems(
        str(tmp_path), repo, revision) == [
            "refs/main is not the exact 40-byte pinned commit"]


def test_main_returns_nonzero_for_download_or_validation_failure(
        tmp_path, monkeypatch):
    downloader = _load_downloader()
    monkeypatch.setenv("OTR_INDEXTTS2_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setattr(downloader, "_EXPECTED", {"required.bin": 4})

    fake_hf = types.ModuleType("huggingface_hub")
    fake_hf.snapshot_download = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)
    assert downloader.main() == 1

    fake_hf.snapshot_download = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("network failed"))
    assert downloader.main() == 1
