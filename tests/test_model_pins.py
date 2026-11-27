"""Kokoro, its ONNX export and the Whisper model are fetched pinned and checked.

Until 2026-11-27 all three came from each repository's current revision with
no check on the bytes, and the torch Kokoro path never read the voice file
the startup fetch had copied: it handed KPipeline a bare voice id, and
KPipeline downloaded ``voices/<id>.pt`` and the model from ``main`` itself.
These tests pin the table, the fetch, the reuse of a cached copy, the torch
wiring and the Whisper folder. CPU only; every network call is faked.
"""
from __future__ import annotations

import hashlib
import os
import sys
import types

import pytest

os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import _otr_model_pins as pins  # noqa: E402
from nodes._otr_audio_engines import _kokoro_backends as kb  # noqa: E402
import nodes._otr_kokoro_voice_prefetch as prefetch  # noqa: E402


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pin(monkeypatch, source, filename, data):
    """Make ``data`` the pinned content of one file."""
    repo, revision, files = pins.SOURCES[source]
    monkeypatch.setitem(pins.SOURCES, source, (repo, revision, dict(
        files, **{filename: (len(data), _sha(data))})))


@pytest.fixture(autouse=True)
def _empty_hub_cache(monkeypatch, tmp_path):
    import huggingface_hub.constants

    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(tmp_path / "hub"))


# --------------------------------------------------------------------------- #
# the table
# --------------------------------------------------------------------------- #
def test_every_source_is_one_commit_and_every_file_a_size_and_digest():
    for source, (repo, revision, files) in pins.SOURCES.items():
        assert len(revision) == 40 and int(revision, 16) >= 0, source
        assert "/" in repo and files, source
        for name, (size, sha256) in files.items():
            assert type(size) is int and size > 0, name
            assert len(sha256) == 64 and int(sha256, 16) >= 0, name


def test_every_voice_the_show_can_cast_has_a_pin():
    pinned = {name[len("voices/"):-3] for name in pins.filenames(pins.KOKORO)
              if name.startswith("voices/")}
    assert set(prefetch.admitted_kokoro_voices()) <= pinned
    assert {"config.json", "kokoro-v1_0.pth"} <= set(pins.filenames(pins.KOKORO))


def test_the_pins_name_the_repositories_the_code_uses():
    assert pins.repo_and_revision(pins.KOKORO)[0] == prefetch.KOKORO_REPO_ID == kb.KOKORO_REPO_ID
    assert pins.repo_and_revision(pins.KOKORO_ONNX)[0] == prefetch.KOKORO_ONNX_REPO_ID
    assert pins.pin_for(pins.KOKORO_ONNX, prefetch.KOKORO_ONNX_FILENAME) is not None
    assert pins.repo_and_revision(pins.WHISPER_BASE)[0] == "Systran/faster-whisper-base"


def test_the_module_loads_by_file_with_no_package():
    """prestartup loads the prefetch, and so this module, before ComfyUI has
    imported any node package: stdlib only, no relative import."""
    import ast

    tree = ast.parse(open(pins.__file__, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0, "a relative import cannot load by file"
            assert node.module in ("__future__", "huggingface_hub"), node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name in ("hashlib", "os", "threading"), alias.name


# --------------------------------------------------------------------------- #
# fetch
# --------------------------------------------------------------------------- #
def test_fetch_asks_for_the_pinned_commit_with_no_token(monkeypatch, tmp_path):
    seen = {}

    def fake_download(**kwargs):
        seen.update(kwargs)
        path = tmp_path / "got.pt"
        path.write_bytes(b"voice")
        return str(path)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        hf_hub_download=fake_download, constants=types.SimpleNamespace(HF_HUB_CACHE=str(tmp_path / "hub"))))
    _pin(monkeypatch, pins.KOKORO, "voices/bf_alice.pt", b"voice")
    out = pins.fetch(pins.KOKORO, "voices/bf_alice.pt", revision="main", token="hf_should_not_be_sent")
    assert out == str(tmp_path / "got.pt")
    assert seen["revision"] == pins.repo_and_revision(pins.KOKORO)[1]
    assert seen["token"] is False
    assert seen["repo_id"] == "hexgrad/Kokoro-82M"


def test_a_wrong_download_is_refused_and_removed_from_a_local_dir(monkeypatch, tmp_path):
    local = tmp_path / "engine_dir"

    def fake_download(**kwargs):
        target = local / "onnx" / "model.onnx"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not the pinned bytes")
        return str(target)

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=fake_download))
    _pin(monkeypatch, pins.KOKORO_ONNX, "onnx/model.onnx", b"the pinned bytes")
    with pytest.raises(pins.PinMismatch, match="not the pinned onnx-community"):
        pins.fetch(pins.KOKORO_ONNX, "onnx/model.onnx", local_dir=str(local))
    assert not (local / "onnx" / "model.onnx").exists()


def test_a_wrong_file_that_was_already_there_is_refused_but_kept(monkeypatch, tmp_path):
    """Sonnet review, 2026-11-27: offline, the hub hands back whatever sits in
    a local_dir without asking which commit it came from. A refusal there
    must not delete the user's file; only a file this call wrote is removed."""
    local = tmp_path / "engine_dir"
    target = local / "onnx" / "model.onnx"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"a file the user already had")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        hf_hub_download=lambda **kwargs: str(target)))
    _pin(monkeypatch, pins.KOKORO_ONNX, "onnx/model.onnx", b"the pinned bytes")
    with pytest.raises(pins.PinMismatch):
        pins.fetch(pins.KOKORO_ONNX, "onnx/model.onnx", local_dir=str(local))
    assert target.read_bytes() == b"a file the user already had"


def test_a_file_with_no_pin_is_never_fetched(monkeypatch):
    def detonate(**_kwargs):
        raise AssertionError("an unpinned file reached the network")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=detonate))
    with pytest.raises(pins.PinMismatch, match="no pinned copy"):
        pins.fetch(pins.KOKORO, "voices/not_a_voice.pt")


def test_a_cached_copy_from_an_older_commit_is_used_offline(monkeypatch, tmp_path):
    """Codex contrarian, 2026-11-27: an offline box whose cache was filled by
    the old unpinned code under an EARLIER commit must keep working. Any
    snapshot holding the pinned bytes is used, with no network call."""
    hub = tmp_path / "hub"
    old = hub / "models--hexgrad--Kokoro-82M" / "snapshots" / ("a" * 40)
    old.mkdir(parents=True)
    (old / "kokoro-v1_0.pth").write_bytes(b"weights")
    wrong = hub / "models--hexgrad--Kokoro-82M" / "snapshots" / ("b" * 40)
    wrong.mkdir(parents=True)
    (wrong / "kokoro-v1_0.pth").write_bytes(b"other weights")

    def detonate(**_kwargs):
        raise AssertionError("a verified cached copy must not touch the network")

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(
        hf_hub_download=detonate, constants=types.SimpleNamespace(HF_HUB_CACHE=str(hub))))
    _pin(monkeypatch, pins.KOKORO, "kokoro-v1_0.pth", b"weights")
    assert pins.fetch(pins.KOKORO, "kokoro-v1_0.pth") == str(old / "kokoro-v1_0.pth")
    # The copy with other bytes is skipped, never deleted.
    assert (wrong / "kokoro-v1_0.pth").read_bytes() == b"other weights"


def test_check_refuses_other_bytes_and_leaves_unpinned_names_alone(tmp_path, monkeypatch):
    _pin(monkeypatch, pins.WHISPER_BASE, "config.json", b"{}")
    good, bad = tmp_path / "good.json", tmp_path / "bad.json"
    good.write_bytes(b"{}")
    bad.write_bytes(b"{ }")
    pins.check(pins.WHISPER_BASE, "config.json", str(good))
    with pytest.raises(pins.PinMismatch, match="Delete it and restart ComfyUI"):
        pins.check(pins.WHISPER_BASE, "config.json", str(bad))
    pins.check(pins.WHISPER_BASE, "a_file_we_never_fetch.bin", str(bad))


# --------------------------------------------------------------------------- #
# the torch Kokoro path reads the pinned files
# --------------------------------------------------------------------------- #
def test_the_torch_model_is_built_from_the_pinned_files(monkeypatch):
    built = []

    class FakeKModel:
        # kokoro 0.9.4's signature, so repo_id is offered.
        def __init__(self, repo_id=None, config=None, model=None, disable_complex=False):
            built.append({"repo_id": repo_id, "config": config, "model": model})
            self.moves = []

        def to(self, device):
            self.moves.append(device)
            return self

        def eval(self):
            self.moves.append("eval")
            return self

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=FakeKModel))
    fetched = []
    monkeypatch.setattr(pins, "fetch", lambda source, name, **kw: fetched.append((source, name)) or "/pinned/" + name)
    model = kb.pinned_kokoro_model("cpu")
    assert fetched == [(pins.KOKORO, "config.json"), (pins.KOKORO, "kokoro-v1_0.pth")]
    assert built == [{"repo_id": "hexgrad/Kokoro-82M", "config": "/pinned/config.json",
                      "model": "/pinned/kokoro-v1_0.pth"}]
    assert model.moves == ["cpu", "eval"]


def test_kokoro_0_7_16_gets_no_repo_id(monkeypatch):
    """Sonnet review, 2026-11-27: the requirements admit kokoro 0.7.16, whose
    KModel takes config= and model= but no repo_id (read from its wheel:
    ``def __init__(self, config=None, model=None)``). Passing repo_id there
    raised TypeError on every torch voice line."""
    built = []

    class KModel0716:
        def __init__(self, config=None, model=None):
            built.append({"config": config, "model": model})

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=KModel0716))
    monkeypatch.setattr(pins, "fetch", lambda source, name, **kw: "/pinned/" + name)
    kb.pinned_kokoro_model("cpu")
    assert built == [{"config": "/pinned/config.json", "model": "/pinned/kokoro-v1_0.pth"}]


def test_cuda_that_is_not_there_is_refused_not_moved_to_the_cpu(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KModel=object))
    monkeypatch.setattr(pins, "fetch", lambda *a, **k: pytest.fail("fetched before the refusal"))
    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        kb.pinned_kokoro_model("cuda")


def test_kpipeline_gets_the_pinned_model_and_the_local_voice_tensor(monkeypatch, tmp_path):
    """The wiring. Given a bare voice id, kokoro's KPipeline downloads
    voices/<id>.pt from the repository's `main`; given a tensor it uses it."""
    seen = {}

    class FakeKPipeline:
        def __init__(self, **kwargs):
            seen["init"] = kwargs

        def __call__(self, text, voice=None, **kwargs):
            seen.setdefault("voices", []).append(voice)
            import numpy as np
            yield ("g", "p", np.full(8, 0.5, dtype=np.float32))

    monkeypatch.setitem(sys.modules, "kokoro", types.SimpleNamespace(KPipeline=FakeKPipeline))
    monkeypatch.setattr(kb, "pinned_kokoro_model", lambda device: ("pinned", device))
    loads = []
    monkeypatch.setattr(kb, "load_local_voice", lambda folder, voice: loads.append((folder, voice)) or ("tensor", voice))
    backend = kb.TorchKokoroBackend("cpu", lang_code="e", voices_dir=str(tmp_path))
    backend.load()
    backend.synthesize("hola", "ef_dora", 1.0)
    backend.synthesize("otra vez", "ef_dora", 1.0)
    assert seen["init"]["model"] == ("pinned", "cpu")
    assert seen["voices"] == [("tensor", "ef_dora"), ("tensor", "ef_dora")]
    assert loads == [(str(tmp_path), "ef_dora")]          # read once per backend


def test_a_local_voice_is_read_from_the_folder_and_nothing_else(tmp_path):
    import torch

    voices = tmp_path / "TTS" / "KokoroTTS" / "voices"
    voices.mkdir(parents=True)
    torch.save(torch.ones(3, 1, 256), voices / "bf_alice.pt")
    pack = kb.load_local_voice(str(voices), "bf_alice")
    assert tuple(pack.shape) == (3, 1, 256)
    with pytest.raises(ValueError, match="not a bare voice name"):
        kb.load_local_voice(str(voices), "../bf_alice")
    with pytest.raises(kb.BackendUnavailable, match="voices folder is not known"):
        kb.load_local_voice(None, "bf_alice")


# --------------------------------------------------------------------------- #
# Whisper
# --------------------------------------------------------------------------- #
def _pin_whisper(monkeypatch):
    repo, revision, files = pins.SOURCES[pins.WHISPER_BASE]
    monkeypatch.setitem(pins.SOURCES, pins.WHISPER_BASE, (repo, revision, {
        name: (len(name.encode()), _sha(name.encode())) for name in files}))
    return revision, sorted(files)


def test_a_whisper_folder_that_is_already_pinned_makes_no_call(monkeypatch, tmp_path):
    from nodes._otr_video_engines import foley_speech as speech

    _revision, names = _pin_whisper(monkeypatch)
    for name in names:
        (tmp_path / name).write_bytes(name.encode())
    out = speech._pinned_whisper_dir(tmp_path, lambda *a, **k: pytest.fail("network on a pinned folder"))
    assert out == str(tmp_path)


def test_whisper_is_fetched_at_the_commit_and_a_wrong_file_is_refused(monkeypatch, tmp_path):
    from nodes._otr_video_engines import foley_speech as speech

    revision, names = _pin_whisper(monkeypatch)
    calls = []

    def fake_snapshot(repo, **kwargs):
        calls.append((repo, kwargs))
        for name in kwargs["allow_patterns"]:
            (tmp_path / name).write_bytes(b"tampered" if name == "model.bin" else name.encode())
        return kwargs["local_dir"]

    with pytest.raises(pins.PinMismatch, match="model.bin"):
        speech._pinned_whisper_dir(tmp_path, fake_snapshot)
    assert calls[0][0] == "Systran/faster-whisper-base"
    assert calls[0][1]["revision"] == revision and calls[0][1]["token"] is False
    assert sorted(calls[0][1]["allow_patterns"]) == names
