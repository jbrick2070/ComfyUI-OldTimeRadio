"""The LLM folder after its reviews (cursor + agy on 8f8ccebb).

Each test is one finding, and the case it names would have re-downloaded a
model the box already had, or blessed a half-finished one:

* a partial folder in an earlier root no longer hides a complete one later;
* a stale ``.tmp`` receipt is never listed and never poisons the check;
* the receipt follows a structural check, not ``snapshot_download``'s return
  (huggingface_hub returns a non-empty ``local_dir`` when the Hub is down);
* ``config.json`` is part of "complete", and an unsharded hand-placed model
  counts;
* with no LLM root at all -- in production, not only under the test gate --
  the hub cache is still the destination.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import _otr_llm_folder as LF  # noqa: E402
from nodes import _otr_model_catalog as cat  # noqa: E402
from tests.test_llm_folder import FOLDER, REPO_ID, SHARDS, _download_env, _hub, _plain  # noqa: E402


def _partial(root: Path):
    folder = root / FOLDER
    folder.mkdir(parents=True)
    (folder / SHARDS[0]).write_bytes(b"w" * 64)
    return folder


def test_a_complete_folder_in_a_later_root_beats_a_partial_one_earlier(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _partial(a)
    complete = _plain(b)
    (r,) = LF.scan_plain_models([a, b])
    assert r == (REPO_ID, complete, True)
    (s,) = cat.scan_local_llm_cache(hub_root=tmp_path / "no-hub", llm_roots=[a, b])
    assert s.on_disk and s.snapshot_path == str(complete)
    assert LF.find_plain_model(REPO_ID, [a, b]) == complete   # the two agree


def test_a_stale_tmp_receipt_is_not_listed_and_the_folder_stays_complete(tmp_path):
    folder = _plain(tmp_path, receipt=False)
    (folder / (LF.RECEIPT_NAME + ".tmp")).write_text("{crashed mid-write", encoding="utf-8")
    assert LF.write_receipt(folder, REPO_ID)
    files = json.loads((folder / LF.RECEIPT_NAME).read_text(encoding="utf-8"))["files"]
    assert not any(name.startswith(LF.RECEIPT_NAME) for name in files)
    assert LF.plain_folder_complete(folder)


def test_the_receipt_follows_the_structural_check_not_the_return(tmp_path, monkeypatch):
    """huggingface_hub returns an existing non-empty local_dir when the Hub is
    unreachable. A first shard was there; the return must not bless it."""
    monkeypatch.setenv("OTR_LLM_DIR", str(tmp_path / "LLM"))
    (tmp_path / "LLM").mkdir()
    _download_env(monkeypatch, tmp_path)
    from nodes._otr_model_inputs import UnknownModelError

    def offline_return(**kwargs):
        return str(_partial(Path(kwargs["local_dir"]).parent))

    with pytest.raises(UnknownModelError, match="did not finish"):
        cat.auto_download_if_missing(REPO_ID, hub_root=tmp_path / "hub",
                                     _snapshot_download=offline_return)
    assert not (tmp_path / "LLM" / FOLDER / LF.RECEIPT_NAME).exists()
    assert not LF.plain_folder_complete(tmp_path / "LLM" / FOLDER)


def test_config_json_is_part_of_complete(tmp_path):
    folder = _plain(tmp_path, receipt=False, config=False)
    assert not LF.plain_folder_complete(folder)             # index + shards, no config
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert LF.plain_folder_complete(folder)
    # and through the receipt path
    assert LF.write_receipt(folder, REPO_ID)
    assert LF.plain_folder_complete(folder)
    (folder / "config.json").unlink()
    assert not LF.plain_folder_complete(folder)


def test_an_unsharded_hand_placed_model_is_complete(tmp_path):
    folder = tmp_path / FOLDER
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    (folder / "model.safetensors").write_bytes(b"w" * 64)
    assert LF.plain_folder_complete(folder)
    # a transfer still in progress says otherwise
    (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (folder / ".cache" / "huggingface" / "download" / "model.safetensors.incomplete").write_bytes(b"?")
    assert not LF.plain_folder_complete(folder)


def test_a_sharded_folder_with_a_transfer_in_progress_is_not_complete(tmp_path):
    """cursor QA on 02758478: index and every shard at their final names, but
    a file still downloading -- not complete on the sharded branch either."""
    folder = _plain(tmp_path, receipt=False)
    assert LF.plain_folder_complete(folder)
    (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (folder / ".cache" / "huggingface" / "download" / "tokenizer.json.incomplete").write_bytes(b"?")
    assert not LF.plain_folder_complete(folder)


def test_a_lone_first_shard_is_never_complete_even_with_config(tmp_path):
    folder = _partial(tmp_path)
    (folder / "config.json").write_text("{}", encoding="utf-8")
    assert not LF.plain_folder_complete(folder)


def test_in_production_with_no_models_root_the_hub_is_still_the_destination(tmp_path, monkeypatch):
    """Not the test gate: OTR_TEST_MODE is unset here, folder_paths is absent,
    and the models root cannot be resolved."""
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    monkeypatch.delenv("OTR_LLM_DIR", raising=False)
    monkeypatch.setattr(LF, "default_llm_root", lambda: None)
    assert LF.llm_roots() == []
    _download_env(monkeypatch, tmp_path)
    hub = tmp_path / "hub"
    hub.mkdir()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        _hub(hub)
        return "x"

    cat.auto_download_if_missing(REPO_ID, hub_root=hub, _snapshot_download=fake_download)
    (kwargs,) = calls
    assert kwargs["cache_dir"] == str(hub) and "local_dir" not in kwargs
