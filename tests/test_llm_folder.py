"""The writer LLM as real files: ComfyUI's ``LLM`` model folder (2026-09-25).

What is pinned, and why each one is here:

* the FOLDER RULES (`_otr_llm_folder`): the name round-trips the repo id;
  other packs' bare-named folders are ignored; under OTR_TEST_MODE the roots
  are empty unless OTR_LLM_DIR pins one, so no test reads real models.
* COMPLETENESS: the receipt path and the index path each say "complete" only
  when every named file is present at size. The case Grok's QA named -- first
  shard present, index not yet landed -- is NOT complete.
* the MERGE (`scan_local_llm_cache`): a complete folder wins; a complete hub
  copy still wins over a half-downloaded folder; a partial folder with no hub
  copy is reported and not on disk.
* the DOWNLOAD: goes to the folder through `local_dir`, the disk check follows
  that drive, the receipt is written only after the download returned, and a
  model already complete in either layout is never downloaded again.
* the LOADER: resolves the folder, and looks for chat_template.jinja in the
  selected directory first (source inspection -- that is the wiring, not the
  helper).
"""
from __future__ import annotations

import inspect
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
from nodes import _otr_hf_env as hfenv  # noqa: E402

REPO_ID = "Qwen/Qwen3.5-4B"
FOLDER = "Qwen--Qwen3.5-4B"
SHARDS = ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")


def _index(folder: Path, shards=SHARDS):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"layer.%d" % i: s for i, s in enumerate(shards)}}), encoding="utf-8")


def _plain(root: Path, repo_id=REPO_ID, shards=SHARDS, receipt=True, config=True):
    folder = root / LF.folder_name(repo_id)
    _index(folder, shards)
    for s in shards:
        (folder / s).write_bytes(b"w" * 64)
    if config:
        (folder / "config.json").write_text(json.dumps({"max_position_embeddings": 32768}), encoding="utf-8")
    if receipt:
        assert LF.write_receipt(folder, repo_id)
    return folder


def _hub(root: Path, repo_id=REPO_ID, shards=SHARDS):
    snap = root / ("models--" + repo_id.replace("/", "--")) / "snapshots" / "abc123"
    _index(snap, shards)
    for s in shards:
        (snap / s).write_bytes(b"h" * 64)
    (snap / "config.json").write_text(json.dumps({"max_position_embeddings": 4096}), encoding="utf-8")
    return snap


@pytest.fixture
def llm_root(tmp_path, monkeypatch):
    root = tmp_path / "LLM"
    root.mkdir()
    monkeypatch.setenv("OTR_LLM_DIR", str(root))
    return root


# --------------------------------------------------------------------------- #
# names and roots
# --------------------------------------------------------------------------- #


def test_the_folder_name_round_trips_and_keeps_the_org():
    assert LF.folder_name("google/gemma-4-12b-it") == "google--gemma-4-12b-it"
    assert LF.repo_id_from_folder("google--gemma-4-12b-it") == "google/gemma-4-12b-it"
    # another pack's bare folder, and a hidden one, are not ours to read
    assert LF.repo_id_from_folder("Qwen3.5-4B") is None
    assert LF.repo_id_from_folder(".cache") is None
    assert LF.repo_id_from_folder("--x") is None


def test_the_roots_are_hermetic_under_test_mode(monkeypatch, tmp_path):
    monkeypatch.delenv("OTR_LLM_DIR", raising=False)
    assert LF.llm_roots() == []
    assert LF.download_destination(REPO_ID) is None
    monkeypatch.setenv("OTR_LLM_DIR", str(tmp_path))
    assert LF.llm_roots() == [tmp_path]
    assert LF.download_destination(REPO_ID) == tmp_path / FOLDER


# --------------------------------------------------------------------------- #
# completeness
# --------------------------------------------------------------------------- #


def test_a_receipted_folder_is_complete_and_a_shrunk_file_is_not(llm_root):
    folder = _plain(llm_root)
    assert LF.plain_folder_complete(folder)
    (folder / SHARDS[1]).write_bytes(b"w" * 10)          # size no longer matches
    assert not LF.plain_folder_complete(folder)


def test_a_receipt_names_every_file_but_not_the_transfer_cache(llm_root):
    folder = _plain(llm_root, receipt=False)
    (folder / ".cache" / "huggingface" / "download").mkdir(parents=True)
    (folder / ".cache" / "huggingface" / "download" / "x.incomplete").write_bytes(b"?")
    assert LF.write_receipt(folder, REPO_ID)
    files = json.loads((folder / LF.RECEIPT_NAME).read_text(encoding="utf-8"))["files"]
    assert set(files) == set(SHARDS) | {"model.safetensors.index.json", "config.json"}


def test_first_shard_present_index_absent_is_not_complete(llm_root):
    """THE CASE FROM THE QA: huggingface_hub downloads files concurrently, so a
    shard can land before its index. Any-nonzero-weight would call this done."""
    folder = llm_root / FOLDER
    folder.mkdir()
    (folder / SHARDS[0]).write_bytes(b"w" * 64)
    assert not LF.plain_folder_complete(folder)


def test_index_present_with_a_missing_shard_is_not_complete(llm_root):
    folder = _plain(llm_root, receipt=False)
    (folder / SHARDS[1]).unlink()
    assert not LF.plain_folder_complete(folder)


def test_a_hand_placed_folder_with_a_full_index_is_complete(llm_root):
    assert LF.plain_folder_complete(_plain(llm_root, receipt=False))


def test_scan_ignores_foreign_folders_and_first_root_wins(tmp_path, monkeypatch):
    a, b = tmp_path / "a", tmp_path / "b"
    _plain(a); _plain(b)
    (a / "Qwen3.5-4B").mkdir()                       # bare name: another pack's
    found = LF.scan_plain_models([a, b])
    assert [(r, c) for r, _p, c in found] == [(REPO_ID, True)]
    assert found[0][1] == a / FOLDER


# --------------------------------------------------------------------------- #
# the merge
# --------------------------------------------------------------------------- #


def test_a_complete_folder_wins_over_the_hub(tmp_path, llm_root):
    hub = tmp_path / "hub"; _hub(hub); folder = _plain(llm_root)
    (r,) = cat.scan_local_llm_cache(hub_root=hub)
    assert r.on_disk and r.snapshot_path == str(folder)
    assert r.advertised_context == 32768                 # read from the folder, not the hub


def test_a_complete_hub_copy_wins_over_a_half_downloaded_folder(tmp_path, llm_root):
    hub = tmp_path / "hub"; snap = _hub(hub)
    folder = llm_root / FOLDER; folder.mkdir()
    (folder / SHARDS[0]).write_bytes(b"w" * 64)          # interrupted new download
    (r,) = cat.scan_local_llm_cache(hub_root=hub)
    assert r.on_disk and r.snapshot_path == str(snap)


def test_a_partial_folder_with_no_hub_copy_is_reported_not_on_disk(tmp_path, llm_root):
    folder = llm_root / FOLDER; folder.mkdir()
    (folder / SHARDS[0]).write_bytes(b"w" * 64)
    (r,) = cat.scan_local_llm_cache(hub_root=tmp_path / "empty-hub")
    assert not r.on_disk and r.snapshot_path == str(folder)


def test_the_hub_scan_is_unchanged_when_no_folder_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("OTR_LLM_DIR", raising=False)
    hub = tmp_path / "hub"; snap = _hub(hub)
    (r,) = cat.scan_local_llm_cache(hub_root=hub)
    assert r.on_disk and r.snapshot_path == str(snap)


# --------------------------------------------------------------------------- #
# the download
# --------------------------------------------------------------------------- #


def _download_env(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setattr(cat, "estimate_model_size_gb", lambda *a, **k: 1.0)
    seen = {}

    def _free(path):
        seen["disk_checked_at"] = Path(path)
        return 10 ** 12

    monkeypatch.setattr(cat, "_free_disk_bytes_for", _free)
    return seen


def test_a_new_download_goes_to_the_folder_with_a_receipt_after(tmp_path, llm_root, monkeypatch):
    seen = _download_env(monkeypatch, tmp_path)
    hub = tmp_path / "hub"; hub.mkdir()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        dest = Path(kwargs["local_dir"])
        assert not (dest / LF.RECEIPT_NAME).exists(), "receipt must not exist before the download returns"
        _plain(llm_root, receipt=False)
        return str(dest)

    out = cat.auto_download_if_missing(REPO_ID, hub_root=hub, _snapshot_download=fake_download)
    (kwargs,) = calls
    assert kwargs["local_dir"] == str(llm_root / FOLDER) and "cache_dir" not in kwargs
    assert out == str(llm_root / FOLDER)
    assert (llm_root / FOLDER / LF.RECEIPT_NAME).is_file()
    assert seen["disk_checked_at"] == llm_root / FOLDER  # the drive the bytes land on
    assert LF.plain_folder_complete(llm_root / FOLDER)


def test_a_model_complete_in_the_folder_is_never_downloaded_again(tmp_path, llm_root, monkeypatch):
    _download_env(monkeypatch, tmp_path)
    folder = _plain(llm_root)

    def never(**kwargs):
        raise AssertionError("download must not run")

    assert cat.auto_download_if_missing(REPO_ID, hub_root=tmp_path / "hub", _snapshot_download=never) == str(folder)


def test_a_model_complete_in_the_hub_is_never_downloaded_again(tmp_path, llm_root, monkeypatch):
    """THE LOAD-BEARING ONE: an existing install keeps its hub copy and pays
    nothing. The folder root exists and is empty -- the case every install
    that upgrades will be in."""
    _download_env(monkeypatch, tmp_path)
    hub = tmp_path / "hub"; snap = _hub(hub)

    def never(**kwargs):
        raise AssertionError("download must not run")

    assert cat.auto_download_if_missing(REPO_ID, hub_root=hub, _snapshot_download=never) == str(snap)
    assert not (llm_root / FOLDER).exists()


def test_with_no_folder_root_the_hub_cache_is_still_the_destination(tmp_path, monkeypatch):
    monkeypatch.delenv("OTR_LLM_DIR", raising=False)
    _download_env(monkeypatch, tmp_path)
    hub = tmp_path / "hub"; hub.mkdir()
    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs); _hub(hub); return "x"

    cat.auto_download_if_missing(REPO_ID, hub_root=hub, _snapshot_download=fake_download)
    (kwargs,) = calls
    assert kwargs["cache_dir"] == str(hub) and "local_dir" not in kwargs


# --------------------------------------------------------------------------- #
# the loader's resolvers
# --------------------------------------------------------------------------- #


def test_the_resolver_prefers_the_folder_and_still_finds_the_hub(tmp_path, llm_root, monkeypatch):
    home = tmp_path / "hf"; snap = _hub(home / "hub")
    assert hfenv.resolve_snapshot_dir(REPO_ID, hf_home=str(home)) == str(snap)
    folder = _plain(llm_root)
    assert hfenv.resolve_snapshot_dir(REPO_ID, hf_home=str(home)) == str(folder)


def test_metadata_resolves_on_its_own_from_a_folder_that_is_not_the_model(tmp_path, llm_root):
    """Bug Bible 02.16: the template may live where the weights do not."""
    home = tmp_path / "hf"; snap = _hub(home / "hub")
    meta_only = llm_root / FOLDER; meta_only.mkdir()
    (meta_only / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    assert hfenv.resolve_snapshot_dir(REPO_ID, hf_home=str(home)) == str(snap)
    assert hfenv.resolve_snapshot_file(REPO_ID, "chat_template.jinja", hf_home=str(home)) \
        == str(meta_only / "chat_template.jinja")


def test_load_llm_looks_in_the_selected_directory_first():
    from nodes import _otr_model_loader as L
    src = inspect.getsource(L.load_llm)
    own = src.index('Path(snapshot_path) / "chat_template.jinja"')
    assert own < src.index("_OTR_HF.resolve_snapshot_file(")
    assert "ComfyUI's LLM model category" in src


def test_the_pack_registers_the_category_at_load():
    src = (_REPO / "__init__.py").read_text(encoding="utf-8")
    assert "register_llm_category" in src
    assert src.index("register_llm_category") > src.index("run_boot_sweep")
