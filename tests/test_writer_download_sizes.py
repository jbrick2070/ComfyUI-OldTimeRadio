"""What a writer download really is (2026-10-14).

The catalog's `approx_safetensors_gb` is the fit basis and is NOT the download
on every row (gemma-4-E2B-it: 6.0 against 10.2 GB measured; mixed units). What
a person is told and what the disk check reserves come from the measured bytes
in `_MEASURED_DOWNLOAD_BYTES`; the Mistral repo's second full copy of its
weights (consolidated.safetensors) is never fetched, and a half-fetched copy of
it left by an older download no longer wedges the folder.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nodes import _otr_model_catalog as C


def test_every_local_and_native_writer_has_a_measured_download():
    for m in C._active_curated_models():
        if getattr(m, "provider", "local") not in ("local", "comfy_native"):
            continue
        assert m.repo_id in C._MEASURED_DOWNLOAD_BYTES, m.repo_id
        assert C.download_size_text(m.repo_id).startswith("about "), m.repo_id


def test_the_native_rows_are_the_bytes_the_pinned_fetch_downloads():
    """One source of truth for the three Comfy-native files: the pinned fetch."""
    from nodes import _otr_visual_assets as VA
    pinned = {fn.rsplit("/", 1)[-1]: size for _cat, _repo, fn, _rev, size, _sha
              in VA._PINNED_SOURCES}
    for repo_id, fname in (
            ("comfy_native:gemma4-e2b-it-int8-convrot", "gemma4_e2b_it_int8_convrot.safetensors"),
            ("comfy_native:gemma4-e4b-it-int8-convrot", "gemma4_e4b_it_int8_convrot.safetensors"),
            ("comfy_native:gemma4-12b-int8-convrot", "gemma4_12b_int8_convrot.safetensors")):
        assert C._MEASURED_DOWNLOAD_BYTES[repo_id] == pinned[fname], repo_id


def test_the_two_multimodal_gemma_rows_report_their_real_download():
    assert C.download_size_text("google/gemma-4-E2B-it") == "about 10.2 GB"
    assert C.download_size_text("google/gemma-4-E4B-it") == "about 16.0 GB"
    # the fit basis is deliberately untouched (fit and admission were proven on it)
    assert C._by_repo_id()["google/gemma-4-E2B-it"].approx_safetensors_gb == 6.0
    assert C._by_repo_id()["google/gemma-4-E4B-it"].approx_safetensors_gb == 9.0


def test_the_retired_qwen_spelling_reads_the_same_download():
    assert C.download_bytes_for(C.DEFAULT_LLM_NF4) == C.download_bytes_for(C.QWEN_LLM)


def test_the_disk_check_reserves_the_measured_download():
    gib = C.estimate_model_size_gb("google/gemma-4-E2B-it")
    assert abs(gib - 10_246_621_918 / 2**30) < 1e-9


class _Sib:
    def __init__(self, name, size):
        self.rfilename, self.size = name, size


def _api(*sibs):
    class _Api:
        def model_info(self, repo_id, files_metadata=True):
            return type("I", (), {"siblings": list(sibs)})()
    return _Api()


def test_the_estimate_counts_exactly_what_the_download_fetches():
    api = _api(_Sib("consolidated.safetensors", 24_495_604_224),
               _Sib("model-00001-of-00002.safetensors", 3 * 2**30),
               _Sib("model-00002-of-00002.safetensors", 2**30),
               _Sib("pytorch_model.bin", 9 * 2**30),           # not in ALLOW_PATTERNS
               _Sib("config.json", 1024))
    assert C.estimate_model_size_gb("someone/uncurated-mistral-like", _hf_api=api) == 4.0
    # huggingface_hub matches the FULL path, case-sensitively: these ARE fetched
    api = _api(_Sib("original/consolidated.safetensors", 2**30),
               _Sib("Consolidated.safetensors", 2**30))
    assert C.estimate_model_size_gb("someone/odd-layout", _hf_api=api) == 2.0


def test_the_download_call_passes_the_ignore_patterns(tmp_path, monkeypatch):
    """The real kwargs reach snapshot_download, not a string in the source."""
    seen = {}

    def fake_snapshot_download(**kwargs):
        seen.update(kwargs)
        raise RuntimeError("stop after the call")

    monkeypatch.setenv("OTR_MODEL_CATALOG_AUTO_DOWNLOAD", "1")
    monkeypatch.setattr(C, "scan_local_llm_cache", lambda **_k: [])   # nothing cached
    monkeypatch.setattr(C, "_free_disk_bytes_for", lambda _p: 10**15)
    monkeypatch.setattr(C._LLM, "download_destination", lambda _w: tmp_path / "m")
    with pytest.raises(RuntimeError, match="stop after the call"):
        C.auto_download_if_missing(
            "mistralai/Mistral-Nemo-Instruct-2407", hub_root=tmp_path,
            _snapshot_download=fake_snapshot_download)
    assert seen["ignore_patterns"] == list(C.IGNORE_PATTERNS)
    assert seen["allow_patterns"] == list(C.ALLOW_PATTERNS)


def test_a_stale_partial_of_the_ignored_copy_is_cleared_and_nothing_else(tmp_path):
    hub_local = pytest.importorskip("huggingface_hub._local_folder")
    folder = tmp_path / "Mistral"
    folder.mkdir()
    stale = hub_local.get_local_download_paths(folder, "consolidated.safetensors") \
        .incomplete_path("abc123")
    kept = hub_local.get_local_download_paths(folder, "model-00001-of-00005.safetensors") \
        .incomplete_path("def456")
    for p in (stale, kept):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"partial")
    removed = C._drop_stale_ignored_partials(folder)
    assert removed == [stale.name]
    assert not Path(stale).exists() and Path(kept).exists()


def test_the_cleanup_creates_nothing_in_a_fresh_folder(tmp_path):
    """A first download must not find its folder already made by the cleanup."""
    fresh = tmp_path / "never_downloaded"
    assert C._drop_stale_ignored_partials(fresh) == []
    assert not fresh.exists()
