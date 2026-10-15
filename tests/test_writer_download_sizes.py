"""What a writer download really is (2026-10-14).

The catalog's `approx_safetensors_gb` is the fit basis and is NOT the download
on every row (gemma-4-E2B-it: 6.0 against 10.2 GB measured; mixed units). What
a person is told and what the disk check reserves come from the measured bytes
in `_MEASURED_DOWNLOAD_BYTES`; the Mistral repo's second full copy of its
weights (consolidated.safetensors) is never fetched.
"""
from __future__ import annotations

import fnmatch

from nodes import _otr_model_catalog as C


def test_every_local_and_native_writer_has_a_measured_download():
    for m in C._active_curated_models():
        if getattr(m, "provider", "local") not in ("local", "comfy_native"):
            continue
        if m.repo_id == C.DEFAULT_LLM_NF4:
            continue
        assert m.repo_id in C._MEASURED_DOWNLOAD_BYTES, m.repo_id
        assert C.download_size_text(m.repo_id).startswith("about "), m.repo_id


def test_the_two_multimodal_gemma_rows_report_their_real_download():
    assert C.download_size_text("google/gemma-4-E2B-it") == "about 10.2 GB"
    assert C.download_size_text("google/gemma-4-E4B-it") == "about 16.0 GB"
    # the fit basis is deliberately untouched (fit and admission were proven on it)
    assert C._by_repo_id()["google/gemma-4-E2B-it"].approx_safetensors_gb == 6.0
    assert C._by_repo_id()["google/gemma-4-E4B-it"].approx_safetensors_gb == 9.0


def test_the_disk_check_reserves_the_measured_download():
    gib = C.estimate_model_size_gb("google/gemma-4-E2B-it")
    assert abs(gib - 10_246_621_918 / 2**30) < 1e-9


def test_mistrals_consolidated_copy_is_never_fetched_or_counted():
    assert "*.safetensors" in C.ALLOW_PATTERNS
    assert any(fnmatch.fnmatch("consolidated.safetensors", p) for p in C.IGNORE_PATTERNS)
    assert not any(fnmatch.fnmatch("model-00001-of-00005.safetensors", p)
                   for p in C.IGNORE_PATTERNS)

    class _Sib:
        def __init__(self, name, size):
            self.rfilename, self.size = name, size

    class _Api:
        def model_info(self, repo_id, files_metadata=True):
            return type("I", (), {"siblings": [
                _Sib("consolidated.safetensors", 24_495_604_224),
                _Sib("model-00001-of-00002.safetensors", 3 * 2**30),
                _Sib("model-00002-of-00002.safetensors", 2**30),
            ]})()

    assert C.estimate_model_size_gb("someone/uncurated-mistral-like", _hf_api=_Api()) == 4.0


def test_the_download_call_passes_the_ignore_patterns():
    src = open(C.__file__, encoding="utf-8").read()
    assert '"ignore_patterns": list(IGNORE_PATTERNS),' in src
