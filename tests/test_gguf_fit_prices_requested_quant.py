"""The VRAM gate must price the quant and context the REQUEST asked for.

PBUG-20260829-08. `_estimate_resident_gb` priced a gguf row from
`approx_artifact_gb()` -- the FIRST pinned artifact, Q8_0 on the gemma row --
and added KV at `row.context_window`, the row's MAXIMUM. Every caller therefore
got 11.8 + 5.6 = 17.4 GB no matter what it requested, so a profile asking for
Q4_K_M at n_ctx 2048 was judged on a Q8_0 load at 8192 and REFUSED with
"pick a smaller model" -- when it had already picked the smaller quant.

Seven profiles could not load their own configured writer, including every
Mac/AMD/8GB one. Invisible on the 16 GB dev box, where the same number lands
WARN instead of FAIL.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as cat
from nodes._otr_gguf_backend import gguf_row_for_repo

GEMMA_GGUF = "unsloth/gemma-4-12b-it-GGUF"


def test_a_small_quant_is_not_priced_as_the_big_one():
    small = cat.check_vram_fit(GEMMA_GGUF, 2048, ceiling_gb=6.8, gguf_quant="Q4_K_M")
    big = cat.check_vram_fit(GEMMA_GGUF, 2048, ceiling_gb=6.8, gguf_quant="Q8_0")
    assert small.estimated_gb < big.estimated_gb, (
        "Q4_K_M priced the same as Q8_0 (%.2f vs %.2f) -- the request's quant "
        "is being ignored again" % (small.estimated_gb, big.estimated_gb))


def test_the_8gb_profiles_own_writer_is_not_refused():
    """The exact request 8gb_lite / otr_8gb_* / otr_amd8_rocm make."""
    v = cat.check_vram_fit(GEMMA_GGUF, 2048, ceiling_gb=6.8, gguf_quant="Q4_K_M")
    assert v.tier != "FAIL", (
        "the 8GB profiles' configured writer is refused at %.2f GB -- they "
        "cannot load the model they ship with" % v.estimated_gb)


def test_mac_and_12gb_profiles_pass():
    for ceiling in (10.0, 10.5):
        v = cat.check_vram_fit(GEMMA_GGUF, 4096, ceiling_gb=ceiling,
                               gguf_quant="Q4_K_M")
        assert v.tier != "FAIL", (
            "ceiling %.1f refused at %.2f GB" % (ceiling, v.estimated_gb))


def test_kv_scales_with_the_requested_context_not_the_row_maximum():
    row = gguf_row_for_repo(GEMMA_GGUF)
    assert row.kv_gb_per_1k, "row has no measured KV cost; test is vacuous"
    small = cat.check_vram_fit(GEMMA_GGUF, 2048, ceiling_gb=99, gguf_quant="Q4_K_M")
    large = cat.check_vram_fit(GEMMA_GGUF, 8192, ceiling_gb=99, gguf_quant="Q4_K_M")
    expected = (8192 - 2048) / 1024.0 * row.kv_gb_per_1k
    assert large.estimated_gb - small.estimated_gb == pytest.approx(expected, rel=1e-6), (
        "KV did not scale with the requested context: %.3f -> %.3f"
        % (small.estimated_gb, large.estimated_gb))


def test_an_oversize_pick_is_still_refused():
    """The gate must not become a rubber stamp: 70B-on-8GB still FAILs."""
    v = cat.check_vram_fit("meta-llama/Llama-3.1-70B-Instruct", 8192,
                           ceiling_gb=6.8, safetensors_gb_hint=140.0)
    assert v.tier == "FAIL", "the oversize guard stopped guarding"


def test_an_unknown_quant_falls_back_rather_than_crashing():
    v = cat.check_vram_fit(GEMMA_GGUF, 2048, ceiling_gb=6.8, gguf_quant="Q2_NOPE")
    assert v.estimated_gb and v.estimated_gb > 0
