"""The admission gate must price the context the LOADER will open.

PBUG-20260829-20. `_assert_policy_admits_vram` passed `ctx_verdict.value` --
the ROW's context_window, 8192 for the gemma GGUF row -- while the loader opens
the context the POLICY requests. A profile asking for n_ctx 4096 was judged at
8192:

    gemma-4-12b-it-GGUF Q4_K_M @ 4096  ->  9.43 GB  WARN  (admitted)
    gemma-4-12b-it-GGUF Q4_K_M @ 8192  -> 12.23 GB  FAIL  (refused)

and it was refused against an 8 GB card that MEASURED 7,751 MiB running exactly
that configuration, 48/48 layers resident, generating coherent text. The gate
refused a load the hardware performs.

Third instance of one family: -08 priced the row's max QUANT, -17 priced the
row's max CONTEXT in the dropdown badge, this priced the row's max context in
the gate. A row's declared maximum is never the right number to judge a
specific request by.
"""
from __future__ import annotations

import types

import pytest

from nodes import _otr_model_catalog as cat
from nodes._otr_model_loader import _assert_policy_admits_vram

GEMMA_GGUF = "unsloth/gemma-4-12b-it-GGUF"


def _policy(ceiling=6.8, quant="Q4_K_M", n_ctx=4096):
    return types.SimpleNamespace(vram_ceiling_gb=ceiling, gguf_quant=quant,
                                 gguf_n_ctx=n_ctx)


def _ctx(value=8192, tier="UNKNOWN"):
    return types.SimpleNamespace(value=value, tier=tier)


def test_the_measured_4060_configuration_is_admitted():
    """n_ctx 4096, Q4_K_M, 6.8 ceiling -- measured at 7,751 MiB on real hardware."""
    _assert_policy_admits_vram(GEMMA_GGUF, _ctx(8192), _policy(n_ctx=4096))


def test_the_row_max_context_does_not_override_the_policy():
    """The bug: ctx_verdict carried 8192 and won over the policy's 4096."""
    at_4096 = cat.check_vram_fit(GEMMA_GGUF, 4096, ceiling_gb=6.8, gguf_quant="Q4_K_M")
    at_8192 = cat.check_vram_fit(GEMMA_GGUF, 8192, ceiling_gb=6.8, gguf_quant="Q4_K_M")
    assert at_4096.estimated_gb < at_8192.estimated_gb, "context is not affecting KV at all"
    # the gate must land on the FIRST of those, given a policy asking for 4096
    _assert_policy_admits_vram(GEMMA_GGUF, _ctx(8192), _policy(n_ctx=4096))


def test_a_policy_that_really_asks_for_8192_is_still_priced_at_8192():
    """Removing the over-price must not become an under-price."""
    with pytest.raises(Exception) as ei:
        _assert_policy_admits_vram(GEMMA_GGUF, _ctx(8192), _policy(n_ctx=8192))
    assert "12.2" in str(ei.value) or "12.23" in str(ei.value), (
        "a genuine 8192 request should still be priced at 8192: %s" % ei.value)


def test_a_policy_without_gguf_n_ctx_falls_back_to_the_context_cap():
    """Transformers rows carry no gguf_n_ctx; the old behaviour must survive."""
    pol = types.SimpleNamespace(vram_ceiling_gb=14.5, gguf_quant=None)
    _assert_policy_admits_vram("google/gemma-4-E2B-it", _ctx(8192), pol)


def test_the_oversize_guard_still_bites():
    """The fix must not turn the gate into a rubber stamp."""
    pol = types.SimpleNamespace(vram_ceiling_gb=6.8, gguf_quant="Q8_0", gguf_n_ctx=8192)
    with pytest.raises(Exception):
        _assert_policy_admits_vram(GEMMA_GGUF, _ctx(8192), pol)
