"""The admission gate must price the REQUEST, never the row's declared maximum.

One defect family, found three times: PBUG-20260829-08 priced a row's maximum
QUANT, -17 priced its maximum CONTEXT in the dropdown badge, and -20 priced its
maximum context in the gate itself. A row's declared maximum is never the right
number to judge a specific request by -- in the worst instance the gate refused
a load against a card that MEASURED 7,751 MiB running exactly that
configuration, 48 of 48 layers resident, generating coherent text.

The rows those three were measured on were retired with their writer backend,
and the quant and context knobs went with them. What remains is the half of the
lesson that still has teeth on every surviving row: a model advertising a
262,144-token window must not be priced as though the pipeline allocates a KV
cache that size. That is the same error wearing different clothes, and it is
what the first test here pins.
"""
from __future__ import annotations

import types

from nodes import _otr_model_catalog as cat
from nodes._otr_model_loader import _assert_policy_admits_vram


def _ctx(value=8192, tier="UNKNOWN"):
    return types.SimpleNamespace(value=value, tier=tier)


def test_native_advertised_window_does_not_price_unallocated_kv():
    """A huge advertised context must not inflate the estimate.

    Qwen3.5-4B advertises 262,144 tokens. The pipeline does not allocate a KV
    cache at that size, so the resident estimate must not move with it -- and
    the model must stay admissible at both ends of the range rather than being
    refused for a window nobody asked to fill.
    """
    from nodes._otr_shared.llm_policy import LLMRuntimePolicy

    policy = LLMRuntimePolicy()
    small = cat.check_vram_fit("Qwen/Qwen3.5-4B", 8192, ceiling_gb=14.5)
    large = cat.check_vram_fit("Qwen/Qwen3.5-4B", 262144, ceiling_gb=14.5)
    assert small.estimated_gb == large.estimated_gb, (
        "the estimate moved with the advertised window (%s vs %s); a context "
        "the pipeline never allocates must not be priced"
        % (small.estimated_gb, large.estimated_gb))
    for capacity in (8192, 262144):
        _assert_policy_admits_vram("Qwen/Qwen3.5-4B", _ctx(capacity), policy)


def test_a_policy_like_caller_without_the_full_policy_shape_is_admitted():
    """The gate reads the ceiling off whatever policy-like object arrives.

    Callers outside the writer hand it a plain namespace rather than a real
    LLMRuntimePolicy. The gate must price and admit from that, not require the
    full dataclass -- this is the caller shape that regressed when policy
    fields were last added and removed.
    """
    pol = types.SimpleNamespace(vram_ceiling_gb=14.5)
    _assert_policy_admits_vram("google/gemma-4-E2B-it", _ctx(8192), pol)


def test_the_oversize_guard_still_bites():
    """The simplification must not turn the gate into a rubber stamp.

    Both halves of the guard are checked, because they reach the estimate by
    different routes: a CURATED row priced from its own catalog size, and an
    uncurated one priced from a caller-supplied size hint. UNKNOWN is the
    documented verdict for an uncurated model with no hint, so a model picked
    only for being famously large would prove nothing here -- it would return
    UNKNOWN and pass a weaker assertion than this file intends.

    The gate itself logs FAIL as a recommendation and lets the runtime be the
    authority, which is why this asserts on the verdict rather than expecting
    an exception.
    """
    curated = cat.check_vram_fit("Qwen/Qwen3.5-4B", 8192, ceiling_gb=1.0)
    assert curated.tier == "FAIL", (
        "a curated 4B row against a 1.0 GB ceiling must price as FAIL, got %s "
        "at %s GB" % (curated.tier, curated.estimated_gb))

    hinted = cat.check_vram_fit(
        "meta-llama/Meta-Llama-3-70B-Instruct", 8192,
        ceiling_gb=6.8, safetensors_gb_hint=140.0)
    assert hinted.tier == "FAIL", (
        "an uncurated row with a 140 GB size hint must price as FAIL, got %s "
        "at %s GB" % (hinted.tier, hinted.estimated_gb))
