"""`_plan_max_memory`'s size tags must be BARE TOKENS, not substrings.

PBUG-20260829-07. `"2b-it" in "google/gemma-4-12b-it"` is True -- "12b-it"
ends in "2b-it" -- so the 12B model was handed the 2-BILLION budget on every
card under 12 GiB and spilled to CPU, where bitsandbytes then refused the
4-bit dispatch. Four live legs on an 8 GB 4060 died on it.

Invisible above 12 GiB because that branch returns first, which is why the
dev box never saw it. These tests pin BOTH sides: the collision cannot come
back, and the >=12 GiB path stays byte-identical.
"""
from __future__ import annotations

import pytest

from nodes._otr_model_loader import _plan_max_memory


def plan(model_id, vram):
    return _plan_max_memory(model_id, vram,
                            cuda_available=True, quant_policy="bnb_nf4")


@pytest.mark.parametrize("model_id", [
    "google/gemma-4-12b-it",
    "GOOGLE/GEMMA-4-12B-IT",          # case must not matter
    "unsloth/gemma-4-12b-it-GGUF",
])
def test_a_12b_id_is_not_mistaken_for_a_2b_id(model_id):
    """The whole bug in one assertion: 12b must NOT get the 2B budget."""
    assert plan(model_id, 8.0) == {0: "6.8GiB", "cpu": "32GiB"}


@pytest.mark.parametrize("model_id", [
    "google/gemma-4-2b-it",
    "google/gemma-2-2b-it",
    "google/gemma-4-E2B-it",
])
def test_a_real_2b_id_still_gets_the_2b_budget(model_id):
    """The fix must not overshoot: genuine 2B models keep 3.2GiB."""
    assert plan(model_id, 8.0) == {0: "3.2GiB", "cpu": "32GiB"}


def test_e4b_is_unaffected():
    assert plan("google/gemma-4-E4B-it", 8.0) == {0: "6.8GiB", "cpu": "32GiB"}


@pytest.mark.parametrize("vram,expected", [
    (12.0, "9.5GiB"),
    (15.99, "13.5GiB"),
])
def test_the_big_card_path_is_untouched(vram, expected):
    """>=12 GiB returns before any tag is read -- the 5080's behaviour must
    not move by a single byte as a side effect of the tag fix."""
    assert plan("google/gemma-4-12b-it", vram) == {0: expected, "cpu": "32GiB"}


def test_an_unquantized_request_still_returns_none():
    """PBUG-20260825-03's half stays fixed: no cap without bnb."""
    assert _plan_max_memory("google/gemma-4-12b-it", 8.0,
                            cuda_available=True, quant_policy="none") is None


def test_no_cuda_still_returns_none():
    assert _plan_max_memory("google/gemma-4-12b-it", 8.0,
                            cuda_available=False, quant_policy="bnb_nf4") is None
