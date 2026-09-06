"""`_plan_max_memory` imposes NO cap. Operator directive 2026-09-06.

WHAT THIS FILE USED TO PIN, and why it changed. It asserted the size-tag
budgets: 3.2GiB for a 2B-tagged id, 6.8GiB for a 9b/12b/e4b/4b-it-tagged id,
and `total_vram - 2.5` above 12 GiB. Those tags were guesses keyed on a
SUBSTRING OF THE MODEL NAME, priced as though every weight quantizes to 4 bits.
Both premises are false, and on a physical 8 GB RTX 4060 all three budgets
produced a partial CPU placement, which is not a saved render -- it is a
useless one, because every forward pass then pays a PCIe round trip per
CPU-resident layer.

Measured from each row's own checkpoint header:

    gemma-4-12b-it   6.95 GiB resident, capped at 6.8  -> missed by 0.15 GiB
                     -> spilled -> 0.4 tok/s
    gemma-4-E2B-it   6.01 GiB resident (5.12 GiB of it NON-quantizable
                     embeddings), capped at 3.2 because its name contains
                     "2b-it" -> never emitted a token
    gemma-4-E4B-it   41 of 42 decoder layers placed on CPU -> 0.5 tok/s

Operator: "remove all caps, just let the system take up as much memory as it
needs" and "no crawling". With no budget the caller sets no device_map either,
so bitsandbytes places the model on ONE device: it fits, or it raises a CUDA
OOM naming the real problem.

PBUG-20260829-07's bare-token matcher is GONE rather than fixed, because the
collision it guarded ("12b-it" ends in "2b-it") can no longer have a
consequence -- there is no tag lookup left to collide in. The old
`test_a_12b_id_is_not_mistaken_for_a_2b_id` is preserved below in spirit: it
now asserts the two ids are treated IDENTICALLY, which is a strictly stronger
statement than "they get different budgets".
"""
from __future__ import annotations

import pytest

from nodes._otr_model_loader import _plan_max_memory


def plan(model_id, vram, quant_policy="bnb_nf4", cuda=True):
    return _plan_max_memory(model_id, vram,
                            cuda_available=cuda, quant_policy=quant_policy)


@pytest.mark.parametrize("model_id", [
    "google/gemma-4-12b-it",
    "GOOGLE/GEMMA-4-12B-IT",
    "unsloth/gemma-4-12b-it-GGUF",
    "google/gemma-4-2b-it",
    "google/gemma-2-2b-it",
    "google/gemma-4-E2B-it",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen3.5-4B",
    "mistralai/Mistral-Nemo-Instruct-2407",
    "someone/an-uncurated-local-model",
    "",
])
@pytest.mark.parametrize("vram", [4.0, 8.0, 11.99, 12.0, 14.5, 15.99, 24.0])
def test_no_model_on_any_card_receives_a_cap(model_id, vram):
    """The whole directive in one assertion."""
    assert plan(model_id, vram) is None


def test_the_12b_and_2b_id_collision_can_no_longer_matter():
    """PBUG-20260829-07 retired. `"2b-it" in "google/gemma-4-12b-it"` is still
    True, but there is no tag lookup left for it to corrupt: both ids now get
    the same answer, which is no budget at all."""
    assert plan("google/gemma-4-12b-it", 8.0) == plan("google/gemma-4-2b-it", 8.0)
    assert plan("google/gemma-4-12b-it", 8.0) is None


def test_no_cpu_offload_lane_is_ever_offered():
    """A `"cpu"` entry in max_memory is what let accelerate spill decoder
    layers. There is no dict to carry one any more, on any card."""
    for vram in (8.0, 16.0):
        assert plan("google/gemma-4-E4B-it", vram) is None


def test_unquantized_and_cuda_less_hosts_are_unchanged():
    """Both historical contracts still hold -- no CUDA-keyed plan without
    CUDA, and no 4-bit-sized cap on an unquantized load -- now satisfied by
    the same unconditional None."""
    assert plan("google/gemma-4-12b-it", 8.0, quant_policy="none") is None
    assert plan("google/gemma-4-12b-it", 8.0, cuda=False) is None
    assert plan("google/gemma-4-12b-it", 8.0, quant_policy="bnb_8bit") is None


def test_the_16gb_card_was_never_actually_capped():
    """Not a behaviour change on the 5080, and the reason is structural: at
    >=14.5 GiB load_llm passes an explicit device_map={"": 0}, and an explicit
    dict device map is used verbatim -- infer_auto_device_map never runs, so
    max_memory was never consulted on that path. This test records the claim;
    the byte-identical proof is the composite-path call diff in the drill log
    (Step 109)."""
    assert plan("google/gemma-4-12b-it", 15.99) is None
