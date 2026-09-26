"""Each writer model samples at its maker's own baseline (2026-09-25).

The creativity dial applied ONE preset map to every model. Measured the day it
went: it ran Qwen3 hotter and Gemma 4 cooler than their makers intend, and
Mistral-Nemo at more than double its card's value. The operator's ruling:
"find the canonical temp baseline for each model and that's it", with "null"
for cloud slots (the provider applies the model's own default). Every test here
calls the real function.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as catalog
from nodes import OTR_LedgerScriptWriter as writer
from nodes._otr_line_composer import LineRequest, compose_line_draft


def test_every_curated_local_model_resolves_and_cloud_slots_send_nothing():
    for row in catalog._by_repo_id().values():
        baseline = catalog.sampling_baseline(row.repo_id)
        if row.provider == "local":
            assert baseline is not None, row.repo_id
            temperature, top_p, top_k = baseline
            assert 0.0 < temperature <= 1.0, (row.repo_id, temperature)
        else:
            assert baseline is None, row.repo_id


def test_a_quant_twin_resolves_to_its_base_models_baseline():
    """Quants are separate catalog rows; the baseline comes from the base
    model through the canonical id, never a per-row copy that could drift."""
    assert (catalog.sampling_baseline(catalog.DEFAULT_LLM_NF4)
            == catalog.sampling_baseline(catalog.DEFAULT_LLM))


def test_the_default_writer_uses_its_model_card_not_the_old_preset():
    assert catalog.sampling_baseline(catalog.DEFAULT_LLM) == (0.7, 0.8, 20)


def test_a_model_that_publishes_nothing_still_samples():
    """Greedy decoding flattens dialogue, so an unlisted local model falls back
    to the old balanced preset instead of sending nothing."""
    assert (catalog.sampling_baseline("someone/uncurated-local-model")
            == catalog.SAMPLING_FALLBACK)
    assert catalog.SAMPLING_FALLBACK[0] > 0


def test_each_slot_gets_its_own_models_numbers():
    scheduler = writer._SlotScheduler(
        creative_id="google/gemma-4-12b-it",
        technical_id=catalog.DEFAULT_LLM,
        min_p=0.05, repetition_penalty=1.03)
    creative = scheduler.sampling_for("creative")
    technical = scheduler.sampling_for("technical")
    assert (creative["top_p"], creative["top_k"]) == (0.95, 64)
    assert (technical["top_p"], technical["top_k"]) == (0.8, 20)
    assert creative["min_p"] == technical["min_p"] == 0.05


def test_a_cloud_slot_sends_no_sampling_keys():
    scheduler = writer._SlotScheduler(
        creative_id="openrouter:slot-a", technical_id=catalog.DEFAULT_LLM,
        min_p=0.0, repetition_penalty=1.0)
    sampling = scheduler.sampling_for("creative")
    assert sampling["top_p"] is None and sampling["top_k"] is None


def _req():
    return LineRequest(
        speaker="ALICE VALE", intent="Answer the signal.", mood="alert",
        canon_header="The relay stands beside a flooded square.", last_lines=[],
        allowed_people=frozenset({"ALICE VALE"}), allowed_things=frozenset())


def _temperatures_seen(base_temperature, attempts=3):
    """Drive the real retry loop: every attempt but the last raises (the path
    that bumps the temperature), and the last returns a line."""
    seen = []

    def creative_fn(messages, *, temperature, max_new_tokens, stop=None):
        seen.append(temperature)
        if len(seen) < attempts:
            raise RuntimeError("transport hiccup")
        return "ALICE VALE: The relay is still warm."

    compose_line_draft(creative_fn=creative_fn, req=_req(),
                       max_attempts=attempts, base_temperature=base_temperature)
    return seen


def test_the_retry_bump_never_crosses_the_collapse_line():
    """Gemma samples at 1.0; +0.1 per retry would reach 1.1 and 1.2, past the
    line where scripts collapsed (BUG-014)."""
    assert _temperatures_seen(1.0) == [1.0, 1.0, 1.0]


def test_a_cooler_model_still_warms_on_retry_up_to_one():
    assert _temperatures_seen(0.7) == pytest.approx([0.7, 0.8, 0.9])
    assert max(_temperatures_seen(0.95)) == pytest.approx(1.0)


def test_a_cloud_slot_sends_no_temperature_on_any_attempt():
    assert _temperatures_seen(None) == [None, None, None]
