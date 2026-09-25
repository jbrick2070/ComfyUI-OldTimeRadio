"""Shipping writer pins: small Qwen locally, 12B on 16 GB NVIDIA, cloud on CPU.

Stops a stale Qwen2.5 id from returning, and keeps the
shipping CPU graph on Sonnet 5 + Luna instead of a local 4B.
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))

from nodes._otr_model_catalog import (  # noqa: E402
    CURATED_LLM_MODELS,
    DEFAULT_LLM,
    default_llm_option,
)
from nodes._otr_shared import capability_profiles as cp  # noqa: E402
import build_variants as bv  # noqa: E402

SMALL = "Qwen/Qwen3.5-4B"
BIG = "google/gemma-4-12b-it"
STALE = (
    "Qwen/Qwen2.5-14B",
    "Qwen/Qwen2.5-14B-Instruct",
    "Qwen/Qwen3-4B-Instruct-2507",
)


def _profile(name: str) -> dict:
    return cp.load_profile(name)


def test_dropdown_default_is_live_small_qwen():
    assert DEFAULT_LLM == SMALL
    assert default_llm_option().startswith(SMALL)
    curated = {m.repo_id for m in CURATED_LLM_MODELS}
    assert SMALL in curated
    for dead in STALE:
        assert dead not in curated, dead


#: The 24/32 GB native tiers pin a BIGGER writer than the 16 GB one, on purpose:
#: a3cf5441 (2026-09-22) shipped them "with the beefier writer baked in". This
#: test encoded a two-way 4B-or-12B policy and had been RED since that commit,
#: unnoticed -- the profile declared an intent the policy test did not know about,
#: which is the drift the profile layer keeps producing. Named here so the next
#: tier that wants its own writer adds a row instead of going quietly red.
BIGGEST = "Qwen/Qwen3.8-27B"


def _shipped_writer_models(pid):
    """The writer models the RENDERED graph carries, in bare-id form.

    Reads the graph rather than the profile, because a matrix row does not carry a
    key it inherits from the canonical -- and because what ships is the question.
    `_strip_label_suffix` turns the stored COMBO label
    ('Qwen/Qwen3.5-4B (8.7 GB download, ...)') back into the bare id; it is
    idempotent, so an already-bare value passes through unchanged.
    """
    from nodes import _otr_workflow_apply as wa
    from nodes._otr_model_catalog import _strip_label_suffix
    import logging

    canonical = json.load(open(
        os.path.join(_REPO, "workflows", "otr_canonical.json"), encoding="utf-8"))
    schemas = wa.build_offline_schemas()
    logging.disable(logging.CRITICAL)
    try:
        graph = wa.apply_profile(canonical, cp.load_profile(pid))
    finally:
        logging.disable(logging.NOTSET)
    names = wa.serialized_slot_names("OTR_LedgerScriptWriter", schemas)
    node = next(n for n in graph["nodes"]
                if n.get("type") == "OTR_LedgerScriptWriter")
    values = node.get("widgets_values") or []
    out = {}
    for widget in ("creative_writing_model", "technical_model"):
        out[widget] = _strip_label_suffix(values[names.index(widget)])
    return out


def test_shipping_writer_split_is_4b_except_16gb_nvidia():
    for pid in bv.SHIPPING_SET:
        shipped = _shipped_writer_models(pid)
        creative = shipped["creative_writing_model"]
        technical = shipped["technical_model"]
        if pid.startswith("otr_16gb_"):
            assert creative == BIG, pid
            assert technical == BIG, pid
        elif pid.startswith(("otr_24gb_", "otr_32gb_")):
            assert creative == BIGGEST, pid
            assert technical == BIGGEST, pid
        elif pid.startswith("otr_cloud_"):
            assert creative == "comfy:slot-a", pid
            assert technical == "comfy:slot-b", pid
        else:
            assert creative == SMALL, pid
            assert technical == SMALL, pid


def test_shipping_variant_widgets_carry_the_live_label():
    """Saved graphs must store the live COMBO label, not a stale bare id."""
    qwen_label = default_llm_option()
    for pid in bv.SHIPPING_SET:
        path = os.path.join(_REPO, "workflows", "variants", f"{pid}.json")
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        text = json.dumps(blob)
        if pid.startswith("otr_cloud_") or pid == "otr_cpu_low":
            assert "comfy:slot-a" in text, pid
            assert "anthropic/claude-sonnet-5" in text, pid
            assert "openai/gpt-5.6-luna" in text, pid
            assert SMALL not in text, pid
        elif pid.startswith("otr_16gb_"):
            assert BIG in text, pid
            assert SMALL not in text, pid
        elif pid.startswith(("otr_24gb_", "otr_32gb_")):
            assert BIGGEST in text, pid
            assert SMALL not in text, pid
        else:
            assert qwen_label in text, pid
            assert "Qwen2.5-14B" not in text, pid
            assert "Qwen3-4B-Instruct-2507" not in text, pid
