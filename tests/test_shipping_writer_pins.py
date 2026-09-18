"""Shipping writer pins: small Qwen locally, 12B on 16 GB NVIDIA, cloud on CPU.

Stops a stale Qwen2.5 / GGUF 2507 id from returning, and keeps the
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


def test_shipping_writer_split_is_4b_except_16gb_nvidia():
    for pid in bv.SHIPPING_SET:
        llm = _profile(pid)["llm"]
        creative = llm["creative_model"]
        technical = llm["technical_model"]
        if pid.startswith("otr_16gb_"):
            assert creative == BIG, pid
            assert technical == BIG, pid
        elif pid.startswith("otr_cloud_") or pid == "otr_cpu_low":
            assert creative == "comfy:slot-a", pid
            assert technical == "comfy:slot-b", pid
        else:
            assert creative == SMALL, pid
            assert technical == SMALL, pid


def test_shipping_cpu_graph_uses_sonnet_and_luna():
    """The graph people open is otr_cpu_low. Lab leftovers are not this pin."""
    llm = _profile("otr_cpu_low")["llm"]
    assert llm["creative_model"] == "comfy:slot-a"
    assert llm["technical_model"] == "comfy:slot-b"
    assert llm["comfy_slot_a_model"] == "anthropic/claude-sonnet-5"
    assert llm["comfy_slot_b_model"] == "openai/gpt-5.6-luna"
    assert "comfy_credits" in llm["lane_allowlist"]
    keys = list((_profile("otr_cpu_low").get("preflight") or {}).get("required_keys") or [])
    assert "OTR_COMFY_API_KEY" in keys


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
        else:
            assert qwen_label in text, pid
            assert "Qwen2.5-14B" not in text, pid
            assert "Qwen3-4B-Instruct-2507" not in text, pid
