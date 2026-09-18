"""Shipping graphs keep the live small Qwen; only 16 GB NVIDIA pins 12B.

The 2026-09-17 Qwen move already landed on the shipping set. This file
stops that split from rotting back to a stale Qwen2.5 / GGUF 2507 id, and
keeps draft cpu_floor on a cloud writer instead of a local 4B it cannot
load.
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
        elif pid.startswith("otr_cloud_"):
            assert creative == "comfy:slot-a", pid
            assert technical == "comfy:slot-b", pid
        else:
            assert creative == SMALL, pid
            assert technical == SMALL, pid


def test_cpu_floor_is_cloud_writer_not_local_qwen():
    """Operator 2026-09-17: floor cannot run the 4B; use a cloud LLM."""
    llm = _profile("cpu_floor")["llm"]
    assert llm["creative_model"] == "comfy:slot-a"
    assert llm["technical_model"] == "comfy:slot-b"
    assert "transformers" not in llm["lane_allowlist"]
    assert "comfy_credits" in llm["lane_allowlist"]
    keys = list((_profile("cpu_floor").get("preflight") or {}).get("required_keys") or [])
    assert "OTR_COMFY_API_KEY" in keys


def test_shipping_variant_widgets_carry_the_live_label():
    """Saved graphs must store the live COMBO label, not a stale bare id."""
    qwen_label = default_llm_option()
    for pid in bv.SHIPPING_SET:
        if pid.startswith("otr_cloud_"):
            continue
        path = os.path.join(_REPO, "workflows", "variants", f"{pid}.json")
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        text = json.dumps(blob)
        if pid.startswith("otr_16gb_"):
            assert BIG in text, pid
            assert SMALL not in text, pid
        else:
            assert qwen_label in text, pid
            assert "Qwen2.5-14B" not in text, pid
            assert "Qwen3-4B-Instruct-2507" not in text, pid
