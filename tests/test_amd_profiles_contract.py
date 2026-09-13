"""AMD presets must stay on the unquantised, portable still-image route."""
from pathlib import Path
import json

import pytest

from nodes._otr_shared import capability_profiles as cp

ROOT = Path(__file__).resolve().parents[1]
AMD_PROFILES = ("otr_amd_still", "otr_amd16_rocm", "otr_amd8_rocm")


@pytest.mark.parametrize("profile_id", AMD_PROFILES)
def test_amd_profiles_load_without_cuda_only_writer_requirements(profile_id):
    profile = cp.load_profile(profile_id)
    assert profile["status"] == "draft"  # No hardware receipt yet.
    assert profile["platform"] == "any"  # Windows and Linux.
    assert profile["gpu_vendor"] == "amd"
    assert profile["device_backend"] == "cuda"  # PyTorch's ROCm API.
    assert profile["allow_sidecars"] is False
    llm = profile["llm"]
    assert llm["quant_policy"] == "none"
    assert llm["attn_impl"] == "sdpa"
    assert llm["creative_model"] == llm["technical_model"] == "Qwen/Qwen3.5-4B"
    assert "transformers" in llm["lane_allowlist"]
    assert profile["launch"]["sage_attention"] is False
    assert profile["image"]["dtype_policy"] == "no_fp8"
    assert profile["video"]["dtype_policy"] == "no_fp8"


@pytest.mark.parametrize("profile_id", AMD_PROFILES)
def test_amd_lab_engines_match_the_shipped_still_route(profile_id):
    shipped = cp.load_profile("otr_amd_still")
    profile = cp.load_profile(profile_id)
    assert profile["role_overrides"] == shipped["role_overrides"]
    assert profile["slot_overrides"] == shipped["slot_overrides"]


def test_amd_memory_tiers_remain_distinct():
    small = cp.load_profile("otr_amd8_rocm")
    full = cp.load_profile("otr_amd16_rocm")
    assert small["llm"]["vram_ceiling_gb"] == 6.8
    assert full["llm"]["vram_ceiling_gb"] == 14.5
    assert small["render"]["frame_budget"] == 17
    assert full["render"]["frame_budget"] == 25


def test_shipped_graph_writer_uses_the_same_unquantised_policy():
    graph = json.loads((ROOT / "workflows/variants/otr_amd_still.json").read_text())
    writer = next(node for node in graph["nodes"]
                  if node["type"] == "OTR_LedgerScriptWriter")
    values = writer["widgets_values"]
    assert all(str(values[index]).startswith("Qwen/Qwen3.5-4B (")
               for index in (2, 3))
    assert values[28] == "none"
    assert "bnb_nf4" not in values
    assert "bnb_8bit" not in values
