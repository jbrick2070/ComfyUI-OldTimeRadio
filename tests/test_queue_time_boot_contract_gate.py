"""The queue-time boot-contract gate (2026-09-26).

MiniMax H3 declares boot contracts without ``default`` (``--reserve-vram 12
--disable-pinned-memory``, no SageAttention). Once its ~39 GB auto-downloads,
the only thing standing between a stock boot and a 39 GB fetch, a written
script and rendered voices -- then a refusal at the first video beat -- is a
check asked at queue time. These tests pin that it is asked, before the
download, with the fix first.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_visual_assets as va

STOCK = {"available": True, "reserve_vram_gb": None,
         "disable_pinned_memory": False, "cpu": False, "sage_attention": False}
H3_BOOT = {"available": True, "reserve_vram_gb": 12.0,
           "disable_pinned_memory": True, "cpu": False, "sage_attention": False}


def test_h3_on_a_stock_boot_is_refused_with_the_restart_first():
    with pytest.raises(va.VisualAssetError) as info:
        va._refuse_unmet_boot_contracts({"minimax_h3_video"}, state=STOCK)
    msg = str(info.value)
    assert msg.startswith("Restart ComfyUI with --reserve-vram 12 --disable-pinned-memory")
    assert "minimax_h3_video" in msg and "Nothing was downloaded" in msg


def test_h3_on_its_own_boot_passes():
    va._refuse_unmet_boot_contracts({"minimax_h3_video", "minimax_h3_audio_in"},
                                    state=H3_BOOT)


def test_sage_on_an_h3_boot_is_still_refused():
    """Sage turns H3 into noise with no error; the flags alone are not enough."""
    with pytest.raises(va.VisualAssetError, match="SageAttention"):
        va._refuse_unmet_boot_contracts({"minimax_h3_video"},
                                        state=dict(H3_BOOT, sage_attention=True))


def test_an_engine_that_runs_on_a_stock_boot_is_untouched():
    va._refuse_unmet_boot_contracts({"ltx_8gb", "animatediff15_v3_haunted_video"},
                                    state=STOCK)


def test_no_readable_boot_state_skips_the_gate():
    """Outside a ComfyUI server (tests, CLI) the render-time check stands."""
    va._refuse_unmet_boot_contracts({"minimax_h3_video"},
                                    state={"available": False, "error": "no comfy"})


def test_the_gate_runs_before_any_download():
    """Wiring at the real site: asked after the node-pack gate and before the
    native requests that decide what to fetch."""
    src = inspect.getsource(va.ensure_prompt_visual_assets)
    boot = src.index("_refuse_unmet_boot_contracts(plan[\"engines\"])")
    assert src.index("_refuse_missing_node_packs(plan[\"engines\"])") < boot
    assert boot < src.index("native_requests(")
