"""The queue-time boot-contract gate (2026-09-26).

MiniMax H3 declares boot contracts without ``default`` (SageAttention off;
since 2026-09-26 nothing else -- the operator ruled out artificial reserves).
Once its ~39 GB auto-downloads, the only thing standing between a Sage boot
and a 39 GB fetch, a written script and rendered voices -- then a refusal at
the first video beat -- is a check asked at queue time. These tests pin that it is asked, before the
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


def test_h3_runs_on_a_stock_sage_free_boot():
    """Operator 2026-09-26: no artificial reserve. H3's contract asks only for
    SageAttention off, so a stock boot passes; an OOM there is recorded."""
    va._refuse_unmet_boot_contracts({"minimax_h3_video"}, state=STOCK)


def test_sage_alone_is_refused_with_the_sage_fix_first():
    with pytest.raises(va.VisualAssetError) as info:
        va._refuse_unmet_boot_contracts(
            {"minimax_h3_video"}, state=dict(STOCK, sage_attention=True))
    msg = str(info.value)
    assert msg.startswith("Start ComfyUI without SageAttention")
    assert "--reserve-vram" not in msg and "Nothing was downloaded" in msg


def test_an_unreadable_sage_state_is_not_blamed_on_sage():
    """A failed Sage probe must not tell a user whose Sage is already off to
    turn it off: the lead says the state could not be confirmed, and the
    probe's own reason follows (Sonnet review of 84c98900)."""
    state = dict(STOCK, sage_attention=None, sage_probe_error="probe raised")
    with pytest.raises(va.VisualAssetError) as info:
        va._refuse_unmet_boot_contracts({"minimax_h3_video"}, state=state)
    msg = str(info.value)
    assert msg.startswith("Could not confirm ComfyUI is running without SageAttention")
    assert "probe raised" in msg


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


def test_an_unknown_contract_name_refuses_by_name(monkeypatch):
    """An engine declaring a contract the table does not know is refused with
    the gate's own message, not a stray BootContractError."""
    from nodes._otr_video_engines import registry as vreg
    real = vreg.get_engine("minimax_h3_video")

    class _Odd:
        compatible_boot_contracts = ("no_such_contract",)
        name = real.name

    monkeypatch.setattr(vreg, "get_engine", lambda eid: _Odd())
    with pytest.raises(va.VisualAssetError, match="no boot contract this pack can check"):
        va._refuse_unmet_boot_contracts({"minimax_h3_video"}, state=STOCK)


def test_an_unknown_name_beside_a_known_one_is_skipped_like_render_time(monkeypatch):
    """Cursor, 78ab8d74: ("no_such_contract", "h3_8gb_lab") must pass a boot
    that satisfies h3_8gb_lab, as the render path's identification would."""
    from nodes._otr_video_engines import registry as vreg

    class _Mixed:
        compatible_boot_contracts = ("no_such_contract", "h3_8gb_lab")

    monkeypatch.setattr(vreg, "get_engine", lambda eid: _Mixed())
    lab = {"available": True, "reserve_vram_gb": None,
           "disable_pinned_memory": True, "sage_attention": False, "cpu": False}
    va._refuse_unmet_boot_contracts({"minimax_h3_video"}, state=lab)
