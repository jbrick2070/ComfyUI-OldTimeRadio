"""M3 wiring tests for the LTX-AV lane in render_driver (CPU; no torch/forward).

Locks the additive driver deltas: the ENGINE_FAMILY entry, the VRAM-safe
render-canvas clamp (the lane is M0-proven only at 512x288 -- the 22B A2V would
blow the budget at the 480x832/1472x832 defaults), and the prompt-gate
membership (ltx_audio_in joins the text-engine scene-prompt branch).
NO FALLBACKS (2026-07-02 rip): the SYNTH_FALLBACKS/chain checks are gone with
the machinery.

The two legacy LTX-AV engines (ltx_av_talk / ltx_av_music) were collapsed into
the single ltx_audio_in engine; these tests reference only ltx_audio_in."""
import pytest

from nodes._otr_video_engines import render_driver as rd


def test_engine_family_entries():
    assert rd.ENGINE_FAMILY["ltx_audio_in"] == "audio_conditioned_video"
    assert rd.engine_family("ltx_audio_in") == "audio_conditioned_video"


def test_no_fallback_machinery_in_driver():
    # NO FALLBACKS (2026-07-02): the chain machinery is ripped from the driver;
    # ltx_audio_in (like every engine) fails LOUD.
    for name in ("SYNTH_FALLBACKS", "make_fallback_of", "UNIVERSAL_FLOOR",
                 "FLOOR_NAMES"):
        assert not hasattr(rd, name)


def _ledger():
    return {"meta": {}, "lines": [], "images": [], "video": {"shots": []}}


def _shot(engine_id, role, **over):
    s = {"shot_id": "shot_b001", "engine_id": engine_id, "role": role,
         "target_frame_count": 120, "render_request_hash": "deadbeef",
         "creative": {"text_prompt": "a vintage radio console, soft glow"}}
    s.update(over)
    return s


@pytest.mark.parametrize("engine", ["ltx_audio_in"])
def test_render_canvas_clamped_to_m0_safe(engine, monkeypatch):
    # 512x288 is the SINGLE-PASS M0-safe clamp; the ia2v_canonical default is
    # the canonical-native 1280x720 (locked in test_ltx_av_ia2v_canonical.py).
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    monkeypatch.delenv("OTR_LTX_AV_RENDER_CANVAS", raising=False)
    role = "music_visual"
    req = rd.build_request_from_shot(_shot(engine, role), _ledger())
    # the lane clamps to the M0-proven-safe native canvas (512x288), NOT the
    # 480x832 portrait / 1472x832 landscape defaults
    assert req["canvas"]["w"] == 512
    assert req["canvas"]["h"] == 288


def test_render_canvas_env_override(monkeypatch):
    # single-pass pin (S4c: an ia2v bookend REQUIRES the radio-face still,
    # out of scope for this canvas-env contract).
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    monkeypatch.setenv("OTR_LTX_AV_RENDER_CANVAS", "832x480")
    req = rd.build_request_from_shot(
        _shot("ltx_audio_in", "music_visual"), _ledger())
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (832, 480)


def test_ltx_audio_in_joins_scene_prompt_branch(monkeypatch):
    # an ltx_audio_in OPEN shot with NO creative prompt gets a composed
    # motion/scene prompt (it joined the ltx_video/wan_i2v text-engine branch),
    # never an empty text_prompt. Single-pass pin (S4c face requirement is
    # exercised in test_ltx_av_ia2v_canonical.py).
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    shot = {"shot_id": "shot_b000_music_open", "engine_id": "ltx_audio_in",
            "role": "music_visual", "target_frame_count": 97,
            "render_request_hash": "cafe", "source_line_ids": []}
    req = rd.build_request_from_shot(shot, _ledger())
    assert req["text_prompt"], "ltx_audio_in open shot must get a composed prompt"
