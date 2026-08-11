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
def test_the_declared_canvas_wins_over_the_landscape_default(engine,
                                                             monkeypatch):
    """The property the old M0 clamp was really protecting, stated directly.

    This used to assert 512x288 -- the single-pass half of an inline,
    RECIPE-DEPENDENT branch in ``build_request_from_shot``. Lane 7 deleted that
    branch: the lane declares 1024x576 once and ``declared_render_canvas`` (the
    LAST write in that function) applies it whatever the recipe. What mattered
    then still matters now, and it is the only thing asserted here: this lane
    does NOT fall through to the 480x832 portrait / 1472x832 landscape
    defaults, which its 22B A2V unet cannot afford.
    """
    from nodes._otr_video_engines.eng_ltx_av import LtxAudioInEngine
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    monkeypatch.delenv("OTR_LTX_AV_RENDER_CANVAS", raising=False)
    req = rd.build_request_from_shot(_shot(engine, "music_visual"), _ledger())
    got = (req["canvas"]["w"], req["canvas"]["h"])
    assert got == tuple(LtxAudioInEngine.render_canvas)
    assert got not in ((480, 832), (1472, 832))


def test_an_env_canvas_that_disagrees_with_the_declaration_is_a_REFUSAL(
        monkeypatch):
    """NOT a quiet re-plan -- the same doctrine ltx_video and ltx_8gb enforce.

    ``OTR_LTX_AV_RENDER_CANVAS`` used to WIN here, and this test asserted that
    it did. It cannot win any more: the declaration is applied last. Handing an
    operator who explicitly asked for 832x480 a 1024x576 render is its own
    small lie, so the adapter refuses by name before any staging or GPU work.
    On this lane the env is not cosmetic either -- ia2v halves the canvas for
    stage A, so a canvas that is not /64 on both axes has no legal stage A.
    """
    from nodes._otr_video_engines.eng_ltx_av import (LtxAudioInEngine,
                                                     assert_env_matches_contract)
    from nodes._otr_video_engines.frame_contract import ContractEnvConflict
    monkeypatch.setenv("OTR_LTX_AV_RENDER_CANVAS", "832x480")
    with pytest.raises(ContractEnvConflict) as excinfo:
        assert_env_matches_contract(LtxAudioInEngine.frame_contract,
                                    LtxAudioInEngine.render_canvas)
    msg = str(excinfo.value)
    assert "OTR_LTX_AV_RENDER_CANVAS" in msg      # names the variable
    assert "1024x576" in msg                      # names the legal value
    assert "NO FALLBACK" in msg
    # and it stays silent when the env AGREES with the declaration
    monkeypatch.setenv("OTR_LTX_AV_RENDER_CANVAS", "1024x576")
    assert_env_matches_contract(LtxAudioInEngine.frame_contract,
                                LtxAudioInEngine.render_canvas)


@pytest.mark.parametrize("raw,needle", [
    ("193", "disagrees"),          # parses, and is not the declaration
    ("not-a-number", "not an integer"),   # cannot be parsed at all
])
def test_a_contract_bearing_env_var_is_checked_RAW_not_after_the_fallback(
        raw, needle, monkeypatch):
    """The hole a reviewer found in the first draft of this check (kibitz r1).

    ``_LTX_AV_MAX_FRAMES`` is parsed at IMPORT by ``_env_num``, which turns a
    malformed value into the declared 497 with a warning. So a check written as
    ``if _LTX_AV_MAX_FRAMES != contract.max_frames`` reports AGREEMENT for a
    variable the operator set and this adapter is ignoring -- the exact silence
    the refusal exists to break.

    Two policies on purpose. ``_env_num`` still keeps a typo from taking the
    import down and deleting the lane from the menu; but a CONTRACT-BEARING
    variable is read RAW here, and both "parses to something else" and "does
    not parse" are refusals, because the planner already partitioned the beat
    against the declaration.
    """
    from nodes._otr_video_engines.eng_ltx_av import (LtxAudioInEngine,
                                                     assert_env_matches_contract)
    from nodes._otr_video_engines.frame_contract import ContractEnvConflict
    monkeypatch.delenv("OTR_LTX_AV_RENDER_CANVAS", raising=False)
    monkeypatch.setenv("OTR_LTX_AV_MAX_FRAMES", raw)
    with pytest.raises(ContractEnvConflict) as excinfo:
        assert_env_matches_contract(LtxAudioInEngine.frame_contract,
                                    LtxAudioInEngine.render_canvas)
    msg = str(excinfo.value)
    assert "OTR_LTX_AV_MAX_FRAMES" in msg
    assert needle in msg
    assert "NO FALLBACK" in msg


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
