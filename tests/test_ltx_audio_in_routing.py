"""Role-driven routing for the unified ltx_audio_in lane (CPU; no torch/forward).

Operator 2026-06-26 + roundtable R1/R2 (docs/2026-06-26-ltx-consolidation): one
audio-in LTX engine serves BOTH scene/bookend beats AND character beats, so the
render driver must route the STILL, the AUDIO, and the PROMPT by the beat ROLE --
not by the engine family. These lock that:

* ltx_audio_in conditions on the beat's WIDE scene still (the scene_open
  radio-bookend still for announcer/music; scene_character for character beats),
  EXACTLY like still_pan/still_flat -- never the vertical portrait, no init='' gap.
* a CHARACTER ltx_audio_in beat is classified character-face -> clean own-voice
  audio (no ambient master slice) + a character prompt (not the scene prompt).
* announcer/music BOOKEND ltx_audio_in beats stay scene (ambient slice + scene
  prompt), and clamp to the VRAM-safe 512x288 AV canvas.

Other engines (ltx_video / wan_i2v / still_pan / humo) are intentionally
UNCHANGED -- these assert that too, so the additive role wiring never drifts them.
"""
import pytest

from nodes._otr_video_engines import render_driver as rd


def _ledger(images=None):
    return {"meta": {}, "lines": [], "images": {"images": images or []},
            "video": {"shots": []}}


def _shot(engine_id, role, **over):
    s = {"shot_id": "shot_b001", "engine_id": engine_id, "role": role,
         "target_frame_count": 120, "render_request_hash": "deadbeef",
         "creative": {"text_prompt": "a vintage radio console, soft glow"}}
    s.update(over)
    return s


# ---- the pure ROLE classifier ------------------------------------------------

def test_classifier_bookend_roles_are_never_character_face():
    assert rd._is_character_face_beat(_shot("ltx_audio_in", "announcer_visual")) is False
    assert rd._is_character_face_beat(_shot("ltx_audio_in", "music_visual")) is False


def test_classifier_ltx_audio_in_character_beat_is_face():
    assert rd._is_character_face_beat(_shot("ltx_audio_in", "character_video")) is True


def test_classifier_humo_character_is_face_via_family():
    assert rd._is_character_face_beat(_shot("humo", "character_video")) is True


def test_classifier_does_not_capture_other_engines():
    # ltx_video / wan_i2v / still_pan are NEVER character-face here -> their
    # audio + prompt routing is untouched by the role wiring.
    assert rd._is_character_face_beat(_shot("ltx_video", "character_video")) is False
    assert rd._is_character_face_beat(_shot("wan_i2v", "character_video")) is False
    assert rd._is_character_face_beat(_shot("still_pan", "scene_broll")) is False


# ---- the ambient-master-audio gate ------------------------------------------

def test_ambient_audio_bookend_uses_master_slice():
    # an announcer/music ltx_audio_in beat (not character-face) keeps the ambient
    # master slice it needs when a beat lacks per-line timing.
    assert rd._uses_ambient_master_audio("ltx_audio_in", "audio_conditioned_video",
                                         False) is True


def test_ambient_audio_character_beat_excluded():
    # a character-face beat must use the character's OWN voice -> NO ambient slice.
    assert rd._uses_ambient_master_audio("ltx_audio_in", "audio_conditioned_video",
                                         True) is False


def test_ambient_audio_two_arg_backcompat():
    # the new is_char_face arg defaults False -> the legacy 2-arg call is unchanged.
    assert rd._uses_ambient_master_audio("ltx_audio_in", "audio_conditioned_video") is True
    assert rd._uses_ambient_master_audio("humo", "audio_driven_face") is False


# ---- the STILL routing (the actual bug fix) ----------------------------------

def test_ltx_audio_in_conditions_on_the_bookend_scene_still():
    # the music bookend's scene_open still is in the ledger (minted by the
    # dispatcher); ltx_audio_in must condition on it (init_source=scene_still),
    # NOT leave init_image='' (the pre-fix crash: "requires init_image (got '')").
    still = "C:/x/otr/episodes/ep/b000_scene_open.png"
    led = _ledger([{"kind": "scene_open", "beat_id": "b000_music_open",
                    "path": still}])
    shot = _shot("ltx_audio_in", "music_visual", shot_id="shot_b000_music_open",
                 creative={})
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"]["init_image"] == still


def test_ltx_audio_in_character_beat_uses_wide_scene_character_still():
    # a CHARACTER ltx_audio_in beat conditions on the WIDE scene_character still
    # (render_aspect=wide), never the vertical portrait (the pillarbox bug).
    still = "C:/x/otr/episodes/ep/b003_scene_character.png"
    led = _ledger([{"kind": "scene_character", "beat_id": "b003",
                    "path": still}])
    shot = _shot("ltx_audio_in", "character_video", shot_id="shot_b003",
                 creative={})
    req = rd.build_request_from_shot(shot, led)
    assert req["asset_refs"]["init_image"] == still


# ---- canvas clamp + scene-prompt join ----------------------------------------

def test_ltx_audio_in_clamps_to_vram_safe_av_canvas(monkeypatch):
    # 512x288 is the SINGLE-PASS recipes' M0-safe clamp; under ia2v_canonical
    # the default is the canonical-native 1280x720 (locked in
    # test_ltx_av_ia2v_canonical.py::test_ia2v_render_canvas_defaults_...).
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    monkeypatch.delenv("OTR_LTX_AV_RENDER_CANVAS", raising=False)
    req = rd.build_request_from_shot(
        _shot("ltx_audio_in", "music_visual"), _ledger())
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (512, 288)


def test_ltx_audio_in_music_open_gets_a_composed_prompt():
    shot = {"shot_id": "shot_b000_music_open", "engine_id": "ltx_audio_in",
            "role": "music_visual", "target_frame_count": 97,
            "render_request_hash": "cafe", "source_line_ids": []}
    req = rd.build_request_from_shot(shot, _ledger())
    assert req["text_prompt"], "ltx_audio_in open shot must get a composed prompt"


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
