"""CPU topology lock for the ia2v_canonical recipe (2026-07-02 transplant).

The canonical comfy.org "LTX-2.3: Image Audio to Video" LIP-SYNC graph,
transplanted into eng_ltx_av behind RECIPE_IA2V (the new DEV-family default;
operator GO after the live isolation smoke proved articulation on OUR radio
still -- docs/2026-07-02-canonical-ia2v/). These tests pin every mechanism
the smoke proved mattered, so a future edit cannot silently regress back to
the frozen-mouth single-pass:

  * TWO-STAGE: base motion pass at HALF canvas -> LTXVLatentUpsampler x2 ->
    audio RE-CONCAT -> 3-step refine;
  * Inplace i2v anchors: 0.7 base (soft -- lets audio actuate the mouth),
    1.0 refine (hard re-anchor);
  * audio latent FROZEN under SetLatentNoiseMask(SolidMask(0));
  * ancestral base sampler, plain refine sampler; front-loaded base ladder,
    0.85->0 refine ladder; CropGuides on the REFINE only;
  * distilled-lora-384 (original, non-1.1) @ 0.5 on the DEV unet;
  * guide chain ImageScale(1.5x) -> ResizeImagesByLongerEdge(1536) ->
    LTXVPreprocess(18);
  * BUG-414 videovae enc/dec split preserved across both stages;
  * i2v REQUIRED (NO FALLBACKS -- no silent t2v downgrade).
UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_ltx_av import (
    IA2V_REFINE_SIGMAS,
    LTX_DISTILLED_SIGMAS,
    LtxAudioInEngine,
    RECIPE_IA2V,
    _recipe_config,
)

W = wb.Wire
DEV_UNET = "ltx-2.3-22b-dev-Q3_K_M.gguf"


@pytest.fixture()
def ia2v_env(monkeypatch):
    monkeypatch.delenv("OTR_LTX_AV_RECIPE", raising=False)
    monkeypatch.delenv("OTR_LTX_AV_DISTILLED_LORA", raising=False)
    monkeypatch.setenv("OTR_LTX_AV_UNET", DEV_UNET)
    return LtxAudioInEngine()


def _graph(eng, image="still.png", w=832, h=480, length=121, seed=7):
    plan = {"text_prompt": "the radio is talking as it speaks", "seed": seed,
            "target_frame_count": length}
    return eng._build_graph(plan, length, w, h, "voice.wav", image)


def test_dev_default_recipe_is_ia2v(ia2v_env):
    assert ia2v_env._recipe() == RECIPE_IA2V


def test_two_stage_topology(ia2v_env):
    g = _graph(ia2v_env)
    # STAGE A: half-canvas empty latent + soft inplace anchor + masked audio
    assert g["emptylatent"]["inputs"]["width"] == 416      # 832 // 2
    assert g["emptylatent"]["inputs"]["height"] == 240     # 480 // 2
    assert g["emptylatent"]["inputs"]["length"] == 121
    assert g["inplace_base"]["inputs"]["strength"] == 0.7
    assert g["inplace_base"]["inputs"]["latent"] == W("emptylatent", 0)
    assert g["concat"]["inputs"]["video_latent"] == W("inplace_base", 0)
    assert g["concat"]["inputs"]["audio_latent"] == W("noisemask", 0)
    # the audio latent rides FROZEN under a ZERO mask
    assert g["solidmask"]["inputs"]["value"] == 0.0
    assert g["noisemask"]["inputs"]["samples"] == W("audioenc", 0)
    assert g["noisemask"]["inputs"]["mask"] == W("solidmask", 0)
    # STAGE B: separate -> upsample x2 -> hard re-anchor -> audio RE-CONCAT
    assert g["separate_base"]["inputs"]["av_latent"] == W("sampler", 0)
    assert g["upscaler"]["inputs"]["samples"] == W("separate_base", 0)
    assert g["inplace_refine"]["inputs"]["latent"] == W("upscaler", 0)
    assert g["inplace_refine"]["inputs"]["strength"] == 1.0
    assert g["concat_refine"]["inputs"]["video_latent"] == W("inplace_refine", 0)
    assert g["concat_refine"]["inputs"]["audio_latent"] == W("separate_base", 1)
    assert g["sampler_refine"]["inputs"]["latent_image"] == W("concat_refine", 0)
    # final: separate the REFINE output; decode VIDEO only (audio = mux-LAST)
    assert g["separate"]["inputs"]["av_latent"] == W("sampler_refine", 0)
    assert g["decode"]["inputs"]["samples"] == W("separate", 0)
    # the single-pass node must be GONE (no plain LTXVImgToVideo in ia2v)
    assert "i2v" not in g


def test_samplers_and_ladders(ia2v_env):
    g = _graph(ia2v_env)
    # base = ancestral over the front-loaded 8-step ladder (motion decisions)
    assert g["ksel"]["inputs"]["sampler_name"] == "euler_ancestral_cfg_pp"
    assert g["sigmas"]["inputs"]["values"] == list(LTX_DISTILLED_SIGMAS)
    # refine = plain euler_cfg_pp over the 3-step 0.85->0 ladder (detail only)
    assert g["ksel_refine"]["inputs"]["sampler_name"] == "euler_cfg_pp"
    assert g["sigmas_refine"]["inputs"]["values"] == list(IA2V_REFINE_SIGMAS)
    # two INDEPENDENT deterministic noise streams per beat
    assert g["noise"]["inputs"]["noise_seed"] == 7
    assert g["noise_refine"]["inputs"]["noise_seed"] == 8


def test_cropguides_refine_only(ia2v_env):
    g = _graph(ia2v_env)
    # base guider takes the conditioning DIRECT...
    assert g["guider"]["inputs"]["positive"] == W("cond", 0)
    assert g["guider"]["inputs"]["negative"] == W("cond", 1)
    # ...the refine guider goes THROUGH CropGuides anchored on the base latent
    assert g["cropguides"]["inputs"]["latent"] == W("separate_base", 0)
    assert g["guider_refine"]["inputs"]["positive"] == W("cropguides", 0)
    assert g["guider_refine"]["inputs"]["negative"] == W("cropguides", 1)


def test_half_distilled_lora_on_dev(ia2v_env):
    g = _graph(ia2v_env)
    assert g["lora"]["inputs"]["model"] == W("unet", 0)
    assert g["lora"]["inputs"]["strength_model"] == 0.5
    assert g["lora"]["inputs"]["lora_name"].endswith(
        "ltx-2.3-22b-distilled-lora-384.safetensors")
    assert "1.1" not in g["lora"]["inputs"]["lora_name"]
    # both guiders ride the SAME LoRA-wrapped model
    assert g["guider"]["inputs"]["model"] == W("lora", 0)
    assert g["guider_refine"]["inputs"]["model"] == W("lora", 0)


def test_guide_image_conditioning_chain(ia2v_env):
    # CANVAS-INDEPENDENT guide prep (lips-dont-talk root-cause fix): the
    # canonical's FIXED 1920x1088 -> longer-edge 1536 keeps the guide SHARP
    # (downscaled) at ANY render canvas; the old 1.5x-of-canvas version built
    # a soft UPSCALED guide at 512x288 -- the production mouth prior killer.
    for w, h in ((832, 480), (512, 288), (1280, 720)):
        g = _graph(ia2v_env, w=w, h=h)
        assert g["imagescale"]["inputs"]["image"] == W("loadimage", 0)
        assert g["imagescale"]["inputs"]["width"] == 1920
        assert g["imagescale"]["inputs"]["height"] == 1088
        assert g["resizelonger"]["inputs"]["longer_edge"] == 1536
        assert g["preprocess"]["inputs"]["img_compression"] == 18
        # BOTH inplace anchors condition on the SAME preprocessed guide
        assert g["inplace_base"]["inputs"]["image"] == W("preprocess", 0)
        assert g["inplace_refine"]["inputs"]["image"] == W("preprocess", 0)


def test_ia2v_render_canvas_defaults_canonical_native(ia2v_env, tmp_path,
                                                      monkeypatch):
    # operator quality catch: under the two-stage recipe the AV canvas
    # defaults to 832x480 (VRAM-laddered live: 704p breached the 14.5GB ceiling in the full pipeline; 512x288 was single-pass
    # VRAM math -> 2.9x upscale mush); single-pass recipes keep 512x288.
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_LTX_AV_RENDER_CANVAS", raising=False)

    req = rd.build_request_from_shot(_announcer_open_shot(),
                                     _ltx_ledger(tmp_path))
    assert (req["canvas"]["w"], req["canvas"]["h"]) == (832, 480)
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    req2 = rd.build_request_from_shot(_announcer_open_shot(),
                                      _ltx_ledger(tmp_path))
    assert (req2["canvas"]["w"], req2["canvas"]["h"]) == (512, 288)


def test_videovae_split_survives_two_stage(ia2v_env):
    # BUG-414 lock, two-stage edition: encode-side VAE is a distinct node whose
    # last consumer runs BEFORE each sampler peak; decode rides its own node.
    g = _graph(ia2v_env)
    assert "videovae" not in g
    for consumer in ("inplace_base", "inplace_refine", "upscaler"):
        assert g[consumer]["inputs"]["vae"] == W("videovae_enc", 0)
    assert g["decode"]["inputs"]["vae"] == W("videovae_dec", 0)


def test_ia2v_requires_init_image(ia2v_env):
    with pytest.raises(Exception):
        _graph(ia2v_env, image="")


def test_node_candidates_and_weights_gated_by_recipe(ia2v_env, monkeypatch):
    cands = ia2v_env._node_candidates()
    for logical in ("inplace_base", "upscaler", "cropguides", "noisemask",
                    "preprocess", "sampler_refine", "concat_refine"):
        assert logical in cands
    labels = [lbl for lbl, _p, _f in ia2v_env._weight_paths()]
    assert any("upscaler" in lbl for lbl in labels)
    assert any("LoRA" in lbl for lbl in labels)
    # distilled_native stays lean: none of the two-stage classes/weights
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")
    lean = LtxAudioInEngine()
    assert "upscaler" not in lean._node_candidates()
    lean_labels = [lbl for lbl, _p, _f in lean._weight_paths()]
    assert not any("upscaler" in lbl for lbl in lean_labels)


def test_keep_set_keeps_lora(ia2v_env):
    rcfg = _recipe_config(RECIPE_IA2V)
    keep = ia2v_env._keep_set("decode", rcfg)
    assert keep == {"unet", "lora", "decode"}


# --------------------------------------------------------------------------- #
# TALKING prompt register (lips-dont-talk kibitz fix, 2026-07-02): probe P4
# proved the scene/console register collapses articulation (4.15 -> 1.18);
# P2 proved music cannot drive lips (register deliberately unchanged there).
# --------------------------------------------------------------------------- #
_PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 80


def _ltx_ledger(tmp_path):
    s = tmp_path / "scene.png"
    s.write_bytes(_PNG)
    imgs = [{"object_id": "still_%s" % b, "kind": kind, "beat_id": b,
             "path": str(s)}
            for b, kind in (("b000", "scene_open"),
                            ("b000_music_open", "scene_open"),
                            ("b001", "scene_announcer"),
                            ("b002", "scene_character"))]
    # cast portrait for c01 -- the S4 ia2v portrait-init route (2026-07-02)
    p = tmp_path / "c01_portrait.png"
    p.write_bytes(_PNG)
    imgs.append({"object_id": "c01", "kind": "portrait", "path": str(p),
                 "w": 832, "h": 1216})
    # WIDE radio-face stills -- S4c: talking bookends REQUIRE them (the
    # toggle retired into default-on under ia2v).
    f = tmp_path / "face.png"
    f.write_bytes(_PNG)
    for _r in ("announcer_visual", "music_visual"):
        imgs.append({"object_id": "still_%s_radio_face_169" % _r,
                     "kind": "portrait", "path": str(f),
                     "w": 1472, "h": 832})
    return {"video": {"video_revision": 1, "shots": []},
            "lines": [{"line_id": "b000", "char_id": "",
                       "start_s": 0.0, "dur_s": 2.0},
                      {"line_id": "b001", "char_id": "announcer",
                       "start_s": 2.0, "dur_s": 2.0},
                      {"line_id": "b002", "char_id": "c01",
                       "start_s": 4.0, "dur_s": 2.0}],
            "images": {"images": imgs}}


def _announcer_open_shot():
    # a REAL announcer bookend (b001): has source lines, NOT the synthetic
    # b000 music open -- _ltx_motion_role_key maps it to the "announcer"
    # register (an empty-source_line_ids shot maps to music_open by design).
    return {"shot_id": "shot_b001", "beat_id": "b001",
            "engine_id": "ltx_audio_in", "role": "announcer_visual",
            "family": "audio_conditioned_video", "target_frame_count": 25,
            "source_line_ids": ["b001"], "char_id": "", "creative": {}}


def _music_open_shot():
    return {"shot_id": "shot_b000_music_open", "beat_id": "b000_music_open",
            "engine_id": "ltx_audio_in", "role": "music_visual",
            "family": "audio_conditioned_video", "target_frame_count": 25,
            "source_line_ids": [], "char_id": "", "creative": {}}


def _char_face_shot():
    m4 = ("face visible, speaking to camera, 40s, Lead Astronomer. Face: "
          "Oval, hooded eyes, aquiline nose, thin lips. Hair: short grey. "
          "Wearing a rumpled observatory jumper over a checked shirt, "
          + "x" * 700)
    return {"shot_id": "shot_b002", "beat_id": "b002",
            "engine_id": "ltx_audio_in", "role": "character_video",
            "family": "audio_conditioned_video", "target_frame_count": 25,
            "source_line_ids": ["b002"], "char_id": "c01",
            "creative": {"text_prompt": m4, "source": "m4"}}


def test_announcer_bookend_gets_talking_register(ia2v_env, tmp_path,
                                                 monkeypatch):
    from nodes._otr_video_engines import render_driver as rd

    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
    req = rd.build_request_from_shot(_announcer_open_shot(),
                                     _ltx_ledger(tmp_path))
    p = req["text_prompt"]
    assert "lips opening and closing" in p and "in sync" in p
    assert "Tuning dial needle" not in p          # console register replaced


def test_music_bookend_keeps_console_register(ia2v_env, tmp_path, monkeypatch):
    # P2: LTX cannot lip-sync to music -- the music bookends deliberately
    # KEEP the console-motion register even under the talking recipe.
    from nodes._otr_video_engines import render_driver as rd

    req = rd.build_request_from_shot(_music_open_shot(), _ltx_ledger(tmp_path))
    assert "Dial whip-pans" in req["text_prompt"]
    assert "lips" not in req["text_prompt"]


def test_announcer_bookend_unchanged_on_single_pass_recipe(tmp_path,
                                                           monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")

    req = rd.build_request_from_shot(_announcer_open_shot(),
                                     _ltx_ledger(tmp_path))
    assert "Tuning dial needle" in req["text_prompt"]   # byte-unchanged
    assert "lips" not in req["text_prompt"]


def test_char_face_beat_gets_compact_talking_prompt(ia2v_env, tmp_path,
                                                    monkeypatch):
    from nodes._otr_video_engines import render_driver as rd

    req = rd.build_request_from_shot(_char_face_shot(), _ltx_ledger(tmp_path))
    p = req["text_prompt"]
    assert "lips opening and closing naturally in sync" in p
    assert p.startswith("face visible, speaking to camera")
    assert len(p) <= rd._LTX_MOTION_PROMPT_MAX
    assert "xxxx" not in p                         # the 900-char wall is gone


@pytest.mark.parametrize(("style_id", "prefix"), [
    ("recur_frac", "surreal recursive fractal style face visible"),
    ("video_art", "video-art feedback style face visible"),
])
def test_char_face_compact_prompt_keeps_non_default_style_cue(
        ia2v_env, tmp_path, monkeypatch, style_id, prefix):
    from nodes._otr_video_engines import render_driver as rd

    led = _ltx_ledger(tmp_path)
    led["meta"] = {"visual_style": style_id}
    req = rd.build_request_from_shot(_char_face_shot(), led)
    p = req["text_prompt"]
    assert p.startswith(prefix)
    assert "lips opening and closing naturally in sync" in p
    assert len(p) <= rd._LTX_MOTION_PROMPT_MAX
    assert req["observability"]["visual_style"] == style_id


def test_announcer_with_m4_still_gets_talking_register(ia2v_env, tmp_path,
                                                       monkeypatch):
    # kibitz r2 must-fix: an announcer bookend carrying an M4 creative prompt
    # must STILL swap to the talking register under ia2v (M4 outranked on
    # announcer bookends only).
    from nodes._otr_video_engines import render_driver as rd

    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
    shot = _announcer_open_shot()
    shot["creative"] = {"text_prompt": "a moody radio studio scene, rain "
                                       "on the window", "source": "m4"}
    req = rd.build_request_from_shot(shot, _ltx_ledger(tmp_path))
    assert "lips opening and closing" in req["text_prompt"]
    assert "rain on the window" not in req["text_prompt"]


def test_announcer_with_m4_keeps_m4_on_single_pass_recipe(tmp_path,
                                                          monkeypatch):
    # kibitz r2 should-fix: M4 precedence on announcer bookends is UNTOUCHED
    # for the single-pass recipes -- only ia2v outranks it.
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")

    shot = _announcer_open_shot()
    shot["creative"] = {"text_prompt": "a moody radio studio scene, rain "
                                       "on the window", "source": "m4"}
    req = rd.build_request_from_shot(shot, _ltx_ledger(tmp_path))
    assert "rain on the window" in req["text_prompt"]
    assert "lips opening and closing" not in req["text_prompt"]


def test_char_face_no_m4_fallback_talks_under_ia2v(ia2v_env, tmp_path,
                                                   monkeypatch):
    # kibitz r3 must-fix: the ShotLock seam-gap fallback (char beat, NO M4)
    # must be a TALKING prompt under ia2v, never the scene-register fallback.
    from nodes._otr_video_engines import render_driver as rd

    shot = _char_face_shot()
    shot["creative"] = {}
    req = rd.build_request_from_shot(shot, _ltx_ledger(tmp_path))
    assert "lips opening and closing naturally in sync" in req["text_prompt"]


def test_char_face_no_m4_fallback_unchanged_on_single_pass(tmp_path,
                                                           monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")

    shot = _char_face_shot()
    shot["creative"] = {}
    req = rd.build_request_from_shot(shot, _ltx_ledger(tmp_path))
    assert req["text_prompt"] == rd._CHAR_FACE_FALLBACK_PROMPT


def test_char_face_beat_keeps_m4_on_single_pass_recipe(tmp_path, monkeypatch):
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")

    req = rd.build_request_from_shot(_char_face_shot(), _ltx_ledger(tmp_path))
    p = req["text_prompt"]
    assert "xxxx" in p                              # full M4 kept
    assert "stable centered subject" in p           # composition clause kept


# --------------------------------------------------------------------------- #
# S4 PORTRAIT INIT (portrait-vs-wide A/B, 2026-07-02): scene-still init left
# proof7 character speech beats at 0.62/0.32/1.27 while the isolation A/B on
# the SAME audio+prompt scored scene 0.57 vs portrait 2.86 (lag 0). Under the
# ia2v talking register a character beat conditions on the cast PORTRAIT.
# --------------------------------------------------------------------------- #


def test_char_beat_inits_on_portrait_under_ia2v(ia2v_env, tmp_path,
                                                monkeypatch):
    from nodes._otr_video_engines import render_driver as rd

    req = rd.build_request_from_shot(_char_face_shot(), _ltx_ledger(tmp_path))
    obs = req["observability"]
    assert obs["init_source"] == "character_portrait_ia2v"
    assert obs["init_image"] == "c01_portrait.png"


def test_char_beat_missing_portrait_fails_loud_under_ia2v(ia2v_env, tmp_path,
                                                          monkeypatch):
    from nodes._otr_video_engines import render_driver as rd

    led = _ltx_ledger(tmp_path)
    led["images"]["images"] = [im for im in led["images"]["images"]
                               if im.get("kind") != "portrait"]
    with pytest.raises(rd.RenderError, match="NO portrait"):
        rd.build_request_from_shot(_char_face_shot(), led)


def test_announcer_bookend_takes_radio_face_by_default_under_ia2v(
        ia2v_env, tmp_path, monkeypatch):
    # S4c: under the talking register the bookend ALWAYS conditions on the
    # WIDE radio-face still -- env toggle NOT required (operator eyeball
    # 2026-07-02, the pinprick faceless-announcer miss).
    from nodes._otr_video_engines import render_driver as rd

    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    req = rd.build_request_from_shot(_announcer_open_shot(),
                                     _ltx_ledger(tmp_path))
    obs = req["observability"]
    assert obs["init_source"] == "ltx_radio_face"
    assert obs["init_image"] == "face.png"


def test_announcer_bookend_missing_face_fails_loud_under_ia2v(
        ia2v_env, tmp_path, monkeypatch):
    # S4c fail-loud: a talking bookend with NO minted radio-face still is a
    # hard error (the faceless proof8 announcer must never silently recur).
    from nodes._otr_video_engines import render_driver as rd

    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    led = _ltx_ledger(tmp_path)
    led["images"]["images"] = [
        im for im in led["images"]["images"]
        if not str(im.get("object_id", "")).endswith("_radio_face_169")]
    with pytest.raises(rd.RenderError, match="radio-face"):
        rd.build_request_from_shot(_announcer_open_shot(), led)


def test_char_beat_keeps_scene_still_on_single_pass_recipe(tmp_path,
                                                           monkeypatch):
    # the 2026-06-20 wide-engines-never-portrait directive HOLDS outside
    # RECIPE_IA2V: pinned distilled_native keeps the scene-still init.
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.setenv("OTR_LTX_AV_RECIPE", "distilled_native")
    monkeypatch.setenv(
        "OTR_LTX_AV_UNET",
        r"distilled-1.1\ltx-2.3-22b-distilled-1.1-Q3_K_M.gguf")

    req = rd.build_request_from_shot(_char_face_shot(), _ltx_ledger(tmp_path))
    obs = req["observability"]
    assert obs["init_source"] == "scene_still"
    assert obs["init_image"] == "scene.png"
