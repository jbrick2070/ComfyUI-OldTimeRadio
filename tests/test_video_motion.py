"""A-S5 / CW-6 CPU tests -- the in-process motion adapters (ltx_video; wan_i2v RETIRED 2026-08-26).

LTX-Video (text->video) and Wan 2.2 (image->video) run IN-PROCESS in the main
cu130 venv (their wrappers are absent in the pytest sandbox), so every test here
exercises the COLD path: the adapters are registered + dark + gated, fail closed
without the flag / install, ltx_video fails closed when SageAttention is resident
(BUG-070), the init_image
aspect plan never stretches, the AS-3 lease is taken + released, and the pure
request / clip helpers are deterministic. The live load + the VRAM<=14.5 GB
boundary + render-twice pixels are the GPU smoke (operator), NOT covered here.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import types

import pytest

from nodes._otr_shared import gpu_residency as gr
from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import motion_common as mc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas as sc
from nodes._otr_video_engines.eng_ltx_video import LtxVideoEngine
from nodes._otr_video_engines.eng_wan_ti2v import WanTi2vEngine

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------- #
# Registration + default-OFF / dark
# --------------------------------------------------------------------------- #
def test_ltx_video_registered_and_dark():
    assert vreg.is_registered("ltx_video")
    eng = vreg.get_engine("ltx_video")
    assert isinstance(eng, LtxVideoEngine)
    # image_to_video since 2026-08-28: the declaration caught up with the
    # behaviour, and `render_driver._SCENE_INIT_FAMILIES` keys the scene-still
    # attachment off exactly this field.
    assert eng.family == "image_to_video" and eng.family in sc.FAMILIES
    # 2026-06-17: announcer + music DEFAULT moved to ltx_audio_in (the audio-in
    # lane). ltx_video keeps its roles (still SELECTABLE) but no longer claims a
    # default role.
    assert eng.default_roles == ()
    assert eng.requires_flag is None               # registry IS the menu (no flag gate)
    assert eng.required_inputs == ("text_prompt", "init_image")
    assert eng.declared_isolation == mc.ISOLATION_IN_PROCESS
    assert eng.binds_seed is True
    assert "ltx_video" in vreg.all_engine_names()   # in the full static dropdown


def test_registry_ltx_selectable_no_flag(monkeypatch):
    # 2026-06-17: ltx_video claims no DEFAULT role (announcer + music moved to
    # ltx_audio_in). Registry IS the menu: it is SELECTABLE for all its roles with
    # NO flag gate.
    monkeypatch.delenv("OTR_ENABLE_LTX_VIDEO", raising=False)
    for role in ("music_visual", "announcer_visual"):
        assert vreg.assert_usable("ltx_video", role) == "ltx_video"


def test_ltx_assert_usable_sage_then_install(monkeypatch):
    eng = vreg.get_engine("ltx_video")
    # No flag gate (registry IS the menu). The first gate is the SageAttention
    # check: resident Sage -> BUG-070 INCOMPATIBLE_PROFILE (fail closed BEFORE
    # the checkpoint check / any forward).
    monkeypatch.setitem(sys.modules, "sageattention",
                        types.ModuleType("sageattention"))
    with pytest.raises(vreg.EngineUnusable) as e2:
        eng.assert_usable(host_caps={}, profile={})
    assert e2.value.reason == vreg.EngineUsabilityReason.INCOMPATIBLE_PROFILE
    # Sage clear but a weight absent -> MISSING_MODEL (still closed). The
    # bogus name stays IN a recognized unet family (S5: recipe resolution runs
    # BEFORE the weight gate and an UNKNOWN family raises MALFORMED_CONFIG --
    # that path is pinned in test_ltx_video_hq_recipe).
    monkeypatch.delitem(sys.modules, "sageattention", raising=False)
    monkeypatch.delenv("OTR_SAGEATTENTION_PATCHED", raising=False)
    monkeypatch.setenv("OTR_LTX_VIDEO_UNET", "ltx-2.3-22b-dev-NO-SUCH.gguf")
    with pytest.raises(vreg.EngineUnusable) as e3:
        eng.assert_usable(host_caps={}, profile={})
    assert e3.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL


def test_sage_gate_helper_pure():
    mc.assert_sage_not_patched("ltx_video", "text_to_video", modules={}, env={})
    with pytest.raises(vreg.EngineUnusable) as ei:
        mc.assert_sage_not_patched("ltx_video", "text_to_video",
                                   modules={"sageattention": object()}, env={})
    assert ei.value.reason == vreg.EngineUsabilityReason.INCOMPATIBLE_PROFILE
    assert mc.sageattention_patched(
        modules={}, env={"OTR_SAGEATTENTION_PATCHED": "1"}) is True
    assert mc.sageattention_patched(modules={}, env={}) is False


def test_aspect_transform_uniform_no_stretch():
    plan = mc.resolve_aspect_transform(480, 832, 1280, 720, "pad")   # portrait->landscape
    sx = plan["scaled_w"] / plan["src_w"]
    sy = plan["scaled_h"] / plan["src_h"]
    assert abs(sx - sy) < 0.01                       # one uniform scale (no stretch)
    assert plan["scaled_w"] % 2 == 0 and plan["scaled_h"] % 2 == 0   # mod-2
    assert plan["scaled_w"] <= 1280 and plan["scaled_h"] <= 720      # pad fits inside
    cplan = mc.resolve_aspect_transform(480, 832, 1280, 720, "crop")
    assert cplan["scaled_w"] >= 1280 and cplan["scaled_h"] >= 720    # crop covers
    with pytest.raises(ValueError):
        mc.resolve_aspect_transform(480, 832, 1280, 720, "stretch")  # unknown policy
    with pytest.raises(ValueError):                  # hand-built stretch is caught
        mc.assert_no_silent_stretch(
            {"src_w": 480, "src_h": 832, "scaled_w": 1280, "scaled_h": 720})


def test_ltx_prepare_releases_lease_on_load_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    # No ComfyUI node classes registered -> load() resolves the GGUF graph
    # classes and raises (WrapperNodeMissing, a RuntimeError); the AS-3 lease
    # must be released, never stranded.
    monkeypatch.setattr(
        "nodes._otr_video_engines.wrapper_bridge.node_class_mappings",
        lambda mapping=None: {} if mapping is None else mapping)
    eng = LtxVideoEngine()
    assert not gr.is_held()
    with pytest.raises(RuntimeError):
        eng.prepare(host_caps={}, profile={}, session_ctx={})
    assert not gr.is_held()                       # lease released, never stranded


def test_teardown_idempotent_no_lease(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    for cls in (LtxVideoEngine, WanTi2vEngine):
        eng = cls()
        eng.unload()                              # no residency -> no crash
        eng.teardown(None)                        # no lease -> no raise
        eng.teardown({"lease": None})
        eng.teardown({"lease": None, "patchers": []})


# --------------------------------------------------------------------------- #
# Determinism contract (V-7): build the request twice -> identical; seed routes
# --------------------------------------------------------------------------- #
def test_ltx_build_render_request_deterministic():
    eng = vreg.get_engine("ltx_video")
    req = {"text_prompt": "a neon diner", "negative_prompt": "blurry",
           "timing": {"target_frame_count": 50}, "seed_bundle": {"request_seed": 7}}
    a, b = eng._build_render_request(req), eng._build_render_request(req)
    assert a == b                                  # render-twice request stability
    assert a == {"text_prompt": "a neon diner", "negative_prompt": "blurry",
                 "fps": 25, "target_frame_count": 50, "seed": 7}
    req2 = dict(req, seed_bundle={"request_seed": 8})
    assert eng._build_render_request(req2)["seed"] == 8 and \
        eng._build_render_request(req2) != a       # seed routes deterministically


def test_canonicalize_silent_bt709():
    for name in ("ltx_video", "wan_ti2v"):
        eng = vreg.get_engine(name)
        clip = eng.canonicalize({"out_path": "C:/o.mp4", "frame_count": 40},
                                {"shot_id": "s1"}, {})
        assert clip["has_audio"] is False              # V-1: mux owns audio
        assert clip["pixel_format"] == "yuv420p"
        assert clip["matrix"] == clip["transfer"] == clip["color_primaries"] == "bt709"
        assert clip["frame_count"] == 40 and clip["fps"] == 25
        assert clip["engine_id"] == name and clip["path"] == "C:/o.mp4"
        assert clip["clip_id"] == "s1"


# --------------------------------------------------------------------------- #
# Cold-import + ASCII source (V-12 / CLAUDE.md)
# --------------------------------------------------------------------------- #
def test_cold_import_motion_adapters_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_ltx_video;"
        "import nodes._otr_video_engines.eng_wan_ti2v;"
        "import nodes._otr_video_engines.motion_common;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs pulled at import:\n{r.stdout}\n{r.stderr}"


def test_new_motion_source_is_ascii_no_em_dash():
    for name in ("motion_common.py", "eng_ltx_video.py", "eng_wan_ti2v.py"):
        src_path = REPO_ROOT / "nodes" / "_otr_video_engines" / name
        src = src_path.read_text(encoding="utf-8")
        assert "—" not in src, f"em-dash forbidden in {name} (CLAUDE.md)"
        src.encode("ascii")                            # ASCII-only source


# --------------------------------------------------------------------------- #
# LTX-I2V ticket Part B (2026-06-11): img2vid branch -- DEFAULT ON since LK-1a
# --------------------------------------------------------------------------- #
def _i2v_req(init_path=""):
    r = {"shot_id": "s1", "text_prompt": "a relay station at dusk",
         "timing": {"target_frame_count": 50},
         "seed_bundle": {"request_seed": 7},
         "canvas": {"w": 1472, "h": 832, "fps": 25}}
    if init_path:
        r["asset_refs"] = {"init_image": init_path}
    return r


def test_the_still_is_REQUIRED_and_declared_like_every_sibling():
    """THE CONTRACT, after OTR_ENABLE_LTX_I2V was retired 2026-08-28.

    Operator ruling: "no switches nor flags, all video models request and
    ingest stills." The declaration is what makes it true -- `render_driver`
    attaches the per-beat scene still by FAMILY (`_SCENE_INIT_FAMILIES`), so a
    lane claiming `text_to_video` would silently stop receiving one. That is
    the exact build-breaker an independent review caught before this shipped.
    """
    eng = vreg.get_engine("ltx_video")
    assert type(eng).family == "image_to_video"
    assert "init_image" in type(eng).required_inputs
    assert getattr(type(eng), "accepts_still", False) is True
    # and the sibling it now matches
    sib = vreg.get_engine("ltx_8gb")
    assert type(sib).family == type(eng).family


def test_a_missing_or_stale_still_REFUSES_with_no_escape_hatch(tmp_path):
    """A4 (2026-07-27) held that a missing still is TERMINAL, not a silent
    degrade to text-only. That is still the policy, and there is no longer an
    opt-out to name in the error: the flag is gone, so the message must not
    advertise one.

    A STALE path is the case `render_driver` cannot catch -- it only checks
    that the still index holds a non-empty string, never that the file exists.
    """
    eng = vreg.get_engine("ltx_video")

    with pytest.raises(RuntimeError) as no_init:
        eng._require_init_image(_i2v_req())                   # no init at all
    assert "no text-only path" in str(no_init.value).lower()
    assert "OTR_ENABLE_LTX_I2V" not in str(no_init.value), (
        "the error must not advertise a flag that no longer exists")

    with pytest.raises(RuntimeError) as stale:
        eng._require_init_image(_i2v_req("Z:/nope/still.png"))
    assert "Z:/nope/still.png" in str(stale.value)

    # ...and a real on-disk still is returned, not merely accepted
    p = tmp_path / "still.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert eng._require_init_image(_i2v_req(str(p))) == str(p)


def test_the_retired_flag_helpers_are_GONE():
    """A guard against a helpful re-introduction. `_use_i2v` and
    `_i2v_enabled` existed only to read the env flag; `_build_graph` was the
    text-only graph builder they selected. All three are deleted, and a
    future reader wiring any of them back would be restoring the second
    render path the ruling forbids."""
    eng = type(vreg.get_engine("ltx_video"))
    for gone in ("_use_i2v", "_i2v_enabled"):
        assert not hasattr(eng, gone), (
            "%s came back -- that is the retired flag gate" % gone)
    # `_build_graph` STAYS and this asserts so deliberately. Both reviews said
    # to delete it as "the text-only builder"; it is not. `_build_graph_i2v`
    # CALLS it to build the shared base graph before layering the image
    # conditioning on, so deleting it broke the render outright. The thing
    # that was actually removed is the `use_i2v` false ARM in render_clip.
    assert hasattr(eng, "_build_graph"), (
        "_build_graph is the shared base graph _build_graph_i2v builds on -- "
        "it is not the text-only path and must not be deleted")


def test_i2v_candidates_and_graph_topology(monkeypatch):
    monkeypatch.setenv("OTR_LTX_SAMPLER", "distilled")   # explicit: test distilled I2V path
    monkeypatch.delenv("OTR_LTX_I2V_STRENGTH", raising=False)
    monkeypatch.delenv("OTR_LTX_VIDEO_CKPT_NAME", raising=False)
    eng = vreg.get_engine("ltx_video")
    cands = eng._node_candidates_i2v()
    # Installed wrapper API (2026-06-12): LTXVImgToVideoConditionOnly CONSUMES the
    # pure-noise EmptyLTXVLatentVideo ("latent") and returns the image-anchored
    # latent, so the latent node stays in the candidate set.
    assert "latent" in cands
    assert cands["img2vid"] == ("LTXVImgToVideoConditionOnly",)
    assert cands["loadimage"] == ("LoadImage",)
    plan = {"text_prompt": "x", "negative_prompt": "", "fps": 25,
            "target_frame_count": 169, "seed": 7}
    g = eng._build_graph_i2v(plan, 169, 1472, 832, "still.png")
    # latent node MUST be present -- img2vid takes it as an input.
    assert "latent" in g
    assert g["loadimage"]["inputs"]["image"] == "still.png"
    iv = g["img2vid"]["inputs"]
    # Installed signature: (vae, image, latent, strength) -> LATENT. No
    # positive/negative/width/height/length; feeding the old kwargs raised
    # "unexpected keyword argument 'positive'". strength MUST be present
    # (BUG-LOCAL-412: 0.75 default restored -- the 6/5 soft-anchor value, correct
    # now that LTX renders at native 832x480 via OTR_LTX_RENDER_CANVAS).
    assert tuple(iv["vae"]) == ("videovae", 0)   # GGUF recipe: the video VAE
    assert tuple(iv["image"]) == ("loadimage", 0)
    assert tuple(iv["latent"]) == ("latent", 0)
    assert iv["strength"] == 0.75                # distilled: HARDCODED to the mini
    assert "positive" not in iv and "negative" not in iv
    # Distilled sampler's latent_image is the image-anchored img2vid LATENT now
    # (NOT the raw pure-noise latent); conditioning flows from the text encoders.
    assert tuple(g["sampleradv"]["inputs"]["latent_image"]) == ("img2vid", 0)
    assert tuple(g["cond"]["inputs"]["positive"]) == ("pos", 0)
    assert tuple(g["cond"]["inputs"]["negative"]) == ("neg", 0)
    # txt2vid graph untouched by the branch
    g2 = eng._build_graph(plan, 169, 1472, 832)
    assert "latent" in g2 and "img2vid" not in g2
    # ksampler rollback mode: latent_image also comes from img2vid
    monkeypatch.setenv("OTR_LTX_SAMPLER", "ksampler")
    g3 = eng._build_graph_i2v(plan, 169, 1472, 832, "still.png")
    assert tuple(g3["ksampler"]["inputs"]["latent_image"]) == ("img2vid", 0)
    # strength is env-overridable + clamped
    monkeypatch.setenv("OTR_LTX_I2V_STRENGTH", "0.5")
    g4 = eng._build_graph_i2v(plan, 169, 1472, 832, "still.png")
    assert g4["img2vid"]["inputs"]["strength"] == 0.5


def test_i2v_driver_attaches_scene_still_with_trace(monkeypatch, tmp_path):
    """DEFAULT-ON (LK-1a) + a scene still in the ledger -> the ltx request
    shows init_source=scene_still; no still -> LOUD structural failure."""
    from nodes._otr_video_engines import render_driver as rd
    monkeypatch.delenv("OTR_LTX_RADIO_PROMPT", raising=False)
    still = tmp_path / "scene_b001.png"
    still.write_bytes(b"\x89PNG\r\n\x1a\n")
    ledger = {
        "meta": {"story_brief_status": "ok", "story_brief": "a relay station",
                 "story_brief_terms": {"setting": ["a relay station"]}},
        "lines": [{"line_id": "b001", "char_id": "announcer",
                   "speaker_role": "announcer", "text": "Tonight...",
                   "start_s": 0.0, "dur_s": 5.0}],
        "images": {"images": [{"object_id": "scene_b001", "kind": "scene_wide",
                               "beat_id": "b001", "path": str(still)}]},
    }
    shot = {"shot_id": "shot_b001", "source_line_ids": ["b001"],
            "role": "announcer_visual", "engine_id": "ltx_video",
            "group_id": "grp_announcer_visual", "target_frame_count": 50,
            "creative": {}}
    req = rd.build_request_from_shot(shot, ledger)
    assert req["observability"]["init_source"] == "scene_still"
    assert req["asset_refs"]["init_image"] == str(still)
    # missing still -> structural failure; no hidden text-only degradation
    ledger2 = dict(ledger, images={"images": []})
    # The refusal now comes from the SHARED scene-init path, not a bespoke
    # ltx_video branch -- that is the whole point of declaring
    # family = "image_to_video". Same policy, one owner.
    with pytest.raises(rd.DeferredImageGapError, match="NO scene still"):
        rd.build_request_from_shot(shot, ledger2)


