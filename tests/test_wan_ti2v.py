"""GO_FORWARD section 4 / 4A (M8 + S2) -- the wan_ti2v 5B TI2V engine.

CPU-only coverage for the new 8GB-tier engine: registration + roles, the GGUF
loader-mode inference, the 5B node-candidate set (Wan22ImageToVideoLatent, NOT
WanImageToVideo), the 5B graph wiring (KSampler takes the CLIPTextEncode
positive/negative DIRECTLY + the latent from Wan22ImageToVideoLatent), the M8
Wan2.2-VAE fail-closed guard, the assert_usable fail-closed ladder, and that the
shared pure helpers (build_render_request / staged name / clip dict) bind to this
engine. No GPU, no model load.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_wan_ti2v import WanTi2vEngine
from nodes._otr_video_engines.registry import (
    EngineUnusable, EngineUsabilityReason,
)


# --------------------------------------------------------------------------- #
# registration / identity
# --------------------------------------------------------------------------- #
def test_registered_and_identity():
    assert vreg.is_registered("wan_ti2v")
    eng = vreg.get_engine("wan_ti2v")
    assert eng.name == "wan_ti2v"
    assert eng.family == "image_to_video"
    assert eng.requires_flag is None            # registry IS the menu (no flag gate)
    assert eng.default_roles == ()              # selectable-not-default
    assert eng.required_inputs == ("init_image",)
    assert eng.commercial_clean is True         # Apache-2.0 (manifest)


def test_serves_the_image_init_roles():
    # FLEXIBLE roles (2026-06-18, commit 1e0fe08): wan_ti2v is selectable for EVERY
    # role -- role_compat is the real gate (it admits wan_ti2v only where the role
    # supplies the required init_image). The engine declares the full ROLES tuple.
    from nodes._otr_shared.role_compat import ROLES
    eng = WanTi2vEngine()
    assert set(eng.roles) == set(ROLES)


def test_distinct_from_wan_i2v_not_a_subclass():
    # Shares only the pure mixin -- it must NOT be a WanI2VEngine subclass.
    from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine
    assert not issubclass(WanTi2vEngine, WanI2VEngine)


# --------------------------------------------------------------------------- #
# loader mode + node candidates (the 5B difference)
# --------------------------------------------------------------------------- #
def test_loader_mode_infers_gguf_from_default_unet(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_LOADER", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_UNET_NAME", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_CKPT", raising=False)
    assert WanTi2vEngine()._loader_mode() == "gguf"


def test_loader_mode_explicit_safetensors(monkeypatch):
    monkeypatch.setenv("OTR_WAN_TI2V_LOADER", "safetensors")
    assert WanTi2vEngine()._loader_mode() == "safetensors"


def test_node_candidates_use_the_5b_latent_node(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_LOADER", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_UNET_NAME", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_CKPT", raising=False)
    cands = WanTi2vEngine()._node_candidates()
    assert cands["latent"] == ("Wan22ImageToVideoLatent",)
    assert cands["unet"] == ("UnetLoaderGGUF",)         # gguf default
    assert "wan" not in cands                           # not the 14B WanImageToVideo


def test_node_candidates_unet_switches_to_core_loader_on_safetensors(monkeypatch):
    monkeypatch.setenv("OTR_WAN_TI2V_LOADER", "safetensors")
    assert WanTi2vEngine()._node_candidates()["unet"] == ("UNETLoader",)


def test_loader_names_default_to_gguf_umt5_and_wan22_vae(monkeypatch):
    # 2026-06-18 roundtable: the floor CLIP default moved OFF fp8 (Mac MPS
    # Float8_e4m3fn TypeError, ComfyUI #9255) to the GGUF umt5 encoder.
    for k in ("OTR_WAN_TI2V_CLIP_NAME", "OTR_WAN_TI2V_VAE_NAME"):
        monkeypatch.delenv(k, raising=False)
    names = WanTi2vEngine()._loader_names()
    assert names["clip"] == "umt5-xxl-encoder-Q5_K_M.gguf"
    assert names["vae"] == "wan2.2_vae.safetensors"


# --------------------------------------------------------------------------- #
# the 5B graph wiring -- KSampler takes pos/neg DIRECT + latent from the 5B node
# --------------------------------------------------------------------------- #
def _graph(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_LOADER", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_UNET_NAME", raising=False)
    monkeypatch.delenv("OTR_WAN_TI2V_CKPT", raising=False)
    eng = WanTi2vEngine()
    req = {"text_prompt": "a slow pan", "canvas": {"w": 832, "h": 480}}
    return eng._build_graph(req, "init.png", {"seed": 7}, 81, 832, 480)


def test_graph_ksampler_takes_pos_neg_directly_and_5b_latent(monkeypatch):
    g = _graph(monkeypatch)
    ks = g["ksampler"]["inputs"]
    assert tuple(ks["positive"]) == ("pos", 0)
    assert tuple(ks["negative"]) == ("neg", 0)
    assert tuple(ks["latent_image"]) == ("latent", 0)


def test_graph_latent_node_takes_vae_and_start_image(monkeypatch):
    g = _graph(monkeypatch)
    lat = g["latent"]["inputs"]
    assert tuple(lat["vae"]) == ("vae", 0)
    assert tuple(lat["start_image"]) == ("loadimage", 0)
    assert lat["width"] == 832 and lat["height"] == 480 and lat["length"] == 81
    assert "positive" not in lat and "negative" not in lat   # 5B latent has no cond


def test_graph_gguf_unet_inputs_have_no_weight_dtype(monkeypatch):
    g = _graph(monkeypatch)
    assert "weight_dtype" not in g["unet"]["inputs"]
    assert g["unet"]["inputs"]["unet_name"].endswith(".gguf")


def test_graph_modelsampling_shift_default_5(monkeypatch):
    monkeypatch.delenv("OTR_WAN_TI2V_SHIFT", raising=False)
    g = _graph(monkeypatch)
    assert g["modelsampling"]["inputs"]["shift"] == 5.0


# --------------------------------------------------------------------------- #
# M8 -- the Wan2.2-VAE fail-closed guard
# --------------------------------------------------------------------------- #
def _stage_full_install(tmp_path, monkeypatch, *, vae_name="wan2.2_vae.safetensors"):
    ckpt = tmp_path / "Wan2.2-TI2V-5B-Q5_K_M.gguf"
    ckpt.write_bytes(b"unet-placeholder")
    monkeypatch.setenv("OTR_ENABLE_WAN_TI2V", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_CKPT", str(ckpt))
    clip_dir = tmp_path / "text_encoders"
    clip_dir.mkdir()
    vae_dir = tmp_path / "vae"
    vae_dir.mkdir()
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_DIR", str(clip_dir))
    monkeypatch.setenv("OTR_WAN_TI2V_VAE_DIR", str(vae_dir))
    if vae_name is not None:
        monkeypatch.setenv("OTR_WAN_TI2V_VAE_NAME", vae_name)
    eng = WanTi2vEngine()
    names = eng._loader_names()
    (clip_dir / names["clip"]).write_bytes(b"clip-placeholder")
    if names["vae"]:
        (vae_dir / names["vae"]).write_bytes(b"vae-placeholder")
    return eng


def test_assert_usable_passes_with_wan22_vae(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    assert eng.assert_usable(host_caps={}, profile={}) == "wan_ti2v"


def test_m8_rejects_the_21_vae(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch,
                              vae_name="wan_2.1_vae.safetensors")
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "M8" in str(exc.value)


def test_m8_rejects_empty_vae_name(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch, vae_name="")
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "M8" in str(exc.value)


def test_m8_rejects_a_wrong_but_present_vae(tmp_path, monkeypatch):
    # 2026-06-18 roundtable: the guard is now an allow-list, so any non-approved
    # VAE basename (not just empty / 2.1) fails closed.
    eng = _stage_full_install(tmp_path, monkeypatch,
                              vae_name="some_other_vae.safetensors")
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL


# --------------------------------------------------------------------------- #
# CLIP-FILL frame budgeting (2026-06-18 -- supersedes the static 17-frame floor)
# --------------------------------------------------------------------------- #
def _clear_floor_env(monkeypatch):
    for k in ("OTR_WAN_TI2V_SAMPLER", "OTR_WAN_TI2V_SCHEDULER", "OTR_WAN_TI2V_STEPS",
              "OTR_WAN_TI2V_CFG", "OTR_WAN_TI2V_SHIFT", "OTR_WAN_TI2V_MAX_FRAMES",
              "OTR_VIDEO_COST_OVERHEAD_MB", "OTR_VIDEO_COST_PER_FRAME_MB",
              "OTR_VIDEO_BUDGET_MARGIN"):
        monkeypatch.delenv(k, raising=False)


def _no_vram_read(monkeypatch):
    """Force free_vram_mb()->None so a test is deterministic regardless of whether
    the box running it has a live GPU (the headless suite box DOES)."""
    from nodes._otr_video_engines import motion_common as mc
    monkeypatch.setattr(mc, "free_vram_mb", lambda: None)


def test_floor_length_default_is_the_motion_floor(monkeypatch):
    # No target + no live VRAM read -> the _TI2V_DEFAULT_FRAMES=17 motion floor.
    # (The old target_fps-as-frame-count bug would have given 33.)
    _clear_floor_env(monkeypatch)
    _no_vram_read(monkeypatch)
    assert WanTi2vEngine()._floor_length(None) == 17
    assert WanTi2vEngine()._floor_length(0) == 17


def test_floor_length_honors_target_without_vram_read(monkeypatch):
    # CLIP-FILL: with NO live VRAM read the beat's audio-derived target is HONORED
    # (snapped to 4n+1 <= target), no longer clamped to 17 -- the freeze fix.
    # A short beat (33) renders natively; a long beat (280) is capped to the engine
    # NATIVE-render max _TI2V_MAX_FRAMES=177 (4n+1) -- the render then ping-pong-
    # extends 177 -> 280 so the beat is still FILLED (no freeze).
    _clear_floor_env(monkeypatch)
    _no_vram_read(monkeypatch)
    assert WanTi2vEngine()._floor_length(33) == 33
    assert WanTi2vEngine()._floor_length(280) == 177


def test_floor_length_predicts_from_live_vram(monkeypatch):
    # With a live free-VRAM read the predictor bounds the target by the budget:
    # overhead 7000 + 185/frame @1472x832; budget = min(free,14500)*0.85.
    _clear_floor_env(monkeypatch)
    from nodes._otr_video_engines import motion_common as mc
    # ~5080 clean box: free ~14775 -> budget 12325 -> (12325-7000)/185 ~= 28 -> 29.
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 14775.0)
    assert WanTi2vEngine()._floor_length(280, 1472, 832) == 29
    # Tight VRAM (8 GB free) cannot fit even one frame past overhead -> the motion
    # floor (17) WINS (a beat always carries some motion; the render-window NVML
    # probe is the real over-budget guard).
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 8000.0)
    assert WanTi2vEngine()._floor_length(280, 1472, 832) == 17


def test_floor_max_override_is_an_absolute_hard_cap(monkeypatch):
    # OTR_WAN_TI2V_MAX_FRAMES pins an absolute hard cap for a tiny/8GB card; with no
    # VRAM read (isolating the cap logic) the target is clamped to it.
    _clear_floor_env(monkeypatch)
    _no_vram_read(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "49")     # 49 = 4*12+1
    assert WanTi2vEngine()._floor_length(49) == 49
    assert WanTi2vEngine()._floor_length(177) == 49        # clamped to the cap


def test_default_sampler_is_portable_euler(monkeypatch):
    _clear_floor_env(monkeypatch)
    cfg = WanTi2vEngine()._resolve_render_config()
    assert cfg["sampler"] == "euler"
    assert cfg["scheduler"] == "simple"


def test_graph_default_sampler_is_euler(monkeypatch):
    _clear_floor_env(monkeypatch)
    g = _graph(monkeypatch)
    assert g["ksampler"]["inputs"]["sampler_name"] == "euler"


def test_non_portable_sampler_fails_closed(monkeypatch):
    # uni_pc/sa_solver/MoEKSampler are not cross-platform -> fail closed (the
    # resolver runs before the install checks, so no staging needed).
    _clear_floor_env(monkeypatch)
    monkeypatch.setenv("OTR_ENABLE_WAN_TI2V", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_SAMPLER", "uni_pc")
    with pytest.raises(EngineUnusable) as exc:
        WanTi2vEngine().assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_out_of_range_steps_fails_closed(monkeypatch):
    _clear_floor_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "999")
    with pytest.raises(EngineUnusable) as exc:
        WanTi2vEngine()._resolve_render_config()
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_non_numeric_cfg_fails_closed(monkeypatch):
    _clear_floor_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_CFG", "high")
    with pytest.raises(EngineUnusable):
        WanTi2vEngine()._resolve_render_config()


# --------------------------------------------------------------------------- #
# CHUNK 2 -- CLIP off-fp8 (Mac-safe GGUF umt5) + tiled VAE (schema-verified)
# --------------------------------------------------------------------------- #
def _clear_clip_vae_env(monkeypatch):
    for k in ("OTR_WAN_TI2V_CLIP_NAME", "OTR_WAN_TI2V_CLIP_LOADER",
              "OTR_WAN_TI2V_TILED_VAE", "OTR_WAN_TI2V_LOADER",
              "OTR_WAN_TI2V_UNET_NAME", "OTR_WAN_TI2V_CKPT"):
        monkeypatch.delenv(k, raising=False)


def test_clip_loader_mode_default_is_gguf(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    assert WanTi2vEngine()._clip_loader_mode() == "gguf"


def test_clip_loader_mode_explicit_safetensors(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_LOADER", "safetensors")
    assert WanTi2vEngine()._clip_loader_mode() == "safetensors"


def test_node_candidates_clip_is_gguf_loader_by_default(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    assert WanTi2vEngine()._node_candidates()["clip"] == ("CLIPLoaderGGUF",)


def test_node_candidates_clip_core_loader_on_fp16(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_NAME", "umt5_xxl_fp16.safetensors")
    assert WanTi2vEngine()._node_candidates()["clip"] == ("CLIPLoader",)


def test_graph_gguf_clip_has_no_device_arg(monkeypatch):
    # CLIPLoaderGGUF takes clip_name + type only (no device), verified vs /object_info.
    _clear_clip_vae_env(monkeypatch)
    g = _graph(monkeypatch)
    clip_in = g["clip"]["inputs"]
    assert clip_in["type"] == "wan"
    assert "device" not in clip_in


def test_graph_safetensors_clip_keeps_device(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_CLIP_NAME", "umt5_xxl_fp16.safetensors")
    g = _graph(monkeypatch)
    assert g["clip"]["inputs"]["device"] == "default"


def test_tiled_vae_default_on(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    eng = WanTi2vEngine()
    assert eng._tiled_vae() is True
    assert eng._node_candidates()["vaedecode"] == ("VAEDecodeTiled",)


def test_tiled_vae_inputs_have_temporal_knobs(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    g = _graph(monkeypatch)
    vd = g["vaedecode"]["inputs"]
    assert vd["tile_size"] == 256
    assert vd["temporal_size"] == 16          # the video-VAE peak lever
    assert "temporal_overlap" in vd and "overlap" in vd


def test_tiled_vae_can_be_disabled(monkeypatch):
    _clear_clip_vae_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_TILED_VAE", "0")
    eng = WanTi2vEngine()
    assert eng._tiled_vae() is False
    assert eng._node_candidates()["vaedecode"] == ("VAEDecode",)
    g = _graph(monkeypatch)
    assert "tile_size" not in g["vaedecode"]["inputs"]


# --------------------------------------------------------------------------- #
# assert_usable fail-closed ladder
# --------------------------------------------------------------------------- #
def test_no_flag_gate_selectable(tmp_path, monkeypatch):
    # No flag gate (registry IS the menu): a full install is usable even with the
    # (vestigial) OTR_ENABLE_WAN_TI2V unset.
    eng = _stage_full_install(tmp_path, monkeypatch)
    monkeypatch.delenv("OTR_ENABLE_WAN_TI2V", raising=False)
    assert eng.assert_usable(host_caps={}, profile={}) == "wan_ti2v"


def test_missing_unet_fails_closed(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_CKPT", str(tmp_path / "nope.gguf"))
    monkeypatch.setenv("OTR_WAN_TI2V_UNET_DIR", str(tmp_path / "empty"))
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "UNET not found" in str(exc.value)


def test_missing_clip_fails_closed(tmp_path, monkeypatch):
    eng = _stage_full_install(tmp_path, monkeypatch)
    # delete the staged clip so only the VAE remains
    import os
    clip = os.path.join(os.environ["OTR_WAN_TI2V_CLIP_DIR"],
                        eng._loader_names()["clip"])
    os.remove(clip)
    with pytest.raises(EngineUnusable) as exc:
        eng.assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    assert "CLIP/umt5" in str(exc.value)


# --------------------------------------------------------------------------- #
# shared pure helpers bind to this engine
# --------------------------------------------------------------------------- #
def test_shared_clip_dict_carries_this_engine_id():
    eng = WanTi2vEngine()
    clip = eng._clip_from_raw({"out_path": "/x/y.mp4", "frame_count": 81},
                              {"shot_id": "b004"})
    assert clip["engine_id"] == "wan_ti2v"
    assert clip["family"] == "image_to_video"
    assert clip["has_audio"] is False
    assert clip["clip_id"] == "b004"


def test_shared_clip_dict_fallback_id_uses_engine_name():
    eng = WanTi2vEngine()
    clip = eng._clip_from_raw({}, {})
    assert clip["clip_id"] == "wan_ti2v_clip"


def test_shared_build_render_request_is_deterministic():
    eng = WanTi2vEngine()
    req = {"asset_refs": {"init_image": "/p.png"},
           "seed_bundle": {"request_seed": 5},
           "timing": {"target_frame_count": 50}}
    a, b = eng._build_render_request(req), eng._build_render_request(req)
    assert a == b
    assert a["seed"] == 5 and a["init_image"] == "/p.png"


def test_shared_staged_name_keys_on_shot_seed_dims():
    eng = WanTi2vEngine()
    name = eng._staged_init_name(
        {"shot_id": "b009", "seed_bundle": {"request_seed": 3}}, 832, 480)
    assert name == "otr_wan_init_b009_s3_832x480.png"
