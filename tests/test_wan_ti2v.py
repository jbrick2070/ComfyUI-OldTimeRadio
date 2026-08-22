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
from nodes._otr_video_engines import eng_wan_ti2v as _WT
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
    # OTR_WAN_TI2V_PREQUALIFICATION is cleared with the rest so every test in
    # this file states which leg it is on: unset here means the PRODUCTION leg,
    # where the frozen recipe binds. A stray consent var in the parent process
    # would otherwise silently change what these tests mean.
    for k in ("OTR_WAN_TI2V_SAMPLER", "OTR_WAN_TI2V_SCHEDULER", "OTR_WAN_TI2V_STEPS",
              "OTR_WAN_TI2V_CFG", "OTR_WAN_TI2V_SHIFT", "OTR_WAN_TI2V_MAX_FRAMES",
              "OTR_WAN_TI2V_PREQUALIFICATION",
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
    """The static budget never SHRINKS a beat's target -- it either returns the
    hard-cap-clamped snapped target or raises. _floor_length clamps target=280
    down to the engine's native hard cap (177, a valid 4n+1) before budgeting.

    REWRITTEN 2026-08-13. This is the exact production path that refused two
    live 45-word render-gate legs, and it no longer refuses on an UNQUALIFIED
    row -- at either VRAM level, and for either half of the cost model:

    # overhead 7000 + 185/frame @1472x832; budget = free*0.85.
    # free 14775 -> budget 12558.75; the SLOPE would price 177 frames at
    #   (12558.75-7000)/185 = 30 affordable and refuse.
    # free 8000  -> budget 6800.00, under the 7000 OVERHEAD, so the fixed-cost
    #   floor would refuse before pricing a single frame.
    # Neither may, because `wan_ti2v` is not in QUALIFIED_COST_ROWS: the slope
    # is 7x a measured ladder and the overhead is an ABSOLUTE-peak figure being
    # compared against FREE bytes. Both return the hard-cap-clamped 177.

    A qualified row still refuses at both levels -- that arithmetic is covered
    in test_clip_fill.py, which qualifies the row to reach it.
    """
    _clear_floor_env(monkeypatch)
    from nodes._otr_video_engines import motion_common as mc
    assert not mc.cost_row_may_refuse("wan_ti2v")
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 14775.0)
    assert WanTi2vEngine()._floor_length(280, 1472, 832) == 177
    # Tight VRAM (8 GB free) is under the overhead floor and still not a refusal
    # -- and still never a silent floor-wins shrink to 17.
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 8000.0)
    assert WanTi2vEngine()._floor_length(280, 1472, 832) == 177


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


def test_non_portable_sampler_fails_closed_under_the_consent_act(monkeypatch):
    # uni_pc/sa_solver/MoEKSampler are not cross-platform -> fail closed. Since
    # the recipe freeze this check lives INSIDE the consent act: on a production
    # leg the knob cannot bind at all, so there is nothing to refuse.
    _clear_floor_env(monkeypatch)
    monkeypatch.setenv("OTR_ENABLE_WAN_TI2V", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_PREQUALIFICATION", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_SAMPLER", "uni_pc")
    with pytest.raises(EngineUnusable) as exc:
        WanTi2vEngine().assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG
    # The adapter's own reason for the whitelist survives the move into the
    # shared helper -- a refusal that cannot say why sends the operator hunting.
    assert "8GB/Mac/AMD floor" in str(exc.value)


def test_out_of_range_steps_fails_closed_under_the_consent_act(monkeypatch):
    _clear_floor_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_PREQUALIFICATION", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "999")
    with pytest.raises(EngineUnusable) as exc:
        WanTi2vEngine()._resolve_render_config()
    assert exc.value.reason is EngineUsabilityReason.MALFORMED_CONFIG


def test_non_numeric_cfg_fails_closed_under_the_consent_act(monkeypatch):
    _clear_floor_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_PREQUALIFICATION", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_CFG", "high")
    with pytest.raises(EngineUnusable):
        WanTi2vEngine()._resolve_render_config()


def test_a_stale_malformed_knob_can_NOT_kill_a_production_leg(monkeypatch):
    """PBUG-20260723-02 wearing the opposite mask.

    A long-booted server may carry a stale `OTR_WAN_TI2V_CFG=high` in its
    environment. On a production leg that value has NO EFFECT on the render --
    the recipe is frozen -- so parsing it just to reject it would kill a leg
    over a knob it does not influence. Named in a warning, never parsed."""
    _clear_floor_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_CFG", "high")
    monkeypatch.setenv("OTR_WAN_TI2V_STEPS", "999")
    monkeypatch.setenv("OTR_WAN_TI2V_SAMPLER", "uni_pc")
    cfg = WanTi2vEngine()._resolve_render_config()
    assert cfg["cfg"] == 5.0 and cfg["steps"] == 30 and cfg["sampler"] == "euler"


# --------------------------------------------------------------------------- #
# CHUNK 2 -- CLIP off-fp8 (Mac-safe GGUF umt5) + tiled VAE (schema-verified)
# --------------------------------------------------------------------------- #
def _clear_clip_vae_env(monkeypatch):
    for k in ("OTR_WAN_TI2V_CLIP_NAME", "OTR_WAN_TI2V_CLIP_LOADER",
              "OTR_WAN_TI2V_TILED_VAE", "OTR_WAN_TI2V_LOADER",
              "OTR_WAN_TI2V_UNET_NAME", "OTR_WAN_TI2V_CKPT",
              "OTR_WAN_TI2V_PREQUALIFICATION",
              "OTR_WAN_TI2V_VAE_TILE", "OTR_WAN_TI2V_VAE_OVERLAP",
              "OTR_WAN_TI2V_VAE_TEMPORAL", "OTR_WAN_TI2V_VAE_TEMPORAL_OVERLAP"):
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


#: The frozen value these four tests are written AGAINST, read from the recipe
#: rather than hardcoded. LANE 6 (2026-08-21) flipped it True -> False, and two
#: of these tests worked by setting an env value that OPPOSES the frozen one --
#: so a hardcoded opposing literal silently stops opposing anything the moment
#: the recipe is bumped. `test_tiled_vae_can_be_disabled_under_the_consent_act`
#: did exactly that: it kept PASSING while no longer discriminating between the
#: environment and the recipe. Deriving both the expectation and its opposite
#: keeps all four live across every future bump.
_FROZEN_TILED = _WT.WAN_TI2V_RECIPE["tiled_vae"]
_OPPOSING_ENV = "0" if _FROZEN_TILED else "1"
_FROZEN_DECODE = ("VAEDecodeTiled",) if _FROZEN_TILED else ("VAEDecode",)


def test_tiled_vae_matches_the_frozen_recipe(monkeypatch):
    """The shipped default IS the recipe's value -- currently OFF (lane 6)."""
    _clear_clip_vae_env(monkeypatch)
    eng = WanTi2vEngine()
    assert eng._tiled_vae() is _FROZEN_TILED
    assert eng._node_candidates()["vaedecode"] == _FROZEN_DECODE


def test_decode_inputs_match_the_frozen_decode_mode(monkeypatch):
    """Tiled decode carries the geometry; untiled carries NONE of it.

    The untiled half is the one that matters after lane 6: the recipe still
    RETAINS vae_tile / vae_overlap / vae_temporal / vae_temporal_overlap so the
    key set stays version-independent, and this proves none of them leak into a
    graph that has no tiled decoder to receive them.
    """
    _clear_clip_vae_env(monkeypatch)
    vd = _graph(monkeypatch)["vaedecode"]["inputs"]
    geometry = ("tile_size", "overlap", "temporal_size", "temporal_overlap")
    if _FROZEN_TILED:
        assert vd["tile_size"] == 256
        assert vd["temporal_size"] == 16      # the video-VAE peak lever
        assert all(k in vd for k in geometry)
    else:
        assert sorted(vd) == ["samples", "vae"]
        assert not any(k in vd for k in geometry)


def test_tiled_vae_can_be_flipped_under_the_consent_act(monkeypatch):
    # The env value OPPOSES the frozen one, so this can still tell whether the
    # environment or the recipe won. That channel exists only inside a
    # measurement run.
    _clear_clip_vae_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_PREQUALIFICATION", "1")
    monkeypatch.setenv("OTR_WAN_TI2V_TILED_VAE", _OPPOSING_ENV)
    eng = WanTi2vEngine()
    assert eng._tiled_vae() is (not _FROZEN_TILED)
    assert eng._node_candidates()["vaedecode"] != _FROZEN_DECODE
    vd = _graph(monkeypatch)["vaedecode"]["inputs"]
    # Flipping AWAY from the frozen mode must also flip the geometry.
    assert ("tile_size" in vd) is (not _FROZEN_TILED)


def test_tiled_vae_can_NOT_be_flipped_on_a_production_leg(monkeypatch):
    # The control for the test above: same OPPOSING value, no consent act. The
    # frozen recipe wins and the shipped decode node stays.
    _clear_clip_vae_env(monkeypatch)
    monkeypatch.setenv("OTR_WAN_TI2V_TILED_VAE", _OPPOSING_ENV)
    eng = WanTi2vEngine()
    assert eng._tiled_vae() is _FROZEN_TILED
    assert eng._node_candidates()["vaedecode"] == _FROZEN_DECODE


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
