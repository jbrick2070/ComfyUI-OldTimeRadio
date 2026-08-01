"""``fastwan_8gb`` -- the 3-step DMD distillation as an equal, additive engine.

Every test here pins something the kibitz arc found, and several would have passed
vacuously before the recipe seam landed -- see
``test_recipe_is_fastwans_not_the_incumbents``.
"""

import pytest

from nodes._otr_shared import public_engines as _PE
from nodes._otr_video_engines import eng_fastwan_8gb as _FW
from nodes._otr_video_engines import eng_wan_ti2v as _WT
from nodes._otr_video_engines import motion_common as _MC
from nodes._otr_video_engines import registry as _REG


def _engine(monkeypatch):
    for k in ("OTR_WAN_TI2V_LOADER", "OTR_WAN_TI2V_UNET_NAME", "OTR_WAN_TI2V_CKPT",
              "OTR_FASTWAN_8GB_LORA_NAME", "OTR_FASTWAN_8GB_PREQUALIFICATION"):
        monkeypatch.delenv(k, raising=False)
    return _FW.FastWan8gbEngine()


def _graph(monkeypatch, seed=7, length=17):
    eng = _engine(monkeypatch)
    req = {"text_prompt": "a slow pan", "canvas": {"w": 832, "h": 480}}
    return eng._build_graph(req, "init.png", {"seed": seed}, length, 832, 480)


# --------------------------------------------------------------------------- #
# registration -- half-registration fails CI
# --------------------------------------------------------------------------- #
def test_engine_is_registered_and_roster_is_clean():
    assert _REG.is_registered("fastwan_8gb")
    audit = _REG.audit_engine_roster()
    assert not audit.get("missing"), audit
    assert not audit.get("unexpected"), audit


def test_capability_row_declares_the_lora_separately():
    """The LoRA is its own model requirement so preflight fails CLOSED. Without
    it the graph renders 3 steps through the UN-distilled base: no error, just
    ruined output wearing a FastWan receipt."""
    row = _REG.CAPABILITIES["fastwan_8gb"]
    assert row["model_requirements"] == ["wan2.2-ti2v-5b", "fastwan-2.2-5b-lora"]
    assert row["device_backends"] == ["cuda"]
    assert row["requires_vendor"] is None
    assert row["needs_fp8_te"] is False and row["needs_fp4_te"] is False


def test_public_menu_is_additive_and_still_a_bijection():
    assert _PE._PUBLIC_ENGINES["fastwan_8gb"] == "fastwan_8gb"
    # The incumbent's row is untouched -- saved graphs still resolve.
    assert _PE._PUBLIC_ENGINES["wan_8gb"] == "wan_ti2v"
    assert len(_PE._PUBLIC_ENGINES) == len(_PE._INTERNAL_TO_PUBLIC)
    assert _PE._INTERNAL_TO_PUBLIC["fastwan_8gb"] == "fastwan_8gb"


def test_label_sells_throughput_not_quality():
    """It is the SAME motion at the SAME canvas for the SAME VRAM, ~2.7x sooner."""
    label = _PE._PUBLIC_LABEL["fastwan_8gb"]
    assert "3-step" in label
    assert not any(w in label.lower() for w in ("better", "hq", "high quality"))


def test_frame_floor_and_cost_row_are_registered():
    """A missing floor row means _DEFAULT_MOTION_FLOOR = 1, which lets a length
    below the 5B VAE's 4-frame quantum reach the model."""
    assert _MC.FRAME_MOTION_FLOOR["fastwan_8gb"] == 17
    assert _MC.FRAME_COST_MODEL["fastwan_8gb"] == (7000.0, 185.0)


# --------------------------------------------------------------------------- #
# the canvas, declared
# --------------------------------------------------------------------------- #
def test_render_canvas_is_declared_and_32_legal():
    w, h = _FW.FastWan8gbEngine.render_canvas
    assert (w, h) == (832, 480)
    assert w % 32 == 0 and h % 32 == 0
    from nodes._otr_video_engines import render_driver as _RD
    assert _RD.declared_render_canvas("fastwan_8gb") == (832, 480)


# --------------------------------------------------------------------------- #
# THE SEED TRAP -- KSampler takes `seed`, SamplerCustom takes `noise_seed`
# --------------------------------------------------------------------------- #
def test_sampler_takes_noise_seed_not_seed(monkeypatch):
    """Copying the incumbent's input dict would land `seed` on a node that does
    not read it, and every clip would render on SamplerCustom's own default --
    no error, no log line. The repo already learned this once: the bench runner
    carries a hand-written dual-key reader for exactly this split."""
    g = _graph(monkeypatch, seed=4242)
    inputs = g["sampler"]["inputs"]
    assert inputs["noise_seed"] == 4242
    assert "seed" not in inputs, "seed lands on a node SamplerCustom never reads"
    assert inputs["add_noise"] is True


def test_ksampler_node_is_gone(monkeypatch):
    g = _graph(monkeypatch)
    assert "ksampler" not in g
    assert "KSampler" not in _engine(monkeypatch)._node_candidates().get(
        "ksampler", ())


# --------------------------------------------------------------------------- #
# the sampler chain
# --------------------------------------------------------------------------- #
def test_graph_wires_the_dmd_chain(monkeypatch):
    g = _graph(monkeypatch)
    s = g["sampler"]["inputs"]
    assert tuple(s["sampler"]) == ("dmdsampler", 0)
    assert tuple(s["sigmas"]) == ("sigmas", 0)
    assert tuple(s["positive"]) == ("pos", 0)
    assert tuple(s["negative"]) == ("neg", 0)
    assert tuple(s["latent_image"]) == ("latent", 0)
    assert tuple(s["model"]) == ("modelsampling", 0)
    assert s["cfg"] == 1.0
    assert g["dmdsampler"]["inputs"] == {}


def test_model_is_routed_through_the_lora(monkeypatch):
    """ModelSamplingSD3 must read the PATCHED model. Wiring the base by mistake
    renders 3 steps through an un-distilled model -- no error, ruined output."""
    g = _graph(monkeypatch)
    assert tuple(g["modelsampling"]["inputs"]["model"]) == ("lora", 0)


def test_vae_decode_follows_the_new_sampler_node(monkeypatch):
    """_vaedecode_inputs used to hardcode W("ksampler", 0); with the sampler node
    renamed that would wire the decode to a node that is not in the graph."""
    g = _graph(monkeypatch)
    assert tuple(g["vaedecode"]["inputs"]["samples"]) == ("sampler", 0)


def test_manual_sigmas_is_a_comma_separated_string(monkeypatch):
    """ManualSigmas takes a STRING, verified against the working bench graph. A
    Python tuple would not serialize into the graph."""
    g = _graph(monkeypatch)
    sig = g["sigmas"]["inputs"]["sigmas"]
    assert isinstance(sig, str)
    assert sig == "1.0, 0.757, 0.522, 0.0"


def test_steps_equal_len_sigmas_minus_one():
    """The recipe cannot claim a step count its own schedule does not produce."""
    n = len([s for s in _FW.FASTWAN_SIGMAS.split(",") if s.strip()])
    assert _FW.FASTWAN_8GB_RECIPE["steps"] == n - 1 == 3


# --------------------------------------------------------------------------- #
# the recipe is FastWan's own (this is the seam paying off)
# --------------------------------------------------------------------------- #
def test_recipe_is_fastwans_not_the_incumbents(monkeypatch):
    """WOULD HAVE FAILED before the recipe seam: the accessors read module-level
    wan_ti2v constants, so this subclass would have resolved 30 steps at cfg 5.0
    and stamped a FastWan receipt on it."""
    cfg = _engine(monkeypatch)._resolve_render_config()
    assert cfg["steps"] == 3
    assert cfg["cfg"] == 1.0
    assert cfg["shift"] == 5.0
    assert cfg["sigmas"] == "1.0, 0.757, 0.522, 0.0"
    assert cfg["lora_strength"] == 1.0
    # ...and the incumbent is unmoved by this engine existing.
    inc = _WT.WanTi2vEngine()._resolve_render_config()
    assert inc["steps"] == 30 and inc["cfg"] == 5.0


def test_receipt_is_its_own_frozen_string(monkeypatch):
    receipt = _engine(monkeypatch)._recipe_receipt()
    assert receipt.startswith("fastwan22_ti2v_5b_dmd3_i2v_v1")
    assert "wan22_ti2v_5b_i2v_single_pass_v1" not in receipt


def test_consent_act_is_not_shared_with_the_incumbent():
    """One switch for two tiers would open both and stamp +prequalification on a
    clip that had rendered frozen."""
    assert (_FW.FastWan8gbEngine.prequalification_env
            != _WT.WanTi2vEngine.prequalification_env)
    assert (_FW.FastWan8gbEngine.max_frames_env
            != _WT.WanTi2vEngine.max_frames_env)


# --------------------------------------------------------------------------- #
# the LoRA: preflight, identity, lifecycle
# --------------------------------------------------------------------------- #
def test_lora_is_in_loader_names_and_aux_preflight(monkeypatch):
    eng = _engine(monkeypatch)
    assert eng._loader_names()["lora"] == _FW.FASTWAN_LORA_NAME
    labels = [row[0] for row in eng._aux_loader_files()]
    assert "LoRA/FastWan" in labels
    # the incumbent's two aux rows survive
    assert "CLIP/umt5" in labels and "VAE" in labels


def test_hoist_graph_carries_the_lora_patched_model(monkeypatch):
    """Both patchers must be hoisted and tracked: _detach_patchers walks
    prepared["patchers"], so an untracked patcher is VRAM nothing reclaims."""
    eng = _engine(monkeypatch)
    g = eng._hoist_graph(eng._loader_names())
    assert set(g) == {"unet", "lora"}
    assert tuple(g["lora"]["inputs"]["model"]) == ("unet", 0)
    assert g["lora"]["inputs"]["lora_name"] == _FW.FASTWAN_LORA_NAME
    assert g["lora"]["inputs"]["strength_model"] == 1.0
    assert _FW.FastWan8gbEngine._SESSION_NODES == ("unet", "lora")


def test_lora_reaches_session_identity(monkeypatch):
    """A swapped or resized LoRA must open a NEW beat session, or a segment would
    reuse a handle its own graph no longer knows how to have produced."""
    eng = _engine(monkeypatch)
    ident = repr(eng.session_identity())
    assert _FW.FASTWAN_LORA_NAME in ident


def test_node_candidates_include_the_chain(monkeypatch):
    cands = _engine(monkeypatch)._node_candidates()
    assert cands["lora"] == ("LoraLoaderModelOnly",)
    assert cands["sigmas"] == ("ManualSigmas",)
    assert cands["sampler"] == ("SamplerCustom",)
    assert cands["dmdsampler"] == ("OTR_DMDRestartSamplerSelect",)


# --------------------------------------------------------------------------- #
# the incumbent is untouched
# --------------------------------------------------------------------------- #
def test_incumbent_graph_still_uses_ksampler(monkeypatch):
    for k in ("OTR_WAN_TI2V_LOADER", "OTR_WAN_TI2V_UNET_NAME", "OTR_WAN_TI2V_CKPT"):
        monkeypatch.delenv(k, raising=False)
    g = _WT.WanTi2vEngine()._build_graph(
        {"text_prompt": "x", "canvas": {"w": 832, "h": 480}},
        "init.png", {"seed": 7}, 81, 832, 480)
    assert "ksampler" in g and "sampler" not in g
    assert g["ksampler"]["inputs"]["seed"] == 7
    assert tuple(g["modelsampling"]["inputs"]["model"]) == ("unet", 0)
    assert tuple(g["vaedecode"]["inputs"]["samples"]) == ("ksampler", 0)


def test_incumbent_declares_no_canvas_this_build():
    """r4 CUT the incumbent canvas declaration from this promotion: FastWan owns
    its canvas, and wan_ti2v's is a separate measured campaign."""
    assert getattr(_WT.WanTi2vEngine, "render_canvas", None) is None


@pytest.mark.parametrize("attr", ["frame_contract", "target_fps", "family"])
def test_substrate_is_inherited_unchanged(attr):
    assert getattr(_FW.FastWan8gbEngine, attr) == getattr(_WT.WanTi2vEngine, attr)


# --------------------------------------------------------------------------- #
# licence + profile
# --------------------------------------------------------------------------- #
def test_commercial_clean_is_false_and_not_inherited():
    """The LoRA actually loaded is a Kijai extraction from a repo with NO
    repo-level licence file. "Both upstreams say apache-2.0" and "this artifact
    is notice-compliant" are different claims; commercial_clean reads as the
    second. Under-claim until the notice chain resolves."""
    assert _FW.FastWan8gbEngine.commercial_clean is False
    assert _WT.WanTi2vEngine.commercial_clean is True     # incumbent unchanged


def _profile():
    import json
    import pathlib
    p = (pathlib.Path(__file__).resolve().parent.parent
         / "config" / "profiles" / "otr_8gb_fastwan.json")
    return json.loads(p.read_text(encoding="utf-8"))


def test_profile_passes_the_real_shape_validator():
    """Reading keys out of the JSON is NOT proof the profile loads.

    My first draft carried a top-level ``notes`` block for rationale; every
    key-reading test below passed while ``load_profile`` rejected the file
    outright (the shape validator fails closed on unknown top-level keys). Only
    a dry run caught it. This test is that dry run, made cheap."""
    from nodes._otr_shared import capability_profiles as _CP
    prof = _CP.load_profile("otr_8gb_fastwan")
    assert prof["id"] == "otr_8gb_fastwan"


def test_profile_routes_every_video_role_to_fastwan():
    prof = _profile()
    assert prof["id"] == "otr_8gb_fastwan"
    for role in ("announcer_visual", "music_visual", "character_visual"):
        assert prof["role_overrides"][role] == "fastwan_8gb"
    assert prof["slot_overrides"]["video_render_engine"] == "fastwan_8gb"


def test_profile_canvas_agrees_with_the_declaration():
    """THE DRIFT GUARD the O1 judgment says the profile channel owes.

    The ADAPTER's declaration is the authority (render_driver applies it last);
    the profile is not. This asserts they agree, so a profile edit that silently
    disagrees with what actually renders is caught here rather than in a ledger."""
    prof = _profile()
    w, h = _FW.FastWan8gbEngine.render_canvas
    assert (prof["render"]["canvas_w"], prof["render"]["canvas_h"]) == (w, h)
    assert prof["launch"]["env"]["OTR_VIDEO_LANDSCAPE_CANVAS"] == "%dx%d" % (w, h)


def test_profile_ships_at_the_floor_not_the_bench_rung():
    """81 frames is a BENCH result, and a bench cell never qualifies an engine.
    The cap moves only after a canonical render proves the rung."""
    prof = _profile()
    assert prof["video"]["max_render_frames"] == 17
    assert prof["launch"]["env"]["OTR_FASTWAN_8GB_MAX_FRAMES"] == "17"
    # ...through its OWN key, never the incumbent's.
    assert "OTR_WAN_TI2V_MAX_FRAMES" not in prof["launch"]["env"]


def test_profile_preflight_requires_the_lora():
    required = _profile()["preflight"]["required_models"]
    assert "fastwan-2.2-5b-lora" in required
    assert set(_REG.CAPABILITIES["fastwan_8gb"]["model_requirements"]) <= set(required)
