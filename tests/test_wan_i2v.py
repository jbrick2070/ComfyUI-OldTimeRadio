"""LANE 1 -- the wan_i2v 14B I2V lane, which could not start until today.

The first lane of the 21-lane video transplant
(`docs/2026-08-10-FINAL-QA-video-build-corpus.md`), and deliberately the first:
it was BROKEN, so the work is unambiguous; its blast radius is nil because it
ships dark (`default_roles = ()`); and it exercises three of the four seed
lessons at once. Whatever shape this file takes is the template the later lanes
copy, so it is organised by PREFLIGHT GATE rather than by function.

There was no `tests/test_wan_i2v.py` at all before this -- no graph test, no
wiring test, no render-path test, no pin on any declaration, on a 14B lane.
That absence is why a wrong checkpoint default survived long enough to be
discovered by a lane audit rather than by CI.

CPU-safe: no CUDA, no model loads, no renders. Every assertion is a read of a
declaration, a resolver, or a config file.
"""

from __future__ import annotations

import json
import os
import pathlib

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_video_director as vd
from nodes._otr_shared import public_engines as pub
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines import wan_shared as ws
from nodes._otr_video_engines.eng_wan_i2v import (
    _I2V_DEFAULT_UNET, WanI2VEngine)
from nodes._otr_video_engines.registry import (
    EngineUnusable, EngineUsabilityReason)

LANE = "wan_i2v"
PUBLIC = "wan22_high_i2v"
DECLARED_CANVAS = (832, 480)
LANDSCAPE = (1472, 832)
REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture()
def engine():
    return vreg.get_engine(LANE)


# ---------------------------------------------------------------------------
# GATE 1 -- weights resolve (lesson L1: the defect that killed this lane)
# ---------------------------------------------------------------------------

def test_the_default_unet_names_the_artifact_that_is_actually_installed():
    """THE LANE-1 BUG, pinned as a literal.

    The default used to be `checkpoints/wan2.2-i2v.safetensors` -- a
    placeholder name in a placeholder category that exists on no box this
    project has run on. `assert_usable` raised MISSING_MODEL before the first
    forward and the lane shipped dead while looking registered.
    """
    assert _I2V_DEFAULT_UNET == (
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors")
    assert "checkpoints" not in WanI2VEngine()._ckpt_path()
    assert "diffusion_models" in WanI2VEngine()._ckpt_path(), (
        "the Wan weights on this project live under diffusion_models, like "
        "the sibling wan_ti2v lane's default -- not under checkpoints")


def test_the_lane_is_named_wan_2_2_because_that_is_what_it_loads():
    """A public id is a claim about the model, and this one was mislabelled
    2.1 once already (registry.py's CAPABILITIES comment records the S5
    correction FROM the stale wan2.1 label TO wan2.2-i2v)."""
    from nodes._otr_video_engines.eng_wan_i2v import RECIPE_WAN_I2V
    assert "wan22" in RECIPE_WAN_I2V
    assert "wan2.2" in _I2V_DEFAULT_UNET
    assert vreg.CAPABILITIES[LANE]["model_requirements"] == ["wan2.2-i2v"]
    assert PUBLIC.startswith("wan22_")


def test_weight_resolution_does_not_stop_at_one_hardcoded_location(engine):
    """G1.1 / L1. `_installed` must consult more than a single joined path.

    Asserted as BEHAVIOUR against a models root that is not the comfy root: a
    box whose weights live elsewhere -- which is this box -- must still resolve.
    """
    assert hasattr(engine, "_resolve_model_file")
    assert engine._installed() is True, (
        "the installed 14B I2V UNET did not resolve; weight resolution is "
        "back to knowing exactly one location (lesson L1)")


def test_a_models_root_override_is_honoured(tmp_path, monkeypatch):
    """The configured-models-root probe is the third and last one, and it is
    what makes the answer true OFF the ComfyUI runtime (in this suite, in the
    preflight matrix, in any tool that asks 'is this lane installed?')."""
    monkeypatch.delenv("OTR_WAN_I2V_CKPT", raising=False)
    monkeypatch.delenv("OTR_WAN_I2V_UNET_DIR", raising=False)
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path))
    assert ws.configured_models_root() == str(tmp_path)
    engine = WanI2VEngine()
    assert engine._installed() is False, (
        "an empty models root must read as NOT installed -- otherwise the "
        "probe is not probing anything")
    staged = tmp_path / "diffusion_models"
    staged.mkdir()
    (staged / _I2V_DEFAULT_UNET).write_bytes(b"unet-placeholder")
    assert engine._installed() is True


def test_a_missing_weight_fails_closed_by_name(tmp_path, monkeypatch):
    """G1.2. A missing weight is a NAMED EngineUnusable from assert_usable,
    never a swallowed import and never a mid-graph death."""
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path))
    monkeypatch.setenv("OTR_WAN_I2V_CKPT", str(tmp_path / "absent.safetensors"))
    with pytest.raises(EngineUnusable) as exc:
        WanI2VEngine().assert_usable(host_caps={}, profile={})
    assert exc.value.reason is EngineUsabilityReason.MISSING_MODEL
    message = str(exc.value)
    assert _I2V_DEFAULT_UNET in message or "absent.safetensors" in message
    for route in ("OTR_WAN_I2V_UNET_DIR", "OTR_WAN_I2V_CKPT", "folder_paths"):
        assert route in message, (
            "the refusal must name every route an operator could use to fix "
            "it; %r is missing from: %s" % (route, message))


def test_the_vestigial_enable_flag_gates_nothing(engine, monkeypatch):
    """The docstring used to say this lane was 'gated behind
    OTR_ENABLE_WAN_I2V', which sent a reader looking for a switch that does not
    exist while the real blocker was the checkpoint default. The registry IS
    the menu; there is no flag gate."""
    monkeypatch.delenv("OTR_ENABLE_WAN_I2V", raising=False)
    assert engine.requires_flag is None
    assert LANE in vreg.all_engine_names()
    assert engine.assert_usable(host_caps={}, profile={}) == LANE


# ---------------------------------------------------------------------------
# GATE 2 -- canvas truth (lesson L2). The two drift guards ltx_video has and
# this lane did not.
# ---------------------------------------------------------------------------

def test_the_engine_declares_its_own_render_canvas(engine):
    """Beside the frame contract and the aspect -- the same kind of fact,
    readable without loading anything."""
    assert tuple(engine.render_canvas) == DECLARED_CANVAS
    assert rd.declared_render_canvas(LANE) == DECLARED_CANVAS


def test_the_declared_canvas_is_32_legal_on_both_axes():
    """Asserted as arithmetic rather than as the literal, so the REASON
    survives a future re-tuning: /32 is the latent grid."""
    width, height = DECLARED_CANVAS
    assert width % 32 == 0 and height % 32 == 0
    assert (width // 32, height // 32) == (26, 15)


def test_wan_i2v_DECLARES_its_canvas_so_the_ENV_cannot_move_it(monkeypatch):
    """The wan_i2v twin of test_engine_contract_roster's ltx_video guard.

    The canvas also arrives through OTR_LTX_RENDER_CANVAS, read at render time
    in build_request_from_shot, and a test over shipped profiles cannot see a
    variable set at BOOT. `declared_render_canvas` is applied LAST on purpose,
    so declaring it is what actually closes the channel.
    """
    monkeypatch.setenv("OTR_LTX_RENDER_CANVAS", "1472x832")
    assert rd.declared_render_canvas(LANE) == DECLARED_CANVAS, (
        "the declaration must not follow an env canvas -- that is the channel "
        "the declaration exists to overrule")


def test_the_declared_canvas_and_the_shipped_profiles_AGREE():
    """The second drift guard. The profile channel keeps exactly one job now
    that the declaration overrules it: it must not be allowed to say something
    different in the config an operator reads."""
    checked = 0
    for path in sorted((REPO / "config" / "profiles").glob("*.json")):
        raw = path.read_text(encoding="utf-8")
        if '"%s"' % LANE not in raw:
            continue
        render = (json.loads(raw).get("render") or {})
        w, h = render.get("canvas_w"), render.get("canvas_h")
        if w is None or h is None:
            continue
        checked += 1
        assert (int(w), int(h)) == DECLARED_CANVAS, (
            "%s renders %s at %sx%s but the adapter declares %s"
            % (path.name, LANE, w, h, DECLARED_CANVAS))
    assert checked >= 1, "expected at least the otr_w45_wan_i2v profile"


def test_the_LEDGER_canvas_cannot_displace_the_declaration():
    """A hostile ledger stamp -- including the landscape default this lane used
    to fall through to -- must not move the render."""
    for hostile in ({"w": 1472, "h": 832}, {"w": 1920, "h": 1080}, None):
        ledger = {
            "episode_id": "ep_lane1",
            "images": {"images": [{"beat_id": "b001", "kind": "scene_beat",
                                   "path": "C:/tmp/scene_b001.png"}]},
            "video": {
                "fps": 25,
                "canonical_canvas": dict(hostile) if hostile else None,
                "shots": [{"shot_id": "shot_b001", "role": "character_video",
                           "group_id": "grp_character_video",
                           "engine_id": LANE, "family": "",
                           "target_frame_count": 33}],
            },
        }
        req = rd.build_request_from_shot(ledger["video"]["shots"][0], ledger)
        assert (req["canvas"]["w"], req["canvas"]["h"]) == DECLARED_CANVAS, (
            "ledger stamp %r displaced the declaration" % (hostile,))


# ---------------------------------------------------------------------------
# GATE 3 -- contract matches runtime (lesson L3)
# ---------------------------------------------------------------------------

def test_the_frame_contract_is_declared_at_the_canvas_rate(engine):
    from nodes._otr_video_engines import frame_contract as fc
    contract = fc.frame_contract_for(engine)
    assert contract.native_fps == 25 and engine.target_fps == 25
    assert (contract.min_frames, contract.max_frames) == (33, 177)
    assert contract.quantum == 4, "Wan's 4n+1 latent stride"
    assert contract.is_legal_length(contract.max_frames)
    assert contract.continuity == "strict_first_frame", (
        "the graph wires LoadImage -> WanImageToVideo start_image, which IS a "
        "hard first-frame lock, so successors may chain")


def test_only_the_f33_rung_is_claimed_as_measured():
    """L7. The contract's f177 ceiling is MODEL-LEGAL. It is not
    machine-qualified, and the manifest must not imply that it is."""
    manifest = json.loads(
        (REPO / "docs" / "evidence" / "video_evidence_manifest.json")
        .read_text(encoding="utf-8"))
    rows = [e for e in manifest["entries"] if e["lane"] == LANE]
    assert rows, "the lane has no evidence row"
    assert any("f33" in row["envelope_key"] for row in rows)
    assert any("f33" in row["note"] for row in rows), (
        "the note must state the RUNG the number was measured at -- a warm "
        "peak without its rung is the corpus defect this manifest exists for")
    assert LANE in manifest["admission_unenforced"], (
        "nothing refuses an over-budget render on this lane, so its receipts "
        "must say admission NOT enforced in words")


# ---------------------------------------------------------------------------
# GATE 7 -- public surface
# ---------------------------------------------------------------------------

def test_exactly_one_live_menu_option_resolves_to_this_lane():
    assert vd.exact_menu_option_for(LANE) == "%s (16:9)" % PUBLIC


def test_every_spelling_of_this_lane_still_resolves():
    """Saved graphs, profiles and the spec's own naming table must all land on
    the internal id. A rename that breaks a saved value is a rename that costs
    the operator an afternoon."""
    for spelling in (LANE, PUBLIC, "%s (16:9)" % PUBLIC, "%s (16:9)" % LANE,
                     "wan21_high_i2v"):
        assert pub.resolve_engine_id(spelling) == LANE, spelling


def test_the_spec_naming_table_string_resolves_but_never_shows(engine):
    """`wan21_high_i2v` is what the spec's table prints for this lane, and the
    lane is Wan 2.2. The live menu states 2.2; the spec's string is a legacy
    alias so it is not a dead end. Flagged for the operator, not silently
    dropped."""
    assert pub._LEGACY_ENGINE_ALIASES["wan21_high_i2v"] == LANE
    menu = [o for o in vd._video_model_combo() if o != vd.ADD_CUSTOM]
    assert not any("wan21" in option for option in menu)
    assert "wan21_high_i2v" not in pub._PUBLIC_ENGINES


def test_the_lane_still_ships_dark(engine):
    """`default_roles = ()` is why lane 1 has a blast radius of nil: nothing
    selects it unless a profile's role_overrides does. If this ever gains a
    default role it stops being a safe first lane."""
    assert tuple(engine.default_roles) == ()
    assert tuple(engine.required_inputs) == ("init_image",)


def test_the_still_plan_is_declared_and_valid(engine):
    from nodes._otr_shared import still_plan_helpers as sph
    assert sph.engine_has_still_plan(engine)
    sph.validate_still_plan(engine.still_plan)


# ---------------------------------------------------------------------------
# The lane's profile and its boot contract
# ---------------------------------------------------------------------------

def _profile():
    return json.loads(
        (REPO / "config" / "profiles" / "otr_w45_wan_i2v.json")
        .read_text(encoding="utf-8"))


def test_the_lane_is_selected_by_role_overrides_not_by_a_node_87_edit():
    """The engine ships dark, so the smoke selects it through a profile -- the
    documented mechanism -- rather than by hand-editing the canonical
    workflow's widget values."""
    profile = _profile()
    assert profile["role_overrides"]["character_visual"] == LANE
    assert profile["slot_overrides"]["video_render_engine"] == LANE


def test_the_boot_contract_rides_launch_env_not_extra_args():
    """L6. `launch.extra_args` is written only into a markdown documentation
    string; no launcher turns it into argv. `launch.env` is the live channel,
    consumed by _otr_soak_server_launch.cmd. A boot pin in the dead channel is
    a pin that clamps nothing."""
    launch = _profile()["launch"]
    env = launch.get("env") or {}
    assert env.get("OTR_WAN_I2V_CKPT"), (
        "the lane's weight pin must live in launch.env")
    assert env["OTR_WAN_I2V_CKPT"].endswith(_I2V_DEFAULT_UNET)
    assert not launch.get("extra_args"), (
        "extra_args is a documentation-only channel; anything load-bearing "
        "placed there is silently ignored at boot")


def test_the_env_pin_and_the_code_default_name_the_same_artifact():
    """Two channels naming one weight must not be allowed to disagree -- that
    is how a preflight passes against one file while the loader loads another.
    """
    pinned = _profile()["launch"]["env"]["OTR_WAN_I2V_CKPT"]
    assert os.path.basename(pinned) == _I2V_DEFAULT_UNET
