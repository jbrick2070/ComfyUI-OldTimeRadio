"""Offline contracts for the confirmed 2026-07-23 video qualification bugs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes._otr_shared import capability_profiles as cp


REPO = Path(__file__).resolve().parents[1]


def test_otr_8gb_wan_profile_pins_low_vram_contract():
    profile = cp.load_profile("otr_8gb_wan")
    assert profile["render"]["canvas_w"] == 832
    assert profile["render"]["canvas_h"] == 480
    # frame_budget is the SOAK per-clip frame count and is a different knob
    # from the planner ceiling -- it stays 17, untouched.
    assert profile["render"]["frame_budget"] == 17
    assert profile["launch"]["env"] == {
        "OTR_VIDEO_LANDSCAPE_CANVAS": "832x480",
        "OTR_WAN_TI2V_MAX_FRAMES": "81",
    }

    variant = json.loads(
        (REPO / "workflows" / "variants" / "otr_8gb_wan.json").read_text(
            encoding="utf-8"))
    validator = next(
        node for node in variant["nodes"]
        if node.get("type") == "OTR_WorkflowValidator")
    render_batch = next(
        node for node in variant["nodes"]
        if node.get("type") == "OTR_VideoRenderBatch")
    env_recipe = json.loads(
        (REPO / "workflows" / "variants" / "otr_8gb_wan.env.json").read_text(
            encoding="utf-8"))
    assert render_batch["widgets_values"][3] == 17    # frame_budget
    assert validator["widgets_values"][4] == env_recipe["master_hash"]
    assert env_recipe["env"]["OTR_WAN_TI2V_MAX_FRAMES"] == "81"


# ---------------------------------------------------------------------------
# WAN 8GB low-VRAM LAUNCH CONTRACT (2026-07-24)
#
# The 2026-07-23 live failure (wan_8gb__lumina_image__media_archive): the engine
# received a 177-frame request while the cost model afforded 30 at the observed
# free VRAM, and died -- correctly, with no silent resize. The 17-frame ceiling
# existed only in the profile's `launch.env`, which a PRODUCTION episode leg can
# never see: the leg is submitted to an already-booted server. These pins hold
# the ceiling on its new channel, profile -> director widget -> ledger -> engine,
# and hold every other tier UNPINNED so no qualified lane changes behaviour.
# ---------------------------------------------------------------------------


def _canonical():
    return json.loads(
        (REPO / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))


def _node_of(graph, node_type):
    return next(n for n in graph["nodes"] if n.get("type") == node_type)


def test_wan_8gb_profile_declares_the_render_ceiling():
    profile = cp.load_profile("otr_8gb_wan")
    # 81 since lane 5 (2026-08-11). The 17 became a planner-narrowing live
    # bug the day wan_ti2v joined PLANNING_CAP_ENGINES.
    assert profile["video"]["max_render_frames"] == 81
    # The ceiling is NOT render.frame_budget: that is the soak/single harness
    # per-clip count (every 16GB tier declares 25 and must not be capped to
    # it), which is why the two numbers stopped agreeing and that is FINE.
    assert profile["render"]["frame_budget"] == 17
    assert "max_render_frames" not in cp.load_profile("16gb_full")["video"]
    assert "max_render_frames" not in cp.load_profile("otr_16gb_ltx_video")["video"]


def test_max_render_frames_is_optional_but_range_checked():
    profile = cp.load_profile("otr_8gb_wan")
    del profile["video"]["max_render_frames"]
    assert cp.validate_profile_shape(profile, source="<absent>")  # legal
    for bad in (999, -1, "17", True):
        profile["video"]["max_render_frames"] = bad
        with pytest.raises(cp.ProfileError, match="max_render_frames"):
            cp.validate_profile_shape(profile, source="<bad>")


def test_canonical_director_ships_the_ceiling_unpinned():
    from nodes.otr_video_director import OTRVideoDirector

    optional = OTRVideoDirector.INPUT_TYPES()["optional"]
    assert "max_render_frames" in optional
    spec = optional["max_render_frames"][1]
    assert (spec["default"], spec["min"], spec["max"]) == (0, 0, 240)
    widgets = _node_of(_canonical(), "OTR_VideoDirector")["widgets_values"]
    # APPENDED last (BUG-LOCAL-097): the 14 prior slots keep their positions.
    assert len(widgets) == 15
    assert widgets[:14][-2:] == ["cuda", "fp8_ok"]
    assert widgets[14] == 0


def test_applied_8gb_variant_pins_its_ceiling_and_other_tiers_stay_unpinned():
    """81, not 17 (lane 5, 2026-08-11). The 17 was a real low-VRAM launch
    contract when it was written, and it became a LIVE BUG on 2026-08-02
    without anyone touching it: `wan_ti2v` joined
    `frame_contract.PLANNING_CAP_ENGINES` and the adapter-side ping-pong
    that made a render cap harmless was ripped the same day. From that
    moment the pin narrowed the PLANNER, so every beat on this profile
    became a chain of 0.68-second segments -- the exact "pile of
    17-frame renders" the adapter's own comment warned about.

    81 = 4*20+1, on the shared ladder, and the value its sibling WAN and
    FastWan profiles already carry. NOT re-measured on real 8 GB
    hardware -- see the lane 5 receipt; if it does not fit there the
    answer is a MEASURED ceiling, not a return to a number that breaks
    the planner.
    """
    def _director_ceiling(stem):
        graph = json.loads(
            (REPO / "workflows" / "variants" / f"{stem}.json").read_text(
                encoding="utf-8"))
        return _node_of(graph, "OTR_VideoDirector")["widgets_values"][14]

    assert _director_ceiling("otr_8gb_wan") == 81
    for stem in ("otr_16gb_full", "otr_16gb_ltx_video", "otr_8gb_ltx",
                 "otr_nv40_12gb", "otr_cpu_floor"):
        assert _director_ceiling(stem) == 0, stem


def test_director_and_shot_lock_carry_the_ceiling_onto_the_ledger():
    from nodes.otr_video_director import OTRVideoDirector, ADD_CUSTOM
    from nodes.otr_shot_lock import OTRShotLock

    def _policy(ceiling):
        return OTRVideoDirector().direct(
            announcer_video_model=ADD_CUSTOM, music_video_model=ADD_CUSTOM,
            character_video_model=ADD_CUSTOM,
            announcer_image_model="Flux (gen 1)",
            music_image_model="Flux (gen 1)",
            character_image_model="Flux (gen 1)",
            fps=25, canvas_w=832, canvas_h=480,
            seed_mode="request_hash", request_seed=0,
            max_render_frames=ceiling,
        )[0]

    assert json.loads(_policy(17))["max_render_frames"] == 17
    assert json.loads(_policy(0))["max_render_frames"] == 0

    led = json.dumps({"cast": [], "lines": [], "meta": {}})
    locked = OTRShotLock().lock(led, audio_done="x",
                                video_policy_json=_policy(17), image_done="")
    assert json.loads(locked[0])["video"]["max_render_frames"] == 17
    # An empty/legacy policy stays unpinned rather than inventing a ceiling.
    legacy = OTRShotLock().lock(led, audio_done="x", video_policy_json="{}",
                                image_done="")
    assert json.loads(legacy[0])["video"]["max_render_frames"] == 0


def test_render_driver_hands_the_ceiling_to_every_adapter():
    from nodes._otr_video_engines import render_driver as rd

    pinned = rd.build_episode_render_policy(
        {"device_policy": "cuda", "dtype_policy": "fp8_ok",
         "max_render_frames": 17})
    assert pinned["max_render_frames"] == 17
    assert pinned["policy_version"] == 2
    assert rd.build_episode_render_policy({})["max_render_frames"] == 0
    assert rd.build_episode_render_policy(None)["max_render_frames"] == 0


def test_prepare_captures_the_episode_ceiling():
    from nodes._otr_video_engines import motion_common as mc

    class _StubMotionEngine(mc.MotionEngineBase):
        name = "stub_ceiling_probe"

        def load(self):
            self._loaded = True

    engine = _StubMotionEngine()
    assert engine.profile_max_render_frames() == 0        # never prepared
    prepared = engine.prepare(
        host_caps={}, profile={"policy_version": 2, "max_render_frames": 17},
        session_ctx={})
    try:
        assert engine.profile_max_render_frames() == 17
    finally:
        mc._GR.release(prepared["lease"])


def test_wan_ti2v_renders_the_8gb_contract_instead_of_failing_closed(monkeypatch):
    """The live 2026-07-23 leg, reproduced: a 177-frame beat at 832x480 with
    only ~30 frames affordable. UNPINNED it raises (correct -- no silent
    resize); with the tier ceiling on the ledger it renders 17 real frames,
    which the render then ping-pong-extends to the beat's full length.

    The row is QUALIFIED here (2026-08-13). This test's subject is the TIER
    CEILING, and the ceiling is only interesting against a budget that would
    otherwise refuse -- so the refusal has to be reachable for the contrast to
    mean anything. In production no row is qualified and the unpinned call
    returns 177 instead of raising; that is pinned in test_wan_ti2v.py, not
    here, because it is a different claim.
    """
    from nodes._otr_video_engines import motion_common as mc
    from nodes._otr_video_engines.eng_wan_ti2v import WanTi2vEngine

    monkeypatch.delenv("OTR_WAN_TI2V_MAX_FRAMES", raising=False)
    monkeypatch.setattr(mc, "QUALIFIED_COST_ROWS", frozenset({"wan_ti2v"}))
    monkeypatch.setattr(mc, "free_vram_mb", lambda: 10365.0)   # affords ~30
    engine = WanTi2vEngine()

    with pytest.raises(mc.MotionBudgetError):
        engine._floor_length(177, 832, 480)

    engine._active_profile = {"policy_version": 2, "max_render_frames": 17}
    assert engine._floor_length(177, 832, 480) == 17
    # The ceiling caps the RENDER, never the beat: a short beat is unaffected.
    assert engine._floor_length(17, 832, 480) == 17


def test_wan_ti2v_ceiling_precedence_and_no_change_when_unpinned(monkeypatch):
    from nodes._otr_video_engines import motion_common as mc
    from nodes._otr_video_engines.eng_wan_ti2v import WanTi2vEngine

    monkeypatch.setattr(mc, "free_vram_mb", lambda: 40000.0)   # affords 177+
    engine = WanTi2vEngine()

    # Unpinned tier (every shipped profile but the 8GB Wan one): unchanged.
    monkeypatch.delenv("OTR_WAN_TI2V_MAX_FRAMES", raising=False)
    engine._active_profile = {"policy_version": 2, "max_render_frames": 0}
    assert engine._floor_length(177, 832, 480) == 177

    # An explicit operator env pin outranks the profile-carried ceiling.
    engine._active_profile = {"policy_version": 2, "max_render_frames": 17}
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "49")
    assert engine._floor_length(177, 832, 480) == 49

    # A malformed pin/stamp falls back to the engine max -- never a crash.
    monkeypatch.setenv("OTR_WAN_TI2V_MAX_FRAMES", "not-an-int")
    engine._active_profile = {"max_render_frames": "nonsense"}
    assert engine._floor_length(177, 832, 480) == 177


def test_the_hand_kept_env_recipe_cannot_drift_from_its_profile():
    """`workflows/variants/*.env.json` is NOT generated by
    `scripts/build_variants.py` -- only four of them exist and they are kept by
    hand, so regenerating a variant silently leaves its env recipe behind.

    Lane 5 (2026-08-11) hit exactly that: the profile moved
    OTR_WAN_TI2V_MAX_FRAMES 17 -> 81, `build_variants` rewrote the graph and the
    launch recipe, and the paired env.json still said 17. Two files describing
    one launch, disagreeing. This asserts the agreement for every pair that
    exists, so the next mover is told rather than discovering it in a leg.
    """
    for env_path in sorted((REPO / "workflows" / "variants").glob("*.env.json")):
        stem = env_path.name[: -len(".env.json")]
        recipe = json.loads(env_path.read_text(encoding="utf-8"))
        profile = cp.load_profile(stem)
        for key, value in (profile["launch"].get("env") or {}).items():
            assert recipe["env"].get(key) == value, (
                "%s carries %s=%r while config/profiles/%s.json says %r -- two "
                "files describing one launch must not disagree"
                % (env_path.name, key, recipe["env"].get(key), stem, value))


def test_the_hand_kept_env_recipe_carries_the_LIVE_master_hash():
    """The other half of the same drift (lane 5). `build_variants` stamps a
    fresh master_hash into the variant's validator node every time it runs; the
    hand-kept env.json carries a copy. When the two disagree, the recipe is
    describing a graph that no longer exists."""
    variants = REPO / "workflows" / "variants"
    for env_path in sorted(variants.glob("*.env.json")):
        stem = env_path.name[: -len(".env.json")]
        variant = json.loads((variants / (stem + ".json")).read_text(
            encoding="utf-8"))
        validator = next(n for n in variant["nodes"]
                         if n.get("type") == "OTR_WorkflowValidator")
        recipe = json.loads(env_path.read_text(encoding="utf-8"))
        assert recipe["master_hash"] == validator["widgets_values"][4], (
            "%s describes a graph that no longer exists" % env_path.name)
