"""Offline contracts for the confirmed 2026-07-23 video qualification bugs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes._otr_shared import capability_profiles as cp


REPO = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# THE PROFILE-CARRIED RENDER CEILING (2026-07-24)
#
# Born of the 2026-07-23 live failure (wan_8gb__lumina_image__media_archive): the
# engine received a 177-frame request while the cost model afforded 30 at the
# observed free VRAM, and died -- correctly, with no silent resize. The ceiling
# existed only in the profile's `launch.env`, which a PRODUCTION episode leg can
# never see: the leg is submitted to an already-booted server. The tier that
# needed it is retired; the channel it built -- profile -> director widget ->
# ledger -> engine -- is what these pins hold, together with every shipped tier
# staying UNPINNED so no qualified lane changes behaviour.
# ---------------------------------------------------------------------------


def _canonical():
    return json.loads(
        (REPO / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))


def _node_of(graph, node_type):
    return next(n for n in graph["nodes"] if n.get("type") == node_type)


#: The 8 GB video row: `ltx_8gb` is a PLANNING_CAP_ENGINES lane, so a ceiling on
#: this row is the case the channel exists for. No matrix row pins one today,
#: so a pinned document is this row with the key added.
PINNED_BASE = "otr_8gb_video"
PINNED_CEILING = 81


def _pinned_variant(tmp_path, monkeypatch):
    """Emit `PINNED_BASE` with a render ceiling, through the real generator."""
    import copy
    import sys
    sys.path.insert(0, str(REPO / "scripts"))
    import build_variants as bv

    doc = copy.deepcopy(cp.load_profile(PINNED_BASE))
    doc["id"] = "otr_8gb_ceiling_probe"
    doc["video"]["max_render_frames"] = PINNED_CEILING
    (tmp_path / "otr_8gb_ceiling_probe.json").write_text(
        json.dumps(doc), encoding="utf-8")
    monkeypatch.setattr(
        bv, "load_profile",
        lambda pid: cp.load_profile(pid, profile_dir=str(tmp_path)))
    graph, _rel, _recipe = bv.build_variant("otr_8gb_ceiling_probe")
    return graph


def test_max_render_frames_is_optional_but_range_checked():
    profile = cp.load_profile(PINNED_BASE)
    profile["video"]["max_render_frames"] = PINNED_CEILING
    assert cp.validate_profile_shape(profile, source="<pinned>")  # legal
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
    # APPENDED last (BUG-LOCAL-097): the 12 prior slots keep their positions.
    # seed_mode/request_seed left the widget list on 2026-09-13 (write-only,
    # never read back), which is why this is 13, not the old 15.
    assert len(widgets) == 13
    # NOT ["cuda", "fp8_ok"]: the canonical carries the picks for whichever
    # machine is under test (operator ruling 2026-09-07). What this guard is for
    # is POSITION -- slots 10 and 11 are device_policy and dtype_policy -- so
    # assert membership in the live dropdowns and let the pick move.
    _opt = OTRVideoDirector.INPUT_TYPES()["optional"]
    _device_options = _opt["device_policy"][0]
    _dtype_options = _opt["dtype_policy"][0]
    assert widgets[10] in _device_options, (
        "slot 10 is device_policy; %r is not one of %r" % (widgets[10], _device_options))
    assert widgets[11] in _dtype_options, (
        "slot 11 is dtype_policy; %r is not one of %r" % (widgets[11], _dtype_options))
    assert widgets[12] == 0


def test_applied_8gb_variant_pins_its_ceiling_and_other_tiers_stay_unpinned(
        tmp_path, monkeypatch):
    """A ceiling a profile states lands in the director's LAST widget slot,
    and the shipped tiers carry none.

    The value that motivated this (81, lane 5, 2026-08-11) sat on a retired
    Wan tier; what stays true is the mechanism, so the pinned graph is the
    8 GB video row emitted with a ceiling added. A shipped tier reading
    anything but 0 here would narrow the planner on a lane nobody measured
    that way.
    """
    def _director_ceiling(graph):
        # max_render_frames is the LAST widget slot -- 12 since seed_mode/
        # request_seed left the list on 2026-09-13 (was 14 with them).
        return _node_of(graph, "OTR_VideoDirector")["widgets_values"][12]

    assert _director_ceiling(_pinned_variant(tmp_path, monkeypatch)) \
        == PINNED_CEILING
    # The 2026-09-13 curation renamed the shipping set; these are the tiers
    # that exist now and legitimately carry no planner ceiling.
    for stem in ("otr_16gb_low", "otr_16gb_video", "otr_8gb_low",
                 "otr_8gb_still"):
        path = REPO / "workflows" / f"{stem}.json"
        graph = json.loads(path.read_text(encoding="utf-8"))
        assert _director_ceiling(graph) == 0, stem


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
            max_render_frames=ceiling,
        )[0]

    assert json.loads(_policy(17))["max_render_frames"] == 17
    assert json.loads(_policy(0))["max_render_frames"] == 0

    led = json.dumps({"cast": [], "lines": [], "meta": {}})
    locked = OTRShotLock().lock(led, audio_done="x",
                                video_policy_json=_policy(17))
    assert json.loads(locked[0])["video"]["max_render_frames"] == 17
    # An empty/legacy policy stays unpinned rather than inventing a ceiling.
    legacy = OTRShotLock().lock(led, audio_done="x", video_policy_json="{}")
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


def test_the_hand_kept_env_recipe_cannot_drift_from_its_profile():
    """`workflows/*.env.json` is NOT generated by
    `scripts/build_variants.py` -- only four of them exist and they are kept by
    hand, so regenerating a variant silently leaves its env recipe behind.

    Lane 5 (2026-08-11) hit exactly that: the profile moved
    OTR_WAN_TI2V_MAX_FRAMES 17 -> 81, `build_variants` rewrote the graph and the
    launch recipe, and the paired env.json still said 17. Two files describing
    one launch, disagreeing. This asserts the agreement for every pair that
    exists, so the next mover is told rather than discovering it in a leg.
    """
    for env_path in sorted((REPO / "workflows").glob("*.env.json")):
        stem = env_path.name[: -len(".env.json")]
        recipe = json.loads(env_path.read_text(encoding="utf-8"))
        profile = cp.load_profile(stem)
        for key, value in (profile["launch"].get("env") or {}).items():
            assert recipe["env"].get(key) == value, (
                "%s carries %s=%r while matrix row %s says %r -- two "
                "files describing one launch must not disagree"
                % (env_path.name, key, recipe["env"].get(key), stem, value))


def test_the_hand_kept_env_recipe_carries_the_LIVE_master_hash():
    """The other half of the same drift (lane 5). `build_variants` stamps a
    fresh master_hash into the variant's validator node every time it runs; the
    hand-kept env.json carries a copy. When the two disagree, the recipe is
    describing a graph that no longer exists."""
    variants = REPO / "workflows"
    for env_path in sorted(variants.glob("*.env.json")):
        stem = env_path.name[: -len(".env.json")]
        variant = json.loads((variants / (stem + ".json")).read_text(
            encoding="utf-8"))
        validator = next(n for n in variant["nodes"]
                         if n.get("type") == "OTR_WorkflowValidator")
        recipe = json.loads(env_path.read_text(encoding="utf-8"))
        assert recipe["master_hash"] == validator["widgets_values"][4], (
            "%s describes a graph that no longer exists" % env_path.name)
