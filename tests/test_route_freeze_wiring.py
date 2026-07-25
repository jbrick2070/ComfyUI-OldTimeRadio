"""Chunk 1b -- the route freeze, wired end to end across BOTH branches.

``workflows/otr_canonical.json`` wires node 87 (OTR_VideoDirector) to node 88
(OTR_ImageDirector) and to node 90 (OTR_ShotLock). Node 88 feeds node 89
(MetaBrief) and node 91 (the dispatcher). There is NO 89 -> 90 edge: the image
branch and the video branch are INDEPENDENT and only reconverge at node 91.
Node 87 is therefore the unique common ancestor, and the only place a route
freeze can inform both branches.

These tests pin that wiring, the environment guard, and -- most importantly --
the decapitation regression the freeze exists to fix.
"""

from __future__ import annotations

import json

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the video registry
from nodes import otr_image_director as idir
from nodes import otr_shot_lock as sl
from nodes import otr_video_director as vd
from nodes._otr_shared import route_freeze as rf
from nodes._otr_video_engines import render_driver as rd


def _resolved(announcer, music, character):
    return {
        "announcer_video_model": {"engine_id": announcer},
        "music_video_model": {"engine_id": music},
        "character_video_model": {"engine_id": character},
    }


def _direct(monkeypatch, announcer="humo_1.7B", music="ltx_audio_in",
            character="wan_i2v", **env):
    """Drive the real OTR_VideoDirector and return its parsed policy."""
    for key in ("OTR_FORCE_ENGINE_MAP", "OTR_ENABLE_HUMO_HOSTS"):
        monkeypatch.delenv(key, raising=False)
    for key, val in env.items():
        monkeypatch.setenv(key, val)
    (raw,) = vd.OTRVideoDirector().direct(
        announcer_video_model=announcer,
        music_video_model=music,
        character_video_model=character,
        announcer_image_model="flux_gen1",
        music_image_model="flux_gen1",
        character_image_model="flux_gen1",
        fps=25, canvas_w=832, canvas_h=480,
        seed_mode="request_hash", request_seed=0,
    )
    return json.loads(raw)


# ---------------------------------------------------------------------------
# THE REGRESSION THIS CHUNK EXISTS FOR
# ---------------------------------------------------------------------------

def test_redirected_bookend_gets_a_WIDE_still_not_a_decapitated_portrait(monkeypatch):
    """THE DECAPITATION BUG. Live under the DEFAULT environment.

    Pick a PORTRAIT HuMo for announcer_visual. With OTR_ENABLE_HUMO_HOSTS unset
    (the default), the radio-is-host guard redirects that beat to the WIDE
    ``ltx_audio_in`` console. Before chunk 1b the aspect map was derived from
    the PICKED portrait engine, so an 832x1216 portrait still was minted and the
    wide render centre-cropped it. ``eng_ltx_av.py:345-347`` records the outcome
    verbatim: "the director defaulted to a 832x1216 PORTRAIT still that the wide
    render then centre-cropped, lopping the subject's head off."

    The aspect must now describe the engine that ACTUALLY renders.
    """
    policy = _direct(monkeypatch, announcer="humo_1.7B")

    # The pick is unchanged and still recorded...
    assert policy["video_models"]["announcer_video_model"]["engine_id"] == "humo_1.7B"
    # ...but the route is frozen to the engine that will really run...
    assert policy["effective_video_models"]["announcer_visual"] == \
        rd._NEVER_HUMO_REDIRECT_ENGINE
    # ...and the still is sized for THAT engine, not for the portrait pick.
    assert policy["aspects"]["announcer_visual"] == "wide"


def test_hosts_on_keeps_the_portrait_still_for_a_portrait_humo(monkeypatch):
    """The other side of the same contract: with the toggle ON the redirect is
    a no-op, the portrait HuMo really does render, and a portrait still is
    correct. A fix that made everything wide would be a different bug."""
    policy = _direct(monkeypatch, announcer="humo_1.7B", OTR_ENABLE_HUMO_HOSTS="1")
    assert policy["effective_video_models"]["announcer_visual"] == "humo_1.7B"
    assert policy["aspects"]["announcer_visual"] == "portrait"


def test_character_video_is_never_redirected(monkeypatch):
    """character_video is outside the guard, so a HuMo pick there survives."""
    policy = _direct(monkeypatch, character="humo_1.7B")
    assert policy["effective_video_models"]["character_video"] == "humo_1.7B"


# ---------------------------------------------------------------------------
# The stamps
# ---------------------------------------------------------------------------

def test_policy_carries_both_maps_and_the_snapshot(monkeypatch):
    policy = _direct(monkeypatch)
    assert isinstance(policy["effective_video_models"], dict)
    assert policy["routing_env_snapshot"] == {
        "force_engine_map": "", "enable_humo_hosts": False}
    # picked and effective are stamped SEPARATELY so a downstream mismatch is
    # replayable rather than a guess.
    assert "video_models" in policy and "effective_video_models" in policy


def test_force_map_is_frozen_into_the_effective_map(monkeypatch):
    policy = _direct(monkeypatch, OTR_FORCE_ENGINE_MAP="*=viz_camera")
    assert set(policy["effective_video_models"].values()) == {"viz_camera"}
    assert policy["routing_env_snapshot"]["force_engine_map"] == "*=viz_camera"


def test_malformed_force_map_is_terminal_at_the_director(monkeypatch):
    """Fail before the image phase, not after ~10 minutes of minting."""
    with pytest.raises(rf.RouteFreezeError):
        _direct(monkeypatch, OTR_FORCE_ENGINE_MAP="garbage-no-equals")


def test_image_director_forwards_both_keys(monkeypatch):
    """Its payload is built KEY BY KEY, so a new upstream key is NOT forwarded
    automatically -- node 89 and node 91 would silently never see the freeze."""
    video_policy = _direct(monkeypatch, announcer="humo_1.7B")
    (raw,) = idir.OTRImageDirector().direct(
        announcer_granularity="per_object",
        music_granularity="per_object",
        character_granularity="per_object",
        fresh_cap=15, seed_mode="request_hash", request_seed=0,
        video_policy_json=json.dumps(video_policy),
    )
    image_policy = json.loads(raw)
    assert image_policy["effective_video_models"] == \
        video_policy["effective_video_models"]
    assert image_policy["routing_env_snapshot"] == \
        video_policy["routing_env_snapshot"]
    # and the derived aspect travels with it
    assert image_policy["aspects"]["announcer_visual"] == "wide"


# ---------------------------------------------------------------------------
# ShotLock consumes the freeze
# ---------------------------------------------------------------------------

def _plan(policy, role="announcer_visual"):
    beats = [{"beat_id": "b001", "role": role, "char_id": "", "dur_s": 2.0}]
    budget = {"per_beat": {"b001": 50}, "total_frames": 50, "warnings": []}
    return sl.build_execution_plan(beats, budget, {}, policy, None)


def test_groups_and_shots_are_minted_from_the_frozen_map(monkeypatch):
    """Groups, shots and the cast-time preflight must all see ONE engine.

    Before 1b the groups and shot rows carried the PICKED engine while the
    preflight re-derived the effective one through its own private mirror.
    """
    policy = _direct(monkeypatch, announcer="humo_1.7B")
    groups, shots = _plan(policy)
    assert shots[0]["engine_id"] == rd._NEVER_HUMO_REDIRECT_ENGINE
    assert groups[0]["engine_id"] == rd._NEVER_HUMO_REDIRECT_ENGINE
    # the invariant the r4 panel asked for, stated directly
    assert groups[0]["engine_id"] == shots[0]["engine_id"]


def test_a_policy_without_the_freeze_falls_back_to_the_picked_map():
    """A pre-1b or hand-built policy keeps its old meaning exactly."""
    policy = {"policy_version": 2,
              "video_models": _resolved("wan_i2v", "wan_i2v", "wan_i2v")}
    _groups, shots = _plan(policy, role="character_video")
    assert shots[0]["engine_id"] == "wan_i2v"


# ---------------------------------------------------------------------------
# The environment guard
# ---------------------------------------------------------------------------

def _lock_with(policy, monkeypatch=None):
    ledger = {
        "episode_id": "ep_guard",
        "meta": {"episode_title": "T", "audio_revision": "1"},
        "script": {"lines": []},
    }
    return sl.OTRShotLock().lock(json.dumps(ledger),
                                 video_policy_json=json.dumps(policy))


def test_shotlock_rejects_an_environment_that_moved_after_the_freeze(monkeypatch):
    """The frozen route stops describing reality the moment the env changes.

    A cached upstream node, an operator exporting the force map between nodes,
    or a headless leg submitted to a differently-booted server all produce this
    state, and every still and prompt planned from the stale map is planned for
    an engine that will not run. Terminal, before any GPU or image time.
    """
    policy = _direct(monkeypatch)          # frozen with a CLEAN environment
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=viz_camera")   # ...then it moves
    with pytest.raises(ValueError, match="routing environment CHANGED"):
        _lock_with(policy)


def test_shotlock_accepts_a_matching_environment(monkeypatch):
    policy = _direct(monkeypatch, OTR_FORCE_ENGINE_MAP="*=viz_camera")
    # environment unchanged since the freeze -> no raise from the guard
    out = _lock_with(policy)
    assert out[3].startswith("shot_lock:done:")


def test_shotlock_ignores_a_policy_with_no_snapshot():
    """Pre-1b / hand-built ledgers are not retro-fitted into the contract."""
    policy = {"policy_version": 2,
              "video_models": _resolved("wan_i2v", "wan_i2v", "wan_i2v")}
    out = _lock_with(policy)
    assert out[3].startswith("shot_lock:done:")


def test_ledger_stamps_both_maps_for_replay(monkeypatch):
    policy = _direct(monkeypatch, announcer="humo_1.7B")
    patched = json.loads(_lock_with(policy)[0])
    video = patched["video"]
    assert video["roles"] == policy["video_models"]                 # PICKED
    assert video["roles_effective"] == policy["effective_video_models"]
    assert video["routing_env_snapshot"] == policy["routing_env_snapshot"]


# ---------------------------------------------------------------------------
# IS_CHANGED -- both ends of the freeze invalidate together
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("node_cls", [vd.OTRVideoDirector, sl.OTRShotLock])
def test_is_changed_tracks_the_routing_environment(node_cls, monkeypatch):
    """Without this ComfyUI serves a cached node across an env change and the
    whole freeze is planned from state the operator already replaced."""
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    base = node_cls.IS_CHANGED()
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=viz_camera")
    forced = node_cls.IS_CHANGED()
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.setenv("OTR_ENABLE_HUMO_HOSTS", "1")
    hosts = node_cls.IS_CHANGED()
    assert len({base, forced, hosts}) == 3


def test_both_nodes_share_one_fingerprint(monkeypatch):
    """The two ends of the freeze must invalidate together, or one re-runs
    against an environment the other has already cached past."""
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=viz_camera")
    assert vd.OTRVideoDirector.IS_CHANGED() == sl.OTRShotLock.IS_CHANGED()
