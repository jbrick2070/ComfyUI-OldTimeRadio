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
            character="wan_ti2v", **env):
    """Drive the real OTR_VideoDirector and return its parsed policy.

    The character slot named the Wan 2.2 **14B** ``wan_i2v`` until that engine
    was RETIRED on 2026-08-26 (it never fit the 14.5 GiB envelope). The
    director calls ``check_retired_engine`` at the selection boundary, so a
    retired pick now raises ``RetiredEngineError`` inside this fixture -- every
    test in this file died before reaching its own assertion, which is why one
    dead default produced twelve failures.

    It is the **5B** ``wan_ti2v`` now: the surviving local image-to-video lane,
    registered, and role-valid for ``character_video``. This is a live stand-in
    for the fixture, NOT a retargeted assertion -- nothing here asserts a
    14B-specific property of the character slot, and the one test that does
    care about the character route (``test_character_video_is_never_redirected``)
    passes its own engine in explicitly.
    """
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
    verbatim: OTR_VideoDirector "defaulted to a 832x1216 PORTRAIT still that the
    wide render then centre-cropped, lopping the subject's head off."

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


def test_talking_also_describes_the_EFFECTIVE_engine(monkeypatch):
    """`talking` carries the identical bug pattern `aspects` was fixed for.

    Added 2026-07-25 by QA: a mutation test showed that reverting `talking` to
    the PICKED engine left the ENTIRE suite green, so the second half of the
    same fix shipped unproven. The failure it guards: an announcer beat picks a
    lip-syncing HuMo that the radio-is-host guard redirects to the
    `ltx_audio_in` console, which does not lip-sync. If `talking` still
    described the picked engine, the image phase would keep minting
    face-forward "talking" portraits for an engine that never moves a mouth.
    """
    from nodes._otr_video_engines import registry as _vreg

    policy = _direct(monkeypatch, announcer="humo_1.7B")
    effective = policy["effective_video_models"]["announcer_visual"]
    assert effective == rd._NEVER_HUMO_REDIRECT_ENGINE

    def _wants_talking(engine_id):
        engine = _vreg.get_engine(engine_id)
        hook = getattr(engine, "wants_talking_prompt", None)
        return bool(hook()) if callable(hook) else False

    # The stamped flag must describe the engine that actually renders, not the
    # operator's pick -- whatever those two engines happen to declare.
    assert policy["talking"]["announcer_visual"] == _wants_talking(effective)
    if _wants_talking("humo_1.7B") != _wants_talking(effective):
        # ...and when they genuinely differ, the pick must NOT be what shipped.
        assert policy["talking"]["announcer_visual"] != _wants_talking("humo_1.7B")


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
    """A pre-1b or hand-built policy keeps its old meaning exactly.

    The engine id is a STAND-IN for "whatever the picked map happens to say".
    It read ``wan_i2v`` and moved to the live 5B ``wan_ti2v`` with the
    2026-08-26 retirement, so this file carries no dead engine name. Nothing
    was orphaned: this path never ran the retired-id guard in the first place
    (ShotLock does not re-validate a hand-built map), so the property under
    test -- the picked map wins when there is no freeze -- is unchanged.
    """
    policy = {"policy_version": 2,
              "video_models": _resolved("wan_ti2v", "wan_ti2v", "wan_ti2v")}
    _groups, shots = _plan(policy, role="character_video")
    assert shots[0]["engine_id"] == "wan_ti2v"


# ---------------------------------------------------------------------------
# The environment guard
# ---------------------------------------------------------------------------

def _lock_with(policy, monkeypatch=None):
    ledger = {
        "episode_id": "ep_guard",
        "meta": {"episode_title": "T"},
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
    # Same stand-in swap as above (wan_i2v retired 2026-08-26): the id is
    # incidental here, only the ABSENCE of routing_env_snapshot is asserted.
    policy = {"policy_version": 2,
              "video_models": _resolved("wan_ti2v", "wan_ti2v", "wan_ti2v")}
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


# ---------------------------------------------------------------------------
# Chunk 1c -- the render-time EQUALITY ASSERTION
# ---------------------------------------------------------------------------

def _frozen_ledger(engine="ltx_audio_in", role="announcer_visual",
                   frozen_engine=None, env=None):
    """A ledger shaped like one ShotLock stamps after the 1b freeze."""
    return {
        "episode_id": "ep_eq",
        "video": {
            "roles_effective": {role: frozen_engine or engine},
            "routing_env_snapshot": env if env is not None else {
                "force_engine_map": "", "enable_humo_hosts": False},
            "execution_groups": [
                {"group_id": "grp_%s" % role, "kind": "consumer",
                 "engine_id": engine, "depends_on": []}],
            "shots": [
                {"shot_id": "shot_b001", "role": role, "group_id": "grp_%s" % role,
                 "engine_id": engine, "family": "", "target_frame_count": 25}],
        },
    }


def test_a_frozen_route_is_verified_and_never_mutated(monkeypatch):
    """The whole point of 1c: at render time we CHECK, we do not repair.

    Repairing is what the old code did, and it is why the defect survived --
    the render phase silently corrected a route the image phase had already
    minted stills for, so the two disagreed and only the pixels showed it.
    """
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    ledger = _frozen_ledger()
    before = json.dumps(ledger, sort_keys=True)
    assert rd.assert_frozen_route(ledger) is True
    out = rd.resolve_final_shot_engines(ledger)
    assert json.dumps(out, sort_keys=True) == before      # byte-identical


def test_a_forced_map_does_NOT_rewrite_a_frozen_ledger(monkeypatch):
    """A frozen plan is not re-forced at render.

    Before 1c, setting the force map late would rewrite the shot rows here --
    after the stills had already been minted for the frozen engine. The
    environment guard catches it instead.
    """
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    ledger = _frozen_ledger()
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=viz_camera")
    with pytest.raises(rd.RenderError, match="routing environment changed"):
        rd.resolve_final_shot_engines(ledger)


def test_shot_that_disagrees_with_the_frozen_route_is_terminal(monkeypatch):
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    ledger = _frozen_ledger(engine="humo", frozen_engine="ltx_audio_in")
    with pytest.raises(rd.RenderError, match="frozen route says"):
        rd.assert_frozen_route(ledger)


def test_shot_that_diverges_from_its_group_is_terminal(monkeypatch):
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    ledger = _frozen_ledger()
    # ANY engine that differs from the shot row proves the divergence check;
    # this read "wan_i2v" until that engine retired 2026-08-26.
    ledger["video"]["execution_groups"][0]["engine_id"] = "wan_ti2v"
    with pytest.raises(rd.RenderError, match="execution group"):
        rd.assert_frozen_route(ledger)


def test_legacy_ledger_keeps_the_historic_mutating_behaviour(monkeypatch):
    """The legacy CPU fixtures (and, until 2026-09-05, the two POST HTTP
    harness routes that have since been removed) build ledgers by hand
    with no OTR_VideoDirector upstream. A strict assertion would break paths
    never in this contract, so the mutating branch survives -- NAMED and
    logged, not silent."""
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    legacy = {
        "video": {
            "shots": [{"shot_id": "s1", "role": "announcer_visual",
                       "engine_id": "humo", "family": ""}],
        },
    }
    assert rd.assert_frozen_route(legacy) is False       # nothing to verify
    rd.resolve_final_shot_engines(legacy)
    # ...and the historic redirect still happens on that path
    assert legacy["video"]["shots"][0]["engine_id"] == rd._NEVER_HUMO_REDIRECT_ENGINE


def test_frozen_route_from_ledger_reads_only_a_real_map():
    assert rd.frozen_route_from_ledger({}) == {}
    assert rd.frozen_route_from_ledger({"video": {}}) == {}
    assert rd.frozen_route_from_ledger({"video": {"roles_effective": {}}}) == {}
    assert rd.frozen_route_from_ledger(
        {"video": {"roles_effective": {"announcer_visual": "ltx_audio_in"}}}
    ) == {"announcer_visual": "ltx_audio_in"}


def test_build_request_from_shot_does_not_redirect_under_a_frozen_route(monkeypatch):
    """~30 later reads of shot["engine_id"]/["family"] live in that function.

    Under a frozen route the row already holds the effective engine, so
    re-deriving could only introduce a disagreement with the minted stills.
    """
    monkeypatch.delenv("OTR_ENABLE_HUMO_HOSTS", raising=False)
    # A frozen ledger whose row deliberately still says "humo": the old code
    # would have rewritten it in place here.
    ledger = _frozen_ledger(engine="humo", frozen_engine="humo")
    shot = ledger["video"]["shots"][0]
    try:
        rd.build_request_from_shot(shot, ledger)
    except Exception:  # noqa: BLE001 -- request building may fail for other reasons
        pass
    assert shot["engine_id"] == "humo"   # NOT mutated to ltx_audio_in


def test_end_to_end_freeze_survives_the_render_boundary(monkeypatch):
    """OTR_VideoDirector -> ShotLock -> render assertion: one route throughout."""
    policy = _direct(monkeypatch, announcer="humo_1.7B")
    patched = json.loads(_lock_with(policy)[0])
    # ShotLock stamped the frozen route...
    assert patched["video"]["roles_effective"]["announcer_visual"] == \
        rd._NEVER_HUMO_REDIRECT_ENGINE
    # ...and the render boundary verifies rather than re-derives it.
    assert rd.assert_frozen_route(patched) is True


# ---------------------------------------------------------------------------
# QA round 2 -- every reader fails closed, no exceptions
# ---------------------------------------------------------------------------

_BAD_MAP = "not-a-role-pair"


def test_creative_directives_do_not_swallow_a_malformed_force_map(monkeypatch):
    """THIRD swallow site, found by the local agy panel after the Sonnet pass.

    ``derive_creative_directives`` asked the dispatcher for still capabilities
    inside a broad ``except Exception: still_capabilities = None``. "Uncertainty
    retains authoring" is right for an INCOMPLETE policy and wrong for a typo'd
    OTR_FORCE_ENGINE_MAP: the lock would quietly author every character beat as
    though capability were unknown, on a plan that cannot render.
    """
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", _BAD_MAP)
    beats = [{"beat_id": "b001", "role": "character_video", "char_id": "c1",
              "dur_s": 2.0}]
    policy = {"policy_version": 2,
              "video_models": _resolved("humo", "humo", "humo")}
    with pytest.raises(rf.RouteFreezeError):
        sl.derive_creative_directives(beats, {}, {}, video_policy=policy)


def test_meta_brief_capability_probe_does_not_swallow_a_malformed_force_map(monkeypatch):
    """FOURTH swallow site, same round, same shape."""
    from nodes import otr_meta_brief_image_prompt as mbp

    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", _BAD_MAP)
    with pytest.raises(rf.RouteFreezeError):
        mbp._still_consumer_capabilities(_resolved("humo", "humo", "humo"))


def test_still_word_roles_do_not_swallow_a_malformed_force_map(monkeypatch):
    """FIRST swallow site (Sonnet pass) -- kept as a standing guard."""
    from nodes import otr_meta_brief_image_prompt as mbp

    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", _BAD_MAP)
    policy = json.dumps({"video_models": _resolved("humo", "humo", "humo")})
    with pytest.raises(rf.RouteFreezeError):
        mbp._still_word_roles_from_policy(policy)


def test_still_consumer_capability_does_not_swallow_a_malformed_force_map(monkeypatch):
    """SECOND swallow site (Sonnet pass) -- kept as a standing guard."""
    from nodes import otr_image_gen_dispatcher as disp

    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", _BAD_MAP)
    policy = {"video_models": _resolved("humo", "humo", "humo")}
    with pytest.raises(rf.RouteFreezeError):
        disp.still_consumer_capability(policy, "character_video")


# (lean-mean order 4, 2026-08-23) Two tests of three_d_locked_slots were
# here -- the effective-vs-picked resolution order and its pre-freeze
# fallback. The lock they pinned is removed with the character_3d family;
# its own docstring called it DORMANT (zero registered declarers of
# requires_mesh_portrait since 2026-06-29), and the effective-first rule
# they defended lives on in mesh_fodder_roles_from_video_policy and the
# aspects map, which the rest of THIS file still pins.
