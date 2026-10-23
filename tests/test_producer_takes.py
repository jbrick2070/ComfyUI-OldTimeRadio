"""Scene Producer takes in every still seed (chunk 3c, 2026-10-22).

A TAKE IS A REDRAW. ``Paint again`` raises a still's take; a character's
``Paint again`` raises that character's take. Each take above zero turns the
seed into a different, reproducible one in every seed branch the dispatcher
has -- the radio bookend, the fixed mode, the scene character's identity seed
and the per-object request hash -- and take 0 is today's seed, byte for byte.

A character's take enters the IDENTITY seed, so the portrait and every scene
still of that character move together: one face, redrawn once. A still's own
take enters last, on that still only.

Headless: the seed resolver is pure, and the dispatch test runs the real
``dispatch_images`` on a stand-in engine.
"""
from __future__ import annotations

import pathlib
import sys
import types

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from nodes import _otr_producer_direction as PD  # noqa: E402
from nodes import otr_image_gen_dispatcher as D  # noqa: E402

REQUEST = {"mode": "request_hash", "request_seed": 7}
FIXED = {"mode": "fixed", "request_seed": 7}

BRANCHES = {
    "bookend": dict(seed_cfg=REQUEST, object_id="still_b000_music_open",
                    prompt_hash="p", kind="scene_open"),
    "fixed": dict(seed_cfg=FIXED, object_id="still_b001", prompt_hash="p",
                  kind="scene_character", char_id="c02"),
    "identity": dict(seed_cfg=REQUEST, object_id="still_b001", prompt_hash="p",
                     kind="scene_character", char_id="c02", portrait_prompt_hash="pp"),
    "request_hash": dict(seed_cfg=REQUEST, object_id="still_b002", prompt_hash="p",
                         kind="scene_beat"),
    "portrait": dict(seed_cfg=REQUEST, object_id="c02", prompt_hash="pp",
                     kind="portrait", char_id="c02"),
}


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_take_zero_is_todays_seed_in_every_branch(branch):
    today = D.resolve_seed_and_mode(**BRANCHES[branch])
    assert D.resolve_seed_and_mode(**BRANCHES[branch], identity_take=0, take=0) == today


@pytest.mark.parametrize("branch", sorted(BRANCHES))
def test_every_take_above_zero_redraws_reproducibly(branch):
    today, _mode = D.resolve_seed_and_mode(**BRANCHES[branch])
    seeds = [D.resolve_seed_and_mode(**BRANCHES[branch], take=t)[0] for t in (1, 2, 3)]
    assert len({today, *seeds}) == 4, "each take is a different draw"
    assert seeds == [D.resolve_seed_and_mode(**BRANCHES[branch], take=t)[0] for t in (1, 2, 3)]


def test_a_characters_take_moves_its_portrait_and_its_scene_stills_together():
    portrait = BRANCHES["portrait"]
    scene = BRANCHES["identity"]
    # Today: the scene still's identity seed IS the portrait's own draw.
    assert D.resolve_seed_and_mode(**portrait)[0] == D.resolve_seed_and_mode(**scene)[0]
    for take in (1, 4):
        p_seed = D.resolve_seed_and_mode(**portrait, identity_take=take)[0]
        s_seed, mode = D.resolve_seed_and_mode(**scene, identity_take=take)
        assert p_seed == s_seed and mode == "seed"
        assert p_seed != D.resolve_seed_and_mode(**portrait)[0]
    # A still's own take then redraws that still alone, keeping the face basis.
    assert (D.resolve_seed_and_mode(**scene, identity_take=1, take=1)[0]
            != D.resolve_seed_and_mode(**scene, identity_take=1)[0])


def test_the_takes_an_object_gets_come_from_its_character_and_its_shot():
    direction = {"takes": {"characters": {"c02": 2}, "stills": {"shot_b001": 3},
                           "clips": {"shot_b001": 9}}}
    assert D._producer_takes(direction, "portrait", "c02", "") == (2, 0)
    assert D._producer_takes(direction, "scene_character", "c02", "b001") == (2, 3)
    assert D._producer_takes(direction, "scene_beat", "", "b001") == (0, 3)
    assert D._producer_takes(direction, "scene_character", "c09", "b007") == (0, 0)
    assert D._producer_takes(None, "scene_character", "c02", "b001") == (0, 0)


def _dispatch(tmp_path, ledger):
    import numpy as np

    from nodes._otr_image_engines import registry as ireg
    from nodes._otr_shared import role_compat as rc

    saved = dict(ireg._IMAGE_REGISTRY._registry)
    ireg._IMAGE_REGISTRY._registry.clear()
    ireg.register(types.SimpleNamespace(
        name="flux_gen1", roles=rc.ROLES, default_roles=rc.ROLES, commercial_clean=True,
        requires_flag=None, required_inputs=("text_prompt",), engine_version="1"))
    policy = {"policy_version": 2,
              "image_models": {"character_image_model": {"engine_id": "flux_gen1"}},
              "video_models": {"announcer_video_model": {"engine_id": "viz_mxc_cpu"},
                               "music_video_model": {"engine_id": "viz_mxc_mandala"},
                               "character_video_model": {"engine_id": "still_motion"}},
              "seed": {"request_seed": 0}}
    payload = {"version": 1, "objects": [
        {"object_id": cid, "kind": "portrait", "role": "character_video", "char_id": cid,
         "w": 832, "h": 1216, "prompt": "a radio actor " + cid, "prompt_hash": "ph_" + cid,
         "source": "llm"} for cid in ("c1", "c2")]}
    try:
        led, _done, _report, _warn = D.dispatch_images(
            ledger, policy, payload,
            gen_fn=lambda _r: np.full((8, 8, 3), 9, dtype=np.uint8),
            output_dir=str(tmp_path), lockdir=tmp_path / "lease.lockdir")
    finally:
        ireg._IMAGE_REGISTRY._registry.clear()
        ireg._IMAGE_REGISTRY._registry.update(saved)
    return {row["object_id"]: row for row in led["images"]["images"]}


def test_a_producer_run_draws_only_the_retaken_character_again(tmp_path):
    first = PD.parse_direction(
        '{"version": 1, "project_id": "a1b2c3d4e5f6", "attempt_id": "0f8e1c2d-3b4a", '
        '"stage": "stills", "out": "01-stills"}')
    plain = _dispatch(tmp_path / "plain", {"episode_id": "ep_t", "cast": [
        {"char_id": "c1"}, {"char_id": "c2"}], PD.RUNTIME_KEY: first})
    retaken = dict(first, takes={"characters": {"c2": 1}, "stills": {}, "clips": {}})
    again = _dispatch(tmp_path / "again", {"episode_id": "ep_t", "cast": [
        {"char_id": "c1"}, {"char_id": "c2"}], PD.RUNTIME_KEY: retaken})
    assert again["c1"]["request_hash"] == plain["c1"]["request_hash"], "untouched: same key"
    assert again["c2"]["request_hash"] != plain["c2"]["request_hash"], "retaken: a new draw"
