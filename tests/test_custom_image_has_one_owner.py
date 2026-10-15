"""A custom image engine is declared once, on OTR_VideoDirector (plan item 9).

The queue-time planners (visual assets, slug and balance preflights) resolve a
'+ Add Custom Model' image pick with VideoDirector's custom_models_json, but
VideoDirector emitted the raw sentinel and ImageDirector resolved it from its
OWN map. One declaration could pass the queue and fail at run time; two could
plan one engine and generate another. VideoDirector now resolves the image
sentinel with admission's own reading of the declaration and emits the bare id.
"""
from __future__ import annotations

import json

import pytest

from nodes import otr_video_director as vd
from nodes._otr_visual_assets import _custom_models, _resolve_slot
from nodes.otr_image_director import OTRImageDirector

SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")


def _video_policy(monkeypatch, custom, **image_picks):
    for key in ("OTR_FORCE_ENGINE_MAP", "OTR_ENABLE_HUMO_HOSTS"):
        monkeypatch.delenv(key, raising=False)
    picks = {slot: image_picks.get(slot, "flux_gen1") for slot in SLOTS}
    (raw,) = vd.OTRVideoDirector().direct(
        announcer_video_model="still_flat", music_video_model="still_flat",
        character_video_model="still_flat", fps=25, canvas_w=832, canvas_h=480,
        custom_models_json=json.dumps(custom), **picks)
    return raw


def _image_engines(policy_raw, image_custom=None):
    (raw,) = OTRImageDirector().direct(
        fresh_cap=15, seed_mode="request_hash", request_seed=0,
        video_policy_json=policy_raw,
        custom_models_json=json.dumps(image_custom or {}))[:1]
    return {slot: row.get("engine_id")
            for slot, row in json.loads(raw)["image_models"].items()}


@pytest.mark.parametrize("slot", SLOTS)
def test_a_producer_declaration_resolves_every_image_role(monkeypatch, slot):
    custom = {slot: "z_image_turbo"}
    policy = _video_policy(monkeypatch, custom, **{slot: vd.ADD_CUSTOM})
    assert json.loads(policy)["image_models"][slot] == "z_image_turbo"
    assert _image_engines(policy)[slot] == "z_image_turbo"
    # Queue-time admission reads the same declaration the same way.
    inputs = {slot: vd.ADD_CUSTOM, "custom_models_json": json.dumps(custom)}
    assert _resolve_slot(inputs, slot, _custom_models(inputs), "image") == "z_image_turbo"


def test_a_conflicting_downstream_map_cannot_override_the_producer(monkeypatch):
    policy = _video_policy(monkeypatch, {"music_image_model": "z_image_turbo"},
                           music_image_model=vd.ADD_CUSTOM)
    got = _image_engines(policy, image_custom={"music_image_model": "flux_gen1"})
    assert got["music_image_model"] == "z_image_turbo"


def test_the_declaration_is_read_the_way_admission_reads_it(monkeypatch):
    policy = _video_policy(monkeypatch, {"music_image_model": "  z_image_turbo  "},
                           music_image_model=vd.ADD_CUSTOM)
    assert json.loads(policy)["image_models"]["music_image_model"] == "z_image_turbo"


def test_an_undeclared_sentinel_passes_through_unchanged(monkeypatch):
    """No new refusal at the producer: the sentinel goes on exactly as before,
    and ImageDirector's direct/legacy path still owns what happens next."""
    policy = _video_policy(monkeypatch, {}, music_image_model=vd.ADD_CUSTOM)
    assert json.loads(policy)["image_models"]["music_image_model"] == vd.ADD_CUSTOM
    # The direct path: a map on ImageDirector still resolves it there...
    assert _image_engines(policy, {"music_image_model": "z_image_turbo"})[
        "music_image_model"] == "z_image_turbo"
    # ...and without one it is an unresolved record, not the flux default.
    assert _image_engines(policy)["music_image_model"] in ("", None)


def test_ordinary_picks_are_untouched(monkeypatch):
    policy = _video_policy(monkeypatch, {"music_image_model": "z_image_turbo"})
    assert json.loads(policy)["image_models"] == {slot: "flux_gen1" for slot in SLOTS}


def _outcome(policy):
    try:
        return ("ok", _image_engines(policy)["music_image_model"])
    except Exception as exc:  # noqa: BLE001 -- the outcome IS the assertion
        return ("raised", type(exc).__name__)


def test_a_declared_engine_is_judged_exactly_like_the_same_pick(monkeypatch):
    """Resolving at the producer must not bypass ImageDirector's own checks:
    a declared id meets the same compatibility/registration verdict as that id
    picked straight from the dropdown -- for a usable and an unusable id."""
    for engine in ("z_image_turbo", "still_flat", "not_a_real_image_engine"):
        declared = _video_policy(monkeypatch, {"music_image_model": engine},
                                 music_image_model=vd.ADD_CUSTOM)
        direct = _video_policy(monkeypatch, {}, music_image_model=engine)
        assert _outcome(declared) == _outcome(direct), engine
