"""word_razzle (Phase 1, 2026-07-03) -- the animated word-card cloud i2v engine.

Offline: the bridge is never called here. Covers registration + the registry
invariants, the dynamic combo surface, the _partner_inputs kwargs (every one
declared in the pinned Pixverse row -- the conformance suite double-guards this),
the asset_refs init-image resolution fix, the fail-LOUD no-init path, and the
env-overridable motion/quality/duration knobs. The LIVE provider spike stays
behind OTR_COMFY_API_KEY + the gated smoke (cannot run without credentials).
"""
from __future__ import annotations

import pytest

from nodes import otr_video_director as vd
from nodes._otr_video_engines import eng_cloud_video as ecv
from nodes._otr_video_engines import registry as vreg


# --------------------------------------------------------------------------- #
# registration + menu surface
# --------------------------------------------------------------------------- #
def test_word_razzle_registered_and_capabilities():
    assert vreg.is_registered("word_razzle")
    row = vreg.CAPABILITIES["word_razzle"]
    assert row["vram_class"] == "cpu" and row["cpu_ok"] is True
    assert set(vreg.CAPABILITIES) == set(vreg.all_engine_names())   # invariant


def test_word_razzle_identity():
    eng = ecv.WordRazzle
    assert eng.name == "word_razzle"
    assert eng.node_key == "cloud_pixverse_i2v"
    assert eng.family == "image_to_video"
    assert eng.required_inputs == ("init_image", "text_prompt")
    assert eng.reactivity == "mute_only" and eng.must_strip_audio is True


def test_word_razzle_never_default():
    assert tuple(ecv.WordRazzle.default_roles) == ()
    for role in ("announcer_visual", "music_visual", "character_video"):
        assert not str(vreg.default_engine_for_role(role)).startswith("cloud_")
        assert vreg.default_engine_for_role(role) != "word_razzle"


def test_word_razzle_in_video_combo():
    parsed = {vd._engine_id_from_pick(c) for c in vd._video_model_combo()}
    assert "word_razzle" in parsed


# --------------------------------------------------------------------------- #
# _partner_inputs -- kwargs, prompt, env knobs (bridge never called)
# --------------------------------------------------------------------------- #
def _png(tmp_path):
    from PIL import Image
    p = tmp_path / "card.png"
    Image.new("RGB", (64, 36), (30, 30, 60)).save(str(p))
    return str(p)


def _request(tmp_path, **over):
    req = {
        "shot_id": "shot_b001",
        "asset_refs": {"init_image": _png(tmp_path)},
        "text_prompt": "a glowing radio marquee at night",
        "seed_bundle": {"request_seed": 11},
        "canvas": {"w": 832, "h": 448, "fps": 25},
        "timing": {"target_frame_count": 125},           # 5s @ 25fps
    }
    req.update(over)
    return req


def test_partner_inputs_kwargs_all_declared(tmp_path):
    ins = ecv.WordRazzle._partner_inputs(_request(tmp_path))
    assert set(ins) == {"image", "prompt", "negative_prompt", "motion_mode",
                        "quality", "duration_seconds", "seed"}
    # every kwarg is a declared input in the pinned Pixverse row.
    from nodes._otr_shared.cloud_media_invoke import partner_rows
    row = partner_rows()["cloud_pixverse_i2v"]
    declared = set(row["inputs"]["required"]) | set(row["inputs"]["optional"])
    assert set(ins) <= declared
    assert ins["seed"] == 11
    assert hasattr(ins["image"], "ndim") and ins["image"].ndim == 4


def test_prompt_leads_with_readability_directive(tmp_path):
    ins = ecv.WordRazzle._partner_inputs(_request(tmp_path))
    assert "legible" in ins["prompt"].lower()
    assert "a glowing radio marquee at night" in ins["prompt"]   # beat appended


def test_asset_refs_init_image_resolves(tmp_path):
    # the whole point of the asset_refs fix: init under asset_refs, not top-level.
    req = _request(tmp_path)
    assert "init_image" not in req                       # only under asset_refs
    ins = ecv.WordRazzle._partner_inputs(req)
    assert ins["image"].ndim == 4


def test_top_level_init_image_still_works(tmp_path):
    # back-compat: a hand-built dict with top-level init_image resolves too.
    png = _png(tmp_path)
    req = {"shot_id": "s", "init_image": png, "text_prompt": "x",
           "seed_bundle": {"request_seed": 1}, "canvas": {"w": 832, "h": 448, "fps": 25}}
    ins = ecv.WordRazzle._partner_inputs(req)
    assert ins["image"].ndim == 4


def test_missing_init_image_fails_loud(tmp_path):
    req = _request(tmp_path, asset_refs={})
    with pytest.raises(RuntimeError, match="init_image missing"):
        ecv.WordRazzle._partner_inputs(req)


def test_duration_derives_from_timing_and_clamps(tmp_path):
    short = ecv.WordRazzle._duration_seconds(_request(tmp_path))          # 125f/25 = 5
    assert short == 5
    long = ecv.WordRazzle._duration_seconds(
        _request(tmp_path, timing={"target_frame_count": 200}))          # 8s -> 8
    assert long == 8


def test_env_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_CLOUD_PIXVERSE_MOTION", "fast")
    monkeypatch.setenv("OTR_CLOUD_PIXVERSE_QUALITY", "720p")
    monkeypatch.setenv("OTR_CLOUD_PIXVERSE_DURATION", "8")
    monkeypatch.setenv("OTR_CLOUD_RAZZLE_MOTION_PROMPT", "custom motion clause")
    ins = ecv.WordRazzle._partner_inputs(_request(tmp_path))
    assert ins["motion_mode"] == "fast"
    assert ins["quality"] == "720p"
    assert ins["duration_seconds"] == 8
    assert ins["prompt"].startswith("custom motion clause")
