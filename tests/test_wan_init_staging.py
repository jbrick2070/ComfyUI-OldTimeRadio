"""GO_FORWARD section 4A S7 + S10 -- Wan init-image staging.

S7: the staged init was a FIXED otr_wan_init_WxH.png, so two shots at the same
dims clobbered each other's staged init. _staged_init_name keys on shot + seed
+ dims -- unique per beat, deterministic per (shot, seed) for V-7.

S10: Pillow is REQUIRED to materialize the init into the canvas. A missing /
unreadable source now fails LOUD; the old silent raw-stage fallback (which
leaned on WanImageToVideo's internal cover-resize, an N9 stretch risk) is gone.
"""

from __future__ import annotations

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines.eng_wan_i2v import WanI2VEngine


# --------------------------------------------------------------------------- #
# S7 -- _staged_init_name
# --------------------------------------------------------------------------- #
def _req(shot_id="s1", seed=3):
    return {"shot_id": shot_id, "seed_bundle": {"request_seed": seed}}


def test_staged_name_encodes_shot_seed_dims():
    eng = WanI2VEngine()
    assert eng._staged_init_name(_req("b007", 42), 832, 480) == \
        "otr_wan_init_b007_s42_832x480.png"


def test_staged_name_distinct_per_shot():
    eng = WanI2VEngine()
    a = eng._staged_init_name(_req("b001", 3), 832, 480)
    b = eng._staged_init_name(_req("b002", 3), 832, 480)
    assert a != b


def test_staged_name_distinct_per_seed():
    eng = WanI2VEngine()
    a = eng._staged_init_name(_req("b001", 3), 832, 480)
    b = eng._staged_init_name(_req("b001", 9), 832, 480)
    assert a != b


def test_staged_name_deterministic_for_same_shot_seed_dims():
    eng = WanI2VEngine()
    a = eng._staged_init_name(_req("b001", 3), 832, 480)
    b = eng._staged_init_name(_req("b001", 3), 832, 480)
    assert a == b   # V-7: render-twice determinism


def test_staged_name_sanitizes_unsafe_chars():
    eng = WanI2VEngine()
    name = eng._staged_init_name(
        {"shot_id": "a/b c:d", "seed_bundle": {"request_seed": 1}}, 64, 64)
    assert "/" not in name and ":" not in name and " " not in name
    assert name.startswith("otr_wan_init_a_b_c_d_s1_")


def test_staged_name_falls_back_to_request_id_then_wan():
    eng = WanI2VEngine()
    assert eng._staged_init_name(
        {"request_id": "r9", "seed_bundle": {}}, 10, 20) == \
        "otr_wan_init_r9_s0_10x20.png"
    assert eng._staged_init_name({}, 10, 20) == "otr_wan_init_wan_s0_10x20.png"


# --------------------------------------------------------------------------- #
# S10 -- _materialize_init_image fail-loud
# --------------------------------------------------------------------------- #
def test_materialize_fails_loud_on_missing_source(tmp_path):
    eng = WanI2VEngine()
    with pytest.raises(wb.GraphExecutionError) as exc:
        eng._materialize_init_image(
            _req(), str(tmp_path / "nope.png"), 64, 64)
    assert "missing" in str(exc.value)


def test_materialize_fails_loud_on_unreadable_source(tmp_path):
    # A non-image file: S10 removes the silent raw-stage fallback, so this now
    # raises instead of staging garbage.
    bad = tmp_path / "p.png"
    bad.write_bytes(b"\x89PNGnot-a-real-image")
    eng = WanI2VEngine()
    with pytest.raises(wb.GraphExecutionError) as exc:
        eng._materialize_init_image(_req(), str(bad), 64, 64)
    assert "unreadable" in str(exc.value)


def test_materialize_real_image_stages_unique_name(tmp_path, monkeypatch):
    Image = pytest.importorskip("PIL.Image")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    src = tmp_path / "p.png"
    Image.new("RGB", (480, 832), (10, 20, 30)).save(src)
    monkeypatch.setattr(wb, "comfy_input_dir", lambda: str(in_dir))
    eng = WanI2VEngine()
    req = {"shot_id": "b003", "seed_bundle": {"request_seed": 5},
           "canvas": {"w": 832, "h": 480, "aspect_policy": "pad"}}
    name = eng._materialize_init_image(req, str(src), 832, 480)
    assert name == "otr_wan_init_b003_s5_832x480.png"
    assert (in_dir / name).exists()
