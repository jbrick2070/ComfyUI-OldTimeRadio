"""The Ghost Signal CADENCE peers -- hold-3 and hold-5.

THE DEFECT THIS SEAM EXISTS TO PREVENT, and it had already been written twice
today before a review panel found it a third time here: ``render_clip`` called
``ghost_unique_source_count``, ``ghost_source_request`` and
``ghost_hold2_selector`` as MODULE functions, every one hard-wired to hold-2. A
subclass declaring ``hold_factor = 5`` would have rendered at hold-2 and stamped
a hold-5 receipt on it -- wrong pixels under a confident label, which is
preflight G1.3 exactly.

So the tests that matter here are BEHAVIOURAL: they run the graph with recorded
fake node classes and read what the sampler was actually asked for. A test that
only asserted ``engine.hold_factor == 5`` would have passed against the broken
code.

THE OTHER HALF is that the golden lane must not move by a single frame. It
renders the operator's dailies and it rendered the published episode. Its
cadence is pinned here against an independently recomputed hold-2 reference
rather than against the implementation being tested.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_video_engines import eng_ghost_signal as gs
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import wrapper_bridge as wb

# THE CADENCE PEERS WERE RETIRED 2026-08-23 and their module is deleted. THIS
# FILE DID NOT GO WITH THEM, because most of what it guards SURVIVED: the hold
# arithmetic lives in `eng_ghost_signal`, which is the haunted lane's own base
# class. Deleting the file would have dropped the golden-cadence pin -- the one
# that catches a hold factor read from a module constant instead of through
# `self`, which is the G1.3 defect this file exists for.
GOLDEN = "animatediff15_v3_haunted_video"
EXPECTED_HOLD = {GOLDEN: 2}

#: A real episode beat. Legs on 2026-08-22 ran 2444-2944 delivered frames over
#: 8 beats, so ~12 seconds each at 25fps.
REAL_BEAT_FRAMES = 300


# --------------------------------------------------------------------------- #
# THE GOLDEN LANE DOES NOT MOVE
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("target", [1, 2, 3, 15, 16, 17, 31, 32, 50, 99, 100,
                                    150, 199, 200, 300, 301])
def test_hold2_is_byte_identical_to_the_pre_seam_arithmetic(target):
    """Recomputed here from first principles, NOT by calling the code under
    test with hold=2. If the seam changed the golden cadence anywhere, this
    catches it at that exact frame count."""
    reference_unique = (target + 1) // 2
    reference_selector = []
    for index in range(reference_unique):
        reference_selector.extend([index, index])
    reference_selector = reference_selector[:target]

    assert gs.ghost_unique_source_count(target) == reference_unique
    assert gs.ghost_hold2_selector(target) == reference_selector
    receipts = gs.ghost_cadence_receipts(target)
    assert receipts["cadence_mode"] == "hold_2"
    assert receipts["cadence_tail_trim"] == (2 * reference_unique) - target


def test_the_golden_lane_still_declares_hold_2():
    assert vreg.get_engine(GOLDEN).hold_factor == 2
    assert gs.GHOST_DEFAULT_HOLD == 2


def test_the_golden_recipe_receipt_is_unchanged():
    """The lane must not silently rename its own recipe.

    Was the mm-p_0.5 receipt while the base lane was the golden one. That lane
    retired 2026-08-23; the receipt pinned here is the surviving lane's, and the
    guard is the same one -- a recipe id is what a render receipt is read
    against, so it may never drift without someone deciding it should.
    """
    assert vreg.get_engine(GOLDEN)._recipe_receipt() == (
        "animatediff_sd15_v3_haunted_static16_512x288_v1")


# --------------------------------------------------------------------------- #
# THE SEAM, PROVEN BY EXECUTION
# --------------------------------------------------------------------------- #

class _Patcher:
    def __init__(self):
        self.detached = False

    def detach(self, unpatch_all=True):
        self.detached = True
        return self


class _Recorder:
    def __init__(self, decoded_frames):
        self.calls = []
        self.decoded_frames = decoded_frames
        self.base_model = _Patcher()
        self.lora_model = _Patcher()
        self.ade_model = _Patcher()
        self.clip = object()
        self.vae = object()

    def classes(self):
        rec = self

        def _node(tag, result):
            class _N:
                FUNCTION = "go"

                def go(self, **kw):
                    rec.calls.append((tag, dict(kw)))
                    return result() if callable(result) else result
            return _N

        def _decode():
            n = rec.decoded_frames
            frames = np.zeros((n, 288, 512, 3), dtype=np.uint8)
            for i in range(n):
                frames[i, 0, 0, 0] = i % 256
            return (frames,)

        return {
            "checkpoint": _node("ckpt", (rec.base_model, rec.clip, rec.vae)),
            "text_encode": _node("text_encode", (("cond",),)),
            "context": _node("context", ("CTX",)),
            # The surviving lane inserts the domain adapter between the
            # checkpoint and the ADE loader; without this node the mocked graph
            # cannot be built at all.
            "lora": _node("lora", (rec.lora_model,)),
            "ade": _node("ade", (rec.ade_model,)),
            "latent": _node("latent", ({"batch": 1},)),
            "sampler": _node("sampler", ("SAMPLED",)),
            "decode": _node("decode", _decode),
        }


def _render(engine_id, monkeypatch, target=REAL_BEAT_FRAMES):
    eng = vreg.get_engine(engine_id)
    expected_source = gs.ghost_source_request(target, eng.hold_factor)
    rec = _Recorder(expected_source)
    monkeypatch.setattr(eng, "_classes", rec.classes())
    monkeypatch.setattr(eng, "_loaded", True)
    monkeypatch.setattr(eng, "_patchers", [rec.base_model])
    prepared = {"engine_id": eng.name, "lease": None,
                "patchers": eng._patchers, "session_ctx": {},
                "base_model": (rec.base_model,), "clip": (rec.clip,),
                "vae": (rec.vae,), "recipe": eng._recipe_receipt()}
    captured = {}

    def _encode(frames, out_path, fps, **kw):
        captured["frames"] = np.asarray(frames)
        return (out_path, int(np.asarray(frames).shape[0]))

    monkeypatch.setattr(wb, "encode_frames_to_silent_mp4", _encode)
    monkeypatch.setattr(wb, "reclaim_idle_models", lambda reason="": None)
    monkeypatch.setattr(gs.GhostSignalEngine, "_ckpt_path", lambda self: "ck")
    monkeypatch.setattr(gs.GhostSignalEngine, "_motion_path", lambda self: "mm")
    # The surviving lane loads a THIRD artifact the retired ones did not -- the
    # removable domain adapter. Stub it here too or the mocked render asks the
    # filesystem for a real .ckpt.
    monkeypatch.setattr(gs.GhostSignalEngine, "_lora_path", lambda self: "lo")
    raw = eng.render_clip({
        "shot_id": "shot_b001", "request_id": "shot_b001",
        "text_prompt": "a tall stooped figure, mid-shot or wider, turns",
        "negative_prompt": "text, watermark, caption, lettering",
        "timing": {"target_frame_count": target},
        "seed_bundle": {"request_seed": 4242},
    }, prepared)
    return rec, raw, captured


@pytest.mark.parametrize("lane", (GOLDEN,))
def test_the_graph_asks_for_this_lanes_own_frame_count(lane, monkeypatch):
    """THE TEST THAT WOULD HAVE FAILED AGAINST THE BROKEN CODE. Reads the batch
    size the latent node actually received, not the declaration."""
    rec, _raw, _cap = _render(lane, monkeypatch)
    latent = [kw for name, kw in rec.calls if name == "latent"][0]
    expected = gs.ghost_source_request(REAL_BEAT_FRAMES, EXPECTED_HOLD[lane])
    assert latent["batch_size"] == expected


@pytest.mark.parametrize("lane", (GOLDEN,))
def test_the_delivered_frame_count_is_the_same_on_every_cadence(
        lane, monkeypatch):
    """THE AUDIO-SYNC CLAIM, and it is the one that must never break. T comes
    from the beat's audio; the hold only decides how many fresh frames fill it.
    Every lane must deliver exactly T frames for the same beat."""
    _rec, _raw, captured = _render(lane, monkeypatch)
    assert captured["frames"].shape[0] == REAL_BEAT_FRAMES


@pytest.mark.parametrize("lane", (GOLDEN,))
def test_the_receipt_names_the_cadence_that_actually_ran(lane, monkeypatch):
    """A constant 'hold_2' here would make every peer's receipt claim the
    golden lane's rate -- the same defect class as the module functions."""
    _rec, raw, _cap = _render(lane, monkeypatch)
    assert raw["cadence_mode"] == "hold_%d" % EXPECTED_HOLD[lane]
    assert raw["cadence_delivered_frame_count"] == REAL_BEAT_FRAMES


@pytest.mark.parametrize("lane", (GOLDEN,))
def test_the_tail_trim_stays_within_its_own_hold(lane, monkeypatch):
    _rec, raw, _cap = _render(lane, monkeypatch)
    assert 0 <= raw["cadence_tail_trim"] < EXPECTED_HOLD[lane]


# --------------------------------------------------------------------------- #
# THE ARITHMETIC THE PROBLEM STATEMENT RESTS ON
# --------------------------------------------------------------------------- #

def _windows(unique):
    """Standard Static: length-16 windows, overlap 4, so stride 12."""
    if unique <= gs.GHOST_CONTEXT_LENGTH:
        return 1
    stride = gs.GHOST_CONTEXT_LENGTH - gs.GHOST_CONTEXT_OVERLAP
    return 1 + math.ceil((unique - gs.GHOST_CONTEXT_LENGTH) / stride)


@pytest.mark.parametrize("lane,expected_windows", [(GOLDEN, 13)])
def test_the_window_count_on_a_real_twelve_second_beat(lane, expected_windows):
    """The number the whole cadence argument turns on. A real episode beat is
    ~12s, and the golden lane fuses THIRTEEN motion gestures into it."""
    hold = EXPECTED_HOLD[lane]
    unique = gs.ghost_unique_source_count(REAL_BEAT_FRAMES, hold)
    assert _windows(unique) == expected_windows


def test_a_short_beat_collapses_to_one_clean_gesture_on_every_lane():
    """use_on_equal_length=False runs the module DIRECTLY at 16, so a short
    beat is one coherent gesture no matter the cadence."""
    for lane in (GOLDEN,):
        unique = gs.ghost_source_request(25, EXPECTED_HOLD[lane])
        assert _windows(unique) == 1


def test_the_source_floor_still_holds_on_the_slowest_lane():
    """At hold-5 a 1s beat wants U=5, below the 16-frame floor, so it still
    asks for a full window and shows only part of the gesture. Documented on
    the engine; pinned here so it cannot change silently."""
    assert gs.ghost_source_request(25, 5) == gs.GHOST_SOURCE_FLOOR


# --------------------------------------------------------------------------- #
# G2 -- canvas truth. Every declaring lane needs a pin.
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# ADDITIVE: everything else is inherited
# --------------------------------------------------------------------------- #
