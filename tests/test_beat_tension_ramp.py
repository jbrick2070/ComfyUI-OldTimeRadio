"""STEP 6 (2026-06-22 story+cast fix): the deterministic escalating
beat_tension ramp over character beats. Pure helper -- no GPU, no LLM."""

from nodes._otr_slot_drama_contract import compute_beat_tension_ramp as R


def test_empty_returns_empty():
    assert R([]) == {}
    assert R(None) == {}


def test_single_beat_is_mid_scale():
    assert R(["b1"]) == {"b1": 3}


def test_two_beats_span_the_extremes():
    # round(1 + 4*0/1)=1 ; round(1 + 4*1/1)=5
    assert R(["a", "b"]) == {"a": 1, "b": 5}


def test_five_beats_climb_one_per_step():
    assert R(["b0", "b1", "b2", "b3", "b4"]) == {
        "b0": 1, "b1": 2, "b2": 3, "b3": 4, "b4": 5,
    }


def test_thirteen_beats_monotone_nondecreasing_start1_peak5():
    ids = [f"b{i:02d}" for i in range(13)]
    m = R(ids)
    vals = [m[i] for i in ids]
    assert vals[0] == 1 and vals[-1] == 5            # starts low, peaks at the end
    assert all(1 <= v <= 5 for v in vals)            # clamped
    assert all(b >= a for a, b in zip(vals, vals[1:]))  # never decreases


def test_deterministic_same_input_same_output():
    ids = ["x", "y", "z", "w"]
    assert R(ids) == R(list(ids))


def test_blank_ids_are_dropped():
    assert R(["", "  ", "b1"]) == {"b1": 3}
