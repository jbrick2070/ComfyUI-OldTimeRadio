"""Tests for the deterministic per-line delivery (emotion) vector."""
from nodes._otr_delivery_vector import DELIVERY_TABLE_VERSION, EMOTIONS, deterministic_delivery_vector


def test_vector_deterministic():
    a = deterministic_delivery_vector("Warning! The reactor is failing!")
    b = deterministic_delivery_vector("Warning! The reactor is failing!")
    assert a == b


def test_vector_dims_and_range():
    v = deterministic_delivery_vector("Help! Run! No escape!", 0.8)
    assert set(v.keys()) == set(EMOTIONS)
    assert all(0.0 <= x <= 1.0 for x in v.values())


def test_neutral_line_is_calm_dominant():
    v = deterministic_delivery_vector("The report is on the desk.")
    others = max(v[e] for e in EMOTIONS if e != "calm")
    assert v["calm"] >= others


def test_fear_line_raises_afraid():
    v = deterministic_delivery_vector("Run! Danger! Hide!", 0.9)
    assert v["afraid"] > v["happy"]


# --------------------------------------------------------------------------- #
# Whiny-fix P1.1: delivery table v2 properties + the stamped-version guard
# --------------------------------------------------------------------------- #
import random as _random

from nodes._otr_delivery_vector import select_delivery_vector


def test_v2_table_version():
    assert DELIVERY_TABLE_VERSION == "v2"


def test_pure_neutral_text_sends_no_aroused_dims():
    v = deterministic_delivery_vector("The report is on the desk.")
    for e in ("angry", "afraid", "surprised", "happy"):
        assert v[e] == 0.0, e


def test_question_only_line_never_saturates_surprised():
    v = deterministic_delivery_vector("Is that the relay station????")
    assert v["surprised"] < 0.5


def test_punctuation_counts_once_not_per_occurrence():
    one = deterministic_delivery_vector("Stop right there!")
    many = deterministic_delivery_vector("Stop!!! right!!! there!!!")
    for e in EMOTIONS:
        if e != "calm":
            assert many[e] == one[e], e


def test_stop_word_interrogatives_demoted():
    v = deterministic_delivery_vector("What do you mean, who was it, how odd.")
    assert v["surprised"] == 0.0


def test_debleat_terminal_question_with_fear_demotes_surprised():
    base = deterministic_delivery_vector("Danger! Is it behind you?")
    # afraid cues present + terminal '?' -> surprised demoted under afraid
    assert base["afraid"] > base["surprised"]


def test_secondary_mass_cap_holds_on_random_corpora():
    rng = _random.Random(20260611)
    vocab = ["help", "run", "danger", "sorry", "lost", "gone", "impossible",
             "suddenly", "laugh", "joy", "fool", "never", "sick", "vile",
             "remember", "once", "the", "a", "station", "relay", "dusk"]
    for _ in range(200):
        text = " ".join(rng.choice(vocab) for _ in range(rng.randint(3, 25)))
        text += rng.choice(["", "?", "!", "...", "?!"])
        v = deterministic_delivery_vector(text, rng.random())
        noncalm = sorted((v[e] for e in EMOTIONS if e != "calm"), reverse=True)
        # primary may be anything <=1; every secondary capped at 0.5
        assert all(x <= 0.5 for x in noncalm[1:]), (text, v)


def test_select_prefers_stamped_on_version_match():
    stamped = {e: 0.0 for e in EMOTIONS}
    stamped["angry"] = 1.0
    line = {"delivery": {"emotion_vector": stamped, "version": DELIVERY_TABLE_VERSION}}
    vec, src = select_delivery_vector(line, "calm words here.")
    assert vec is stamped and src == f"stamped:{DELIVERY_TABLE_VERSION}"


def test_select_rederives_on_stale_stamp_version():
    stale = {e: 0.0 for e in EMOTIONS}
    stale["surprised"] = 1.0
    line = {"delivery": {"emotion_vector": stale, "version": "v1"},
            "line_id": "L1"}
    vec, src = select_delivery_vector(line, "The report is on the desk.")
    assert vec is not stale
    assert src.startswith("rederived_stale_stamp")
    assert vec["surprised"] == 0.0


def test_select_derives_when_no_stamp():
    vec, src = select_delivery_vector({}, "Danger! Run!")
    assert src == "derived" and vec["afraid"] > 0.0


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
