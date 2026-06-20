"""Tests for nodes/_otr_anti_loop.py -- deterministic mechanical anti-loop /
dedupe (story-quality Phase 1, A4).

Covers the acceptance fixtures: a verbatim / near-duplicate loop is detected
and repaired; the "What if...?" rhetorical loop is detected; the announcer
open/close taglines are exempt; targets carry a voiced index + hint; a single
rhetorical question is left alone; the spine-side repairer recomposes only
CHARACTER targets in place and never touches the announcer.
"""

from nodes import _otr_anti_loop as AL


def _line(role, text, beat_id, char_id="c1"):
    return {
        "speaker_role": role,
        "text": text,
        "beat_id": beat_id,
        "char_id": char_id,
        "beat_intent": "advance the scene",
    }


def test_near_duplicate_flags_the_later_twin():
    voiced = [
        _line("character", "We have to reach the old reactor before midnight.", "b1"),
        _line("character", "Tell me about the weather on the coast today.", "b2", "c2"),
        _line("character", "We have to reach the old reactor before midnight!", "b3"),
    ]
    targets = AL.find_anti_loop_targets(voiced)
    idxs = {t.voiced_index for t in targets}
    assert 2 in idxs, "the near-identical later line must be flagged"
    assert 0 not in idxs, "the first occurrence is kept, not flagged"
    t = next(t for t in targets if t.voiced_index == 2)
    assert "near_duplicate" in t.kind
    assert t.hint and isinstance(t.hint, str)
    assert t.beat_id == "b3"


def test_short_repeats_not_flagged_as_duplicate():
    voiced = [
        _line("character", "Yes, of course.", "b1"),
        _line("character", "Yes, of course.", "b2"),
    ]
    targets = AL.find_anti_loop_targets(voiced)
    assert targets == [], "lines under the 6-word floor are not dedupe targets"


def test_question_loop_three_total_flags_all_but_first():
    voiced = [
        _line("character", "What if the signal is a warning?", "b1"),
        _line("character", "We should keep moving north.", "b2", "c2"),
        _line("character", "What if the signal is a trap?", "b3"),
        _line("character", "What if the signal is the end?", "b4", "c2"),
    ]
    targets = AL.find_anti_loop_targets(voiced)
    idxs = {t.voiced_index for t in targets}
    assert idxs == {2, 3}, "first 'What if' kept; the 2 later ones flagged"
    assert all(t.kind.endswith("question_loop") or "question_loop" in t.kind
               for t in targets)


def test_two_consecutive_questions_flag_the_second():
    voiced = [
        _line("character", "What if we are already too late?", "b1"),
        _line("character", "What if we are wrong about all of it?", "b2", "c2"),
        _line("character", "Let us check the manifest.", "b3"),
    ]
    targets = AL.find_anti_loop_targets(voiced)
    idxs = {t.voiced_index for t in targets}
    assert 1 in idxs, "the second consecutive same-prefix question is flagged"
    assert 0 not in idxs


def test_single_rhetorical_question_is_fine():
    voiced = [
        _line("character", "What if the readings are wrong?", "b1"),
        _line("character", "We move at dawn regardless.", "b2", "c2"),
    ]
    assert AL.find_anti_loop_targets(voiced) == []


def test_announcer_taglines_are_exempt():
    # first + last announcer lines are the open/close taglines -> exempt,
    # even though they are byte-identical to each other.
    tag = "You are listening to the Old Time Radio science hour tonight."
    voiced = [
        _line("announcer", tag, "a0", char_id="ann"),
        _line("character", "We have to reach the old reactor before midnight.", "b1"),
        _line("character", "We have to reach the old reactor before midnight.", "b2"),
        _line("announcer", tag, "a1", char_id="ann"),
    ]
    targets = AL.find_anti_loop_targets(voiced)
    idxs = {t.voiced_index for t in targets}
    assert 0 not in idxs and 3 not in idxs, "announcer taglines are exempt"
    assert 2 in idxs, "the duplicated character line is still flagged"


def test_repair_recomposes_only_character_targets_in_place():
    voiced = [
        _line("character", "What if the core is already breached?", "b1"),
        _line("character", "What if the core is already lost?", "b2", "c2"),
    ]

    def recompose_fn(idx, original, hint):
        return f"FIXED-{idx}: a concrete new line."

    summary = AL.repair_character_anti_loops(None, voiced, recompose_fn)
    assert summary["status"] == "ok"
    assert summary["repaired"] >= 1
    # the flagged consecutive question (index 1) is rewritten in place.
    assert voiced[1]["text"].startswith("FIXED-1")
    # the kept first occurrence is untouched.
    assert voiced[0]["text"] == "What if the core is already breached?"


def test_repair_never_touches_announcer_lines():
    tag = "Stay tuned for more thrilling tales after these messages tonight."
    voiced = [
        _line("announcer", tag, "a0", char_id="ann"),
        _line("character", "We have to reach the old reactor before midnight.", "b1"),
        _line("character", "We have to reach the old reactor before midnight.", "b2"),
        _line("announcer", tag, "a1", char_id="ann"),
    ]

    def recompose_fn(idx, original, hint):
        return "REWRITTEN"

    summary = AL.repair_character_anti_loops(None, voiced, recompose_fn)
    assert voiced[0]["text"] == tag and voiced[3]["text"] == tag
    assert voiced[2]["text"] == "REWRITTEN"
    assert summary["character_targets"] >= 1


def test_detection_is_deterministic():
    voiced = [
        _line("character", "We have to reach the old reactor before midnight.", "b1"),
        _line("character", "We have to reach the old reactor before midnight.", "b2"),
    ]
    a = AL.find_anti_loop_targets(voiced)
    b = AL.find_anti_loop_targets(voiced)
    assert [t.voiced_index for t in a] == [t.voiced_index for t in b]
    assert [t.hint for t in a] == [t.hint for t in b]
