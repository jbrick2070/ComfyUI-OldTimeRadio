"""Story-Quality R2 Final QA -- the lever metrics in story_quality_scan.

r2_lever_metrics counts the shipped craft-lever flags + structural signals over a
frozen ledger (read-only). Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

_spec = importlib.util.spec_from_file_location(
    "story_quality_scan", str(_REPO / "scripts" / "story_quality_scan.py"))
_scan = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scan)


def _ledger(lines, meta=None, cast=None):
    return {"lines": lines, "meta": meta or {}, "cast": cast or []}


def test_counts_weak_flags():
    led = _ledger([
        {"speaker_role": "character", "text": "You're playing with fire, Skip."},
        {"speaker_role": "character", "text": "I'm scared, this is dangerous."},
        {"speaker_role": "character", "text": "clenches jaw You mean it?"},
        {"speaker_role": "announcer", "text": "And so the lesson is haste makes waste."},
    ])
    m = _scan.r2_lever_metrics(led)
    assert m["cliche_lines"] == 1
    assert m["on_the_nose_lines"] == 1
    assert m["leading_stage_dir_lines"] == 1
    assert m["thesis_close"] is True


def test_clean_episode_all_zero():
    led = _ledger([
        {"speaker_role": "character", "text": "The decay curve gives us ten months."},
        {"speaker_role": "announcer", "text": "The pencil lies still on the empty console."},
    ])
    m = _scan.r2_lever_metrics(led)
    assert m["cliche_lines"] == 0 and m["on_the_nose_lines"] == 0
    assert m["leading_stage_dir_lines"] == 0 and m["thesis_close"] is False


def test_structural_signals():
    led = _ledger(
        [{"speaker_role": "character", "text": "Hello."}],
        meta={
            "specificity_anchors": ["three robotic arms", "Swift"],
            "central_object": "the pencil",
            "dramatic_state": {
                "character_a_wants": "CHRIS: honor the established commitment, whatever the cost",
                "character_b_wants": "SKIP: force a compromise that protects the status quo",
            },
        },
        cast=[
            {"name": "CHRIS", "speech_signature": "clipped and terse"},
            {"name": "SKIP", "speech_signature": "warm and rambling"},
            {"name": "ANNOUNCER", "speech_signature": "narrator"},
        ],
    )
    m = _scan.r2_lever_metrics(led)
    assert m["has_specificity_anchors"] is True and m["n_specificity_anchors"] == 2
    assert m["has_central_object"] is True
    assert m["wants_default"] is True            # boilerplate wants detected
    assert m["voice_distinct_ratio"] == 1.0      # two distinct char registers


def test_scan_ledger_merges_lever_metrics(tmp_path):
    import json
    p = tmp_path / "ep_ledger.json"
    p.write_text(json.dumps(_ledger(
        [{"speaker_role": "character", "text": "You're playing with fire."}],
        meta={"central_object": "the pencil"})), encoding="utf-8")
    row = _scan.scan_ledger(str(p), None)
    assert row["cliche_lines"] == 1
    assert row["has_central_object"] is True


_ANCHORS = ["41.3 degrees C", "837/835 form", "the rail death", "Swift Observatory"]


def test_mf4_r2_helpers_loaded():
    """MF-4: if a detector import silently fell back to the no-op stubs, every R2
    metric zeros out. This guard fails loudly so a typo cannot hide it."""
    assert _scan._HAS_R2_HELPERS is True
    m = _scan.r3_quality_metrics(_ledger([]))
    assert m["r2_helpers_loaded"] is True


def test_r3_v2_detectors_on_final_text():
    stuffed = ("the 837/835 form logged the rail death right as it hit "
               "41.3 degrees C near the Swift Observatory gate")
    one_breath = " ".join(["word"] * 30)
    leak = "snaps off the cap, cracks the seal, drains the vial"
    cost = "It's the trust they will lose either way, Marlowe."
    led = _ledger(
        [
            {"speaker_role": "character", "text": stuffed},
            {"speaker_role": "character", "text": one_breath},
            {"speaker_role": "character", "text": leak},
            {"speaker_role": "character", "text": cost},
        ],
        meta={"specificity_anchors": _ANCHORS},
    )
    m = _scan.r3_quality_metrics(led)
    assert m["anchor_stuffing_lines"] == 1
    assert m["one_breath_violation_lines"] == 1
    assert m["stage_action_leak_lines"] == 1
    assert m["personal_cost_boilerplate_lines"] == 1


def test_r3_compose_flag_counters():
    led = _ledger([
        {"speaker_role": "character", "text": "ok", "compose_flags": ["cliche_retry"]},
        {"speaker_role": "character", "text": "ok",
         "compose_flags": ["quality_residual:cliche"]},
        {"speaker_role": "character", "text": "ok",
         "compose_flags": ["cliche_retry", "quality_reroll_degraded",
                           "quality_residual:cliche"]},
    ])
    m = _scan.r3_quality_metrics(led)
    assert m["quality_retry_lines"] == 2
    assert m["quality_residual_lines"] == 2
    assert m["quality_reroll_degraded_lines"] == 1


def test_r3_coda_counters():
    led = _ledger([
        {"speaker_role": "announcer", "text": "Open."},
        {"speaker_role": "character", "text": "A clean line lands."},
        {"speaker_role": "announcer",
         "text": "The real story: the launch failed.",
         "compose_flags": ["news_coda_fallback", "news_coda_truncated"]},
    ])
    m = _scan.r3_quality_metrics(led)
    assert m["news_coda_truncated_count"] == 1
    assert m["news_coda_fallback_count"] == 1
    assert m["news_coda_generic_bridge_count"] == 1
    assert m["news_coda_mojibake_count"] == 0


def test_r3_dignity_counters():
    led = _ledger(
        [{"speaker_role": "character", "text": "Hello."}],
        meta={
            "dramatic_state_source": "fallback",
            "dramatic_state_fallback_term_replaced": True,
            "central_object": "transgender people",
            "dramatic_state": {
                "character_a_wants": "take sole credit for transgender people",
            },
        },
    )
    m = _scan.r3_quality_metrics(led)
    assert m["dramatic_state_fallback_count"] == 1
    assert m["dramatic_state_fallback_replaced_count"] == 1
    assert m["ownable_people_object_count"] == 1
    assert m["ownership_template_on_nonownable_count"] == 1


def test_r3_register_near_duplicate():
    led = _ledger(
        [{"speaker_role": "character", "speaker": "A", "text": "Hello."},
         {"speaker_role": "character", "speaker": "B", "text": "Hi there."}],
        cast=[
            {"name": "A", "speech_signature": "measured, precise, weary"},
            {"name": "B", "speech_signature": "measured, precise"},
            {"name": "ANNOUNCER", "speech_signature": "narrator"},
        ],
    )
    m = _scan.r3_quality_metrics(led)
    # A + B are the two principals (one line each) and share most of their
    # register -> near-duplicate + high overlap between the principals.
    assert m["speech_signature_near_duplicate_count"] >= 1
    assert m["register_overlap_ratio"] >= 0.5


def test_r3_register_overlap_uses_principals_not_minor_pair():
    # The two PRINCIPALS (most dialogue lines) have DISTINCT registers; a MINOR
    # third voice nearly duplicates a principal. register_overlap must reflect the
    # principals (low), NOT the minor pair the old all-pairs max would surface.
    led = _ledger(
        [{"speaker_role": "character", "speaker": "LEAD1", "text": "The seal cracked."},
         {"speaker_role": "character", "speaker": "LEAD1", "text": "We move at dawn."},
         {"speaker_role": "character", "speaker": "LEAD2", "text": "I disagree, plainly."},
         {"speaker_role": "character", "speaker": "LEAD2", "text": "Hold the line."},
         {"speaker_role": "character", "speaker": "EXTRA", "text": "Yes, sir."}],
        cast=[
            {"name": "LEAD1", "speech_signature": "clipped, procedural, cold"},
            {"name": "LEAD2", "speech_signature": "warm, rambling, wry"},
            {"name": "EXTRA", "speech_signature": "clipped, procedural, cold"},
            {"name": "ANNOUNCER", "speech_signature": "narrator"},
        ],
    )
    m = _scan.r3_quality_metrics(led)
    # principals = LEAD1 (2 lines) + LEAD2 (2 lines); disjoint registers -> ~0.
    assert m["register_overlap_ratio"] < 0.5
    # EXTRA duplicates LEAD1's register -> still caught by the all-pairs near-dup.
    assert m["speech_signature_near_duplicate_count"] >= 1


def test_r3_clean_episode_all_zero():
    led = _ledger(
        [{"speaker_role": "character", "text": "The seal cracked before midnight."}],
        meta={"specificity_anchors": _ANCHORS},
        cast=[
            {"name": "A", "speech_signature": "clipped, procedural"},
            {"name": "B", "speech_signature": "warm, rambling"},
        ],
    )
    m = _scan.r3_quality_metrics(led)
    for k in ("anchor_stuffing_lines", "one_breath_violation_lines",
              "stage_action_leak_lines", "personal_cost_boilerplate_lines",
              "quality_retry_lines", "quality_residual_lines",
              "news_coda_truncated_count", "news_coda_mojibake_count",
              "speech_signature_near_duplicate_count",
              "ownership_template_on_nonownable_count"):
        assert m[k] == 0, k


def test_scan_ledger_merges_r3(tmp_path):
    import json
    p = tmp_path / "ep_ledger.json"
    p.write_text(json.dumps(_ledger(
        [{"speaker_role": "character", "text": "You're playing with fire."}],
        meta={"central_object": "the pencil"})), encoding="utf-8")
    row = _scan.scan_ledger(str(p), None)
    assert "stage_action_leak_lines" in row
    assert "speech_signature_near_duplicate_count" in row
    assert row["r2_helpers_loaded"] is True


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
