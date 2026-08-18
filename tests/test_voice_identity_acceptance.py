"""The voice-identity acceptance instrument reads receipts. Test that it reads.

WHY THIS FILE EXISTS. The instrument returned three findings on the FIRST live
arm and all three were its own fault: it checked "every line is char_v1 with
alpha 0.4" across the whole log, and a real episode carries TWO lanes --
char_voice on a clone engine (opted into character seeding, carrying an emotion
vector) and announcer_voice on kokoro (deliberately NOT opted in, no emotion
vector at all). The announcer's two perfectly correct rows failed every check.

That is the dominant defect shape in this repo -- ONE FIELD, TWO MEANINGS --
and it is worse than harmless here: an instrument that cries wolf on every arm
teaches the next reader to disbelieve the arms that matter. The fixtures below
are verbatim P-OBS lines from the live arm A render (episode published to
otr/obs/ 2026-08-18 02:38), so they cannot drift from what the dispatch writes
without this file noticing.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
_SPEC = importlib.util.spec_from_file_location(
    "otr_voice_identity_acceptance",
    REPO / "scripts" / "otr_voice_identity_acceptance.py")
ACC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ACC)


# Verbatim from the live arm A server log.
CHARACTER_LINES = """\
[OTR voice P-OBS] char_voice: line=b002 char=c02 -> voice_ref_id=vz_caro_davy ref=vz_caro_davy.wav engine=indextts2 alpha=0.4 delivery=v2:nonzero(derived) emo_mass=0.3996(capped) seed=6612433960155474569 policy=char_v1 seed_ref=vz_caro_davy
[OTR voice P-OBS] char_voice: line=b004 char=c02 -> voice_ref_id=vz_caro_davy ref=vz_caro_davy.wav engine=indextts2 alpha=0.4 delivery=v2:nonzero(derived) emo_mass=0.4 seed=6612433960155474569 policy=char_v1 seed_ref=vz_caro_davy
[OTR voice P-OBS] char_voice: line=b003 char=c03 -> voice_ref_id=vz_donor_andrea ref=vz_donor_andrea.wav engine=indextts2 alpha=0.4 delivery=v2:nonzero(derived) emo_mass=0.32 seed=4555013281669016880 policy=char_v1 seed_ref=vz_donor_andrea
"""

ANNOUNCER_LINES = """\
[OTR voice P-OBS] announcer_voice: line=b001 char=announcer -> voice_ref_id=bf_emma ref=bf_emma engine=kokoro alpha=n/a delivery=v2:nonzero(derived) emo_mass=n/a seed=1140790997092050882 policy=line_v1 seed_ref=bf_emma
[OTR voice P-OBS] announcer_voice: line=b011 char=announcer -> voice_ref_id=bf_emma ref=bf_emma engine=kokoro alpha=n/a delivery=v2:nonzero(derived) emo_mass=n/a seed=2532113877630814928 policy=line_v1 seed_ref=bf_emma
"""


def _audit(text):
    return ACC.audit_rows(ACC.parse_pobs(text))


# --------------------------------------------------------------------------- #
# THE BUG THAT SHIPPED ON ARM A
# --------------------------------------------------------------------------- #
def test_a_real_two_lane_episode_reads_CLEAN():
    """Both lanes together, exactly as a live episode emits them."""
    report = _audit(CHARACTER_LINES + ANNOUNCER_LINES)

    assert ACC.verdict(report, "char_v1", "0.4") == []


def test_the_announcer_lane_does_not_pollute_the_character_checks():
    """The announcer is on the legacy seed BY DESIGN [QA-6] -- he is not a
    cloned character and has no identity to hold steady across beats. His rows
    must not appear in the character lane's policy or alpha tallies."""
    report = _audit(CHARACTER_LINES + ANNOUNCER_LINES)

    assert report["character_lane_lines"] == 3
    assert report["other_lane_lines"] == 2
    assert report["policies"] == {"char_v1": 3}
    assert report["alphas"] == {"0.4": 3}


def test_the_announcer_drawing_a_seed_per_line_is_not_a_split_character():
    """He legitimately draws one per line. Counting that as the defect is what
    made the first live arm look broken when it was clean."""
    report = _audit(CHARACTER_LINES + ANNOUNCER_LINES)

    assert report["characters_with_split_seeds"] == {}


# --------------------------------------------------------------------------- #
# IT STILL CATCHES THE THINGS IT IS FOR
# --------------------------------------------------------------------------- #
def test_a_split_character_seed_is_still_reported():
    """The defect itself: one character, two seeds across his own lines."""
    broken = CHARACTER_LINES.replace(
        "seed=6612433960155474569 policy=char_v1 seed_ref=vz_caro_davy\n"
        "[OTR voice P-OBS] char_voice: line=b003",
        "seed=9999999999999999999 policy=char_v1 seed_ref=vz_caro_davy\n"
        "[OTR voice P-OBS] char_voice: line=b003", 1)
    report = _audit(broken)

    assert "c02" in report["characters_with_split_seeds"]
    assert any("DEFECT IS STILL PRESENT" in f
               for f in ACC.verdict(report, "char_v1", "0.4"))


def test_emotion_mass_over_the_ceiling_is_still_reported():
    report = _audit(CHARACTER_LINES.replace("emo_mass=0.4 ", "emo_mass=0.97 ", 1))

    assert report["lines_over_the_cap"]
    assert any("OVER THE CEILING" in f
               for f in ACC.verdict(report, "char_v1", "0.4"))


def test_a_scope_leak_onto_the_announcer_is_reported():
    """THE INVERSE CHECK [QA-6]. If the opt-in ever reaches an announcer
    profile, the announcer lane turns up on char_v1 and that is a real defect --
    measured on live evidence rather than assumed from the YAML."""
    leaked = ANNOUNCER_LINES.replace("policy=line_v1", "policy=char_v1")
    report = _audit(CHARACTER_LINES + leaked)

    assert report["other_lane_rows_on_the_character_policy"]
    assert any("SCOPE LEAK" in f for f in ACC.verdict(report, "char_v1", "0.4"))


def test_the_control_arm_must_actually_reproduce_the_defect():
    """A line_v1 arm with no split seeds means the arm did not boot with the
    character seed disabled, so the 2x2 has no contrast to measure."""
    control = CHARACTER_LINES.replace("policy=char_v1", "policy=line_v1")
    report = _audit(control)

    assert any("CONTROL ARM DID NOT REPRODUCE" in f
               for f in ACC.verdict(report, "line_v1", "0.4"))


def test_a_log_with_no_character_lane_proves_nothing():
    report = _audit(ANNOUNCER_LINES)

    assert any("NO CHARACTER-LANE RECEIPTS" in f
               for f in ACC.verdict(report, "char_v1", "0.4"))


def test_the_massless_lane_does_not_crash_the_ceiling_check():
    """chatterbox and dia carry the character SEED but emit `emo_mass=n/a`;
    both opted in, so their rows must parse rather than being dropped."""
    dia = CHARACTER_LINES.replace("engine=indextts2", "engine=dia").replace(
        "emo_mass=0.3996(capped)", "emo_mass=n/a").replace(
        "emo_mass=0.4 ", "emo_mass=n/a ").replace("emo_mass=0.32", "emo_mass=n/a")
    report = _audit(dia)

    assert report["character_lane_lines"] == 3
    assert report["rows_without_emotion"] == 3
    assert report["lines_over_the_cap"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
