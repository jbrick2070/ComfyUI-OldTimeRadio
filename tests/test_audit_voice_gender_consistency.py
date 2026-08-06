"""Item 8 chunk 5 -- the offline voice/gender consistency audit.

The audit is the standing receipt for this item, so its EXIT CODES are the
contract: 0 clean, 1 a real violation on a policy-era ledger, 2 the scan did not
finish and its verdict must not be read as a pass.
"""
import importlib.util
import json
import os

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPT = os.path.join(_REPO, "scripts", "audit_voice_gender_consistency.py")

_spec = importlib.util.spec_from_file_location("otr_gender_audit", _SCRIPT)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

from nodes._otr_roster_gender import (  # noqa: E402
    VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY,
    VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION,
)


def _write_ledger(root, name, cast, *, policy=True, char_engine="indextts2"):
    meta = {"char_voice_engine": char_engine, "announcer_voice_engine": "kokoro"}
    if policy:
        meta[VOICE_PORTRAIT_CONSISTENCY_POLICY_KEY] = (
            VOICE_PORTRAIT_CONSISTENCY_POLICY_REVISION)
    ep = root / name
    ep.mkdir(parents=True, exist_ok=True)
    (ep / f"{name}_ledger.json").write_text(
        json.dumps({"meta": meta, "cast": cast}), encoding="utf-8")
    return ep


# --------------------------------------------------------------------------- #
# Engine routing -- the whole point of "engine-aware"


def test_bark_speaks_the_preset_other_engines_speak_the_reference():
    assert audit.active_field_for("bark") == "voice_preset"
    assert audit.active_field_for("kokoro") == "voice_ref_id"
    assert audit.active_field_for("elevenlabs") == "provider_voice_id"
    assert audit.active_field_for("google_tts") == "provider_voice_id"
    # indextts2 / chatterbox / dia -- the clone default
    assert audit.active_field_for("indextts2") == "voice_ref_id"
    assert audit.active_field_for("") == "voice_ref_id"


def test_a_dormant_field_is_not_a_violation():
    """1,147 of 1,559 recent rows render on indextts2, which speaks the
    REFERENCE. A wrong-gender bark preset on such a row is a ledger that
    disagrees with itself -- worth reporting, not the same claim as 'the
    audience heard the wrong gender'."""
    ref_g = {"vz_f": "female"}
    preset_g = {"v2/en_speaker_0": "male"}
    row = {"name": "X", "gender": "female", "presentation_gender": "female",
           "voice_ref_id": "vz_f", "voice_preset": "v2/en_speaker_0"}
    res = audit.audit_row(row, "indextts2", ref_g, preset_g)
    assert res["verdict"] == "agree"
    assert res["dormant"] and res["dormant"][0]["field"] == "voice_preset"


def test_the_same_row_IS_a_violation_on_bark():
    """Same data, different engine: bark speaks the preset."""
    ref_g = {"vz_f": "female"}
    preset_g = {"v2/en_speaker_0": "male"}
    row = {"name": "X", "gender": "female", "presentation_gender": "female",
           "voice_ref_id": "vz_f", "voice_preset": "v2/en_speaker_0"}
    res = audit.audit_row(row, "bark", ref_g, preset_g)
    assert res["verdict"] == "DISAGREE"


def test_absent_and_unresolved_are_never_reported_as_agreement():
    ref_g, preset_g = {}, {}
    absent = audit.audit_row({"name": "X", "gender": "male"},
                             "indextts2", ref_g, preset_g)
    assert absent["verdict"] == "active_absent"
    unresolved = audit.audit_row(
        {"name": "X", "gender": "male", "voice_ref_id": "vz_unknown"},
        "indextts2", ref_g, preset_g)
    assert unresolved["verdict"] == "active_unresolved"


def test_presentation_gender_beats_the_label():
    """The announcer's reference is episode-seeded and never read its label, so
    the stamped presentation is the only honest thing to compare."""
    ref_g = {"af_x": "female"}
    row = {"name": "ANNOUNCER", "gender": "male",
           "presentation_gender": "female", "voice_ref_id": "af_x"}
    res = audit.audit_row(row, "kokoro", ref_g, {})
    assert res["verdict"] == "agree"


# --------------------------------------------------------------------------- #
# Exit codes


def test_exit_2_on_a_bad_root(tmp_path, capsys):
    assert audit.main(["--root", str(tmp_path / "nope")]) == 2


def test_exit_2_when_no_ledgers_are_found(tmp_path):
    """A zero-ledger scan reporting 'no violations' is the same false green."""
    assert audit.main(["--root", str(tmp_path)]) == 2


def test_exit_2_on_an_unreadable_ledger(tmp_path):
    ep = tmp_path / "ep1"
    ep.mkdir()
    (ep / "ep1_ledger.json").write_text("{not json", encoding="utf-8")
    assert audit.main(["--root", str(tmp_path)]) == 2


def test_exit_1_on_a_policy_era_violation(tmp_path):
    _write_ledger(tmp_path, "ep1", [{
        "char_id": "c01", "name": "SCROOGE", "gender": "male",
        "presentation_gender": "male",
        "voice_ref_id": "vz_donor_andrea",
    }], char_engine="bark")
    # bark speaks voice_preset; give it a female preset against a male row
    led = tmp_path / "ep1" / "ep1_ledger.json"
    data = json.loads(led.read_text(encoding="utf-8"))
    data["cast"][0]["voice_preset"] = "v2/en_speaker_9"   # female
    led.write_text(json.dumps(data), encoding="utf-8")
    assert audit.main(["--root", str(tmp_path)]) == 1


def test_exit_0_when_policy_ledgers_are_consistent(tmp_path):
    _write_ledger(tmp_path, "ep1", [{
        "char_id": "c01", "name": "SCROOGE", "gender": "male",
        "presentation_gender": "male", "voice_preset": "v2/en_speaker_0",
    }], char_engine="bark")
    assert audit.main(["--root", str(tmp_path)]) == 0


def test_legacy_ledgers_do_not_fail_by_default_but_do_with_the_flag(tmp_path):
    """1,587 ledgers were frozen before this contract existed. They are evidence,
    not violations -- unless the operator asks."""
    _write_ledger(tmp_path, "ep1", [{
        "char_id": "c01", "name": "SCROOGE", "gender": "male",
        "voice_preset": "v2/en_speaker_9",      # female preset, male row
    }], policy=False, char_engine="bark")
    assert audit.main(["--root", str(tmp_path)]) == 0
    assert audit.main(["--root", str(tmp_path), "--fail-on-legacy"]) == 1


def test_authority_failure_is_exit_2_not_a_clean_pass(tmp_path, monkeypatch):
    """An audit that loses its gender authority would resolve nothing and call
    the corpus clean."""
    def _boom():
        raise audit.AuthorityError("bank gone")
    monkeypatch.setattr(audit, "load_authorities", _boom)
    _write_ledger(tmp_path, "ep1", [{"char_id": "c01", "name": "X",
                                     "gender": "male"}])
    assert audit.main(["--root", str(tmp_path)]) == 2
