"""Portable IndexTTS2 bank creation preserves the shared casting surface."""
from __future__ import annotations

import importlib.util
import hashlib
import json
from pathlib import Path
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "otr_make_portable_voice_bank.py"
    spec = importlib.util.spec_from_file_location("_otr_portable_bank_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _wav(path: Path, sample: int) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22_050)
        handle.writeframes(
            int(sample).to_bytes(2, "little", signed=True) * 22_050)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset(models: Path, row: dict) -> Path:
    return models / Path(row["ref_path"]).relative_to("models")


def test_portable_bank_preserves_non_index_rows_and_replaces_only_index(tmp_path):
    tool = _load()
    male = tmp_path / "male.wav"
    female = tmp_path / "female.wav"
    _wav(male, 21)
    _wav(female, -21)
    models = tmp_path / "models"
    output = tmp_path / "config" / "portable-bank.json"

    result = tool.build_portable_bank(
        shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
        models_root=str(models), male_wav=str(male), female_wav=str(female),
        output=str(output))
    shipped = json.loads((ROOT / "config" / "voice_reference_bank.json").read_text(
        encoding="utf-8"))
    expected_non_index = {row["voice_ref_id"] for row in shipped["voices"]
                          if row.get("engine") != "indextts2"}
    actual_non_index = {row["voice_ref_id"] for row in result["voices"]
                        if row.get("engine") != "indextts2"}
    index = [row for row in result["voices"] if row.get("engine") == "indextts2"]

    assert actual_non_index == expected_non_index
    # THE ROUTE-EXCEPTION ASSERTION IS GONE, and dropping it is more honest
    # than retargeting it. It used to compare the script's output against
    # `LEMMY_VOICE_POLICY`, which was an INDEPENDENT source and therefore
    # caught the script's hardcoded literal drifting from the live casting
    # policy. Pointing it at the script's own constant instead -- the obvious
    # way to survive the policy's deletion -- made it compare the producer to
    # itself: true for any value the constant could hold, so it could not fail.
    # A tautology that reads as coverage is worse than no assertion, and the
    # field itself is removed two commits from here. The other eleven
    # assertions in this test are unaffected and still independent.
    assert [(row["voice_ref_id"], row["gender"]) for row in index] == [
        ("idx_portable_male_v1", "male"),
        ("idx_portable_female_v1", "female"),
    ]
    assert all(row["roles"] == ["char_voice"] for row in index)
    assert all(len(row["ref_sha256"]) == 64 for row in index)
    assert all(row["commercial_clean"] is False for row in index)
    assert output.exists()
    assert all(_asset(models, row).exists() for row in index)
    assert all(_digest(_asset(models, row)) == row["ref_sha256"]
               for row in index)
    assert all(row["ref_sha256"] in Path(row["ref_path"]).stem
               for row in index)
    assert "idx_lemmy_algenib_cockney_v1" not in {
        row["voice_ref_id"] for row in result["voices"]}

    # THE RESERVATION TRAVELS WITH THE ROW, and the replacements do not inherit
    # it. Reservation moved onto the bank row itself on 2026-09-24; before that
    # it was derived from a casting policy, so it was the same answer no matter
    # which bank was loaded. Now it is a property of THIS file, which is exactly
    # why the portable build has to be checked: the two retained clone rows must
    # keep it, and the two GENERATED rows must not gain it -- a portable bank
    # that reserved its own replacements would refuse to cast the only
    # indextts2 voices it ships.
    reserved_here = {row["voice_ref_id"] for row in result["voices"]
                     if str(row.get("reserved_for") or "").strip()}
    assert reserved_here == {"cb_lemmy_algenib_cockney_v1",
                             "dia_lemmy_algenib_cockney_v1"}, reserved_here
    assert all("reserved_for" not in row or not row["reserved_for"]
               for row in index), "a generated replacement row was reserved"


def test_portable_bank_refuses_one_recording_labeled_as_both_genders(tmp_path):
    tool = _load()
    ref = tmp_path / "one.wav"
    _wav(ref, 12)

    try:
        tool.build_portable_bank(
            shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
            models_root=str(tmp_path / "models"), male_wav=str(ref),
            female_wav=str(ref), output=str(tmp_path / "bank.json"))
    except ValueError as exc:
        assert "must be distinct" in str(exc)
    else:
        raise AssertionError("identical male/female references were accepted")

    refs = tmp_path / "models" / "TTS" / "refs" / "indextts2"
    if refs.exists():
        assert not list(refs.glob("otr_portable_*.wav"))


def test_portable_bank_refuses_output_that_aliases_generated_asset(tmp_path):
    tool = _load()
    male = tmp_path / "male.wav"
    female = tmp_path / "female.wav"
    _wav(male, 13)
    _wav(female, -13)
    models = tmp_path / "models"
    output = (models / "TTS" / "refs" / "indextts2" /
              ("otr_portable_male_%s.wav" % _digest(male)))

    with pytest.raises(ValueError, match="must not alias a generated asset"):
        tool.build_portable_bank(
            shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
            models_root=str(models), male_wav=str(male), female_wav=str(female),
            output=str(output))

    assert not output.exists()


def test_portable_bank_rejects_too_short_vendor_incompatible_wav(tmp_path):
    tool = _load()
    short = tmp_path / "short.wav"
    female = tmp_path / "female.wav"
    with wave.open(str(short), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(22_050)
        handle.writeframes(b"\x01\x00" * 64)
    _wav(female, -4)

    with pytest.raises(ValueError, match="too short for IndexTTS2"):
        tool.build_portable_bank(
            shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
            models_root=str(tmp_path / "models"), male_wav=str(short),
            female_wav=str(female), output=str(tmp_path / "bank.json"))


def test_bank_publish_failure_keeps_previous_bank_and_assets_valid(
        tmp_path, monkeypatch):
    tool = _load()
    models = tmp_path / "models"
    output = tmp_path / "portable.json"
    male1, female1 = tmp_path / "male1.wav", tmp_path / "female1.wav"
    male2, female2 = tmp_path / "male2.wav", tmp_path / "female2.wav"
    for path, sample in ((male1, 11), (female1, -11),
                         (male2, 22), (female2, -22)):
        _wav(path, sample)

    first = tool.build_portable_bank(
        shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
        models_root=str(models), male_wav=str(male1), female_wav=str(female1),
        output=str(output))
    before = output.read_bytes()
    old_rows = [row for row in first["voices"]
                if row.get("engine") == "indextts2"]
    real_replace = tool.os.replace

    def fail_bank_publish(source, destination):
        if tool._canonical(destination) == tool._canonical(str(output)):
            raise OSError("injected bank publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(tool.os, "replace", fail_bank_publish)
    with pytest.raises(OSError, match="injected bank publish failure"):
        tool.build_portable_bank(
            shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
            models_root=str(models), male_wav=str(male2),
            female_wav=str(female2), output=str(output))

    assert output.read_bytes() == before
    assert all(_digest(_asset(models, row)) == row["ref_sha256"]
               for row in old_rows)


def test_swapped_content_addressed_sources_remain_distinct(tmp_path):
    tool = _load()
    models = tmp_path / "models"
    first_output = tmp_path / "first.json"
    second_output = tmp_path / "second.json"
    male, female = tmp_path / "male.wav", tmp_path / "female.wav"
    _wav(male, 31)
    _wav(female, -31)
    first = tool.build_portable_bank(
        shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
        models_root=str(models), male_wav=str(male), female_wav=str(female),
        output=str(first_output))
    first_rows = {row["gender"]: row for row in first["voices"]
                  if row.get("engine") == "indextts2"}

    second = tool.build_portable_bank(
        shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
        models_root=str(models),
        male_wav=str(_asset(models, first_rows["female"])),
        female_wav=str(_asset(models, first_rows["male"])),
        output=str(second_output))
    second_rows = {row["gender"]: row for row in second["voices"]
                   if row.get("engine") == "indextts2"}

    assert second_rows["male"]["ref_sha256"] == first_rows["female"]["ref_sha256"]
    assert second_rows["female"]["ref_sha256"] == first_rows["male"]["ref_sha256"]
    assert second_rows["male"]["ref_sha256"] != second_rows["female"]["ref_sha256"]
    assert all(_digest(_asset(models, row)) == row["ref_sha256"]
               for row in second_rows.values())


def test_cli_emits_runtime_override_and_schema_valid_bank(tmp_path, capsys):
    tool = _load()
    male = tmp_path / "male.wav"
    female = tmp_path / "female.wav"
    _wav(male, 2)
    _wav(female, -2)
    output = tmp_path / "portable.json"

    assert tool.main([
        "--models-root", str(tmp_path / "models"),
        "--male-wav", str(male), "--female-wav", str(female),
        "--output", str(output),
    ]) == 0
    assert "OTR_VOICE_REFERENCE_BANK=" in capsys.readouterr().out

    from nodes import _otr_voice_bank
    entries, _digest = _otr_voice_bank.load_voice_bank(str(output))
    assert _otr_voice_bank.unavailable_qualified_route_ids(str(output)) == {
        "lemmy-indextts2-algenib-cockney-v2"}
    assert any(row.engine == "kokoro" for row in entries)
    assert {row.gender for row in entries if row.engine == "indextts2"} == {
        "male", "female"}


@pytest.mark.parametrize("value", [
    "route-a",
    [""],
    [" route-a"],
    ["route-a "],
    ["route-a", "route-a"],
    [17],
])
def test_malformed_unavailable_route_ids_are_rejected(tmp_path, value):
    from nodes import _otr_voice_bank

    source = json.loads((ROOT / "config" / "voice_reference_bank.json").read_text(
        encoding="utf-8"))
    source["unavailable_qualified_route_ids"] = value
    bank = tmp_path / "bad-route-exceptions.json"
    bank.write_text(json.dumps(source), encoding="utf-8")

    with pytest.raises(
            _otr_voice_bank.VoiceBankError,
            match="unavailable_qualified_route_ids"):
        _otr_voice_bank.load_voice_bank(str(bank))


def test_shipped_bank_has_no_route_exception_and_metadata_is_sha_bound(tmp_path):
    from nodes import _otr_voice_bank

    source = json.loads((ROOT / "config" / "voice_reference_bank.json").read_text(
        encoding="utf-8"))
    bank = tmp_path / "bank.json"
    bank.write_text(json.dumps(source), encoding="utf-8")
    _entries, original_sha = _otr_voice_bank.load_voice_bank(str(bank))
    assert _otr_voice_bank.unavailable_qualified_route_ids(
        source_sha256=original_sha) == frozenset()

    source["unavailable_qualified_route_ids"] = ["later-route"]
    bank.write_text(json.dumps(source), encoding="utf-8")
    assert _otr_voice_bank.unavailable_qualified_route_ids(
        source_sha256=original_sha) == frozenset()
    _entries, later_sha = _otr_voice_bank.load_voice_bank(str(bank))
    assert _otr_voice_bank.unavailable_qualified_route_ids(
        source_sha256=later_sha) == {"later-route"}










def test_exact_exception_is_safe_in_preserve_ledger_mode(tmp_path, monkeypatch):
    tool = _load()
    male = tmp_path / "male.wav"
    female = tmp_path / "female.wav"
    _wav(male, 10)
    _wav(female, -10)
    output = tmp_path / "portable.json"
    tool.build_portable_bank(
        shipped_bank=str(ROOT / "config" / "voice_reference_bank.json"),
        models_root=str(tmp_path / "models"), male_wav=str(male),
        female_wav=str(female), output=str(output))
    monkeypatch.setenv("OTR_VOICE_REFERENCE_BANK", str(output))

    from nodes.cast_lock import CastLock

    ledger = json.dumps({
        "meta": {"episode_seed": 42},
        "cast": [{
            "char_id": "c02", "name": "LEMMY", "gender": "male",
            "voice_preset": "v2/en_speaker_8",
        }],
        "lines": [],
    })
    locked = json.loads(CastLock().lock(
        script_json=ledger, cast_voice_policy="preserve_ledger")[0])
    lemmy = locked["cast"][0]

    # The portable exception must not prove a qualified route. A non-bark
    # stamp (provisional or otherwise) clears leftover v2/* so the ledger
    # does not credit Bark for a voice nobody heard (Lime 20260917).
    assert "voice_route" not in lemmy
    engine = str(lemmy.get("voice_engine") or "")
    if engine and engine != "bark":
        assert not str(lemmy.get("voice_preset") or "").startswith("v2/")
    else:
        assert lemmy["voice_preset"] == "v2/en_speaker_8"
