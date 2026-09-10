"""Required delivery follows this run through rename, publication and cache reuse."""
import json
from pathlib import Path

import pytest

from nodes import _otr_ledger as OTRL, _otr_publication_eligibility as PE
from nodes import otr_master_audio_mux as MUX


def ledger(required=True, bank="my_story", token="run-one"):
    return {"episode_id": "pending_old", "meta": {"source_bank": bank,
            "delivery_intent": {"schema_version": MUX.DELIVERY_INTENT_VERSION,
                "source_bank": bank, "publication_required": required,
                "delivery_token": token, "draft_digest": "a" * 64 if required else ""}}}


@pytest.mark.parametrize("wire", ["", "not-json", "[]", "null", "{\"meta\": []}"])
def test_present_bad_wire_refuses(wire):
    with pytest.raises(MUX.DeliveryContractError):
        MUX._delivery_intent(wire)


def test_absent_wire_and_legacy_ledger_keep_legacy_behavior():
    assert MUX._delivery_intent(None) is None
    assert MUX._delivery_intent(json.dumps({"meta": {"source_bank": "original"}})) is None


@pytest.mark.parametrize("change", ["absent", "null", "false", "token", "bank", "digest", "version"])
def test_my_story_cannot_omit_or_weaken_its_contract(change):
    data = ledger()
    intent = data["meta"]["delivery_intent"]
    if change == "absent":
        del data["meta"]["delivery_intent"]
    elif change == "null":
        data["meta"]["delivery_intent"] = None
    else:
        key, value = {"false": ("publication_required", False), "token": ("delivery_token", 123),
                      "bank": ("source_bank", "original"), "digest": ("draft_digest", ""),
                      "version": ("schema_version", "unknown")}[change]
        intent[key] = value
    with pytest.raises(MUX.DeliveryContractError):
        MUX._delivery_intent(json.dumps(data))


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))
    root = tmp_path / "otr" / "episodes"
    directory = root / "the_fog_bell"
    audio = directory / "audio"
    audio.mkdir(parents=True)
    obs = tmp_path / "otr" / "obs" / "the_fog_bell_final.mp4"
    obs.parent.mkdir(parents=True)
    path = audio / "the_fog_bell_ledger.json"
    data = ledger()
    data["episode_id"] = directory.name
    PE.stamp_publication_eligibility(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: path)
    monkeypatch.setattr(MUX, "_episodes_root", lambda: root)
    monkeypatch.setattr(MUX, "_master_wav_owes_a_delivery_gain", lambda *a: False)
    monkeypatch.setattr(MUX, "_reresolve_master_audio", lambda p: p)
    final = directory / "the_fog_bell_final.mp4"
    calls = []
    def mux(video, master, output, **kw):
        calls.append("mux")
        final.write_bytes(b"archival test artifact")
        return str(final), ["mux checked"]
    def publish(self, source):
        calls.append("publish")
        obs.write_bytes(Path(source).read_bytes())
        return str(obs)
    monkeypatch.setattr(MUX, "mux_master_audio", mux)
    monkeypatch.setattr(MUX.OTRMasterAudioMux, "_publish_to_obs", publish)
    args = dict(silent_video_path=str(directory / "the_fog_bell_silent.mp4"),
                master_audio_path=str(audio / "master.wav"), script_json=json.dumps(ledger()))
    return args, data, path, obs, final, calls


def test_renamed_episode_publishes_and_stamps_the_real_file(episode):
    args, _, path, obs, final, calls = episode
    result, report = MUX.OTRMasterAudioMux().mux(**args)
    assert Path(result) == final and obs.is_file()
    assert calls == ["mux", "publish"]
    assert json.loads(path.read_text())["meta"]["obs_final_path"] == str(obs)
    assert "delivery OK" in report


@pytest.mark.parametrize("mismatch", ["token", "stem"])
def test_unrelated_input_fails_before_any_media_write(episode, mismatch):
    args, _, _, _, _, calls = episode
    if mismatch == "token":
        args["script_json"] = json.dumps(ledger(token="other-run"))
    else:
        args["silent_video_path"] = args["silent_video_path"].replace("the_fog_bell_silent", "other_silent")
    with pytest.raises(MUX.DeliveryContractError):
        MUX.OTRMasterAudioMux().mux(**args)
    assert calls == []


def test_withheld_required_delivery_fails_but_keeps_archive(episode):
    args, data, path, obs, final, calls = episode
    data["meta"]["provenance"] = {"status": "research_only", "blocks_publish": True}
    PE.stamp_publication_eligibility(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(MUX.DeliveryContractError, match="withheld"):
        MUX.OTRMasterAudioMux().mux(**args)
    assert final.is_file() and not obs.exists() and calls == ["mux"]


def test_failed_stamp_cannot_claim_delivery(episode, monkeypatch):
    args, *_ = episode
    monkeypatch.setattr(MUX.OTRMasterAudioMux, "_stamp_terminal_paths", lambda *a: "stamp skipped")
    with pytest.raises(MUX.DeliveryContractError, match="records"):
        MUX.OTRMasterAudioMux().mux(**args)


def test_cache_changes_when_required_obs_file_disappears(episode):
    args, _, _, obs, _, _ = episode
    MUX.OTRMasterAudioMux().mux(**args)
    before = MUX.OTRMasterAudioMux.IS_CHANGED(**args)
    assert before == MUX.OTRMasterAudioMux.IS_CHANGED(**args)
    obs.unlink()
    assert before != MUX.OTRMasterAudioMux.IS_CHANGED(**args)
