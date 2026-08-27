"""Canonical workflow wiring for Google video SFX-bed muxing."""
from __future__ import annotations

import json
import pathlib

from nodes.otr_master_audio_mux import OTRMasterAudioMux, _reresolve_master_audio

WF = pathlib.Path(__file__).resolve().parent.parent / "workflows" / "otr_canonical.json"


def _workflow():
    return json.loads(WF.read_text(encoding="utf-8"))


def test_master_audio_reresolve_uses_active_ledger_not_newest_sibling(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    episodes = tmp_path / "otr" / "episodes"
    active_audio = episodes / "signal_lost_active" / "audio"
    sibling_audio = episodes / "signal_lost_newer_sibling" / "audio"
    active_audio.mkdir(parents=True)
    sibling_audio.mkdir(parents=True)
    basename = "pending_123_master.wav"
    active_master = active_audio / basename
    sibling_master = sibling_audio / basename
    active_master.write_bytes(b"active")
    sibling_master.write_bytes(b"sibling")
    active_ledger = active_audio / "signal_lost_active_ledger.json"
    active_ledger.write_text('{"episode_id":"signal_lost_active"}', encoding="utf-8")
    stale = str(episodes / "pending_123" / "audio" / basename)
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path", lambda: active_ledger,
    )

    assert _reresolve_master_audio(stale) == str(active_master)


def test_master_audio_reresolve_fails_closed_without_active_ledger(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    stale = str(tmp_path / "pending" / "audio" / "master.wav")
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path", lambda: None,
    )

    assert _reresolve_master_audio(stale) == stale


def test_master_audio_reresolve_rejects_ledger_directory_identity_mismatch(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    audio = tmp_path / "otr" / "episodes" / "signal_lost_active" / "audio"
    audio.mkdir(parents=True)
    basename = "pending_123_master.wav"
    (audio / basename).write_bytes(b"active")
    active_ledger = audio / "signal_lost_active_ledger.json"
    active_ledger.write_text(
        '{"episode_id":"signal_lost_different"}', encoding="utf-8",
    )
    stale = str(tmp_path / "episodes" / "pending_123" / "audio" / basename)
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path", lambda: active_ledger,
    )

    assert _reresolve_master_audio(stale) == stale


def test_master_audio_mux_declares_connector_only_clip_manifest_input():
    it = OTRMasterAudioMux.INPUT_TYPES()
    spec = it["optional"]["clip_manifest_json"]
    assert spec[0] == "STRING"
    assert spec[1]["forceInput"] is True
    # Connector-only SFX input must not add to the saved widget vector.
    widget_backed = [
        name for name, field in it["optional"].items()
        if field[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")
        and not field[1].get("forceInput")
    ]
    assert widget_backed == ["fps", "ffmpeg", "output_path"]


def test_canonical_workflow_wires_clip_manifest_to_master_audio_mux():
    wf = _workflow()
    nodes = {n["id"]: n for n in wf["nodes"]}
    n85 = nodes[85]
    n92 = nodes[92]
    # S5 platform-portability (2026-07-10): OLD pin 278 -> NEW pin 279 (link
    # 279 = OTR_WorkflowValidator.validation_report -> node-1 gate_in; this
    # file's link IDs 85/92-side are unaffected, only the vector's ceiling).
    # 720-bakeoff C3 (2026-07-11): music cue fanout added links 280-283 (node
    # 83 -> nodes 3/7). PBUG-20260721-14 adds link 284, the post-rename
    # SignalLostVideo completion gate into ShotLock; the 85/92 SFX-manifest
    # link IDs below are untouched.
    #
    # Title-card legibility (2026-08-12) adds link 285: SignalLostVideo's new
    # title_card_plan_json output into OTR_CaptionBurn, so the hero title can be
    # drawn AFTER the procgen blend where an outline is not a no-op. Again a
    # NEW link id appended; nothing on the 85/92 side moves.
    #
    # The LTX 2.5 FOLEY BED (2026-08-26) adds links 286-288: the video policy
    # into the EpisodeAssembler and into this mux, and the clip manifest into
    # this mux's NEW foley_receipts_json connector. This is the SIXTH amendment
    # for a link that is not this test's subject -- and the first one that
    # touches node 85 at all, so read the next paragraph before assuming it is
    # the same kind of bump as the five before it.
    #
    # THE FOLEY RECEIPTS RIDE A NEW CONNECTOR, NOT THIS ONE, AND THAT WAS AN
    # OPERATOR DECISION (RULING 5, docs/2026-08-26-foley-bed-OPERATOR-RULINGS).
    # Reusing `clip_manifest_json` would have made this file's own subject --
    # a retired connector that is accepted, hashed and unused -- quietly false.
    # Every assertion below about link 278, its slot, and its fanout is
    # therefore UNCHANGED and still asserting exactly what it always did. What
    # changed is only what was APPENDED after it, which is the point of the
    # canary: it fires on any graph edit, and the diff it forces you to look at
    # is what proves the edit was additive.
    #
    # NOTE for whoever bumps this next -- retiring the global-counter line in
    # favour of the scoped assertions would be a deliberate contract change and
    # belongs in its own commit, not in passing.
    assert wf["last_link_id"] == 288
    assert [i["name"] for i in n85["inputs"]] == [
        "silent_video_path",
        "master_audio_path",
        "audio_done",
        "declared_credits_tail_s",
        "clip_manifest_json",
        "fps",
        "ffmpeg",
        "output_path",
        "video_policy_json",
        "foley_receipts_json",
    ]
    assert [i.get("link") for i in n85["inputs"][:5]] == [274, 263, 249, 276, 278]
    assert n85["widgets_values"] == [25, "ffmpeg", ""]
    assert n92["outputs"][1]["name"] == "clip_manifest_json"
    assert n92["outputs"][1]["links"] == [261, 271, 275, 278, 288]
    links = {l[0]: l for l in wf["links"]}
    assert links[278] == [278, 92, 1, 85, 4, "STRING"]
    assert links[261] == [261, 92, 1, 84, 2, "STRING"]
    assert links[271] == [271, 92, 1, 94, 1, "STRING"]
    assert links[275] == [275, 92, 1, 95, 1, "STRING"]
    assert links[274][3:5] == [85, 0]
    assert links[263][3:5] == [85, 1]
    assert links[249][3:5] == [85, 2]
    assert links[276][3:5] == [85, 3]
