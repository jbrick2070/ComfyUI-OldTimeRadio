"""Canonical workflow wiring for Google video SFX-bed muxing."""
from __future__ import annotations

import json
import pathlib

from nodes.otr_master_audio_mux import OTRMasterAudioMux, _reresolve_master_audio

WF = pathlib.Path(__file__).resolve().parent.parent / "workflows" / "otr_canonical.json"


def _workflow():
    return json.loads(WF.read_text(encoding="utf-8"))


def test_master_audio_reresolve_uses_active_ledger_not_newest_sibling(
    tmp_path, monkeypatch, caplog,
):
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    caplog.set_level("INFO")
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
    assert any("PATH RECONCILED" in r.message and r.levelname == "INFO"
               for r in caplog.records)
    assert not any(r.levelno >= 30 for r in caplog.records)


def test_master_audio_reresolve_follows_rename_when_test_mode_leaked_outside_pytest(
    tmp_path, monkeypatch, caplog,
):
    """Live Comfy can inherit OTR_TEST_MODE=1 from a pytest parent shell.
    That must not skip pending_ re-resolve -- the 2026-09-16 Three Boxes
    mux miss. Pytest itself still skips, via PYTEST_CURRENT_TEST.
    """
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    caplog.set_level("INFO")
    episodes = tmp_path / "otr" / "episodes"
    active_audio = episodes / "signal_lost_active" / "audio"
    active_audio.mkdir(parents=True)
    basename = "pending_123_master.wav"
    active_master = active_audio / basename
    active_master.write_bytes(b"active")
    active_ledger = active_audio / "signal_lost_active_ledger.json"
    active_ledger.write_text('{"episode_id":"signal_lost_active"}', encoding="utf-8")
    stale = str(episodes / "pending_123" / "audio" / basename)
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path", lambda: active_ledger,
    )

    assert _reresolve_master_audio(stale) == str(active_master)
    assert any("PATH RECONCILED" in r.message and r.levelname == "INFO"
               for r in caplog.records)


def test_master_audio_reresolve_skipped_under_pytest_test_mode(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_google_video_sfx_workflow.py")
    stale = str(tmp_path / "pending" / "audio" / "pending_123_master.wav")
    monkeypatch.setattr(
        "nodes._otr_ledger.in_flight_ledger_path",
        lambda: (_ for _ in ()).throw(AssertionError("pytest must not consult ledger")),
    )

    assert _reresolve_master_audio(stale) == stale


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
    tmp_path, monkeypatch, caplog,
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
    assert any("REJECTED" in r.message and r.levelname == "WARNING"
               for r in caplog.records)


def test_master_audio_mux_declares_connector_only_clip_manifest_input():
    it = OTRMasterAudioMux.INPUT_TYPES()
    spec = it["optional"]["clip_manifest_json"]
    assert spec[0] == "STRING"
    assert spec[1]["forceInput"] is True
    # The `ffmpeg` widget was REMOVED 2026-09-13 (it had been discarded at
    # every execute-method boundary since 2026-09-04, so the declaration was
    # a channel to nowhere). The absence is asserted directly so a re-added
    # widget has to answer for itself here, not just fail a shape check.
    assert "ffmpeg" not in it["optional"]
    # Connector-only SFX input must not add to the saved widget vector.
    widget_backed = [
        name for name, field in it["optional"].items()
        if field[0] in ("INT", "FLOAT", "STRING", "BOOLEAN")
        and not field[1].get("forceInput")
    ]
    assert widget_backed == ["fps", "output_path"]


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
    # 292 since 8171e994 took the bypassed overlay out of the canonical. That
    # commit removed nodes 93/94 and RE-ROUTED what they sat between, so the
    # new link 292 is OTR_SilentComposite.video_path -> OTR_CaptionBurn
    # ([292, 84, 0, 86, 0, "STRING"]) -- the caption burn now reads the
    # composite directly instead of through a blend that shipped bypassed.
    #
    # This is the SEVENTH bump of a counter that is not this test's subject,
    # so here is the proof it is additive rather than the usual assurance:
    # link 292 touches neither node 85 nor node 92, and link 278 and its audio
    # fanout below are byte-identical. Read the tuple above -- src 84, dst 86 --
    # and neither id appears in any assertion in this function.
    # EIGHTH bump, 293 (plan 0k, 2026-09-26): [293, 97, 0, 63, 6, "STRING"],
    # the new OTR_ComfyCredential node's token into the Workflow Validator's
    # appended `credential` socket. Neither node 85 nor node 92 is touched,
    # and link 278 and its fanout below are byte-identical.
    assert wf["last_link_id"] == 293
    # `ffmpeg` left node 85's inputs on 2026-09-13 (the widget was discarded at
    # the method boundary since 2026-09-04 and had no live effect; the
    # declaration closed the channel rather than keep sanitising it). It sat
    # mid-list, so removal re-indexed every input after it -- output_path,
    # video_policy_json, foley_receipts_json and script_json each moved down
    # one slot. Repaired by IDENTITY in the workflow JSON (each link's
    # dst_slot set to the index whose inputs[i].link equals that link's id),
    # so only link 291 (script_json) below actually changes value.
    assert [i["name"] for i in n85["inputs"]] == [
        "silent_video_path",
        "master_audio_path",
        "audio_done",
        "declared_credits_tail_s",
        "clip_manifest_json",
        "fps",
        "output_path",
        "video_policy_json",
        "foley_receipts_json",
        "script_json",
    ]
    assert [i.get("link") for i in n85["inputs"][:5]] == [274, 263, 249, 276, 278]
    assert n85["widgets_values"] == [25, ""]
    assert n92["outputs"][1]["name"] == "clip_manifest_json"
    # 271 is gone, and its absence is the CLEAN consequence of 8171e994 rather
    # than collateral: link 271 was [271, 92, 1, 94, 1, "STRING"] -- this same
    # clip_manifest_json output feeding node 94, OTR_SceneAwareScopes. Node 94
    # left the canonical, so the wire into it left with it, and no other
    # consumer lost a feed.
    #
    # The subject of this test is untouched: link 278 is still
    # [278, 92, 1, 85, 4, "STRING"], byte-identical, and still in this fanout.
    assert n92["outputs"][1]["links"] == [261, 275, 278, 288]
    links = {l[0]: l for l in wf["links"]}
    # dst_slot 9, not 10 -- script_json shifted down one slot when the
    # ffmpeg widget in front of it was removed (see the note above).
    assert links[291] == [291, 1, 1, 85, 9, "STRING"]
    assert links[278] == [278, 92, 1, 85, 4, "STRING"]
    assert links[261] == [261, 92, 1, 84, 2, "STRING"]
    # 271 used to be asserted here as [271, 92, 1, 94, 1, "STRING"]. It is
    # ABSENT on purpose: node 94 (OTR_SceneAwareScopes) left the canonical in
    # 8171e994, so the wire into it went too. Assert the absence rather than
    # dropping the line, so a node 94 that comes back has to answer for itself.
    assert 271 not in links, (
        "link 271 is back -- it fed OTR_SceneAwareScopes, which 8171e994 took "
        "off the canonical because the blend it served shipped bypassed"
    )
    assert links[275] == [275, 92, 1, 95, 1, "STRING"]
    assert links[274][3:5] == [85, 0]
    assert links[263][3:5] == [85, 1]
    assert links[249][3:5] == [85, 2]
    assert links[276][3:5] == [85, 3]
