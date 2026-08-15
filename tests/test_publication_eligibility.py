"""Publication eligibility, end to end -- one producer, one consumer, no guess.

THE DEFECT THIS SUITE PINS. The operator rule "a research_only source BLOCKS
publish" was realised by raising at the FREEZE, which killed a finished render
rather than withholding a copy: the operator lost the whole episode, including
the archival copy a research-only source is actually cleared for. The rule now
lands at the publication boundary. These tests assert the rule still bites, and
that the render survives it.

What is deliberately asymmetric, and why each half is tested separately: the
PRODUCER never blocks for a reason it cannot substantiate (an absent rights
record is the ordinary state on four of six banks), while the CONSUMER blocks on
anything it cannot read (a missing receipt is not permission). Test both
directions or the asymmetry silently collapses into whichever one somebody
edited last.

CPU-only: plain dicts, tmp_path, no ffmpeg, no GPU, no model load.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_ledger as OTRL
from nodes import _otr_ledger_freeze as LF
from nodes import _otr_publication_eligibility as PE
from nodes import otr_master_audio_mux as MUX


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _ledger(provenance=None, source_meta=None, *, episode_id="the_test_episode",
            source_bank="public_domain"):
    """A ledger complete enough to FREEZE.

    The structural keys matter: several tests below drive the real Phase 10,
    and a fixture missing `cast`/`beats`/`shots` fails the freeze for reasons
    that have nothing to do with publication -- which would make a
    "research_only still freezes" test pass or fail for the wrong reason.
    """
    meta = {"source_bank": source_bank, "episode_title": "T", "style": "s"}
    if provenance is not None:
        meta["provenance"] = provenance
    if source_meta is not None:
        meta["source_meta"] = source_meta
    return {
        "schema_version": LF.EXPECTED_SCHEMA_VERSION,
        "episode_id": episode_id,
        "meta": meta,
        "cast": [{"char_id": "c01", "name": "NARRATOR"}],
        "lines": [{"line_id": "b1", "char_id": "c01",
                   "speaker_role": "character", "text": "hi"}],
        "beats": [],
        "scenes": [],
        "shots": [],
        "music": [],
        "clips": [],
    }


_CLEARED = {"status": "public_domain_us", "blocks_publish": False}
_RESEARCH = {"status": "research_only", "blocks_publish": True,
             "source_label": "Restricted Collection"}
_NAMED = {"title": "The Time Machine", "author": "H. G. Wells"}


# --------------------------------------------------------------------------- #
# the producer: rights decide, identity only explains
# --------------------------------------------------------------------------- #
class TestProducer:
    def test_a_cleared_source_is_eligible(self):
        el = PE.evaluate_publication_eligibility(_ledger(_CLEARED, _NAMED))
        assert el.eligible is True
        assert PE.REASON_RIGHTS_CLEARED in el.reasons
        assert el.blocking_reasons == ()

    def test_research_only_is_the_block(self):
        el = PE.evaluate_publication_eligibility(_ledger(_RESEARCH, _NAMED))
        assert el.eligible is False
        assert el.blocking_reasons == (PE.REASON_RIGHTS_RESEARCH_ONLY,)

    def test_an_unstamped_rights_record_is_NOT_a_block(self):
        """The normaliser is opt-in per bank. Four of six banks never stamp it,
        so reading its absence as a denial would withhold every original,
        media_archive and news episode ever rendered."""
        el = PE.evaluate_publication_eligibility(_ledger(None))
        assert el.eligible is True
        assert PE.REASON_RIGHTS_NOT_STAMPED in el.reasons

    def test_a_malformed_rights_record_is_named_but_not_a_block(self):
        el = PE.evaluate_publication_eligibility(_ledger("research_only"))
        assert el.eligible is True
        assert PE.REASON_RIGHTS_MALFORMED in el.reasons

    def test_a_degraded_identity_never_blocks(self):
        """Zero of 65 units lack a title or author (schema-enforced), so an
        identity rule could only ever fire on a corpus fault -- and the answer
        to a corpus fault is a receipt, not a withheld episode."""
        el = PE.evaluate_publication_eligibility(_ledger(_CLEARED, {"title": ""}))
        assert el.eligible is True
        assert PE.REASON_IDENTITY_DEGRADED in el.reasons

    def test_a_complete_identity_is_recorded(self):
        el = PE.evaluate_publication_eligibility(_ledger(_CLEARED, _NAMED))
        assert PE.REASON_IDENTITY_COMPLETE in el.reasons

    def test_a_lane_with_no_bibliographic_identity_is_not_applicable(self):
        el = PE.evaluate_publication_eligibility(
            _ledger(None, None, source_bank="original"))
        assert PE.REASON_IDENTITY_NOT_APPLICABLE in el.reasons

    def test_a_junk_ledger_cannot_raise(self):
        for junk in (None, [], "ledger", 7):
            assert PE.evaluate_publication_eligibility(junk).eligible is True

    def test_evaluate_does_not_stamp_and_stamp_does(self):
        led = _ledger(_CLEARED, _NAMED)
        PE.evaluate_publication_eligibility(led)
        assert PE.PUBLICATION_ELIGIBILITY_META_KEY not in led["meta"]
        PE.stamp_publication_eligibility(led)
        assert led["meta"][PE.PUBLICATION_ELIGIBILITY_META_KEY]["eligible"] is True

    def test_the_stamp_is_idempotent(self):
        """A receipt that changes when nothing changed re-runs the terminal mux
        on every save, because the mux's cache key is built on its digest."""
        led = _ledger(_CLEARED, _NAMED)
        first = PE.stamp_publication_eligibility(led)
        snapshot = json.dumps(led["meta"][PE.PUBLICATION_ELIGIBILITY_META_KEY],
                              sort_keys=True)
        second = PE.stamp_publication_eligibility(led)
        assert first.digest == second.digest
        assert json.dumps(led["meta"][PE.PUBLICATION_ELIGIBILITY_META_KEY],
                          sort_keys=True) == snapshot


class TestDigest:
    def test_the_digest_is_a_stable_content_hash_not_pythons_salted_hash(self):
        """`hash()` on a str is salted per interpreter, so the key it produces
        changes at every server boot -- harmless until it gates a deliverable."""
        el = PE.evaluate_publication_eligibility(_ledger(_CLEARED, _NAMED))
        assert len(el.digest) == 64
        int(el.digest, 16)  # raises if it is not hex
        assert el.digest == PE.evaluate_publication_eligibility(
            _ledger(_CLEARED, _NAMED)).digest

    def test_the_digest_moves_when_the_verdict_moves(self):
        cleared = PE.evaluate_publication_eligibility(_ledger(_CLEARED, _NAMED))
        blocked = PE.evaluate_publication_eligibility(_ledger(_RESEARCH, _NAMED))
        assert cleared.digest != blocked.digest


# --------------------------------------------------------------------------- #
# the consumer: anything unreadable is blocked
# --------------------------------------------------------------------------- #
class TestConsumerFailsClosed:
    def test_an_eligible_receipt_publishes(self):
        led = _ledger(_CLEARED, _NAMED)
        PE.stamp_publication_eligibility(led)
        d = PE.decide_from_meta(led["meta"], expected_episode_id=led["episode_id"])
        assert d.publishable is True
        assert d.digest

    def test_an_ineligible_receipt_names_its_blocking_reason(self):
        led = _ledger(_RESEARCH, _NAMED)
        PE.stamp_publication_eligibility(led)
        d = PE.decide_from_meta(led["meta"])
        assert d.publishable is False
        assert d.reason == "ineligible"
        assert PE.REASON_RIGHTS_RESEARCH_ONLY in d.detail

    def test_no_receipt_is_not_permission(self):
        d = PE.decide_from_meta({"source_bank": "public_domain"})
        assert d.publishable is False
        assert d.reason == PE.DECISION_NO_RECEIPT

    @pytest.mark.parametrize("receipt", ["yes", 1, [], True])
    def test_a_malformed_receipt_is_blocked(self, receipt):
        d = PE.decide_from_meta({PE.PUBLICATION_ELIGIBILITY_META_KEY: receipt})
        assert d.publishable is False
        assert d.reason == PE.DECISION_MALFORMED

    def test_an_unknown_version_is_blocked(self):
        """An old node must never approve a publication under rules it has
        never seen."""
        d = PE.decide_from_meta({PE.PUBLICATION_ELIGIBILITY_META_KEY: {
            "version": "publication_eligibility_v99", "eligible": True,
            "episode_id": "e", "reasons": [],
        }})
        assert d.publishable is False
        assert d.reason == PE.DECISION_VERSION_UNKNOWN

    def test_a_non_bool_eligible_field_is_blocked(self):
        d = PE.decide_from_meta({PE.PUBLICATION_ELIGIBILITY_META_KEY: {
            "version": PE.PUBLICATION_ELIGIBILITY_VERSION,
            "eligible": "true", "episode_id": "e", "reasons": [],
        }})
        assert d.publishable is False
        assert d.reason == PE.DECISION_MALFORMED

    def test_a_receipt_for_a_DIFFERENT_episode_is_blocked(self):
        """The stale-singleton case. One episode must never be gated on
        another's verdict, in either direction."""
        led = _ledger(_CLEARED, _NAMED, episode_id="an_earlier_episode")
        PE.stamp_publication_eligibility(led)
        d = PE.decide_from_meta(led["meta"], expected_episode_id="tonights_episode")
        assert d.publishable is False
        assert d.reason == PE.DECISION_EPISODE_MISMATCH

    def test_unreadable_meta_is_blocked(self):
        assert PE.decide_from_meta(None).publishable is False
        assert PE.decide_from_meta("meta").publishable is False


# --------------------------------------------------------------------------- #
# the freeze: G14 warns, the render survives, and the receipt is stamped there
# --------------------------------------------------------------------------- #
class TestFreezeIsTheProducer:
    def test_g14_is_a_WARNING_not_a_freeze_error(self):
        report = LF.run_gap_audit(_ledger(_RESEARCH, _NAMED), label="pre")
        assert not [e for e in report.errors if "G14" in e]
        assert [w for w in report.warnings if "G14" in w]

    def test_a_research_only_episode_STILL_FREEZES(self):
        """The whole point. This used to raise FreezeAssertionError and destroy
        a finished render to prevent a copy nobody had made yet."""
        led = _ledger(_RESEARCH, _NAMED)
        LF.phase_10_gap_audit_post_and_freeze(led)
        assert led["meta"]["freeze_verdict"] == "frozen_with_warns"
        assert led["meta"]["cleanup_locked"] is True

    def test_phase_10_stamps_the_receipt_and_it_says_blocked(self):
        led = _ledger(_RESEARCH, _NAMED)
        LF.phase_10_gap_audit_post_and_freeze(led)
        receipt = led["meta"][PE.PUBLICATION_ELIGIBILITY_META_KEY]
        assert receipt["eligible"] is False
        assert PE.REASON_RIGHTS_RESEARCH_ONLY in receipt["blocking_reasons"]
        assert receipt["episode_id"] == "the_test_episode"

    def test_phase_10_stamps_a_cleared_episode_eligible(self):
        led = _ledger(_CLEARED, _NAMED)
        LF.phase_10_gap_audit_post_and_freeze(led)
        assert led["meta"][PE.PUBLICATION_ELIGIBILITY_META_KEY]["eligible"] is True

    def test_a_structural_error_still_rejects_the_freeze(self):
        """Softening G14 must not soften the gate. A real structural fault
        (schema_version) still raises."""
        led = _ledger(_CLEARED, _NAMED)
        led["schema_version"] = "l0-ancient"
        with pytest.raises(LF.FreezeAssertionError):
            LF.phase_10_gap_audit_post_and_freeze(led)

    def test_a_rejected_freeze_stamps_NO_receipt_so_the_mux_blocks(self):
        led = _ledger(_CLEARED, _NAMED)
        led["schema_version"] = "l0-ancient"
        with pytest.raises(LF.FreezeAssertionError):
            LF.phase_10_gap_audit_post_and_freeze(led)
        assert PE.PUBLICATION_ELIGIBILITY_META_KEY not in led["meta"]
        assert PE.decide_from_meta(led["meta"]).publishable is False

    def test_the_read_only_audit_stamps_nothing(self):
        """`run_gap_audit` is read-only by contract; a stamping audit is how a
        read-only guarantee quietly stops being true."""
        led = _ledger(_RESEARCH, _NAMED)
        before = json.dumps(led, sort_keys=True)
        LF.run_gap_audit(led, label="pre")
        assert json.dumps(led, sort_keys=True) == before


# --------------------------------------------------------------------------- #
# the ledger save: a blocked episode carries no OBS pointer
# --------------------------------------------------------------------------- #
class TestObsAliasesOnSave:
    @pytest.fixture
    def workspace(self, tmp_path):
        ep_id = "the_test_episode"
        otr_root = tmp_path / "output" / "otr"
        audio_dir = otr_root / "episodes" / ep_id / "audio"
        audio_dir.mkdir(parents=True)
        (otr_root / "obs").mkdir(parents=True)
        return audio_dir / f"{ep_id}_ledger.json"

    def _saved(self, ledger_path, provenance):
        led = _ledger(provenance, _NAMED)
        PE.stamp_publication_eligibility(led)
        assert OTRL.save_ledger_safe(ledger_path, led) is True
        return json.loads(Path(ledger_path).read_text(encoding="utf-8"))

    def test_an_eligible_episode_keeps_its_planned_obs_pointer(self, workspace):
        saved = self._saved(workspace, _CLEARED)
        assert saved["meta"]["paths"]["obs_final"]
        assert saved["meta"]["paths"]["obs_dir"]

    def test_a_blocked_episode_carries_NO_obs_path_at_either_alias(self, workspace):
        """A path key reads as "the deliverable is there" to every downstream
        reader. On a blocked episode nothing ever arrives at it."""
        saved = self._saved(workspace, _RESEARCH)
        assert "obs_final" not in saved["meta"]["paths"]
        assert "obs_dir" not in saved["meta"]["paths"]
        assert "obs_final_path" not in saved["meta"]

    def test_a_stale_published_claim_is_cleared_when_the_block_lands(self, workspace):
        led = _ledger(_RESEARCH, _NAMED)
        led["meta"]["obs_final_path"] = str(
            workspace.parent.parent.parent.parent / "obs" / "the_test_episode.mp4")
        PE.stamp_publication_eligibility(led)
        assert OTRL.save_ledger_safe(workspace, led) is True
        saved = json.loads(workspace.read_text(encoding="utf-8"))
        assert "obs_final_path" not in saved["meta"]

    def test_a_ledger_with_no_receipt_is_left_alone(self, workspace):
        """"We cannot tell" is not "we know it is blocked" -- the producer side
        stays permissive and the mux is the one that fails closed."""
        led = _ledger(_CLEARED, _NAMED)
        assert OTRL.save_ledger_safe(workspace, led) is True
        saved = json.loads(workspace.read_text(encoding="utf-8"))
        assert saved["meta"]["paths"]["obs_final"]


# --------------------------------------------------------------------------- #
# the mux: the consumer, the blocked path and the cache key
# --------------------------------------------------------------------------- #
class TestMuxConsumesTheReceipt:
    @pytest.fixture
    def episode(self, tmp_path, monkeypatch):
        ep_id = "the_test_episode"
        episodes_root = tmp_path / "output" / "otr" / "episodes"
        audio_dir = episodes_root / ep_id / "audio"
        audio_dir.mkdir(parents=True)
        (tmp_path / "output" / "otr" / "obs").mkdir(parents=True)
        ledger_path = audio_dir / f"{ep_id}_ledger.json"
        monkeypatch.setattr(MUX, "_episodes_root", lambda: episodes_root)
        monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: ledger_path)
        return {
            "ep_id": ep_id,
            "ledger_path": ledger_path,
            "video": str(episodes_root / ep_id / f"{ep_id}_silent.mp4"),
        }

    def _write(self, episode, provenance):
        led = _ledger(provenance, _NAMED, episode_id=episode["ep_id"])
        PE.stamp_publication_eligibility(led)
        assert OTRL.save_ledger_safe(episode["ledger_path"], led) is True
        return led

    def test_a_cleared_episode_is_publishable(self, episode):
        self._write(episode, _CLEARED)
        assert MUX._publication_decision(episode["video"]).publishable is True

    def test_a_research_only_episode_is_withheld(self, episode):
        self._write(episode, _RESEARCH)
        d = MUX._publication_decision(episode["video"])
        assert d.publishable is False
        assert PE.REASON_RIGHTS_RESEARCH_ONLY in d.detail
        # The node owns the "obs_publish BLOCKED" marker; the summary must not
        # carry a second verdict word of its own.
        assert "BLOCKED" not in d.summary()
        assert PE.REASON_RIGHTS_RESEARCH_ONLY in d.summary()

    def test_no_ledger_at_all_fails_closed(self, tmp_path, monkeypatch):
        monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: None)
        d = MUX._publication_decision(str(tmp_path / "whatever_silent.mp4"))
        assert d.publishable is False
        assert d.reason == PE.DECISION_NO_RECEIPT

    def test_a_STALE_singleton_from_another_episode_fails_closed(self, episode):
        """The in-flight ledger belongs to `the_test_episode`; this video does
        not. Gating tonight's episode on last night's verdict is the exact
        failure the stem-prefix check exists to prevent."""
        self._write(episode, _CLEARED)
        d = MUX._publication_decision("a_completely_different_episode_silent.mp4")
        assert d.publishable is False
        assert d.reason == PE.DECISION_NO_RECEIPT


class TestMuxStampAndCacheKey:
    @pytest.fixture
    def episode(self, tmp_path, monkeypatch):
        ep_id = "the_test_episode"
        episodes_root = tmp_path / "output" / "otr" / "episodes"
        audio_dir = episodes_root / ep_id / "audio"
        audio_dir.mkdir(parents=True)
        (tmp_path / "output" / "otr" / "obs").mkdir(parents=True)
        ledger_path = audio_dir / f"{ep_id}_ledger.json"
        monkeypatch.setattr(MUX, "_episodes_root", lambda: episodes_root)
        monkeypatch.setattr(OTRL, "in_flight_ledger_path", lambda: ledger_path)
        return {
            "ep_id": ep_id,
            "ledger_path": ledger_path,
            "video": str(episodes_root / ep_id / f"{ep_id}_silent.mp4"),
            "final": str(episodes_root / ep_id / f"{ep_id}_silent_final.mp4"),
            "audio": str(audio_dir / f"{ep_id}_master.wav"),
        }

    def _write(self, episode, provenance):
        led = _ledger(provenance, _NAMED, episode_id=episode["ep_id"])
        PE.stamp_publication_eligibility(led)
        assert OTRL.save_ledger_safe(episode["ledger_path"], led) is True

    def test_a_withheld_publication_still_stamps_the_archival_final(self, episode):
        self._write(episode, _RESEARCH)
        line = MUX.OTRMasterAudioMux()._stamp_terminal_paths(
            episode["final"], None, episode["audio"])
        saved = json.loads(episode["ledger_path"].read_text(encoding="utf-8"))
        assert "publication withheld" in line
        assert saved["final_video_path"] == episode["final"]
        assert saved["final_audio_path"] == episode["audio"]
        assert "obs_final_path" not in saved["meta"]
        assert "obs_final" not in saved["meta"]["paths"]

    def test_a_published_episode_stamps_the_obs_pointer(self, episode):
        self._write(episode, _CLEARED)
        obs = str(episode["ledger_path"].parents[3] / "obs" / "the_test_episode.mp4")
        Path(obs).write_bytes(b"published")
        MUX.OTRMasterAudioMux()._stamp_terminal_paths(
            episode["final"], obs, episode["audio"])
        saved = json.loads(episode["ledger_path"].read_text(encoding="utf-8"))
        assert saved["meta"]["obs_final_path"]

    def test_IDENTICAL_INPUTS_WITH_CHANGED_ELIGIBILITY_BUST_THE_CACHE(self, episode):
        """The D5a acceptance assertion, stated as code.

        A cached node does not execute, and a mux that does not execute cannot
        withhold anything -- the receipt would say blocked while the previous
        run's published output stood in for it. The old key hashed only
        `clip_manifest_json`, a RETIRED connector that feeds nothing, so these
        two runs were indistinguishable."""
        kwargs = {"silent_video_path": episode["video"], "clip_manifest_json": "{}"}
        self._write(episode, _CLEARED)
        cleared_key = MUX.OTRMasterAudioMux.IS_CHANGED(**kwargs)
        self._write(episode, _RESEARCH)
        blocked_key = MUX.OTRMasterAudioMux.IS_CHANGED(**kwargs)
        assert cleared_key != blocked_key

    def test_the_cache_key_is_stable_when_nothing_changed(self, episode):
        kwargs = {"silent_video_path": episode["video"], "clip_manifest_json": "{}"}
        self._write(episode, _CLEARED)
        assert (MUX.OTRMasterAudioMux.IS_CHANGED(**kwargs)
                == MUX.OTRMasterAudioMux.IS_CHANGED(**kwargs))

    def test_two_different_episodes_never_share_a_cache_entry(self, episode):
        """Episode identity was absent from the old key entirely."""
        self._write(episode, _CLEARED)
        mine = MUX.OTRMasterAudioMux.IS_CHANGED(
            silent_video_path=episode["video"], clip_manifest_json="{}")
        theirs = MUX.OTRMasterAudioMux.IS_CHANGED(
            silent_video_path="somebody_elses_episode_silent.mp4",
            clip_manifest_json="{}")
        assert mine != theirs

    def test_the_cache_key_is_a_reproducible_hex_digest(self, episode):
        self._write(episode, _CLEARED)
        key = MUX.OTRMasterAudioMux.IS_CHANGED(
            silent_video_path=episode["video"], clip_manifest_json="{}")
        assert isinstance(key, str) and len(key) == 64
        int(key, 16)

    def _run_mux(self, episode, monkeypatch):
        """Drive the real ``mux()`` with the ffmpeg work stubbed out.

        Only ``mux_master_audio`` is replaced -- the branch under test, the
        stamping and the report are all the shipped code. ``_publish_to_obs``
        is instrumented rather than stubbed to nothing, so "was the copy made"
        is answered by observation instead of by inference from a log string.
        """
        published = []
        monkeypatch.setattr(
            MUX, "mux_master_audio",
            lambda *a, **k: (episode["final"], ["mux gate OK"]),
        )
        monkeypatch.setattr(
            MUX.OTRMasterAudioMux, "_publish_to_obs",
            lambda self, final: published.append(final) or "obs/copy.mp4",
        )
        path, report = MUX.OTRMasterAudioMux().mux(
            episode["video"], episode["audio"])
        return path, report, published

    def test_an_INELIGIBLE_episode_still_SUCCEEDS_and_keeps_its_final(
            self, episode, monkeypatch):
        """The acceptance assertion for D5a, and the whole reason this chunk
        exists: withholding a copy is not failing a render."""
        self._write(episode, _RESEARCH)
        path, report, published = self._run_mux(episode, monkeypatch)
        assert path == episode["final"]
        assert published == []
        assert "obs_publish BLOCKED" in report
        assert "obs_publish OK" not in report
        saved = json.loads(episode["ledger_path"].read_text(encoding="utf-8"))
        assert saved["final_video_path"] == episode["final"]
        assert "obs_final_path" not in saved["meta"]
        assert "obs_final" not in saved["meta"]["paths"]

    def test_an_ELIGIBLE_episode_publishes(self, episode, monkeypatch):
        """The other half. Without this, "nothing published" would pass for
        both answers and the test above would prove nothing."""
        self._write(episode, _CLEARED)
        path, report, published = self._run_mux(episode, monkeypatch)
        assert path == episode["final"]
        assert published == [episode["final"]]
        assert "obs_publish OK" in report
