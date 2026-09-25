"""Asset cleanup after publish (GO_FORWARD_PLAN row 0b, 2026-09-25).

The writer's `asset_cleanup` choice, the ledger stamp that carries it, and the
terminal mux that carries it out. Everything runs on a `tmp_path` tree: no
ComfyUI, no GPU, no saved fixtures.

What is pinned, and why each one is here:

* the PLANNER (`plan_asset_cleanup`) -- what each mode would delete and keep,
  and one refusal per guard. The previous space-saver wiped the wrong episode
  on its first day (BUG-LOCAL-014); every refusal below is a way that happens.
* the EXECUTOR (`execute_asset_cleanup`) -- the planner tests do not prove it,
  so two runs hold a real Windows lock on one file and assert it is skipped,
  named, and the rest still goes.
* the MUX WIRING -- `_asset_cleanup` reads the RE-READ ledger for the mode,
  the token and the published path, and the call sits after the preview and
  after the delivery gate.
* the WRITER -- the widget, the stamp, and the replay that must not inherit
  its source's choice.
"""
from __future__ import annotations

import inspect
import json
import os
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes import _otr_asset_cleanup as AC  # noqa: E402

TOKEN = "a" * 32
EP_ID = "signal_lost_20260925_120000"

MEDIA = [
    "audio/%s_master.wav" % EP_ID,
    "audio/stems/line_001.flac",
    "stills/scene_b001.png",
    "portraits/c02.jpg",
    "clips/shot_001.mp4",
    "clips/latents.npy",
    "%s_silent.mp4" % EP_ID,
    "%s_final.mp4" % EP_ID,
]
TEXT = [
    "audio/%s_ledger.json" % EP_ID,
    "episode_canon.json",
    "treatment.md",
    "captions/%s.srt" % EP_ID,
    "qa/report.txt",
    "notes.unknownext",          # an extension nobody listed is KEPT
]


def _tree(tmp_path):
    root = tmp_path / "episodes"
    ep = root / EP_ID
    for rel in MEDIA + TEXT:
        path = ep / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x" * 1024)
    obs_dir = tmp_path / "obs"
    obs_dir.mkdir()
    obs = obs_dir / (EP_ID + ".mp4")
    obs.write_bytes(b"published" * 100)
    return root, ep, obs_dir, obs


def _facts(root, ep, obs_dir, obs, **over):
    facts = dict(
        episodes_root=root,
        obs_copy=str(obs),
        video_paths=(str(ep / ("%s_silent.mp4" % EP_ID)),
                     str(ep / ("%s_final.mp4" % EP_ID))),
        obs_dir=obs_dir,
        wire_token=TOKEN,
        ledger_token=TOKEN,
        ledger_obs_path=str(obs),
    )
    facts.update(over)
    return facts


def _rel(ep, paths):
    return sorted(Path(p).relative_to(ep).as_posix() for p in paths)


# --------------------------------------------------------------------------- #
# the labels and the stamp
# --------------------------------------------------------------------------- #


def test_every_label_says_what_it_keeps_and_its_first_word_is_the_slug():
    assert [AC.slug_from_label(label) for label in AC.LABELS] == list(AC.MODES)
    assert AC.LABELS == (
        "off (keep everything)",
        "partial (keep only the text files)",
        "full (keep only the published video)",
    )
    assert AC.DEFAULT_LABEL == "off (keep everything)"
    assert AC.slug_from_label("") == "off"          # a graph saved before the widget


def test_an_unknown_label_fails_at_the_writer():
    with pytest.raises(ValueError):
        AC.slug_from_label("everything (burn it down)")


def test_an_unknown_ledger_value_reads_as_off_with_one_note():
    assert AC.mode_from_meta({}) == ("off", None)
    assert AC.mode_from_meta({"asset_cleanup": "partial"}) == ("partial", None)
    mode, note = AC.mode_from_meta({"asset_cleanup": "scorched"})
    assert mode == "off" and "scorched" in note


# --------------------------------------------------------------------------- #
# the planner
# --------------------------------------------------------------------------- #


def test_off_plans_nothing(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    assert AC.plan_asset_cleanup(ep, "off", **_facts(root, ep, obs_dir, obs)) == ([], [], None)


def test_partial_deletes_the_sound_and_pictures_and_keeps_every_text_file(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    delete, keep, refusal = AC.plan_asset_cleanup(
        ep, "partial", **_facts(root, ep, obs_dir, obs))
    assert refusal is None
    assert _rel(ep, delete) == sorted(MEDIA)
    assert _rel(ep, keep) == sorted(TEXT)


def test_full_lists_the_folder_itself(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    delete, keep, refusal = AC.plan_asset_cleanup(
        ep, "full", **_facts(root, ep, obs_dir, obs))
    assert refusal is None and delete == [ep] and keep == []


def _refusal(tmp_path, mode="partial", episode_dir=None, **over):
    root, ep, obs_dir, obs = _tree(tmp_path)
    facts = _facts(root, ep, obs_dir, obs)
    for key, value in over.items():
        facts[key] = value(root, ep, obs_dir, obs) if callable(value) else value
    target = ep if episode_dir is None else episode_dir(root, ep, obs_dir, obs)
    delete, keep, refusal = AC.plan_asset_cleanup(target, mode, **facts)
    assert delete == [] and keep == []
    return refusal


REFUSALS = {
    "unknown mode": dict(mode="scorched"),
    "no in-flight episode": dict(episode_dir=lambda r, e, d, o: None),
    "not a direct child of the episodes root": dict(
        episodes_root=lambda r, e, d, o: r.parent),
    "a reserved underscore folder": dict(
        episode_dir=lambda r, e, d, o: (r / "_shared").mkdir() or (r / "_shared")),
    "the obs copy is inside the folder": dict(
        obs_copy=lambda r, e, d, o: str(e / ("%s_final.mp4" % EP_ID)),
        ledger_obs_path=lambda r, e, d, o: str(e / ("%s_final.mp4" % EP_ID))),
    "the obs folder is inside the folder": dict(
        obs_dir=lambda r, e, d, o: e / "audio"),
    "a video outside the folder": dict(
        video_paths=lambda r, e, d, o: (str(d / "elsewhere.mp4"),)),
    "no video to bind to": dict(video_paths=()),
    "nothing was published": dict(obs_copy=None),
    "the obs copy is missing": dict(
        obs_copy=lambda r, e, d, o: str(d / "gone.mp4"),
        ledger_obs_path=lambda r, e, d, o: str(d / "gone.mp4")),
    "the obs copy is empty": dict(
        obs_copy=lambda r, e, d, o: (o.write_bytes(b"") or str(o))),
    "the ledger does not record the obs copy": dict(
        ledger_obs_path=lambda r, e, d, o: str(d / "another.mp4")),
    "the ledger records nothing": dict(ledger_obs_path=None),
    "no token on the wire": dict(wire_token=""),
    "no token on the ledger": dict(ledger_token=None),
    "another run's token": dict(ledger_token="b" * 32),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_every_guard_refuses_and_deletes_nothing(tmp_path, case):
    refusal = _refusal(tmp_path, **REFUSALS[case])
    assert refusal, case
    # and the tree is untouched: the planner never deletes
    assert (tmp_path / "episodes" / EP_ID / ("%s_final.mp4" % EP_ID)).exists()


def _make_dir_link(link: Path, target: Path) -> bool:
    """A directory symlink, else a Windows junction (no privilege needed)."""
    try:
        os.symlink(target, link, target_is_directory=True)
        return True
    except (OSError, NotImplementedError):
        pass
    if os.name == "nt":
        try:
            import _winapi
            _winapi.CreateJunction(str(target), str(link))
            return True
        except (OSError, AttributeError):
            pass
    return False


@pytest.mark.parametrize("mode", ["partial", "full"])
def test_a_linked_folder_inside_the_episode_is_a_refusal(tmp_path, mode):
    root, ep, obs_dir, obs = _tree(tmp_path)
    outside = tmp_path / "someone_elses_work"
    outside.mkdir()
    (outside / "precious.wav").write_bytes(b"do not touch")
    if not _make_dir_link(ep / "stills" / "linked", outside):
        pytest.skip("this box can create neither a symlink nor a junction")
    delete, keep, refusal = AC.plan_asset_cleanup(
        ep, mode, **_facts(root, ep, obs_dir, obs))
    assert refusal and "linked" in refusal
    assert delete == [] and keep == []
    assert (outside / "precious.wav").read_bytes() == b"do not touch"


# --------------------------------------------------------------------------- #
# the executor
# --------------------------------------------------------------------------- #


class _Receipts:
    def __init__(self, ok=True):
        self.calls = []
        self.ok = ok

    def __call__(self, receipt):
        self.calls.append(json.loads(json.dumps(receipt)))
        return self.ok


def test_partial_runs_two_phase_and_removes_emptied_folders(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    delete, keep, _ = AC.plan_asset_cleanup(ep, "partial", **_facts(root, ep, obs_dir, obs))
    receipts = _Receipts()
    line = AC.execute_asset_cleanup("partial", ep, delete, keep,
                                    write_receipt=receipts, obs_copy=str(obs))
    assert [c["state"] for c in receipts.calls] == ["started", "done"]
    done = receipts.calls[-1]
    assert done["mode"] == "partial" and done["replay_freeze_possible"] is False
    assert len(done["removed"]) == len(MEDIA) and done["skipped"] == []
    assert done["removed_bytes"] == 1024 * len(MEDIA)
    for rel in TEXT:
        assert (ep / rel).is_file(), rel
    for rel in MEDIA:
        assert not (ep / rel).exists(), rel
    # folders the media left empty are gone; folders holding text stay
    assert not (ep / "stills").exists() and not (ep / "clips").exists()
    assert not (ep / "audio" / "stems").exists() and (ep / "audio").is_dir()
    assert obs.is_file()
    assert line.startswith("asset_cleanup partial: removed %d files" % len(MEDIA))
    assert "kept %d, skipped 0" % len(TEXT) in line and str(obs) in line


def test_nothing_is_deleted_when_the_started_receipt_does_not_save(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    delete, keep, _ = AC.plan_asset_cleanup(ep, "partial", **_facts(root, ep, obs_dir, obs))
    line = AC.execute_asset_cleanup("partial", ep, delete, keep,
                                    write_receipt=_Receipts(ok=False))
    assert "skipped" in line and "nothing was deleted" in line
    for rel in MEDIA:
        assert (ep / rel).is_file(), rel


@pytest.mark.skipif(os.name != "nt", reason="an open handle blocks a delete only on Windows")
def test_partial_skips_a_locked_file_names_it_and_still_writes_done(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    locked = ep / "audio" / ("%s_master.wav" % EP_ID)
    delete, keep, _ = AC.plan_asset_cleanup(ep, "partial", **_facts(root, ep, obs_dir, obs))
    receipts = _Receipts()
    with open(locked, "rb"):
        line = AC.execute_asset_cleanup("partial", ep, delete, keep,
                                        write_receipt=receipts, obs_copy=str(obs))
    done = receipts.calls[-1]
    assert done["state"] == "done"
    assert done["skipped"] == [str(locked)]
    assert "skipped 1" in line and str(locked) in line
    assert locked.is_file()
    assert not (ep / "stills" / "scene_b001.png").exists()


@pytest.mark.skipif(os.name != "nt", reason="an open handle blocks a delete only on Windows")
def test_full_survives_a_failing_rmtree_entry_and_names_it(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    locked = ep / "clips" / "shot_001.mp4"
    delete, keep, _ = AC.plan_asset_cleanup(ep, "full", **_facts(root, ep, obs_dir, obs))
    receipts = _Receipts()
    with open(locked, "rb"):
        line = AC.execute_asset_cleanup("full", ep, delete, keep,
                                        write_receipt=receipts, obs_copy=str(obs))
    assert receipts.calls[0]["state"] == "started"
    assert receipts.calls[-1]["skipped"] == [str(locked)]
    assert locked.is_file()
    assert not (ep / "episode_canon.json").exists()
    assert not (ep / "stills").exists()
    assert "skipped 1" in line and str(locked) in line
    assert obs.is_file()


def test_full_removes_the_whole_folder_and_leaves_obs(tmp_path):
    root, ep, obs_dir, obs = _tree(tmp_path)
    delete, keep, _ = AC.plan_asset_cleanup(ep, "full", **_facts(root, ep, obs_dir, obs))
    line = AC.execute_asset_cleanup("full", ep, delete, keep,
                                    write_receipt=_Receipts(), obs_copy=str(obs))
    assert not ep.exists() and obs.is_file() and root.is_dir()
    assert "removed %d files" % (len(MEDIA) + len(TEXT)) in line


# --------------------------------------------------------------------------- #
# the mux: the facts it gathers, and where the call sits
# --------------------------------------------------------------------------- #


def _mux_rig(tmp_path, monkeypatch, *, mode="partial", ledger_token=TOKEN):
    from nodes import otr_master_audio_mux as M
    root, ep, obs_dir, obs = _tree(tmp_path)
    ledger_path = ep / "audio" / ("%s_ledger.json" % EP_ID)
    meta = {"source_bank": "media_archive",
            "delivery_intent": {"schema_version": M.DELIVERY_INTENT_VERSION,
                                "source_bank": "media_archive",
                                "publication_required": False,
                                "draft_digest": "", "delivery_token": ledger_token},
            "obs_final_path": str(obs)}
    if mode is not None:
        meta["asset_cleanup"] = mode
    ledger_path.write_text(json.dumps({"episode_id": EP_ID, "meta": meta}), encoding="utf-8")
    monkeypatch.setattr(M, "_inflight_episode_for_stem", lambda stem: (ledger_path, ep))
    monkeypatch.setattr(M, "_episodes_root", lambda: root)
    monkeypatch.setattr(M, "_obs_dir", lambda: obs_dir)
    intent = {"delivery_token": TOKEN, "source_bank": "media_archive"}
    silent = str(ep / ("%s_silent.mp4" % EP_ID))
    final = str(ep / ("%s_final.mp4" % EP_ID))
    return M, ep, obs, ledger_path, intent, silent, final


def test_the_mux_cleans_the_bound_folder_and_leaves_the_receipt_on_the_ledger(tmp_path, monkeypatch):
    M, ep, obs, ledger_path, intent, silent, final = _mux_rig(tmp_path, monkeypatch)
    line, final_removed = M.OTRMasterAudioMux()._asset_cleanup(silent, final, str(obs), intent)
    assert line.startswith("asset_cleanup partial: removed %d files" % len(MEDIA)), line
    assert final_removed is True and obs.is_file()
    receipt = json.loads(ledger_path.read_text(encoding="utf-8"))["meta"]["asset_cleanup_receipt"]
    assert receipt["state"] == "done" and receipt["mode"] == "partial"


def test_the_mux_says_nothing_and_touches_nothing_when_off(tmp_path, monkeypatch):
    M, ep, obs, ledger_path, intent, silent, final = _mux_rig(tmp_path, monkeypatch, mode=None)
    assert M.OTRMasterAudioMux()._asset_cleanup(silent, final, str(obs), intent) == (None, False)
    assert Path(final).is_file()


def test_the_mux_refuses_another_runs_ledger(tmp_path, monkeypatch):
    """The mtime fallback can hand the singleton a ledger from another run.
    That ledger cannot carry this run's token, so nothing goes."""
    M, ep, obs, ledger_path, intent, silent, final = _mux_rig(
        tmp_path, monkeypatch, ledger_token="c" * 32)
    line, final_removed = M.OTRMasterAudioMux()._asset_cleanup(silent, final, str(obs), intent)
    assert "refused" in line and "nothing was deleted" in line and not final_removed
    for rel in MEDIA:
        assert (ep / rel).is_file(), rel


def test_the_mux_refuses_a_wire_with_no_delivery_intent(tmp_path, monkeypatch):
    M, ep, obs, ledger_path, intent, silent, final = _mux_rig(tmp_path, monkeypatch)
    line, _ = M.OTRMasterAudioMux()._asset_cleanup(silent, final, str(obs), None)
    assert "refused" in line and Path(final).is_file()


def test_the_mux_never_cleans_a_withheld_episode(tmp_path, monkeypatch):
    M, ep, obs, ledger_path, intent, silent, final = _mux_rig(tmp_path, monkeypatch)
    line, _ = M.OTRMasterAudioMux()._asset_cleanup(silent, final, None, intent)
    assert "refused" in line and Path(final).is_file()


def test_the_mux_calls_the_cleanup_last_after_the_preview_and_the_delivery_gate():
    """Source inspection is the right tool for exactly this: WHERE the call
    sits. The preview extracts its frame from the archival final, which the
    cleanup deletes; the delivery gate must have passed before anything goes;
    and a cancel must be re-raised before the broad catch."""
    from nodes import otr_master_audio_mux as M
    src = inspect.getsource(M.OTRMasterAudioMux.mux)
    call = src.index("self._asset_cleanup(")
    assert src.index("ui = _canvas_preview(final, obs_copy)") < call
    assert src.index("delivery OK -- required publication verified") < call
    assert src.index("sweep_shared_tmp()") < call
    assert src.index("self._stamp_terminal_paths(") < call
    tail = src[call:]
    assert tail.index("except _Interrupted:") < tail.index("except Exception")
    assert "raise" in tail[tail.index("except _Interrupted:"):tail.index("except Exception")]
    helper = inspect.getsource(M.OTRMasterAudioMux._asset_cleanup)
    assert "plan_asset_cleanup(" in helper and "execute_asset_cleanup(" in helper
    assert "_assert_delivery_binding(intent, stem)" in helper


# --------------------------------------------------------------------------- #
# the writer and the replay
# --------------------------------------------------------------------------- #


def test_the_writer_widget_ships_off_and_offers_the_three_full_labels():
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W
    choices, meta = W.INPUT_TYPES()["optional"]["asset_cleanup"]
    assert list(choices) == list(AC.LABELS)
    assert meta["default"] == "off (keep everything)"
    for phrase in ("otr/obs", "replay bundle", "text file"):
        assert phrase in meta["tooltip"], phrase


def test_the_writer_stamps_beside_the_delivery_intent_on_the_fresh_path():
    from nodes import OTR_LedgerScriptWriter as W
    src = inspect.getsource(W.OTR_LedgerScriptWriter.run)
    stamp = src.index('meta["delivery_intent"] = {')
    after = src[stamp:]
    assert after.index('meta["asset_cleanup"] = _cleanup_slug') < 1200


def test_asset_cleanup_is_run_volatile_on_a_replay():
    from nodes import production_ledger as PL
    assert "asset_cleanup" in PL._REPLAY_RUN_VOLATILE_META


@pytest.fixture
def frozen_full(tmp_path, monkeypatch):
    """A frozen bundle whose SOURCE ledger chose `full`."""
    from nodes import production_ledger as PL
    from tests.test_canonical_replay import _freeze, _make_episode
    # The replay import REBINDS the process ledger singleton. Registering it
    # with monkeypatch restores it at teardown; left bound, it pointed the
    # next module's tests (test_audio_cache_wiring, alphabetically next) at
    # this temp episode and turned three of them red in the full suite.
    monkeypatch.setattr(PL, "_CURRENT", PL._CURRENT)
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    ep, ledger = _make_episode(tmp_path / "episodes")
    ledger["meta"]["asset_cleanup"] = "full"
    lp = ep / "audio" / (ep.name + "_ledger.json")
    lp.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return _freeze(ep, tmp_path / "bundles")


def test_a_replay_never_inherits_its_sources_cleanup(frozen_full, monkeypatch):
    """THE LOAD-BEARING ONE (row 0b): the source said `full`, this run's
    widget says off, and the replay must not delete anything."""
    from nodes import OTR_LedgerScriptWriter as W
    from nodes import production_ledger as PL
    monkeypatch.setattr(W, "_ROLLS", None)
    out = W.OTR_LedgerScriptWriter().run(
        replay_from=str(frozen_full), asset_cleanup="off (keep everything)")
    assert "asset_cleanup" not in json.loads(out[1])["meta"]
    on_disk = json.loads(Path(PL.peek_ledger().path).read_text(encoding="utf-8"))
    assert "asset_cleanup" not in on_disk["meta"]


def test_a_replay_stamps_this_runs_choice_on_the_wire_and_the_disk(frozen_full, monkeypatch):
    from nodes import OTR_LedgerScriptWriter as W
    from nodes import production_ledger as PL
    monkeypatch.setattr(W, "_ROLLS", None)
    out = W.OTR_LedgerScriptWriter().run(
        replay_from=str(frozen_full),
        asset_cleanup="partial (keep only the text files)")
    assert json.loads(out[1])["meta"]["asset_cleanup"] == "partial"
    on_disk = json.loads(Path(PL.peek_ledger().path).read_text(encoding="utf-8"))
    assert on_disk["meta"]["asset_cleanup"] == "partial"
