"""The replay freezer as a library (Scene Producer chunk 1, 2026-10-22).

The freezing moved from ``scripts/otr_freeze_replay_bundle.py`` into
``nodes/_otr_replay_freezer.py`` so a node can freeze a stage inside the ComfyUI
server. These tests pin what that move promised:

* refusals raise ``ReplayBundleError``, never ``SystemExit`` (a SystemExit raised
  in a node escapes ComfyUI's prompt worker and stops it for every later prompt);
* the helper never asks git; the source commit is an argument, and the script
  is the one caller that fills it from git;
* ``dest=`` names the bundle directory itself, and a refused freeze leaves
  nothing behind;
* clips and foley stems are frozen only when ``include_render_outputs=True``,
  so an ordinary bundle is exactly what it was.

Offline: the episode on disk is the replay suite's own fixture.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from nodes import _otr_replay_freezer as RF  # noqa: E402
from nodes import production_ledger as PL  # noqa: E402


def _replay_suite():
    """The canonical replay suite, loaded as a module for its episode builder."""
    spec = importlib.util.spec_from_file_location(
        "otr_test_canonical_replay_for_freezer",
        pathlib.Path(__file__).with_name("test_canonical_replay.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def episode(tmp_path, monkeypatch):
    monkeypatch.setattr(PL, "_default_out_dir",
                        lambda ep=None: str(tmp_path / "episodes" / (ep or "pending") / "audio"))
    ep, ledger = _replay_suite()._make_episode(tmp_path / "episodes")
    return ep


def _with_render_outputs(ep: pathlib.Path) -> None:
    (ep / "clips").mkdir()
    (ep / "clips" / "shot_b001.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42 clip bytes")
    (ep / "audio" / "foley").mkdir()
    (ep / "audio" / "foley" / "shot_b001_foley.wav").write_bytes(b"RIFF foley stem")


def _paths(bundle: pathlib.Path) -> set:
    return {row["path"] for row in PL.load_replay_manifest(str(bundle))["files"]}


def test_refusals_raise_the_bundle_error_never_system_exit(episode, tmp_path):
    first = RF.freeze(episode, tmp_path / "bundles")
    with pytest.raises(PL.ReplayBundleError, match="already exists"):
        RF.freeze(episode, tmp_path / "bundles")
    assert not issubclass(PL.ReplayBundleError, SystemExit)
    assert first.is_dir()


def test_dest_names_the_bundle_directory_and_is_immutable(episode, tmp_path):
    dest = tmp_path / "_replay" / "p" / "proj1" / "01-stills"
    got = RF.freeze(episode, None, dest=dest, source_commit="unknown")
    assert got == dest and (dest / PL.REPLAY_MANIFEST_NAME).is_file()
    assert PL.load_replay_manifest(str(dest))["source_commit"] == "unknown"
    with pytest.raises(PL.ReplayBundleError, match="already exists"):
        RF.freeze(episode, None, dest=dest)
    assert not (dest.parent / ".01-stills.building").exists()


def test_a_refused_freeze_leaves_no_building_directory(episode, tmp_path):
    led_path = next((episode / "audio").glob("*_ledger.json"))
    led = json.loads(led_path.read_text(encoding="utf-8"))
    led["images"]["images"].append({"image_id": "gone", "path": str(episode / "stills" / "nope.png")})
    led_path.write_text(json.dumps(led), encoding="utf-8")
    dest = tmp_path / "bundles" / "01-stills"
    with pytest.raises(PL.ReplayBundleError, match="image rows not found"):
        RF.freeze(episode, None, dest=dest)
    assert not dest.exists()
    assert not (dest.parent / ".01-stills.building").exists()


def test_render_outputs_are_frozen_only_when_asked(episode, tmp_path):
    _with_render_outputs(episode)
    plain = RF.freeze(episode, None, dest=tmp_path / "plain")
    assert not any(p.startswith(("clips/", "audio/foley/")) for p in _paths(plain))
    full = RF.freeze(episode, None, dest=tmp_path / "full", include_render_outputs=True)
    assert {"clips/shot_b001.mp4", "audio/foley/shot_b001_foley.wav"} <= _paths(full)
    assert _paths(full) - _paths(plain) == {"clips/shot_b001.mp4",
                                            "audio/foley/shot_b001_foley.wav"}


def test_the_helper_never_spawns_a_process_or_exits():
    source = pathlib.Path(RF.__file__).read_text(encoding="utf-8")
    assert "import subprocess" not in source
    assert "raise SystemExit" not in source
    assert RF.freeze.__kwdefaults__["source_commit"] == ""
    assert RF.freeze.__kwdefaults__["include_render_outputs"] is False


def test_the_script_is_a_thin_front_end_that_supplies_the_commit(episode, tmp_path, monkeypatch):
    import scripts.otr_freeze_replay_bundle as fz
    assert fz.find_master is RF.find_master
    assert fz.derive_engine_bundle is RF.derive_engine_bundle
    assert fz.ReplayBundleError is PL.ReplayBundleError
    monkeypatch.setattr(fz, "_git_commit", lambda: "c0ffee1")
    bundle = fz.freeze(episode, tmp_path / "b")
    assert PL.load_replay_manifest(str(bundle))["source_commit"] == "c0ffee1"
    named = fz.freeze(episode, tmp_path / "b2", source_commit="build-7")
    assert PL.load_replay_manifest(str(named))["source_commit"] == "build-7"


def test_the_command_line_turns_a_refusal_into_a_clean_exit(episode, tmp_path):
    import scripts.otr_freeze_replay_bundle as fz
    out = tmp_path / "cli"
    assert fz.main([str(episode), "--out", str(out)]) == 0
    with pytest.raises(SystemExit, match="refused: bundle already exists"):
        fz.main([str(episode), "--out", str(out)])


def test_a_corrupted_episode_id_cannot_lead_the_freeze_outside_its_root(episode, tmp_path):
    # Codex QA, 2026-10-22: the ledger's episode_id becomes part of a path the
    # helper deletes and renames, so it must be one folder name.
    led_path = next((episode / "audio").glob("*_ledger.json"))
    led = json.loads(led_path.read_text(encoding="utf-8"))
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("must survive", encoding="utf-8")
    for bad in ("../victim", "..", "a/b", "C:x", " padded "):
        led["episode_id"] = bad
        led_path.write_text(json.dumps(led), encoding="utf-8")
        with pytest.raises(PL.ReplayBundleError, match="single safe folder name"):
            RF.freeze(episode, tmp_path / "bundles")
    assert (victim / "keep.txt").read_text(encoding="utf-8") == "must survive"
    assert not (tmp_path / ".victim.building").exists()


def test_a_dest_that_names_no_folder_is_refused(episode, tmp_path):
    with pytest.raises(PL.ReplayBundleError, match="not a usable bundle directory"):
        RF.freeze(episode, None, dest=pathlib.Path(tmp_path.anchor))
