"""
tests/test_meta_paths.py -- Phase E (BUG-LOCAL-018) regression suite

Covers _build_meta_paths and the meta.paths block stamping in both
write paths (Ledger.save and save_ledger_safe).

Acceptance gates from QA pass Phase E:
  * an old ledger from before the bump still loads cleanly
  * a new ledger has the new fields populated
  * no downstream node throws KeyError on either shape
  * paths block reflects ACTUAL on-disk location, not a reconstruction
    from episode_id
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_ledger as OTRL
from nodes.production_ledger import Ledger


# ---------------------------------------------------------------------------
# _build_meta_paths: per-episode workspace layout
# ---------------------------------------------------------------------------

class TestBuildMetaPathsPerEpisode:
    """Layout: <root>/output/otr/episodes/<ep>/audio/<ep>_ledger.json"""

    @pytest.fixture
    def workspace(self, tmp_path):
        ep_id = "the_test_episode"
        otr_root = tmp_path / "output" / "otr"
        ep_dir = otr_root / "episodes" / ep_id
        audio_dir = ep_dir / "audio"
        audio_dir.mkdir(parents=True)
        # Pre-create obs dir to exercise the obs_final stamping
        (otr_root / "obs").mkdir(parents=True)
        ledger_path = audio_dir / f"{ep_id}_ledger.json"
        ledger_path.write_text("{}", encoding="utf-8")
        return {
            "ep_id": ep_id,
            "otr_root": otr_root,
            "ep_dir": ep_dir,
            "audio_dir": audio_dir,
            "ledger_path": ledger_path,
        }

    def test_layout_detected_as_per_episode_workspace(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        assert paths["layout"] == "per-episode-workspace"

    def test_all_six_per_episode_dirs_stamped(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        ep = workspace["ep_dir"].resolve()
        assert Path(paths["episode_root"]) == ep
        assert Path(paths["audio_dir"]) == ep / "audio"
        assert Path(paths["stills_dir"]) == ep / "stills"
        assert Path(paths["portraits_dir"]) == ep / "portraits"
        assert Path(paths["videos_dir"]) == ep / "videos"
        assert Path(paths["composited_dir"]) == ep / "composited"

    def test_obs_final_stamped_when_obs_dir_exists(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        otr = workspace["otr_root"].resolve()
        assert Path(paths["obs_final"]) == otr / "obs" / f"{workspace['ep_id']}.mp4"
        assert Path(paths["obs_dir"]) == otr / "obs"

    def test_published_obs_filename_replaces_planned_alias(self, workspace):
        published = (
            workspace["otr_root"] / "obs" /
            f"{workspace['ep_id']}_captioned_with_credits_final.mp4"
        )
        published.write_bytes(b"published")
        paths = OTRL._build_meta_paths(
            workspace["ledger_path"],
            workspace["ep_id"],
            published_obs_path=published,
        )
        assert Path(paths["obs_final"]) == published.resolve()
        assert Path(paths["obs_dir"]) == published.parent.resolve()

    @pytest.mark.parametrize("candidate_name", [
        "another_episode_final.mp4",
        "the_test_episode_final.mov",
        "the_test_episode_missing.mp4",
    ])
    def test_invalid_published_obs_path_cannot_redirect_owner(
            self, workspace, candidate_name):
        candidate = workspace["otr_root"] / "obs" / candidate_name
        if "missing" not in candidate_name:
            candidate.write_bytes(b"not accepted")
        paths = OTRL._build_meta_paths(
            workspace["ledger_path"],
            workspace["ep_id"],
            published_obs_path=candidate,
        )
        expected = (
            workspace["otr_root"] / "obs" /
            f"{workspace['ep_id']}.mp4"
        )
        assert Path(paths["obs_final"]) == expected.resolve()

    def test_ledger_path_stamped_as_absolute(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        assert Path(paths["ledger_path"]) == workspace["ledger_path"].resolve()
        assert Path(paths["ledger_path"]).is_absolute()


# ---------------------------------------------------------------------------
# _build_meta_paths: legacy flat layout
# ---------------------------------------------------------------------------

class TestBuildMetaPathsLegacyFlat:
    """Layout: <root>/output/audio/<ep>_ledger.json (no per-episode dir)"""

    @pytest.fixture
    def workspace(self, tmp_path):
        ep_id = "legacy_episode"
        audio_dir = tmp_path / "output" / "audio"
        audio_dir.mkdir(parents=True)
        ledger_path = audio_dir / f"{ep_id}_ledger.json"
        ledger_path.write_text("{}", encoding="utf-8")
        return {
            "ep_id": ep_id,
            "audio_dir": audio_dir,
            "ledger_path": ledger_path,
        }

    def test_layout_detected_as_legacy_flat(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        assert paths["layout"] == "legacy-flat"

    def test_no_fabricated_subdirs_in_legacy(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        for k in ("stills_dir", "portraits_dir", "videos_dir", "composited_dir"):
            assert k not in paths, (
                f"legacy layout should NOT fabricate {k}"
            )

    def test_no_obs_final_in_legacy(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        assert "obs_final" not in paths
        assert "obs_dir" not in paths

    def test_minimal_paths_present(self, workspace):
        paths = OTRL._build_meta_paths(workspace["ledger_path"], workspace["ep_id"])
        assert paths["ledger_path"]
        assert paths["audio_dir"]
        assert paths["episode_root"]


# ---------------------------------------------------------------------------
# save_ledger_safe stamps meta.paths and bumps schema_version
# ---------------------------------------------------------------------------

def test_save_ledger_safe_stamps_meta_paths(tmp_path):
    ep_id = "stamp_test"
    audio_dir = tmp_path / "output" / "otr" / "episodes" / ep_id / "audio"
    audio_dir.mkdir(parents=True)
    (tmp_path / "output" / "otr" / "obs").mkdir(parents=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"

    led = {"episode_id": ep_id, "lines": []}
    ok = OTRL.save_ledger_safe(ledger_path, led)
    assert ok

    # Read back from disk
    saved = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert saved["schema_version"] == OTRL.CURRENT_SCHEMA_VERSION
    assert saved["meta"]["schema_version"] == OTRL.CURRENT_SCHEMA_VERSION
    assert "paths" in saved["meta"]
    assert saved["meta"]["paths"]["layout"] == "per-episode-workspace"
    assert ep_id in saved["meta"]["paths"]["obs_final"]


def test_save_ledger_safe_does_not_corrupt_existing_meta(tmp_path):
    ep_id = "preserve_test"
    audio_dir = tmp_path / "output" / "otr" / "episodes" / ep_id / "audio"
    audio_dir.mkdir(parents=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"

    led = {
        "episode_id": ep_id,
        "lines": [],
        "meta": {
            "perfect_run_spacesaver": True,
            "news_seed": {"hash": "abcd"},
        },
    }
    OTRL.save_ledger_safe(ledger_path, led)
    saved = json.loads(ledger_path.read_text(encoding="utf-8"))
    # Pre-existing meta keys must survive
    assert saved["meta"]["perfect_run_spacesaver"] is True
    assert saved["meta"]["news_seed"]["hash"] == "abcd"
    # New stamping happened too
    assert "paths" in saved["meta"]
    assert saved["meta"]["schema_version"] == OTRL.CURRENT_SCHEMA_VERSION


def test_save_ledger_safe_synchronizes_terminal_obs_surfaces(tmp_path):
    ep_id = "terminal_truth"
    otr_root = tmp_path / "output" / "otr"
    audio_dir = otr_root / "episodes" / ep_id / "audio"
    audio_dir.mkdir(parents=True)
    obs_path = otr_root / "obs" / f"{ep_id}_with_credits_final.mp4"
    obs_path.parent.mkdir(parents=True)
    obs_path.write_bytes(b"published")
    ledger_path = audio_dir / f"{ep_id}_ledger.json"
    led = {
        "episode_id": ep_id,
        "meta": {"obs_final_path": str(obs_path)},
    }

    assert OTRL.save_ledger_safe(ledger_path, led)
    saved = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert saved["meta"]["obs_final_path"] == str(obs_path.resolve())
    assert saved["meta"]["paths"]["obs_final"] == str(obs_path.resolve())


# ---------------------------------------------------------------------------
# Backward-compat: old ledgers without meta.paths still load cleanly
# ---------------------------------------------------------------------------

def test_old_ledger_without_meta_paths_loads_clean(tmp_path):
    ledger_path = tmp_path / "old_ledger.json"
    # Synthesize an l3-2026-04-28 shape (no meta.paths)
    old = {
        "schema_version": "l3-2026-04-28",
        "episode_id": "old_ep",
        "lines": [],
        "meta": {
            "schema_version": "l3-2026-04-28",
            "perfect_run_spacesaver": False,
        },
    }
    ledger_path.write_text(json.dumps(old), encoding="utf-8")
    loaded = OTRL.load_ledger_safe(ledger_path)
    assert loaded is not None
    # Reader contract: dict.get returns None for missing meta.paths.
    assert loaded.get("meta", {}).get("paths") is None
    # Specifically, no KeyError on the safe-read pattern.
    audio_dir = loaded.get("meta", {}).get("paths", {}).get("audio_dir")
    assert audio_dir is None


# ---------------------------------------------------------------------------
# Ledger.save also stamps meta.paths and self-corrects after rename
# ---------------------------------------------------------------------------

def test_ledger_class_save_stamps_meta_paths(tmp_path):
    ep_id = "pending_20260502_120000"
    new_id = "the_finalized_episode"
    audio_dir = tmp_path / "episodes" / ep_id / "audio"
    audio_dir.mkdir(parents=True)
    # Create the destination obs sibling for the per-episode-workspace
    # layout detection in _build_meta_paths -- requires "otr" parent
    # name. Build the realistic structure.
    workspace_audio = tmp_path / "output" / "otr" / "episodes" / ep_id / "audio"
    workspace_audio.mkdir(parents=True)
    (tmp_path / "output" / "otr" / "obs").mkdir(parents=True)

    led = Ledger(ep_id, str(workspace_audio))
    led.save()

    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
    assert saved["schema_version"] == OTRL.CURRENT_SCHEMA_VERSION
    assert saved["meta"]["paths"]["layout"] == "per-episode-workspace"
    assert ep_id in saved["meta"]["paths"]["audio_dir"]


def test_ledger_meta_paths_self_corrects_after_rename(tmp_path):
    """After Ledger.rename_episode moves the per-episode dir, the next
    save's meta.paths must point at the NEW location, not the pending."""
    ep_id = "pending_20260502_130000"
    new_id = "the_renamed_episode"
    workspace_audio = tmp_path / "output" / "otr" / "episodes" / ep_id / "audio"
    workspace_audio.mkdir(parents=True)
    (tmp_path / "output" / "otr" / "obs").mkdir(parents=True)

    led = Ledger(ep_id, str(workspace_audio))
    led.save()
    led.rename_episode(new_id)
    led.save()

    new_audio_dir = tmp_path / "output" / "otr" / "episodes" / new_id / "audio"
    saved = json.loads(Path(led.path).read_text(encoding="utf-8"))
    paths = saved["meta"]["paths"]
    assert new_id in paths["audio_dir"]
    assert ep_id not in paths["audio_dir"], (
        "meta.paths.audio_dir still points at the pending dir after rename"
    )
    assert new_id in paths["obs_final"]
