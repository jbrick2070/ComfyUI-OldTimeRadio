"""Regression tests for scripts/render_episode_concat.py:find_humo_clip.

BUG-LOCAL-075: render_humo_batch.py's move-block renames each SaveVideo
output from ``humo_<line_id>_NNNNN_.mp4`` to ``<line_id>.mp4`` (line 1053
of render_humo_batch.py). The concat-side discovery only globbed for the
SaveVideo raw form, so it never matched the post-move canonical form.
Net effect: an episode with N rendered clips concat'd from at most 1
clip (the orphan SaveVideo file from a partial-run history). The fix
adds the canonical ``<line_id>.mp4`` form to the search and keeps the
raw form as a partial-run-recovery fallback.

These tests pin both forms so the discovery contract cannot drift.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import render_episode_concat as rec  # noqa: E402


def _make_clip(p: Path, mtime: float | None = None) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"\x00\x00\x00\x18ftypmp42")  # minimal MP4 header
    if mtime is not None:
        import os
        os.utime(p, (mtime, mtime))
    return p


def test_find_humo_clip_resolves_canonical_post_move_form(tmp_path: Path) -> None:
    """The canonical post-move filename ``<line_id>.mp4`` must resolve.
    Pre-fix, the discovery only matched ``humo_<line_id>_*.mp4`` and
    missed every successful run."""
    comfy_out = tmp_path / "ComfyUI" / "output"
    ep_id = "test_episode_001"
    clips_dir = comfy_out / "otr" / "videos" / ep_id

    _make_clip(clips_dir / "l001.mp4")
    _make_clip(clips_dir / "l002.mp4")
    _make_clip(clips_dir / "l003.mp4")

    assert rec.find_humo_clip(comfy_out, ep_id, "l001") == clips_dir / "l001.mp4"
    assert rec.find_humo_clip(comfy_out, ep_id, "l002") == clips_dir / "l002.mp4"
    assert rec.find_humo_clip(comfy_out, ep_id, "l003") == clips_dir / "l003.mp4"


def test_find_humo_clip_still_resolves_raw_savevideo_form(tmp_path: Path) -> None:
    """Partial-run recovery: if render_humo_batch crashed before the
    move-rename happened, the raw ``humo_<line_id>_NNNNN_.mp4`` form is
    still on disk. Discovery must fall through to it."""
    comfy_out = tmp_path / "ComfyUI" / "output"
    ep_id = "partial_run_episode"
    clips_dir = comfy_out / "otr" / "videos" / ep_id

    _make_clip(clips_dir / "humo_l001_00001_.mp4")

    found = rec.find_humo_clip(comfy_out, ep_id, "l001")
    assert found == clips_dir / "humo_l001_00001_.mp4"


def test_find_humo_clip_prefers_newest_when_both_forms_present(tmp_path: Path) -> None:
    """When both forms exist (e.g., a re-run that produced a new
    canonical clip alongside the orphan raw clip from the failed prior
    run), the NEWER one wins. This is the actual Storms Sentience
    scenario that produced BUG-LOCAL-075."""
    comfy_out = tmp_path / "ComfyUI" / "output"
    ep_id = "rerun_after_partial"
    clips_dir = comfy_out / "otr" / "videos" / ep_id

    # Orphan from prior failed run (older mtime)
    older = _make_clip(clips_dir / "humo_l001_00001_.mp4", mtime=time.time() - 3600)
    # Canonical from successful re-run (newer mtime)
    newer = _make_clip(clips_dir / "l001.mp4", mtime=time.time())

    found = rec.find_humo_clip(comfy_out, ep_id, "l001")
    assert found == newer, f"expected newer clip, got {found}"


def test_find_humo_clip_legacy_otr_videos_root_still_works(tmp_path: Path) -> None:
    """Back-compat: episodes rendered before the path rename live under
    the legacy output/otr_videos/<id>/ root. Discovery must still find
    them so old episodes can be re-concatenated."""
    comfy_out = tmp_path / "ComfyUI" / "output"
    ep_id = "legacy_episode"
    legacy_dir = comfy_out / "otr_videos" / ep_id

    _make_clip(legacy_dir / "l001.mp4")
    _make_clip(legacy_dir / "humo_l002_00001_.mp4")

    assert rec.find_humo_clip(comfy_out, ep_id, "l001") == legacy_dir / "l001.mp4"
    assert rec.find_humo_clip(comfy_out, ep_id, "l002") == legacy_dir / "humo_l002_00001_.mp4"


def test_find_humo_clip_returns_none_when_no_match(tmp_path: Path) -> None:
    """Missing line_id with no clip on disk returns None gracefully."""
    comfy_out = tmp_path / "ComfyUI" / "output"
    ep_id = "empty_episode"
    (comfy_out / "otr" / "videos" / ep_id).mkdir(parents=True, exist_ok=True)

    assert rec.find_humo_clip(comfy_out, ep_id, "l999") is None


def test_find_humo_clip_no_episode_dir_returns_none(tmp_path: Path) -> None:
    """If the episode dir doesn't exist anywhere, return None without
    raising."""
    comfy_out = tmp_path / "ComfyUI" / "output"
    assert rec.find_humo_clip(comfy_out, "nonexistent_episode", "l001") is None


def test_default_out_dir_is_in_broadcast_tree(tmp_path: Path) -> None:
    """BUG-LOCAL-084: the concat default output dir must live in
    output/episodes_for_obs/<id>/ -- a sibling of output/otr/, NOT a
    child -- so OBS's directory_sorter can stream finished episodes
    without picking up the per-line clip pieces under output/otr/videos/.
    Pin by source-string check on the active default expression.
    Supersedes the old BUG-LOCAL-075 'everything under otr/' rule for
    final episode outputs only; intermediate work files (audio, stills,
    portraits, per-line video pieces) still live under output/otr/."""
    src = (SCRIPTS_DIR / "render_episode_concat.py").read_text(encoding="utf-8")
    assert '"episodes_for_obs" / episode_id' in src, \
        "default out_dir no longer composes output/episodes_for_obs/<id>"
