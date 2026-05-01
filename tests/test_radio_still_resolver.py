"""
test_radio_still_resolver.py
============================

Coverage for ``_resolve_radio_still_path`` in
``nodes/batch_humo_render.py`` (Step 5 of the ROADMAP P0 lock).

The helper picks up the radio still path from the ledger -- in
either schema location (top-level or under meta) -- and returns
``Path`` only when it points at an actually-existing file.  Any
ambiguity falls through to ``None`` so radio-role lines fall back
to the legacy portrait resolver gracefully.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
_NODES_DIR = os.path.join(_REPO_ROOT, "nodes")
for p in (_REPO_ROOT, _NODES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from batch_humo_render import _resolve_radio_still_path  # noqa: E402


@pytest.fixture()
def radio_png(tmp_path: Path) -> Path:
    """Create a dummy PNG file on disk so the file-existence check
    succeeds.  Content doesn't matter; the resolver only checks
    is_file()."""
    p = tmp_path / "radio_bookend_episode.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n")  # PNG magic + crlf so it's identifiable
    return p


class TestNoneCases:
    """Resolver returns None on missing / non-existent / hostile."""

    def test_none_ledger(self):
        assert _resolve_radio_still_path(None) is None

    def test_empty_dict(self):
        assert _resolve_radio_still_path({}) is None

    def test_missing_field(self):
        led = {"episode_id": "x", "lines": []}
        assert _resolve_radio_still_path(led) is None

    def test_empty_string_field(self):
        led = {"radio_bookend_path": ""}
        assert _resolve_radio_still_path(led) is None

    def test_meta_empty_string(self):
        led = {"meta": {"radio_bookend_path": ""}}
        assert _resolve_radio_still_path(led) is None

    @pytest.mark.parametrize("hostile", [
        "string", 42, [], (1, 2), True, b"bytes",
    ])
    def test_hostile_ledger_type(self, hostile):
        # Non-dict ledger should never raise, just return None.
        assert _resolve_radio_still_path(hostile) is None

    def test_meta_not_a_dict(self):
        led = {"meta": "broken"}
        assert _resolve_radio_still_path(led) is None

    def test_path_does_not_exist_returns_none(self, tmp_path):
        # File is gone but ledger still has the path stamped.  Resolver
        # MUST return None so radio-role lines fall through to the
        # portrait resolver instead of crashing on a missing image
        # mid-render.
        missing = tmp_path / "definitely_not_here.png"
        led = {"radio_bookend_path": str(missing)}
        assert _resolve_radio_still_path(led) is None


class TestTopLevelField:
    """The canonical post-2026-04-30 stamp location."""

    def test_resolves_when_top_level_exists(self, radio_png):
        led = {"radio_bookend_path": str(radio_png)}
        result = _resolve_radio_still_path(led)
        assert result is not None
        assert result == radio_png
        assert result.is_file()


class TestMetaField:
    """Belt-and-suspenders fallback location."""

    def test_resolves_when_only_meta_set(self, radio_png):
        led = {"meta": {"radio_bookend_path": str(radio_png)}}
        result = _resolve_radio_still_path(led)
        assert result is not None
        assert result == radio_png

    def test_top_level_takes_precedence_over_meta(self, tmp_path, radio_png):
        # Two different paths; top-level wins (BatchFluxRender now
        # stamps both, so they should match in practice -- but if
        # there's drift, top-level is the canonical post-2026-04-30
        # location and should win).
        meta_only = tmp_path / "radio_meta.png"
        meta_only.write_bytes(b"\x89PNG\r\n\x1a\n")
        led = {
            "radio_bookend_path": str(radio_png),
            "meta": {"radio_bookend_path": str(meta_only)},
        }
        result = _resolve_radio_still_path(led)
        assert result == radio_png  # top-level wins
        assert result != meta_only


class TestPathConversion:
    """Stamped paths can be plain strings, Path objects, or weird
    Windows paths -- resolver normalizes."""

    def test_path_object_input_works(self, radio_png):
        # If someone stamped a Path object instead of str (shouldn't
        # happen via JSON ledger but defensive), resolver still
        # handles it.
        led = {"radio_bookend_path": radio_png}
        result = _resolve_radio_still_path(led)
        assert result == radio_png

    def test_garbled_path_returns_none(self):
        led = {"radio_bookend_path": "\x00\x00not a real path"}
        # Path() may or may not raise on this; either way resolver
        # should land on None (file doesn't exist).
        assert _resolve_radio_still_path(led) is None


class TestFilesystemFallback:
    """BUG-LOCAL-121 filesystem-fallback layer (2026-04-30).

    When ledger stamping fails silently but the FLUX phase still saved
    the radio still to its deterministic location, the resolver should
    find it by reconstructing the path from ``ledger.episode_id``.
    """

    def test_filesystem_fallback_finds_by_episode_id(
        self, tmp_path: Path, monkeypatch
    ):
        # Monkeypatch otr_stills_dir to point at tmp.
        import batch_humo_render as bhr  # noqa: E402

        monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)

        # Place the radio still where FLUX would have saved it, with
        # episode_id-based naming.
        ep_id = "pending_20260430_210737"
        radio_file = tmp_path / f"radio_bookend_{ep_id}.png"
        radio_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        # Ledger has episode_id but NO radio_bookend_path stamp.
        led = {"episode_id": ep_id}
        result = _resolve_radio_still_path(led)
        assert result is not None
        assert result == radio_file
        assert result.is_file()

    def test_filesystem_fallback_when_ledger_path_stale(
        self, tmp_path: Path, monkeypatch
    ):
        # Ledger has a stamped path that no longer exists on disk
        # (e.g. file deleted, drive reorg) but the FLUX-saved file
        # IS at the deterministic episode_id location.  Resolver
        # should fall through to filesystem layer.
        import batch_humo_render as bhr  # noqa: E402

        monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)
        ep_id = "ep001"
        radio_file = tmp_path / f"radio_bookend_{ep_id}.png"
        radio_file.write_bytes(b"\x89PNG\r\n\x1a\n")

        stale = tmp_path / "old_stamped_path_gone.png"
        led = {
            "episode_id": ep_id,
            "radio_bookend_path": str(stale),
        }
        result = _resolve_radio_still_path(led)
        assert result == radio_file

    def test_filesystem_fallback_returns_none_if_no_episode_id(
        self, tmp_path: Path, monkeypatch
    ):
        import batch_humo_render as bhr  # noqa: E402

        monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)
        # Even with a radio file present, no episode_id means we
        # can't reconstruct the deterministic name.
        (tmp_path / "radio_bookend_anything.png").write_bytes(b"\x89PNG")
        led = {}  # no episode_id
        assert _resolve_radio_still_path(led) is None

    def test_filesystem_fallback_returns_none_if_episode_id_blank(
        self, tmp_path: Path, monkeypatch
    ):
        import batch_humo_render as bhr  # noqa: E402

        monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)
        led = {"episode_id": "   "}  # whitespace-only
        assert _resolve_radio_still_path(led) is None

    def test_filesystem_fallback_returns_none_if_episode_id_not_string(
        self, tmp_path: Path, monkeypatch
    ):
        import batch_humo_render as bhr  # noqa: E402

        monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)
        led = {"episode_id": 12345}  # int, not str
        assert _resolve_radio_still_path(led) is None

    def test_filesystem_fallback_returns_none_if_file_missing(
        self, tmp_path: Path, monkeypatch
    ):
        import batch_humo_render as bhr  # noqa: E402

        monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)
        # episode_id present but no matching file in stills dir.
        led = {"episode_id": "ep_no_match"}
        assert _resolve_radio_still_path(led) is None

    def test_ledger_path_takes_precedence_over_filesystem(
        self, tmp_path: Path, monkeypatch, radio_png
    ):
        # If both the ledger-stamped path AND a filesystem-deterministic
        # file exist, the ledger path wins (it's the canonical signal).
        import batch_humo_render as bhr  # noqa: E402

        monkeypatch.setattr(bhr, "otr_stills_dir", lambda: tmp_path)
        ep_id = "ep_precedence"
        fs_path = tmp_path / f"radio_bookend_{ep_id}.png"
        fs_path.write_bytes(b"\x89PNG\r\n\x1a\n")

        led = {
            "episode_id": ep_id,
            "radio_bookend_path": str(radio_png),  # ledger-stamped
        }
        result = _resolve_radio_still_path(led)
        assert result == radio_png  # ledger wins
        assert result != fs_path
