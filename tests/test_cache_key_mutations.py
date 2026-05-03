"""
tests/test_cache_key_mutations.py -- Phase D (BUG-LOCAL-017) regression

Covers the cache-key fix for MusicGen + AudioGen. Two test surfaces:

1. **Mutation tests:** every dimension that affects content identity
   (prompt, duration, episode_seed, cue_id) must produce a fresh sha
   prefix. Cosmetic dimensions (safe_name) must not.

2. **Lookup tests:** _find_cached returns the canonical <prefix>.wav
   when present, falls back to the newest legacy <prefix>_<ts>.wav
   timestamp when only legacy files exist, returns None on miss.

3. **Atomic write test:** _save_wav writes through .tmp + os.replace.

4. **Back-compat:** _cache_key still resolves to the canonical filename.

Consult source: docs/2026-05-02-phase-d-cache-key-consult__*
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from nodes import musicgen_theme as MT
from nodes import batch_audiogen_generator as AG


# ---------------------------------------------------------------------------
# Mutation: each input dimension that should affect identity must change
# the prefix; cosmetic ones must not.
# ---------------------------------------------------------------------------

class TestMusicGenPrefixMutations:
    BASE = dict(
        cue_id="opening",
        prompt="warm brass fanfare with snare brushes",
        duration_sec=12,
        episode_seed="seed_xyz",
    )

    def _prefix(self, **overrides):
        kwargs = {**self.BASE, **overrides}
        return MT._cache_prefix(**kwargs)

    def test_baseline_prefix_is_deterministic(self):
        a = self._prefix()
        b = self._prefix()
        assert a == b
        assert a.startswith("opening_")

    def test_seed_change_yields_fresh_prefix(self):
        assert self._prefix() != self._prefix(episode_seed="other_seed")

    def test_prompt_change_yields_fresh_prefix(self):
        assert self._prefix() != self._prefix(prompt="completely different prompt text")

    def test_duration_change_yields_fresh_prefix(self):
        assert self._prefix() != self._prefix(duration_sec=8)

    def test_cue_id_change_yields_fresh_prefix(self):
        assert self._prefix() != self._prefix(cue_id="closing")


class TestAudioGenPrefixMutations:
    BASE = dict(
        prompt="thunder rumble in distance",
        duration_sec=3.0,
        episode_seed="seed_xyz",
    )

    def _prefix(self, **overrides):
        kwargs = {**self.BASE, **overrides}
        return AG._cache_prefix(**kwargs)

    def test_baseline_prefix_is_deterministic(self):
        a = self._prefix()
        b = self._prefix()
        assert a == b
        assert a.startswith("sfx_")

    def test_seed_change_yields_fresh_prefix(self):
        assert self._prefix() != self._prefix(episode_seed="other_seed")

    def test_prompt_change_within_first_20_chars_yields_fresh_prefix(self):
        # Both safe_name and digest change here
        assert self._prefix() != self._prefix(prompt="THUNDER rumble in distance")

    def test_prompt_change_after_first_20_chars_yields_fresh_prefix(self):
        # safe_name (first 20 chars) is identical, but digest must change
        # because the full prompt feeds the hash. This is the regression
        # guard against accidentally using safe_name as an identity input.
        a = self._prefix()
        b = self._prefix(prompt="thunder rumble in distance, with massive bass drop")
        assert a != b, (
            "audiogen prefix must change on full-prompt change even when "
            "first 20 chars (safe_name) are identical"
        )

    def test_duration_change_yields_fresh_prefix(self):
        assert self._prefix() != self._prefix(duration_sec=5.0)


# ---------------------------------------------------------------------------
# Backward-compatible _cache_key still works
# ---------------------------------------------------------------------------

def test_musicgen_cache_key_returns_canonical_filename():
    name = MT._cache_key("opening", "warm brass", 12, "seed1")
    # No timestamp in canonical name
    assert name.endswith(".wav")
    assert name.count("_") == 1, (
        f"musicgen canonical filename should be <cue>_<sha>.wav (one underscore), got {name}"
    )


def test_audiogen_cache_key_returns_canonical_filename():
    name = AG._cache_key("thunder rumble", 3.0, "seed1")
    assert name.endswith(".wav")
    # sfx_<safe_name>_<sha>.wav has 3 underscores between sfx, safe_name, sha
    # safe_name itself may contain underscores (sanitized prompt). Test
    # that no millisecond timestamp suffix remains.
    stem = name[:-4]  # strip .wav
    parts = stem.split("_")
    # Last component is the 8-char hex digest; no all-digits ts at the end.
    assert len(parts[-1]) == 8 and all(
        c in "0123456789abcdef" for c in parts[-1]
    ), f"audiogen canonical filename last component should be 8-char hex sha, got {name}"


# ---------------------------------------------------------------------------
# Lookup behavior
# ---------------------------------------------------------------------------

@pytest.fixture
def cache_workspace(tmp_path):
    """Build a writable cache_dir + helper to seed canonical/legacy files."""
    return tmp_path


class TestFindCached:
    """Each module exposes its own _find_cached. Same semantics."""

    @pytest.mark.parametrize("module", [MT, AG])
    def test_returns_none_on_empty_dir(self, cache_workspace, module):
        assert module._find_cached(str(cache_workspace), "opening_abcd1234") is None

    @pytest.mark.parametrize("module", [MT, AG])
    def test_returns_canonical_when_present(self, cache_workspace, module):
        prefix = "opening_abcd1234"
        canonical = cache_workspace / f"{prefix}.wav"
        canonical.write_bytes(b"placeholder")
        result = module._find_cached(str(cache_workspace), prefix)
        assert result == str(canonical)

    @pytest.mark.parametrize("module", [MT, AG])
    def test_falls_back_to_legacy_when_only_legacy_present(
        self, cache_workspace, module
    ):
        prefix = "opening_abcd1234"
        legacy = cache_workspace / f"{prefix}_1712345678901.wav"
        legacy.write_bytes(b"placeholder")
        result = module._find_cached(str(cache_workspace), prefix)
        assert result == str(legacy)

    @pytest.mark.parametrize("module", [MT, AG])
    def test_canonical_wins_when_both_present(self, cache_workspace, module):
        prefix = "opening_abcd1234"
        canonical = cache_workspace / f"{prefix}.wav"
        canonical.write_bytes(b"canonical")
        legacy = cache_workspace / f"{prefix}_1712345678901.wav"
        legacy.write_bytes(b"legacy")
        result = module._find_cached(str(cache_workspace), prefix)
        assert result == str(canonical)

    @pytest.mark.parametrize("module", [MT, AG])
    def test_newest_filename_timestamp_wins_among_legacy(
        self, cache_workspace, module
    ):
        prefix = "opening_abcd1234"
        old = cache_workspace / f"{prefix}_1700000000000.wav"
        new = cache_workspace / f"{prefix}_1799999999999.wav"
        # Write in reverse order to verify mtime is NOT the selector
        new.write_bytes(b"new")
        old.write_bytes(b"old")
        result = module._find_cached(str(cache_workspace), prefix)
        assert result == str(new), (
            "must prefer newest filename timestamp, not mtime"
        )

    @pytest.mark.parametrize("module", [MT, AG])
    def test_does_not_match_unrelated_prefix(self, cache_workspace, module):
        # Files for a different prefix must not be returned
        (cache_workspace / "closing_zzzz9999_1700000000000.wav").write_bytes(b"x")
        result = module._find_cached(str(cache_workspace), "opening_abcd1234")
        assert result is None

    @pytest.mark.parametrize("module", [MT, AG])
    def test_handles_prefix_with_glob_metacharacters(
        self, cache_workspace, module
    ):
        """The Phase D consult flagged that ``Path.glob()`` chokes on
        glob metacharacters like ``[``. In practice the real prefix from
        ``_cache_prefix`` is always ``<role>_<8-hex>`` and cannot contain
        such characters, but the lookup function should tolerate them
        anyway because it uses ``iterdir() + startswith``, not glob.

        We use ``[`` only -- ``*``, ``?``, ``"``, ``<``, ``>``, ``|``,
        ``\\``, ``/``, ``:`` are illegal in Windows filenames.
        """
        weird = "opening_[abcd1234"
        (cache_workspace / f"{weird}.wav").write_bytes(b"weird")
        # iterdir + startswith must find this without crashing on the [
        result = module._find_cached(str(cache_workspace), weird)
        assert result == str(cache_workspace / f"{weird}.wav")


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [MT, AG])
def test_save_wav_writes_atomically(cache_workspace, module, monkeypatch):
    """_save_wav must write through a sibling .tmp file and os.replace.
    If the process is killed mid-write, the canonical path should NEVER
    contain a partial file (it either doesn't exist or is the previous
    complete version)."""
    target = cache_workspace / "opening_abcd1234.wav"

    # Simulate a fresh write
    samples = np.zeros(1000, dtype=np.float32)
    module._save_wav(str(target), samples, 32000)

    assert target.exists()
    assert not (cache_workspace / "opening_abcd1234.wav.tmp").exists(), (
        "tmp file must be cleaned up via os.replace"
    )
    # The file should be a real WAV (soundfile readable)
    import soundfile as sf
    data, sr = sf.read(str(target), dtype="float32")
    assert len(data) == 1000
    assert sr == 32000


@pytest.mark.parametrize("module", [MT, AG])
def test_save_wav_failure_cleans_up_tmp(cache_workspace, module, monkeypatch):
    """If sf.write fails after creating the .tmp file, the .tmp must be
    cleaned up so a future cache lookup doesn't see stale partial content."""
    import soundfile as sf

    target = cache_workspace / "opening_abcd1234.wav"
    tmp = cache_workspace / "opening_abcd1234.wav.tmp"

    real_write = sf.write
    call_count = {"n": 0}

    def fake_write(path, *args, **kwargs):
        call_count["n"] += 1
        # Write a partial tmp file then raise
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated failure mid-write")

    monkeypatch.setattr(sf, "write", fake_write)

    # _save_wav swallows the exception (logs warning) and cleans up tmp
    module._save_wav(str(target), np.zeros(10, dtype=np.float32), 32000)

    assert not target.exists(), "canonical path must not exist on failure"
    assert not tmp.exists(), (
        "tmp file must be cleaned up after sf.write raised"
    )
