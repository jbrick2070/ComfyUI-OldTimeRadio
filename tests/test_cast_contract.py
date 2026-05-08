"""tests/test_cast_contract.py — Phase 0+ Cast Contract Extensions §1+§2."""
from __future__ import annotations

from pathlib import Path

import pytest

from nodes._otr_cast_contract import (
    LOCKED_FILENAME,
    CastContract,
    CastContractMismatch,
    CharacterEntry,
    load_locked,
    lock_to_episode,
)


def test_character_entry_matches_canonical_and_aliases():
    e = CharacterEntry(
        character_id="c01",
        canonical_name="MONTY",
        aliases=["MONTGOMERY", "MONTY T."],
        voice_spec="bark:v2/en_speaker_3",
    )
    assert e.matches("MONTY")
    assert e.matches("monty")  # case-insensitive
    assert e.matches("MONTGOMERY")
    assert e.matches("monty t.")
    assert not e.matches("SAILOR")


def test_contract_lookup_returns_canonical_or_alias():
    monty = CharacterEntry("c01", "MONTY", ["MONTGOMERY"], "bark:v2/en_speaker_3")
    sailor = CharacterEntry("c02", "SAILOR", [], "bark:v2/en_speaker_9")
    contract = CastContract(characters=[monty, sailor])
    assert contract.lookup("MONTY") is monty
    assert contract.lookup("MONTGOMERY") is monty
    assert contract.lookup("SAILOR") is sailor
    assert contract.lookup("AEGEUS") is None


def test_version_stamp_deterministic_across_insertion_order():
    monty = CharacterEntry("c01", "MONTY", ["MONTGOMERY"], "bark:v2/en_speaker_3")
    sailor = CharacterEntry("c02", "SAILOR", [], "bark:v2/en_speaker_9")
    a = CastContract(characters=[monty, sailor])
    b = CastContract(characters=[sailor, monty])
    a.stamp_version()
    b.stamp_version()
    assert a.version == b.version
    assert a.version.startswith("sha:")


def test_version_stamp_is_64_bit_sha_16_chars():
    """Round-robin Item E (2026-05-08): bumped from 8 hex chars
    (32 bits) to 16 hex chars (64 bits, ~280 trillion possibilities)
    to push collision probability past any realistic episode count."""
    contract = CastContract(
        characters=[CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")]
    )
    contract.stamp_version()
    # Format: "sha:<16 lowercase hex chars>" = 4 + 16 = 20 chars total.
    assert contract.version.startswith("sha:")
    hex_part = contract.version[4:]
    assert len(hex_part) == 16
    assert all(c in "0123456789abcdef" for c in hex_part)


def test_cast_contract_mismatch_is_exception_not_runtime_error():
    """Round-robin Item G (2026-05-08): inheritance bumped from
    RuntimeError to plain Exception so a downstream
    `except RuntimeError` doesn't silently swallow the structured
    drift signal."""
    assert issubclass(CastContractMismatch, Exception)
    assert not issubclass(CastContractMismatch, RuntimeError)


def test_version_stamp_changes_with_alias_addition():
    e1 = CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")
    e2 = CharacterEntry("c01", "MONTY", ["MONTGOMERY"], "bark:v2/en_speaker_3")
    a = CastContract(characters=[e1])
    b = CastContract(characters=[e2])
    a.stamp_version()
    b.stamp_version()
    assert a.version != b.version


def test_to_dict_from_dict_roundtrip():
    monty = CharacterEntry("c01", "MONTY", ["MONTGOMERY"], "bark:v2/en_speaker_3")
    a = CastContract(characters=[monty])
    a.stamp_version()
    b = CastContract.from_dict(a.to_dict())
    assert b.version == a.version
    assert b.characters[0].character_id == "c01"
    assert b.characters[0].aliases == ["MONTGOMERY"]


def test_lock_to_episode_writes_file_and_idempotent_on_same_version(tmp_path: Path):
    """First call writes; second call with identical contract passes
    through silently (BUG-LOCAL-122 read-and-compare-version upgrade).
    """
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    contract = CastContract(
        characters=[CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")]
    )
    locked_path = lock_to_episode(contract, episode_dir)
    assert locked_path == episode_dir / LOCKED_FILENAME
    assert locked_path.is_file()
    # Second call with same contract -- pass through, NOT raise.
    locked_path_again = lock_to_episode(contract, episode_dir)
    assert locked_path_again == locked_path


def test_lock_to_episode_raises_on_version_mismatch(tmp_path: Path):
    """Different contract on the same episode dir is a real failure."""
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    contract_a = CastContract(
        characters=[CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")]
    )
    contract_b = CastContract(
        characters=[CharacterEntry("c01", "AEGEUS", [], "bark:v2/en_speaker_5")]
    )
    lock_to_episode(contract_a, episode_dir)
    with pytest.raises(CastContractMismatch, match="drifted"):
        lock_to_episode(contract_b, episode_dir)


def test_lock_to_episode_raises_on_corrupt_existing(tmp_path: Path):
    """A corrupt locked file is a hard fault, not a silent overwrite."""
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    (episode_dir / LOCKED_FILENAME).write_text("{not json", encoding="utf-8")
    contract = CastContract(
        characters=[CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")]
    )
    with pytest.raises(RuntimeError, match="unreadable"):
        lock_to_episode(contract, episode_dir)


def test_lock_to_episode_writes_atomically_no_temp_debris(tmp_path: Path):
    """Round-robin Element 1 catch (2026-05-08): write must use a
    temp file + os.replace so an interrupted write doesn't leave a
    truncated lockfile that bricks the episode dir on rerun.

    Verifies the *post-write* directory state contains the lockfile
    and NO `.cast_contract.lock.*.tmp` debris.
    """
    episode_dir = tmp_path / "ep_atomic"
    episode_dir.mkdir()
    contract = CastContract(
        characters=[CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")]
    )
    lock_to_episode(contract, episode_dir)
    files = sorted(p.name for p in episode_dir.iterdir())
    # Exactly the lockfile, no temp debris.
    assert files == [LOCKED_FILENAME]
    # Lockfile is well-formed JSON (would have been broken under the
    # pre-fix non-atomic write_text path if interrupted).
    import json as _json
    payload = _json.loads((episode_dir / LOCKED_FILENAME).read_text(encoding="utf-8"))
    assert payload["version"].startswith("sha:")


def test_lock_to_episode_truncated_existing_raises_unreadable(tmp_path: Path):
    """Simulates the failure mode the atomic-write fix prevents:
    pre-existing 0-byte / truncated lockfile on disk. The
    read-and-compare path must surface that as `unreadable` rather
    than silently overwriting.
    """
    episode_dir = tmp_path / "ep_truncated"
    episode_dir.mkdir()
    (episode_dir / LOCKED_FILENAME).write_text("", encoding="utf-8")  # 0-byte
    contract = CastContract(
        characters=[CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")]
    )
    with pytest.raises(RuntimeError, match="unreadable"):
        lock_to_episode(contract, episode_dir)


def test_load_locked_round_trip(tmp_path: Path):
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    monty = CharacterEntry("c01", "MONTY", ["MONTGOMERY"], "bark:v2/en_speaker_3")
    contract = CastContract(characters=[monty])
    lock_to_episode(contract, episode_dir)
    loaded = load_locked(episode_dir)
    assert loaded is not None
    assert loaded.characters[0].character_id == "c01"
    assert loaded.characters[0].aliases == ["MONTGOMERY"]


def test_load_locked_returns_none_when_not_present(tmp_path: Path):
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    assert load_locked(episode_dir) is None


def test_lock_stamps_version_if_unstamped(tmp_path: Path):
    episode_dir = tmp_path / "ep_test"
    episode_dir.mkdir()
    contract = CastContract(
        characters=[CharacterEntry("c01", "MONTY", [], "bark:v2/en_speaker_3")]
    )
    assert contract.version == ""
    lock_to_episode(contract, episode_dir)
    assert contract.version.startswith("sha:")
