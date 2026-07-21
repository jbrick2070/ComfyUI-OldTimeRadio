"""Unit tests for the production ledger (L1 scope: write-only, incremental
saves, fault-tolerant writes)."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from nodes.production_ledger import (  # noqa: E402
    Ledger,
    get_ledger,
    new_ledger,
    _rebase_episode_local_paths,
    _same_durable_run,
    _slugify,
    _word_count,
    _char_count,
    music_cue_spec_sha256,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_out(tmp_path) -> Path:
    return tmp_path


# ---------------------------------------------------------------------------
# Utility coverage
# ---------------------------------------------------------------------------

class TestUtilities:
    def test_durable_identity_rejects_one_sided_freeze(self):
        current = {
            "episode_id": "same_name",
            "meta": {"freeze_timestamp": "freeze-current"},
        }
        legacy_or_stale = {"episode_id": "same_name", "meta": {}}

        assert not _same_durable_run(current, legacy_or_stale)
        assert not _same_durable_run(legacy_or_stale, current)
        assert _same_durable_run(
            legacy_or_stale,
            {"episode_id": "same_name", "meta": {}},
        )

    def test_slugify_basic(self):
        assert _slugify("Black Sphere") == "black_sphere"

    def test_slugify_strips_junk(self):
        assert _slugify("  I am Artemis!  Peter Rossoni ") == "i_am_artemis_peter_rossoni"

    def test_slugify_empty_returns_default(self):
        assert _slugify("") == "episode"
        assert _slugify(None) == "episode"

    def test_slugify_limits_length(self):
        assert len(_slugify("a" * 200, limit=40)) == 40

    def test_word_count_ignores_punctuation(self):
        assert _word_count("Hello, world! How's it going?") == 5

    def test_word_count_handles_empty(self):
        assert _word_count("") == 0
        assert _word_count(None) == 0

    def test_char_count_is_len(self):
        assert _char_count("abcdef") == 6
        assert _char_count("") == 0
        assert _char_count(None) == 0


# ---------------------------------------------------------------------------
# Ledger construction + save
# ---------------------------------------------------------------------------

class TestLedgerBasics:
    def test_new_ledger_creates_structure(self, tmp_out):
        led = Ledger("signal_lost_test_20260424_000000", str(tmp_out))
        assert led.episode_id == "signal_lost_test_20260424_000000"
        # Schema bumped 2026-04-29 (l2 -> l3) for the BUG-LOCAL-100..107
        # diagnostic expansion (text_for_tts, audio_gates, transitions,
        # warmup_pad_ms, mp4_dur_s, etc). Live-pulled from
        # _otr_ledger.CURRENT_SCHEMA_VERSION so both write paths stay
        # in lockstep.
        assert led.data["schema_version"].startswith("l3-")
        # S26-A3: "sfx" key removed from schema scaffold -- sfx are
        # first-class lines[] rows in the v2 ledger; the standalone
        # sfx[] array (and its tolerant consumers) is dead.
        for key in ("cast", "scenes", "shots", "beats", "lines",
                    "music", "clips"):
            assert led.data[key] == []
        assert "sfx" not in led.data
        assert led.data["audio"] == {}
        assert led.data["final_audio_path"] is None
        assert led.data["final_video_path"] is None
        assert led.data["total_episode_dur_s"] is None
        assert led.data["total_beats"] == 0


class TestLedgerBeats:
    def test_set_beats_round_trips(self, tmp_out):
        led = Ledger("test_beats", str(tmp_out))
        led.set_beats([
            {"beat_id": "shot_001_b1", "shot_id": "shot_001",
             "scene_id": "scene_lab", "speaker": "ALICE", "char_id": "c01",
             "line_ids": ["l001", "l002"], "start_s": 0.0, "dur_s": 5.5},
            {"beat_id": "shot_001_b2", "shot_id": "shot_001",
             "scene_id": "scene_lab", "speaker": "BOB", "char_id": "c02",
             "line_ids": ["l003"], "start_s": 5.5, "dur_s": 3.2},
        ])
        assert len(led.data["beats"]) == 2
        assert led.data["total_beats"] == 2
        b1 = led.data["beats"][0]
        assert b1["beat_id"] == "shot_001_b1"
        assert b1["speaker"] == "ALICE"
        assert b1["line_ids"] == ["l001", "l002"]
        assert b1["start_s"] == 0.0
        assert b1["dur_s"] == 5.5

    def test_set_lines_carries_beat_id_and_boundary(self, tmp_out):
        led = Ledger("test_beat_lines", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "shot_id": "shot_001",
             "beat_id": "shot_001_b1", "boundary": "shot_start",
             "char_id": "c01", "text": "Hello", "start_s": 0.0, "dur_s": 1.0},
            {"line_id": "l002", "shot_id": "shot_001",
             "beat_id": "shot_001_b2", "boundary": "beat_start",
             "char_id": "c02", "text": "Hi", "start_s": 1.0, "dur_s": 1.0},
            {"line_id": "l003", "shot_id": "shot_001",
             "beat_id": "shot_001_b2", "boundary": "continue",
             "char_id": "c02", "text": "There", "start_s": 2.0, "dur_s": 1.0},
        ])
        rows = led.data["lines"]
        assert rows[0]["beat_id"] == "shot_001_b1"
        assert rows[0]["boundary"] == "shot_start"
        assert rows[1]["boundary"] == "beat_start"
        assert rows[2]["boundary"] == "continue"

    def test_set_lines_omitting_beat_fields_back_compat(self, tmp_out):
        """Older callers pre-dating beats can omit beat_id + boundary;
        ledger stores None and downstream treats as shot_start."""
        led = Ledger("test_back_compat", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "shot_id": "shot_001", "char_id": "c01",
             "text": "Hello", "start_s": 0.0, "dur_s": 1.0},
        ])
        row = led.data["lines"][0]
        assert row["beat_id"] is None
        assert row["boundary"] is None

    def test_set_lines_defaults_speaker_role_to_character(self, tmp_out):
        """STEP 1 (2026-06-22 story+cast fix): every line row is born with
        an explicit speaker_role. set_lines used to DROP the field, so the
        streaming partial-ledger path produced rows with no speaker_role
        that reached the cast auditor with an undefined role. A row that
        omits speaker_role now defaults to 'character' (matching
        init_lines_from_outline)."""
        led = Ledger("test_role_default", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "char_id": "c01", "text": "Hello"},
        ])
        row = led.data["lines"][0]
        assert row["speaker_role"] == "character"

    def test_set_lines_preserves_supplied_speaker_role(self, tmp_out):
        """A supplied speaker_role rides through set_lines unchanged."""
        led = Ledger("test_role_keep", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "char_id": "announcer",
             "speaker_role": "announcer", "text": "Tonight..."},
            {"line_id": "l002", "speaker_role": "music_open", "text": ""},
        ])
        rows = led.data["lines"]
        assert rows[0]["speaker_role"] == "announcer"
        assert rows[1]["speaker_role"] == "music_open"

    def test_path_follows_episode_id(self, tmp_out):
        led = Ledger("signal_lost_BLACK SPHERE_20260424", str(tmp_out))
        assert led.path.endswith("signal_lost_black_sphere_20260424_ledger.json")

    def test_save_writes_valid_json(self, tmp_out):
        led = Ledger("test_ep", str(tmp_out))
        path = led.save()
        assert path is not None
        assert Path(path).exists()
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        assert loaded["episode_id"] == "test_ep"
        assert loaded["total_dialogue_lines"] == 0

    def test_save_is_atomic(self, tmp_out):
        led = Ledger("atomic_test", str(tmp_out))
        led.save()
        # No stray .tmp file should remain after a successful save
        assert not Path(tmp_out / "atomic_test_ledger.json.tmp").exists()

    def test_rename_updates_path_and_data(self, tmp_out):
        # BUG-LOCAL-019 (Phase B fallout): use a proper per-episode dir
        # structure so rename_episode's hard-fail invariant has clean
        # room to operate. Earlier shape (passing tmp_out as audio_dir
        # directly) walked os.path.dirname up to the user's TEMP root,
        # which is shared across pytest sessions and quickly accumulates
        # signal_lost_*/ siblings from prior runs -- triggering the
        # Phase B both-source-and-destination-exist guard.
        old_id = "pending_20260424_000000"
        new_id = "signal_lost_black_sphere_20260424_142006"
        ep_dir = tmp_out / "episodes" / old_id
        audio_dir = ep_dir / "audio"
        audio_dir.mkdir(parents=True)
        led = Ledger(old_id, str(audio_dir))
        led.rename_episode(new_id)
        assert led.data["episode_id"] == new_id
        assert "black_sphere" in led.path


# ---------------------------------------------------------------------------
# BUG-LOCAL-108: dual-ledger fix
# ---------------------------------------------------------------------------

class TestDualLedgerFix:
    """rename_episode must atomically move the on-disk file so audio
    nodes' schema-l3 writes don't get orphaned at the pending path.
    save() must merge any schema-l3 fields from disk so it doesn't
    nuke fields the Ledger class doesn't manage."""

    def test_rename_episode_moves_file_on_disk(self, tmp_out):
        # BUG-LOCAL-019 (Phase B fallout): see test_rename_updates_path_and_data
        # comment -- same fix shape, proper per-episode dir structure.
        import json as _json
        old_id = "pending_20260424_000000"
        new_id = "signal_lost_black_sphere_20260424_142006"
        ep_dir = tmp_out / "episodes" / old_id
        audio_dir = ep_dir / "audio"
        audio_dir.mkdir(parents=True)
        led = Ledger(old_id, str(audio_dir))
        led.save()
        old_path = Path(led.path)
        assert old_path.exists()
        led.rename_episode(new_id)
        new_path = Path(led.path)
        # Old pending file should NO LONGER exist; new canonical
        # should exist with the same contents.
        assert not old_path.exists(), "BUG-108: old pending file orphaned"
        assert new_path.exists()

    def test_rename_episode_idempotent_when_id_unchanged(self, tmp_out):
        led = Ledger("pending_test", str(tmp_out))
        led.save()
        path1 = Path(led.path)
        led.rename_episode("pending_test")  # same id
        assert Path(led.path) == path1
        assert path1.exists()

    def test_path_rebase_handles_windows_slashes_and_component_boundaries(self):
        old = r"C:\output\otr\episodes\pending_1"
        new = r"C:\output\otr\episodes\signal_lost_final"
        payload = {
            "backslash": old + r"\audio\cue.wav",
            "forward": "C:/output/otr/episodes/pending_1/clips/shot.mp4",
            "external": r"C:\models\voice.wav",
            "prefix_sibling": old + r"_other\audio\not_this_run.wav",
            "prose": "the pending_1 directory moved",
        }

        rebased, count = _rebase_episode_local_paths(payload, old, new)

        assert count == 2
        assert rebased["backslash"] == new + r"\audio\cue.wav"
        assert rebased["forward"] == new + r"\clips\shot.mp4"
        assert rebased["external"] == payload["external"]
        assert rebased["prefix_sibling"] == payload["prefix_sibling"]
        assert rebased["prose"] == payload["prose"]

    @pytest.mark.parametrize("bank", [
        "media_archive", "original", "public_domain", "shakespeare",
        "scifi_news", "scifi_news_pro",
    ])
    def test_rename_rebases_shared_six_bank_episode_paths(self, tmp_out, bank):
        old_id = f"pending_{bank}"
        new_id = f"signal_lost_{bank}_final"
        old_root = tmp_out / bank / "episodes" / old_id
        old_audio = old_root / "audio"
        old_audio.mkdir(parents=True)
        cue_path = old_audio / "music_cue_opening.wav"
        cue_path.write_bytes(b"cue")
        clip_path = old_root / "clips" / "shot.mp4"
        clip_path.parent.mkdir()
        clip_path.write_bytes(b"clip")
        master_path = old_audio / f"{old_id}_master.wav"
        master_path.write_bytes(b"master")
        external_ref = tmp_out / "shared" / "voice.wav"
        external_ref.parent.mkdir(exist_ok=True)
        external_ref.write_bytes(b"voice")

        led = Ledger(old_id, str(old_audio))
        led.data["meta"] = {
            "source_bank": bank,
            "freeze_timestamp": f"freeze-{bank}",
            "audio_readiness": {"ok": True},
            "readiness_passes": [{"text_hash_before": 11, "text_hash_after": 11}],
            "content_authorship": {"sha256": "ab" * 32},
            "external_voice_ref": str(external_ref),
        }
        led.set_lines([{
            "line_id": "l001", "text": "A clean line.",
            "char_id": "c01", "speaker_role": "character",
        }])
        led.set_music([{
            "cue_id": "opening", "generation_prompt": "low radio pulse",
            "target_duration_s": 1.0, "placement": "opening",
            "anchor_line_id": "l001",
        }])
        led.data["clips"] = [{
            "line_id": "l001", "video_clip_path": str(clip_path),
        }]
        led.data["audio"] = {
            "master_audio_sha256": "cd" * 32,
            "master_audio_path": str(master_path),
            "ledger_frozen": True,
        }
        led.data["renderer_receipts"] = {
            "nested_assets": [str(cue_path), str(clip_path), str(external_ref)],
        }
        led.save()

        pending_path = Path(led.path)
        disk = json.loads(pending_path.read_text(encoding="utf-8"))
        disk_cue = disk["music"][0]
        disk_cue.update({
            "wav_path": str(cue_path), "start_s": 0.0, "dur_s": 1.0,
            "start_s_space": "master_mix",
        })
        disk_cue["cue_spec_sha256"] = music_cue_spec_sha256(disk_cue)
        disk["lines"].append({
            "line_id": "music_opening_001", "speaker_role": "music_open",
            "mirrored_from": "music", "music_cue_id": "opening",
            "start_s": 0.0, "dur_s": 1.0,
            "start_s_space": "master_mix",
        })
        pending_path.write_text(json.dumps(disk, indent=2), encoding="utf-8")

        sealed_before = {
            "freeze_timestamp": disk["meta"]["freeze_timestamp"],
            "audio_readiness": disk["meta"]["audio_readiness"],
            "readiness_passes": disk["meta"]["readiness_passes"],
            "content_authorship": disk["meta"]["content_authorship"],
            "master_audio_sha256": disk["audio"]["master_audio_sha256"],
            "cue_spec_sha256": disk_cue["cue_spec_sha256"],
        }

        led.rename_episode(new_id)
        led.save()

        final_path = Path(led.path)
        final = json.loads(final_path.read_text(encoding="utf-8"))
        final_root = old_root.parent / new_id
        assert not old_root.exists()
        assert final_path.exists()
        assert Path(final["music"][0]["wav_path"]).is_file()
        assert Path(final["clips"][0]["video_clip_path"]).is_file()
        assert Path(final["audio"]["master_audio_path"]).is_file()
        _unused, stale_path_count = _rebase_episode_local_paths(
            final, str(old_root), str(final_root),
        )
        assert stale_path_count == 0
        assert Path(final["meta"]["paths"]["episode_root"]) == final_root
        assert Path(final["meta"]["paths"]["audio_dir"]) == final_root / "audio"
        assert Path(final["renderer_receipts"]["nested_assets"][0]).is_file()
        assert Path(final["renderer_receipts"]["nested_assets"][1]).is_file()
        assert final["renderer_receipts"]["nested_assets"][2] == str(external_ref)
        assert final["meta"]["external_voice_ref"] == str(external_ref)
        mirrors = [
            row for row in final["lines"]
            if row.get("mirrored_from") == "music"
        ]
        assert [row["line_id"] for row in mirrors] == ["music_opening_001"]
        assert final["meta"]["freeze_timestamp"] == sealed_before["freeze_timestamp"]
        assert final["meta"]["audio_readiness"] == sealed_before["audio_readiness"]
        assert final["meta"]["readiness_passes"] == sealed_before["readiness_passes"]
        assert final["meta"]["content_authorship"] == sealed_before["content_authorship"]
        assert final["audio"]["master_audio_sha256"] == sealed_before["master_audio_sha256"]
        assert final["music"][0]["cue_spec_sha256"] == sealed_before["cue_spec_sha256"]
        assert Path(final["music"][0]["wav_path"]).is_relative_to(final_root)

    def test_merge_rejects_foreign_or_reauthored_disk_only_music_mirrors(self, tmp_out):
        led = Ledger("ep_mirror_gate", str(tmp_out))
        led.data["meta"] = {"freeze_timestamp": "freeze-current"}
        led.set_lines([{
            "line_id": "l001", "text": "Current line.",
            "char_id": "c01", "speaker_role": "character",
        }])
        led.set_music([{
            "cue_id": "opening", "generation_prompt": "current pulse",
            "target_duration_s": 1.0, "placement": "opening",
            "anchor_line_id": "l001",
        }])
        led.save()
        path = Path(led.path)
        disk = json.loads(path.read_text(encoding="utf-8"))
        disk["meta"]["freeze_timestamp"] = "freeze-foreign"
        disk["music"][0]["generation_prompt"] = "stale pulse"
        disk["music"][0]["cue_spec_sha256"] = music_cue_spec_sha256(
            disk["music"][0]
        )
        disk["lines"].extend([
            {
                "line_id": "music_opening_001", "speaker_role": "music_open",
                "mirrored_from": "music", "music_cue_id": "opening",
                "start_s": 0.0, "dur_s": 1.0,
                "start_s_space": "master_mix",
            },
            {"line_id": "l999", "text": "stale ordinary disk content"},
        ])
        path.write_text(json.dumps(disk, indent=2), encoding="utf-8")

        led.save()
        merged = json.loads(path.read_text(encoding="utf-8"))

        assert not any(
            row.get("mirrored_from") == "music" for row in merged["lines"]
        )
        assert not any(row.get("line_id") == "l999" for row in merged["lines"])
        assert merged["meta"]["freeze_timestamp"] == "freeze-current"

    def test_save_merges_schema_l3_fields_from_disk(self, tmp_out):
        """Audio nodes write schema-l3 fields directly via
        _otr_ledger.save_ledger_safe. The Ledger class must
        NOT clobber those fields when its own .save() fires."""
        import json as _json
        led = Ledger("merge_test", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "text": "hello",
             "char_id": "c01", "char_count": 5, "word_count": 1},
        ])
        led.save()
        path = Path(led.path)
        # Simulate an audio-node write: read the on-disk file, add
        # schema-l3 fields, write back.
        on_disk = _json.loads(path.read_text(encoding="utf-8"))
        on_disk["audio_gates"] = [
            {"gate": "post_bark", "sha256_first_kb": "abc123",
             "dur_s": 10.0, "sample_count": 240000, "sample_rate": 24000},
        ]
        on_disk["audio"] = {
            "master_audio_sha256": "ab" * 32,
            "ledger_frozen": True,
        }
        on_disk["meta"] = {"phase_ms": {"bark": 145000}, "git_commit": "deadbee"}
        on_disk["transitions"] = [
            {"from_line_id": "opening_theme", "to_line_id": "scene_audio",
             "crossfade_ms": 500, "boundary_s": 9.5},
        ]
        on_disk["radio_bookend_path"] = "/tmp/radio.png"
        on_disk["lines"][0]["text_for_tts"] = "hello"
        on_disk["lines"][0]["bark_wav_dur_s"] = 0.42
        on_disk["lines"][0]["start_s"] = 9.5
        on_disk["lines"][0]["start_s_space"] = "master_mix"
        path.write_text(_json.dumps(on_disk, indent=2), encoding="utf-8")
        # Now the Ledger class re-saves (e.g. SignalLostVideo's
        # post-rename save). Its in-memory state has only the
        # original lines[] without schema-l3. The merge must
        # preserve all the schema-l3 fields above.
        led.save()
        merged = _json.loads(path.read_text(encoding="utf-8"))
        assert "audio_gates" in merged and len(merged["audio_gates"]) == 1
        assert merged["audio"] == {
            "master_audio_sha256": "ab" * 32,
            "ledger_frozen": True,
        }
        assert merged["audio_gates"][0]["gate"] == "post_bark"
        assert merged["meta"]["phase_ms"]["bark"] == 145000
        assert merged["meta"]["git_commit"] == "deadbee"
        assert "transitions" in merged and len(merged["transitions"]) == 1
        assert merged["radio_bookend_path"] == "/tmp/radio.png"
        l001 = merged["lines"][0]
        assert l001["text_for_tts"] == "hello"
        assert l001["bark_wav_dur_s"] == 0.42
        assert l001["start_s"] == 9.5
        assert l001["start_s_space"] == "master_mix"

    def test_save_does_not_overwrite_in_mem_with_disk(self, tmp_out):
        """In-memory values for fields the Ledger class manages must
        win over disk values (it's the fresh state). Only fields the
        class doesn't manage get merged from disk."""
        import json as _json
        led = Ledger("freshness_test", str(tmp_out))
        led.set_lines([{"line_id": "l001", "text": "ORIGINAL",
                        "char_id": "c01", "char_count": 8, "word_count": 1}])
        led.save()
        path = Path(led.path)
        # Disk value for `text` is "ORIGINAL" too. Update in-memory.
        led.set_lines([{"line_id": "l001", "text": "UPDATED",
                        "char_id": "c01", "char_count": 7, "word_count": 1}])
        led.save()
        merged = _json.loads(path.read_text(encoding="utf-8"))
        assert merged["lines"][0]["text"] == "UPDATED", \
            "in-memory text must win over stale on-disk value"

    def test_save_preserves_disk_rows_when_in_mem_array_empty(self, tmp_out):
        """If memory has no rows for a ROW_KEYED array but disk has
        rows, keep the disk rows. Was sfx-based pre-S27; migrated to
        music in lockstep with the S27 Item 2 sfx-scaffold deletion."""
        import json as _json
        led = Ledger("empty_mem_test", str(tmp_out))
        led.save()
        path = Path(led.path)
        on_disk = _json.loads(path.read_text(encoding="utf-8"))
        on_disk["music"] = [
            {"cue_id": "opening_theme", "start_s": 0.0, "dur_s": 12.0,
             "description": "opening theme", "wav_path": "/tmp/opening.wav"},
        ]
        path.write_text(_json.dumps(on_disk, indent=2), encoding="utf-8")
        led.save()  # in-mem music is empty []
        merged = _json.loads(path.read_text(encoding="utf-8"))
        assert len(merged["music"]) == 1
        assert merged["music"][0]["cue_id"] == "opening_theme"

    # -- durable-field identity (S2 P1.1 / 720-bakeoff C1) -------------

    def _seed_line_with_disk_render(self, tmp_out, in_mem_text, disk_text):
        """Save a ledger with a single line, then simulate an audio node
        writing DURABLE render fields onto the on-disk row (with its own
        ``disk_text``). Return (led, path) with the in-memory line carrying
        ``in_mem_text``, ready for a second .save() to exercise the merge."""
        import json as _json
        led = Ledger("durable_line", str(tmp_out))
        led.set_lines([{"line_id": "l001", "text": disk_text,
                        "char_id": "c01", "char_count": len(disk_text),
                        "word_count": 1}])
        led.save()
        path = Path(led.path)
        on_disk = _json.loads(path.read_text(encoding="utf-8"))
        on_disk["lines"][0]["text_for_tts"] = disk_text
        on_disk["lines"][0]["bark_wav_dur_s"] = 0.42
        on_disk["lines"][0]["bark_wav_path"] = "/tmp/l001.wav"
        on_disk["lines"][0]["start_s"] = 9.5
        path.write_text(_json.dumps(on_disk, indent=2), encoding="utf-8")
        # Point in-memory at the (possibly edited) text.
        led.set_lines([{"line_id": "l001", "text": in_mem_text,
                        "char_id": "c01", "char_count": len(in_mem_text),
                        "word_count": 1}])
        return led, path

    def test_unchanged_line_text_retains_durable_render_fields(self, tmp_out):
        """Same-hash-retain: an unedited line keeps the on-disk durable
        render fields (BUG-108 preservation still holds)."""
        import json as _json
        led, path = self._seed_line_with_disk_render(
            tmp_out, in_mem_text="hello", disk_text="hello")
        led.save()
        row = _json.loads(path.read_text(encoding="utf-8"))["lines"][0]
        assert row["text"] == "hello"
        assert row["text_for_tts"] == "hello"
        assert row["bark_wav_dur_s"] == 0.42
        assert row["bark_wav_path"] == "/tmp/l001.wav"
        assert row["start_s"] == 9.5

    def test_changed_line_text_invalidates_durable_render_fields(self, tmp_out):
        """Changed-hash-invalidate: editing the line text drops the stale
        on-disk render fields instead of gluing them onto the new text."""
        import json as _json
        led, path = self._seed_line_with_disk_render(
            tmp_out, in_mem_text="a completely different line", disk_text="hello")
        led.save()
        row = _json.loads(path.read_text(encoding="utf-8"))["lines"][0]
        assert row["text"] == "a completely different line"
        # The stale render fields must NOT be resurrected onto the edit.
        assert row.get("text_for_tts") is None
        assert row.get("bark_wav_dur_s") is None
        assert row.get("bark_wav_path") is None
        assert row.get("start_s") is None

    def test_legacy_music_row_without_hash_migrates_on_identity_match(self, tmp_out):
        """Legacy read-compat: an on-disk music row written before
        ``cue_spec_sha256`` existed still carries its durable wav forward,
        because identity is recomputed from CONTENT (not a stored hash)."""
        import json as _json
        led = Ledger("legacy_music", str(tmp_out))
        led.save()
        path = Path(led.path)
        on_disk = _json.loads(path.read_text(encoding="utf-8"))
        # Legacy row: NO cue_spec_sha256, NO authored spec fields, just the
        # old six-key shape + a rendered wav.
        on_disk["music"] = [{
            "cue_id": "opening", "description": "eerie theremin",
            "generation_prompt": "eerie theremin", "wav_path": "/tmp/open.wav",
            "start_s": 0.0, "dur_s": 12.0,
        }]
        path.write_text(_json.dumps(on_disk, indent=2), encoding="utf-8")
        # Fresh in-memory authoring with the SAME generation_prompt.
        led.set_music([{"cue_id": "opening", "description": "eerie theremin",
                        "generation_prompt": "eerie theremin"}])
        led.save()
        row = _json.loads(path.read_text(encoding="utf-8"))["music"][0]
        assert row["wav_path"] == "/tmp/open.wav"
        assert row["dur_s"] == 12.0

    def test_changed_music_cue_spec_invalidates_wav(self, tmp_out):
        """Re-authoring a cue (different generation_prompt) invalidates the
        stale rendered wav rather than keeping the wrong audio."""
        import json as _json
        led = Ledger("music_invalidate", str(tmp_out))
        led.set_music([{"cue_id": "opening",
                        "generation_prompt": "eerie theremin"}])
        led.save()
        path = Path(led.path)
        on_disk = _json.loads(path.read_text(encoding="utf-8"))
        on_disk["music"][0]["wav_path"] = "/tmp/old_theremin.wav"
        on_disk["music"][0]["start_s"] = 0.0
        on_disk["music"][0]["dur_s"] = 12.0
        path.write_text(_json.dumps(on_disk, indent=2), encoding="utf-8")
        # Re-author the cue with a NEW prompt (changed cue_spec).
        led.set_music([{"cue_id": "opening",
                        "generation_prompt": "bright brass fanfare"}])
        led.save()
        row = _json.loads(path.read_text(encoding="utf-8"))["music"][0]
        assert row["generation_prompt"] == "bright brass fanfare"
        assert row.get("wav_path") is None, "stale wav must not resurrect"
        assert row.get("dur_s") is None

    def test_set_music_carries_authored_fields_and_stamps_hash(self, tmp_out):
        """set_music carries the authored cue-spec fields it dropped before
        C1 and stamps a deterministic cue_spec_sha256."""
        led = Ledger("music_authored", str(tmp_out))
        led.set_music([{
            "cue_id": "inter_01", "description": "rising strings",
            "generation_prompt": "rising strings under dialogue",
            "placement": "inter", "anchor_line_id": "shot_002_b1",
            "target_duration_s": 4.0,
        }])
        row = led.data["music"][0]
        assert row["placement"] == "inter"
        assert row["anchor_line_id"] == "shot_002_b1"
        assert row["target_duration_s"] == 4.0
        h = row["cue_spec_sha256"]
        assert isinstance(h, str) and len(h) == 64
        # Deterministic: same spec -> same hash; changed prompt -> different.
        led.set_music([{
            "cue_id": "inter_01", "description": "rising strings",
            "generation_prompt": "rising strings under dialogue",
            "placement": "inter", "anchor_line_id": "shot_002_b1",
            "target_duration_s": 4.0,
        }])
        assert led.data["music"][0]["cue_spec_sha256"] == h
        led.set_music([{
            "cue_id": "inter_01", "description": "rising strings",
            "generation_prompt": "DIFFERENT prompt",
            "placement": "inter", "anchor_line_id": "shot_002_b1",
            "target_duration_s": 4.0,
        }])
        assert led.data["music"][0]["cue_spec_sha256"] != h

    def test_identity_match_preserves_render_owned_music_shot_id(self, tmp_out):
        """A later Ledger.save must not erase SceneSequencer's anchor-derived
        music shot_id while line/clip shot ownership stays authoritative."""
        import json as _json

        led = Ledger("music_shot_receipt", str(tmp_out))
        led.set_music([{
            "cue_id": "inter_01", "generation_prompt": "brief bridge",
            "placement": "interstitial", "anchor_line_id": "l002",
        }])
        led.save()
        path = Path(led.path)
        on_disk = _json.loads(path.read_text(encoding="utf-8"))
        on_disk["music"][0].update({
            "shot_id": "shot_002", "start_s": 3.0, "dur_s": 1.0,
            "start_s_space": "master_mix",
        })
        path.write_text(_json.dumps(on_disk, indent=2), encoding="utf-8")

        led.save()
        row = _json.loads(path.read_text(encoding="utf-8"))["music"][0]
        assert row["shot_id"] == "shot_002"
        assert row["start_s_space"] == "master_mix"


# ---------------------------------------------------------------------------
# Setters
# ---------------------------------------------------------------------------

class TestSetters:
    def _sample_cast(self):
        return [
            {"char_id": "c01", "name": "EDNA",
             "gender": "female", "voice_preset": "v2/en_speaker_2"},
            {"char_id": "c02", "name": "BOB",
             "gender": "male", "voice_preset": "v2/en_speaker_1"},
        ]

    def _sample_lines(self):
        return [
            {"line_id": "l001", "shot_id": "sh01", "char_id": "c01",
             "text": "We're flying blind without GPS.",
             "traits": "Female, 40s, urgent"},
            {"line_id": "l002", "shot_id": "sh01", "char_id": "c02",
             "text": "Edna, I've taken control of some military satellites.",
             "traits": "Male, 30s, clipped"},
            {"line_id": "l003", "shot_id": "sh02", "char_id": "c01",
             "text": "Bob? What are you doing here?",
             "traits": "startled"},
        ]

    def test_set_cast_normalizes_rows(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_cast(self._sample_cast())
        assert len(led.data["cast"]) == 2
        assert led.data["cast"][0]["char_id"] == "c01"
        assert led.data["cast"][0]["gender"] == "female"

    def test_set_cast_preserves_tts_model_when_supplied(self, tmp_out):
        """If the caller supplies tts_model on the input row, set_cast
        writes it through verbatim."""
        led = Ledger("t", str(tmp_out))
        led.set_cast([
            {"char_id": "c01", "name": "ANNOUNCER",
             "gender": "male", "tts_model": "kokoro",
             "voice_preset": "bm_george"},
            {"char_id": "c02", "name": "LEMMY",
             "gender": "male", "tts_model": "bark",
             "voice_preset": "v2/en_speaker_8"},
        ])
        assert led.data["cast"][0]["tts_model"] == "kokoro"
        assert led.data["cast"][1]["tts_model"] == "bark"

    # S26-B3: legacy `_derive_tts_model_from_voice_preset` shim removed;
    # the two derivation tests (bark-preset + kokoro-preset) deleted with it
    # per the "delete the test AND the dead caller" pattern. The
    # test_set_cast_tts_model_none_for_unknown_preset case below still
    # pins the contract: if tts_model is not supplied, it lands as None.

    def test_set_cast_tts_model_none_for_unknown_preset(self, tmp_out):
        """If the voice_preset doesn't match any known TTS prefix and
        tts_model isn't supplied, the field lands as None and the
        consumer is responsible for error-checking."""
        led = Ledger("t", str(tmp_out))
        led.set_cast([
            {"char_id": "c01", "name": "MYSTERY",
             "gender": "other", "voice_preset": "future_tts_xyz"},
        ])
        assert led.data["cast"][0]["tts_model"] is None

    def test_set_cast_voice_params_defaults_to_none(self, tmp_out):
        """If the caller doesn't supply voice_params, the field
        defaults to None and consumers fall back to their defaults."""
        led = Ledger("t", str(tmp_out))
        led.set_cast([
            {"char_id": "c01", "name": "BOB",
             "gender": "male", "tts_model": "bark",
             "voice_preset": "v2/en_speaker_1"},
        ])
        assert led.data["cast"][0]["voice_params"] is None

    def test_set_cast_voice_params_preserves_dict(self, tmp_out):
        """A caller-supplied voice_params dict flows through verbatim."""
        led = Ledger("t", str(tmp_out))
        led.set_cast([
            {"char_id": "c01", "name": "BOB",
             "gender": "male", "tts_model": "bark",
             "voice_preset": "v2/en_speaker_1",
             "voice_params": {"temperature": 0.75}},
        ])
        assert led.data["cast"][0]["voice_params"] == {"temperature": 0.75}

    def test_set_cast_voice_params_non_dict_coerced_to_none(self, tmp_out):
        """A stray non-dict / non-None voice_params (e.g. a string)
        gets coerced to None so the ledger schema stays clean."""
        led = Ledger("t", str(tmp_out))
        led.set_cast([
            {"char_id": "c01", "name": "BOB",
             "gender": "male", "tts_model": "bark",
             "voice_preset": "v2/en_speaker_1",
             "voice_params": "not a dict"},
        ])
        assert led.data["cast"][0]["voice_params"] is None

    def test_set_lines_computes_counts(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines(self._sample_lines())
        rows = led.data["lines"]
        assert len(rows) == 3
        assert rows[0]["char_count"] == len("We're flying blind without GPS.")
        assert rows[0]["word_count"] == 5
        assert led.data["total_dialogue_lines"] == 3
        total_words = sum(r["word_count"] for r in rows)
        assert led.data["total_word_count"] == total_words

    def test_totals_roll_up_per_character(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_cast(self._sample_cast())
        led.set_lines(self._sample_lines())
        edna = next(c for c in led.data["cast"] if c["char_id"] == "c01")
        bob = next(c for c in led.data["cast"] if c["char_id"] == "c02")
        assert edna["line_count"] == 2
        assert bob["line_count"] == 1
        assert edna["word_count"] > 0
        assert bob["word_count"] > 0

    def test_totals_roll_up_per_scene_via_shot(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_scenes([{"scene_id": "s01", "env": "NASA control room"}])
        led.set_shots([
            {"shot_id": "sh01", "scene_id": "s01", "visual_prompt": "X"},
            {"shot_id": "sh02", "scene_id": "s01", "visual_prompt": "Y"},
        ])
        led.set_lines(self._sample_lines())
        s01 = led.data["scenes"][0]
        assert s01["line_count"] == 3  # all three lines land in s01

    def test_set_final_paths(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_final_paths(
            audio_path="/tmp/final.wav",
            video_path="/tmp/final.mp4",
            total_episode_dur_s=421.5,
        )
        assert led.data["final_audio_path"] == "/tmp/final.wav"
        assert led.data["final_video_path"] == "/tmp/final.mp4"
        assert led.data["total_episode_dur_s"] == 421.5


# ---------------------------------------------------------------------------
# Timing back-fill
# ---------------------------------------------------------------------------

class TestTimingBackfill:
    def test_apply_line_timings(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines([
            {"line_id": "l001", "char_id": "c01", "text": "A"},
            {"line_id": "l002", "char_id": "c02", "text": "B"},
        ])
        led.apply_line_timings({
            "l001": {"start_s": 12.4, "dur_s": 2.1, "bark_wav_path": "/tmp/a.wav"},
            "l002": {"start_s": 14.5, "dur_s": 1.8, "bark_wav_path": "/tmp/b.wav"},
        })
        rows = led.data["lines"]
        assert rows[0]["start_s"] == 12.4
        assert rows[0]["bark_wav_path"] == "/tmp/a.wav"
        assert rows[1]["start_s"] == 14.5

    def test_apply_line_timings_ignores_unknown_ids(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines([{"line_id": "l001", "text": "Hi"}])
        led.apply_line_timings({"l999": {"start_s": 1.0, "dur_s": 1.0}})
        assert led.data["lines"][0]["start_s"] is None

    def test_apply_music_timings(self, tmp_out):
        # Sfx half of the prior `test_apply_sfx_and_music_timings`
        # deleted S27 (cleanbreak-tail Item 2) along with set_sfx /
        # apply_sfx_timings. The music half stays -- music timings
        # are still part of the active ledger contract.
        led = Ledger("t", str(tmp_out))
        led.set_music([{"cue_id": "opening"}])
        led.apply_music_timings({"opening": {"start_s": 0.0, "dur_s": 12.0, "wav_path": "/tmp/m.wav"}})
        assert led.data["music"][0]["dur_s"] == 12.0
        assert led.data["music"][0]["wav_path"] == "/tmp/m.wav"


# ---------------------------------------------------------------------------
# Fault tolerance
# ---------------------------------------------------------------------------

class TestFaultTolerance:
    def test_bad_input_does_not_raise(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_cast([{"char_id": None, "line_count": "bogus"}])
        # Should have coerced line_count to 0 rather than raising.
        assert led.data["cast"][0]["line_count"] == 0

    def test_save_never_raises_even_with_bad_out_dir(self, tmp_out, monkeypatch):
        led = Ledger("t", str(tmp_out))
        # Simulate disk failure by monkeypatching os.replace.
        import nodes.production_ledger as pl
        def _boom(*a, **kw):
            raise OSError("disk full")
        monkeypatch.setattr(pl.os, "replace", _boom)
        assert led.save() is None  # returned None, did not raise

    def test_save_is_idempotent(self, tmp_out):
        led = Ledger("t", str(tmp_out))
        led.set_lines([{"line_id": "l001", "text": "Hi"}])
        p1 = led.save()
        p2 = led.save()
        assert p1 == p2
        assert Path(p1).exists()


# ---------------------------------------------------------------------------
# Singleton accessors
# ---------------------------------------------------------------------------

class TestSingletonAccessors:
    def test_new_ledger_resets_current(self, tmp_out):
        a = new_ledger("ep_a", str(tmp_out))
        assert get_ledger() is a
        b = new_ledger("ep_b", str(tmp_out))
        assert get_ledger() is b
        assert b.episode_id == "ep_b"

    def test_get_ledger_creates_placeholder_if_none(self, tmp_out):
        # Reset singleton via new_ledger then null it through a fresh import would
        # be overkill; just confirm get_ledger never returns None.
        led = get_ledger()
        assert led is not None


# ---------------------------------------------------------------------------
# C0 (2026-06-22 story-quality R3) -- compose_flags + arc_phase plumbing
# ---------------------------------------------------------------------------

class TestComposeFlagsPlumbing:
    def test_set_lines_preserves_compose_flags_and_arc_phase(self, tmp_out):
        """set_lines used to DROP compose_flags + arc_phase entirely. A row
        carrying them must round-trip through set_lines unchanged."""
        led = Ledger("test_cf_preserve", str(tmp_out))
        led.set_lines([
            {"line_id": "b001", "char_id": "c01", "text": "Hi",
             "compose_flags": ["role_coerce:x", "stage_dir_stripped:bare"],
             "arc_phase": "rising"},
        ])
        row = led.data["lines"][0]
        assert row["compose_flags"] == ["role_coerce:x", "stage_dir_stripped:bare"]
        assert row["arc_phase"] == "rising"

    def test_set_lines_defaults_compose_flags_to_empty_list(self, tmp_out):
        """A row omitting compose_flags / arc_phase gets [] / None (not a
        missing key) so downstream readers never KeyError."""
        led = Ledger("test_cf_default", str(tmp_out))
        led.set_lines([{"line_id": "b001", "char_id": "c01", "text": "Hi"}])
        row = led.data["lines"][0]
        assert row["compose_flags"] == []
        assert row["arc_phase"] is None

    def test_set_lines_coerces_non_list_compose_flags(self, tmp_out):
        """A malformed (non-list) compose_flags is coerced to [] -- never
        propagated as a string/None into the schema."""
        led = Ledger("test_cf_coerce", str(tmp_out))
        led.set_lines([
            {"line_id": "b001", "char_id": "c01", "text": "Hi",
             "compose_flags": "oops"},
        ])
        assert led.data["lines"][0]["compose_flags"] == []

    def test_update_line_text_appends_compose_flags(self, tmp_out):
        led = Ledger("test_cf_append", str(tmp_out))
        led.set_lines([
            {"line_id": "b001", "char_id": "c01", "text": "old",
             "compose_flags": ["pre_existing"]},
        ])
        ok = led.update_line_text(
            "b001", "new line", compose_flags_append=["objective_literal_retry"],
        )
        assert ok is True
        row = led.data["lines"][0]
        assert row["text"] == "new line"
        assert row["compose_flags"] == ["pre_existing", "objective_literal_retry"]
        assert row["word_count"] == 2

    def test_update_line_text_append_coerces_missing_flags(self, tmp_out):
        """update_line_text(append) on a row with no compose_flags coerces to
        [] before appending."""
        led = Ledger("test_cf_append2", str(tmp_out))
        led.set_lines([{"line_id": "b001", "char_id": "c01", "text": "old"}])
        led.data["lines"][0].pop("compose_flags", None)
        led.update_line_text("b001", "x", compose_flags_append=["f1", "f1"])
        # duplicates preserved -- a count of the same flag is meaningful.
        assert led.data["lines"][0]["compose_flags"] == ["f1", "f1"]

    def test_update_line_text_no_flags_is_unchanged_behaviour(self, tmp_out):
        """Calling without compose_flags_append leaves an existing flags list
        untouched (byte-identical to pre-C0 behaviour)."""
        led = Ledger("test_cf_noop", str(tmp_out))
        led.set_lines([
            {"line_id": "b001", "char_id": "c01", "text": "old",
             "compose_flags": ["keep"]},
        ])
        led.update_line_text("b001", "new")
        assert led.data["lines"][0]["compose_flags"] == ["keep"]

    def test_update_line_text_clears_skip_state_on_recomposed_text(self, tmp_out):
        """A rerolled replacement must not remain editorially skipped.

        Script Doctor can clear a row and stamp skip metadata before the
        targeted reroll puts fresh text back.  Keeping that metadata creates
        the production-only Phase 10 invariant failure ``skip=True + text``.
        """
        led = Ledger("test_reroll_clears_skip", str(tmp_out))
        led.set_lines([{"line_id": "b004", "char_id": "c01", "text": ""}])
        row = led.data["lines"][0]
        row.update({
            "skip": True,
            "tts_skip_reason": "script_doctor_skip",
            "reviewer_skip_reason": "flat_line",
        })

        assert led.update_line_text("b004", "The signal returns.") is True

        assert row["text"] == "The signal returns."
        assert row.get("skip") is None
        assert row.get("tts_skip_reason") is None
        assert row.get("reviewer_skip_reason") is None

    def test_update_line_text_preserves_skip_state_for_empty_text(self, tmp_out):
        """An empty replacement remains skipped until real text arrives."""
        led = Ledger("test_empty_keeps_skip", str(tmp_out))
        led.set_lines([{"line_id": "b004", "char_id": "c01", "text": ""}])
        row = led.data["lines"][0]
        row.update({"skip": True, "tts_skip_reason": "empty"})

        assert led.update_line_text("b004", "   ") is True

        assert row["skip"] is True
        assert row["tts_skip_reason"] == "empty"

    def test_compose_flags_survive_full_round_trip(self, tmp_out):
        """The plan's regression: init/set -> update_line_text(append) ->
        set_lines -> save -> reload -> flags still present."""
        led = Ledger("test_cf_roundtrip", str(tmp_out))
        led.set_lines([
            {"line_id": "b001", "char_id": "c01", "text": "old",
             "arc_phase": "climax"},
        ])
        led.update_line_text("b001", "mid", compose_flags_append=["minted_in_reroll"])
        # round-trip the rows back through set_lines (the path that dropped them)
        led.set_lines(led.data["lines"])
        led.save()
        with open(led.path, "r", encoding="utf-8") as fh:
            reloaded = json.load(fh)
        row = reloaded["lines"][0]
        assert "minted_in_reroll" in row["compose_flags"]
        assert row["arc_phase"] == "climax"
