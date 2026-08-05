"""Render-driver coverage for Google video SFX-bed routing."""
from __future__ import annotations

import logging
import pathlib
import shutil

import pytest

from nodes._otr_video_engines import render_driver as rd


def _sfx_open_ledger(engine_id="google_vid_sfx_veo_lite"):
    return {
        "audio": {"master_audio_sha256": rd.FROZEN_AUDIO_SHA, "ledger_frozen": True},
        "meta": {
            "visual_style": "video_art",
            "style": "archive signal mystery",
            "story_brief_terms": {
                "setting": ["tape archive"],
                "atmosphere": ["electrical unease"],
            },
        },
        "lines": [
            {"line_id": "b000_music_open", "char_id": "",
             "speaker_role": "music_open", "start_s": 0.0, "dur_s": 4.0},
        ],
        "images": {"images": [
            {"object_id": "still_b000_music_open", "kind": "scene_open",
             "beat_id": "b000_music_open", "path": "X:/img/open.png"},
        ]},
        "video": {"video_revision": 1, "fps": 25, "shots": [
            {"shot_id": "shot_b000_music_open",
             "source_line_ids": ["b000_music_open"],
             "role": "music_visual",
             "engine_id": engine_id,
             "family": "text_to_video",
             "group_id": "grp_music",
             "target_frame_count": 100,
             "degradation_trail": [],
             "render_request_hash": "hash_sfx_open",
             "cache_keys": {"request_hash": "hash_sfx_open"},
             "creative": {}},
        ]},
    }


def test_sfx_google_prompt_routing_avoids_silent_google_visual_finishing():
    led = _sfx_open_ledger()
    shot = led["video"]["shots"][0]
    req = rd.build_request_from_shot(shot, led)
    prompt = req["text_prompt"].lower()
    assert req["audio_ref"] is None
    assert req["asset_refs"]["init_image"] == "X:/img/open.png"
    assert req["observability"]["prompt_source"] == "motion_role"
    assert "single luminous radio receiver" not in prompt
    assert "anthropomorphic radio receiver" not in prompt
    assert "no spoken words" not in prompt
    assert "no blood" not in prompt
    assert req["observability"].get("visual_safety_prompt") is None


def test_silent_google_prompt_routing_still_uses_visual_google_branch():
    led = _sfx_open_ledger(engine_id="google_veo_video")
    shot = led["video"]["shots"][0]
    req = rd.build_request_from_shot(shot, led)
    prompt = req["text_prompt"].lower()
    assert "single luminous radio receiver" in prompt
    # Content clause retired 2026-08-05 (operator directive); the SFX bed's
    # functional no-speech text still applies and is asserted elsewhere.
    assert "no explicit guns" not in prompt
    # NOT "applied" any more, and that is correct. `_apply_visual_safety_prompt`
    # early-returns when the helper changes nothing, and the retired clause
    # changes nothing -- so stamping "applied" would assert an enforcement that
    # did not happen. Matches the sibling assertion above.
    assert req["observability"].get("visual_safety_prompt") is None


def test_persist_sfx_stem_even_when_mp4_already_persisted(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "out"))
    dest = tmp_path / "out" / "otr" / "episodes" / "ep_sfx" / "clips"
    dest.mkdir(parents=True)
    video = dest / "shot1_music_visual_google_vid_sfx_veo_lite.mp4"
    video.write_bytes(b"persisted video")
    sfx_tmp = tmp_path / "stem.sfx.wav"
    sfx_tmp.write_bytes(b"sfx")
    result = {
        "ledger": {"video": {"shots": [
            {"shot_id": "shot1", "role": "music_visual",
             "engine_id": "google_vid_sfx_veo_lite"},
        ]}},
        "clips": {
            "shot1": {
                "type": "video",
                "path": str(video),
                "engine_id": "google_vid_sfx_veo_lite",
                "sfx_stem_path": str(sfx_tmp),
            }
        },
    }
    rd.persist_episode_clips(result, "ep_sfx")
    expected = dest / "shot1_music_visual_google_vid_sfx_veo_lite.sfx.wav"
    assert expected.is_file()
    assert result["clips"]["shot1"]["path"] == str(video)
    assert result["clips"]["shot1"]["sfx_stem_path"] == str(expected)


def test_persist_sfx_move_failure_clears_dangling_path(
    tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "out"))
    sfx_tmp = tmp_path / "stem.sfx.wav"
    sfx_tmp.write_bytes(b"sfx")

    def _fail_move(src, dst):
        raise OSError("blocked")

    monkeypatch.setattr(shutil, "move", _fail_move)
    result = {
        "ledger": {"video": {"shots": [
            {"shot_id": "shot1", "role": "music_visual",
             "engine_id": "google_vid_sfx_veo_lite"},
        ]}},
        "clips": {
            "shot1": {
                "type": "video",
                "path": "",
                "engine_id": "google_vid_sfx_veo_lite",
                "sfx_stem_path": str(sfx_tmp),
            }
        },
    }
    caplog.set_level(logging.WARNING, logger="OTR.video.render_driver")
    rd.persist_episode_clips(result, "ep_sfx")
    assert result["clips"]["shot1"]["sfx_stem_path"] == ""
    assert "SFX move FAILED" in caplog.text


def test_build_clip_manifest_carries_sfx_fields_only_for_sfx_clip():
    result = {
        "ledger": {"video": {
            "video_revision": 1,
            "fps": 25,
            "canonical_canvas": {"w": 1472, "h": 832},
            "shots": [
                {"shot_id": "shot_sfx", "source_line_ids": ["b001"],
                 "role": "music_visual", "engine_id": "google_vid_sfx_veo_lite",
                 "family": "text_to_video", "target_frame_count": 50,
                 "start_s": 0.0},
                {"shot_id": "shot_plain", "source_line_ids": ["b002"],
                 "role": "music_visual", "engine_id": "google_veo_video",
                 "family": "text_to_video", "target_frame_count": 50,
                 "start_s": 2.0},
            ]}},
        "clips": {
            "shot_sfx": {
                "engine_id": "google_vid_sfx_veo_lite",
                "family": "text_to_video",
                "frame_count": 50,
                "path": __file__,
                "sfx_stem_path": str(pathlib.Path(__file__)),
                "sfx_duration_s": 2.0,
                "sfx_sha256": "a" * 64,
            },
            "shot_plain": {
                "engine_id": "google_veo_video",
                "family": "text_to_video",
                "frame_count": 50,
                "path": __file__,
            },
        },
    }
    manifest = rd.build_clip_manifest(result, episode_id="ep_sfx")
    first, second = manifest["clips"]
    assert first["sfx_stem_path"] == str(pathlib.Path(__file__))
    assert first["sfx_duration_s"] == 2.0
    assert first["sfx_sha256"] == "a" * 64
    assert "sfx_stem_path" not in second
    assert "sfx_duration_s" not in second
    assert "sfx_sha256" not in second
