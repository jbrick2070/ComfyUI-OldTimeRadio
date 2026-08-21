"""BUG-LOCAL-413 guard tests: check_ltx_open_health surfaces a radio-open beat
that fell to the procgen/still floor instead of an LTX engine (the 6/15 silent
soft-open). CPU-only; pure manifest read."""
import pytest

from nodes._otr_video_engines import render_driver as rd


def _manifest(rows):
    return {"episode_id": "e", "clips": rows}


def _row(beat_id, role, engine_id, exists=True):
    return {"order": 0, "shot_id": "shot_" + beat_id, "beat_id": beat_id,
            "role": role, "engine_id": engine_id, "exists": exists}


def test_healthy_ltx_open_passes():
    m = _manifest([
        _row("b000_music_open", "music_visual", "ltx_video"),
        _row("b001", "announcer_visual", "ltx_video"),
        _row("b002", "character_video", "humo"),
    ])
    assert rd.check_ltx_open_health(m) == []


def test_ltx_av_open_also_healthy():
    m = _manifest([_row("b001", "announcer_visual", "ltx_audio_in"),
                   _row("b000_music_open", "music_visual", "ltx_audio_in")])
    assert rd.check_ltx_open_health(m) == []


def test_ltx25_HQ_open_also_healthy():
    m = _manifest([_row("b001", "announcer_visual", "ltx25_video"),
                   _row("b000_music_open", "music_visual", "ltx25_video")])
    assert rd.check_ltx_open_health(m, strict=True) == []


def test_procgen_fallback_open_flagged():
    # the 6/15 failure: the announcer/music open rendered on the still floor
    m = _manifest([
        _row("b000_music_open", "music_visual", "still_motion"),
        _row("b001", "announcer_visual", "abstract"),
        _row("b002", "character_video", "humo"),    # non-open: ignored
    ])
    bad = rd.check_ltx_open_health(m)
    assert len(bad) == 2
    assert {b["beat_id"] for b in bad} == {"b000_music_open", "b001"}
    # the non-open character beat is never flagged
    assert all(b["role"] != "character_video" for b in bad)


def test_clips_zero_open_flagged():
    # clips=0: the open beat exists=False (no LTX clip on disk at all)
    m = _manifest([_row("b001", "announcer_visual", "ltx_video", exists=False)])
    bad = rd.check_ltx_open_health(m)
    assert len(bad) == 1
    assert bad[0]["exists"] is False


def test_strict_mode_raises():
    m = _manifest([_row("b001", "announcer_visual", "still_motion")])
    with pytest.raises(rd.RenderFloorError):
        rd.check_ltx_open_health(m, strict=True)
    # non-strict only warns (returns the offenders, no raise)
    assert len(rd.check_ltx_open_health(m, strict=False)) == 1


def test_strict_env_flag(monkeypatch):
    monkeypatch.setenv("OTR_LTX_OPEN_STRICT", "1")
    m = _manifest([_row("b001", "announcer_visual", "abstract")])
    with pytest.raises(rd.RenderFloorError):
        rd.check_ltx_open_health(m)


def test_build_clip_manifest_nonstrict_does_not_raise(monkeypatch):
    # build_clip_manifest calls the guard; in the default (non-strict) mode a
    # procgen open must NOT raise (warn only) so production never aborts.
    monkeypatch.delenv("OTR_LTX_OPEN_STRICT", raising=False)
    result = {
        "ledger": {"video": {"shots": [
            {"shot_id": "s0", "role": "announcer_visual",
             "engine_id": "still_motion", "target_frame_count": 10}]},
            "lines": []},
        "clips": {"s0": {"path": "", "engine_id": "still_motion"}},
        "trace": [],
    }
    man = rd.build_clip_manifest(result, episode_id="e")   # must not raise
    assert man["n_beats"] == 1
    assert man["clips"][0]["role"] == "announcer_visual"
