"""OTR_CreditsRoll spec (credits enrichment 2026-07-03, GO_FORWARD v4).

This is the contract for the NEW late credits surface. It absorbs, as the
node's spec, the tests that the S3+S1 atomic rip retires:
- tests/test_hud_dossier_bug3.py::test_dossier_has_render_engines_block_video_and_image
- tests/test_hud_dossier_bug3.py::test_dossier_image_engines_from_meta_primary
- tests/test_hud_dossier_bug3.py::test_dossier_image_meta_takes_precedence_over_legacy
- tests/test_video_render_path_cw4.py::test_assemble_extends_to_floor_for_credits_tail_bug410
  (the looped-last-clip LOOK CONTRACT, now owned by plan_backdrop + the roll)

Cleanbreak deltas vs the old dossier behavior, by design:
- NO "(not recorded)" placeholders: a missing receipt RAISES CreditsDataError.
- meta.image_engines is the ONLY image source (S2 durable stamp); the legacy
  ledger['images'] row-scan fallback is NOT reproduced.
- Credits are DELIVERED-truth: cast voices come from CastLock's stamp
  (voice_ref_id/voice_preset + voice_engine), never meta.voice_cast_decision.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from nodes import otr_credits_roll as cr

HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG,
                                  reason="ffmpeg/ffprobe required")


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                   check=True, capture_output=True)


def _silent_video(path, dur=2.0, size="64x64"):
    _ff("-f", "lavfi", "-i", f"color=c=gray:s={size}:d={dur}", "-r", "25",
        "-pix_fmt", "yuv420p", "-t", f"{dur}", "-an", str(path))


def _count_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-count_packets", "-show_entries", "stream=nb_read_packets",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    return int(out.stdout.strip() or 0)


def _count_audio_streams(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True)
    return len([l for l in out.stdout.splitlines() if l.strip()])


def _led():
    """A durable ledger AFTER the S2 stamps have landed."""
    return {
        "meta": {
            "episode_seed": 42,
            "gen_params_initial": {
                "creative_writing_model": "openrouter:~anthropic/claude-opus-latest",
                "technical_model": "mistralai/Mistral-Nemo-Instruct-2407",
            },
            "news": {"script_brief": "UCLA lab integrity vs commercialization"},
            "image_engines": {"by_role": {
                "character_video": {"flux2_klein": 3},
                "music_visual": {"flux_gen1": 1}},
                "image_revision": 1},
            "render_engines": {"by_role": {
                "character_video": {"ltx_av": 3},
                "announcer_visual": {"still_flat": 2}},
                "histogram": {"ltx_av": 3, "still_flat": 2}},
            "music_engine": "stable_audio_music",
            "cast_voice_slots": {"c01": {"speech_signature": "clipped, dry"}},
        },
        "cast": [
            {"char_id": "c01", "name": "HAYES VANCE",
             "voice_ref_id": "vz_bill_boerst", "voice_engine": "indextts2"},
            {"char_id": "a1", "name": "ANNOUNCER",
             "voice_preset": "v2/en_speaker_6", "voice_engine": "bark"},
        ],
        "lines": [{"line_id": "b002"}],
    }


def _body(sections, header):
    return "\n".join(l for s in sections if s["header"] == header
                     for l in s["lines"])


def _headers(sections):
    return [s["header"] for s in sections]


# --------------------------------------------------------------------------- #
# Receipts blocks (relocated dossier spec)
# --------------------------------------------------------------------------- #
def test_roll_has_motion_and_images_blocks_video_and_image():
    sections = cr.build_credits_sections(_led())
    motion = _body(sections, "MOTION")
    assert "character_video" in motion and "ltx_av" in motion
    images = _body(sections, "IMAGES")
    assert "flux2_klein" in images               # image engine per role
    assert "music_visual" in images


def test_roll_image_engines_from_meta_is_the_only_source():
    """Relocated 'meta primary' + 'meta over legacy': cleanbreak makes meta
    the ONLY source -- a legacy ledger['images'] section is ignored, and a
    MISSING meta.image_engines RAISES (no quiet fallback scan)."""
    led = _led()
    led["images"] = {"images": [
        {"role": "character_video", "engine_id": "z_image_turbo"}]}
    sections = cr.build_credits_sections(led)
    images = _body(sections, "IMAGES")
    assert "flux2_klein" in images               # meta wins
    assert "z_image_turbo" not in images         # legacy rows never scanned

    del led["meta"]["image_engines"]
    with pytest.raises(cr.CreditsDataError):
        cr.build_credits_sections(led)


def test_roll_missing_receipts_raise_not_placeholder():
    for key in ("render_engines", "music_engine"):
        led = _led()
        del led["meta"][key]
        with pytest.raises(cr.CreditsDataError):
            cr.build_credits_sections(led)
    with pytest.raises(cr.CreditsDataError):
        cr.build_credits_sections({})
    with pytest.raises(cr.CreditsDataError):
        cr.build_credits_sections(None)


def test_roll_cast_voices_from_delivered_stamp_only():
    sections = cr.build_credits_sections(_led())
    cast = _body(sections, "CAST & VOICES")
    assert "HAYES VANCE" in cast
    assert "indextts2" in cast and "vz_bill_boerst" in cast
    assert "clipped, dry" in cast                # speech signature quoted
    assert "v2/en_speaker_6" in cast             # bark preset stamp honored

    # an unstamped character is a HARD failure (S2 guarantees the stamp)
    led = _led()
    del led["cast"][0]["voice_ref_id"]
    with pytest.raises(cr.CreditsDataError):
        cr.build_credits_sections(led)


def test_roll_planned_voice_decision_is_never_credited():
    """CastLock can fall closed PAST meta.voice_cast_decision.accepted_id --
    crediting it would print a voice that never spoke."""
    led = _led()
    led["meta"]["voice_cast_decision"] = {
        "c01": {"accepted_id": "planned_voice_never_used",
                "engine": "indextts2"}}
    sections = cr.build_credits_sections(led)
    assert "planned_voice_never_used" not in _body(sections, "CAST & VOICES")


def test_roll_music_and_news_and_seed_blocks():
    sections = cr.build_credits_sections(_led())
    assert "stable_audio_music" in _body(sections, "MUSIC")
    assert "UCLA lab integrity" in _body(sections, "NEWS SEED")
    seed = _body(sections, "SEED / COMMIT")
    assert "42" in seed
    assert len(seed.split("·")[-1].strip()) >= 7   # git short sha present


# --------------------------------------------------------------------------- #
# Duration -- the DECLARED credits tail (the credits-aware mux guard input)
# --------------------------------------------------------------------------- #
def test_roll_duration_is_content_driven():
    # full transit: canvas enters from the bottom, exits the top
    assert cr.compute_roll_duration_s(650, 350, scroll_pps=100.0) == \
        pytest.approx(10.0)
    # taller content -> longer declared tail (NEVER clamped "to be safe")
    short = cr.compute_roll_duration_s(500, 832)
    tall = cr.compute_roll_duration_s(5000, 832)
    assert tall > short
    with pytest.raises(cr.CreditsDataError):
        cr.compute_roll_duration_s(500, 832, scroll_pps=0)


def test_roll_canvas_grows_with_sections():
    a = cr.render_roll_canvas(cr.build_credits_sections(_led()), 640)
    led = _led()
    led["cast"] = led["cast"] * 10
    b = cr.render_roll_canvas(cr.build_credits_sections(led), 640)
    assert b.height > a.height


# --------------------------------------------------------------------------- #
# Backdrop -- the looped-last-clip LOOK CONTRACT (relocated BUG-410 owner)
# --------------------------------------------------------------------------- #
def test_backdrop_is_last_existing_clip_looped_never_black():
    manifest = {"fps": 25, "clips": [
        {"shot_id": "s0", "path": "a.mp4", "exists": True},
        {"shot_id": "s1", "path": "", "exists": False},
        {"shot_id": "s2", "path": "b.mp4", "exists": True},
    ]}
    row = cr.plan_backdrop(manifest)
    assert row["shot_id"] == "s2"                # LAST existing clip
    # positioned manifests pick by start_s, not list order
    manifest["clips"][0]["start_s"] = 30.0
    manifest["clips"][2]["start_s"] = 10.0
    assert cr.plan_backdrop(manifest)["shot_id"] == "s0"
    # NO usable clip -> RAISE (credits-over-black bounces at eyeball)
    with pytest.raises(cr.CreditsDataError):
        cr.plan_backdrop({"clips": [{"path": "", "exists": False}]})
    with pytest.raises(cr.CreditsDataError):
        cr.plan_backdrop({})


# --------------------------------------------------------------------------- #
# Render + append mechanics (ffmpeg; silent tail; no source-copy fallback)
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_credits_clip_renders_silent_and_matches_declared_duration(tmp_path):
    backdrop = tmp_path / "last_clip.mp4"
    _silent_video(backdrop, 1.0)                 # SHORT clip -> must loop
    out = tmp_path / "credits.mp4"
    sections = cr.build_credits_sections(_led())
    dur = cr.render_credits_clip(sections, str(backdrop), str(out),
                                 w=64, h=64, fps=25.0, scroll_pps=400.0)
    assert out.exists() and out.stat().st_size > 0
    assert _count_audio_streams(str(out)) == 0   # SILENT tail model
    frames = _count_frames(str(out))
    assert frames == pytest.approx(dur * 25.0, abs=3)
    # the 1s backdrop looped to cover the whole roll (no early cut)
    assert frames > 25 + 3


@needs_ffmpeg
def test_append_credits_extends_body_and_stays_silent(tmp_path):
    body = tmp_path / "body.mp4"
    backdrop = tmp_path / "clip.mp4"
    _silent_video(body, 2.0)
    _silent_video(backdrop, 1.0)
    credits = tmp_path / "credits.mp4"
    dur = cr.render_credits_clip(cr.build_credits_sections(_led()),
                                 str(backdrop), str(credits),
                                 w=64, h=64, fps=25.0, scroll_pps=400.0)
    out = tmp_path / "with_credits.mp4"
    cr.append_credits(str(body), str(credits), str(out))
    assert _count_audio_streams(str(out)) == 0
    total = _count_frames(str(out))
    assert total == pytest.approx(50 + dur * 25.0, abs=5)
    # video ENDS at credit-end: nothing after the roll
    assert total <= 50 + int(dur * 25.0) + 5


def test_append_credits_raises_never_source_copies(tmp_path):
    """The node-93 shutil.copy2 fallback is the exact behavior this node
    exists to refuse: a missing/failed input RAISES, never copies through."""
    with pytest.raises(cr.CreditsDataError):
        cr.append_credits(str(tmp_path / "missing_body.mp4"),
                          str(tmp_path / "missing_credits.mp4"),
                          str(tmp_path / "out.mp4"))
    src = (cr.__file__ and open(cr.__file__, encoding="utf-8").read()) or ""
    assert "copy2" not in src                    # no source-copy anywhere


# --------------------------------------------------------------------------- #
# Node surface (wired at S3; widget drift guard)
# --------------------------------------------------------------------------- #
def test_node_surface_two_force_inputs_no_widgets():
    it = cr.OTRCreditsRoll.INPUT_TYPES()
    req = it["required"]
    assert set(req) == {"video_path", "clip_manifest_json"}
    for spec in req.values():
        assert spec[1].get("forceInput") is True   # zero widgets_values rows
    assert cr.OTRCreditsRoll.RETURN_NAMES == (
        "video_with_credits_path", "declared_credits_tail_s", "report")
