"""OTR_CreditsRoll spec -- the 3-column SIGNAL LOST console (credits enrichment
redesign 2026-07-03, docs/2026-07-03-credits-enrichment/CREDITS_OVERLAY_BUILD_PLAN.md).

Cols 1-2 static dashboard (title / MODELS / [PRODUCTION LEDGER] / [SYSTEM] |
CAST & VOICES / [WRITER-LLM-CONFIG]); col-3 SCROLLS the full narrative (STORY
SPINE -> full CLASSIFIED TRANSCRIPT -> SOURCE INTERCEPT -> DIAGNOSTIC). No
fallbacks: a missing receipt RAISES; frozen story facts omit-if-absent; probe
fields ([SYSTEM]/VRAM) degrade.
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


def _silent_video(path, dur=2.0, size="256x144"):
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
    """A durable ledger AFTER the S2 stamps + the redesign stamps have landed."""
    return {
        "meta": {
            "episode_title": "Neon Truth",
            "style": "silent_scientific_protest",
            "episode_seed": 42,
            "cast_contract": {"cast_seed": 70303, "seed_source": "os-entropy"},
            "gen_params_initial": {
                "creative_writing_model": "mistralai/Mistral-Nemo-Instruct-2407",
                "technical_model": "mistralai/Mistral-Nemo-Instruct-2407",
                "creativity": "balanced", "temperature": 0.85, "top_p": 0.95,
                "target_words": 120, "seed_source": "os-entropy",
            },
            "slot_transitions": 0,
            "total_word_count": 83, "character_word_count": 37,
            "announcer_word_count": 46,
            "news": {"script_brief": "UCLA lab integrity vs commercialization"},
            "dramatic_state": {"dramatic_question": "Will Hayes expose the data?",
                               "character_a_wants": "protect the lab",
                               "character_b_wants": "ship the product",
                               "ending_change": "Hayes goes public"},
            "image_engines": {"by_role": {
                "announcer_visual": {"flux2_klein": 5},
                "character_video": {"flux2_klein": 5},
                "music_visual": {"flux2_klein": 2}}, "image_revision": 1},
            "render_engines": {
                "by_role": {"announcer_visual": {"humo": 2},
                            "music_visual": {"ltx_video": 1},
                            "character_video": {"wan_i2v": 3}},
                "by_engine": {
                    "humo": {"family": "audio_driven_face", "recipe": None,
                             "quant": None, "use_lora": False},
                    "ltx_video": {"family": "text_to_video", "recipe": "ia2v",
                                  "quant": "q4_K_M", "use_lora": True},
                    "wan_i2v": {"family": "image_to_video"}},
                "video_revision": 1, "vram_peak_mb": 8245.0},
            "music_engine": "stable_audio_3",
            "cast_voice_slots": {"c01": {"speech_signature": "warm · baritone"}},
        },
        "cast": [
            {"char_id": "a1", "name": "ANNOUNCER",
             "voice_preset": "bm_fable", "voice_engine": "kokoro"},
            {"char_id": "c01", "name": "KANE SIRIKIT",
             "voice_ref_id": "vz_bill_boerst", "voice_engine": "indextts2"},
            {"char_id": "c02", "name": "ALICE MALONE",
             "voice_ref_id": "vz_caro_davy", "voice_engine": "indextts2"},
        ],
        "lines": [
            {"line_id": "b001", "speaker_role": "announcer", "char_id": "a1",
             "text": "After midnight, on rain-kissed streets."},
            {"line_id": "b002", "speaker_role": "character", "char_id": "c02",
             "text": "Kane, the deadline's looming. July thirteenth."},
            {"line_id": "b003", "speaker_role": "character", "char_id": "c01",
             "text": "I'll have to expose my sources, Alice."},
        ],
    }


def _layout(**over):
    led = _led()
    return cr.build_credits_layout(led, w=1920, h=1080,
                                   manifest={"clips": [
                                       {"shot_id": "s0", "path": "a.mp4",
                                        "exists": True, "start_s": 0.0}],
                                       "total_target_frames": 400, "fps": 25,
                                       "clip_count": 3})


def _flat(blocks):
    return json.dumps(blocks, default=str, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# Hero / subtitle (title tweak)
# --------------------------------------------------------------------------- #
def test_hero_is_episode_title_subtitle_is_signal_lost():
    lay = _layout()
    assert lay["hero"] == "NEON TRUTH"                 # episode title = hero
    assert lay["subtitle"] == "SIGNAL LOST"            # 50% subtitle below
    assert "silent_scientific_protest" in lay["meta_strip"]
    assert "1920x1080" in lay["meta_strip"]


def test_missing_title_or_style_raises():
    for key in ("episode_title", "style"):
        led = _led()
        del led["meta"][key]
        with pytest.raises(cr.CreditsDataError):
            cr.build_credits_layout(led, w=1920, h=1080, manifest={})


# --------------------------------------------------------------------------- #
# COL 1 -- MODELS (video family suffix from the S-B stamp) + PRODUCTION LEDGER
# --------------------------------------------------------------------------- #
def test_models_block_video_family_and_image_and_music():
    lay = _layout()
    models = dict(lay["col1"])["models"]
    body = _flat(models)
    assert "flux2_klein" in body                       # image engine
    assert "humo" in body and "wan_i2v" in body        # video engines per role
    assert "audio_driven_face" in _flat(models["video_rows"]) or \
        "audio-driven face" in body                    # family label (S-B)
    assert "stable_audio_3" in body                    # music engine


def test_production_ledger_block_has_seed_commit_rev_vram():
    lay = _layout()
    grids = [b for k, b in lay["col1"] if k == "grid"]
    led_block = next(g for g in grids if "PRODUCTION LEDGER" in g["header"])
    keys = " ".join(r[0] for r in led_block["rows"])
    assert "SEED:" in keys and "COMMIT:" in keys and "REV:" in keys
    assert "VRAM:" in keys
    body = _flat(led_block["rows"])
    assert "70303" in body                             # cast seed
    assert "8.1 GiB" in body or "8.0 GiB" in body      # vram peak formatted


def test_missing_receipts_raise_not_placeholder():
    for key in ("render_engines", "image_engines", "music_engine"):
        led = _led()
        del led["meta"][key]
        with pytest.raises(cr.CreditsDataError):
            cr.build_credits_layout(led, w=1920, h=1080, manifest={})
    led = _led()
    del led["meta"]["cast_contract"]
    led["meta"].pop("episode_seed")
    with pytest.raises(cr.CreditsDataError):
        cr.build_credits_layout(led, w=1920, h=1080, manifest={})
    with pytest.raises(cr.CreditsDataError):
        cr.build_credits_layout({}, w=1920, h=1080, manifest={})


# --------------------------------------------------------------------------- #
# COL 2 -- cast (delivered stamp only) + writer config
# --------------------------------------------------------------------------- #
def test_cast_voices_from_delivered_stamp_with_signature():
    lay = _layout()
    body = _flat(lay["col2"]["cast_rows"])
    assert "KANE SIRIKIT" in body and "indextts2" in body
    assert "vz_bill_boerst" in body
    assert "warm · baritone" in body                   # speech signature quoted
    assert "bm_fable" in body and "kokoro" in body     # bark/kokoro preset stamp


def test_unstamped_cast_voice_raises():
    led = _led()
    del led["cast"][1]["voice_ref_id"]
    with pytest.raises(cr.CreditsDataError):
        cr.build_credits_layout(led, w=1920, h=1080, manifest={})


def test_planned_voice_decision_is_never_credited():
    led = _led()
    led["meta"]["voice_cast_decision"] = {
        "c01": {"accepted_id": "planned_voice_never_used"}}
    lay = cr.build_credits_layout(led, w=1920, h=1080, manifest={"clips": []})
    assert "planned_voice_never_used" not in _flat(lay["col2"]["cast_rows"])


def test_writer_config_block():
    lay = _layout()
    body = _flat(lay["col2"]["writer_grid"])
    assert "Mistral-Nemo-Instruct-2407" in body
    assert "0.85" in body and "0.95" in body
    assert "target 120" in body and "actual 83" in body


# --------------------------------------------------------------------------- #
# COL 3 -- scroll flow: spine + FULL transcript + intercept + diagnostic
# --------------------------------------------------------------------------- #
def test_col3_flow_system_spine_full_transcript_intercept_diagnostic():
    lay = _layout()
    kinds = [k for k, _ in lay["col3_flow"]]
    # SYSTEM now LEADS the scroll (operator/R1: all the same details in the scroll)
    assert kinds == ["system", "spine", "transcript", "intercept", "diagnostic"]
    system = dict(lay["col3_flow"])["system"]
    assert any("CPU" in r[0] for r in system["rows"])   # SYSTEM in the scroll
    assert any("GPU" in r[0] for r in system["rows"])
    spine = dict(lay["col3_flow"])["spine"]
    assert "protect the lab" in _flat(spine) and "ship the product" in _flat(spine)
    transcript = dict(lay["col3_flow"])["transcript"]
    # FULL transcript -- every dialogue line present, nothing dropped
    assert len(transcript["lines"]) == 3
    assert "expose my sources" in _flat(transcript["lines"])


def test_system_left_col1_static_dashboard():
    lay = _layout()
    # SYSTEM must NOT be duplicated in the static col 1 anymore.
    col1_headers = [b.get("header", "") for k, b in lay["col1"] if k == "grid"]
    assert not any("SYSTEM" in h for h in col1_headers)
    assert any("PRODUCTION LEDGER" in h for h in col1_headers)   # ledger stays


def test_system_block_carries_full_detail_and_correct_sysd_keys(monkeypatch):
    """kibitz R3: the scroll SYSTEM block must carry ALL the detail (host / cpu +
    cores / ram + peak / gpu + vram / cuda / torch / python) and read the RIGHT
    collect_system_specs keys (hostname, vram, cpu_cores, ram_peak) -- a
    regression back to host / gpu_vram or dropped fields fails here."""
    monkeypatch.setattr(cr, "_sys_specs", lambda: {
        "hostname": "SENTINEL_HOST", "os": "SENTINEL_OS",
        "cpu": "SENTINEL_CPU", "cpu_cores": "SENTINEL_CORES",
        "ram": "SENTINEL_RAM", "ram_peak": "SENTINEL_PEAK",
        "gpu": "SENTINEL_GPU", "vram": "SENTINEL_VRAM",
        "cuda": "13.0", "torch": "2.10", "python": "3.12"})
    lay = _layout()
    sysblock = _flat(dict(lay["col3_flow"])["system"])
    for s in ("SENTINEL_HOST", "SENTINEL_CPU", "SENTINEL_CORES", "SENTINEL_RAM",
              "SENTINEL_PEAK", "SENTINEL_GPU", "SENTINEL_VRAM"):
        assert s in sysblock, s
    # the PRODUCTION LEDGER VRAM total reads the "vram" key (not "gpu_vram")
    grids = [b for k, b in lay["col1"] if k == "grid"]
    led_block = next(g for g in grids if "PRODUCTION LEDGER" in g["header"])
    assert "SENTINEL_VRAM" in _flat(led_block["rows"])


def test_diagnostic_is_seeded_and_never_fabricates_a_number():
    lay = _layout()
    diag = dict(lay["col3_flow"])["diagnostic"]["text"]
    assert diag.startswith(">> DIAGNOSTIC")
    assert "???" not in diag                            # no fabricated value
    # deterministic per cast_seed
    lay2 = _layout()
    assert dict(lay2["col3_flow"])["diagnostic"]["text"] == diag


def test_news_intercept_omitted_when_absent():
    led = _led()
    led["meta"]["news"] = None
    lay = cr.build_credits_layout(led, w=1920, h=1080, manifest={"clips": []})
    kinds = [k for k, _ in lay["col3_flow"]]
    assert "intercept" not in kinds                     # omit, no placeholder
    assert "transcript" in kinds and "diagnostic" in kinds


# --------------------------------------------------------------------------- #
# Duration -- col-3 scroll drives the declared tail (never truncates)
# --------------------------------------------------------------------------- #
def test_duration_is_scroll_driven_with_speedup_and_static():
    # a tall scroll -> longer tail
    short = cr.compute_credits_duration_s(200, 900)[0]
    tall = cr.compute_credits_duration_s(6000, 900)[0]
    assert tall > short
    # ceiling caps the DURATION by speeding pps up -- never truncates content
    dur, pps = cr.compute_credits_duration_s(100000, 900)
    assert dur <= cr._MAX_HOLD_S + 0.01 and pps > cr._SCROLL_PPS
    # nothing to scroll -> a static readable hold
    assert cr.compute_credits_duration_s(0, 900)[0] > 0
    with pytest.raises(cr.CreditsDataError):
        cr.compute_credits_duration_s(500, 900, pps=0)


# --------------------------------------------------------------------------- #
# Backdrop -- looped last clip (look contract)
# --------------------------------------------------------------------------- #
def test_backdrop_is_last_existing_clip_never_black():
    manifest = {"clips": [
        {"shot_id": "s0", "path": "a.mp4", "exists": True},
        {"shot_id": "s2", "path": "b.mp4", "exists": True}]}
    assert cr.plan_backdrop(manifest)["shot_id"] == "s2"
    with pytest.raises(cr.CreditsDataError):
        cr.plan_backdrop({"clips": [{"path": "", "exists": False}]})
    with pytest.raises(cr.CreditsDataError):
        cr.plan_backdrop({})


def test_backdrop_excludes_directory_clips(tmp_path):
    """kibitz R3: a 3D frame-DIRECTORY clip must NOT be chosen as the loop
    backdrop (ffmpeg -stream_loop -i <dir> crashes). Pick the file instead;
    raise if the only clip is a directory."""
    d = tmp_path / "frames_dir"
    d.mkdir()
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")
    manifest = {"clips": [
        {"shot_id": "s0", "path": str(f), "exists": True},
        {"shot_id": "s1", "path": str(d), "exists": True}]}   # dir is "last"
    assert cr.plan_backdrop(manifest)["shot_id"] == "s0"       # the file, not the dir
    with pytest.raises(cr.CreditsDataError):
        cr.plan_backdrop({"clips": [
            {"shot_id": "s1", "path": str(d), "exists": True}]})


# --------------------------------------------------------------------------- #
# PIL renderers (no ffmpeg needed -- PIL/numpy only)
# --------------------------------------------------------------------------- #
def test_static_base_renders_full_frame():
    img = cr.render_static_base(_layout(), 1920, 1080)
    assert img.size == (1920, 1080)


def test_scroll_canvas_grows_with_transcript():
    lay = _layout()
    a = cr.render_scroll_canvas(lay["col3_flow"], 600, 1080)
    led = _led()
    led["lines"] = led["lines"] * 12                   # much longer script
    lay2 = cr.build_credits_layout(led, w=1920, h=1080, manifest={"clips": [
        {"shot_id": "s0", "path": "a.mp4", "exists": True}]})
    b = cr.render_scroll_canvas(lay2["col3_flow"], 600, 1080)
    assert b.height > a.height                          # nothing dropped -> taller


# --------------------------------------------------------------------------- #
# ffmpeg render + append (silent tail, no source-copy)
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_console_clip_renders_silent_and_declares_duration(tmp_path):
    backdrop = tmp_path / "clip.mp4"
    _silent_video(backdrop, 1.0)                        # short -> must loop
    out = tmp_path / "credits.mp4"
    dur = cr.render_credits_clip(_layout(), str(backdrop), str(out),
                                 w=1280, h=720, fps=25.0)
    assert out.exists() and out.stat().st_size > 0
    assert _count_audio_streams(str(out)) == 0         # silent tail
    assert _count_frames(str(out)) == pytest.approx(dur * 25.0, abs=6)


@needs_ffmpeg
def test_append_extends_body_and_stays_silent(tmp_path):
    body = tmp_path / "body.mp4"
    backdrop = tmp_path / "clip.mp4"
    _silent_video(body, 2.0, size="1280x720")
    _silent_video(backdrop, 1.0, size="1280x720")
    credits = tmp_path / "credits.mp4"
    dur = cr.render_credits_clip(_layout(), str(backdrop), str(credits),
                                 w=1280, h=720, fps=25.0)
    out = tmp_path / "with_credits.mp4"
    cr.append_credits(str(body), str(credits), str(out))
    assert _count_audio_streams(str(out)) == 0
    total = _count_frames(str(out))
    assert total == pytest.approx(50 + dur * 25.0, abs=8)


@needs_ffmpeg
def test_col3_text_scrolls_even_for_short_episode(tmp_path):
    """The real-obs bug: col 3 was STATIC + clipped on a short episode. Classic
    roll must ALWAYS scroll. Render over a CONSTANT gray backdrop so any change in
    the col-3 region is TEXT motion (not the looped backdrop that masked the bug),
    and assert two in-scroll times differ -- for the SHORT fixture transcript."""
    import hashlib
    from PIL import Image
    backdrop = tmp_path / "clip.mp4"
    _silent_video(backdrop, 1.0, size="1280x720")      # constant gray
    out = tmp_path / "credits.mp4"
    dur = cr.render_credits_clip(_layout(), str(backdrop), str(out),
                                 w=1280, h=720, fps=25.0)

    def col3_hash(ts):
        f = tmp_path / ("f_%.2f.png" % ts)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", "%.2f" % ts,
                        "-i", str(out), "-frames:v", "1", str(f)], check=True)
        im = Image.open(f).convert("RGB")
        x = cr._sc(cr._COL3_X, 720)
        reg = im.crop((x, cr._sc(cr._COL3_VIEW_Y, 720), im.width, 720 - 40))
        return hashlib.sha1(reg.tobytes()).hexdigest()

    t_early = cr._LEAD_HOLD_S + 2.0
    t_mid = min(dur - cr._TAIL_HOLD_S - 2.0, t_early + 6.0)
    assert col3_hash(t_early) != col3_hash(t_mid), \
        "col 3 text did not scroll (static-crop / overflow-only regression)"


def test_classic_roll_distance_includes_viewport():
    """Guard the model regression: the roll distance is content_h + view_h (the
    canvas is padded view_h top AND bottom) so col 3 ALWAYS rolls and nothing is
    clipped -- NOT the old overflow-only content_h - view_h that left short
    episodes static. A tiny canvas still yields a roll_px >= view_h."""
    view_h = 600
    # content 100px tall -> padded 100+1200 -> roll_px = 1300 - 600 = 700 > view_h
    assert cr.compute_credits_duration_s(100 + view_h, view_h)[0] > \
        cr._LEAD_HOLD_S + cr._TAIL_HOLD_S
    # duration scales with content (longer transcript -> longer roll)
    assert cr.compute_credits_duration_s(5000, view_h)[0] > \
        cr.compute_credits_duration_s(500, view_h)[0]


def test_append_raises_never_source_copies(tmp_path):
    with pytest.raises(cr.CreditsDataError):
        cr.append_credits(str(tmp_path / "missing_body.mp4"),
                          str(tmp_path / "missing_credits.mp4"),
                          str(tmp_path / "out.mp4"))
    src = open(cr.__file__, encoding="utf-8").read()
    assert "copy2" not in src


# --------------------------------------------------------------------------- #
# Node surface (widget-drift guard)
# --------------------------------------------------------------------------- #
def test_node_surface_two_force_inputs_no_widgets():
    it = cr.OTRCreditsRoll.INPUT_TYPES()
    req = it["required"]
    assert set(req) == {"video_path", "clip_manifest_json"}
    for spec in req.values():
        assert spec[1].get("forceInput") is True
    assert cr.OTRCreditsRoll.RETURN_NAMES == (
        "video_with_credits_path", "declared_credits_tail_s", "report")
