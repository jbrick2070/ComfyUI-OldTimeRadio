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
import os
import shutil
import logging
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
            "visual_style": "sci_fi_radio",
            "source_bank": "scifi_news_pro",
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
                # DELIBERATELY the PRE-2026-07-28 by_engine shape: no
                # ``varied``, no ``clip_count``, and the wan_i2v row carrying
                # only ``family``. _build_render_engines_payload can no longer
                # emit this, and that is the point -- it pins that the credits
                # readers still read an episode stamped before the per-field
                # roll-up landed. Do not "modernise" it.
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
    # The strip shows BANK and LOOK (operator ruling 2026-08-03), never the
    # story scaffold -- which lied on screen twice in one day
    # ('asteroid_mining_labor_dispute' over a cartographer's-guild tale,
    # 'pirate_radio_resistance_drama' over a film-reel story).
    assert "scifi_news_pro" in lay["meta_strip"]
    assert "sci_fi_radio" in lay["meta_strip"]
    assert "silent_scientific_protest" not in lay["meta_strip"]
    assert "1920x1080" in lay["meta_strip"]


def test_missing_title_or_identity_raises():
    # The no-fallback contract, updated for the display swap: the strip's
    # inputs are episode_title, visual_style and source_bank. meta.style (the
    # scaffold) is deliberately NOT required -- a scaffold-off episode has
    # none, by the bank's own definition.
    for key in ("episode_title", "visual_style", "source_bank"):
        led = _led()
        del led["meta"][key]
        with pytest.raises(cr.CreditsDataError):
            cr.build_credits_layout(led, w=1920, h=1080, manifest={})


def test_scaffold_absence_never_reaches_the_strip():
    """INVERTED 2026-08-03 (operator: the scaffold does not belong in the
    credits). The previous pin demanded the strip show the
    'story_scaffold_off' status token and REFUSE visual_style. The ruling is
    the opposite: the strip always shows the LOOK, and neither the scaffold
    slug nor its status token is display content. The status pair remains a
    ledger receipt (the old _story_style_receipt helper was deleted with the
    swap -- zero callers)."""
    led = _led()
    del led["meta"]["style"]
    led["meta"]["visual_style"] = "recur_frac"
    led["meta"]["story_scaffold_enabled"] = False
    led["meta"]["story_style_status"] = "story_scaffold_off"
    lay = cr.build_credits_layout(led, w=1920, h=1080, manifest={"clips": []})
    assert "recur_frac" in lay["meta_strip"]
    assert "story_scaffold_off" not in lay["meta_strip"]
    # And a scaffold-ON ledger's slug still never leaks into the strip.
    led2 = _led()
    led2["meta"]["visual_style"] = "recur_frac"
    lay2 = cr.build_credits_layout(led2, w=1920, h=1080, manifest={"clips": []})
    assert "silent_scientific_protest" not in lay2["meta_strip"]


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
    # 2026-08-14: was `"target 120" and "actual 83"`. There is no requested
    # length any more, so the row carries the OBSERVED counts alone. The row
    # must keep RENDERING -- it briefly stopped entirely, because it was
    # gated on a target_words that is never written now.
    assert "83 (char 37 / ann 46)" in body


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
    # CONTENT PASS 2026-08-03: the dramatic_state rows (Question / A wants /
    # B wants / Ending) are OUT of the scroll -- they are derived BEFORE any
    # dialogue exists and the tempests_chart specimen scrolled them naming
    # characters the delivered episode replaced. The spine now carries only
    # story-derived fields (premise from the produced logline; brief / arc /
    # palette / atmosphere when the reflection stamped them).
    assert "protect the lab" not in _flat(spine)
    assert "ship the product" not in _flat(spine)
    assert any("Premise:" in r[0] for r in spine["rows"])
    transcript = dict(lay["col3_flow"])["transcript"]
    # FULL transcript -- every dialogue line present, nothing dropped
    assert len(transcript["lines"]) == 3
    assert "expose my sources" in _flat(transcript["lines"])


def test_meta_strip_is_clamped_with_a_marked_cut_at_narrow_canvases():
    """QA 2026-08-03 (measured with the real font): the source_bank prefix
    pushed the strip past its column at 720p/480p worst-case pairs. The strip
    now goes through _fit_text, whose cut is always MARKED with an ellipsis --
    a value silently shortened to fit reads as the whole value."""
    from PIL import Image, ImageDraw
    d = ImageDraw.Draw(Image.new("RGB", (64, 64)))
    font = cr._load_font(18)
    long = "media_archive · shakespeare_stage_realism · 1280x720 · 2026-08-03"
    fitted = cr._fit_text(d, long, font, 200)
    assert fitted.endswith("...")
    assert cr._fw(d, fitted, font) <= 200
    short = cr._fit_text(d, "original · anime", font, 10_000)
    assert short == "original · anime"          # untouched when it fits
    assert cr._fit_text(d, long, font, 0) == ""  # degenerate budget


def test_spine_carries_the_story_derived_fields_when_stamped():
    """The reflection's fields -- live-verified truthful on the 2026-08-03
    specimens -- are the spine's content now: brief, arc (underscores
    humanized), palette and atmosphere. All optional; absent fields add no
    row."""
    led = _led()
    led["meta"]["story_brief"] = "A standoff over lab data in a midnight tower"
    led["meta"]["arc_shape"] = "investigation_without_answer"
    led["meta"]["visual_palette"] = ["glass", "rain", "monitors"]
    led["meta"]["story_brief_terms"] = {
        "atmosphere": ["hum", "static"], "lighting": ["sodium"]}
    lay = cr.build_credits_layout(led, w=1920, h=1080,
                                  manifest={"clips": []})
    spine = dict(lay["col3_flow"])["spine"]
    flat = _flat(spine)
    assert "A standoff over lab data" in flat
    assert "investigation without answer" in flat      # humanized, no underscores
    assert "glass" in flat and "rain" in flat
    assert "hum" in flat and "sodium" in flat


def test_intercept_prefers_key_terms_over_stale_brief():
    """The intercept scrolls the source's true atoms when key_terms exist;
    the pre-generation script_brief remains only as the legacy fallback."""
    led = _led()
    led["meta"]["news"]["key_terms"] = ["lighthouse", "map", "storm"]
    lay = cr.build_credits_layout(led, w=1920, h=1080, manifest={"clips": []})
    texts = [b.get("text", "") for k, b in lay["col3_flow"] if k == "intercept"]
    assert any("lighthouse" in t and "storm" in t for t in texts)
    assert not any("UCLA lab integrity" in t for t in texts)


def test_transcript_voice_resolves_announcer_role_tag_alias():
    led = _led()
    # Content-owned lanes preserve the role tag on the line while the cast row
    # may use a canonical slot id. Display name and delivered voice must join
    # through the same alias-aware cast authority.
    led["cast"][0]["char_id"] = "host01"
    led["lines"][0]["char_id"] = "announcer"
    lay = cr.build_credits_layout(
        led, w=1920, h=1080,
        manifest={"clips": [{"shot_id": "s0", "path": "a.mp4",
                             "exists": True, "start_s": 0.0}]},
    )
    transcript = dict(lay["col3_flow"])["transcript"]["lines"]
    assert transcript[0]["speaker"] == "ANNOUNCER"
    assert transcript[0]["voice"] == "bm_fable"


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
# Backdrop -- the BODY VIDEO's frozen final frame (WIRE-W6, 2026-07-29)
# --------------------------------------------------------------------------- #
def _backdrop_png(path, size=(1280, 720)):
    """A still standing in for the body video's extracted final frame."""
    from PIL import Image
    Image.new("RGB", size, (40, 44, 52)).save(str(path))
    return path


def test_plan_backdrop_is_GONE_not_merely_unused():
    """It hunted the clip manifest for a loopable FILE clip and raised when it
    found none -- which is why an all-``mesh_stage`` episode (frame
    DIRECTORIES, no mp4) could render 7 of 7 shots and then be refused by the
    terminal node of its own graph.

    The backdrop comes from the body video now, so the manifest search has no
    remaining caller. Leaving it importable would leave a SECOND backdrop
    authority in the file for someone to wire back up -- the shape this build
    has paid for repeatedly. Deleted, and asserted deleted.
    """
    assert not hasattr(cr, "plan_backdrop")
    assert "plan_backdrop" not in cr.__all__


@needs_ffmpeg
def test_the_backdrop_comes_from_the_body_video_itself(tmp_path):
    body = tmp_path / "body.mp4"
    _silent_video(body, 2.0, size="1280x720")
    png = tmp_path / "bd.png"
    cr.extract_final_frame(str(body), str(png))
    assert png.exists() and png.stat().st_size > 0
    from PIL import Image
    assert Image.open(png).size == (1280, 720)


@needs_ffmpeg
def test_a_body_of_only_a_few_frames_still_yields_a_backdrop(tmp_path):
    """`-sseof -3` finds nothing to decode on a body shorter than the seek, so
    the extractor falls back to frame 0. A short episode must not lose its
    credits over an ffmpeg seek that had nothing to seek to."""
    body = tmp_path / "tiny.mp4"
    _silent_video(body, 0.2, size="640x360")
    png = tmp_path / "bd.png"
    cr.extract_final_frame(str(body), str(png))
    assert png.exists() and png.stat().st_size > 0


def test_an_unreadable_body_is_TERMINAL_at_the_extractor(tmp_path):
    """A body whose frame cannot be read is not a presentation problem -- there
    is no picture to make and no episode to hand back either."""
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"this is not an mp4")
    with pytest.raises(cr.CreditsDataError, match="backdrop frame"):
        cr.extract_final_frame(str(bad), str(tmp_path / "bd.png"))


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
    backdrop = _backdrop_png(tmp_path / "bd.png")       # a HELD frame now
    out = tmp_path / "credits.mp4"
    dur = cr.render_credits_clip(_layout(), str(backdrop), str(out),
                                 w=1280, h=720, fps=25.0)
    assert out.exists() and out.stat().st_size > 0
    assert _count_audio_streams(str(out)) == 0         # silent tail
    assert _count_frames(str(out)) == pytest.approx(dur * 25.0, abs=6)


@needs_ffmpeg
def test_append_extends_body_and_stays_silent(tmp_path):
    body = tmp_path / "body.mp4"
    _silent_video(body, 2.0, size="1280x720")
    backdrop = _backdrop_png(tmp_path / "bd.png")
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
    backdrop = _backdrop_png(tmp_path / "bd.png")      # constant flat colour
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


def test_scroll_still_inputs_are_looped_timed(monkeypatch, tmp_path):
    """Regression (shipped-obs bug 2026-07-04): col 3 rendered BLANK the whole roll
    because base_png/scroll_png were fed as plain single-frame `-i <png>` inputs.
    A single still is one frame at t=0, so the col-3 crop y-expr (which scrolls on
    `t`) was frozen at y=0 = the blank top pad. They MUST be LOOPED, fps-timed image
    inputs so `t` advances. Capture the render argv and assert `-loop 1 -framerate`
    precedes each still input -- a guard the fade-masked motion test above missed."""
    backdrop = _backdrop_png(tmp_path / "bd.png")
    seen = {}
    real_run = cr.subprocess.run

    def _spy(cmd, *a, **k):
        if isinstance(cmd, (list, tuple)) and any(
                isinstance(c, str) and c.endswith(".scroll.png") for c in cmd):
            seen["cmd"] = [str(c) for c in cmd]
            with open(cmd[-1], "wb") as fh:            # satisfy the size>0 check
                fh.write(b"\x00" * 32)

            class _R:
                returncode = 0
                stderr = ""
            return _R()
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(cr.subprocess, "run", _spy)
    cr.render_credits_clip(_layout(), str(backdrop), str(tmp_path / "credits.mp4"),
                           w=1280, h=720, fps=25.0)
    cmd = seen["cmd"]
    for suffix in (".base.png", ".scroll.png"):
        idxs = [j for j, c in enumerate(cmd) if c.endswith(suffix)]
        assert idxs, "no %s input in the ffmpeg argv" % suffix
        j = idxs[0]
        assert cmd[j - 1] == "-i", "%s not an -i input" % suffix
        window = cmd[max(0, j - 6):j]
        assert "-loop" in window and "-framerate" in window, (
            "%s is not looped/fps-timed -> the col-3 scroll would FREEZE "
            "(single frame at t=0). argv window=%r" % (suffix, window))


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


# --------------------------------------------------------------------------- #
# MODELS.VIDEO -- the recipe reaches the CARD, and the row is CLAMPED
# (2026-07-28). `video_suffix` had ONE write and ZERO readers since the S-E5
# stamp: the durable ledger knew what rendered each beat and the credits sheet
# did not. And `_row` right-aligned by subtracting pixel width with no bound,
# so a ~90-character LANE 2 receipt would have begun LEFT of its own label.
#
# These drive the REAL _draw_models with a REAL font onto a REAL canvas and
# record the coordinates it actually draws at. Nothing here asserts a value the
# fixture handed in.
# --------------------------------------------------------------------------- #
_LANE2 = ("RECIPE_LTX8_I2V_v2+prequalification[tiled_vae=off] "
          "· Q8_0 · 512x288")


def _spy_models(models, w=1920, h=1080):
    """Render the MODELS block and return (draw_calls, end_y, draw, image).

    Each call is (x, y, text, font). The draw object is real -- textlength,
    font metrics and the canvas are all the production ones."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    d = ImageDraw.Draw(img)
    calls = []
    real_text = d.text

    def spy(xy, text, *a, **k):
        calls.append((int(xy[0]), int(xy[1]), text, k.get("font")))
        return real_text(xy, text, *a, **k)

    d.text = spy
    end_y = cr._draw_models(d, int(cr._COL1_X * h / cr._REF_H), 48, models,
                            w, h)
    return calls, end_y, d, img


def _models_block(video_suffix=None, video_rows=None):
    return {"header": "MODELS", "tag": "GENERATIVE STACK", "img_rev": 1,
            "vid_rev": 3, "image_rows": [("stills", "flux2_klein x5")],
            "video_rows": video_rows or [("music_visual", "ltx_8gb",
                                          "image-to-video")],
            "video_suffix": video_suffix or {},
            "music": "stable_audio_3"}


def test_the_card_draws_the_recipe_it_had_been_carrying_unread():
    """THE DEFECT: one write, zero readers. Proven by pixels, not by reading
    the source -- the same block with and without a recipe must not render to
    the same image."""
    without = _spy_models(_models_block())[3]
    with_it = _spy_models(_models_block({"ltx_8gb": _LANE2}))[3]
    assert without.tobytes() != with_it.tobytes()


def test_rendering_the_same_block_twice_is_identical():
    """CONTROL for the test above -- if the renderer were nondeterministic,
    an image inequality would prove nothing."""
    a = _spy_models(_models_block({"ltx_8gb": _LANE2}))[3]
    b = _spy_models(_models_block({"ltx_8gb": _LANE2}))[3]
    assert a.tobytes() == b.tobytes()


def test_the_recipe_text_is_actually_drawn_and_says_the_recipe():
    calls, _end, _d, _img = _spy_models(_models_block({"ltx_8gb": _LANE2}))
    drawn = " ".join(c[2] for c in calls)
    assert "RECIPE_LTX8_I2V_v2" in drawn
    assert "tiled_vae=off" in drawn          # the DEPARTURE, not just the name


def test_an_engine_with_no_recipe_adds_no_line():
    """A still_* / humo row stamps no recipe, and an empty note must not open
    a blank line under it."""
    tall = _spy_models(_models_block({"ltx_8gb": _LANE2}))[1]
    short = _spy_models(_models_block({"ltx_8gb": ""}))[1]
    bare = _spy_models(_models_block())[1]
    assert short == bare < tall


def test_a_long_value_never_starts_left_of_its_own_label():
    """THE RIDER, measured at the pixels. Before the clamp, vx was
    x + colw - width with no bound."""
    from PIL import Image, ImageDraw
    h = 1080
    long_engine = "ltx_8gb_" + ("x" * 120)
    calls, _end, d, _img = _spy_models(
        _models_block(video_rows=[("music_visual", long_engine, "")]))
    fbody = cr._load_font(cr._sc(cr._PT_BODY, h))
    label = next(c for c in calls if c[2] == "music_visual")
    value = next(c for c in calls if c[2].startswith("ltx_8gb_x"))
    assert value[0] >= label[0] + cr._fw(d, "music_visual", fbody)
    assert value[2].endswith("...")          # the cut is MARKED, not silent


def test_a_clamped_row_stays_inside_its_column():
    """BOTH bounds. The right edge alone is a TAUTOLOGY of the positioning
    formula (vx is DEFINED as x + colw - width, so value_x + width == x + colw
    for any string, clamped or not) -- the pre-push fan-out caught this test
    asserting only that. The LEFT bound is the one the old code violated: it
    put vx at -754 on a 120-character engine id."""
    h = 1080
    long_engine = "ltx_8gb_" + ("x" * 120)
    calls, _end, d, _img = _spy_models(
        _models_block(video_rows=[("music_visual", long_engine, "")]))
    x0 = int(cr._COL1_X * h / cr._REF_H)
    colw = int(cr._COL1_W * h / cr._REF_H)
    fbody = cr._load_font(cr._sc(cr._PT_BODY, h))
    value = next(c for c in calls if c[2].startswith("ltx_8gb_x"))
    assert value[0] >= x0                                   # <- the real bound
    assert value[0] + cr._fw(d, value[2], fbody) <= x0 + colw + 1


def test_a_long_family_suffix_gives_way_before_the_engine_id():
    """The clamp's first branch: an engine id that cannot be read is worse
    than a missing family annotation, so the annotation is trimmed first."""
    calls, _end, _d, _img = _spy_models(_models_block(
        video_rows=[("music_visual", "wan_i2v", "family " * 30)]))
    engine = [c for c in calls if c[2] == "wan_i2v"]
    assert engine, "the engine id was trimmed while the annotation survived"
    suffix = [c for c in calls if c[2].startswith("· family")]
    assert suffix and suffix[0][2].endswith("...")


def test_a_row_that_clamps_and_carries_a_recipe_does_both():
    """The two mechanisms meet on one row; each was only ever tested alone."""
    long_engine = "ltx_8gb_" + ("x" * 120)
    calls, _end, _d, _img = _spy_models(_models_block(
        {long_engine: _LANE2},
        video_rows=[("music_visual", long_engine, "")]))
    drawn = [c[2] for c in calls]
    assert any(t.startswith("ltx_8gb_x") and t.endswith("...") for t in drawn)
    assert any("RECIPE_LTX8_I2V_v2" in t for t in drawn)


def test_only_the_roles_that_carry_a_receipt_get_a_note():
    """A real ledger mixes engines that stamp a recipe with engines that do
    not (still_* and humo stamp none) in the SAME render."""
    calls, _end, _d, _img = _spy_models(_models_block(
        {"ltx_8gb": _LANE2},
        video_rows=[("announcer_visual", "humo", "audio-driven face"),
                    ("music_visual", "ltx_8gb", "image-to-video")]))
    ys = {c[2]: c[1] for c in calls}
    note = [c for c in calls if "RECIPE_LTX8_I2V_v2" in c[2]]
    assert len(note) == 1
    assert note[0][1] > ys["ltx_8gb"]          # BELOW its own engine row
    assert note[0][1] > ys["humo"]             # ...and not under humo's


def test_an_unbreakable_receipt_token_is_cut_not_run_off_the_edge():
    """_wrap splits on whitespace and cannot break one long token -- and a
    departure list is exactly one long token."""
    h = 1080
    token = "RECIPE_LTX8_I2V_v2+prequalification[" + ("k=v," * 40) + "]"
    calls, _end, d, _img = _spy_models(_models_block({"ltx_8gb": token}))
    fmicro = cr._load_font(cr._sc(cr._PT_MICRO, h))
    x0 = int(cr._COL1_X * h / cr._REF_H)
    colw = int(cr._COL1_W * h / cr._REF_H)
    note = [c for c in calls if c[2].startswith("RECIPE_LTX8_I2V_v2+prequal")]
    assert note, "the receipt line was dropped entirely"
    for c in note:
        assert c[0] + cr._fw(d, c[2], fmicro) <= x0 + colw + 1
    assert any(c[2].endswith("...") for c in note)


def test_the_recipe_note_is_bounded_at_the_columns_allowance():
    """col1 flows downward with no backstop, so an unbounded note pushes the
    [PRODUCTION LEDGER] and [SYSTEM] grids toward the footer."""
    h = 1080
    huge = " ".join("departure_%02d=value" % i for i in range(60))
    calls, _end, _d, _img = _spy_models(_models_block({"ltx_8gb": huge}))
    fmicro = cr._load_font(cr._sc(cr._PT_MICRO, h))
    note_lines = [c for c in calls
                  if c[3] is fmicro and c[2].startswith("departure_")]
    # The LITERAL, not cr._NOTE_LINES_MAX. Asserting against the constant is
    # tautological -- the mutation round raised the ceiling to 9 and this test
    # happily agreed with it, which is how a two-line note becomes a wall of
    # micro text on a card with room to spare. TWO lines is the product
    # decision: enough for a full LANE 2 receipt with its departure list.
    assert len(note_lines) == 2
    assert note_lines[-1][2].endswith("...")     # overflow folded, not dropped
    # ...and line 2 is a CONTINUATION of line 1, not an unrelated micro call:
    assert note_lines[1][0] == note_lines[0][0]
    assert note_lines[1][1] == note_lines[0][1] + cr._fh(fmicro)


_LONG_RECIPE = ("RECIPE_LTX8_I2V_v2+prequalification"
                "[tiled_vae=off,t5_device=cpu,attn=sage2]")


def _led_with(recipe):
    led = _led()
    led["meta"]["render_engines"]["by_engine"] = {
        eng: {"family": "image_to_video", "recipe": recipe, "quant": "Q8_0",
              "render_canvas": "704x400", "use_lora": True, "varied": [],
              "clip_count": 3}
        for eng in ("humo", "ltx_video", "wan_i2v")}
    return led


def _col1_end_y(recipe, w, h):
    from PIL import Image, ImageDraw
    lay = cr.build_credits_layout(_led_with(recipe), w=w, h=h, manifest={
        "clips": [{"shot_id": "s0", "path": "a.mp4", "exists": True,
                   "start_s": 0.0}],
        "total_target_frames": 400, "fps": 25, "clip_count": 3})
    img = Image.new("RGBA", (w, h), (0, 0, 0, 255))
    return cr._draw_col1(ImageDraw.Draw(img),
                         int(cr._COL1_X * h / cr._REF_H),
                         int(cr._MARGIN_TOP * h / cr._REF_H), lay, w, h)


@pytest.mark.parametrize("w,h", [(1280, 720), (1920, 1080), (3840, 2160)])
def test_col1_with_a_long_receipt_on_every_role_clears_the_footer(w, h):
    """THE REGRESSION THE PRE-PUSH FAN-OUT CAUGHT. A FIXED two-line note
    allowance overran the footer by 27px at 1280x720 -- the size this repo's
    own render tests already use -- so the [PRODUCTION LEDGER] rows drew into
    the footer band. The column measures what it can afford now."""
    end_y = _col1_end_y(_LONG_RECIPE, w, h)
    footer_top = h - int(56 * h / cr._REF_H)
    assert end_y <= footer_top, (
        "col1 ran into the footer at y=%d (footer starts %d)"
        % (end_y, footer_top))


def test_where_the_column_cannot_afford_a_note_it_adds_nothing():
    """854x480 overflows on its REQUIRED content alone and did so before any
    of this. The allowance drops to zero there, so the recipe note must cost
    exactly nothing rather than deepening a pre-existing overflow."""
    w, h = 854, 480
    assert _col1_end_y(_LONG_RECIPE, w, h) == _col1_end_y(None, w, h)


def test_the_card_shows_mixed_rather_than_one_clips_recipe():
    """Row 1 and row 2 meet here: the card draws what the PER-FIELD roll-up
    reports, so an engine that rendered two recipes says so on screen."""
    from nodes.otr_video_render_batch import _build_render_engines_payload
    # `exists: True` on both rows (2026-08-26). Since the sanctioned-gap work,
    # only rows with a clip actually on disk populate delivered-engine
    # accounting -- a refused beat must never be credited as motion somebody
    # rendered. Both beats here DID render, which is the whole premise of a
    # "two recipes on one engine" card, so they say so.
    ren = _build_render_engines_payload({"clips": [
        {"shot_id": "s1", "role": "music_visual", "engine_id": "ltx_8gb",
         "recipe": "RECIPE_LTX8_I2V_v2", "quant": "Q8_0", "exists": True},
        {"shot_id": "s2", "role": "music_visual", "engine_id": "ltx_8gb",
         "recipe": "RECIPE_LTX8_I2V_v2+prequalification[tiled_vae=off]",
         "quant": "Q8_0", "exists": True}]}, None)
    block = _models_block(
        {"ltx_8gb": cr._recipe_suffix(ren, "ltx_8gb")},
        video_rows=cr._video_role_rows(ren))
    drawn = " ".join(c[2] for c in _spy_models(block)[0])
    assert "mixed recipe" in drawn
    assert "RECIPE_LTX8_I2V_v2" not in drawn


# --------------------------------------------------------------------------- #
# THE COL1 LADDER (2026-07-28). The card is a VIEW of the durable ledger, not
# the ledger: it may show less than it knows, never claim more than it shows.
#
# This was filed as latent -- "reachable only if something renders the card at
# 480p". It is not: roll() sizes the card from the FINISHED VIDEO
# (_probe_video), the canonical workflow's VideoDirector ships 832x480, and the
# ltx_8gb tier renders 512x288. The shipped default was overflowing its own
# footer and PIL was clipping it in silence.
# --------------------------------------------------------------------------- #

def _col1_bottom(w, h, led=None):
    """Run the REAL ladder and return (floor_y, final_y, layout)."""
    lay = cr.build_credits_layout(led or _led_with("recipe_v2[tiled_vae=on]"),
                                  w=w, h=h, manifest={"clips": []})
    sx = h / cr._REF_H
    floor_y = h - int(56 * sx)
    y = cr._draw_col1(cr._scratch_draw(w, h), int(cr._COL1_X * sx),
                      int(cr._MARGIN_TOP * sx), lay, w, h)
    return floor_y, y, lay


@pytest.mark.parametrize("w,h", [(3840, 2160), (1920, 1080), (1280, 720),
                                 (854, 480), (832, 480)])
def test_col1_clears_its_footer_at_every_shipped_canvas(w, h):
    """832x480 is the CANONICAL WORKFLOW's own canvas. It used to overflow."""
    floor_y, y, _lay = _col1_bottom(w, h)
    assert y <= floor_y, (
        "col1 ends %dpx past the footer at %dx%d" % (y - floor_y, w, h))


def test_the_canonical_canvas_keeps_its_WHOLE_ledger():
    """The fix at 832x480 is bought with WHITESPACE, not information.

    Asserted separately from "it fits", because a ladder that fits by dropping
    the frame budget and the VRAM peak off the shipped end-card would satisfy
    that test while quietly costing the operator the two rows they most often
    want.

    It observes WHAT WAS DRAWN, not the layout handed in. The first version of
    this test read the input layout -- which `_abridge` deliberately COPIES
    rather than mutates -- so it passed no matter what the ladder did, and the
    mutation round duly found `_GAP_TIERS = (1.0,)` surviving: with the
    whitespace rung deleted the column falls through to dropping rows, still
    "fits", and the old assertion never noticed."""
    drawn = {}
    real_grid = cr._draw_grid

    def _spy(draw, x, y, header, rows, h, **kw):
        drawn.setdefault("grids", []).append((header, [r[0] for r in rows]))
        return real_grid(draw, x, y, header, rows, h, **kw)

    cr._draw_grid = _spy
    try:
        _floor, _y, lay = _col1_bottom(832, 480)
    finally:
        cr._draw_grid = real_grid

    header, labels = drawn["grids"][-1]
    # Derived from the layout, not hard-coded: which ledger rows exist at all
    # depends on the manifest (FRAMES only appears when it carries a frame
    # count), and a hard-coded list would fail for the wrong reason.
    expected = [r[0] for k, b in lay["col1"] if k == "grid" for r in b["rows"]]
    assert labels == expected, (labels, expected)
    assert "ABRIDGED" not in header, header
    for stamp in ("SEED:", "COMMIT:"):
        assert stamp in labels, (stamp, labels)


def test_the_canonical_canvas_abridges_NOTHING_and_says_nothing(caplog):
    """The other half of the same fact, from the log rather than the pixels:
    at the shipped canvas the ladder is SILENT. A warning here would mean the
    end-card of every canonical episode is quietly shorter than the ledger."""
    with caplog.at_level(logging.WARNING, logger=cr.log.name):
        _col1_bottom(832, 480)
    assert "ABRIDGED" not in caplog.text, caplog.text
    assert "OVERFLOWS" not in caplog.text, caplog.text


def test_the_whitespace_tiers_never_touch_TYPE():
    """The whole reason whitespace is spent before content: type is a
    legibility floor, and a receipt in unreadable type is a receipt-shaped
    object claiming credit for a disclosure that never happened.

    Compares the FONT SIZES the flow asks for at the loosest and tightest
    tier. Only the gaps may differ."""
    lay = cr.build_credits_layout(_led_with("recipe_v2"), w=832, h=480,
                                  manifest={"clips": []})
    real_load = cr._load_font

    def _sizes_at(tier):
        seen = []

        def _spy(pt):
            seen.append(pt)
            return real_load(pt)

        cr._load_font = _spy
        try:
            cr._flow_col1(cr._scratch_draw(832, 480), 30, 20, lay, 832, 480,
                          0, gaps=tier)
        finally:
            cr._load_font = real_load
        return seen

    loose, tight = _sizes_at(1.0), _sizes_at(0.25)
    assert loose == tight, (
        "the tightest whitespace tier changed a font size: %r vs %r"
        % (loose, tight))


def test_a_canvas_too_small_for_the_card_ABRIDGES_and_SAYS_SO(caplog):
    """512x288 is the ltx_8gb tier. Even abridged the card does not fit there
    -- so it is drawn anyway (a terminal node never destroys a finished
    episode) and the shortfall is LOGGED. What is not acceptable is the old
    behaviour: drawn, clipped by PIL, and silent."""
    with caplog.at_level(logging.WARNING, logger=cr.log.name):
        _floor, _y, lay = _col1_bottom(512, 288)
    grid = [b for k, b in lay["col1"] if k == "grid"][0]
    # The LAYOUT is untouched -- abridging is a drawing-time view, so the
    # ledger the rest of the card reads from still has every row.
    assert grid["header"] == "[ PRODUCTION LEDGER ]"
    text = caplog.text
    assert "512x288" in text
    assert "OVERFLOWS" in text
    assert "small-canvas variant" in text


def test_the_abridged_view_keeps_the_reproducibility_stamps():
    """SEED and COMMIT are the two rows that make an episode findable again,
    and they are deliberately absent from the drop order."""
    assert "SEED:" not in cr._LEDGER_DROP_ORDER
    assert "COMMIT:" not in cr._LEDGER_DROP_ORDER
    # Fine print goes before marquee: a frame budget and a VRAM peak are
    # telemetry, a revision pair is a footnote.
    assert cr._LEDGER_DROP_ORDER == ("FRAMES:", "VRAM:", "REV:")

    lay = cr.build_credits_layout(_led_with("recipe_v2"), w=512, h=288,
                                  manifest={"clips": []})
    trimmed = cr._abridge(lay, list(cr._LEDGER_DROP_ORDER))
    grid = [b for k, b in trimmed["col1"] if k == "grid"][0]
    labels = [r[0] for r in grid["rows"]]
    assert "SEED:" in labels and "COMMIT:" in labels
    assert "FRAMES:" not in labels and "VRAM:" not in labels
    # EVERY CUT IS MARKED -- on the header, which is the tier still legible at
    # the size that forced the cut, and again in a tail row with the count.
    assert grid["header"] == "[ PRODUCTION LEDGER -- ABRIDGED ]"
    assert labels[-1] == "+3 CUT"


def test_abridging_never_mutates_the_callers_layout():
    """The abridged card is a VIEW. If it edited the layout in place, the
    scroll column and the footers would start reading a ledger the drawing
    pass had quietly shortened."""
    lay = cr.build_credits_layout(_led_with("recipe_v2"), w=832, h=480,
                                  manifest={"clips": []})
    before = [r[0] for k, b in lay["col1"] if k == "grid" for r in b["rows"]]
    cr._abridge(lay, ["FRAMES:", "VRAM:"])
    after = [r[0] for k, b in lay["col1"] if k == "grid" for r in b["rows"]]
    assert before == after


# --------------------------------------------------------------------------- #
# WIRE-W6 -- the failure boundary: TRUTH is terminal, GLASS degrades
# --------------------------------------------------------------------------- #
#
# This node is the LAST in the graph, so anything it raises costs a whole
# rendered episode. It still may not be incapable of failing, because the
# standing policy is that the card is a VIEW of the durable ledger: a record
# may never elide. r4/A7 drew the line -- an unreadable body, a malformed
# manifest and missing ledger TRUTH stay terminal; everything that merely makes
# a PICTURE degrades to "the finished episode, no credits tail, and a receipt
# that says so".


def _roll_ledger(monkeypatch, data=None):
    """Point the node's `get_ledger()` at a fixture ledger."""
    from nodes import production_ledger as pl

    class _L:
        pass

    led = _L()
    led.data = _led() if data is None else data
    monkeypatch.setattr(pl, "get_ledger", lambda: led)
    return led


@needs_ffmpeg
def test_a_presentation_failure_returns_the_EPISODE_with_a_zero_tail(
        tmp_path, monkeypatch):
    """The whole point of the boundary. The console failed to compose; the
    episode is finished and sitting on disk. Hand it back."""
    _roll_ledger(monkeypatch)
    body = tmp_path / "body.mp4"
    _silent_video(body, 1.0, size="1280x720")

    def _explode(*_a, **_k):
        raise RuntimeError("libx264 fell over")

    monkeypatch.setattr(cr, "render_credits_clip", _explode)
    out, tail, report = cr.OTRCreditsRoll().roll(
        str(body), json.dumps({"clips": []}))

    assert out == str(body), "the finished episode must be handed back as-is"
    assert tail == 0.0, (
        "the mux's credits-aware guard must be told there is NO tail; a "
        "non-zero declaration here would make it reserve time for a console "
        "that was never appended")
    rep = json.loads(report)
    assert rep["ok"] is False and rep["credits_rendered"] is False
    assert rep["reason"] == "presentation_failure"
    assert "libx264 fell over" in rep["error"]


@needs_ffmpeg
def test_a_backdrop_extraction_failure_is_also_presentation_only(
        tmp_path, monkeypatch):
    """Frame extraction sits INSIDE the glass half: the body is readable (it
    probed fine two lines earlier), we simply could not make a picture."""
    _roll_ledger(monkeypatch)
    body = tmp_path / "body.mp4"
    _silent_video(body, 1.0, size="1280x720")
    monkeypatch.setattr(cr, "extract_final_frame", lambda *a, **k: (_ for _ in ()).throw(
        cr.CreditsDataError("no frame")))
    out, tail, report = cr.OTRCreditsRoll().roll(
        str(body), json.dumps({"clips": []}))
    assert out == str(body) and tail == 0.0
    assert json.loads(report)["ok"] is False


def test_a_missing_body_is_TERMINAL(tmp_path, monkeypatch):
    _roll_ledger(monkeypatch)
    with pytest.raises(cr.CreditsDataError, match="credits input video missing"):
        cr.OTRCreditsRoll().roll(str(tmp_path / "nope.mp4"), "{}")


@needs_ffmpeg
def test_a_malformed_manifest_is_TERMINAL(tmp_path, monkeypatch):
    _roll_ledger(monkeypatch)
    body = tmp_path / "body.mp4"
    _silent_video(body, 1.0, size="1280x720")
    with pytest.raises(cr.CreditsDataError, match="unparseable"):
        cr.OTRCreditsRoll().roll(str(body), "{not json")


@needs_ffmpeg
def test_missing_ledger_TRUTH_is_TERMINAL_not_degraded(tmp_path, monkeypatch):
    """The half that must NOT soften. A ledger missing a receipt the card is
    obliged to show would publish credits claiming less than the build knows --
    the one failure this card exists to prevent. It raises from
    build_credits_layout, ABOVE the try, so the glass half never sees it."""
    hollow = _led()
    hollow["meta"] = dict(hollow.get("meta") or {})
    hollow["meta"].pop("episode_title", None)
    _roll_ledger(monkeypatch, hollow)
    body = tmp_path / "body.mp4"
    _silent_video(body, 1.0, size="1280x720")
    with pytest.raises(cr.CreditsDataError):
        cr.OTRCreditsRoll().roll(str(body), json.dumps({"clips": []}))


@needs_ffmpeg
def test_an_all_DIRECTORY_episode_now_publishes(tmp_path, monkeypatch):
    """THE mesh_stage DEFECT, closed.

    An episode rendered entirely by mesh_stage has only frame DIRECTORIES in
    its clip manifest and no mp4 at all. The old backdrop planner searched that
    manifest for a loopable file clip, found none, and raised -- so the
    terminal node refused an episode that had rendered every one of its shots.
    The manifest is no longer consulted: the backdrop is the body video's own
    final frame, and the body always exists because it is this node's input.
    """
    _roll_ledger(monkeypatch)
    body = tmp_path / "body.mp4"
    _silent_video(body, 1.0, size="1280x720")
    frames_dir = tmp_path / "shot_000_frames"
    frames_dir.mkdir()
    manifest = {"clips": [
        {"shot_id": "s0", "path": str(frames_dir), "exists": True},
        {"shot_id": "s1", "path": str(frames_dir), "exists": True}]}

    out, tail, report = cr.OTRCreditsRoll().roll(str(body), json.dumps(manifest))
    rep = json.loads(report)
    assert rep["ok"] is True and rep["credits_rendered"] is True
    assert rep["backdrop_source"] == "body_final_frame"
    assert tail > 0.0 and os.path.exists(out) and out != str(body)


# ---------------------------------------------------------------------------
# THE NON-COMMERCIAL NOTICE (2026-08-07)
#
# meta["noncommercial_notice"] has been stamped by the writer since the
# provenance work and NOTHING RENDERED IT -- a rights warning with no human
# surface. These tests pin the two conditions that are easy to get wrong: it
# must render INDEPENDENTLY of credits_source_line, and it must not invent a
# second label in front of the notice's own wording.
#
# Every assertion reads the ORDERED col3_flow list. Converting it to a dict
# collapses duplicate "intercept" keys and would silently pass.
# ---------------------------------------------------------------------------

_NOTICE = (
    "NON-COMMERCIAL SOURCE: this episode adapts Folger Shakespeare text "
    "licensed CC BY-NC 3.0, which does NOT permit commercial use. Do not sell "
    "it, and do not publish it on a monetized channel. Personal and "
    "non-commercial sharing is fine."
)


def _intercepts(lay):
    """The intercept texts, IN ORDER. Never dict() -- duplicates collapse."""
    return [b.get("text", "") for k, b in lay["col3_flow"] if k == "intercept"]


def _lay_with_meta(**meta_over):
    led = _led()
    led["meta"].update(meta_over)
    return cr.build_credits_layout(led, w=1920, h=1080, manifest={"clips": []})


def test_noncommercial_notice_renders_exactly_once_with_its_own_prefix():
    lay = _lay_with_meta(noncommercial_notice=_NOTICE)
    hits = [t for t in _intercepts(lay) if "NON-COMMERCIAL SOURCE:" in t]
    assert len(hits) == 1, f"expected exactly one notice, got {hits!r}"
    # ">> " + the notice's own wording. No second label: a
    # ">> NON-COMMERCIAL NOTICE: %s" wrapper would stutter.
    assert hits[0] == ">> " + _NOTICE
    assert "NOTICE: NON-COMMERCIAL" not in hits[0]


def test_noncommercial_notice_renders_without_a_source_line():
    """THE ONE THAT MATTERS. A malformed legacy ledger can carry the notice and
    no credits_source_line, and the rights warning is exactly the line that must
    survive the other field's absence -- which is why it is its own `if`."""
    lay = _lay_with_meta(noncommercial_notice=_NOTICE,
                         credits_source_line="")
    texts = _intercepts(lay)
    assert any(t == ">> " + _NOTICE for t in texts)
    assert not any(t.startswith(">> SOURCE:") for t in texts)


def test_noncommercial_notice_follows_the_source_line_when_both_exist():
    lay = _lay_with_meta(credits_source_line="adapted from Hamlet (Folger)",
                         noncommercial_notice=_NOTICE)
    texts = _intercepts(lay)
    src = next(i for i, t in enumerate(texts) if t.startswith(">> SOURCE:"))
    notice = next(i for i, t in enumerate(texts) if "NON-COMMERCIAL SOURCE:" in t)
    assert notice == src + 1, (
        f"notice must immediately follow the source line; got {texts!r}")


def test_no_notice_intercept_when_the_field_is_absent():
    lay = _lay_with_meta()
    assert not any("NON-COMMERCIAL" in t for t in _intercepts(lay))


def test_whitespace_only_notice_emits_no_bare_marker():
    """Without .strip() a whitespace-only field renders a naked '>>'."""
    lay = _lay_with_meta(noncommercial_notice="   \n\t ")
    texts = _intercepts(lay)
    assert not any(t.strip() == ">>" for t in texts), f"bare marker in {texts!r}"
    assert not any("NON-COMMERCIAL" in t for t in texts)
