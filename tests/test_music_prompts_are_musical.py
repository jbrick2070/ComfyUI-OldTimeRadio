"""Every music prompt the pack composes is MUSICAL and story-relevant
(operator, 2026-09-11 evening, after `moonlit_deception`: "the 'music' needs to
be improved, doesn't sound like music ... sounds like radio hiss, which is what
we asked. So I am asking to go over ALL musical prompts and make them more
musical ... ideally it is relevant to the story").

Two products from one composer (`_otr_music_prompt`): the ROW text the ledger
stores and hashes, and the ENGINE prompt every adapter hears -- the story
palette's instruments and a clean production anchor in front of the row text,
plus the one negative prompt. No engine prepends its own texture any more.
"""
from __future__ import annotations

import inspect
import itertools

import pytest

from nodes import _otr_music_palette as P
from nodes import _otr_music_prompt as MP
from nodes._otr_audio_engines import eng_stable_audio_3 as SA3

_NOISE_WORDS = ("hiss", "static", "noise", "crackle", "squelch", "tape",
                "vintage", "lo-fi", "lofi", "degraded", "pristine")
_BANKS = ("shakespeare", "public_domain", "original", "media_archive",
          "my_story", "scifi_news_pro")
_YEARS = (None, "c. 1595", 1750, 1897, 1935, 1975)


def _meta(bank, year, moods=("tense", "moonlit", "secretive"),
          setting=("forest", "moonlit night")):
    meta = {"story_brief_status": "ok",
            "story_brief_terms": {"setting": list(setting)},
            "music_mood_terms": list(moods),
            "source_bank": bank}
    if year is not None:
        meta["source_meta"] = {"year": year}
    return meta


def _instrument_count(text):
    low = text.lower()
    return sum(1 for name in (
        "lute", "viol", "recorder", "harpsichord", "strings", "brass", "clarinet",
        "vibraphone", "bass", "piano", "quartet", "horn", "oboe", "bassoon",
        "guitar", "organ", "drum", "theremin", "timpani", "cello", "harp",
        "celesta", "flute", "woodwinds", "snare") if name in low)


@pytest.mark.parametrize("bank,year,cue", list(itertools.product(_BANKS, _YEARS, MP.CUE_DURATIONS)))
def test_every_bank_period_and_cue_composes_music_not_texture(bank, year, cue):
    row, duration = MP.compose_music_prompt(_meta(bank, year), cue)
    engine = MP.compose_engine_prompt(_meta(bank, year), row)
    for text in (row, engine.text):
        low = text.lower()
        assert not [w for w in _NOISE_WORDS if w in low], text
    assert _instrument_count(engine.text) >= 2, engine.text
    assert row.endswith("instrumental only, no dialogue, no vocals")
    assert engine.text.endswith("instrumental only, no dialogue, no vocals")
    assert engine.text.startswith(P.story_palette(_meta(bank, year)).instruments)
    assert MP.PRODUCTION_ANCHOR in engine.text
    assert duration == MP.CUE_DURATIONS[cue]
    assert len(engine.text) <= MP.ENGINE_PROMPT_MAX_CHARS


def test_the_row_text_reads_as_a_musical_instruction_about_this_story():
    row, _ = MP.compose_music_prompt(_meta("shakespeare", "c. 1595",
                                           moods=("enchanted", "playful", "tension-building")),
                                     "opening")
    assert row.startswith("enchanted, playful, tension-building"), "the brief's words lead"
    assert "shimmering sustained strings, soft celesta colour" in row
    assert "Elizabethan consort music" in row
    assert "evokes forest, moonlit night" in row
    assert "a rising overture that settles into a flowing theme" in row
    # the magic/dream group asks for a floating pulse rather than grief's rubato
    assert "slow tempo, floating and unmetered" in row, "the model is told the pace"
    # "enchanted" is slow and "playful" is fast: the contradiction is dropped
    assert "dancing woodwind" not in row


def test_the_engine_prompt_leads_with_the_ensemble_and_a_clean_recording():
    meta = _meta("shakespeare", "c. 1595")
    row, _ = MP.compose_music_prompt(meta, "closing")
    engine = MP.compose_engine_prompt(meta, row)
    assert engine.text.startswith("viol consort, recorders, soft bowed strings, gentle lute, "
                                  "clearly recorded, clean balanced studio mix, natural room. ")
    assert engine.text.endswith(row)
    assert engine.palette_key == "early_consort"
    assert engine.negative == MP.NEGATIVE_PROMPT_DEFAULT


def test_the_negative_prompt_names_the_withdrawn_texture_and_speech():
    low = MP.NEGATIVE_PROMPT_DEFAULT.lower()
    for word in ("hiss", "static", "noise", "vocals", "speech", "clipping"):
        assert word in low


def test_the_negative_prompt_says_this_is_not_a_loop():
    """Measured 2026-09-12 as the single biggest lever: these words alone
    took the closing cue's envelope periodicity from 0.78 to 0.26 over four
    seeds. Stable Audio Open is built to make loops, so a cue must say it is
    not one."""
    low = MP.NEGATIVE_PROMPT_DEFAULT.lower()
    for word in ("loop", "repetitive", "ostinato", "sequencer", "metronome",
                 "drum machine", "click track", "beat"):
        assert word in low, word


def test_the_positive_prompt_never_asks_for_the_thing_the_negative_forbids():
    """codex r1, 2026-09-12: the negative prompt spent the campaign banning
    beats and loops while `_MOOD_TAGS` -- the last-ditch keyword path into the
    SAME row text -- was still asking for "tighter rhythm, percussive accents"
    on any brief containing the word "urgent". A cue of four to twelve seconds
    that is told to be percussive has only one way to answer."""
    banned = ("rhythm", "rhythmic", "percussive", "percussion", "drum", "beat",
              "pulse", "ostinato", "loop", "metronome", "steady")
    for keyword, tag in MP._MOOD_TAGS.items():
        low = tag.lower()
        assert not [w for w in banned if w in low], (keyword, tag)


def test_an_authored_prompt_survives_verbatim_inside_the_engine_prompt():
    authored = "slow open on a lonely trumpet over a city at night"
    engine = MP.compose_engine_prompt(_meta("my_story", None), authored)
    assert engine.text.endswith(". " + authored)
    assert engine.text.startswith(P.HOUSE_PALETTE.instruments)


def test_the_engine_prompt_is_capped_below_the_smallest_engine_budget_without_raising():
    long_row = ", ".join("a long clause number %d" % i for i in range(120))
    assert len(long_row) > MP.ENGINE_PROMPT_MAX_CHARS
    engine = MP.compose_engine_prompt(_meta("my_story", None), long_row)
    assert len(engine.text) <= MP.ENGINE_PROMPT_MAX_CHARS
    assert engine.text.startswith(P.HOUSE_PALETTE.instruments)
    assert engine.text.endswith("clause number %d" % (
        int(engine.text.rsplit("number ", 1)[-1])))  # cut at a clause boundary
    assert not engine.text.endswith(",")


def test_an_overflowing_composed_row_keeps_its_instrumental_tail():
    meta = _meta("my_story", None,
                 moods=tuple("mood word number %d" % i for i in range(3)),
                 setting=tuple("a setting phrase that runs long %d" % i for i in range(2)))
    row, _ = MP.compose_music_prompt(meta, "opening")
    long_row = row.replace("evokes", "evokes " + ", ".join(
        "an extra scenic clause %d" % i for i in range(60)) + ",")
    assert len(long_row) > MP.ENGINE_PROMPT_MAX_CHARS
    assert long_row.endswith("instrumental only, no dialogue, no vocals")
    engine = MP.compose_engine_prompt(meta, long_row)
    assert len(engine.text) <= MP.ENGINE_PROMPT_MAX_CHARS
    assert engine.text.endswith("instrumental only, no dialogue, no vocals")
    assert engine.text.startswith(P.HOUSE_PALETTE.instruments)


def test_junk_meta_still_composes_both_products():
    for junk in (None, {}, [], "x", {"source_meta": "c. 1595", "music_mood_terms": 3}):
        row, _ = MP.compose_music_prompt(junk if isinstance(junk, dict) else {}, "opening")
        engine = MP.compose_engine_prompt(junk, row)
        assert engine.text and engine.negative and engine.palette_key


def test_the_dead_period_voice_overlay_is_gone():
    """`gen_params_initial.period_voice` had zero producers (grep, 2026-09-11)
    and the palette now owns the period. Half-removed is the defect."""
    source = inspect.getsource(MP)
    assert "period_voice" not in source


# --------------------------------------------------------------------------- #
# the SA3 engine no longer prepends its own era anchor
# --------------------------------------------------------------------------- #
def test_sa3_sends_the_composed_prompt_verbatim_and_uses_the_composer_negative():
    source = inspect.getsource(SA3)
    for gone in ("_sa3_augment_prompt", "_SA3_PERIOD_GENRE", "_SA3_DEFAULT_GENRE",
                 "_SA3_NEG_DEFAULT", "analog tape warmth", "modern pristine mix"):
        assert gone not in source, gone
    body = inspect.getsource(SA3.StableAudio3Engine.generate_clip)
    assert "pos_text = str(prompt or \"\").strip()" in body
    assert "NEGATIVE_PROMPT_DEFAULT" in body
    assert "OTR_SA3_NEG_PROMPT" in body, "the operator's one override stays"
    assert "placement or prompt" in body, "the cue's placement names the window"
    # THE RECEIPT RECORDS WHAT THE ENGINE HEARD, NOT WHAT IT WAS OFFERED
    # (codex r2, 2026-09-12). `OTR_SA3_NEG_PROMPT` overrides the composer's
    # negative here, so a receipt carrying only the composer's would describe a
    # cue nobody heard -- and `scripts/otr_music_ab.py` reads exactly these two
    # fields to prove an A/B arm actually ran.
    assert '"negative_prompt": str(neg_text)' in body
    assert '"denoise": float(denoise)' in body
    # AND THE RATIO IT USED (cursor r3): `seconds_total` alone cannot tell a
    # 1.0 control arm from the shipped default, because a 36 s window satisfies
    # "at least 1x a 12 s cue". The one knob this campaign most wants to A/B was
    # the one the receipt could not identify.
    assert '"context_ratio": _sa3_context_ratio()' in body


def test_sa3_window_follows_the_placement_not_a_word_in_the_prompt(monkeypatch):
    # THE ENVIRONMENT IS AN INPUT TO THIS FUNCTION (cursor r3, 2026-09-12).
    # `_env_float` reads it live, so an A/B shell that exported the control
    # ratio turned these assertions red -- or, far worse, green while the
    # defect they guard was live.
    monkeypatch.delenv("OTR_SA3_CONTEXT_RATIO", raising=False)
    """The window is placed by PLACEMENT, and is always longer than the cue
    (the floor below), so no cue is ever a self-contained piece."""
    ctx = 12.0
    start, total = SA3._sa3_clip_window("opening", 12.0, ctx)
    assert start == 0.0 and total == 36.0, "12 s cue in a 12 s window was a loop"
    start, total = SA3._sa3_clip_window("closing", 8.0, ctx)
    assert abs(start - 16.0) < 1e-6 and total == 24.0
    start, total = SA3._sa3_clip_window("interstitial", 4.0, ctx)
    assert abs(start - 4.0) < 1e-6 and total == 12.0, "already 3x: unchanged"


def test_a_cue_is_never_a_self_contained_piece(monkeypatch):
    monkeypatch.delenv("OTR_SA3_CONTEXT_RATIO", raising=False)
    """BUG-408 existed to kill seconds_total == dur, and it stayed live for
    the longest cue: at the shipped 12 s context the 12 s opening cue got
    exactly that, which is a loop-shaped request (measured 2026-09-12)."""
    for placement, dur in (("opening", 12.0), ("closing", 8.0), ("interstitial", 4.0)):
        for ctx in (4.0, 8.0, 12.0, 45.0):
            start, total = SA3._sa3_clip_window(placement, dur, ctx)
            assert total >= dur * SA3._SA3_MIN_CONTEXT_RATIO_DEFAULT - 1e-6, (placement, dur, ctx, total)
            assert total > dur, (placement, dur, ctx, total)
            assert 0.0 <= start <= total - dur + 1e-6, (placement, dur, ctx, start)
    # an operator who asks for MORE context still gets it
    assert SA3._sa3_clip_window("opening", 12.0, 90.0)[1] == 90.0


def test_a_ratio_below_the_floor_is_a_control_arm_and_warns_that_it_breaks_the_invariant(
        monkeypatch, caplog):
    """The 3x floor is the change most likely to have over-corrected the music
    into formlessness -- the opening cue renders the first third of a 36 s arc
    and may never reach the theme it is told to settle into -- and a module
    constant cannot be A/B'd by scripts/otr_music_ab.py (Fable, 2026-09-12).

    BUT 1.0 DOES NOT MERELY RESTORE THE OLD BEHAVIOUR, IT RESTORES THE DEFECT
    (codex, 2026-09-12): seconds_total == dur is the self-contained,
    loop-shaped request `test_a_cue_is_never_a_self_contained_piece` exists to
    forbid. It stays reachable because a control arm has to reproduce what it
    controls for, and it has to say so out loud every time."""
    monkeypatch.setenv("OTR_SA3_CONTEXT_RATIO", "1.0")
    with caplog.at_level("WARNING"):
        assert SA3._sa3_clip_window("opening", 12.0, 12.0) == (0.0, 12.0)
    assert any("control arm" in r.getMessage() for r in caplog.records), caplog.text
    caplog.clear()
    monkeypatch.setenv("OTR_SA3_CONTEXT_RATIO", "2.0")
    assert SA3._sa3_clip_window("opening", 12.0, 12.0) == (0.0, 24.0)
    # never BELOW the cue itself, whatever is asked for
    monkeypatch.setenv("OTR_SA3_CONTEXT_RATIO", "0.1")
    start, total = SA3._sa3_clip_window("opening", 12.0, 12.0)
    assert total >= 12.0 and start == 0.0
    # junk falls back to the default rather than crashing a render -- and the
    # default is not a control arm, so it does not warn
    caplog.clear()
    monkeypatch.setenv("OTR_SA3_CONTEXT_RATIO", "not-a-number")
    with caplog.at_level("WARNING"):
        assert SA3._sa3_clip_window("opening", 12.0, 12.0)[1] == 36.0
    assert not [r for r in caplog.records if "control arm" in r.getMessage()]


# --------------------------------------------------------------------------- #
# every adapter accepts the two keyword arguments the theme node now passes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("module,cls", [
    ("eng_stable_audio_3", "StableAudio3Engine"),
    ("eng_musicgen", None), ("eng_google_lyria", None),
    ("eng_cloud_sonilo", None), ("eng_stable_audio", None),
])
def test_every_music_adapter_accepts_placement_and_negative_prompt(module, cls):
    import importlib
    mod = importlib.import_module("nodes._otr_audio_engines." + module)
    classes = [getattr(mod, cls)] if cls else [
        obj for name, obj in vars(mod).items()
        if inspect.isclass(obj) and hasattr(obj, "generate_clip")
        and obj.__module__ == mod.__name__]
    assert classes, module
    for klass in classes:
        params = inspect.signature(klass.generate_clip).parameters
        assert params["placement"].kind is inspect.Parameter.KEYWORD_ONLY, klass
        assert params["negative_prompt"].kind is inspect.Parameter.KEYWORD_ONLY, klass


# --------------------------------------------------------------------------- #
# the guidance follows the checkpoint (measured 2026-09-12, 65 renders)
# --------------------------------------------------------------------------- #
def test_a_post_trained_checkpoint_never_gets_a_guidance_it_cannot_answer(monkeypatch):
    """THE SCRATCH, AND WHY IT HAPPENED. Stability ships each Stable Audio 3
    model in a BASE and a POST-TRAINED form and documents that `cfg_scale` and
    `negative_prompt` "have no effect on post-trained checkpoints". ComfyUI
    drops the unconditional branch only at cfg 1.0, so at 7.0 it extrapolates
    sevenfold off a branch the post-training collapsed, and a few latent frames
    land off the manifold. Measured: 74 bursts in 17 renders of that
    combination against 2 in 48 renders of every other one.

    Every anti-loop word this pack wrote lives in the negative prompt, so the
    base file is preferred when it exists -- that is the only place the work is
    live."""
    engine = SA3.StableAudio3Engine
    assert SA3._CKPT_PREFERENCE[0].endswith("_base.safetensors"), "base is preferred"
    assert SA3._CFG_FOR_POST_TRAINED == 1.0, (
        "the only value ComfyUI turns the extrapolation off at")
    assert SA3._CFG_FOR_BASE > 1.0, "a base checkpoint can be steered"

    # an explicit override always wins -- that is how the A/B harness moves it
    monkeypatch.setattr(SA3, "_CKPT", "stable_audio_3_medium_base.safetensors")
    assert engine.resolve_ckpt() == ("stable_audio_3_medium_base.safetensors", True)
    monkeypatch.setattr(SA3, "_CKPT", "stable_audio_3_small_music.safetensors")
    assert engine.resolve_ckpt() == ("stable_audio_3_small_music.safetensors", False)


def test_an_install_with_only_the_post_trained_file_still_renders(monkeypatch):
    """Nobody has to download anything. A box that has only the file it always
    had keeps working and simply gets the quieter guidance."""
    monkeypatch.setattr(SA3, "_CKPT", "")
    import sys
    import types
    fake = types.ModuleType("folder_paths")
    fake.get_full_path = lambda kind, name: (
        r"C:\models\%s" % name if name == SA3._CKPT_PREFERENCE[-1] else None)
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    name, is_base = SA3.StableAudio3Engine.resolve_ckpt()
    assert name == SA3._CKPT_PREFERENCE[-1] and is_base is False

    # and when the base file IS there, it wins
    fake.get_full_path = lambda kind, name: r"C:\models\%s" % name
    name, is_base = SA3.StableAudio3Engine.resolve_ckpt()
    assert name == SA3._CKPT_PREFERENCE[0] and is_base is True


def test_resolution_never_raises_when_comfy_is_absent(monkeypatch):
    """This runs inside a render and off it; an import that is not there is not
    a reason to lose an episode."""
    monkeypatch.setattr(SA3, "_CKPT", "")
    import sys
    monkeypatch.setitem(sys.modules, "folder_paths", None)
    name, is_base = SA3.StableAudio3Engine.resolve_ckpt()
    assert name == SA3._CKPT_PREFERENCE[-1] and is_base is False


def test_the_guidance_a_checkpoint_gets_is_the_one_it_can_answer(monkeypatch):
    """The mechanism, callable. A constant here is what put a sevenfold
    extrapolation on a model whose publisher documents that guidance does
    nothing -- and it is the one combination, of four measured, that makes the
    scratch."""
    monkeypatch.delenv("OTR_SA3_CFG", raising=False)
    assert SA3._sa3_guidance(True) == SA3._CFG_FOR_BASE
    assert SA3._sa3_guidance(False) == SA3._CFG_FOR_POST_TRAINED
    assert SA3._sa3_guidance(True) != SA3._sa3_guidance(False), (
        "the whole point is that they differ")
    # the operator's override still wins on both
    monkeypatch.setenv("OTR_SA3_CFG", "6.5")
    assert SA3._sa3_guidance(True) == 6.5 and SA3._sa3_guidance(False) == 6.5


def test_the_placement_knob_does_not_claim_to_have_reached_the_model():
    """VERIFIED IN THE RUNTIME, 2026-09-12. `StableAudio3.extra_conds` in
    comfy/model_base.py embeds `seconds_total` and nothing else; the checkpoint
    has no seconds_start embedder, and only the older StableAudio1 class carries
    one. So "an outro sits at the TAIL, an intro at the HEAD" has never reached
    this model -- it was computed, sent, logged and receipted all the same.

    The value still goes to the node, because the node takes it and a future
    checkpoint may read it. What must never happen again is a log line or a
    receipt field that lets a reader believe it worked."""
    assert SA3._SA3_READS_SECONDS_START is False
    body = inspect.getsource(SA3.StableAudio3Engine.generate_clip)
    assert "IGNORED by SA3" in body, "the log tells the truth"
    assert '"seconds_start_read_by_model": _SA3_READS_SECONDS_START' in body, (
        "the receipt tells the truth")
    # and the window is still computed and still sent -- this is a truth fix,
    # not a removal
    assert "seconds_start, seconds_total = _sa3_clip_window(" in body


# --------------------------------------------------------------------------- #
# a bank whose genre has a beat must not be forbidden a beat (2026-09-12)
# --------------------------------------------------------------------------- #
def test_a_rhythmic_bank_is_never_forbidden_its_own_genre():
    """THE SELF-CANCELLING REQUEST, GUARDED AT THE SOURCE. The underscore's
    negative bans loop, looping, repetitive, ostinato, sequencer, arpeggiator,
    drum machine, metronome, click track, drum loop and beat. Every one of those
    is a REQUIREMENT of techno, house, salsa and a swinging jazz quartet.

    Shipping the underscore's negative onto a dance bank is the identical defect
    that tore the cues earlier the same day -- a prompt asking for "raw distorted
    TR-909" while the negative banned distortion."""
    forbidden_of_a_groove = ("loop", "ostinato", "sequencer", "arpeggiator",
                             "drum machine", "metronome", "click track", "beat")
    for bank in ("scifi_news_pro", "media_archive", "original", "public_domain"):
        meta = _meta(bank, None)
        row, _ = MP.compose_music_prompt(meta, "opening")
        engine = MP.compose_engine_prompt(meta, row)
        assert P.story_palette(meta).rhythmic is True, bank
        low = engine.negative.lower()
        for word in forbidden_of_a_groove:
            assert word not in low, (bank, word)
        # what is a defect in ANY music is still banned
        for word in ("noise", "hiss", "distortion", "clipping", "vocals"):
            assert word in low, (bank, word)

    # and a sustained bank keeps the full anti-loop negative
    meta = _meta("shakespeare", "c. 1595")
    row, _ = MP.compose_music_prompt(meta, "opening")
    assert MP.compose_engine_prompt(meta, row).negative == MP.NEGATIVE_PROMPT_DEFAULT


def test_a_rhythmic_bank_is_not_also_told_to_play_slow_and_sustained():
    """The other half of the same contradiction. `mood_devices` and
    `tempo_phrase` are orchestral instructions written for a held cue -- "slow
    tempo, sustained and taut, no rubato" is a fine thing to tell a string
    section and a contradiction to tell a 128 BPM drum machine. On a genre bank
    the idiom carries the musical instruction instead, including the BPM."""
    meta = _meta("scifi_news_pro", None, moods=("ominous", "suspenseful"))
    row, _ = MP.compose_music_prompt(meta, "opening")
    assert "Detroit techno at 128 BPM" in row
    assert "sustained and taut" not in row
    assert "slow rising strings" not in row
    # the brief's own words still lead: story relevance survives the genre
    assert row.startswith("ominous, suspenseful")


def test_the_theme_node_offers_the_style_field_and_threads_it():
    """The widget exists, is APPENDED last (widgets_values is positional), and
    its value actually reaches the composer -- a field that changes nothing is
    worse than no field."""
    from nodes import stable_audio_theme as T
    spec = T.StableAudioTheme.INPUT_TYPES()
    assert "music_style" in spec["optional"]
    assert spec["optional"]["music_style"][1]["default"] == ""
    assert not spec["optional"]["music_style"][1].get("forceInput"), (
        "it must be a WIDGET, not a wired input")
    # appended last among the widgets, which is the only safe position
    widgets = [name for name, (_t, opts) in
               list(spec["required"].items()) + list(spec["optional"].items())
               if not opts.get("forceInput")]
    assert widgets[-1] == "music_style", widgets
    # and it is carried to the palette by the one route the composers use
    body = inspect.getsource(T.StableAudioTheme._render_clips)
    assert 'meta = dict(meta, music_style=' in body


def test_no_other_module_reads_the_checkpoint_constant_as_a_filename():
    """A BUILD-BREAKER THAT SHIPPED, and the class is worth a guard.

    `eng_stable_audio_3._CKPT` used to BE the checkpoint filename. Since the
    engine began choosing between the base and post-trained checkpoints at load
    time it is the operator's OVERRIDE, and it is empty by default. One other
    module still read it as a filename -- the visual-asset preflight -- so it
    asked ComfyUI to find a weight called "" and killed every canonical render
    that did not set OTR_SA3_CKPT. The unit suite could not see it; the first
    live leg after the change died in twelve seconds.

    Anything outside the engine wanting to know which checkpoint will actually
    load must ASK (`StableAudio3Engine.resolve_ckpt()`), never read the
    constant."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "nodes"
    offenders = []
    for path in root.rglob("*.py"):
        if path.name == "eng_stable_audio_3.py":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if "sa3._CKPT" in line or "eng_stable_audio_3._CKPT" in line:
                offenders.append("%s:%d %s" % (path.name, n, line.strip()))
    assert not offenders, (
        "these read the override constant as if it were a filename:\n  %s"
        % "\n  ".join(offenders))


def test_the_visual_preflight_resolves_the_checkpoint_it_will_load():
    """The positive half: the preflight must name the file that will actually
    be loaded, so a box missing it is told before a render starts rather than
    after twenty minutes of work."""
    import inspect
    from nodes import _otr_visual_assets as VA
    body = inspect.getsource(VA.native_requests)
    assert "resolve_ckpt()" in body
    name, is_base = SA3.StableAudio3Engine.resolve_ckpt()
    assert name.endswith(".safetensors") and name, name
    assert isinstance(is_base, bool)
