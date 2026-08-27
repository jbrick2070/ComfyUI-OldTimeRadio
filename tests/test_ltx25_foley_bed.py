"""The LTX 2.5 FOLEY BED -- the stem format, the cut, the mix, and the wiring.

WHAT THIS FILE IS FOR. The foley bed is picture-locked audio travelling through
four stages that each own a different piece of it: the engine writes a stem per
rendered segment, the coverage assembler cuts and concatenates them per beat,
the manifest carries the receipts, and ``OTR_MasterAudioMux`` splats them onto
the frozen master. Every test below is about a way those four can come to
disagree -- because when they do, the failure is not an exception. It is a bed
playing under the wrong picture, in a file that looks completely normal.

NOT THE SFX BED. That was separately GENERATED effects from a dedicated model,
ripped 2026-08-06 and staying dead (``tests/test_rip_sfx_bed_guard.py``). This
is the video model's own output. Operator, 2026-08-26: *"sfx bed is different
than foley bed, i won't get the two confused."*

CPU-safe and pure: no renders, no CUDA, no model loads. Every stem here is a
handful of samples written to tmp.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pytest

from nodes import otr_master_audio_mux as MUX
from nodes import scene_sequencer as SEQ
from nodes._otr_video_engines import foley_stems as fs

FPS = 25
RATE = 48000
STEP = RATE // FPS          # 1920 samples per frame at 25 fps


def _stem(tmp_path, name, frames, value, *, rate=RATE, channels=2):
    """A constant-valued stem of exactly ``frames`` frames. Constant on purpose:
    a cut is then trivially checkable by reading the value back."""
    path = tmp_path / name
    step = rate // FPS
    fs.write_pcm16_wav(
        path, np.full((channels, frames * step), value, dtype=np.float32), rate)
    return str(path)


# ---------------------------------------------------------------------------
# The frames-to-samples conversion -- the arithmetic everything else rests on
# ---------------------------------------------------------------------------

def test_samples_per_frame_is_exact_or_it_is_a_refusal():
    """The decode's rate comes off the WEIGHTS at runtime, not from a constant,
    so it can change under us with a re-quant. Both rates seen in practice
    divide evenly by 25; one that did not would make every conversion a
    rounding, and a rounding compounds across a chained beat until the bed
    slides audibly off its own picture."""
    assert fs.samples_per_frame(48000, 25) == 1920
    assert fs.samples_per_frame(44100, 25) == 1764
    for bad in (44101, 0, -48000):
        with pytest.raises(fs.FoleyStemError):
            fs.samples_per_frame(bad, 25)
    with pytest.raises(fs.FoleyStemError):
        fs.samples_per_frame(48000, 0)


def test_a_stem_round_trips_through_the_one_writer_and_the_one_reader():
    """One format for every OTR audio artifact -- 16-bit PCM, the same as the
    episode master -- so the mux never has to ask what it is reading."""
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as tmp:
        path = pathlib.Path(tmp) / "s.wav"
        source = np.stack([np.linspace(-0.9, 0.9, 4800, dtype=np.float32),
                           np.linspace(0.9, -0.9, 4800, dtype=np.float32)])
        n, ch = fs.write_pcm16_wav(path, source, RATE)
        assert (n, ch) == (4800, 2)
        back, rate = fs.read_pcm16_wav(path)
        assert rate == RATE
        assert back.shape == (2, 4800)
        # 16-bit quantisation is the only difference permitted.
        assert np.max(np.abs(back - source)) < 1.0 / 32000


# ---------------------------------------------------------------------------
# The cut -- the coverage assembler's half
# ---------------------------------------------------------------------------

def test_the_cut_drops_the_head_and_keeps_exactly_the_visible_frames(tmp_path):
    """`drop_head` / `keep_frames` are applied HERE, in sample space, and only
    here. The engine emits a stem as long as the mp4 it wrote; cutting in both
    places would trim picture-locked audio twice -- and every chained successor
    carries `drop_head=1`, so it would be wrong on every beat over one rung
    rather than occasionally."""
    a = _stem(tmp_path, "a.wav", 10, 0.5)
    b = _stem(tmp_path, "b.wav", 10, 0.25)
    out = tmp_path / "beat.wav"
    receipts = fs.assemble_beat_foley_segments(
        [(a, 0, 10), (b, 1, 9)], out, expect_frames=19, fps=FPS)

    assert receipts["foley_samples"] == 19 * STEP
    assert receipts["foley_sample_rate"] == RATE
    assert receipts["foley_channels"] == 2
    assert receipts["foley_duration_s"] == pytest.approx(19 * STEP / RATE)
    assert receipts["foley_sha256"] == fs.sha256_of_file(out)

    arr, _rate = fs.read_pcm16_wav(out)
    # Segment A whole, then segment B with its first frame gone.
    assert np.allclose(arr[:, :10 * STEP], 0.5, atol=1e-3)
    assert np.allclose(arr[:, 10 * STEP:], 0.25, atol=1e-3)


def test_a_single_segment_beat_goes_through_the_cutter_too(tmp_path):
    """A ONE-segment plan still owes a tail trim whenever its length was
    rounded up to a legal rung. Routing only multi-segment beats through the
    cutter is how that surplus survives on exactly the beats nobody checks."""
    only = _stem(tmp_path, "only.wav", 97, 0.4)
    out = tmp_path / "beat.wav"
    receipts = fs.assemble_beat_foley_segments(
        [(only, 0, 50)], out, expect_frames=50, fps=FPS)
    assert receipts["foley_samples"] == 50 * STEP


def test_the_assembled_length_is_PROVEN_against_the_beat(tmp_path):
    """Transactional, like its video sibling: a stem that disagrees with its
    beat's frame count is not a receipt problem, it is audio that will play
    under the wrong picture."""
    a = _stem(tmp_path, "a.wav", 10, 0.5)
    out = tmp_path / "beat.wav"
    with pytest.raises(fs.FoleyStemError) as exc:
        fs.assemble_beat_foley_segments(
            [(a, 0, 10)], out, expect_frames=11, fps=FPS)
    assert "disagree" in str(exc.value)


def test_a_missing_or_short_stem_is_a_refusal_never_silence(tmp_path):
    """A bed that quietly drops a beat is a lane reporting success for work it
    did not deliver."""
    out = tmp_path / "beat.wav"
    with pytest.raises(fs.FoleyStemError):
        fs.assemble_beat_foley_segments(
            [(str(tmp_path / "nope.wav"), 0, 10)], out, expect_frames=10,
            fps=FPS)
    short = _stem(tmp_path, "short.wav", 3, 0.5)
    with pytest.raises(fs.FoleyStemError) as exc:
        fs.assemble_beat_foley_segments(
            [(short, 0, 10)], out, expect_frames=10, fps=FPS)
    assert "shorter than the picture" in str(exc.value)


def test_a_format_change_mid_beat_is_a_refusal(tmp_path):
    """Bug Bible 12.29's silent mismatch: differing rates concatenate without
    complaint and play the back half at the wrong speed."""
    a = _stem(tmp_path, "a.wav", 10, 0.5)
    b = _stem(tmp_path, "b.wav", 10, 0.5, rate=44100)
    with pytest.raises(fs.FoleyStemError) as exc:
        fs.assemble_beat_foley_segments(
            [(a, 0, 10), (b, 0, 10)], tmp_path / "beat.wav",
            expect_frames=20, fps=FPS)
    assert "one format" in str(exc.value)


# ---------------------------------------------------------------------------
# The mix -- the mux's half
# ---------------------------------------------------------------------------

def _row(path, start_s, frames, space="master_mix",
         engine="ltx25_foley_plus"):
    """One manifest row. `engine_id` is load-bearing, not decoration: it is
    what tells the mix WHICH gains this row mixes at, and the two lanes
    attenuate the master differently."""
    return {"foley_path": path, "start_s": start_s, "frame_count": frames,
            "start_s_space": space, "engine_id": engine}


def test_the_operator_ratio_is_applied_exactly_once(tmp_path):
    """0.20 foley / 0.80 master, on the FULL master -- dialogue, room tone,
    themes and cues together, because the assembler folds all of it into one
    WAV and no separate voice bus exists to duck against."""
    master = np.full((2, 100 * STEP), 0.5, dtype=np.float32)
    bed = _stem(tmp_path, "bed.wav", 100, 0.25)
    mixed, stats = fs.mix_foley_under_master(
        master, RATE, [_row(bed, 0.0, 100)], fps=FPS)
    assert stats["placed"] == 1
    assert stats["lanes"] == {"ltx25_foley_plus": 1}
    assert stats["global_master_gain"] == 0.80
    assert stats["muted_samples"] == 0
    assert np.allclose(mixed, 0.5 * 0.8 + 0.25 * 0.2, atol=1e-3)


def test_a_beat_with_no_bed_is_SILENCE_and_the_master_is_not_boosted(tmp_path):
    """Two rulings in one assertion. A beat with no stem gets silence -- never a
    neighbour's bed, which would put audio generated for one picture under
    another. And the master holds 0.80 whether or not a bed exists, so a beat
    without foley does not get louder than its neighbours."""
    master = np.full((2, 100 * STEP), 0.5, dtype=np.float32)
    bed = _stem(tmp_path, "bed.wav", 10, 0.25)
    mixed, _stats = fs.mix_foley_under_master(
        master, RATE, [_row(bed, 0.0, 10)], fps=FPS)
    assert np.allclose(mixed[:, :10 * STEP], 0.4 + 0.05, atol=1e-3)
    assert np.allclose(mixed[:, 10 * STEP:], 0.4, atol=1e-3)


def test_stems_SPLAT_at_their_own_offsets_and_OVERLAPS_add(tmp_path):
    """The manifest legitimately contains overlapping positioned rows -- the
    opening/body and body/closing seams overlap by design and the assembler
    crossfades across them -- so "each beat follows the last" is false about
    this timeline. Additive, because where two beats overlap the picture shows
    both; copy-over would silence the outgoing beat's bed instantly."""
    master = np.zeros((1, 100 * STEP), dtype=np.float32)
    a = _stem(tmp_path, "a.wav", 20, 0.5, channels=1)
    b = _stem(tmp_path, "b.wav", 20, 0.25, channels=1)
    # b starts at frame 10 -- ten frames INSIDE a.
    mixed, stats = fs.mix_foley_under_master(
        master, RATE, [_row(a, 0.0, 20), _row(b, 10 / FPS, 20)], fps=FPS)
    assert stats["placed"] == 2
    gain = fs.FOLEY_LANE_GAINS["ltx25_foley_plus"][0]
    assert np.allclose(mixed[:, :10 * STEP], 0.5 * gain, atol=1e-3)
    assert np.allclose(mixed[:, 10 * STEP:20 * STEP],
                       (0.5 + 0.25) * gain, atol=1e-3)
    assert np.allclose(mixed[:, 20 * STEP:30 * STEP], 0.25 * gain, atol=1e-3)
    assert np.allclose(mixed[:, 30 * STEP:], 0.0, atol=1e-3)


def test_the_offset_is_integer_frames_not_a_float_multiplication(tmp_path):
    """`round(start_s * fps) * (rate // fps)` lands every bed on a frame
    boundary. `int(start_s * rate)` accumulates a sub-sample error per beat with
    no defined rounding -- exactly the drift this feature would be blamed for.

    3.88 s is one rung, and 3.88 * 48000 = 186240.00000000003 in float."""
    master = np.zeros((1, 300 * STEP), dtype=np.float32)
    bed = _stem(tmp_path, "bed.wav", 10, 0.5, channels=1)
    mixed, _stats = fs.mix_foley_under_master(
        master, RATE, [_row(bed, 3.88, 10)], fps=FPS)
    edge = 97 * STEP                      # 3.88 s at 25 fps is frame 97 exactly
    assert np.allclose(mixed[:, edge - 1], 0.0, atol=1e-4)
    assert np.allclose(
        mixed[:, edge], 0.5 * fs.FOLEY_LANE_GAINS["ltx25_foley_plus"][0],
        atol=1e-3)


def test_a_row_positioned_in_the_wrong_CLOCK_is_a_refusal(tmp_path):
    """The assembler rewrites the ledger's line rows from scene_audio space to
    master_mix space once the themes are folded in. Splatting a bed at a
    scene-audio offset into a master-mix timeline puts EVERY beat's audio early
    by the length of the opening theme -- and nothing raises."""
    master = np.zeros((1, 100 * STEP), dtype=np.float32)
    bed = _stem(tmp_path, "bed.wav", 10, 0.5, channels=1)
    with pytest.raises(fs.FoleyStemError) as exc:
        fs.mix_foley_under_master(
            master, RATE, [_row(bed, 0.0, 10, space="scene_audio")], fps=FPS)
    assert "master_mix" in str(exc.value)


def test_a_stem_longer_than_its_own_slot_is_a_refusal(tmp_path):
    """A stem longer than its own picture has been cut against the wrong plan.
    Note the boundary: overflow past the NEXT row's start_s is legal (rows
    overlap); overflow past its OWN slot is not."""
    master = np.zeros((1, 100 * STEP), dtype=np.float32)
    bed = _stem(tmp_path, "bed.wav", 20, 0.5, channels=1)
    with pytest.raises(fs.FoleyStemError) as exc:
        fs.mix_foley_under_master(
            master, RATE, [_row(bed, 0.0, 10)], fps=FPS)
    assert "longer than its own picture" in str(exc.value)


def test_a_stem_the_manifest_names_but_disk_lacks_is_a_refusal(tmp_path):
    master = np.zeros((1, 100 * STEP), dtype=np.float32)
    with pytest.raises(fs.FoleyStemError) as exc:
        fs.mix_foley_under_master(
            master, RATE, [_row(str(tmp_path / "gone.wav"), 0.0, 10)], fps=FPS)
    assert "NO SILENT SKIP" in str(exc.value)


def test_a_mismatched_stem_is_conformed_EXPLICITLY_and_said_so(tmp_path):
    """Explicit conformance before mixing (Bug Bible 12.29). A silent channel
    mismatch lands the bed on one side only; a silent rate mismatch plays it at
    the wrong pitch under a picture that is still correct."""
    stem = np.full((1, 10 * (44100 // FPS)), 0.5, dtype=np.float32)
    conformed, notes = fs.conform_to_master(stem, 44100, RATE, 2)
    assert conformed.shape[0] == 2
    assert conformed.shape[-1] == pytest.approx(10 * STEP, rel=0.01)
    assert any("44100->48000" in n for n in notes)
    assert any("channels 1->2" in n for n in notes)
    # A stem that already matches is left alone and says nothing.
    same, quiet = fs.conform_to_master(
        np.zeros((2, 100), dtype=np.float32), RATE, RATE, 2)
    assert quiet == [] and same.shape == (2, 100)


def test_mime_zeroes_ONLY_its_own_windows_never_the_whole_episode(tmp_path):
    """THE RULING THAT SHAPES THIS LANE. Engines are ROLE-WIDE dropdowns, so
    `ltx25_high_mime` on the character role makes every character beat a silent
    performance -- while the announcer and music roles still speak, out of the
    SAME single master WAV. A global 0.00 would silence the episode. So mime
    zeroes its own beats' samples and leaves the rest of the timeline alone."""
    master = np.full((1, 100 * STEP), 0.5, dtype=np.float32)
    bed = _stem(tmp_path, "mime.wav", 10, 0.25, channels=1)
    mixed, stats = fs.mix_foley_under_master(
        master, RATE, [_row(bed, 10 / FPS, 10, engine="ltx25_mime")],
        fps=FPS, lane_ids={"ltx25_mime"})

    assert stats["lanes"] == {"ltx25_mime": 1}
    assert stats["global_master_gain"] == 1.0, (
        "mime must NOT attenuate the whole timeline -- that is the foley "
        "lane's global 0.80, and applying it here would duck every role")
    assert stats["muted_samples"] == 10 * STEP
    # Before the mime window: the master, untouched.
    assert np.allclose(mixed[:, :10 * STEP], 0.5, atol=1e-3)
    # Inside it: the master is GONE and the video's own audio is at full.
    assert np.allclose(mixed[:, 10 * STEP:20 * STEP], 0.25, atol=1e-3)
    # After it: the master again, untouched.
    assert np.allclose(mixed[:, 20 * STEP:], 0.5, atol=1e-3)


def test_a_mixed_episode_gives_each_lane_its_OWN_gains(tmp_path):
    """A mixed-role episode is legal -- one role on foley, another on mime --
    and the two attenuate the master differently. Reading the gains off the
    EPISODE rather than off each row would mix one lane at the other's
    balance."""
    master = np.full((1, 100 * STEP), 0.5, dtype=np.float32)
    foley = _stem(tmp_path, "f.wav", 10, 0.25, channels=1)
    mime = _stem(tmp_path, "m.wav", 10, 0.25, channels=1)
    mixed, stats = fs.mix_foley_under_master(
        master, RATE,
        [_row(foley, 0.0, 10), _row(mime, 20 / FPS, 10, engine="ltx25_mime")],
        fps=FPS, lane_ids={"ltx25_foley_plus", "ltx25_mime"})

    assert stats["lanes"] == {"ltx25_foley_plus": 1, "ltx25_mime": 1}
    # The foley lane's 0.80 is GLOBAL, so it floors the whole timeline...
    assert stats["global_master_gain"] == 0.80
    assert np.allclose(mixed[:, :10 * STEP], 0.5 * 0.8 + 0.25 * 0.2, atol=1e-3)
    assert np.allclose(mixed[:, 10 * STEP:20 * STEP], 0.5 * 0.8, atol=1e-3)
    # ...and the mime window still goes to zero on top of it, at full foley.
    assert np.allclose(mixed[:, 20 * STEP:30 * STEP], 0.25, atol=1e-3)
    assert np.allclose(mixed[:, 30 * STEP:], 0.5 * 0.8, atol=1e-3)


def test_the_global_080_applies_to_beats_with_no_bed_at_all(tmp_path):
    """RULING 1, and it is why `lane_ids` exists as a parameter at all: "voice
    holds 0.80 whether or not a foley stem exists for that beat, so a beat
    without foley does not get louder". An episode whose foley role rendered
    NO beats still gets the 0.80 floor -- which cannot be inferred from the
    rows, because there are none."""
    master = np.full((1, 100 * STEP), 0.5, dtype=np.float32)
    mixed, stats = fs.mix_foley_under_master(
        master, RATE, [], fps=FPS, lane_ids={"ltx25_foley_plus"})
    assert stats["placed"] == 0
    assert stats["global_master_gain"] == 0.80
    assert np.allclose(mixed, 0.4, atol=1e-3)


def test_an_UNPOSITIONED_beat_is_skipped_loudly_not_fatal(tmp_path):
    """THE LIVE LEG THAT KILLED AN EPISODE (2026-08-26), pinned.

    This raised FoleyStemError until a canonical leg died on it at the very
    last node -- 3h17m of render lost, nothing published. A `music_inter` beat
    is a video-only bridge: ledger b006 read `start_s=None, dur_s=None,
    text=''`, and its neighbours b005 (33.936 + 3.499) and b007 (37.435) were
    exactly contiguous, so it owned NO time in the master mix at all. It still
    rendered a picture, and on a foley lane it still produced a stem.

    A beat with no window cannot carry a bed. Skipping is the only honest
    answer -- and skipping is NOT the position-guessing the old guard rightly
    forbade. What the guard was really protecting against is a bed silently
    vanishing, so the skip is LOUD and COUNTED instead."""
    master = np.full((1, 100 * STEP), 0.5, dtype=np.float32)
    placed_stem = _stem(tmp_path, "ok.wav", 10, 0.25, channels=1)
    orphan = _stem(tmp_path, "orphan.wav", 10, 0.25, channels=1)
    rows = [
        _row(placed_stem, 0.0, 10),
        # Exactly what build_clip_manifest emits for an unpositioned line.
        {"foley_path": orphan, "start_s": None, "frame_count": 10,
         "start_s_space": "master_mix", "engine_id": "ltx25_foley_plus"},
    ]
    mixed, stats = fs.mix_foley_under_master(master, RATE, rows, fps=FPS)

    assert stats["placed"] == 1
    assert stats["unpositioned"] == 1, (
        "an unpositioned beat must be COUNTED -- a bed missing from half an "
        "episode has to be visible in the receipt, not just a log line")
    # The positioned bed still landed; the episode still ships.
    assert np.allclose(mixed[:, :10 * STEP], 0.5 * 0.8 + 0.25 * 0.2, atol=1e-3)
    assert np.allclose(mixed[:, 10 * STEP:], 0.5 * 0.8, atol=1e-3)


def test_every_beat_unpositioned_still_delivers_a_master(tmp_path):
    """The degenerate case must not become the old hard failure by another
    route: if NOTHING can be placed, the master is still returned (scaled and
    then normalised downstream), because the episode is not the bed's hostage."""
    master = np.full((1, 50 * STEP), 0.5, dtype=np.float32)
    orphan = _stem(tmp_path, "orphan.wav", 10, 0.25, channels=1)
    mixed, stats = fs.mix_foley_under_master(
        master, RATE,
        [{"foley_path": orphan, "start_s": None, "frame_count": 10,
          "engine_id": "ltx25_foley_plus"}], fps=FPS)
    assert (stats["placed"], stats["unpositioned"]) == (0, 1)
    assert np.allclose(mixed, 0.5 * 0.8, atol=1e-3)


def test_a_stem_from_a_lane_with_no_gains_is_a_refusal(tmp_path):
    """Mixing a stem at gains chosen for a different lane is how a bed ends up
    over or under the dialogue it was balanced against. NO GUESS."""
    master = np.zeros((1, 100 * STEP), dtype=np.float32)
    bed = _stem(tmp_path, "bed.wav", 10, 0.5, channels=1)
    with pytest.raises(fs.FoleyStemError) as exc:
        fs.mix_foley_under_master(
            master, RATE, [_row(bed, 0.0, 10, engine="wan_ti2v")], fps=FPS)
    assert "not an audio-keeping lane" in str(exc.value)


# ---------------------------------------------------------------------------
# The route -- ONE answer, read by two nodes
# ---------------------------------------------------------------------------

def test_the_route_is_decided_by_ONE_function_for_both_audio_stages():
    """The assembler decides which flavour of master WAV to write and the mux
    decides mix-vs-copy. Two different answers means either a double loudness
    gain or an un-levelled deliverable, and neither is visible in any log."""
    def policy(**roles):
        return json.dumps({"effective_video_models": roles})

    assert fs.is_foley_route(policy(music="ltx25_foley_plus"))
    # BOTH audio-keeping lanes put the episode on the route. They mix
    # differently, but they need the same provisional master to mix INTO.
    assert fs.is_foley_route(policy(character="ltx25_mime"))
    assert fs.is_foley_route(policy(character="ltx25_high_mime (16:9)"))
    assert fs.route_lane_ids(
        policy(character="ltx25_high_mime", music="ltx25_high_foley_plus")
    ) == {"ltx25_mime", "ltx25_foley_plus"}
    assert fs.route_lane_ids(policy(music="ltx25_video")) == frozenset()
    # PUBLIC ids and display suffixes resolve too -- a bare == would answer
    # False for an episode that really is on the route.
    assert fs.is_foley_route(policy(music="ltx25_high_foley_plus"))
    assert fs.is_foley_route(policy(music="ltx25_high_foley_plus (16:9)"))
    # ANY role, not all: the episode has ONE master WAV, so one foley role puts
    # the whole thing on the route.
    assert fs.is_foley_route(
        policy(announcer="ltx25_video", music="ltx25_foley_plus",
               character="still_flat"))
    assert not fs.is_foley_route(policy(music="ltx25_video"))
    # Total on garbage: absent policy means the historical path, which is what
    # every existing episode expects.
    for junk in ("", None, "{", "[]", json.dumps({"effective_video_models": 3})):
        assert fs.is_foley_route(junk) is False

    # And the mux really uses it, rather than a second copy of the question.
    assert MUX._foley_route(policy(music="ltx25_foley_plus")) is True
    assert MUX._foley_route(policy(music="ltx25_video")) is False


# ---------------------------------------------------------------------------
# The wiring -- the two node surfaces
# ---------------------------------------------------------------------------

def test_the_mux_connectors_are_APPENDED_and_the_tripwire_is_untouched():
    """BUG-LOCAL-097: widgets_values is positional, so a new input goes at the
    END. And the operator's decision: the foley receipts ride their OWN
    connector so `clip_manifest_json` stays the deliberate tripwire it is --
    "accepted, hashed, unused", never a use invented for it."""
    optional = MUX.OTRMasterAudioMux.INPUT_TYPES()["optional"]
    names = list(optional)
    assert names[-2:] == ["video_policy_json", "foley_receipts_json"]
    for name in ("video_policy_json", "foley_receipts_json"):
        assert optional[name][0] == "STRING"
        assert optional[name][1]["forceInput"] is True

    # UNCHANGED, and this is the assertion the whole option-(b) decision buys.
    retired = optional["clip_manifest_json"]
    assert "retired" in retired[1]["tooltip"].lower()
    params = set(inspect.signature(MUX.OTRMasterAudioMux.mux).parameters)
    assert {"video_policy_json", "foley_receipts_json"} <= params
    assert "clip_manifest_json" in params
    # STILL UNUSED, and this is the half a signature check cannot see. The
    # parameter is accepted and hashed; the body must never read it. `mux()`
    # binds it and mentions it only in the comment that says it is retired.
    body = inspect.getsource(MUX.OTRMasterAudioMux.mux)
    body = body[body.index("):") + 2:]
    reads = [line for line in body.splitlines()
             if "clip_manifest_json" in line and not line.strip().startswith("#")]
    assert reads == [], reads


def test_the_assembler_connector_is_APPENDED_and_defaults_to_the_old_path():
    """Absent / empty -> the historical behaviour, byte for byte. Every
    non-foley episode must be unaffected by this build."""
    optional = SEQ.EpisodeAssembler.INPUT_TYPES()["optional"]
    assert list(optional)[-1] == "video_policy_json"
    assert optional["video_policy_json"][1]["forceInput"] is True
    sig = inspect.signature(SEQ.EpisodeAssembler.assemble)
    assert sig.parameters["video_policy_json"].default == ""


def test_the_canonical_workflow_wires_all_three_new_links():
    """CLAUDE.md section 0: code that is not wired into the canonical JSON is
    DEAD. The policy reaches BOTH audio stages from ONE source, and the clip
    manifest reaches the mux on the new connector -- while link 278 keeps
    feeding the retired one."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[1]
    doc = json.loads(
        (root / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in doc["nodes"]}
    assert by_id[7]["type"] == "OTR_EpisodeAssembler"
    assert by_id[85]["type"] == "OTR_MasterAudioMux"

    def slot(node_id, name):
        node = by_id[node_id]
        return next((i for i, inp in enumerate(node["inputs"])
                     if inp.get("name") == name), None)

    wired = {(l[1], l[3], l[4]) for l in doc["links"]}
    assert (87, 7, slot(7, "video_policy_json")) in wired
    assert (87, 85, slot(85, "video_policy_json")) in wired
    assert (92, 85, slot(85, "foley_receipts_json")) in wired
    # APPENDED, not inserted: each new input is the last on its node.
    assert slot(7, "video_policy_json") == len(by_id[7]["inputs"]) - 1
    assert slot(85, "foley_receipts_json") == len(by_id[85]["inputs"]) - 1
    # The retired connector is still fed, and still by link 278.
    assert any(l[0] == 278 and l[3] == 85 for l in doc["links"])
