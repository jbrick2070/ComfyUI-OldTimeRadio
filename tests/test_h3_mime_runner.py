"""LANE 21 -- the `h3_low_mime` standalone runner, and it INVERTS the audio law.

Every other H3 path in this repo proves SILENCE. Lanes 19 and 20 never decode
the audio latent and `canonicalize` ffprobes their file to prove
`has_audio: False`. This runner is the G5.2 exemption: the clip's invented score
IS its audio. So the tests below assert the OPPOSITE of the rest of the video
suite, and they assert WHY it is not a V-1 violation -- these outputs never enter
episode assembly.

The runner is a script, not an engine, so the CPU-testable surface is its pure
half: the seconds->grid snap, the graph it builds, the mux command, and the
not-registered invariant. The live half (submit, probe, receipt) is proved by
the lane's solo smoke.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_shared import public_engines as pub
from nodes._otr_video_engines import registry as vreg


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNNER_PATH = os.path.join(REPO_ROOT, "scripts", "otr_h3_mime_runner.py")


def _runner():
    """Import the runner by PATH -- it lives in scripts/, not a package."""
    spec = importlib.util.spec_from_file_location("otr_h3_mime_runner",
                                                  RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def runner():
    return _runner()


# ---------------------------------------------------------------------------
# NOT REGISTERED -- the invariant the whole lane shape rests on
# ---------------------------------------------------------------------------
def test_the_mime_lane_is_NOT_in_the_video_registry():
    """r3 proved a registered engine cannot be kept OUT of the episode
    dropdowns: role_compat grants every role the full input vocabulary and the
    dropdown IS the registry. A mime engine in the registry would immediately be
    selectable for ordinary beats -- and a mime beat's audio does not exist
    until its video renders, which the audio-first master freeze cannot absorb
    without a phase inversion that is a LATER spec.
    """
    assert "minimax_h3_mime" not in vreg.all_engine_names()
    assert "h3_low_mime" not in pub._PUBLIC_ENGINES
    assert "minimax_h3_mime" not in vreg.CAPABILITIES
    # And no registered engine answers to the mime id by any alias route.
    assert pub.resolve_engine_id("h3_low_mime") not in vreg.all_engine_names()


def test_importing_the_runner_REGISTERS_NOTHING():
    """Structural, and the first draft of this test was not.

    It grepped the runner's source for `@register` and failed -- on the COMMENT
    explaining that the decorators live on the adapter classes rather than here.
    That is L26 biting for the fourth time in one session, in the test written
    minutes after the lesson. So: import the runner and prove the registry and
    the public menu are byte-identical across it. A decorator that did fire
    would move one of those sets; a comment cannot.
    """
    before = (sorted(vreg.all_engine_names()), sorted(pub._PUBLIC_ENGINES))
    mod = _runner()
    after = (sorted(vreg.all_engine_names()), sorted(pub._PUBLIC_ENGINES))
    assert before == after
    assert os.path.isfile(RUNNER_PATH)
    # It is a script: it has a main() and an argument parser, not a node class.
    assert callable(mod.main)
    assert not [v for v in vars(mod).values()
                if isinstance(v, type) and hasattr(v, "INPUT_TYPES")]


# ---------------------------------------------------------------------------
# The grid snap
# ---------------------------------------------------------------------------
def test_seconds_snap_UP_onto_the_models_own_grid(runner):
    """The snap goes through `align_frame_count`, the node's own rule, so the
    length handed to the graph is one the node will not silently move."""
    for seconds in (0.5, 1.0, 3.75, 5.0, 8.0, 15.0):
        n = runner.model_frames_for_seconds(seconds)
        assert n % 17 == 5, (seconds, n)
        assert runner.delivered_seconds(n) >= seconds - 1e-9


def test_the_lab_mime_length_reproduces_exactly(runner):
    """The two passing lab legs are model f90 = 3.750 s at 24 fps."""
    assert runner.model_frames_for_seconds(3.75) == 90
    assert runner.delivered_seconds(90) == 3.75


def test_a_nonpositive_target_is_refused(runner):
    for bad in (0, -1, -0.5):
        with pytest.raises(ValueError):
            runner.model_frames_for_seconds(bad)


def test_the_runner_accepts_BELOW_the_trained_band_and_says_so(runner):
    """The registered lanes publish only the trained band because a beat that
    renders below it renders badly. A mime clip is a short authored moment, not
    a beat -- and the lab's two passing legs are f90, below that floor."""
    assert runner.below_trained_band(90) is True
    assert runner.below_trained_band(124) is False
    assert runner.below_trained_band(192) is False


# ---------------------------------------------------------------------------
# THE AUDIO, which is the whole lane
# ---------------------------------------------------------------------------
def test_the_graph_DECODES_AND_KEEPS_the_audio(runner):
    """The inversion, stated as structure: the registered lanes have no
    `VAEDecodeAudio` at all; this graph has one, wired into the container."""
    g = runner.build_mime_graph(
        image_name="p.png", prompt="x", model_frames=90, seed=42,
        width=864, height=480, filename_prefix="pfx")
    classes = {n["class_type"] for n in g.values()}
    assert "VAEDecodeAudio" in classes
    assert "CreateVideo" in classes and "SaveVideo" in classes
    # The audio VAE feeds the audio decode, and that decode feeds the container.
    audio_decode = [k for k, n in g.items()
                    if n["class_type"] == "VAEDecodeAudio"][0]
    create = [k for k, n in g.items() if n["class_type"] == "CreateVideo"][0]
    assert g[create]["inputs"]["audio"] == [audio_decode, 0]
    # ... off the SAME sampled latent the picture comes from, so the score and
    # the frames are the same generation rather than two unrelated ones.
    sample = [k for k, n in g.items()
              if n["class_type"] == "SamplerCustomAdvanced"][0]
    video_decode = [k for k, n in g.items() if n["class_type"] == "VAEDecode"][0]
    assert g[audio_decode]["inputs"]["samples"] == [sample, 0]
    assert g[video_decode]["inputs"]["samples"] == [sample, 0]


def test_both_vaes_are_loaded_and_they_are_DIFFERENT_files(runner):
    g = runner.build_mime_graph(
        image_name="p.png", prompt="x", model_frames=90, seed=42,
        width=864, height=480, filename_prefix="pfx")
    vaes = [n["inputs"]["vae_name"] for n in g.values()
            if n["class_type"] == "VAELoader"]
    assert len(vaes) == 2 and len(set(vaes)) == 2
    assert runner.MIME_AUDIO_VAE in vaes and runner.MIME_VAE in vaes


def test_delivery_is_at_the_MODELS_rate_and_is_never_converted(runner):
    """The one place in the build that does NOT convert to the 25 fps canvas.

    Lanes 19/20 remap to 25 because they hand clips to a 25 fps timeline. This
    clip carries its own generated score and never reaches that timeline, so
    resampling the picture would desync it from the music it was scored for.
    """
    g = runner.build_mime_graph(
        image_name="p.png", prompt="x", model_frames=90, seed=42,
        width=864, height=480, filename_prefix="pfx")
    create = [n for n in g.values() if n["class_type"] == "CreateVideo"][0]
    assert create["inputs"]["fps"] == float(runner.H3_MODEL_FPS) == 24.0
    assert create["inputs"]["fps"] != float(runner.H3_CANVAS_FPS)


# ---------------------------------------------------------------------------
# The graph otherwise matches the lanes it shares a model with
# ---------------------------------------------------------------------------
def test_the_frozen_recipe_is_the_SAME_one_the_registered_lanes_render_with(
        runner):
    """Imported from the adapter module rather than re-typed, so the runner and
    the lanes cannot drift into two H3 recipes."""
    from nodes._otr_video_engines import eng_minimax_h3 as h3

    g = runner.build_mime_graph(
        image_name="p.png", prompt="x", model_frames=90, seed=7,
        width=864, height=480, filename_prefix="pfx")
    sched = [n for n in g.values() if n["class_type"] == "BasicScheduler"][0]
    sampler = [n for n in g.values() if n["class_type"] == "KSamplerSelect"][0]
    assert sched["inputs"]["steps"] == h3.H3_RECIPE["steps"] == 20
    assert sched["inputs"]["scheduler"] == h3.H3_RECIPE["scheduler"]
    assert sampler["inputs"]["sampler_name"] == h3.H3_RECIPE["sampler"]
    assert runner.MIME_UNET == h3._H3_DEFAULT_UNET      # the fl2va DiT


def test_the_seed_and_length_reach_the_graph(runner):
    for seed, frames in ((42, 90), (43, 124), (7, 192)):
        g = runner.build_mime_graph(
            image_name="p.png", prompt="x", model_frames=frames, seed=seed,
            width=864, height=480, filename_prefix="pfx")
        noise = [n for n in g.values() if n["class_type"] == "RandomNoise"][0]
        h3n = [n for n in g.values()
               if n["class_type"] == "MiniMaxH3ImageToVideo"][0]
        assert noise["inputs"]["noise_seed"] == seed
        assert h3n["inputs"]["length"] == frames


# ---------------------------------------------------------------------------
# --stem-wav and --mux-tts
# ---------------------------------------------------------------------------
def test_the_stem_is_OPTIONAL_and_costs_no_second_decode(runner):
    without = runner.build_mime_graph(
        image_name="p.png", prompt="x", model_frames=90, seed=42,
        width=864, height=480, filename_prefix="pfx")
    assert not [n for n in without.values()
                if n["class_type"] == runner.STEM_NODE_CLASS]

    with_stem = runner.build_mime_graph(
        image_name="p.png", prompt="x", model_frames=90, seed=42,
        width=864, height=480, filename_prefix="pfx", stem_prefix="pfx_score")
    stem = [n for n in with_stem.values()
            if n["class_type"] == runner.STEM_NODE_CLASS]
    assert len(stem) == 1
    # It REUSES the existing decode rather than adding another.
    audio_decode = [k for k, n in with_stem.items()
                    if n["class_type"] == "VAEDecodeAudio"][0]
    assert stem[0]["inputs"]["audio"] == [audio_decode, 0]
    assert len([n for n in with_stem.values()
                if n["class_type"] == "VAEDecodeAudio"]) == 1
    # Lossless, because the stem is meant to be reusable material -- and the
    # FLAT SELECTOR STRING, because this is an API graph. `format` is a
    # DynamicCombo: the executor expands the selector (plus any dotted option
    # sub-inputs) into the dict `execute` receives, so handing it that dict
    # directly fails live with "missing 1 required positional argument".
    # Caught by the lane's first live attempt; pinned here so the in-process
    # shape cannot come back.
    assert stem[0]["inputs"]["format"] == "flac"
    assert not isinstance(stem[0]["inputs"]["format"], dict)


def test_the_review_mix_writes_a_THIRD_file_and_touches_neither_original(
        runner):
    cmd = runner.ducked_mux_cmd("clip.mp4", "voice.wav", "review.mp4")
    assert cmd[-1] == "review.mp4"
    assert "clip.mp4" in cmd and "voice.wav" in cmd
    # The picture is copied, never re-encoded -- a review copy must not change
    # what it is being used to review.
    assert "copy" in cmd
    joined = " ".join(cmd)
    assert "volume=" in joined and "amix" in joined      # music ducked under
    assert "-y" in cmd


def test_the_mux_ducks_the_music_DOWN(runner):
    """A positive gain would push the score OVER the voice, which is the
    opposite of the flag's purpose."""
    joined = " ".join(runner.ducked_mux_cmd("c.mp4", "v.wav", "o.mp4"))
    import re
    m = re.search(r"volume=(-?\d+\.\d+)dB", joined)
    assert m, joined
    assert float(m.group(1)) < 0


# ---------------------------------------------------------------------------
# Deliverables land where deliverables live
# ---------------------------------------------------------------------------
def test_the_default_output_is_a_DURABLE_path_not_tmp(runner):
    """CLAUDE.md section 6: point a render at its canonical path the FIRST
    time; an asset staged in tmp 'to move later' is where work dies."""
    out = runner.DEFAULT_OUT_DIR.replace("\\", "/").lower()
    assert "/otr/episodes/" in out
    for forbidden in ("/tmp", "temp", "scratch", "_shared/tmp"):
        assert forbidden not in out


def test_the_review_mix_NEVER_raises_and_reports_its_failure(runner, tmp_path):
    """A failing optional extra must not take down a deliverable that already
    succeeded -- BEHAVIOURALLY, by making it actually fail.

    THE DEFECT, caught by post-coding QA: `--mux-tts` ran AFTER the clip had
    been validated and copied to its durable path but BEFORE the receipt was
    written, with a bare `subprocess.check_call`. A missing ffmpeg or a
    malformed voice raised, and a valid audio-carrying clip was left on disk
    with NO receipt beside it -- silently breaking the one thing this runner's
    docstring promises.
    """
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    out = tmp_path / "review.mp4"

    # A voice file that does not exist: ffmpeg fails, and the helper must not.
    path, err = runner.write_review_mix(str(clip), str(tmp_path / "nope.wav"),
                                        str(out))
    assert path == ""
    assert err, "the failure must be REPORTED, not swallowed"
    assert not out.exists()


def test_a_failed_review_mix_changes_the_exit_code(runner):
    """It must not fail silently either: the operator asked for a review copy
    and did not get one, so main() returns a distinct non-zero code."""
    import inspect
    src = inspect.getsource(runner.main)
    # The receipt records it ...
    assert "review_mix_error" in src
    # ... and the exit code is 2, distinct from both 0 and the hard failures' 1.
    assert "return 2" in src


def test_a_deliverable_is_never_SILENTLY_overwritten(runner, tmp_path):
    """Behavioural: create each deliverable in turn and prove it is DETECTED.

    The default name carries the seed and a one-SECOND timestamp, so two runs
    with the same seed inside one second collide -- and an explicit --name
    collides on every repeat. These are outputs the operator keeps.
    """
    name = "collide_me"
    assert runner.existing_deliverables(str(tmp_path), name) == []
    for suffix in (".mp4", ".flac", ".receipt.json"):
        (tmp_path / (name + suffix)).write_text("x", encoding="utf-8")
        found = runner.existing_deliverables(str(tmp_path), name)
        assert any(p.endswith(suffix) for p in found), suffix
    assert len(runner.existing_deliverables(str(tmp_path), name)) == 3
    # A DIFFERENT name is unaffected -- the check is per-run, not per-directory.
    assert runner.existing_deliverables(str(tmp_path), "something_else") == []


def test_more_than_one_video_output_is_REFUSED_not_silently_narrowed(runner):
    """This graph has exactly one SaveVideo. Taking [0] from a longer list would
    deliver one file and discard the rest with no log line."""
    import inspect
    assert "len(videos) != 1" in inspect.getsource(runner.main)


def test_the_runner_names_every_node_class_it_submits(runner):
    """Preflight collects EVERY miss before raising (lane 8's rule), so the
    declared list must actually cover the graph it builds."""
    g = runner.build_mime_graph(
        image_name="p.png", prompt="x", model_frames=90, seed=42,
        width=864, height=480, filename_prefix="pfx", stem_prefix="s")
    used = {n["class_type"] for n in g.values()}
    declared = set(runner.REQUIRED_NODE_CLASSES) | {runner.STEM_NODE_CLASS}
    assert used <= declared, used - declared
