"""CW-4 -- the NEW render path (OTR_SilentComposite + terminal OTR_MasterAudioMux)
+ the cheap radio-floor families. CPU tests.

The audio-critical assertions use real ffmpeg on SYNTHESIZED media (sine master +
black silent video) -- the same headless approach the A-S2 mux probe used -- so
the mux byte-identity (V-1 / C7) and the silent-composite 0-audio invariants are
proven without a GPU. The live end-to-end episode render is an INTERACTIVE
ComfyUI smoke (operator), not covered here.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import pathlib

import pytest

from nodes.otr_master_audio_mux import mux_master_audio, audio_pcm_sha, OTRMasterAudioMux
from nodes.otr_silent_composite import (
    normalize_to_silent_canonical, count_audio_streams, probe_video, OTRSilentComposite,
)
from nodes._otr_video_engines import registry as vreg
from nodes._otr_shared import role_compat as rc

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
HAVE_FFMPEG = bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))
needs_ffmpeg = pytest.mark.skipif(not HAVE_FFMPEG, reason="ffmpeg/ffprobe required")


def _ff(*args):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args],
                   check=True, capture_output=True)


def _sine(path, dur=2.0):
    _ff("-f", "lavfi", "-i", f"sine=frequency=440:duration={dur}",
        "-ar", "24000", "-ac", "1", str(path))


def _silent_video(path, dur=2.0, with_audio=False):
    args = ["-f", "lavfi", "-i", f"color=c=black:s=64x64:d={dur}", "-r", "25"]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=300:duration={dur}"]
    args += ["-pix_fmt", "yuv420p", "-t", f"{dur}"]
    if not with_audio:
        args += ["-an"]
    args += [str(path)]
    _ff(*args)


# --------------------------------------------------------------------------- #
# OTR_MasterAudioMux -- the terminal, audio-critical node
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_master_audio_mux_stream_hash_byte_identical(tmp_path):
    master = tmp_path / "master.wav"
    silent = tmp_path / "silent.mp4"
    out = tmp_path / "final.mkv"
    _sine(master, 2.0)
    _silent_video(silent, 2.0)
    final, report = mux_master_audio(str(silent), str(master), str(out), fps=25)
    assert os.path.isfile(final)
    # the output audio is byte-identical to the frozen master (no re-encode)
    assert audio_pcm_sha(final) == audio_pcm_sha(str(master))
    # and the final actually carries exactly one audio stream
    assert count_audio_streams(final) == 1


@needs_ffmpeg
def test_master_audio_mux_no_shortest_and_duration_guard(tmp_path):
    master = tmp_path / "m.wav"
    silent = tmp_path / "s.mp4"
    _sine(master, 2.0)
    _silent_video(silent, 3.0)   # 1s longer than the audio -> must fail closed
    with pytest.raises(ValueError):
        mux_master_audio(str(silent), str(master), str(tmp_path / "o.mkv"), fps=25)
    # the node never silently truncates (no -shortest); the source proves it -- the
    # only '-shortest' token in the mux module is the V-2 guard assert, never a cmd arg.
    src = (REPO_ROOT / "nodes" / "otr_master_audio_mux.py").read_text(encoding="utf-8")
    assert 'assert "-shortest" not in cmd' in src


def test_master_audio_mux_missing_inputs_fail_closed(tmp_path):
    # missing files -> ValueError (never a half-muxed episode)
    with pytest.raises(ValueError):
        mux_master_audio(str(tmp_path / "nope.mp4"), str(tmp_path / "nope.wav"),
                         str(tmp_path / "o.mkv"))


def test_master_audio_mux_is_output_node():
    assert OTRMasterAudioMux.OUTPUT_NODE is True
    it = OTRMasterAudioMux.INPUT_TYPES()
    assert "silent_video_path" in it["required"] and "master_audio_path" in it["required"]
    assert "audio_done" in it["optional"]            # gate mirrors audio_done


# --------------------------------------------------------------------------- #
# OTR_SilentComposite -- always-silent canonical output (V-1)
# --------------------------------------------------------------------------- #
@needs_ffmpeg
def test_silent_composite_strips_audio_and_is_canonical(tmp_path):
    base = tmp_path / "base.mp4"
    out = tmp_path / "silent.mp4"
    _silent_video(base, 2.0, with_audio=True)        # base HAS audio
    assert count_audio_streams(str(base)) == 1
    silent, report = normalize_to_silent_canonical(
        str(base), str(out), w=320, h=240, fps=25,
    )
    # V-1: the composite is ALWAYS silent
    assert count_audio_streams(silent) == 0
    info = probe_video(silent)
    assert info.get("pix_fmt") == "yuv420p"
    assert int(info["width"]) % 2 == 0 and int(info["height"]) % 2 == 0


@needs_ffmpeg
def test_post_chain_zero_audio_then_mux_roundtrip(tmp_path):
    """End-to-end (CPU): base(+audio) -> SilentComposite (0 audio) -> MasterAudioMux
    -> final whose audio is byte-identical to the frozen master."""
    base = tmp_path / "base.mp4"
    silent = tmp_path / "silent.mp4"
    final = tmp_path / "final.mkv"
    _silent_video(base, 2.0, with_audio=True)
    normalize_to_silent_canonical(str(base), str(silent), w=320, h=240, fps=25)
    assert count_audio_streams(str(silent)) == 0      # post-chain zero audio

    # synth the master to the composite's ACTUAL duration so the 1/fps gate holds
    from nodes.otr_master_audio_mux import _probe_float
    dur = _probe_float(str(silent), "v:0")
    master = tmp_path / "master.wav"
    _sine(master, round(dur, 3))
    out, _r = mux_master_audio(str(silent), str(master), str(final), fps=25)
    assert count_audio_streams(out) == 1
    assert audio_pcm_sha(out) == audio_pcm_sha(str(master))


# --------------------------------------------------------------------------- #
# cheap radio-floor families -- registered, model-agnostic
# --------------------------------------------------------------------------- #
def test_cheap_families_registered():
    names = set(vreg.all_engine_names())
    for fam in ("abstract", "still_kenburns", "station_card", "visualizer", "flux_still"):
        assert fam in names, f"cheap family {fam} not registered in the video registry"


def test_cheap_family_usable_and_role_filtered():
    # abstract serves background_abstract and is usable there (no opt-in flag)
    assert vreg.assert_usable("abstract", "background_abstract") == "abstract"
    # the SHARED role filter (AS-1) offers cheap families per their roles
    descs = [
        {"engine_id": n, "roles": tuple(vreg.get_engine(n).roles),
         "required_inputs": tuple(getattr(vreg.get_engine(n), "required_inputs", ()))}
        for n in vreg.all_engine_names()
    ]
    bg = rc.filter_engines_for_role("background_abstract", descs)
    assert "abstract" in bg and "visualizer" in bg
    # incompatible role fails closed
    with pytest.raises(vreg.EngineUnusable):
        vreg.assert_usable("visualizer", "character_video")  # not in visualizer.roles


def test_cheap_families_cold_import_clean():
    """The render namespace + the new render nodes import NO heavy lib (V-12)."""
    code = (
        "import sys;"
        "import nodes._otr_video_engines;"            # triggers cheap_families register
        "import nodes._otr_video_engines.cheap_families;"
        "import nodes.otr_silent_composite;"
        "import nodes.otr_master_audio_mux;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy); sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_new_render_nodes_have_no_shortest_cmd():
    """Neither new render node passes -shortest to ffmpeg. The token may appear
    ONLY in MasterAudioMux's V-2 guard assert/comment, never in a cmd list."""
    comp = (REPO_ROOT / "nodes" / "otr_silent_composite.py").read_text(encoding="utf-8")
    # composite builds a cmd list; -shortest must not be an element of it
    assert '"-shortest"' not in comp.replace('assert "-shortest" not in cmd', "")
