"""The shared ffprobe boundary -- `nodes/_otr_shared/ffprobe.py` (lean-mean order 8).

The boundary owns exactly two facts about this BOX: which ffprobe binary to run,
and how to read a rational frame-rate string. Every FAILURE POLICY stays with
its caller, which is what separates this from the cancelled order-7
consolidation, so nothing here asserts what a failed probe COSTS -- only that
the tool is found, launched, and read correctly.

The resolution-order tests are the point of the file. The bug order 8 exists to
kill is that a bare literal ``"ffprobe"``, written as a default in a caller's
own signature, used to out-rank the operator's ``OTR_FFPROBE`` pin at every
call site in the pack.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_shared import ffprobe as ffp  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return str(path)


@pytest.fixture()
def clean_env(monkeypatch):
    """No OTR pins and nothing on PATH -- every test opts its own pieces back in."""
    monkeypatch.delenv("OTR_FFPROBE", raising=False)
    monkeypatch.delenv("OTR_FFMPEG", raising=False)
    monkeypatch.setattr(ffp.shutil, "which", lambda name: None)
    return monkeypatch


def _which_map(monkeypatch, mapping):
    monkeypatch.setattr(ffp.shutil, "which", lambda name: mapping.get(name))


# --------------------------------------------------------------------------- #
# resolve_ffprobe -- the order, and the bare-name rule that motivates order 8
# --------------------------------------------------------------------------- #
def test_resolves_to_none_when_the_box_has_nothing(clean_env):
    assert ffp.resolve_ffprobe() is None
    assert ffp.resolve_ffprobe("ffprobe") is None


def test_a_bare_default_name_is_not_a_preference(clean_env, tmp_path):
    """THE ORDER-8 BUG: `ffprobe="ffprobe"` is a signature default, not a choice."""
    pinned = _touch(tmp_path / "pinned" / "ffprobe.exe")
    clean_env.setenv("OTR_FFPROBE", pinned)
    assert ffp.resolve_ffprobe("ffprobe") == pinned
    assert ffp.resolve_ffprobe("ffprobe.exe") == pinned
    assert ffp.resolve_ffprobe("  ") == pinned
    assert ffp.resolve_ffprobe(None) == pinned


def test_a_real_path_outranks_the_operator_pin(clean_env, tmp_path):
    """A full path whose BASENAME is bare is still a choice -- the directory says so."""
    pinned = _touch(tmp_path / "pinned" / "ffprobe.exe")
    chosen = _touch(tmp_path / "chosen" / "ffprobe.exe")
    clean_env.setenv("OTR_FFPROBE", pinned)
    assert ffp.resolve_ffprobe(chosen) == chosen


def test_a_caller_supplied_ffmpeg_offers_its_sibling(clean_env, tmp_path):
    probe = _touch(tmp_path / "bin" / "ffprobe.exe")
    ffmpeg = _touch(tmp_path / "bin" / "ffmpeg.exe")
    clean_env.setenv("OTR_FFPROBE", _touch(tmp_path / "pinned" / "ffprobe.exe"))
    assert ffp.resolve_ffprobe(ffmpeg=ffmpeg) == probe


def test_a_bare_caller_ffmpeg_is_not_a_preference_either(clean_env, tmp_path):
    pinned = _touch(tmp_path / "pinned" / "ffprobe.exe")
    clean_env.setenv("OTR_FFPROBE", pinned)
    assert ffp.resolve_ffprobe(ffmpeg="ffmpeg") == pinned


def test_path_ffprobe_beats_the_ffmpeg_sibling_guesses(clean_env, tmp_path):
    """A normal install keeps resolving exactly as it always did."""
    on_path = _touch(tmp_path / "path" / "ffprobe.exe")
    _touch(tmp_path / "cfg" / "ffprobe.exe")
    clean_env.setenv("OTR_FFMPEG", _touch(tmp_path / "cfg" / "ffmpeg.exe"))
    _which_map(clean_env, {"ffprobe": on_path})
    assert ffp.resolve_ffprobe() == on_path


def test_the_configured_ffmpeg_sibling_rescues_a_box_without_ffprobe_on_path(
        clean_env, tmp_path):
    sibling = _touch(tmp_path / "cfg" / "ffprobe.exe")
    clean_env.setenv("OTR_FFMPEG", _touch(tmp_path / "cfg" / "ffmpeg.exe"))
    assert ffp.resolve_ffprobe() == sibling


def test_the_path_ffmpeg_sibling_is_the_last_resort(clean_env, tmp_path):
    sibling = _touch(tmp_path / "bin" / "ffprobe.exe")
    ffmpeg = _touch(tmp_path / "bin" / "ffmpeg.exe")
    _which_map(clean_env, {"ffmpeg": ffmpeg})
    assert ffp.resolve_ffprobe() == sibling


def test_the_sibling_swap_preserves_decoration_and_extension(clean_env, tmp_path):
    sibling = _touch(tmp_path / "b" / "ffprobe-7.1.exe")
    ffmpeg = _touch(tmp_path / "b" / "ffmpeg-7.1.exe")
    assert ffp.resolve_ffprobe(ffmpeg=ffmpeg) == sibling


def test_a_constructed_sibling_that_does_not_exist_is_not_offered(clean_env, tmp_path):
    ffmpeg = _touch(tmp_path / "lonely" / "ffmpeg.exe")
    assert ffp.resolve_ffprobe(ffmpeg=ffmpeg) is None


def test_a_pin_that_does_not_exist_is_skipped_not_returned(clean_env, tmp_path):
    on_path = _touch(tmp_path / "path" / "ffprobe.exe")
    clean_env.setenv("OTR_FFPROBE", str(tmp_path / "gone" / "ffprobe.exe"))
    _which_map(clean_env, {"ffprobe": on_path})
    assert ffp.resolve_ffprobe() == on_path


def test_resolve_never_raises_on_junk(clean_env):
    for junk in (0, "", "   ", b"", [], {}):
        assert ffp.resolve_ffprobe(junk) is None


# --------------------------------------------------------------------------- #
# probe_raw -- launches, and holds no opinion about the return code
# --------------------------------------------------------------------------- #
def test_probe_raw_refuses_by_name_when_no_binary_resolves(clean_env):
    with pytest.raises(ffp.FFprobeMissing) as excinfo:
        ffp.probe_raw(["-version"])
    assert "OTR_FFPROBE" in str(excinfo.value)


def test_probe_raw_hands_back_a_non_zero_return_code_without_raising(
        clean_env, tmp_path):
    binary = _touch(tmp_path / "ffprobe.exe")
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(argv, 1, "", "boom")

    clean_env.setattr(ffp.subprocess, "run", fake_run)
    proc = ffp.probe_raw(["-i", 42], ffprobe=binary, timeout=9)
    assert proc.returncode == 1 and proc.stderr == "boom"
    assert seen["argv"] == [binary, "-i", "42"]
    assert seen["kwargs"]["timeout"] == 9
    assert seen["kwargs"]["encoding"] == "utf-8"


def test_probe_raw_names_a_timeout_and_a_launch_failure(clean_env, tmp_path):
    binary = _touch(tmp_path / "ffprobe.exe")

    def timing_out(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, 3)

    clean_env.setattr(ffp.subprocess, "run", timing_out)
    with pytest.raises(ffp.FFprobeError) as excinfo:
        ffp.probe_raw(["-version"], ffprobe=binary, timeout=3)
    assert "timed out" in str(excinfo.value)

    def not_executable(argv, **kwargs):
        raise OSError(8, "Exec format error")

    clean_env.setattr(ffp.subprocess, "run", not_executable)
    with pytest.raises(ffp.FFprobeError):
        ffp.probe_raw(["-version"], ffprobe=binary)


def test_a_vanished_binary_is_reported_as_missing_not_as_a_probe_failure(
        clean_env, tmp_path):
    binary = _touch(tmp_path / "ffprobe.exe")

    def gone(argv, **kwargs):
        raise FileNotFoundError(2, "No such file")

    clean_env.setattr(ffp.subprocess, "run", gone)
    with pytest.raises(ffp.FFprobeMissing):
        ffp.probe_raw(["-version"], ffprobe=binary)


# --------------------------------------------------------------------------- #
# probe_json -- the query each caller keeps, assembled once
# --------------------------------------------------------------------------- #
def _stub_json(monkeypatch, tmp_path, stdout, returncode=0):
    binary = _touch(tmp_path / "ffprobe.exe")
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(argv, returncode, stdout, "why not")

    monkeypatch.setattr(ffp.subprocess, "run", fake_run)
    return binary, seen


def test_probe_json_builds_the_query_and_parses_the_document(clean_env, tmp_path):
    binary, seen = _stub_json(clean_env, tmp_path, '{"streams": [{"width": 1920}]}')
    doc = ffp.probe_json("clip.mp4", ["stream=width", "format=duration"],
                         select_streams="v:0", extra_args=["-count_frames"],
                         ffprobe=binary)
    assert doc["streams"][0]["width"] == 1920
    assert seen["argv"] == [
        binary, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=width", "-show_entries", "format=duration",
        "-of", "json", "clip.mp4"]


def test_probe_json_accepts_a_single_entries_string(clean_env, tmp_path):
    binary, seen = _stub_json(clean_env, tmp_path, "{}")
    ffp.probe_json("clip.mp4", "stream=width", ffprobe=binary)
    assert seen["argv"].count("-show_entries") == 1


def test_probe_json_refuses_a_failed_probe_and_unreadable_output(clean_env, tmp_path):
    binary, _ = _stub_json(clean_env, tmp_path, "", returncode=3)
    with pytest.raises(ffp.FFprobeError) as excinfo:
        ffp.probe_json("clip.mp4", "stream=width", ffprobe=binary)
    assert "why not" in str(excinfo.value)

    binary, _ = _stub_json(clean_env, tmp_path, "not json at all")
    with pytest.raises(ffp.FFprobeError) as excinfo:
        ffp.probe_json("clip.mp4", "stream=width", ffprobe=binary)
    assert "unparseable" in str(excinfo.value)


def test_probe_json_treats_empty_output_as_an_empty_document(clean_env, tmp_path):
    binary, _ = _stub_json(clean_env, tmp_path, "")
    assert ffp.probe_json("clip.mp4", "stream=width", ffprobe=binary) == {}


# --------------------------------------------------------------------------- #
# parse_rate / parse_fps_int -- the parse that was re-fixed three times
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rate,expected", [
    ("25/1", 25.0),
    ("50/2", 25.0),
    ("24", 24.0),
    (30.0, 30.0),
])
def test_parse_rate_reads_rationals_and_plain_numbers(rate, expected):
    assert ffp.parse_rate(rate) == pytest.approx(expected)


def test_parse_rate_reads_ntsc_without_rounding_it_away():
    assert ffp.parse_rate("30000/1001") == pytest.approx(29.97, abs=0.01)


@pytest.mark.parametrize("rate", [
    None, "", "   ", "N/A", "n/a", "0/0", "25/0", "garbage", "x/y", "0/1", "-25/1",
])
def test_parse_rate_says_unknown_rather_than_guessing_zero(rate):
    assert ffp.parse_rate(rate) is None


@pytest.mark.parametrize("rate,expected", [
    ("25/1", 25),
    ("30000/1001", 30),
    ("0/0", 0),
    (None, 0),
    ("garbage", 0),
])
def test_parse_fps_int_matches_the_shipped_clip_contract_values(rate, expected):
    """The four cases `wan_shared._parse_fps` has always been pinned to."""
    assert ffp.parse_fps_int(rate) == expected


# --------------------------------------------------------------------------- #
# invariant V-12
# --------------------------------------------------------------------------- #
def test_the_boundary_is_cold_import_clean():
    """Measuring a file must never drag a model framework into memory."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys; import nodes._otr_shared.ffprobe as m; "
         "print(sorted(k for k in sys.modules "
         "if k.split('.')[0] in {'torch', 'transformers', 'diffusers', "
         "'folder_paths', 'comfy'}))"],
        cwd=str(_REPO_ROOT), capture_output=True, text=True,
        encoding="utf-8", errors="replace")
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "[]", probe.stdout
