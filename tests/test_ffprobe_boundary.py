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

import ast
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
# THE RATCHET -- the boundary is the only door out of nodes/
#
# Not a style rule. Every one of these three patterns is a way a caller found
# its own ffprobe, and every caller that found its own found a DIFFERENT one:
# four independent binary-resolution strategies across eleven files, only one
# of which had ever heard of OTR_FFPROBE. A new caller that copies the nearest
# example is how that happened, so the nearest example has to stop existing.
# --------------------------------------------------------------------------- #
_BOUNDARY = _REPO_ROOT / "nodes" / "_otr_shared" / "ffprobe.py"
_PROBE_NAMES = {"ffprobe", "ffprobe.exe"}


def _module_string_constants(tree):
    """Module-level ``NAME = "literal"`` bindings, so the scans see one hop.

    The first version of these tests only recognised a string sitting DIRECTLY
    in the argument position, and a QA pass proved the hole by building a file
    that does everything the boundary exists to prevent while routing both
    literals through named module constants. A rule that a rename defeats is
    not a rule.
    """
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    found[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) \
                and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            found[node.target.id] = node.value.value
    return found


def _nodes_sources():
    for path in sorted((_REPO_ROOT / "nodes").rglob("*.py")):
        if path == _BOUNDARY or "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        yield path, tree, _module_string_constants(tree)


def _constant_str(node, consts=None):
    """The string this expression IS -- a literal, or a name bound to one."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if consts and isinstance(node, ast.Name):
        return consts.get(node.id)
    return None


def _searches_path(tree, consts):
    """``shutil.which("ffprobe")`` -- PATH is one of six places, not the place."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(
            func, "id", "")
        if name != "which":
            continue
        first = _constant_str(node.args[0], consts)
        if first and first.lower() in _PROBE_NAMES:
            yield node.lineno


def _reads_the_operator_pin(tree, consts):
    """``OTR_FFPROBE`` read anywhere but the one reader."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            if _constant_str(node.slice, consts) == "OTR_FFPROBE":
                yield node.lineno
        elif isinstance(node, ast.Call):
            for arg in node.args:
                if _constant_str(arg, consts) == "OTR_FFPROBE":
                    yield node.lineno


def _builds_a_bare_argv(tree, consts):
    """``["ffprobe", ...]`` -- a hope, not a configuration."""
    for node in ast.walk(tree):
        if not isinstance(node, (ast.List, ast.Tuple)) or not node.elts:
            continue
        first = _constant_str(node.elts[0], consts)
        if first and first.lower() in _PROBE_NAMES:
            yield node.lineno


_SCANS = [
    ("searches PATH for ffprobe itself", _searches_path),
    ("reads OTR_FFPROBE itself", _reads_the_operator_pin),
    ("builds an argv on a bare ffprobe literal", _builds_a_bare_argv),
]
_SCAN_IDS = ["which", "pin", "argv"]


@pytest.mark.parametrize("label,scan", _SCANS, ids=_SCAN_IDS)
def test_no_module_outside_the_boundary(label, scan):
    offenders = []
    for path, tree, consts in _nodes_sources():
        offenders += ["%s:%d" % (path.name, line) for line in scan(tree, consts)]
    assert not offenders, (
        "a module under nodes/ %s, outside nodes/_otr_shared/ffprobe.py: %s"
        % (label, ", ".join(offenders)))


#: A caller that does everything the boundary exists to prevent, while routing
#: both literals through named module constants one hop from the call site.
#: A QA pass built exactly this and walked it past the first version of the
#: three scans above, all of which came back empty. It is checked in as a
#: FIXTURE rather than fixed once and forgotten: the scans have to keep
#: catching it.
_EVASIVE_CALLER = '''
import os
import shutil
import subprocess

_FFPROBE_ENV = "OTR_FFPROBE"
_FFPROBE_EXE = "ffprobe"


def probe(path):
    binary = os.environ.get(_FFPROBE_ENV) or shutil.which(_FFPROBE_EXE)
    return subprocess.run([_FFPROBE_EXE, "-v", "error", str(path)],
                          capture_output=True, text=True)
'''


@pytest.mark.parametrize("label,scan", _SCANS, ids=_SCAN_IDS)
def test_the_ratchet_catches_a_caller_that_hides_its_literals(label, scan):
    tree = ast.parse(_EVASIVE_CALLER)
    assert list(scan(tree, _module_string_constants(tree))), (
        "the scan for %r did not see the evasive caller -- a rule a rename "
        "defeats is not a rule" % label)


@pytest.mark.parametrize("relative", [
    "otr_credits_roll.py",
    "otr_master_audio_mux.py",
    "otr_silent_composite.py",
    "otr_post_upscale_procgen_blend.py",
    "otr_scene_aware_scopes.py",
    "_otr_shared/cloud_media_canonical.py",
    "_otr_video_engines/wan_shared.py",
    "_otr_video_engines/eng_cloud_video.py",
    "_otr_video_engines/eng_google_omni_video.py",
    "_otr_video_engines/eng_google_veo_video.py",
    "_otr_upscale_engines/_pipeline.py",
])
def test_every_migrated_caller_actually_goes_through_the_boundary(relative):
    """The absence tests above pass just as well on a module that stopped
    probing. This one says the door is USED.

    It is a FIXED list, and that is a known limit rather than an oversight: a
    brand-new file is invisible here until someone adds it. The guard against a
    new file is the three offender scans above, which see every module under
    ``nodes/`` and need no list at all. This one guards against the OTHER
    direction -- a migrated caller quietly reverting."""
    source = (_REPO_ROOT / "nodes" / relative).read_text(encoding="utf-8")
    assert any(name in source for name in
               ("resolve_ffprobe", "probe_raw", "probe_json", "parse_rate",
                "parse_fps_int")), relative


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
