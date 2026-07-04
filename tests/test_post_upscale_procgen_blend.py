"""BUG-LOCAL-030 Phase B regression: post-RTXUpscale procgen blend node."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")


@pytest.fixture(autouse=True)
def _isolate_obs_dir(monkeypatch, tmp_path):
    """Pin OTR_OUTPUT_DIR so otr_obs_dir() resolves under tmp_path.

    BUG-LOCAL-108 path cleanup (2026-05-05): the blend node now writes
    its output into otr_obs_dir() rather than next to the source. Without
    this fixture every blend test would write into Jeffrey's real
    ComfyUI output tree.
    """
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path))


@pytest.fixture
def node():
    from nodes.otr_post_upscale_procgen_blend import PostUpscaleProcgenBlend

    return PostUpscaleProcgenBlend()


def _capture_run():
    captured = {}

    def _fake_run(cmd, **kwargs):
        # Always update captured["cmd"] to the LAST cmd (back-compat).
        # Also track ffmpeg-only cmds so tests can find the blend cmd
        # even when the node also invokes git rev-parse via ledger save.
        captured["cmd"] = cmd
        captured.setdefault("all", []).append(cmd)
        if cmd and isinstance(cmd, list):
            head = str(cmd[0]).lower()
            if (head.endswith("ffmpeg") or head.endswith("ffmpeg.exe")) and \
               captured.get("ffmpeg") is None:
                captured["ffmpeg"] = cmd

        class _R:
            returncode = 0

        return _R()

    return captured, _fake_run


def test_input_types_shape(node):
    spec = node.INPUT_TYPES()
    req = spec["required"]
    opt = spec["optional"]
    assert "source_mp4_path" in req
    assert "procgen_mp4_path" in req
    assert "blend_mode" in opt
    assert "blend_opacity" in opt
    assert "bypass" in opt
    assert "out_suffix" in opt


def test_returns_two_strings(node):
    assert node.RETURN_TYPES == ("STRING", "STRING")
    assert node.RETURN_NAMES == ("final_mp4_path", "report")
    assert node.OUTPUT_NODE is True


# -- P1: SDH caption burn wiring -------------------------------------------

def test_build_cmd_no_captions_is_unchanged():
    from nodes.otr_post_upscale_procgen_blend import _build_blend_cmd

    cmd = _build_blend_cmd(
        source_mp4=Path("s.mp4"), procgen_mp4=Path("p.mp4"),
        out_mp4=Path("o.mp4"), blend_mode="screen", blend_opacity=1.0,
        ffmpeg="ffmpeg",
    )
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert fc.endswith("[v]")
    assert "ass=" not in fc            # no caption step when not requested
    assert "-c:a" in cmd and cmd[cmd.index("-c:a") + 1] == "copy"


def test_build_cmd_with_captions_appends_ass_by_basename():
    from nodes.otr_post_upscale_procgen_blend import _build_blend_cmd

    cmd = _build_blend_cmd(
        source_mp4=Path("s.mp4"), procgen_mp4=Path("p.mp4"),
        out_mp4=Path("o.mp4"), blend_mode="screen", blend_opacity=1.0,
        ffmpeg="ffmpeg",
        captions_ass_path=r"C:\Users\x\ep\audio\ep_captions.ass",
    )
    fc = cmd[cmd.index("-filter_complex") + 1]
    # blend routes through [vpre]; ass burns by BASENAME (no drive colon, no path)
    assert "[vpre]" in fc
    assert ";[vpre]ass=ep_captions.ass[v]" in fc
    assert "C:" not in fc and "/" not in fc.split("ass=")[1]
    # audio still copied -- C7 untouched
    assert cmd[cmd.index("-c:a") + 1] == "copy"


def test_build_cmd_3input_scopes_no_double_format_gbrp_bug402():
    """BUG-LOCAL-402: the §4D 3-input blend (procgen floor + scene-aware scopes)
    must emit a WELL-FORMED filter. green_only_step already ends with
    ',format=gbrp', so a second pin used to collapse into 'format=gbrpformat=gbrp';
    ffmpeg rejected the bogus 'gbrpformat' option and the WHOLE blend + SDH caption
    burn silently fell back to source-copy on every episode. Guard both overlay
    states + the caption burn on top of the 3-input graph."""
    from nodes.otr_post_upscale_procgen_blend import _build_blend_cmd
    for green in (True, False):
        cmd = _build_blend_cmd(
            source_mp4=Path("s.mp4"), procgen_mp4=Path("p.mp4"),
            out_mp4=Path("o.mp4"), blend_mode="screen", blend_opacity=1.0,
            ffmpeg="ffmpeg", scopes_mp4=Path("scopes.mp4"),
            green_only_overlay=green,
        )
        fc = cmd[cmd.index("-filter_complex") + 1]
        assert "gbrpformat" not in fc, f"green={green}: malformed filter -> {fc}"
        # exactly three gbrp pins (main, pgn, scp); none collapsed to gbrpformat
        assert fc.count("format=gbrp") == 3, f"green={green}: {fc}"
        # BUG-LOCAL-410: the scopes track is BLACK-padded past its end so the
        # shortest=1 lighten-blend does not re-clamp the credits post-roll back
        # to the (shorter) scopes length.
        assert "tpad=stop_mode=add" in fc and "[scp]" in fc, f"green={green}: {fc}"
        # captions still burn on top of the 3-input blend (not lost to fallback)
        cmd_cap = _build_blend_cmd(
            source_mp4=Path("s.mp4"), procgen_mp4=Path("p.mp4"),
            out_mp4=Path("o.mp4"), blend_mode="screen", blend_opacity=1.0,
            ffmpeg="ffmpeg", scopes_mp4=Path("scopes.mp4"),
            green_only_overlay=green,
            captions_ass_path=r"C:\ep\audio\ep_captions.ass",
        )
        fc_cap = cmd_cap[cmd_cap.index("-filter_complex") + 1]
        assert "gbrpformat" not in fc_cap
        assert ";[vpre]ass=ep_captions.ass[v]" in fc_cap


def test_ass_filter_arg_returns_basename_and_parent():
    from nodes.otr_post_upscale_procgen_blend import _ass_filter_arg

    name, cwd = _ass_filter_arg(r"C:\Users\x\ep\audio\ep_captions.ass")
    assert name == "ep_captions.ass"
    assert cwd.endswith("audio")


# -- P4: caption widgets + node<->JSON widget-vector contract --------------

def test_caption_widgets_removed(node):
    # 2026-07-04 widget-audit Batch 3: caption ownership migrated to node 86
    # OTR_CaptionBurn (chain 84 -> 93 -> 86 -> 95 -> 85); node 93 no longer
    # exposes the burn_captions / caption_style widgets or burns captions.
    it = node.INPUT_TYPES()
    opt = it["optional"]
    req = it.get("required", {})
    assert "burn_captions" not in opt and "caption_style" not in opt
    assert "burn_captions" not in req and "caption_style" not in req


def test_default_blend_opacity_is_full_strength(node):
    """BUG-LOCAL-096 (2026-05-04 EVENING): default opacity bumped from
    0.5 to 1.0 so the procgen overlay shows at full intensity by
    default. Pre-096 default produced a "weak" mix that washed out
    procgen colors. Pin the new default so a silent revert is caught.
    """
    spec = node.INPUT_TYPES()
    assert spec["optional"]["blend_opacity"][1]["default"] == 1.0


def test_default_blend_mode_is_screen_with_green_only(node):
    """BUG-LOCAL-106 (2026-05-04 LATE EVENING): default flipped from
    "lighten" + green_only=False to "screen" + green_only=True after
    BUG-105 A/B confirmed `screen_GREEN_crush18` was the visibly
    correct combo on echo_in_stasis source. screen + green_only_overlay
    pairs cleanly because:
      - green_only_overlay zeros procgen R and B before the blend
      - screen formula `out = A + B - A*B` preserves source R and B
        (B contribution from procgen is 0 -> source channel passes
        through) and lifts source G exactly where the green CRT
        wireframe lives.
    Pre-106 "lighten" default collapsed to white over fully-saturated
    source pixels (max((255,0,255),(0,255,0)) == (255,255,255)) --
    looked like glare, not phosphor. Pin the new default so a silent
    revert is caught.
    """
    spec = node.INPUT_TYPES()
    assert spec["optional"]["blend_mode"][1]["default"] == "screen"
    assert spec["optional"]["green_only_overlay"][1]["default"] is True


def test_blend_cmd_uses_audio_passthrough(node, tmp_path):
    src = tmp_path / "rtx_upscale.mp4"
    src.write_bytes(b"src")
    pgn = tmp_path / "procgen_1080p.mp4"
    pgn.write_bytes(b"pgn")
    captured, fake_run = _capture_run()
    from nodes import otr_post_upscale_procgen_blend as M

    # Stub the ledger stamp helper so its subprocess side-effects (git
    # rev-parse via the canonical Ledger.save commit lookup) don't
    # contaminate the captured ffmpeg cmd. Stamp behavior is covered by
    # its own dedicated tests further down.
    with mock.patch.object(M.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(M, "_stamp_ledger_final_video_path",
                           return_value=(False, "stubbed for cmd-shape test")):
        node.blend(
            source_mp4_path=str(src),
            procgen_mp4_path=str(pgn),
            blend_mode="lighten",
            blend_opacity=0.5,
        )
    # Look at the ffmpeg cmd specifically (filtering out any other
    # subprocess invocations e.g. git rev-parse from ledger save).
    cmd = captured.get("ffmpeg") or captured["cmd"]
    # C7 audio passthrough
    assert "-c:a" in cmd, f"expected -c:a in ffmpeg cmd; got {cmd}"
    assert cmd[cmd.index("-c:a") + 1] == "copy", (
        "audio MUST be -c:a copy (zero re-encodes for C7 byte-identity)"
    )
    # Both inputs present
    assert str(src) in cmd
    assert str(pgn) in cmd
    # Filter chain references both
    assert "-filter_complex" in cmd
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "[0:v]" in fc and "[1:v]" in fc
    assert "blend=all_mode=lighten" in fc
    assert "all_opacity=0.500" in fc


def test_blend_cmd_does_NOT_use_shortest_for_c7_safety(node, tmp_path):
    """BUG-LOCAL-030 C7 hardening (post-round-robin Gemini catch):
    ``-shortest`` MUST NOT appear in the blend cmd. ffmpeg ``-shortest``
    on an A/V mux stops writing audio when the shortest input ends --
    if procgen is even 40ms shorter than source (likely from 24fps vs
    25fps quantization), trailing audio gets truncated and Rule C7
    byte-identity silently breaks.

    Without ``-shortest``, framesync holds the last procgen frame as
    the overlay continues; audio passes through via ``-c:a copy`` from
    source until source EOF. Audio reaches full duration. C7 holds.
    """
    src = tmp_path / "src.mp4"; src.write_bytes(b"x")
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"x")
    captured, fake_run = _capture_run()
    from nodes import otr_post_upscale_procgen_blend as M

    # Stub ledger stamp -- this test is about the ffmpeg cmd shape, not
    # the ledger stamp side-effect (covered by other tests below).
    with mock.patch.object(M.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(M, "_stamp_ledger_final_video_path",
                           return_value=(False, "stubbed for cmd-shape test")):
        node.blend(source_mp4_path=str(src), procgen_mp4_path=str(pgn))
    cmd = captured.get("ffmpeg") or captured["cmd"]
    assert "-shortest" not in cmd, (
        f"-shortest must NOT be present in PostUpscaleProcgenBlend cmd "
        f"(C7 audio truncation risk per BUG-LOCAL-030 round-robin); got {cmd}"
    )
    # Audio MUST still be -c:a copy (passthrough, no re-encode).
    assert "-c:a" in cmd
    assert cmd[cmd.index("-c:a") + 1] == "copy"
    # Source audio MUST still be mapped (-map "0:a?").
    map_idx = [i for i, x in enumerate(cmd) if x == "-map"]
    map_args = [cmd[i + 1] for i in map_idx]
    assert "0:a?" in map_args, f"audio must still be mapped from source; got maps={map_args}"


def test_blend_cmd_includes_longform_hardening_flags(node, tmp_path):
    """BUG-LOCAL-030 long-form hardening (post round-robin risk-#10):
    the blend cmd MUST include ``-max_muxing_queue_size 1024`` (guards
    against "Too many packets buffered" on long-form blends) AND thread
    caps (``-filter_complex_threads 2``, ``-filter_threads 2``,
    ``-threads 4``) so a 5+ min episode does not spike DRAM via
    thread x framebuffer multiplication. Per Gemini round-robin.
    """
    src = tmp_path / "src.mp4"; src.write_bytes(b"x")
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"x")
    captured, fake_run = _capture_run()
    from nodes import otr_post_upscale_procgen_blend as M

    with mock.patch.object(M.subprocess, "run", side_effect=fake_run), \
         mock.patch.object(M, "_stamp_ledger_final_video_path",
                           return_value=(False, "stubbed for cmd-shape test")):
        node.blend(source_mp4_path=str(src), procgen_mp4_path=str(pgn))
    cmd = captured.get("ffmpeg") or captured["cmd"]
    assert "-max_muxing_queue_size" in cmd, (
        f"-max_muxing_queue_size MUST be present for long-form safety; got {cmd}"
    )
    assert cmd[cmd.index("-max_muxing_queue_size") + 1] == "1024"
    assert "-filter_complex_threads" in cmd
    assert cmd[cmd.index("-filter_complex_threads") + 1] == "2"
    assert "-filter_threads" in cmd
    assert cmd[cmd.index("-filter_threads") + 1] == "2"
    assert "-threads" in cmd
    assert cmd[cmd.index("-threads") + 1] == "4"


def test_bypass_mode_copies_source_to_output(node, tmp_path):
    """bypass=True must skip ffmpeg entirely and copy source -> out."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"original-content")
    pgn = tmp_path / "pgn.mp4"
    pgn.write_bytes(b"pgn")
    from nodes import otr_post_upscale_procgen_blend as M

    with mock.patch.object(M.subprocess, "run") as mock_run:
        out, report = node.blend(
            source_mp4_path=str(src), procgen_mp4_path=str(pgn),
            bypass=True,
        )
        mock_run.assert_not_called()
    out_p = Path(out)
    assert out_p.exists()
    assert out_p.read_bytes() == b"original-content"
    assert "bypass" in report.lower()


def test_missing_source_returns_empty_path(node, tmp_path):
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"x")
    out, report = node.blend(
        source_mp4_path=str(tmp_path / "does_not_exist.mp4"),
        procgen_mp4_path=str(pgn),
    )
    assert out == ""
    assert "missing" in report.lower()


def test_missing_procgen_falls_back_to_source_copy(node, tmp_path):
    """If procgen mp4 isn't on disk, gracefully copy source -> output
    so the pipeline still produces a deliverable. Logs a warning."""
    src = tmp_path / "src.mp4"
    src.write_bytes(b"src-content")
    out, report = node.blend(
        source_mp4_path=str(src),
        procgen_mp4_path=str(tmp_path / "missing_procgen.mp4"),
    )
    assert out != ""
    out_p = Path(out)
    assert out_p.exists()
    assert out_p.read_bytes() == b"src-content"
    assert "skipped blend" in report.lower() or "procgen" in report.lower()


def test_ffmpeg_failure_falls_back_to_source_copy(node, tmp_path):
    """If ffmpeg blend itself fails, copy source -> output as fallback
    so the pipeline doesn't drop the deliverable."""
    src = tmp_path / "src.mp4"; src.write_bytes(b"src-content")
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"pgn")
    import subprocess as _sub
    from nodes import otr_post_upscale_procgen_blend as M

    def _fail_run(cmd, **kwargs):
        raise _sub.CalledProcessError(returncode=1, cmd=cmd, stderr=b"ffmpeg boom")

    with mock.patch.object(M.subprocess, "run", side_effect=_fail_run):
        out, report = node.blend(source_mp4_path=str(src), procgen_mp4_path=str(pgn))
    out_p = Path(out)
    assert out_p.exists()
    assert out_p.read_bytes() == b"src-content"
    assert "ffmpeg" in report.lower() or "failed" in report.lower()


def test_node_registered_in_init():
    """OTR_PostUpscaleProcgenBlend must be in NODE_CLASS_MAPPINGS."""
    import importlib
    sys.path.insert(0, str(_REPO_ROOT.parent))
    try:
        pkg = importlib.import_module(_REPO_ROOT.name)
        assert "OTR_PostUpscaleProcgenBlend" in pkg.NODE_CLASS_MAPPINGS, (
            "node not registered in __init__.py"
        )
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------------------
# BUG-LOCAL-030 audit gap fix (2026-05-03 EVENING): ledger stamp
# ---------------------------------------------------------------------------

def test_ledger_stamped_after_successful_blend(node, tmp_path):
    """On successful blend: ledger.final_video_path + meta.paths.obs_final_blended
    + meta.post_upscale_blend forensics block all updated."""
    import json
    src = tmp_path / "src_1080p.mp4"; src.write_bytes(b"src")
    pgn = tmp_path / "pgn_1080p.mp4"; pgn.write_bytes(b"pgn")
    # Build a fake ledger on disk + point in_flight_ledger_path at it.
    ep_id = "signal_lost_test_20260503_010101"
    ep_dir = tmp_path / "fake_output" / "otr" / "episodes" / ep_id
    audio_dir = ep_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"
    initial_ledger = {
        "episode_id": ep_id,
        "schema_version": "l3-test",
        "final_video_path": str(tmp_path / "obs" / f"{ep_id}_1080p.mp4"),  # pre-blend
        "meta": {"schema_version": "l3-test", "paths": {"audio_dir": str(audio_dir)}},
    }
    ledger_path.write_text(json.dumps(initial_ledger), encoding="utf-8")

    captured, fake_run = _capture_run()
    from nodes import otr_post_upscale_procgen_blend as M

    with mock.patch.object(M.subprocess, "run", side_effect=fake_run), \
         mock.patch("nodes._otr_ledger.in_flight_ledger_path", return_value=ledger_path):
        out, report = node.blend(
            source_mp4_path=str(src),
            procgen_mp4_path=str(pgn),
            blend_mode="lighten",
            blend_opacity=0.5,
        )

    # Ledger on disk should now reflect the blended path.
    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert led["final_video_path"].endswith("_procgen_blended.mp4"), (
        f"final_video_path must point at the blended output; got {led['final_video_path']}"
    )
    # NOTE: meta.paths is owned by _build_meta_paths (per schema doc rule
    # "Never write to meta.paths from outside _build_meta_paths"). The
    # blended path is discoverable via top-level final_video_path AND
    # via meta.post_upscale_blend.blended_mp4 (the forensics block).
    assert led["meta"]["post_upscale_blend"]["blended_mp4"].endswith("_procgen_blended.mp4")

    # Forensics block populated correctly.
    fb = led["meta"]["post_upscale_blend"]
    assert fb["source_mp4"] == str(src)
    assert fb["procgen_mp4"] == str(pgn)
    assert fb["blended_mp4"].endswith("_procgen_blended.mp4")
    assert fb["blend_mode"] == "lighten"
    assert fb["blend_opacity"] == 0.5

    # Report mentions the stamp.
    assert "ledger" in report.lower()
    assert "stamped" in report.lower()


def test_ledger_NOT_stamped_on_bypass_mode(node, tmp_path):
    """bypass=True: do NOT touch the ledger (output is a verbatim copy of
    source; final_video_path should remain whatever RTXUpscale set)."""
    import json
    src = tmp_path / "src.mp4"; src.write_bytes(b"src")
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"pgn")
    ep_id = "test_ep_001"
    audio_dir = tmp_path / "fake_output" / "otr" / "episodes" / ep_id / "audio"
    audio_dir.mkdir(parents=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"
    pre_blend_path = "C:/some/preblend.mp4"
    initial_ledger = {
        "episode_id": ep_id,
        "schema_version": "l3-test",
        "final_video_path": pre_blend_path,
        "meta": {"schema_version": "l3-test"},
    }
    ledger_path.write_text(json.dumps(initial_ledger), encoding="utf-8")

    from nodes import otr_post_upscale_procgen_blend as M
    with mock.patch("nodes._otr_ledger.in_flight_ledger_path", return_value=ledger_path):
        node.blend(
            source_mp4_path=str(src), procgen_mp4_path=str(pgn),
            bypass=True,
        )

    # Ledger should be UNCHANGED — bypass means the output is just a
    # copy of source, the original final_video_path is still correct.
    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert led["final_video_path"] == pre_blend_path, (
        f"bypass mode must NOT change ledger.final_video_path; was {pre_blend_path}, "
        f"now {led['final_video_path']}"
    )
    assert "post_upscale_blend" not in led["meta"], (
        "bypass mode must NOT add post_upscale_blend forensics block"
    )


def test_ledger_NOT_stamped_when_procgen_missing(node, tmp_path):
    """Procgen file missing: degraded copy mode, do NOT stamp ledger
    (output is a copy of source, not a real blend)."""
    import json
    src = tmp_path / "src.mp4"; src.write_bytes(b"src")
    ep_id = "test_ep_002"
    audio_dir = tmp_path / "fake_output" / "otr" / "episodes" / ep_id / "audio"
    audio_dir.mkdir(parents=True)
    ledger_path = audio_dir / f"{ep_id}_ledger.json"
    pre_blend_path = "C:/some/preblend.mp4"
    initial_ledger = {
        "episode_id": ep_id,
        "schema_version": "l3-test",
        "final_video_path": pre_blend_path,
        "meta": {"schema_version": "l3-test"},
    }
    ledger_path.write_text(json.dumps(initial_ledger), encoding="utf-8")

    with mock.patch("nodes._otr_ledger.in_flight_ledger_path", return_value=ledger_path):
        node.blend(
            source_mp4_path=str(src),
            procgen_mp4_path=str(tmp_path / "missing_procgen.mp4"),
        )

    led = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert led["final_video_path"] == pre_blend_path
    assert "post_upscale_blend" not in led["meta"]


def test_ledger_stamp_helper_never_raises_on_no_singleton(tmp_path):
    """If no in-flight ledger singleton + no fallback ledger, the helper
    must return (False, msg) -- never raise. The blend node must still
    succeed and produce its output."""
    from nodes import otr_post_upscale_procgen_blend as M
    src = tmp_path / "src.mp4"; src.write_bytes(b"src")
    pgn = tmp_path / "pgn.mp4"; pgn.write_bytes(b"pgn")
    blended = tmp_path / "src_procgen_blended.mp4"

    with mock.patch("nodes._otr_ledger.in_flight_ledger_path", return_value=None):
        stamped, msg = M._stamp_ledger_final_video_path(
            blended_mp4=blended, source_mp4=src, procgen_mp4=pgn,
            blend_mode="lighten", blend_opacity=0.5,
        )
    assert stamped is False
    assert "no in-flight ledger" in msg.lower() or "skipped" in msg.lower()


def test_signal_lost_video_resolution_default_is_1080p():
    """BUG-030 Phase B: procgen renders at native 1920x1080 by default."""
    src = (_REPO_ROOT / "nodes" / "video_engine.py").read_text(encoding="utf-8")
    import re
    m = re.search(r'"resolution":\s*\(\[([^\]]+)\],\s*\{\s*"default":\s*"([^"]+)"', src)
    assert m is not None, "could not locate resolution widget in video_engine.py"
    choices, default = m.group(1), m.group(2)
    assert default == "1920x1080", (
        f"expected resolution default=1920x1080 (BUG-030 Phase B); got {default}"
    )
    assert "1920x1080" in choices and "832x480" in choices, (
        "both 1920x1080 (new default) and 832x480 (legacy) must remain in choices"
    )
