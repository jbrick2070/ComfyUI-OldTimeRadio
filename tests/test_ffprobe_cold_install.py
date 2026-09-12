"""A cold install has ffmpeg but no ffprobe, and the render used to die there.

`pip install -r requirements.txt` brings imageio-ffmpeg, which ships ONE binary.
`resolve_ffprobe()` then answers None, and the first clip probe-back --
`wan_shared.ffprobe_clip_fields` -> `probe_json` -> `probe_raw` -> FFprobeMissing
-> GraphExecutionError -> `_render_shot` LOUD -- ended the leg minutes in with
nothing in otr/obs (live artifact: a Mac mini M4, 2026-09-07, requirements.txt).
A lane that cleared the engines died at the LAST node instead: the credits'
`_ffprobe_bin()` raised CreditsDataError on a fully rendered episode.

The fix lives in ONE place. `probe_json` builds the same document from PyAV
(a ComfyUI core dependency, `av>=17`) when no binary resolves: same keys, same
types (counts, rates and seconds are STRINGS, sizes are ints), same omissions
(an unspecified colour tag is ABSENT, never "unknown"). `probe_raw` still
refuses by name -- a caller that wants ffprobe's own text cannot be served any
other way -- which is why `ffprobe_counted_frames` moved its query onto
`probe_json`, and the credits no longer force a binary.

The parity half of this file runs only where a real ffprobe resolves and
compares the two documents key for key on a clip PyAV writes itself; the shape
half runs everywhere.
"""
from __future__ import annotations

import inspect
import os
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

import pytest

av = pytest.importorskip("av")

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("OTR_TEST_MODE", "1")

from nodes._otr_shared import ffprobe as ffp  # noqa: E402
from nodes._otr_video_engines import wan_shared as ws  # noqa: E402
from nodes._otr_video_engines import wrapper_bridge as wb  # noqa: E402


# --------------------------------------------------------------------------- #
# fixtures -- a clip no tool is needed to write
# --------------------------------------------------------------------------- #
def _write_clip(path, *, with_audio):
    """A 12-frame 64x48 mpeg4 clip at 24 fps with no colour tags, plus an
    optional 16 kHz aac track -- written by PyAV, so the file needs no tool."""
    with av.open(str(path), "w") as out:
        video = out.add_stream("mpeg4", rate=24)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio = out.add_stream("aac", rate=16000) if with_audio else None
        for i in range(12):
            frame = av.VideoFrame(64, 48, "rgb24")
            frame.planes[0].update(bytes([i * 20]) * frame.planes[0].buffer_size)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        if audio is not None:
            for i in range(8):
                frame = av.AudioFrame(format="fltp", layout="mono", samples=1024)
                frame.sample_rate = 16000
                for plane in frame.planes:
                    plane.update(bytes(plane.buffer_size))
                frame.pts = i * 1024
                for packet in audio.encode(frame):
                    out.mux(packet)
            for packet in audio.encode():
                out.mux(packet)
    return str(path)


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    return _write_clip(tmp_path_factory.mktemp("cold") / "untagged_av.mp4",
                       with_audio=True)


@pytest.fixture(scope="module")
def silent_clip(tmp_path_factory):
    return _write_clip(tmp_path_factory.mktemp("cold") / "untagged_v.mp4",
                       with_audio=False)


@pytest.fixture()
def no_binary(monkeypatch):
    """The cold install: nothing resolves, whatever this box really has."""
    monkeypatch.setattr(ffp, "resolve_ffprobe", lambda *a, **k: None)
    return monkeypatch


#: Every probe_json query the pack makes, by caller (grounded 2026-09-11).
QUERIES = [
    ("wan_shared.ffprobe_clip_fields", dict(entries=(
        "stream=codec_type,codec_name,pix_fmt,color_primaries,color_transfer,"
        "color_space,avg_frame_rate,r_frame_rate,width,height,nb_frames"))),
    ("wan_shared.ffprobe_counted_frames", dict(
        entries="stream=nb_read_frames", select_streams="v:0",
        extra_args=("-count_frames",))),
    ("otr_credits_roll._probe_video", dict(
        entries=["stream=width,height,r_frame_rate", "format=duration"],
        select_streams="v:0")),
    ("cloud_media_canonical._ffprobe_streams", dict(
        entries=None, extra_args=("-show_streams", "-show_format"), timeout=120)),
    ("foley_stems mux proof", dict(entries="stream=codec_type,duration")),
    ("an audio-first query", dict(
        entries="stream=codec_type,duration,sample_rate,channels",
        select_streams="a:0")),
    ("a decoded count of every stream", dict(
        entries="stream=codec_type,nb_frames,nb_read_frames",
        extra_args=("-count_frames",))),
]
_AUDITED_STREAM_KEYS = (
    "index", "codec_type", "codec_name", "width", "height", "pix_fmt",
    "color_space", "color_primaries", "color_transfer", "r_frame_rate",
    "avg_frame_rate", "nb_frames", "nb_read_frames", "duration", "sample_rate",
    "channels", "time_base", "start_time", "bit_rate")
_AUDITED_FORMAT_KEYS = ("duration", "nb_streams", "format_name", "size",
                        "bit_rate", "start_time")
_REAL_BINARY = ffp.resolve_ffprobe()


def _agrees(mine, theirs, audited, everything):
    """Every key the fallback wrote equals the tool's; every audited key is
    present or absent on BOTH sides. Under a field list the two documents
    are simply equal."""
    if not everything:
        assert mine == theirs, (mine, theirs)
        return
    for key, value in mine.items():
        assert theirs.get(key) == value, (key, value, theirs.get(key))
    for key in audited:
        assert (key in mine) == (key in theirs), (key, mine, theirs)


# --------------------------------------------------------------------------- #
# parity -- the real tool beside the fallback, key for key
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _REAL_BINARY,
                    reason="no ffprobe on this box; the parity half needs the tool")
@pytest.mark.parametrize("caller,query", QUERIES, ids=[q[0] for q in QUERIES])
def test_the_fallback_matches_the_real_tool_key_for_key(clip, caller, query,
                                                        monkeypatch):
    real = ffp.probe_json(clip, **query)
    monkeypatch.setattr(ffp, "resolve_ffprobe", lambda *a, **k: None)
    emulated = ffp.probe_json(clip, **query)
    everything = "-show_streams" in query.get("extra_args", ())
    real_streams, mine_streams = real.get("streams", []), emulated.get("streams", [])
    assert len(mine_streams) == len(real_streams), (mine_streams, real_streams)
    for mine, theirs in zip(mine_streams, real_streams):
        _agrees(mine, theirs, _AUDITED_STREAM_KEYS, everything)
    assert ("format" in real) == ("format" in emulated)
    if "format" in real:
        _agrees(emulated["format"], real["format"], _AUDITED_FORMAT_KEYS,
                "-show_format" in query.get("extra_args", ()))


# --------------------------------------------------------------------------- #
# shape -- what each caller reads, with no binary anywhere
# --------------------------------------------------------------------------- #
def test_the_clip_contract_probe_reads_the_same_fields_without_a_binary(
        no_binary, clip):
    fields = ws.ffprobe_clip_fields(clip)
    assert fields["codec_types"] == ["video", "audio"]
    assert fields["video_codec"] == "mpeg4" and fields["pix_fmt"] == "yuv420p"
    assert (int(fields["fps"]), fields["width"], fields["height"],
            fields["nb_frames"]) == (24, 64, 48, 12)
    assert fields["color_space"] is None and fields["color_primaries"] is None


def test_the_decoded_count_is_measured_not_read_from_the_header(no_binary, clip):
    assert ws.ffprobe_counted_frames(clip) == 12


class _Packet:
    def __init__(self, stream, frames):
        self.stream, self._frames = stream, frames

    def decode(self):
        return [object()] * self._frames


class _FakeStream:
    def __init__(self, index, kind, header_frames):
        self.index, self.type, self.frames = index, kind, header_frames


class _FakeContainer:
    """A container whose HEADER count disagrees with its picture data -- the
    concatenated-beat case `ffprobe_counted_frames` exists for."""
    def __init__(self):
        self.video = _FakeStream(0, "video", header_frames=99)
        self.audio = _FakeStream(1, "audio", header_frames=99)
        self.streams = [self.video, self.audio]

    def demux(self, *streams):
        wanted = {s.index for s in streams}
        for packet in (_Packet(self.video, 3), _Packet(self.audio, 2),
                       _Packet(self.video, 4), _Packet(self.video, 0)):
            if packet.stream.index in wanted:
                yield packet


def test_select_streams_reads_kind_ordinal_and_index_like_ffprobe():
    """A clip with one stream of each kind cannot tell v:0 from v; this can."""
    v0, a0, a1 = (_FakeStream(0, "video", 1), _FakeStream(1, "audio", 1),
                  _FakeStream(2, "audio", 1))
    streams = [v0, a0, a1]
    assert ffp._select(streams, "") == streams
    assert ffp._select(streams, "a") == [a0, a1]
    assert ffp._select(streams, "a:1") == [a1]
    assert ffp._select(streams, "a:0") == [a0]
    assert ffp._select(streams, "a:5") == []
    assert ffp._select(streams, "1") == [a0], "a bare number is a stream INDEX"
    assert ffp._select(streams, "V:0") == [v0]


def test_the_decoded_count_comes_from_decoding_not_from_the_header():
    container = _FakeContainer()
    assert ffp._decoded_counts(container, [container.video]) == {0: 7}
    assert ffp._decoded_counts(container, container.streams) == {0: 7, 1: 2}
    assert ffp._decoded_counts(container, []) == {}


def test_the_counted_frames_probe_names_its_failure_without_a_binary(
        no_binary, tmp_path):
    with pytest.raises(wb.GraphExecutionError, match="frame-count probe failed"):
        ws.ffprobe_counted_frames(str(tmp_path / "gone.mp4"))


def test_the_credits_measure_the_source_without_forcing_a_binary(no_binary, clip):
    from nodes import otr_credits_roll as credits
    probed = credits._probe_video(clip)
    assert (probed["w"], probed["h"], probed["fps"]) == (64, 48, 24.0)
    assert probed["duration"] == pytest.approx(0.512, abs=0.002)


def test_per_stream_and_container_durations_are_served(no_binary, clip):
    doc = ffp.probe_json(clip, "stream=codec_type,duration")
    by_kind = {s["codec_type"]: float(s["duration"]) for s in doc["streams"]}
    assert by_kind["video"] == pytest.approx(0.5, abs=0.002)
    assert by_kind["audio"] == pytest.approx(0.512, abs=0.002)
    full = ffp.probe_json(clip, extra_args=("-show_streams", "-show_format"),
                          timeout=120)
    assert float(full["format"]["duration"]) == pytest.approx(0.512, abs=0.002)
    assert [s["codec_type"] for s in full["streams"]] == ["video", "audio"]
    assert full["streams"][1]["sample_rate"] == "16000"
    assert full["streams"][1]["channels"] >= 1


def test_ffprobe_types_are_kept_strings_where_the_tool_prints_strings(
        no_binary, clip):
    doc = ffp.probe_json(clip, extra_args=("-show_streams", "-show_format"))
    video, audio = doc["streams"]
    assert isinstance(video["nb_frames"], str) and isinstance(video["duration"], str)
    assert isinstance(video["width"], int) and isinstance(video["index"], int)
    assert video["r_frame_rate"] == "24/1" and video["avg_frame_rate"] == "24/1"
    assert audio["r_frame_rate"] == "0/0", "ffprobe prints 0/0 for audio"
    assert isinstance(doc["format"]["nb_streams"], int)
    assert isinstance(doc["format"]["size"], str)


def test_selecting_a_stream_kind_the_file_lacks_yields_no_streams(
        no_binary, silent_clip):
    assert ffp.probe_json(silent_clip, "stream=codec_type",
                          select_streams="a:0") == {"streams": []}
    assert ws.ffprobe_clip_fields(silent_clip)["codec_types"] == ["video"]


def test_an_unspecified_colour_tag_is_absent_never_unknown(no_binary, clip):
    video = ffp.probe_json(
        clip, "stream=color_space,color_primaries,color_transfer",
        select_streams="v:0")["streams"][0]
    assert video == {}, "ffprobe omits an unspecified tag; so must the fallback"


class _Context:
    name = "h264"
    pix_fmt = "yuv420p"
    colorspace = 1
    color_primaries = 2
    color_trc = 2


class _Stream:
    """Tonight's libx264 clip as PyAV reports it (measured 2026-09-11)."""
    index = 0
    type = "video"
    width, height = 1472, 832
    codec_context = _Context()
    base_rate = average_rate = Fraction(25, 1)
    time_base = Fraction(1, 12800)
    start_time, duration = 0, 320000
    bit_rate, frames = 589893, 625


def test_the_colour_names_follow_ffprobe_bt709_present_unspecified_absent():
    doc = ffp._stream_document(_Stream(), None)
    assert doc["color_space"] == "bt709"
    assert "color_primaries" not in doc and "color_transfer" not in doc
    assert doc["r_frame_rate"] == "25/1" and doc["nb_frames"] == "625"
    assert doc["duration"] == "25.000000" and doc["bit_rate"] == "589893"
    assert ffp._stream_document(_Stream(), {"width", "nb_frames"}) == {
        "width": 1472, "nb_frames": "625"}


# --------------------------------------------------------------------------- #
# refusals -- the same split the binary path makes
# --------------------------------------------------------------------------- #
def test_a_query_only_ffprobe_can_answer_is_refused_by_name(no_binary, clip):
    with pytest.raises(ffp.FFprobeMissing) as excinfo:
        ffp.probe_json(clip, "stream=width", extra_args=("-show_frames",))
    assert "-show_frames" in str(excinfo.value)
    assert "OTR_FFPROBE" in str(excinfo.value)
    with pytest.raises(ffp.FFprobeMissing):
        ffp.probe_json(clip, "stream=width", select_streams="x:0")
    with pytest.raises(ffp.FFprobeMissing):
        ffp.probe_json(clip, "frame=pkt_pts")


def test_probe_raw_still_refuses_by_name(no_binary):
    with pytest.raises(ffp.FFprobeMissing):
        ffp.probe_raw(["-version"])


def test_unreadable_media_is_a_probe_failure_not_a_missing_tool(
        no_binary, tmp_path):
    with pytest.raises(ffp.FFprobeError) as excinfo:
        ffp.probe_json(str(tmp_path / "gone.mp4"), "stream=width")
    assert not isinstance(excinfo.value, ffp.FFprobeMissing)
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"\x00" * 64)
    with pytest.raises(ffp.FFprobeError) as excinfo:
        ffp.probe_json(str(junk), "stream=width")
    assert not isinstance(excinfo.value, ffp.FFprobeMissing)


def test_without_pyav_the_refusal_names_both_absences(no_binary, clip):
    no_binary.setitem(sys.modules, "av", None)
    with pytest.raises(ffp.FFprobeMissing) as excinfo:
        ffp.probe_json(clip, "stream=width")
    assert "PyAV" in str(excinfo.value) and "OTR_FFPROBE" in str(excinfo.value)


def test_a_present_binary_is_used_and_the_fallback_is_never_consulted(
        monkeypatch, tmp_path):
    binary = tmp_path / "ffprobe.exe"
    binary.write_bytes(b"")
    monkeypatch.setattr(ffp, "resolve_ffprobe", lambda *a, **k: str(binary))
    seen = {}

    def fake_run(argv, **kwargs):
        seen["argv"] = argv
        return subprocess.CompletedProcess(
            argv, 0, '{"streams": [{"width": 1920}]}', "")
    monkeypatch.setattr(ffp.otr_proc, "run", fake_run)

    def never(*a, **k):
        raise AssertionError("the PyAV fallback was consulted with a binary present")
    monkeypatch.setattr(ffp, "_pyav_document", never)
    assert ffp.probe_json("clip.mp4", "stream=width")["streams"][0]["width"] == 1920
    assert seen["argv"][0] == str(binary)


# --------------------------------------------------------------------------- #
# wiring -- the two callers that used to bypass the healed path
# --------------------------------------------------------------------------- #
def test_the_counted_frames_query_goes_through_probe_json_so_it_heals_too():
    source = inspect.getsource(ws.ffprobe_counted_frames)
    assert "probe_json(" in source and "probe_raw(" not in source
    assert '"-count_frames"' in source and '"stream=nb_read_frames"' in source
    assert '"v:0"' in source


def test_the_credits_no_longer_force_a_binary():
    from nodes import otr_credits_roll as credits
    assert not hasattr(credits, "_ffprobe_bin"), "zero callers: ripped, not kept"
    assert "ffprobe=" not in inspect.getsource(credits._probe_video)


def test_pyav_is_a_declared_dependency():
    text = (_REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert any(line.strip().startswith("av>=") for line in text.splitlines()), \
        "requirements.txt must declare PyAV, the measuring tool of a cold install"
