"""CPU tests for the shared in-process wrapper bridge (A-ship forward mechanics).

The in-process motion forwards (humo / ltx_video / wan_i2v) and the radio floor
all run through nodes/_otr_video_engines/wrapper_bridge: node-class resolution,
the generic declarative graph executor, the 4n+1 quantizer, IMAGE->uint8, and the
silent bt709 / yuv420p ffmpeg encode. Every mechanic is proven here on the CPU box
(the real wrapper-node forward is the operator GPU slice). The ffmpeg round-trips
run when ffmpeg / ffprobe are on PATH (they are on the build box) and skip cleanly
otherwise. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

from nodes._otr_video_engines import wrapper_bridge as wb

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_HAS_FFMPEG = shutil.which("ffmpeg") is not None
_HAS_FFPROBE = shutil.which("ffprobe") is not None


# --- fake ComfyUI node classes for the executor ---------------------------- #
class _Add:
    FUNCTION = "go"

    def go(self, a, b):
        return (a + b,)


class _Double:
    FUNCTION = "run"

    def run(self, x):
        return (x * 2,)


class _Pair:
    FUNCTION = "run"

    def run(self, x):
        return (x, x + 1)              # two outputs (slot 0, slot 1)


class _NoTuple:
    FUNCTION = "run"

    def run(self, x):
        return x * 3                   # bare return normalised to a 1-tuple


# --- resolution ------------------------------------------------------------ #
def test_resolve_node_class_first_present_wins():
    mapping = {"B": object, "C": list}
    assert wb.resolve_node_class(("A", "B", "C"), mapping=mapping) is mapping["B"]
    assert wb.resolve_node_class("C", mapping=mapping) is mapping["C"]


def test_resolve_node_class_missing_is_named():
    with pytest.raises(wb.WrapperNodeMissing) as e:
        wb.resolve_node_class(("X", "Y"), mapping={})
    assert "X" in str(e.value) and "Y" in str(e.value)


def test_resolve_graph_classes_aggregates_missing():
    mapping = {"Have": object}
    with pytest.raises(wb.WrapperNodeMissing) as e:
        wb.resolve_graph_classes(
            {"a": "Have", "b": ("Miss1", "Miss2"), "c": "Gone"}, mapping=mapping)
    msg = str(e.value)
    assert "b=" in msg and "c=" in msg and "a=" not in msg
    ok = wb.resolve_graph_classes({"a": "Have"}, mapping=mapping)
    assert ok == {"a": mapping["Have"]}


def test_node_class_mappings_returns_dict_and_empty_fails_closed():
    assert isinstance(wb.node_class_mappings(), dict)
    with pytest.raises(wb.WrapperNodeMissing):
        wb.resolve_node_class("Anything", mapping={})


# --- generic executor ------------------------------------------------------ #
def test_run_graph_topo_and_wiring():
    graph = {
        "n1": {"class": _Add, "inputs": {"a": 2, "b": 3}},               # -> 5
        "n2": {"class": _Double, "inputs": {"x": wb.Wire("n1", 0)}},     # -> 10
        "n3": {"class": _Add,
               "inputs": {"a": wb.Wire("n1", 0), "b": wb.Wire("n2", 0)}},  # -> 15
    }
    assert wb.run_graph(graph, terminal="n3") == (15,)
    allr = wb.run_graph(graph)
    assert allr["n1"] == (5,) and allr["n2"] == (10,) and allr["n3"] == (15,)


def test_run_graph_multi_output_slot():
    graph = {
        "p": {"class": _Pair, "inputs": {"x": 7}},                      # (7, 8)
        "d": {"class": _Double, "inputs": {"x": wb.Wire("p", 1)}},      # -> 16
    }
    assert wb.run_graph(graph, terminal="d") == (16,)


def test_run_graph_bare_return_normalised():
    g = {"n": {"class": _NoTuple, "inputs": {"x": 4}}}
    assert wb.run_graph(g, terminal="n") == (12,)


def test_run_graph_class_by_name_from_classes():
    g = {"n": {"class": "Adder", "inputs": {"a": 1, "b": 1}}}
    assert wb.run_graph(g, classes={"Adder": _Add}, terminal="n") == (2,)


def test_run_graph_cycle_is_named():
    g = {"a": {"class": _Double, "inputs": {"x": wb.Wire("b", 0)}},
         "b": {"class": _Double, "inputs": {"x": wb.Wire("a", 0)}}}
    with pytest.raises(wb.GraphExecutionError):
        wb.run_graph(g)


def test_run_graph_dangling_wire_is_named():
    g = {"a": {"class": _Double, "inputs": {"x": wb.Wire("ghost", 0)}}}
    with pytest.raises(wb.GraphExecutionError):
        wb.run_graph(g)


def test_run_graph_unresolved_name_and_node_raise():
    with pytest.raises(wb.GraphExecutionError):
        wb.run_graph({"n": {"class": "Missing", "inputs": {}}},
                     classes={}, terminal="n")

    class _Boom:
        FUNCTION = "run"

        def run(self):
            raise ValueError("boom")

    with pytest.raises(wb.GraphExecutionError) as e:
        wb.run_graph({"n": {"class": _Boom, "inputs": {}}})
    assert "boom" in str(e.value)


def test_wire_accessors():
    w = wb.Wire("src", 2)
    assert w.src == "src" and w.slot == 2 and tuple(w) == ("src", 2)


# --- quantize / dims ------------------------------------------------------- #
def test_quantize_frames_4n1():
    assert wb.quantize_frames_4n1(97) == 97
    assert wb.quantize_frames_4n1(96) == 97
    assert wb.quantize_frames_4n1(98) == 101
    assert wb.quantize_frames_4n1(1) == 1
    assert wb.quantize_frames_4n1(5, min_frames=33) == 33
    capped = wb.quantize_frames_4n1(1000, max_frames=177)
    assert capped == 177 and (capped - 1) % 4 == 0
    assert (wb.quantize_frames_4n1(50) - 1) % 4 == 0


def test_even_dim():
    assert wb.even_dim(833) == 832
    assert wb.even_dim(0) == 2 and wb.even_dim(1) == 2
    assert wb.even_dim(480) == 480


# --- images_to_uint8 ------------------------------------------------------- #
def test_images_to_uint8_from_float_numpy():
    np = pytest.importorskip("numpy")
    img = np.zeros((2, 4, 4, 3), dtype="float32")
    img[..., 0] = 1.0
    img[..., 1] = 0.5
    out = wb.images_to_uint8(img)
    assert out.dtype == np.uint8 and out.shape == (2, 4, 4, 3)
    assert out[0, 0, 0, 0] == 255 and out[0, 0, 0, 1] == 128
    single = wb.images_to_uint8(np.zeros((4, 4, 3), dtype="float32"))
    assert single.shape == (1, 4, 4, 3)


# --- ffmpeg cmd builders (pure) -------------------------------------------- #
def test_ffmpeg_silent_cmd_contract():
    cmd = wb.ffmpeg_silent_mp4_cmd("o.mp4", 832, 480, 25)
    assert "-an" in cmd                                  # V-1 no audio
    pix_vals = [cmd[k + 1] for k, v in enumerate(cmd) if v == "-pix_fmt"]
    assert pix_vals == ["rgb24", "yuv420p"]              # input raw rgb24, out yuv420p
    assert "bt709" in cmd and cmd[-1] == "o.mp4"
    assert "832x480" in " ".join(cmd)


def test_the_declared_size_is_the_PIPED_size_never_rounded():
    """This assertion used to read the other way round -- it REQUIRED that an
    833-wide ask be declared as 832 ("odd width -> even"), which is the defect
    written down as the contract.

    The ``-s`` describes the INPUT byte stream, and
    ``encode_frames_to_silent_mp4`` pipes ``frames.tobytes()`` at the array's
    real width. Declaring 832 while piping 833-pixel rows makes ffmpeg slice
    every row on the wrong boundary: measured, a (5,63,47,3) batch wrote a
    46x62 clip of skewed pixels and PASSED the frame-count proof, because five
    frames went in and five came out. Rounding is now the caller's business
    and an odd canvas is refused by name at the encoder."""
    joined = " ".join(wb.ffmpeg_silent_mp4_cmd("o.mp4", 833, 481, 25))
    assert "833x481" in joined
    assert "832x480" not in joined
    # The three builders that SCALE or PAD to a target keep even_dim: there
    # ffmpeg is told what to PRODUCE, and yuv420p requires even. Asserted so
    # the two cases cannot be "simplified" into one. (They spell their target
    # as an ffmpeg filter argument, `scale=W:H`, not as a `-s WxH`.)
    static = wb.ffmpeg_still_static_cmd("s.png", "o.mp4", 833, 481, 25, 10)
    vf = static[static.index("-vf") + 1]
    assert vf.startswith("scale=832:480:"), vf
    assert "833" not in vf and "481" not in vf


def test_floor_and_still_cmd_frame_count():
    floor = wb.ffmpeg_lavfi_floor_cmd("o.mp4", 320, 240, 25, 13)
    assert floor[floor.index("-frames:v") + 1] == "13"
    assert "-an" in floor and "yuv420p" in floor
    still = wb.ffmpeg_still_motion_cmd("in.png", "o.mp4", 320, 240, 25, 10)
    assert still[still.index("-frames:v") + 1] == "10"
    assert "zoompan" in " ".join(still) and "-an" in still


def test_encode_rejects_bad_shape(tmp_path):
    np = pytest.importorskip("numpy")
    with pytest.raises(wb.GraphExecutionError):
        wb.encode_frames_to_silent_mp4(
            np.zeros((4, 4), dtype="uint8"), str(tmp_path / "x.mp4"), 25)


# --- ffmpeg round-trips (the leaf; only when ffmpeg/ffprobe present) -------- #
def _probe(path, *entries):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=" + ",".join(entries),
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True)
    return out.stdout.split()


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_encode_frames_roundtrip(tmp_path):
    np = pytest.importorskip("numpy")
    frames = np.full((5, 64, 48, 3), 120, dtype="uint8")
    out = tmp_path / "clip.mp4"
    path, n = wb.encode_frames_to_silent_mp4(frames, str(out), 25)
    assert n == 5 and out.exists() and out.stat().st_size > 0
    fields = _probe(out, "nb_read_frames", "pix_fmt")
    assert "yuv420p" in fields and "5" in fields
    a = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=index", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True)
    assert a.stdout.strip() == ""                        # V-1: silent clip


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_lavfi_floor_roundtrip(tmp_path):
    out = tmp_path / "floor.mp4"
    wb.run_ffmpeg(wb.ffmpeg_lavfi_floor_cmd(str(out), 96, 64, 25, 8))
    assert out.exists() and out.stat().st_size > 0
    assert "8" in _probe(out, "nb_read_frames")


# --- cold-import + ASCII / no BOM / no em-dash ----------------------------- #
def test_cold_import_bridge_no_heavy_libs():
    code = ("import sys;"
            "import nodes._otr_video_engines.wrapper_bridge;"
            "heavy=[m for m in ('torch','transformers','diffusers','numpy') "
            "if m in sys.modules];"
            "print('HEAVY', heavy); sys.exit(1 if heavy else 0)")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs at import:\n{r.stdout}\n{r.stderr}"


def test_bridge_source_ascii_no_bom_no_em_dash():
    p = REPO_ROOT / "nodes" / "_otr_video_engines" / "wrapper_bridge.py"
    raw = p.read_bytes()
    assert raw[:3] != b"\xef\xbb\xbf"                    # no BOM
    src = raw.decode("utf-8")
    assert chr(0x2014) not in src                        # no em-dash
    src.encode("ascii")                                  # ASCII-only


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# --------------------------------------------------------------------------- #
# The encoder returns a PROVEN frame count, not the input array's length
# (2026-07-28). The M7 sweep at 58e288af proved the colour/stream contract on
# every emitted clip but NOT the COUNT: encode_frames_to_silent_mp4 returned
# int(len(frames)), and for a single-render beat that number IS
# CanonicalClip.frame_count -- the integer timing authority. Multi-segment
# beats were already covered by assemble_beat_segments' decode.
#
# It costs no decode: nb_frames rides the stream read ffprobe_clip_fields
# already performs on every clip. The decode is the FALLBACK, for a container
# that records no count at all.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_the_container_reports_its_own_count_to_ffprobe_clip_fields(tmp_path):
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import wan_shared as ws
    out = tmp_path / "clip.mp4"
    wb.encode_frames_to_silent_mp4(
        np.full((7, 64, 48, 3), 90, dtype="uint8"), str(out), 25)
    assert ws.ffprobe_clip_fields(str(out))["nb_frames"] == 7


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_a_stream_with_no_count_reports_None_and_not_zero(tmp_path):
    """None means "no count here". Zero would mean "an empty clip", and the
    encoder boundary would then refuse every good clip it could not read a
    header from."""
    from nodes._otr_video_engines import wan_shared as ws
    aud = tmp_path / "silence.m4a"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "anullsrc=r=44100:cl=mono", "-t", "0.2", str(aud)],
                   check=True, capture_output=True)
    assert ws.ffprobe_clip_fields(str(aud))["nb_frames"] is None


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_a_count_the_file_does_not_support_is_REFUSED(tmp_path):
    """THE DEFECT, stated at the boundary: a declared count the clip does not
    hold must not become the timing authority."""
    np = pytest.importorskip("numpy")
    out = tmp_path / "clip.mp4"
    wb.encode_frames_to_silent_mp4(
        np.full((5, 64, 48, 3), 90, dtype="uint8"), str(out), 25)
    with pytest.raises(wb.GraphExecutionError,
                       match="timing authority") as exc_info:
        wb.proven_frame_count(str(out), 9)
    # ...and the refusal names BOTH numbers, so the log says what disagreed.
    # Read off exc_info, NOT a second bare try/except: a try/except whose
    # asserts live in the handler passes vacuously the day the code stops
    # raising, which is the one regression this test exists to catch.
    assert "5 frame(s)" in str(exc_info.value)
    assert "9 were piped" in str(exc_info.value)


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_zero_frames_is_refused_by_name_at_the_encoder(tmp_path):
    """ndim==4 is satisfied by a batch of ZERO frames and ffmpeg exits 0 on
    the empty pipe, writing a container with no video stream. The old encoder
    returned frame_count=0 for that and the beat carried on."""
    np = pytest.importorskip("numpy")
    out = tmp_path / "clip.mp4"
    with pytest.raises(wb.GraphExecutionError, match="ZERO frames") as info:
        wb.encode_frames_to_silent_mp4(
            np.zeros((0, 64, 48, 3), dtype="uint8"), str(out), 25)
    # ...and it does NOT describe itself as a failed multi-segment assembly,
    # which is what the count proof one call later would have said.
    assert "assembled beat" not in str(info.value)


@pytest.mark.skipif(not (_HAS_FFMPEG and _HAS_FFPROBE),
                    reason="ffmpeg/ffprobe not on PATH")
def test_the_normal_path_never_pays_for_a_decode(monkeypatch, tmp_path):
    """The reason this proof can be unconditional.

    Detonating the decoder is only half the test: the pre-2026-07-28 encoder
    probed NOTHING and would also pass it. So the header read is COUNTED --
    exactly once, which is what distinguishes "used the free proof" from
    "never proved anything"."""
    np = pytest.importorskip("numpy")
    from nodes._otr_video_engines import wan_shared as ws
    seen = {"header": 0}
    real_fields = ws.ffprobe_clip_fields

    def _spy(*a, **k):
        seen["header"] += 1
        return real_fields(*a, **k)

    def _boom(*_a, **_k):
        raise AssertionError("decoded per clip in the render loop")

    monkeypatch.setattr(ws, "ffprobe_clip_fields", _spy)
    monkeypatch.setattr(ws, "ffprobe_counted_frames", _boom)
    out = tmp_path / "clip.mp4"
    _p, n = wb.encode_frames_to_silent_mp4(
        np.full((11, 64, 48, 3), 90, dtype="uint8"), str(out), 25)
    assert n == 11
    assert seen["header"] == 1


def test_a_container_with_no_count_falls_back_to_the_decode(monkeypatch):
    """A count that cannot be read is not an excuse to return the array's
    length -- that is the number this whole change exists to stop trusting."""
    from nodes._otr_video_engines import wan_shared as ws
    monkeypatch.setattr(ws, "ffprobe_clip_fields",
                        lambda *_a, **_k: {"nb_frames": None})
    monkeypatch.setattr(ws, "ffprobe_counted_frames", lambda *_a, **_k: 33)
    assert wb.proven_frame_count("x.mp4", 33) == 33
    with pytest.raises(wb.GraphExecutionError, match="timing authority"):
        wb.proven_frame_count("x.mp4", 34)


def test_a_clip_LONGER_than_it_was_piped_is_refused_too(monkeypatch):
    """A beat with MORE frames than were piped drifts exactly as badly as a
    short one -- the check is a disagreement, not a shortfall. Found by the
    mutation round: every fixture until now had counted < declared, so a
    refusal that only caught short clips stayed green."""
    from nodes._otr_video_engines import wan_shared as ws
    monkeypatch.setattr(ws, "ffprobe_clip_fields",
                        lambda *_a, **_k: {"nb_frames": 40})
    with pytest.raises(wb.GraphExecutionError, match="timing authority"):
        wb.proven_frame_count("x.mp4", 33)


def test_the_fallback_says_which_proof_it_used(monkeypatch):
    """A receipt that cannot say HOW it knows is the mux defect at 54b3626b.

    Both blocks REQUIRE the raise (pytest.raises, not a bare try/except): a
    handler-only assertion passes vacuously the day the refusal stops firing,
    and this test would then be guarding nothing at all."""
    from nodes._otr_video_engines import wan_shared as ws
    monkeypatch.setattr(ws, "ffprobe_clip_fields",
                        lambda *_a, **_k: {"nb_frames": None})
    monkeypatch.setattr(ws, "ffprobe_counted_frames", lambda *_a, **_k: 33)
    with pytest.raises(wb.GraphExecutionError, match="timing authority") as a:
        wb.proven_frame_count("x.mp4", 34)
    assert "decode" in str(a.value)
    monkeypatch.setattr(ws, "ffprobe_clip_fields",
                        lambda *_a, **_k: {"nb_frames": 33})
    with pytest.raises(wb.GraphExecutionError, match="timing authority") as b:
        wb.proven_frame_count("x.mp4", 34)
    assert "container" in str(b.value) and "decode" not in str(b.value)


def test_the_encoder_returns_what_the_boundary_proved(monkeypatch, tmp_path):
    """CONTROL against the obvious wrong fix: returning int(b) and calling
    proven_frame_count only for its side effect. The returned value must BE
    the proof's answer."""
    np = pytest.importorskip("numpy")
    seen = {}

    def _fake(out_path, declared, **_k):
        seen["declared"] = declared
        return 4242

    monkeypatch.setattr(wb, "proven_frame_count", _fake)
    monkeypatch.setattr(wb, "run_ffmpeg", lambda cmd: cmd)

    class _P:
        returncode = 0

        def communicate(self, _data):
            return b"", b""

    out = tmp_path / "clip.mp4"
    out.write_bytes(b"x")
    monkeypatch.setattr(wb.otr_proc, "popen", lambda *_a, **_k: _P())
    _p, n = wb.encode_frames_to_silent_mp4(
        np.zeros((6, 8, 8, 3), dtype="uint8"), str(out), 25)
    assert seen["declared"] == 6            # it was told the array's length
    assert n == 4242                        # ...and returned the PROOF's
