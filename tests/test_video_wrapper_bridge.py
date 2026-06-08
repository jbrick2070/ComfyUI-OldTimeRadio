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
    cmd = wb.ffmpeg_silent_mp4_cmd("o.mp4", 833, 480, 25)
    assert "-an" in cmd                                  # V-1 no audio
    pix_vals = [cmd[k + 1] for k, v in enumerate(cmd) if v == "-pix_fmt"]
    assert pix_vals == ["rgb24", "yuv420p"]              # input raw rgb24, out yuv420p
    assert "bt709" in cmd and cmd[-1] == "o.mp4"
    joined = " ".join(cmd)
    assert "832x480" in joined and "833x480" not in joined  # odd width -> even


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
