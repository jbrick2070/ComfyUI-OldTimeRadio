# -*- coding: utf-8 -*-
"""The 8 GB LTX 2.5 lanes run stage one only and decode at 832x480.

WHY (4060 heavy runs, 2026-10-15): every 8 GB LTX 2.5 leg reached its first
shot and died with an out-of-memory at ``decode``. The two-stage recipe
decodes at 1664x960 after a 2x latent upscale, measured at 8,080 MB on a 16 GB
card, and an 8 GB 4060 offers ~7,400 MB. Operator ruling the same day: the
8 GB lanes skip the upscale; the 16 GB lanes stay exactly as they were.
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent.parent))

import nodes._otr_video_engines  # noqa: E402,F401 -- populates the registry
from nodes._otr_video_engines import eng_ltx25 as E  # noqa: E402
from nodes._otr_video_engines import ltx25_recipe as R  # noqa: E402
from nodes._otr_video_engines import registry as vreg  # noqa: E402

#: 8 GB lane -> the 16 GB lane it must equal in everything but stage two.
PAIRS = {
    "ltx25_video_8gb": "ltx25_video",
    "ltx25_foley_8gb": "ltx25_foley_16gb",
    "ltx25_mime_8gb": "ltx25_mime_16gb",
    "ltx25_audio_in_8gb": "ltx25_audio_in_16gb",
}


def _plan(engine, tmp_path):
    plan = {"text_prompt": "an operator working the console", "seed": 7}
    if "audio_ref" in (getattr(engine, "required_inputs", ()) or ()):
        import wave
        path = tmp_path / "beat_audio.wav"
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(44100)
            handle.writeframes(b"\x00\x00" * 1024)
        plan["audio_path"] = str(path)
    return plan


def _graph(engine, tmp_path):
    return engine._build_graph(_plan(engine, tmp_path), "beat_still.png",
                               R.LTX25_FRAMES, R.LTX25_CANVAS_W,
                               R.LTX25_CANVAS_H)


@pytest.mark.parametrize("lane,parent", sorted(PAIRS.items()))
def test_the_8gb_lane_is_its_16gb_parent_minus_stage_two(lane, parent,
                                                         tmp_path):
    small, big = vreg.get_engine(lane), vreg.get_engine(parent)
    assert small._two_stage is False and big._two_stage is True
    g_small, g_big = _graph(small, tmp_path), _graph(big, tmp_path)
    stage_two = set(E.Ltx25VideoEngine._STAGE_TWO_NODES)
    assert stage_two <= set(g_big)
    assert not stage_two & set(g_small)
    # Everything else is the parent's graph, node for node.
    assert set(g_small) == set(g_big) - stage_two
    assert g_small["decode"]["inputs"]["samples"] == ("separate", 0)
    assert g_big["decode"]["inputs"]["samples"] == ("refine_separate", 0)
    # Same weights and the same tier formatter.
    assert small._dit_name() == big._dit_name()
    # render_driver reads the formatter off the class's own __dict__, so the
    # 8 GB lane must bind the same one its parent binds (None for audio-in).
    assert (type(small).__dict__.get("compose_prompt")
            is type(big).__dict__.get("compose_prompt"))


@pytest.mark.parametrize("lane", sorted(PAIRS))
def test_the_8gb_lane_needs_no_upscaler_and_decodes_832x480(lane):
    engine = vreg.get_engine(lane)
    labels = [label for label, _path, _floor in engine._weight_paths()]
    assert not [x for x in labels if "upscaler" in x], labels
    assert engine._output_canvas() == (R.LTX25_CANVAS_W, R.LTX25_CANVAS_H)
    assert not set(E.Ltx25VideoEngine._STAGE_TWO_NODES) & set(
        engine._node_candidates())


def test_the_16gb_lanes_are_untouched():
    for parent in PAIRS.values():
        engine = vreg.get_engine(parent)
        labels = [label for label, _p, _f in engine._weight_paths()]
        assert [x for x in labels if "upscaler" in x], parent
        assert engine._output_canvas() == (R.LTX25_RENDER_CANVAS_W,
                                           R.LTX25_RENDER_CANVAS_H)


def _records(audit, shape):
    out = []
    for ordinal, (node_id, cls) in enumerate(audit, 1):
        out.append({"node_id": node_id, "class_name": cls,
                    "ordinal": ordinal})
    out[-1]["output_shapes"] = [shape]
    return out


def test_execution_proof_accepts_each_shape_and_refuses_the_other():
    single = _records(E._SINGLE_STAGE_AUDIT,
                      [97, R.LTX25_CANVAS_H, R.LTX25_CANVAS_W, 3])
    double = _records(E._TWO_STAGE_AUDIT,
                      [97, R.LTX25_RENDER_CANVAS_H, R.LTX25_RENDER_CANVAS_W, 3])
    assert E._assert_two_stage_execution(single, 97, single_stage=True)
    assert E._assert_two_stage_execution(double, 97)
    with pytest.raises(RuntimeError):
        E._assert_two_stage_execution(single, 97)
    with pytest.raises(RuntimeError):
        E._assert_two_stage_execution(double, 97, single_stage=True)
    # A single-stage run that somehow decoded at the doubled canvas is refused.
    wrong = _records(E._SINGLE_STAGE_AUDIT,
                     [97, R.LTX25_RENDER_CANVAS_H, R.LTX25_RENDER_CANVAS_W, 3])
    with pytest.raises(RuntimeError):
        E._assert_two_stage_execution(wrong, 97, single_stage=True)


@pytest.mark.parametrize("lane,source", [("ltx25_foley_8gb", "separate"),
                                         ("ltx25_foley_16gb",
                                          "refine_separate")])
def test_the_foley_harvest_reads_the_node_that_feeds_the_decode(lane, source):
    engine = vreg.get_engine(lane)
    assert engine._final_separate_node() == source
    latent = {"samples": __import__("torch").zeros(1, 2, 3)}
    other = "refine_separate" if source == "separate" else "separate"
    engine._pending_audio_latent = None
    engine._on_graph_result(other, ({"samples": None}, latent))
    assert engine._pending_audio_latent is None
    engine._on_graph_result(source, ({"samples": None}, latent))
    assert engine._pending_audio_latent is not None


def test_the_8gb_workflows_pick_the_8gb_lanes():
    root = _ROOT / "workflows"
    for wf, lane in (("otr_8gb_ltx25_foley.json", "ltx25_foley_8gb"),
                     ("otr_8gb_ltx25_mime.json", "ltx25_mime_8gb"),
                     ("otr_8gb_ltx25_audio_in.json", "ltx25_audio_in_8gb")):
        text = (root / wf).read_text(encoding="utf-8")
        assert text.count('"%s (16:9)"' % lane) == 3, wf
        assert "_16gb (16:9)" not in text, wf
