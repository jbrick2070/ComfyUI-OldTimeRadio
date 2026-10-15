"""A floored voice line keeps its planned length through the sequencer.

Plan item 6, 2026-10-14. A classified cloud-TTS failure is replaced by silence
the length of the line (floored_line_silence), so the episode still assembles.
pack_audio_batch then pads every clip to the longest, and SceneSequencer trims
each row's trailing silence -- a floor is ALL silence, so it was cut to 100
samples, the next line started early and the episode came up short. The voice
node now marks each floor's exact span on the AUDIO dict; the sequencer keeps
exactly that span and trims every other row as before.

CPU-only: the real producer, the real packer and the real sequencer.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from nodes import _otr_voice_node_common as voice
from nodes._otr_audio_engines.base import pack_audio_batch
from nodes.scene_sequencer import SceneSequencer
from tests._support.ledger_stub import make_stub_ledger

REPO = Path(__file__).resolve().parents[1]
SR = 24000
FLOOR_TEXT = "this whole line never came back from the provider at all"


def _tone(seconds, amp=0.2, sr=SR):
    return {"waveform": torch.full((1, 1, int(seconds * sr)), amp), "sample_rate": sr}


def _bus(*items):
    """('tone', seconds) / ('floor', text) rows, marked as the voice node does."""
    clips, spans = [], []
    for kind, value in items:
        if kind == "floor":
            clips.append(voice.floored_line_silence(value, SR))
            spans.append(voice.floor_span(clips, "L%d" % len(clips), SR))
        else:
            clips.append(_tone(value) if not isinstance(value, dict) else value)
    packed = pack_audio_batch(clips, sample_rate=SR, mono=True)
    return voice.attach_floor_spans(packed, spans), clips


def _lengths(audio):
    return [len(c) for c, _sr in SceneSequencer()._extract_clips_from_audio(audio)]


def _floor_samples(text=FLOOR_TEXT):
    return int(voice.floored_line_silence(text, SR)["waveform"].shape[-1])


def test_a_floor_between_healthy_lines_keeps_its_exact_span():
    packed, _ = _bus(("tone", 0.5), ("floor", FLOOR_TEXT), ("tone", 1.25))
    unmarked = {k: v for k, v in packed.items() if k != voice.FLOOR_SPANS_KEY}
    got, legacy = _lengths(packed), _lengths(unmarked)
    assert got[1] == _floor_samples()          # the planned pause, all of it
    assert legacy[1] == 100                     # what the trim used to leave
    assert got[0] == legacy[0] and got[2] == legacy[2]   # healthy rows: as before


def test_differently_sized_floors_each_keep_their_own_span_never_the_batch_max():
    short, long = "one two three", " ".join(["word"] * 60)
    packed, _ = _bus(("floor", short), ("tone", 3.0), ("floor", long))
    got = _lengths(packed)
    assert got[0] == _floor_samples(short)
    assert got[2] == _floor_samples(long)
    assert got[0] < int(packed["waveform"].shape[-1])


def test_a_floor_keeps_its_span_when_a_neighbour_is_resampled():
    packed, _ = _bus(("tone", _tone(0.5, sr=48000)), ("floor", FLOOR_TEXT))
    assert packed["sample_rate"] == SR
    assert _lengths(packed)[1] == _floor_samples()


def test_no_floor_means_no_marker_and_the_old_trim():
    packed, _ = _bus(("tone", 0.5), ("tone", 0.75))
    assert voice.FLOOR_SPANS_KEY not in packed


@pytest.mark.parametrize("mutate", [
    lambda s: s.update(sample_rate=48000),
    lambda s: s.update(index=99),
    lambda s: s.update(index=-1),
    lambda s: s.update(samples=10 ** 9),
    lambda s: s.update(samples=0),
    lambda s: s.update(samples="400"),
    lambda s: s.pop("samples"),
])
def test_a_marker_that_does_not_fit_falls_back_whole_and_never_raises(mutate):
    packed, _ = _bus(("tone", 0.5), ("floor", FLOOR_TEXT))
    mutate(packed[voice.FLOOR_SPANS_KEY][0])
    assert _lengths(packed)[1] == 100


def test_a_duplicate_row_in_the_marker_falls_back():
    packed, _ = _bus(("floor", FLOOR_TEXT), ("tone", 0.5))
    packed[voice.FLOOR_SPANS_KEY].append(dict(packed[voice.FLOOR_SPANS_KEY][0]))
    assert _lengths(packed)[0] == 100


def test_the_voice_node_marks_every_floor_and_attaches_after_packing():
    """The wiring, read where it lives (structurally, not by whitespace): the
    floor branch of the commit loop records a span right after flooring, the
    healthy branch never does, and the packed batch is what gets the spans."""
    import ast
    tree = ast.parse((REPO / "nodes" / "_otr_voice_node_common.py").read_text(
        encoding="utf-8"))

    def calls(node):
        names = set()
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                f = sub.func
                names.add(f.id if isinstance(f, ast.Name) else getattr(f, "attr", ""))
        return names

    floor_branches = [n for n in ast.walk(tree)
                      if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
                      and n.test.id == "_floor_why"]
    assert len(floor_branches) == 1
    branch = ast.Module(body=floor_branches[0].body, type_ignores=[])
    assert "_floor_voice_line" in calls(branch)

    def is_span_append(node):
        """``_floor_spans.append(floor_span(...))`` -- exactly that list,
        exactly that value, so a computed-and-dropped span cannot pass."""
        return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "append"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "_floor_spans"
                and len(node.args) == 1 and isinstance(node.args[0], ast.Call)
                and getattr(node.args[0].func, "id", "") == "floor_span")

    assert any(is_span_append(n) for n in ast.walk(branch))
    # ...as a TOP-LEVEL statement of the branch, right after the floor is
    # appended to clips, with the real clips and sample rate -- not before it,
    # not after the `continue`, not inside a dead `if`.
    body = floor_branches[0].body
    stmts = [s.value if isinstance(s, ast.Expr) else None for s in body]
    i_floor = next(i for i, c in enumerate(stmts) if isinstance(c, ast.Call)
                   and getattr(c.func, "id", "") == "_floor_voice_line")
    i_span = next(i for i, c in enumerate(stmts) if c is not None and is_span_append(c))
    i_cont = next((i for i, s in enumerate(body) if isinstance(s, ast.Continue)), len(body))
    assert i_floor < i_span < i_cont
    span_args = stmts[i_span].args[0].args
    assert [getattr(a, "id", None) for a in (span_args[0], span_args[2])] == ["clips", "sr"]
    # Nothing outside that branch records a span.
    assert sum(1 for n in ast.walk(tree) if is_span_append(n)) == 1
    assert sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call)
               and getattr(n.func, "id", "") == "floor_span") == 1
    attaches = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and getattr(n.func, "id", "") == "attach_floor_spans"]
    assert len(attaches) == 1
    packed, spans = attaches[0].args
    assert isinstance(packed, ast.Call) and getattr(packed.func, "id", "") == "pack_audio_batch"
    assert isinstance(spans, ast.Name) and spans.id == "_floor_spans"


@pytest.fixture
def hermetic_sequencer(tmp_path):
    state = {"led_disk": None}
    with patch("nodes._otr_model_loader.unload_llm", side_effect=lambda *a, **k: None), \
         patch("nodes.scene_sequencer._runtime_log", side_effect=lambda *a, **k: None), \
         patch("nodes._otr_ledger.in_flight_ledger_path",
               side_effect=lambda: tmp_path / "stub_ledger.json"), \
         patch("nodes._otr_ledger.load_ledger_safe",
               side_effect=lambda _p: state["led_disk"]), \
         patch("nodes._otr_ledger.save_ledger_safe",
               side_effect=lambda _p, led: state.update(led_disk=led) or True):
        yield state


def test_the_assembled_timeline_holds_the_floor_and_starts_the_next_line_after_it(
        hermetic_sequencer):
    led = make_stub_ledger()      # music_open, announcer b002, b003, b004, music_close
    hermetic_sequencer["led_disk"] = json.loads(json.dumps(led))
    character, _ = _bus(("floor", FLOOR_TEXT), ("tone", 0.75))
    announcer, _ = _bus(("tone", 0.5))
    SceneSequencer().sequence(script_json=json.dumps(led),
                              tts_audio_clips=character,
                              announcer_audio_clips=announcer)
    rows = {ln["line_id"]: ln for ln in hermetic_sequencer["led_disk"]["lines"]}
    floor_s = _floor_samples() / SR
    assert rows["b003"]["dur_s"] == pytest.approx(floor_s, abs=0.01)
    assert rows["b004"]["start_s"] >= rows["b003"]["start_s"] + floor_s - 0.01
