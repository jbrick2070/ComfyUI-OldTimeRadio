# -*- coding: utf-8 -*-
"""EVERY registered LTX 2.5 lane must be able to build its graph.

WHY THIS FILE EXISTS, and it is worth reading before deleting a line of it.

On 2026-09-23 commit ``7e309f1b`` added ``_ingraph_upscale`` to
``Ltx25NativeFoleyBase`` and read it from ``Ltx25VideoEngine._build_graph``.
``_build_graph`` is five inheritance levels ABOVE the class that declared the
attribute, so every lane that does not descend from the native foley base raised
``AttributeError`` the moment it built a graph: ``ltx25_video``,
``ltx25_foley_plus``, ``ltx25_mime``, ``ltx25_foley_plus_24gb`` and
``ltx25_foley_plus_32gb``. Five working lanes, dead.

THE WHOLE SUITE PASSED. Every fixture, every roster, every parity check, the
widget audits, the AST parse -- all green, because no test anywhere built a graph
for those five lanes. The commit message claimed "BLAST RADIUS: additive" and
that claim was false. It was caught by a review lane reading the inheritance
chain, which is exactly the mechanism CLAUDE.md's 2026-09-07 entry describes: a
decorator that bound to the wrong object, valid syntax, green tests, a feature
gone from every dropdown on every platform.

SO THIS TEST DOES THE DUMBEST POSSIBLE THING AND THAT IS THE POINT. It asks each
registered LTX 2.5 lane to build its own graph and fails if the call raises. It
asserts almost nothing about the result, because the value here is coverage of
the CALL, not of the output -- a richer assertion on three lanes would have
missed this, and a trivial assertion on all of them catches it.
"""
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(_ROOT.parent.parent))


def _registry():
    import nodes._otr_video_engines  # noqa: F401 -- populates the registry
    from nodes._otr_video_engines import registry as vreg
    return vreg


def _ltx25_lane_ids():
    """Every registered engine whose class comes from the LTX 2.5 module.

    DERIVED, NEVER LISTED. A hand-written list is the defect this file exists to
    catch -- a new lane that forgot to join it would be exactly as invisible as
    the five broken ones were.
    """
    vreg = _registry()
    ids = []
    for name in vreg.all_engine_names():
        try:
            engine = vreg.get_engine(name)
        except Exception:                    # noqa: BLE001 -- a different fault
            continue
        module = type(engine).__module__
        if module.endswith("eng_ltx25"):
            ids.append(name)
    return sorted(ids)


def _silent_wav(directory):
    """A real, tiny, valid WAV on disk. 16-bit mono, 1024 silent frames.

    The audio-in lane STAGES its waveform into ComfyUI's input directory, so a
    plausible filename is not enough -- ``stage_into_comfy_input`` raises "input
    file missing" on a path that does not exist. Written with the stdlib ``wave``
    module so the file is genuinely well-formed rather than a renamed blob.
    """
    import wave
    path = directory / "beat_audio.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44100)
        handle.writeframes(b"\x00\x00" * 1024)
    return path


def _plan_for(engine, tmp_path):
    """A minimal beat plan that satisfies THIS lane's declared required_inputs.

    THE AUDIO-IN LANES REFUSE A BEAT WITH NO WAVEFORM, on purpose and with a good
    reason -- "quietly generating one instead would ship a foley clip under an
    audio-in name" -- so a one-size plan made this file report those lanes broken
    while the engine was behaving exactly as designed. The test was wrong, not the
    code, and it took two corrections to get right:

    1. the key `_build_graph` reads is ``audio_path``, not the ``audio_ref`` the
       lane DECLARES in `required_inputs`; an earlier plan step resolves one into
       the other, so satisfying the declared contract was not enough;
    2. the value has to be a FILE THAT EXISTS, because the lane stages it into
       ComfyUI's input directory before wiring it.

    Built from `required_inputs` so a lane that adds a new one fails loudly here
    rather than being quietly under-tested.
    """
    plan = {"text_prompt": "an operator working the console", "seed": 7}
    required = getattr(engine, "required_inputs", ()) or ()
    if "audio_ref" in required:
        plan["audio_path"] = str(_silent_wav(tmp_path))
    return plan


def test_there_are_ltx25_lanes_to_check_at_all():
    """A discovery bug here would make every test below vacuously pass."""
    ids = _ltx25_lane_ids()
    assert len(ids) >= 8, (
        "expected the LTX 2.5 family; found %r. If the module moved, fix the "
        "discovery in _ltx25_lane_ids -- do not delete the tests." % (ids,))


@pytest.mark.parametrize("engine_id", _ltx25_lane_ids())
def test_the_lane_can_build_its_graph(engine_id, tmp_path):
    """The call must not raise. That is the entire contract.

    An AttributeError here means a class attribute is declared BELOW the method
    that reads it. Put it on the class whose method reads it.
    """
    from nodes._otr_video_engines import ltx25_recipe as R
    engine = _registry().get_engine(engine_id)

    graph = engine._build_graph(
        _plan_for(engine, tmp_path), "beat_still.png", R.LTX25_FRAMES,
        R.LTX25_CANVAS_W, R.LTX25_CANVAS_H)

    assert isinstance(graph, dict) and graph, (
        "%s built %r" % (engine_id, type(graph).__name__))
    assert engine._TERMINAL in graph, (
        "%s built a graph with no terminal node %r; keys=%r"
        % (engine_id, engine._TERMINAL, sorted(graph)))


@pytest.mark.parametrize("engine_id", _ltx25_lane_ids())
def test_every_wire_points_at_a_node_that_exists(engine_id, tmp_path):
    """A pruned node leaves dangling wires, and a dangling wire is a crash.

    The low-res lane removes six nodes from the graph. If any surviving input
    still referenced one of them the executor would fail at run time with a
    missing-node error -- late, on the GPU, after the models had loaded. This
    catches it in a second, on CPU.
    """
    from nodes._otr_video_engines import ltx25_recipe as R
    engine = _registry().get_engine(engine_id)
    graph = engine._build_graph(
        _plan_for(engine, tmp_path), "beat_still.png", R.LTX25_FRAMES,
        R.LTX25_CANVAS_W, R.LTX25_CANVAS_H)

    dangling = []
    for node_id, spec in graph.items():
        for field, value in (spec.get("inputs") or {}).items():
            # A wire is (source_node_id, slot_index).
            if (isinstance(value, (list, tuple)) and len(value) == 2
                    and isinstance(value[0], str) and value[0] not in graph):
                dangling.append("%s.%s -> %r" % (node_id, field, value[0]))
    assert not dangling, "%s has dangling wires: %s" % (engine_id, dangling)


def test_only_the_low_res_lane_skips_the_ingraph_upscale():
    """The flag's default must stay True, or every HQ lane silently degrades.

    Written as a membership rule rather than a count: a tally goes stale on the
    next lane, and this file is precisely about what goes stale invisibly.
    """
    vreg = _registry()
    skipping = {name for name in _ltx25_lane_ids()
                if not getattr(vreg.get_engine(name), "_ingraph_upscale", True)}
    assert skipping == {"ltx25_native_foley_lowres"}, (
        "exactly one lane should decode at its native canvas; these do: %r"
        % (sorted(skipping),))


def test_the_flag_is_declared_on_the_class_that_reads_it():
    """The structural fix, asserted structurally.

    ``_build_graph`` lives on ``Ltx25VideoEngine``, so the default must live
    there too. If someone moves it back down to a subclass, every lane that does
    not inherit that subclass breaks again -- and no behavioural test in this
    file would necessarily say WHY.
    """
    from nodes._otr_video_engines.eng_ltx25 import Ltx25VideoEngine
    assert "_ingraph_upscale" in vars(Ltx25VideoEngine), (
        "_ingraph_upscale must be declared on Ltx25VideoEngine itself, because "
        "that is the class whose _build_graph reads it. Declaring it on a "
        "subclass leaves every sibling lane raising AttributeError.")
    assert vars(Ltx25VideoEngine)["_ingraph_upscale"] is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
