"""NO-MIRROR, step 2 -- the documents the grader and the composite will read.

Three producers, all landing BEFORE the consumers that need them (the spec's
producer-first order, which exists because an armed consumer with a starved
producer is the ``PBUG-20260805-04`` class):

* ``frame_receipt_version`` on the manifest -- the version that makes a MISSING
  receipt detectable. Without it, "reject an explicit non-none mode" can never
  catch an adapter that regresses by dropping its receipt, because absence stays
  permanently legal.
* ``closing_theme_frame_window`` -- the frame-domain evidence for the ONE
  sanctioned reuse in the tree. Every malformed input must yield NOTHING, since
  the composite treats absence as "do not loop" and the alternative is looping
  on doubt.
* ``shot["frame_bounded"]`` -- the 7.3 stamp that tells the grader which shots
  owe native-frame evidence, recorded at freeze time so a re-grade months later
  judges the episode by the contract it was RENDERED under.

Plus the centralized manifest index, which replaced two hand-rolled dict
comprehensions that both overwrote duplicates and both crashed on a non-record.
"""

from __future__ import annotations

import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.normpath(os.path.join(_HERE, ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import nodes._otr_video_engines  # noqa: F401,E402  -- populate the registry
from nodes import otr_shot_lock as sl  # noqa: E402
from nodes._otr_video_engines import acceptance as acc  # noqa: E402
from nodes._otr_video_engines import frame_contract as fc  # noqa: E402
from nodes._otr_video_engines import registry as vreg  # noqa: E402
from nodes._otr_video_engines import render_driver as rd  # noqa: E402


# ---------------------------------------------------------------------------
# The closing-theme window
# ---------------------------------------------------------------------------
def _chunks(count=3, start=100.0, dur=6.0, cue="closing", space="master_mix"):
    """One closing cue mirrored into ``count`` chunks, exactly as
    ``scene_sequencer`` mints them: equal durations, contiguous, 1-based index."""
    each = dur / count
    return [{"line_id": "music_%s_%03d" % (cue, i + 1),
             "speaker_role": "music_close",
             "start_s_space": space,
             "start_s": start + i * each,
             "dur_s": each,
             "music_cue_id": cue,
             "music_chunk_index": i + 1,
             "music_chunk_count": count}
            for i in range(count)]


def test_a_well_formed_closing_cue_yields_its_frame_span():
    window = rd.closing_theme_frame_window(_chunks(), 25)
    # 100.0 s -> round(2500) ; 106.0 s -> ceil(2650)
    assert window == {"start": 2500, "end": 2650}


def test_the_window_spans_the_WHOLE_cue_not_one_chunk():
    """The chunking is a transport detail; the theme is one continuous thing."""
    one = rd.closing_theme_frame_window(_chunks(count=1), 25)
    many = rd.closing_theme_frame_window(_chunks(count=6), 25)
    assert one == many


def test_the_terminal_boundary_ROUNDS_UP_so_the_last_partial_frame_is_covered():
    """A window that stopped short would refuse to authorize the final frame of
    the very thing it describes. The manifest quantizes its own timeline the
    same way."""
    window = rd.closing_theme_frame_window(
        _chunks(count=1, start=10.0, dur=0.5), 24)
    assert window == {"start": 240, "end": 252}      # ceil(10.5 * 24) == 252


@pytest.mark.parametrize("mutate,why", [
    (lambda rows: [dict(r, start_s_space="scene_audio") for r in rows],
     "scene-audio seconds are measured from a different zero"),
    (lambda rows: [dict(r, speaker_role="music_open") for r in rows],
     "a different cue entirely"),
    (lambda rows: rows[:-1],
     "a missing chunk leaves a hole of unknown extent"),
    (lambda rows: [dict(r, music_cue_id=("a" if i else "b"))
                   for i, r in enumerate(rows)],
     "two cues is an ambiguity, not a wider window"),
    (lambda rows: [dict(r, music_chunk_index=1) for r in rows],
     "repeated indices cannot prove completeness"),
    (lambda rows: [dict(r, dur_s=0.0) for r in rows],
     "a zero-length chunk is not a span"),
    (lambda rows: [dict(r, dur_s=None) for r in rows],
     "an unreadable duration is not a short one"),
    (lambda rows: [dict(r, start_s=-1.0) for r in rows],
     "a negative start is not a position"),
    (lambda rows: [dict(r, start_s=float("inf")) for r in rows],
     "infinity is not a time"),
    (lambda rows: [dict(r, dur_s=True) for r in rows],
     "True is an int and would silently become one second"),
    (lambda rows: [dict(r, music_chunk_count=99) for r in rows],
     "the declared count must match the rows present"),
])
def test_a_malformed_closing_cue_authorizes_NOTHING(mutate, why):
    assert rd.closing_theme_frame_window(mutate(_chunks()), 25) is None, why


def test_a_REAL_GAP_between_chunks_is_refused_rather_than_bridged():
    """Bridging would extend the loop authorization across silence."""
    rows = _chunks(count=2, start=100.0, dur=4.0)
    rows[1]["start_s"] = 110.0                       # a 8-second hole
    assert rd.closing_theme_frame_window(rows, 25) is None


def test_overlapping_chunks_are_ACCEPTED_because_overlap_is_not_a_hole():
    rows = _chunks(count=2, start=100.0, dur=4.0)
    rows[1]["start_s"] = 101.0        # spans (100,102) and (101,103)
    assert rd.closing_theme_frame_window(rows, 25) == {"start": 2500, "end": 2575}


def test_a_LONG_chunk_containing_a_SHORT_one_reports_the_full_reach():
    """The window is the UNION of the chunks, not the last-sorted one's end.

    Taking ``spans[-1][1]`` looked right while the chunks were equal and in
    order -- which is how ``scene_sequencer`` mints them -- and silently stopped
    the authorization short the moment they were not. It would also have
    reported a GAP that the containing chunk already covers. Both are fixed by
    unioning properly, and both directions are pinned here."""
    rows = _chunks(count=3, start=100.0, dur=3.0)
    rows[0].update(start_s=100.0, dur_s=10.0)        # (100, 110) contains the rest
    rows[1].update(start_s=101.0, dur_s=1.0)         # (101, 102)
    rows[2].update(start_s=109.0, dur_s=1.0)         # (109, 110) -- only reachable
    assert rd.closing_theme_frame_window(rows, 25) == {"start": 2500, "end": 2750}


@pytest.mark.parametrize("lines", [None, [], (), ["not-a-row"], [None]])
def test_no_evidence_means_no_window(lines):
    assert rd.closing_theme_frame_window(lines, 25) is None


@pytest.mark.parametrize("fps", [0, -1, None, "twenty-five", float("inf"),
                                 float("-inf"), float("nan")])
def test_an_unusable_fps_authorizes_nothing(fps):
    """``int(float('inf'))`` raises OverflowError, NOT ValueError, so it escaped
    a two-exception guard -- the same hole ``acceptance.frame_count`` documents
    for the bare ``Infinity`` token that ``json.load`` accepts."""
    assert rd.closing_theme_frame_window(_chunks(), fps) is None


# ---------------------------------------------------------------------------
# "NEVER RAISES" is a contract, and these are the inputs that broke it
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("count", [[3], {"n": 3}, {3}, None, "3", 3.0, True])
def test_an_unhashable_or_wrong_typed_chunk_count_REFUSES_rather_than_raising(count):
    """This was a set comprehension over the RAW field, so ``[3]`` raised
    "unhashable type: 'list'" three lines BEFORE the isinstance check written to
    reject it. A type guard placed after the operation it guards is not a guard."""
    rows = [dict(r, music_chunk_count=count) for r in _chunks()]
    assert rd.closing_theme_frame_window(rows, 25) is None


@pytest.mark.parametrize("index", [[1], {"i": 1}, None, "1", 1.5])
def test_an_unhashable_or_wrong_typed_chunk_index_REFUSES_rather_than_raising(index):
    rows = [dict(r, music_chunk_index=index) for r in _chunks()]
    assert rd.closing_theme_frame_window(rows, 25) is None


@pytest.mark.parametrize("cue", [["closing"], {"c": 1}, 7, None])
def test_an_odd_cue_id_REFUSES_rather_than_raising(cue):
    rows = [dict(r, music_cue_id=cue) for r in _chunks()]
    assert rd.closing_theme_frame_window(rows, 25) in (None, {"start": 2500,
                                                              "end": 2650})


@pytest.mark.parametrize("lines", [5, 3.5, True, "music_close", {"a": 1}])
def test_a_LINES_FIELD_THAT_IS_NOT_A_LIST_refuses_rather_than_raising(lines):
    """``lines or ()`` short-circuits on any TRUTHY value, so a scalar passed
    straight through and ``for ln in 5`` raised TypeError."""
    assert rd.closing_theme_frame_window(lines, 25) is None


def test_a_finite_but_enormous_start_does_not_overflow_the_frame_conversion():
    """Each value is finite on its own; the PRODUCT with fps is not. Checking
    the inputs and not the arithmetic is how a total function stops being one."""
    rows = _chunks(count=1, start=1e308, dur=1.0)
    assert rd.closing_theme_frame_window(rows, 25) is None


def test_a_MALFORMED_CLOSING_ROW_NEVER_DESTROYS_THE_WHOLE_MANIFEST():
    """The blast radius is the point, and it is why these are not nitpicks.

    ``build_clip_manifest`` is called unconditionally on every render and nothing
    wraps it, so a raise inside the window derivation does not cost the loop
    authorization -- it costs the MANIFEST, and with it the composite, the mux
    and the published episode. A corrupt ledger field must refuse the window and
    leave everything else standing."""
    result = {
        "ledger": {
            "video": {"fps": 25, "shots": [], "canonical_canvas": {"w": 8, "h": 8}},
            "lines": [dict(_chunks(count=1)[0], music_chunk_count=[3])],
        },
        "clips": {},
    }
    manifest = rd.build_clip_manifest(result, episode_id="probe")
    assert manifest["frame_receipt_version"] == rd.FRAME_RECEIPT_VERSION
    assert "closing_theme_frame_window" not in manifest


def test_the_window_is_NOT_video_engines_synthesized_title_card_window():
    """``video_engine`` SYNTHESIZES a close window when it cannot resolve one,
    because a title card has to be drawn somewhere. If this window were sourced
    from that, an episode with no closing cue at all would authorize a loop over
    its last few seconds -- the exact 'loops for any unexplained tail' defect.

    Proven by behaviour rather than by reading: the SAME lines that make
    ``video_engine`` synthesize a close window make this one return nothing.

    NO ``hasattr`` GUARD, deliberately. A first draft of this test probed a
    function name that does not exist in the tree (``_bookend_timing``) behind
    exactly such a guard, so the video_engine half never ran, the docstring's
    "proven by behaviour" was false, and the whole test passed vacuously -- it
    would have kept passing after the very unification it exists to forbid. If
    ``_resolve_title_timing`` is ever renamed, this must FAIL and be re-pointed,
    not quietly skip."""
    from nodes import video_engine as ve
    lines = [{"line_id": "l001", "speaker_role": "character",
              "start_s": 0.0, "dur_s": 4.0}]

    # video_engine INVENTS a closing window for these lines...
    synthesized = ve._resolve_title_timing({"lines": lines}, None, 25, 500)
    assert synthesized.get("close_synthesized") is True
    assert "music_close_start_f" in synthesized
    assert "music_close_end_f" in synthesized

    # ...and the loop authorization refuses them. There is no closing cue here,
    # so there is nothing to authorize.
    assert rd.closing_theme_frame_window(lines, 25) is None


# ---------------------------------------------------------------------------
# The manifest index
# ---------------------------------------------------------------------------
def test_a_duplicate_shot_id_keeps_the_FIRST_row_and_never_silently_swaps():
    """The old comprehension kept the LAST one, so which row got graded depended
    on file order. The duplicate is a defect to report, not a tie to break."""
    manifest = {"clips": [{"shot_id": "s1", "engine_id": "first"},
                          {"shot_id": "s1", "engine_id": "second"}]}
    assert acc.clip_rows_by_shot(manifest)["s1"]["engine_id"] == "first"


def test_a_non_record_inside_clips_is_skipped_rather_than_raising():
    """``r.get`` on a list raised AttributeError straight out of the grader and
    past the durable script's exit-code contract."""
    manifest = {"clips": [["not", "a", "row"], 7, None, "s",
                          {"shot_id": "s1"}]}
    assert set(acc.clip_rows_by_shot(manifest)) == {"s1"}


def test_a_clips_STRING_does_not_iterate_into_its_characters():
    assert acc.manifest_clips({"clips": "s1,s2"}) == []
    assert acc.clip_rows_by_shot({"clips": "s1,s2"}) == {}


@pytest.mark.parametrize("manifest", [None, {}, [], "x", 7, {"clips": None}])
def test_the_index_is_total_over_any_document(manifest):
    assert acc.manifest_clips(manifest) == []
    assert acc.clip_rows_by_shot(manifest) == {}


def test_rows_with_no_shot_id_do_not_all_collapse_onto_one_key():
    """Two things with no identity are not the same thing."""
    manifest = {"clips": [{"engine_id": "a"}, {"engine_id": "b"},
                          {"shot_id": "s1"}]}
    assert set(acc.clip_rows_by_shot(manifest)) == {"s1"}


# ---------------------------------------------------------------------------
# The boundedness stamp (7.3)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("engine_id", ["wan_ti2v", "ltx_video", "humo",
                                       "ltx_8gb", "ltx_audio_in"])
def test_a_bounded_engine_is_stamped_TRUE(engine_id):
    shot = {"engine_id": engine_id}
    sl._stamp_frame_bounded(shot)
    assert shot["frame_bounded"] is True
    # And it agrees with the ONE definition, rather than restating it.
    assert shot["frame_bounded"] == fc.can_split(vreg.get_engine(engine_id))


@pytest.mark.parametrize("engine_id", ["still_flat", "still_pan", "viz_green",
                                       "viz_camera"])
def test_an_unbounded_engine_is_stamped_FALSE_not_left_absent(engine_id):
    """False is a RECORDED answer -- 'we asked, and it has no ceiling'. Leaving
    it absent would be indistinguishable from never having asked."""
    shot = {"engine_id": engine_id}
    sl._stamp_frame_bounded(shot)
    assert shot["frame_bounded"] is False


@pytest.mark.parametrize("engine_id", ["", "not_a_registered_engine"])
def test_an_unknown_engine_states_NOTHING_rather_than_guessing(engine_id):
    """Absence means nobody asked. Defaulting to False would claim we checked;
    defaulting to True would indict every legacy ledger."""
    shot = {"engine_id": engine_id}
    sl._stamp_frame_bounded(shot)
    assert "frame_bounded" not in shot


def test_a_zero_length_beat_still_gets_its_boundedness():
    """It is stamped by its own function precisely so it does not inherit
    ``_stamp_coverage_plan``'s early return for a beat that renders nothing --
    the engine is not in doubt just because the beat is empty."""
    shot = {"engine_id": "wan_ti2v", "target_frame_count": 0}
    sl._stamp_frame_bounded(shot)
    assert shot["frame_bounded"] is True
