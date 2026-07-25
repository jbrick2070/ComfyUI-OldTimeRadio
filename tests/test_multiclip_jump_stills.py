"""Chunk 4 -- the jump-still image-phase consumer.

WITHOUT THIS A JUMP CUT HAS NO STILL. ShotLock plans N independent renders for
one beat, the image phase mints exactly ONE still per beat, so every segment
after the first would reach the render with no init image at all -- the
text-only / dark-floor degradation this build removes.

All three seams are pinned here: ShotLock mints the requests, the image
dispatcher merges them into ``objects`` + ``required_scene_targets``, and the
still spine proves EVERY segment before a render starts. The merge is pinned as
LOAD-BEARING, not merely present: reverting it to a passthrough must make the
spine fail closed, which is the check that would have caught the unproven
decapitation half in QA round 1.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes import otr_image_gen_dispatcher as gd
from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import coverage_plan as cp
from nodes._otr_video_engines import frame_contract as fc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd

#: A real multi-clip JUMP contract: an 8n+1 ladder with a ceiling, plus a SOFT
#: reference (the HuMo / Veo class) that cannot lock frame 0 -- so it jump cuts
#: rather than pretending to chain.
JUMP_CONTRACT = fc.FrameContract(
    min_frames=9, max_frames=41, quantum=8,
    supports_multi_clip=True, continuity=fc.CONTINUITY_SOFT_REFERENCE)

#: The same ladder with a strict first-frame lock. It CHAINS, so its successors
#: begin on a real rendered frame and owe the image phase nothing.
CHAIN_CONTRACT = fc.FrameContract(
    min_frames=9, max_frames=41, quantum=8,
    supports_multi_clip=True, continuity=fc.CONTINUITY_STRICT_FIRST_FRAME)


def _use_contract(monkeypatch, contract, engine_id="wan_i2v"):
    engine = vreg.get_engine(engine_id)
    monkeypatch.setattr(engine, "frame_contract", lambda: contract,
                        raising=False)


def _beats(role="character_video", n=1):
    return [{"beat_id": "b%03d" % i, "role": role, "char_id": "c1",
             "dur_s": 2.0} for i in range(1, n + 1)]


def _budget(beats, frames=50):
    per = {b["beat_id"]: frames for b in beats}
    return {"per_beat": per, "total_frames": frames * len(beats),
            "warnings": []}


def _policy(engine="wan_i2v"):
    return {
        "policy_version": 2,
        "video_models": {
            "announcer_video_model": {"engine_id": engine},
            "music_video_model": {"engine_id": engine},
            "character_video_model": {"engine_id": engine},
        },
    }


# ---------------------------------------------------------------------------
# The pure authority
# ---------------------------------------------------------------------------

def test_a_single_clip_beat_owes_no_jump_stills():
    plan = cp.partition_beat(30, fc.SINGLE_ONLY)
    assert cp.jump_still_requests(plan, "b001") == ()


def test_a_chain_owes_no_jump_stills():
    """A chained successor begins on a REAL rendered frame, not a new still."""
    plan = cp.partition_beat(49, CHAIN_CONTRACT)
    assert plan.join_mode == cp.JOIN_CHAIN and plan.is_multi_clip
    assert cp.jump_still_requests(plan, "b001") == ()


def test_a_jump_owes_one_still_per_segment_after_the_first():
    plan = cp.partition_beat(99, JUMP_CONTRACT)
    assert plan.join_mode == cp.JOIN_JUMP and plan.segment_count == 3
    rows = cp.jump_still_requests(plan, "b001", role="character_video",
                                  engine_id="wan_i2v", char_id="c1")
    # Segment 0 uses the beat's OWN still, which already exists and is already
    # validated -- minting a second one for it would orphan the first.
    assert [row["segment_index"] for row in rows] == [1, 2]
    assert [row["object_id"] for row in rows] == ["jumpstill_b001_s1",
                                                  "jumpstill_b001_s2"]
    assert all(row["kind"] == cp.JUMP_STILL_KIND for row in rows)
    assert all(row["beat_id"] == "b001" for row in rows)
    assert all(row["engine_id"] == "wan_i2v" for row in rows)
    assert all(row["char_id"] == "c1" for row in rows)


def test_an_unkeyed_jump_beat_is_terminal():
    """An id-less request could never be matched to a rendered still."""
    plan = cp.partition_beat(50, JUMP_CONTRACT)
    with pytest.raises(cp.CoveragePlanError, match="beat_id"):
        cp.jump_still_requests(plan, "")


# ---------------------------------------------------------------------------
# Seam 1 -- ShotLock stamps the requests durably
# ---------------------------------------------------------------------------

def test_shotlock_stamps_one_request_per_jump_segment(monkeypatch):
    _use_contract(monkeypatch, JUMP_CONTRACT)
    beats = _beats()
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, frames=99), {}, _policy())
    rows = shots[0]["jump_still_requests"]
    assert [row["object_id"] for row in rows] == ["jumpstill_b001_s1",
                                                  "jumpstill_b001_s2"]
    assert all(row["role"] == "character_video" for row in rows)


def test_the_stamp_survives_the_wire_verbatim(monkeypatch):
    """It rides the durable ledger, so it must be plain JSON."""
    import json
    _use_contract(monkeypatch, JUMP_CONTRACT)
    beats = _beats()
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, frames=50), {}, _policy())
    raw = json.dumps(shots[0]["jump_still_requests"], ensure_ascii=True)
    assert json.loads(raw) == shots[0]["jump_still_requests"]


def test_a_chained_beat_gets_no_stamp_at_all(monkeypatch):
    _use_contract(monkeypatch, CHAIN_CONTRACT)
    beats = _beats()
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, frames=49), {}, _policy())
    assert cp.CoveragePlan.from_dict(shots[0]["coverage_plan"]).is_multi_clip
    assert "jump_still_requests" not in shots[0]


def test_chunk_4_is_behaviour_inert_today():
    """Every adapter is still single_only, so no beat owes a segment still.

    If this fails, an adapter opted in to multi-clip without its own live
    proof -- exactly what the per-adapter declaration exists to prevent, and it
    must surface here rather than on a render.
    """
    beats = _beats(n=3)
    _groups, shots = sl.build_execution_plan(beats, _budget(beats), {},
                                             _policy())
    assert all("jump_still_requests" not in shot for shot in shots)


# ---------------------------------------------------------------------------
# Seam 2 -- the dispatcher merges them into the image phase
# ---------------------------------------------------------------------------

def _objects(beat="b001", role="character_video", kind="scene_beat"):
    return [{"object_id": "still_%s" % beat, "kind": kind, "role": role,
             "beat_id": beat, "w": 832, "h": 480,
             "prompt": "a lonely transmitter tower at dusk",
             "prompt_hash": "ph_%s" % beat, "visual_style": "sci_fi_radio",
             "char_id": "c1"}]


def _targets(beat="b001", role="character_video", kind="scene_beat"):
    return [{"object_id": "still_%s" % beat, "kind": kind, "role": role,
             "beat_id": beat}]


def _requests(beat="b001", segments=(1,), role="character_video"):
    return [{"object_id": cp.jump_still_object_id(beat, index),
             "kind": cp.JUMP_STILL_KIND, "beat_id": beat,
             "segment_index": index, "role": role, "engine_id": "wan_i2v"}
            for index in segments]


def _request_ledger(beat="b001", segments=(1,), role="character_video"):
    return {"video": {"shots": [{
        "shot_id": "shot_%s" % beat, "role": role, "engine_id": "wan_i2v",
        "source_line_ids": [beat],
        "jump_still_requests": _requests(beat, segments, role),
    }]}}


def test_the_merge_mints_one_real_object_per_segment():
    objects, targets, report = gd.merge_jump_still_requests(
        _request_ledger(segments=(1, 2)), _objects(), _targets())
    minted = [obj for obj in objects
              if obj["kind"] == cp.JUMP_STILL_KIND]
    assert [obj["object_id"] for obj in minted] == ["jumpstill_b001_s1",
                                                    "jumpstill_b001_s2"]
    # Cloned from the beat's own still: same scene, same size, same style.
    assert all(obj["prompt"] == "a lonely transmitter tower at dusk"
               for obj in minted)
    assert all((obj["w"], obj["h"]) == (832, 480) for obj in minted)
    assert all(obj["visual_style"] == "sci_fi_radio" for obj in minted)
    assert all(obj["source"] == "jump_segment_cover" for obj in minted)
    # ...and every one is a REQUIRED target, so an unmaterialized segment is
    # terminal at the dispatcher's own completion contract.
    assert [t["object_id"] for t in targets] == ["still_b001",
                                                 "jumpstill_b001_s1",
                                                 "jumpstill_b001_s2"]
    assert len(report) == 2


def test_the_merge_never_mutates_its_inputs():
    objects, targets = _objects(), _targets()
    gd.merge_jump_still_requests(_request_ledger(), objects, targets)
    assert len(objects) == 1 and len(targets) == 1


def test_the_merge_is_free_when_nothing_jump_cuts():
    objects, targets = _objects(), _targets()
    out_objects, out_targets, report = gd.merge_jump_still_requests(
        {"video": {"shots": [{"shot_id": "shot_b001"}]}}, objects, targets)
    assert out_objects is objects and out_targets is targets and report == []


def test_each_segment_gets_its_own_seed_so_the_cut_is_a_real_cut():
    """A jump CUT, not the same frame twice.

    The per-object seed is derived from the object id, so cloning the prompt
    under a new id yields a different image of the SAME scene -- which is what
    a jump cut is. (An operator who selects ``mode=fixed`` gets identical
    stills by that choice, not by this merge.)
    """
    objects, _targets_out, _report = gd.merge_jump_still_requests(
        _request_ledger(segments=(1, 2)), _objects(), _targets())
    seeds = {gd.resolve_object_seed({}, obj["object_id"], obj["prompt_hash"],
                                    kind=obj["kind"]) for obj in objects}
    assert len(seeds) == 3


def test_a_required_beat_with_no_scene_still_is_TERMINAL():
    """The exact bug chunk 4 exists to close, stated as a refusal."""
    with pytest.raises(gd.ImageRenderError, match="renders from nothing"):
        gd.merge_jump_still_requests(_request_ledger(), [], _targets())


def test_a_mesh_beat_asking_for_jump_stills_is_TERMINAL():
    """Fodder or plate? Neither answer is provable, so refuse rather than pick."""
    mesh_objects = [
        {"object_id": "meshfodder_b001", "kind": "mesh_fodder",
         "role": "character_video", "beat_id": "b001", "prompt": "x"},
        {"object_id": "plate_b001", "kind": "scene_background_plate",
         "role": "character_video", "beat_id": "b001", "prompt": "y"},
    ]
    with pytest.raises(gd.ImageRenderError, match="NO GUESS"):
        gd.merge_jump_still_requests(_request_ledger(), mesh_objects,
                                     _targets())


def test_two_candidate_scene_stills_for_one_beat_is_TERMINAL():
    ambiguous = _objects() + [dict(_objects()[0], object_id="still_b001_alt",
                                   kind="scene_character")]
    with pytest.raises(gd.ImageRenderError, match="more than one scene still"):
        gd.merge_jump_still_requests(_request_ledger(), ambiguous, _targets())


def test_a_lane_that_consumes_no_still_needs_no_segment_stills():
    """A visualizer beat: no scene object AND no required target.

    Reported rather than silently dropped -- the distinction between "this lane
    genuinely needs none" and "the still went missing" is the required-target
    receipt, not a guess.
    """
    objects, targets, report = gd.merge_jump_still_requests(
        _request_ledger(), [], [])
    assert objects == [] and targets == []
    assert report and "consumes no scene still" in report[0]


def test_a_malformed_request_stamp_is_TERMINAL():
    ledger = _request_ledger()
    ledger["video"]["shots"][0]["jump_still_requests"] = {"not": "a list"}
    with pytest.raises(gd.ImageRenderError, match="malformed"):
        gd.merge_jump_still_requests(ledger, _objects(), _targets())


def test_a_request_without_an_object_id_is_TERMINAL():
    ledger = _request_ledger()
    ledger["video"]["shots"][0]["jump_still_requests"] = [{"beat_id": "b001"}]
    with pytest.raises(gd.ImageRenderError, match="without an object_id"):
        gd.merge_jump_still_requests(ledger, _objects(), _targets())


def test_a_duplicate_segment_id_is_TERMINAL():
    objects = _objects() + [dict(_objects()[0],
                                 object_id="jumpstill_b001_s1",
                                 kind=cp.JUMP_STILL_KIND, beat_id="")]
    with pytest.raises(gd.ImageRenderError, match="already exists"):
        gd.merge_jump_still_requests(_request_ledger(), objects, _targets())


# ---------------------------------------------------------------------------
# Seam 3 -- the still spine proves EVERY segment
# ---------------------------------------------------------------------------

def _pool_file(tmp_path, name):
    path = tmp_path / ("%s.png" % name)
    path.write_bytes(b"still")
    return str(path)


def _spine_ledger(tmp_path, *, segments=(1,), rows=True, manifest=True):
    beat = "b001"
    image_rows = [{
        "object_id": "still_b001", "kind": "scene_beat", "beat_id": beat,
        "path": "", "pool_path": _pool_file(tmp_path, "base"),
        "content_hash": "abc123456789",
    }]
    targets = [{"object_id": "still_b001", "kind": "scene_beat",
                "beat_id": beat, "path": ""}]
    for index in segments:
        oid = cp.jump_still_object_id(beat, index)
        if rows:
            image_rows.append({
                "object_id": oid, "kind": cp.JUMP_STILL_KIND, "beat_id": beat,
                "segment_index": index, "path": "",
                "pool_path": _pool_file(tmp_path, oid),
                "content_hash": "def%03d456789" % index,
            })
        if manifest:
            targets.append({"object_id": oid, "kind": cp.JUMP_STILL_KIND,
                            "beat_id": beat})
    return {
        "episode_id": "ep_jump",
        "lines": [{"line_id": beat, "char_id": "",
                   "speaker_role": "background"}],
        "images": {"images": image_rows, "required_scene_targets": targets},
        "video": {"shots": [{
            "shot_id": "shot_b001", "source_line_ids": [beat],
            "engine_id": "still_pan", "family": "static_image_gen",
            "role": "character_video",
            "jump_still_requests": _requests(beat, segments),
        }]},
    }


def test_the_spine_validates_every_jump_segment(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = _spine_ledger(tmp_path, segments=(1, 2))

    receipt = rd.validate_and_repair_still_spine(ledger)

    segments = [row for row in receipt["validated"]
                if row["kind"] == cp.JUMP_STILL_KIND]
    assert [row["segment_index"] for row in segments] == [1, 2]
    assert all(Path(row["path"]).is_file() for row in segments)
    # ...and each one landed in the ACTIVE episode's stills directory.
    stills = tmp_path / "output" / "otr" / "episodes" / "ep_jump" / "stills"
    assert all(Path(row["path"]).parent == stills for row in segments)


def test_a_missing_segment_still_is_TERMINAL(tmp_path, monkeypatch):
    """The beat's own still must NOT be borrowed to cover a segment.

    Substituting it would put the identical image at both ends of the cut --
    the held-frame degradation this build removes -- so the base still being
    present and materialized is deliberately not enough.
    """
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = _spine_ledger(tmp_path, rows=False)
    with pytest.raises(rd.RenderError, match="would render from nothing"):
        rd.validate_and_repair_still_spine(ledger)


def test_a_segment_outside_the_target_receipt_is_TERMINAL(tmp_path,
                                                          monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = _spine_ledger(tmp_path, manifest=False)
    with pytest.raises(rd.RenderError, match="not in the required_scene_targets"):
        rd.validate_and_repair_still_spine(ledger)


def test_a_jump_shot_always_needs_a_target_manifest(tmp_path, monkeypatch):
    """Even a lane the scene check would let through unmanifested."""
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = _spine_ledger(tmp_path)
    ledger["images"].pop("required_scene_targets")
    ledger["video"]["shots"][0].update(engine_id="viz_mxc_cpu",
                                       family="abstract")
    with pytest.raises(rd.RenderError, match="required_scene_targets receipt"):
        rd.validate_and_repair_still_spine(ledger)


def test_a_malformed_stamp_is_TERMINAL_at_the_spine(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = _spine_ledger(tmp_path)
    ledger["video"]["shots"][0]["jump_still_requests"] = "b001"
    with pytest.raises(rd.RenderError, match="malformed"):
        rd.validate_and_repair_still_spine(ledger)


def test_REVERTING_THE_MERGE_MAKES_THE_RENDER_FAIL_CLOSED(tmp_path,
                                                          monkeypatch):
    """THE MUTATION CHECK -- is the merge load-bearing, or is it theatre?

    A passthrough ``merge_jump_still_requests`` produces exactly this ledger:
    no segment object was dispatched, so no segment image row exists, and no
    segment target reached the receipt. If the spine still passed here, the
    merge could be deleted without a single test noticing -- which is how the
    decapitation fix's twin shipped unproven in QA round 1.
    """
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    ledger = _spine_ledger(tmp_path, rows=False, manifest=False)
    with pytest.raises(rd.RenderError):
        rd.validate_and_repair_still_spine(ledger)


def test_a_segment_still_never_shadows_the_beats_own_still(tmp_path):
    """Why the segment rows do NOT wear a ``scene_*`` kind.

    Both beat-keyed still lookups take the LAST matching scene row, so a
    segment still wearing a scene kind would shadow the beat's own image and
    segment 0 would render from the LAST segment's still.
    """
    ledger = _spine_ledger(tmp_path, segments=(1, 2))
    rows = ledger["images"]["images"]
    for row in rows:
        row["path"] = row["pool_path"]
    assert rd._still_index(ledger)["b001"] == rows[0]["path"]
    assert rd._still_spine_row_for_beat(rows, "b001")["object_id"] == "still_b001"


def test_end_to_end_shotlock_to_dispatcher_to_spine(tmp_path, monkeypatch):
    """One set of ids, minted once and READ twice -- never re-derived."""
    monkeypatch.setenv("OTR_OUTPUT_DIR", str(tmp_path / "output"))
    _use_contract(monkeypatch, JUMP_CONTRACT)
    beats = _beats()
    _groups, shots = sl.build_execution_plan(
        beats, _budget(beats, frames=99), {}, _policy())
    stamped = [row["object_id"] for row in shots[0]["jump_still_requests"]]

    objects, targets, _report = gd.merge_jump_still_requests(
        {"video": {"shots": shots}}, _objects(), _targets())
    assert set(stamped) <= {obj["object_id"] for obj in objects}
    assert set(stamped) <= {target["object_id"] for target in targets}

    image_rows = [{"object_id": "still_b001", "kind": "scene_beat",
                   "beat_id": "b001", "path": "",
                   "pool_path": _pool_file(tmp_path, "e2e_base"),
                   "content_hash": "abc123456789"}]
    image_rows.extend({
        "object_id": oid, "kind": cp.JUMP_STILL_KIND, "beat_id": "b001",
        "path": "", "pool_path": _pool_file(tmp_path, oid),
        "content_hash": "e2e%s" % oid,
    } for oid in stamped)
    ledger = {
        "episode_id": "ep_e2e",
        "lines": [{"line_id": "b001", "char_id": "",
                   "speaker_role": "background"}],
        "images": {"images": image_rows, "required_scene_targets": targets},
        "video": {"shots": shots},
    }

    receipt = rd.validate_and_repair_still_spine(ledger)

    proven = [row["path"] for row in receipt["validated"]
              if row["kind"] == cp.JUMP_STILL_KIND]
    assert len(proven) == len(stamped) == 2
