"""3D IMAGE STREAMS (2026-06-21) -- the mesh-fodder fork (chunks 1-7).

CPU-only: no Blender / hy3d / GPU. Covers the capability-gated routing seam --
the director computes which image-prompt roles are mesh-fodder from the selected
video engines, MetaBrief forks those beats to a clean mesh_fodder subject + a
subject-free scene_background_plate (never a cinematic scene still -> never the
clay blob), and render_driver feeds the mesher the fodder, not the scene still.
"""
from __future__ import annotations

from nodes._otr_video_engines import eng_mesh_stage  # noqa: F401 (register)
from nodes import otr_image_director as idir
from nodes import otr_meta_brief_image_prompt as mb


# --------------------------------------------------------------------------- #
# OTR_ImageDirector: which roles are mesh-fodder (capability, not engine name)
# --------------------------------------------------------------------------- #
def test_mesh_fodder_roles_from_video_policy():
    vp = {"video_models": {
        "other_beats_video_model": {"engine_id": "mesh_stage"},
        "announcer_video_model": {"engine_id": "ltx_video"},
        "music_video_model": {"engine_id": "mesh_stage"},
    }}
    roles = set(idir.mesh_fodder_roles_from_video_policy(vp))
    # other_beats drives character_video + the two other-beats roles; music too.
    assert roles == {"character_video", "background_abstract", "scene_broll",
                     "music_visual"}
    # announcer paired with ltx_video -> NOT fodder.
    assert "announcer_visual" not in roles


def test_mesh_fodder_roles_empty_when_no_3d():
    vp = {"video_models": {"other_beats_video_model": {"engine_id": "ltx_video"}}}
    assert idir.mesh_fodder_roles_from_video_policy(vp) == []
    # unknown/custom engine is tolerantly NOT-fodder (never raises).
    vp2 = {"video_models": {"other_beats_video_model": {"engine_id": "not_a_real_engine"}}}
    assert idir.mesh_fodder_roles_from_video_policy(vp2) == []


# --------------------------------------------------------------------------- #
# MetaBrief: the prompt fork
# --------------------------------------------------------------------------- #
def _meta():
    return {"story_brief_terms": {"setting": ["a fog-bound harbor town"]}}


def _lines():
    return [
        {"line_id": "b001", "speaker_role": "character", "char_id": "c01",
         "text": "We have to go back.", "beat_intent": "warns the others",
         "start_s": 1.0, "dur_s": 2.0},
        {"line_id": "b002", "speaker_role": "announcer", "char_id": "announcer",
         "text": "And now, our story.", "start_s": 3.0, "dur_s": 1.0},
    ]


def _cast():
    return [{"char_id": "c01",
             "character_description": "a weathered dock inspector in an oilskin coat"}]


def test_fork_mints_fodder_and_plate_not_scene_still():
    payload, _warn = mb.derive_image_prompts(
        _cast(), _meta(), lines=_lines(),
        mesh_fodder_roles={"character_video", "announcer_visual", "music_visual"})
    objs = payload["objects"]
    by_kind = {}
    for o in objs:
        by_kind.setdefault(o["kind"], []).append(o)
    # the fork replaced the cinematic scene still -> NO scene_character/scene_beat
    assert "scene_character" not in by_kind
    assert "scene_beat" not in by_kind
    # exactly one fodder + one plate per forked beat (b001 character, b002 announcer,
    # b000 open music) -- each scene target became two objects.
    fodder = by_kind.get("mesh_fodder", [])
    plate = by_kind.get("scene_background_plate", [])
    fodder_beats = {o["beat_id"] for o in fodder}
    assert {"b001", "b002"} <= fodder_beats
    assert fodder_beats == {o["beat_id"] for o in plate}      # paired 1:1
    # the character fodder leads with the character + the isolation scaffold,
    # carries char_id + mesh_subject_id, and is portrait/near-square.
    f001 = next(o for o in fodder if o["beat_id"] == "b001")
    assert f001["char_id"] == "c01" and f001["mesh_subject_id"] == "c01"
    assert "oilskin" in f001["prompt"] and "neutral mid-grey" in f001["prompt"]
    assert f001["w"] == mb.MESH_FODDER_W and f001["h"] == mb.MESH_FODDER_H
    assert f001["negative_prompt"] == mb.MESH_FODDER_NEG_SCAFFOLD
    # the plate is subject-free (the scaffold says so) and 16:9-ish (wide).
    p001 = next(o for o in plate if o["beat_id"] == "b001")
    assert "no people" in p001["prompt"] and "no subject" in p001["prompt"]
    assert p001["w"] >= p001["h"]


# --------------------------------------------------------------------------- #
# Chunk 4: kind-specific indices (render_driver)
# --------------------------------------------------------------------------- #
def test_portrait_index_ignores_mesh_fodder_rows():
    from nodes._otr_video_engines import render_driver as rd
    ledger = {"images": {"images": [
        {"object_id": "c01", "kind": "portrait", "char_id": "c01",
         "path": "portrait_c01.png"},
        {"object_id": "meshfodder_b001", "kind": "mesh_fodder", "char_id": "c01",
         "beat_id": "b001", "path": "fodder_c01.png"},
        {"object_id": "still_b001", "kind": "scene_character", "char_id": "c01",
         "beat_id": "b001", "path": "scene_c01.png"},
    ]}}
    # the HuMo portrait lookup sees ONLY the real portrait, never the fodder.
    assert rd._portrait_index(ledger) == {"c01": "portrait_c01.png"}


def test_still_index_prioritizes_background_plate():
    from nodes._otr_video_engines import render_driver as rd
    # plate appears BEFORE a stale scene_beat row -> plate still wins.
    ledger = {"images": {"images": [
        {"object_id": "plate_b002", "kind": "scene_background_plate",
         "beat_id": "b002", "path": "plate_b002.png"},
        {"object_id": "still_b002", "kind": "scene_beat",
         "beat_id": "b002", "path": "scene_b002.png"},
    ]}}
    assert rd._still_index(ledger)["b002"] == "plate_b002.png"


# --------------------------------------------------------------------------- #
# Chunk 5: render_driver stamps the mesh subject id onto the request
# --------------------------------------------------------------------------- #
def _mesh_ledger(tmp_path, char_id=""):
    fodder = tmp_path / "fodder.png"
    fodder.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 80)
    row = {"kind": "mesh_fodder", "beat_id": "b001", "path": str(fodder)}
    if char_id:
        row["char_id"] = char_id
        row["mesh_subject_id"] = char_id
    return {
        "video": {"video_revision": 1, "shots": []},
        "lines": [{"line_id": "b001", "char_id": char_id,
                   "start_s": 1.0, "dur_s": 2.0}],
        "images": {"images": [row]},
    }


def test_build_request_stamps_mesh_subject_id_char(tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    led = _mesh_ledger(tmp_path, char_id="c01")
    shot = {"shot_id": "shot_b001", "beat_id": "b001", "engine_id": "mesh_stage",
            "family": "image_to_video", "target_frame_count": 25,
            "source_line_ids": ["b001"], "char_id": "c01", "creative": {}}
    req = rd.build_request_from_shot(shot, led)
    assert req["mesh_subject_id"] == "c01"


def test_build_request_stamps_mesh_subject_id_beat_when_no_char(tmp_path):
    from nodes._otr_video_engines import render_driver as rd
    led = _mesh_ledger(tmp_path, char_id="")        # announcer/music object beat
    # the fodder row has no char id, so it resolves by beat_id; subject = beat id.
    shot = {"shot_id": "shot_b001", "beat_id": "b001", "engine_id": "mesh_stage",
            "family": "image_to_video", "target_frame_count": 25,
            "source_line_ids": ["b001"], "char_id": "", "creative": {}}
    req = rd.build_request_from_shot(shot, led)
    assert req["mesh_subject_id"] == "b001"


def test_no_fork_without_mesh_fodder_roles():
    """Default (no fodder roles) keeps the legacy cinematic-scene-still look."""
    payload, _warn = mb.derive_image_prompts(_cast(), _meta(), lines=_lines())
    kinds = {o["kind"] for o in payload["objects"]}
    assert "mesh_fodder" not in kinds
    assert "scene_background_plate" not in kinds
    # the legacy character scene still is present.
    assert "scene_character" in kinds
