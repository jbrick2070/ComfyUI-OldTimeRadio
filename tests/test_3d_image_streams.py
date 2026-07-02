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


def test_mesh_fodder_roles_honor_force_engine_map(monkeypatch):
    # OTR_FORCE_ENGINE_MAP=*=mesh_stage forces mesh video at render time; the
    # image phase must ALSO see it and fork fodder, else b000 FamilyInputGaps.
    vp = {"video_models": {"other_beats_video_model": {"engine_id": "ltx_video"},
                           "announcer_video_model": {"engine_id": "ltx_video"}}}
    assert idir.mesh_fodder_roles_from_video_policy(vp) == []   # no force -> none
    monkeypatch.setenv("OTR_FORCE_ENGINE_MAP", "*=mesh_stage")
    roles = set(idir.mesh_fodder_roles_from_video_policy(vp))
    assert "announcer_visual" in roles and "music_visual" in roles
    monkeypatch.delenv("OTR_FORCE_ENGINE_MAP", raising=False)


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
    # mesh-friendly scaffold (v1.1): the subject is forced clean/smooth for the mesher
    assert "smooth solid form" in f001["prompt"]
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
    # matches the minted object's mesh_subject_id convention ("obj_<beat>").
    assert req["mesh_subject_id"] == "obj_b001"


def test_music_visual_fodder_is_the_radio_not_a_generic_object():
    """2026-06-30 mesh-improve item 1: a no-character music-role fodder beat
    meshes the RADIO ITSELF -- never the old generic "an emblematic object
    representing X" (an arbitrary, unrelated prop) and never a character
    body. Verified against the real derive_image_prompts output (not
    guessed)."""
    lines = _lines() + [
        {"line_id": "b005", "speaker_role": "music_open", "char_id": "",
         "text": "", "start_s": 5.0, "dur_s": 2.0},
    ]
    payload, _w = mb.derive_image_prompts(
        _cast(), _meta(), lines=lines, mesh_fodder_roles={"music_visual"})
    fodder = [o for o in payload["objects"]
              if o["kind"] == "mesh_fodder" and o["role"] == "music_visual"]
    assert fodder, "the music_open beat should mint radio mesh fodder"
    f0 = fodder[0]
    assert "char_id" not in f0
    assert mb.MESH_RADIO_HOST_SUBJECT_ID == "radio_host"
    assert f0["mesh_subject_id"] == "radio_host"
    assert "radio" in f0["prompt"].lower()
    assert "emblematic object" not in f0["prompt"].lower()


def test_music_visual_fodder_shares_one_id_across_open_and_close():
    """Every music_visual fodder beat -- open AND close -- shares the SAME
    radio_host id (2026-06-30 mesh-improve item 4): the radio is ONE
    recurring on-air object, never a fresh unrelated mesh per beat (the old
    per-beat obj_<beat> cache)."""
    lines = _lines() + [
        {"line_id": "b005", "speaker_role": "music_open", "char_id": "",
         "text": "", "start_s": 5.0, "dur_s": 2.0},
        {"line_id": "b099", "speaker_role": "music_close", "char_id": "",
         "text": "", "start_s": 90.0, "dur_s": 2.0},
    ]
    payload, _w = mb.derive_image_prompts(
        _cast(), _meta(), lines=lines, mesh_fodder_roles={"music_visual"})
    fodder = [o for o in payload["objects"]
              if o["kind"] == "mesh_fodder" and o["role"] == "music_visual"]
    beat_ids = {o["beat_id"] for o in fodder}
    assert beat_ids == {"b005", "b099"}
    ids = {o["mesh_subject_id"] for o in fodder}
    assert ids == {"radio_host"}


def test_character_and_announcer_fodder_ids_unaffected_by_radio_host_change():
    """The radio_host stable-id change is scoped to role=='music_visual' only.
    Character fodder keeps its char_id-driven subject id; the pre-existing
    announcer fodder id (obj_<beat> -- a separate, pre-existing gap, out of
    scope for this item) is UNCHANGED by this fix either way."""
    payload, _w = mb.derive_image_prompts(
        _cast(), _meta(), lines=_lines(),
        mesh_fodder_roles={"character_video", "announcer_visual"})
    fodder = {o["beat_id"]: o for o in payload["objects"]
              if o["kind"] == "mesh_fodder"}
    assert fodder["b001"]["mesh_subject_id"] == "c01"
    assert fodder["b002"]["mesh_subject_id"] == "obj_b002"
    assert fodder["b002"]["mesh_subject_id"] != "radio_host"


def test_announcer_beat_meshes_announcer_not_uncast(tmp_path):
    """Chunk 6: the announcer carries char_id 'announcer', so a 3D announcer
    beat meshes the ANNOUNCER subject -- never 'uncast' on the environment."""
    from nodes._otr_video_engines import render_driver as rd
    led = _mesh_ledger(tmp_path, char_id="announcer")
    shot = {"shot_id": "shot_b001", "beat_id": "b001", "engine_id": "mesh_stage",
            "family": "image_to_video", "target_frame_count": 25,
            "source_line_ids": ["b001"], "char_id": "announcer", "creative": {}}
    req = rd.build_request_from_shot(shot, led)
    assert req["mesh_subject_id"] == "announcer"


def test_fork_object_subject_id_matches_driver_convention():
    """Chunk 6: a no-character forked beat mints mesh_subject_id 'obj_<beat>',
    the SAME id render_driver stamps -> the mesh cache + the minted fodder
    agree on the subject."""
    payload, _w = mb.derive_image_prompts(
        [], _meta(),
        lines=[{"line_id": "b007", "speaker_role": "", "text": "a sound",
                "start_s": 1.0, "dur_s": 2.0}],
        mesh_fodder_roles={"background_abstract"})
    fodder = [o for o in payload["objects"]
              if o["kind"] == "mesh_fodder" and o["beat_id"] == "b007"]
    assert fodder, "no-character beat should mint object fodder"
    f0 = fodder[0]
    assert "char_id" not in f0
    assert f0["mesh_subject_id"] == "obj_b007"


# --------------------------------------------------------------------------- #
# Chunk 7: opaque source-over composite (mesh over the background plate)
# --------------------------------------------------------------------------- #
def test_mesh_composite_is_opaque_source_over(tmp_path):
    """An OPAQUE mesh pixel (alpha==255) fully REPLACES the background plate --
    no double-exposure ghost. Real-ffmpeg proof: a solid-green opaque mesh clip
    over a solid-RED still plate -> the output is GREEN, never a red/green blend."""
    import shutil
    import subprocess
    if not shutil.which("ffmpeg"):
        import pytest
        pytest.skip("ffmpeg required for the composite proof")
    from PIL import Image
    from nodes import otr_silent_composite as sc
    w = h = 64
    n = 4
    fdir = tmp_path / "mesh"
    fdir.mkdir()
    for i in range(n):
        Image.new("RGBA", (w, h), (0, 255, 0, 255)).save(
            fdir / ("frame_%04d.png" % i))
    plate = tmp_path / "plate.png"
    Image.new("RGB", (w, h), (255, 0, 0)).save(plate)
    seg = tmp_path / "seg.mp4"
    sc._encode_segment_from_dir("ffmpeg", str(fdir), n, str(seg),
                                w=w, h=h, fps=5,
                                bg_path=str(plate), bg_is_still=True)
    assert seg.exists()
    probe = tmp_path / "probe.png"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(seg),
                    "-frames:v", "1", str(probe)], check=True)
    r, g, b = Image.open(probe).convert("RGB").getpixel((w // 2, h // 2))
    # opaque green mesh, NOT a blend with the red plate (a blend would give r~128).
    assert g > 180 and r < 80 and b < 80, (r, g, b)


def test_no_fork_without_mesh_fodder_roles():
    """Default (no fodder roles) keeps the legacy cinematic-scene-still look."""
    payload, _warn = mb.derive_image_prompts(_cast(), _meta(), lines=_lines())
    kinds = {o["kind"] for o in payload["objects"]}
    assert "mesh_fodder" not in kinds
    assert "scene_background_plate" not in kinds
    # the legacy character scene still is present.
    assert "scene_character" in kinds
