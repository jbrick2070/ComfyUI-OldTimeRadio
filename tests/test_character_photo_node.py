"""OTR_CharacterPhoto: a character's photo on the canvas, the ComfyUI way.

Operator, 2026-10-20: "is that all best practice for comfy workflows? I know they
usually go to the assets folder" -- then "build the Character Photo node with an
image picker, upload and preview, chained into Image Director". Codex's design
review shaped it: ONE required picker per node (the canvas only gives the first
required picker its upload button), chained for an amalgamation; ComfyUI's own
file names ("pasted/x.png", "x.png [output]"); IS_CHANGED on the bytes; a typed
chain; and one blank node shipped wired in the canonical workflow.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import otr_character_photo as N
from nodes._otr_shared import character_photos as cp
from nodes.otr_image_director import OTRImageDirector

REPO = Path(__file__).resolve().parents[1]
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root = tmp_path / "input"
    (root / "pasted").mkdir(parents=True)
    (root / "jeff.png").write_bytes(PNG)
    (root / "pasted" / "image (1).png").write_bytes(PNG + b"paste")
    monkeypatch.setattr(cp, "input_dir", lambda: str(root))
    monkeypatch.setattr(cp, "_root_dir", lambda kind: str(root) if kind == "input"
                        else str(tmp_path / kind))
    return root


def test_the_picker_is_declared_like_core_load_image():
    req = N.OTRCharacterPhoto.INPUT_TYPES()["required"]
    assert req["image"][1]["image_upload"] is True
    assert list(req) == ["name", "image"]          # ONE required picker
    assert "photos_in" in N.OTRCharacterPhoto.INPUT_TYPES()["optional"]


def test_blank_passes_through_and_half_filled_is_refused(inputs):
    v = N.OTRCharacterPhoto.VALIDATE_INPUTS
    assert v(name="", image="") is True
    assert "give the character's name" in v(name="", image="jeff.png")
    assert "pick a photo" in v(name="Jeffrey", image="")
    assert "not a picture" in v(name="Jeffrey", image="nope.png")
    assert v(name="Jeffrey", image="jeff.png") is True
    (chain,) = N.OTRCharacterPhoto().add(photos_in={"version": 1, "photos": {"Cat": ["c.png"]}})
    assert chain["photos"] == {"Cat": ["c.png"]}


def test_a_chain_gathers_one_face_and_refuses_a_fourth(inputs):
    node = N.OTRCharacterPhoto()
    (c,) = node.add(name="Jeffrey", image="jeff.png")
    (c,) = node.add(name=" jeffrey ", image="pasted/image (1).png", photos_in=c)
    (c,) = node.add(name="JEFFREY", image="jeff.png", photos_in=c)     # a repeat
    assert c["photos"] == {"Jeffrey": ["jeff.png", "pasted/image (1).png"]}
    (c,) = node.add(name="Jeffrey", image="a.png", photos_in=c)
    with pytest.raises(ValueError, match="at most 3"):
        node.add(name="Jeffrey", image="b.png", photos_in=c)


def test_comfyui_file_names_resolve_and_stay_inside_their_folder(inputs, tmp_path):
    assert cp.resolve_photo("jeff.png").endswith("jeff.png")
    assert cp.resolve_photo("pasted/image (1).png").endswith("image (1).png")
    assert cp.resolve_photo("jeff.png [input]").endswith("jeff.png")
    (tmp_path / "output").mkdir()
    (tmp_path / "output" / "still.png").write_bytes(PNG)
    assert cp.resolve_photo("still.png [output]").endswith("still.png")
    for bad in ("../secret.png", "pasted/../../x.png", "/etc/x.png", "C:/x.png",
                "a\\b.png", "x.exe", "jeff.png [models]"):
        assert cp.resolve_photo(bad) == "", bad


def test_a_new_picture_under_the_same_name_changes_the_fingerprint(inputs):
    before = N.OTRCharacterPhoto.IS_CHANGED(name="Jeffrey", image="jeff.png")
    (inputs / "jeff.png").write_bytes(PNG + b"retaken")
    assert N.OTRCharacterPhoto.IS_CHANGED(name="Jeffrey", image="jeff.png") != before
    assert N.OTRCharacterPhoto.IS_CHANGED(name="", image="") == ""
    assert OTRImageDirector.IS_CHANGED(character_photos="") == ""
    assert OTRImageDirector.IS_CHANGED(character_photos="Jeffrey = jeff.png") != ""


def _video_policy():
    return json.dumps({"video_models": {}, "image_models": {
        "announcer_image_model": "flux3_klein", "music_image_model": "flux3_klein",
        "character_image_model": "flux3_klein"}})


def test_image_director_merges_the_chain_and_the_typed_lines_win():
    chain = {"version": 1, "photos": {"Jeffrey": ["a.png", "b.png"], "Cat": ["cat.png"]}}
    (raw,) = OTRImageDirector().direct(
        15, "request_hash", 42, video_policy_json=_video_policy(),
        character_photos="jeffrey = typed.png", character_photos_in=chain)
    photos = json.loads(raw)["character_photos"]
    assert photos == {"Cat": ["cat.png"], "jeffrey": ["typed.png"]}
    (raw,) = OTRImageDirector().direct(15, "request_hash", 42,
                                       video_policy_json=_video_policy(),
                                       character_photos_in=chain)
    assert json.loads(raw)["character_photos"] == chain["photos"]


def test_the_socket_is_the_last_input_and_the_node_is_registered():
    opt = list(OTRImageDirector.INPUT_TYPES()["optional"])
    assert opt[-2:] == ["character_photos", "character_photos_in"]
    manifest = json.loads((REPO / "node_list.json").read_text(encoding="utf-8"))
    assert manifest["OTR_CharacterPhoto"] == "Character Photo"
    init = (REPO / "__init__.py").read_text(encoding="utf-8")
    assert '"OTR_CharacterPhoto":' in init and ".nodes.otr_character_photo" in init


def test_the_canonical_ships_one_blank_node_wired_to_image_director():
    wf = json.loads((REPO / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
    photo = [n for n in wf["nodes"] if n["type"] == "OTR_CharacterPhoto"]
    # The upload button saves "image" in its own slot, as core's LoadImage does.
    assert len(photo) == 1 and photo[0]["widgets_values"] == ["", "", "image"]
    director = next(n for n in wf["nodes"] if n["type"] == "OTR_ImageDirector")
    link_id = next(i["link"] for i in director["inputs"] if i["name"] == "character_photos_in")
    link = next(lk for lk in wf["links"] if lk[0] == link_id)
    assert (link[1], link[3]) == (photo[0]["id"], director["id"])


# --------------------------------------------------------------------------- #
# The upload button saves a slot, and every converter reads past it
# (live canvas check, 2026-10-20: the frontend serializes the node as
# ["", "<file>", "image"] -- the same shape as core's LoadImage)
# --------------------------------------------------------------------------- #
def _schemas():
    return {"OTR_CharacterPhoto": {"input": OTR_INPUTS()}}


def OTR_INPUTS():
    spec = N.OTRCharacterPhoto.INPUT_TYPES()
    spec["required"]["image"] = (["jeff.png"], {"image_upload": True})
    return spec


def _node(values):
    return {"id": 98, "type": "OTR_CharacterPhoto", "inputs": [], "widgets_values": values}


@pytest.mark.parametrize("values", [["Jeffrey", "jeff.png", "image"]])
def test_the_pack_converters_read_past_the_upload_slot(values):
    import importlib.util
    spec = importlib.util.spec_from_file_location("otr_api", REPO / "scripts" / "otr_api.py")
    api = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(api)
    assert api._serialized_slot_names("OTR_CharacterPhoto", _schemas()) == [
        "name", "image", "image__upload"]
    from nodes import _otr_workflow_apply as wa
    assert wa.serialized_slot_names("OTR_CharacterPhoto", _schemas()) == [
        "name", "image", "image__upload"]
    from nodes import _otr_workflow_validator as wv
    assert wv._expected_slot_count(OTR_INPUTS()) == 3
