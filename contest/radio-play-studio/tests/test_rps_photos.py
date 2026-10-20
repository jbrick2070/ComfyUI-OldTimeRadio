"""Character photos: checked, stored in ComfyUI by content, and set on the writer."""
from __future__ import annotations

import base64
import io
import json

import pytest

from radio_play_studio.form import IMAGE_DIRECTOR, apply_choices, describe_form, locate_nodes
from radio_play_studio.graph import GraphError, workflow_to_api_prompt
from radio_play_studio.photos import (
    MAX_PHOTO_BYTES, PhotoError, decode_photo, stored_name, upload_photo,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
WEBP = b"RIFF\x10\x00\x00\x00WEBPVP8 " + b"\x00" * 32


def _data_url(raw: bytes, mime: str = "image/png") -> str:
    return f"data:{mime};base64," + base64.b64encode(raw).decode("ascii")


@pytest.mark.parametrize("raw, ext", [(PNG, ".png"), (JPEG, ".jpg"), (WEBP, ".webp")])
def test_a_picture_is_known_by_its_first_bytes(raw, ext):
    got, got_ext, _mime = decode_photo(_data_url(raw))
    assert got == raw and got_ext == ext


def test_anything_else_is_refused_before_comfyui_sees_it():
    with pytest.raises(PhotoError, match="not a PNG"):
        decode_photo(_data_url(b"<svg onload=alert(1)>"))
    with pytest.raises(PhotoError, match="intact"):
        decode_photo("data:image/png;base64,***")
    with pytest.raises(PhotoError, match="no picture"):
        decode_photo("")
    with pytest.raises(PhotoError, match="over"):
        decode_photo("A" * (MAX_PHOTO_BYTES * 2))


def test_the_stored_name_is_the_content_so_it_can_never_carry_a_path():
    a, b = stored_name(PNG, ".png"), stored_name(PNG + b"x", ".png")
    assert a.startswith("otr_cast_") and a.endswith(".png") and a != b
    assert "/" not in a and "\\" not in a and ".." not in a


class _Reply(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_upload_goes_to_comfyuis_input_folder_and_returns_its_name():
    seen = {}

    def opener(req, timeout):
        seen["url"], seen["body"] = req.full_url, req.data
        seen["type"] = req.headers["Content-type"]
        return _Reply(json.dumps({"name": stored_name(PNG, ".png"), "subfolder": "",
                                  "type": "input"}).encode())

    name = upload_photo("http://comfy.invalid/", _data_url(PNG), opener=opener)
    assert name == stored_name(PNG, ".png")
    assert seen["url"] == "http://comfy.invalid/upload/image"
    assert seen["type"].startswith("multipart/form-data; boundary=")
    assert PNG in seen["body"] and b'name="overwrite"' in seen["body"]


def test_a_stored_name_the_pack_could_not_read_is_refused():
    def opener(req, timeout):
        return _Reply(json.dumps({"name": "x.png", "subfolder": "elsewhere"}).encode())

    with pytest.raises(PhotoError, match="cannot read"):
        upload_photo("http://comfy.invalid", _data_url(PNG), opener=opener)


def _with_photos(schemas, workflow, *, widget=True):
    """The pack's OTR_ImageDirector, with or without the character_photos input."""
    optional = {"character_photos": ("STRING", {"default": "", "multiline": True})} if widget else {}
    schemas[IMAGE_DIRECTOR] = {"input": {"required": {}, "optional": optional}}
    workflow["nodes"].append({"id": 88, "type": IMAGE_DIRECTOR, "inputs": [],
                              "widgets_values": [""] if widget else []})


def test_the_photo_row_shows_only_where_it_can_be_used(schemas, workflow):
    graph, ids = workflow_to_api_prompt(workflow, schemas), locate_nodes(workflow)
    assert ids[IMAGE_DIRECTOR] is None          # an audio-only workflow: no stills
    assert "my_photos" not in [r["key"] for r in describe_form(schemas, graph, ids)]
    _with_photos(schemas, workflow)
    graph, ids = workflow_to_api_prompt(workflow, schemas), locate_nodes(workflow)
    assert "my_photos" in [r["key"] for r in describe_form(schemas, graph, ids)]


def test_photo_lines_land_on_the_image_director(schemas, workflow):
    _with_photos(schemas, workflow)
    graph, ids = workflow_to_api_prompt(workflow, schemas), locate_nodes(workflow)
    apply_choices(graph, schemas, ids, {"bank": "my_story",
                                        "my_photos": "Ada = otr_cast_ab.png"})
    assert graph[ids[IMAGE_DIRECTOR]]["inputs"]["character_photos"] == "Ada = otr_cast_ab.png"


def test_an_older_pack_refuses_photos_but_not_an_empty_box(schemas, workflow):
    _with_photos(schemas, workflow, widget=False)
    graph, ids = workflow_to_api_prompt(workflow, schemas), locate_nodes(workflow)
    apply_choices(graph, schemas, ids, {"bank": "my_story", "my_photos": ""})
    with pytest.raises(GraphError, match="newer Old-Time Radio pack"):
        apply_choices(graph, schemas, ids, {"bank": "my_story", "my_photos": "Ada = a.png"})


def test_a_workflow_with_no_stills_refuses_photos(schemas, workflow):
    graph, ids = workflow_to_api_prompt(workflow, schemas), locate_nodes(workflow)
    with pytest.raises(GraphError, match="draws no stills"):
        apply_choices(graph, schemas, ids, {"bank": "my_story", "my_photos": "Ada = a.png"})
