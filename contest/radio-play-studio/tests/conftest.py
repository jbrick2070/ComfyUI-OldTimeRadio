"""Shared fixtures: a small stand-in for the pack's node definitions and a
matching workflow, so the app is tested without ComfyUI, a GPU or the pack."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

APP_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_ROOT))

from radio_play_studio.config import Settings  # noqa: E402
from radio_play_studio.form import EPISODE_NODE_TYPE, WRITER  # noqa: E402

BANKS = ["roll (any eligible bank)", "original", "shakespeare", "my_story"]
STYLES = ["roll (any style)", "noir", "pastel"]
DIRECTOR = "OTR_VideoDirector"
#: As the pack lists them: local engines, cloud lanes and the custom row together.
STILL_ENGINES = ["cloud_flux_pro", "flux3_klein", "google_image", "ideo", "moss_image",
                 "z_image_turbo", "+ Add Custom Model"]


def writer_schema() -> dict:
    """The writer's inputs the form touches, in the pack's declared order."""
    text = lambda **o: ("STRING", {"default": "", **o})  # noqa: E731
    return {"input": {
        "required": {
            "source_bank": (BANKS, {"default": BANKS[0]}),
            "act_count": (["1", "2", "3"], {"default": "1"}),
            "num_characters": ("INT", {"default": 2, "min": 1, "max": 6}),
            "episode_language": (["English", "Spanish"], {"default": "English"}),
        },
        "optional": {
            "source_ref": text(),
            "visual_style": (STYLES, {"default": STYLES[0]}),
            "story_characters": text(multiline=True),
            "story_plot": text(multiline=True),
            "story_setting": text(multiline=True),
            "story_author": text(),
        },
    }}


def director_schema() -> dict:
    """The video director's three still-engine dropdowns, in the pack's order."""
    pick = lambda: (STILL_ENGINES, {"default": "flux3_klein"})  # noqa: E731
    return {"input": {"required": {"announcer_image_model": pick(),
                                   "character_image_model": pick(),
                                   "music_image_model": pick()}}}


@pytest.fixture
def schemas() -> dict:
    return {WRITER: writer_schema(), EPISODE_NODE_TYPE: {"input": {"required": {}}},
            DIRECTOR: director_schema()}


@pytest.fixture
def workflow(schemas) -> dict:
    """A small canvas workflow whose saved widget values are the defaults."""
    spec = schemas[WRITER]["input"]
    values = [opts[1]["default"] if isinstance(opts[0], str) else opts[1]["default"]
              for opts in list(spec["required"].values()) + list(spec["optional"].values())]
    return {
        "nodes": [
            {"id": 7, "type": WRITER, "inputs": [], "widgets_values": values},
            {"id": 9, "type": EPISODE_NODE_TYPE, "inputs": [], "widgets_values": []},
            {"id": 5, "type": DIRECTOR, "inputs": [],
             "widgets_values": ["flux3_klein", "flux3_klein", "flux3_klein"]},
            {"id": 12, "type": "MarkdownNote", "widgets_values": ["note"]},
        ],
        "links": [],
    }


@pytest.fixture
def settings(tmp_path, workflow) -> Settings:
    path = tmp_path / "otr_app.json"
    path.write_text(json.dumps(workflow), encoding="utf-8")
    return Settings(comfyui_url="http://comfy.invalid", api_url="http://bridge.invalid",
                    workflow_path=path, out_dir=tmp_path / "episodes")
