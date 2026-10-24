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
#: The writer's model list as the pack shapes it: local models, then the cloud
#: handles (slot A creative, slot B technical).
WRITERS = ["google/gemma-4-12b-it", "comfy_native:gemma4-12b-int8-convrot",
           "openrouter:slot-a", "openrouter:slot-b", "google_api:slot-a", "google_api:slot-b",
           "comfy:slot-a", "comfy:slot-b"]
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
            "creative_writing_model": (WRITERS, {"default": WRITERS[1]}),
            "openrouter_slot_a_model": (["(enable OpenRouter)", "~anthropic/claude-sonnet-latest"],
                                        {"default": "(enable OpenRouter)"}),
            "google_api_slot_a_model": (["(select Google API model)", "gemini-pro-latest"],
                                        {"default": "(select Google API model)"}),
            "comfy_slot_a_model": (["(enable Comfy Credits)", "anthropic/claude-sonnet-5"],
                                   {"default": "(enable Comfy Credits)"}),
        },
    }}


def director_schema() -> dict:
    """The video director's three still-engine dropdowns, in the pack's order."""
    # A fresh list each: a test that adds an engine to one dropdown must not
    # change the others, or the next test's.
    pick = lambda: (list(STILL_ENGINES), {"default": "flux3_klein"})  # noqa: E731
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
