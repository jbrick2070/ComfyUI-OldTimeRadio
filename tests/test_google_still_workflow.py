# -*- coding: utf-8 -*-
"""The Google stills lane ships as a workflow: otr_google_still.

WHY (2026-09-25, plan row 0h). The stills-only Google lane was proven on
2026-09-19 as a hand-applied preset (config/profiles/google_still_1act.json,
added 67332dc5) and published an English Hamlet 1-act in 218 s. That preset
was deleted with the other unreferenced profile files in 2ac5ed88, which left
README, apple/CLOUD.md and apple/VIDEO_MODELS.md pointing readers at presets
that no longer existed and no Google workflow in the gallery at all.

The row copies the proven preset's lanes. What this file guards is the part a
reader relies on: every stage is a Google lane, nothing is downloaded, no GPU
is asked for, and the one key it needs is named.

Headless. No engine, no model, no GPU, no network.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from nodes._otr_shared import capability_profiles as cp
from nodes.otr_video_director import exact_menu_option_for
from tests._support.writer_slots import value

REPO = Path(__file__).resolve().parents[1]
ROW_ID = "otr_google_still"
WORKFLOW = REPO / "workflows" / f"{ROW_ID}.json"


@pytest.fixture(scope="module")
def graph():
    return json.loads(WORKFLOW.read_text(encoding="utf-8"))


def _node(graph, node_type):
    found = [n for n in graph["nodes"] if n["type"] == node_type]
    assert len(found) == 1, (node_type, len(found))
    return found[0]


def test_the_row_ships_on_the_cpu_and_names_its_key():
    row = cp.load_profile(ROW_ID)
    assert row["device_backend"] == "cpu"
    assert row["preflight"]["required_keys"] == ["OTR_GOOGLE_API_KEY"]
    assert WORKFLOW.is_file()


def test_the_shape_and_every_device_are_what_the_card_says(graph):
    """3 acts, 3 characters, and nothing asks for a GPU. The card quotes a
    1-act proof and says so; this pins the shape it actually opens at."""
    writer = _node(graph, "OTR_LedgerScriptWriter")
    assert value(writer, "act_count") == "3"
    assert value(writer, "num_characters") == 3
    assert value(writer, "llm_device") == "cpu"
    assert value(writer, "llm_quant_policy") == "none"
    assert value(_node(graph, "OTR_CastLock"), "voice_device") == "cpu"
    assert value(_node(graph, "OTR_VideoDirector"), "device_policy") == "cpu"


def test_the_writer_is_gemini_flash_and_flash_lite(graph):
    writer = _node(graph, "OTR_LedgerScriptWriter")
    assert value(writer, "creative_writing_model") == "google_api:slot-a"
    assert value(writer, "technical_model") == "google_api:slot-b"
    assert value(writer, "google_api_slot_a_model") == "gemini-flash-latest"
    assert value(writer, "google_api_slot_b_model") == "gemini-flash-lite-latest"


def test_voices_and_music_are_google(graph):
    castlock = _node(graph, "OTR_CastLock")
    assert value(castlock, "char_voice_engine") == "google_tts"
    assert value(castlock, "announcer_voice_engine") == "google_tts"
    assert value(_node(graph, "OTR_StableAudioTheme"), "engine") == "google_lyria"


def test_pictures_are_gemini_stills_and_veo_is_never_called(graph):
    """Tier 1 allows 10 Veo requests a day per model; an episode needs about
    sixteen. The stills are composited on the CPU instead."""
    director = _node(graph, "OTR_VideoDirector")
    still = exact_menu_option_for("still_flat")
    for role in ("announcer", "music", "character"):
        assert value(director, f"{role}_video_model") == still, role
        assert value(director, f"{role}_image_model") == "google_image", role
    assert "veo" not in json.dumps(graph["nodes"]).lower()


def test_nothing_downloads():
    path = REPO / "scripts" / "otr_provision.py"
    spec = importlib.util.spec_from_file_location("_otr_provision_google_still", path)
    provision = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provision)
    lanes = provision.profile_lanes(ROW_ID)
    assert lanes["automatic"] == [] and lanes["manual"] == [], lanes
