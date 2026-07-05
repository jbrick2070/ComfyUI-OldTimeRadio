"""Stage 1 story-pack foundation tests (multi-modal story schema).

Proves: (a) the science pack is BYTE-IDENTICAL to the live prompt constants
(runtime-import equality), (b) exact seam-key set, (c) the fail-loud matrix,
(d) accessor semantics, (e) DORMANCY -- no production node consumes the loader
yet, so the sci-fi run is unchanged.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_story_pack as sp
from nodes import _otr_line_composer as L
from nodes import _otr_outline as O
from nodes import _otr_style_picker as S

REPO = Path(__file__).resolve().parents[1]
PACK_PATH = REPO / "nodes" / "story_packs" / "science_news" / "science_news_default.json"

EXPECTED_SEAMS = frozenset({
    "outline_macro_system", "outline_phase_system", "outline_beat_system",
    "line_composer_system", "coda_system",
    "announcer_intro_system", "announcer_intro_safe_system", "announcer_outro_system",
    "style_pick_inventor_system", "style_pick_inventor_user",
    "style_pick_chooser_system", "style_pick_chooser_user",
})


def _expected_live() -> "dict[str, str]":
    """Seam -> the exact live runtime string it must mirror."""
    return {
        "outline_macro_system": O._MACRO_SYSTEM_PROMPT,
        "outline_phase_system": O._PHASE_SYSTEM_PROMPT,
        "outline_beat_system": O._BEAT_SYSTEM_PROMPT,
        "line_composer_system": L._SYSTEM_PROMPT,
        # coda is the UNCONDITIONAL runtime join (_otr_line_composer.py:3407).
        "coda_system": L._NEWS_CODA_SYSTEM + L._NEWS_CODA_SYSTEM_V2_EXAMPLES,
        "announcer_intro_system": L._ANNOUNCER_INTRO_SYSTEM,
        "announcer_intro_safe_system": L._ANNOUNCER_INTRO_SYSTEM_SAFE,
        "announcer_outro_system": L._ANNOUNCER_OUTRO_SYSTEM,
        "style_pick_inventor_system": S._INVENTOR_SYSTEM,
        "style_pick_inventor_user": S._INVENTOR_USER_TEMPLATE,
        "style_pick_chooser_system": S._CHOOSER_SYSTEM,
        "style_pick_chooser_user": S._CHOOSER_USER_TEMPLATE,
    }


@pytest.fixture
def pack():
    sp._PACK_CACHE.clear()
    return sp.load_pack(PACK_PATH)


# -- (a) byte-identity + (b) exact seam set --------------------------------

def test_allowlist_equals_authored_set():
    assert sp.PRODUCTION_SEAM_ALLOWLIST == EXPECTED_SEAMS


def test_pack_seam_keys_exact(pack):
    assert set(pack.prompt_stages) == EXPECTED_SEAMS


def test_pack_is_byte_identical_to_live_constants(pack):
    expected = _expected_live()
    assert set(expected) == EXPECTED_SEAMS  # keep this map complete
    for seam, live in expected.items():
        assert pack.prompt_stages[seam] == live, f"drift in seam {seam!r}"


def test_pack_metadata(pack):
    assert pack.source_bank_id == "science_news"
    assert pack.story_model_id == "science_news_default"
    assert pack.story_pipeline_id == "legacy_many_pass"
    assert pack.schema_version == "v2.0"


# -- (c) fail-loud matrix ---------------------------------------------------

def _valid() -> dict:
    return {
        "source_bank_id": "b",
        "story_model_id": "m",
        "story_pipeline_id": "p",
        "schema_version": "v2.0",
        "prompt_stages": {"line_composer_system": "hi"},
    }


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "pack.json"
    p.write_text(text, encoding="utf-8")
    return p


def _write_obj(tmp_path: Path, obj: dict) -> Path:
    return _write(tmp_path, json.dumps(obj, ensure_ascii=False))


def test_missing_file():
    sp._PACK_CACHE.clear()
    with pytest.raises(sp.StoryPackNotFoundError):
        sp.load_pack(REPO / "nodes" / "story_packs" / "does_not_exist.json")


def test_malformed_json(tmp_path):
    sp._PACK_CACHE.clear()
    with pytest.raises(sp.StoryPackParseError):
        sp.load_pack(_write(tmp_path, "{ not json"))


def test_duplicate_key_top_level(tmp_path):
    sp._PACK_CACHE.clear()
    text = ('{"source_bank_id":"a","source_bank_id":"b",'
            '"story_model_id":"m","story_pipeline_id":"p",'
            '"schema_version":"v2.0","prompt_stages":{}}')
    with pytest.raises(sp.StoryPackParseError):
        sp.load_pack(_write(tmp_path, text))


def test_duplicate_key_nested(tmp_path):
    sp._PACK_CACHE.clear()
    text = ('{"source_bank_id":"a","story_model_id":"m","story_pipeline_id":"p",'
            '"schema_version":"v2.0","prompt_stages":'
            '{"line_composer_system":"x","line_composer_system":"y"}}')
    with pytest.raises(sp.StoryPackParseError):
        sp.load_pack(_write(tmp_path, text))


def test_unknown_top_level_key(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["bogus_field"] = 1
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_missing_required_key(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    del obj["schema_version"]
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_unknown_schema_version(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["schema_version"] = "v9.9"
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_unknown_seam_key(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["prompt_stages"] = {"not_a_real_seam": "x"}
    with pytest.raises(sp.UnknownSeamError):
        sp.load_pack(_write_obj(tmp_path, obj))


def test_whitespace_only_seam_value(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["prompt_stages"] = {"line_composer_system": "   \n\t"}
    with pytest.raises(sp.StoryPackValidationError):
        sp.load_pack(_write_obj(tmp_path, obj))


# -- (d) accessor semantics -------------------------------------------------

def test_get_pack_prompt_present(pack):
    val = pack.prompt_stages["coda_system"]
    assert sp.get_pack_prompt(pack, "coda_system") == val
    assert sp.get_pack_prompt_or_none(pack, "coda_system") == val


def test_accessor_absent_seam(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()  # only line_composer_system present
    p = sp.load_pack(_write_obj(tmp_path, obj))
    assert sp.get_pack_prompt_or_none(p, "coda_system") is None
    with pytest.raises(sp.StoryPackValidationError):
        sp.get_pack_prompt(p, "coda_system")


def test_accessor_unknown_seam_raises(pack):
    with pytest.raises(sp.UnknownSeamError):
        sp.get_pack_prompt(pack, "not_a_real_seam")
    with pytest.raises(sp.UnknownSeamError):
        sp.get_pack_prompt_or_none(pack, "not_a_real_seam")


def test_or_none_preserves_exact_content(tmp_path):
    sp._PACK_CACHE.clear()
    obj = _valid()
    obj["prompt_stages"] = {"line_composer_system": "  padded content  "}
    p = sp.load_pack(_write_obj(tmp_path, obj))
    # non-empty after strip -> returns the ORIGINAL untrimmed bytes
    assert sp.get_pack_prompt_or_none(p, "line_composer_system") == "  padded content  "


# -- (e) dormancy guard -----------------------------------------------------

def test_stage1_is_dormant_no_production_consumer():
    """No production node imports/calls the loader in Stage 1 -> the sci-fi run
    is byte-for-byte unchanged. Only the module itself + tests may reference it."""
    nodes_dir = REPO / "nodes"
    offenders = []
    for py in nodes_dir.rglob("*.py"):
        if py.name == "_otr_story_pack.py":
            continue
        text = py.read_text(encoding="utf-8", errors="replace")
        if "_otr_story_pack" in text or "get_pack_prompt" in text:
            offenders.append(str(py.relative_to(REPO)))
    assert not offenders, f"Stage 1 must stay dormant; found consumers: {offenders}"
