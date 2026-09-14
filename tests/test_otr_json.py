"""Shared LLM JSON extractor: fences, Gemma pretty-print, fail-closed nested child.

CPU only. No model. UTF-8 no BOM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_json as OJ  # noqa: E402
from nodes import _otr_my_story as MS  # noqa: E402


_ACT = {
    "n": 2,
    "scene_setting": "The playground, moving toward the toy kitchen",
    "lines": [
        {"speaker": "Stomp", "text": "Move! We have to see what is inside!"},
        {"speaker": "Tiptoe",
         "text": "Wait, Stomp. Stay back. We don't know what these things are."},
    ],
}


def _fenced(body: str) -> str:
    return "```json\n%s\n```" % body


def test_plain_object_still_parses():
    raw = json.dumps(_ACT)
    assert OJ.parse_first_json_object(raw) == _ACT


def test_clean_fence_still_parses():
    raw = _fenced(json.dumps(_ACT, indent=2))
    assert OJ.parse_first_json_object(raw) == _ACT


def test_fence_with_prose_before_the_object_parses():
    """Repair attempts wrap the act in fences plus a one-line preamble.
    Decoding the fence body from column 0 fail-closed the 2026-09-14
    RunPod my_story_act_2 ladder."""
    raw = _fenced("Here is the repaired act:\n" + json.dumps(_ACT))
    assert OJ.parse_first_json_object(raw) == _ACT


def test_unescaped_newline_inside_a_string_parses():
    """Gemma pretty-prints dialogue across real line breaks. The ladder
    heartbeat collapses whitespace, so the live log looks valid."""
    raw = _fenced(
        '{\n  "n": 2,\n  "scene_setting": "yard",\n  "lines": [\n'
        '    {"speaker": "Stomp", "text": "Move!\\nWe have to see."},\n'
        '    {"speaker": "Tiptoe", "text": "Wait, Stomp.\nStay back."}\n'
        "  ]\n}"
    )
    parsed = OJ.parse_first_json_object(raw)
    assert parsed["n"] == 2
    assert parsed["lines"][1]["text"] == "Wait, Stomp.\nStay back."


def test_repaired_block_need_not_be_a_substring_of_raw():
    raw = '{"n": 1, "text": "Wait, Stomp.\nStay back."}'
    block = OJ.extract_first_json_block(raw)
    assert block not in raw
    assert json.loads(block) == OJ.parse_first_json_object(raw)
    assert json.loads(block)["text"] == "Wait, Stomp.\nStay back."


def test_trailing_comma_before_closing_brace_parses():
    raw = _fenced('{"n": 2, "scene_setting": "yard", "lines": ['
                  '{"speaker": "Stomp", "text": "Move!"},],}')
    parsed = OJ.parse_first_json_object(raw)
    assert parsed["lines"][0]["speaker"] == "Stomp"


def test_fence_preamble_plus_raw_newline_together_parse():
    body = (
        "Here is the repaired act:\n"
        '{\n  "n": 2,\n  "scene_setting": "yard",\n  "lines": [\n'
        '    {"speaker": "Stomp", "text": "Move!"},\n'
        '    {"speaker": "Tiptoe", "text": "Wait, Stomp.\nStay back."}\n'
        "  ]\n}"
    )
    parsed = OJ.parse_first_json_object(_fenced(body))
    assert parsed["lines"][1]["text"] == "Wait, Stomp.\nStay back."


def test_malformed_outer_does_not_salvage_a_nested_child():
    """Codex P5: an unclosed envelope must not return the inner line object."""
    raw = '{ "n": 1, "lines": [ {"speaker": "Ada", "text": "Hi"} ]'
    assert OJ.extract_first_json_block(raw) == ""
    with pytest.raises(json.JSONDecodeError, match="no decodable top-level"):
        OJ.parse_first_json_object(raw)


def test_undecodable_fence_does_not_search_past_the_fence():
    raw = "```json\nnot an object\n```\n" + json.dumps(_ACT)
    assert OJ.extract_first_json_block(raw) == ""


def test_repaired_empty_object_from_a_stray_comma_stays_a_syntax_miss():
    assert OJ.extract_first_json_block("{,}") == ""


def test_empty_preamble_object_does_not_hide_the_real_artifact():
    raw = "Here is {} " + json.dumps(_ACT)
    assert OJ.parse_first_json_object(raw) == _ACT


def test_genuine_empty_object_still_extracts():
    assert json.loads(OJ.extract_first_json_block("{}")) == {}


def test_unescaped_inner_quote_still_fail_closed():
    raw = '{"n": 1, "text": "He said "hello""}'
    assert OJ.extract_first_json_block(raw) == ""


def test_act_script_accepts_a_tex_line_from_parsed_json():
    """Live attempt-1 path: JSON parsed; lines[10] used ``tex`` not ``text``."""
    raw = json.dumps({
        "n": 2,
        "scene_setting": "yard",
        "lines": [
            {"speaker": "Stomp", "text": "Move!"},
            {"speaker": "Stomp", "tex": "It's glowing like a tiny star."},
        ],
    })
    act = MS.ActScript.model_validate(OJ.parse_first_json_object(raw))
    assert act.lines[1].text == "It's glowing like a tiny star."
