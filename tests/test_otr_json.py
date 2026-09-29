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


def test_empty_object_does_not_salvage_a_nested_brace_in_leftover_keys():
    raw = '{} "lines": [ {"speaker": "Ada", "text": "Hi"} ]'
    assert json.loads(OJ.extract_first_json_block(raw)) == {}


def test_stray_comma_in_an_empty_nested_object_stays_a_syntax_miss():
    raw = '{"n": 2, "scene_setting": "yard", "lines": [{,}]}'
    assert OJ.extract_first_json_block(raw) == ""


def test_json_fence_wins_over_an_earlier_thinking_fence():
    raw = (
        "```\nthinking about the act\n```\n"
        + _fenced(json.dumps(_ACT))
    )
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


def test_padded_nested_keys_match_the_unpadded_object():
    """Live Gemma 2026-09-14: leftover ``"speaker "`` / ``"lines "``."""
    raw = json.dumps({
        "n ": _ACT["n"],
        "scene_setting": _ACT["scene_setting"],
        "lines ": [
            {"speaker ": "Stomp", "text": _ACT["lines"][0]["text"]},
            {"speaker": "Tiptoe", "text": _ACT["lines"][1]["text"]},
        ],
    })
    parsed = OJ.parse_first_json_object(raw)
    assert parsed == _ACT
    assert OJ.parse_first_json_object(_fenced(raw)) == _ACT


def test_padded_key_collision_is_last_wins():
    raw = '{"speaker": "Ada", "speaker ": "Tom", "text": "Hi."}'
    assert OJ.parse_first_json_object(raw) == {"speaker": "Tom", "text": "Hi."}


def test_empty_key_after_strip_is_dropped():
    raw = '{"": 1, "  ": 2, "n": 3}'
    assert OJ.parse_first_json_object(raw) == {"n": 3}


def test_normalize_json_keys_does_not_mutate_string_values():
    raw = {"speaker ": "Stomp ", "title ": "  Fog  ", "nested": [{"a ": " x "}]}
    assert OJ.normalize_json_keys(raw) == {
        "speaker": "Stomp ",
        "title": "  Fog  ",
        "nested": [{"a": " x "}],
    }


def test_act_script_accepts_padded_lines_from_parsed_json():
    raw = json.dumps({
        "n ": 2,
        "scene_setting": "yard",
        "lines ": [
            {"speaker ": "Stomp", "text": "Move!"},
            {"speaker": "Tiptoe", "text": "Wait."},
        ],
    })
    act = MS.ActScript.model_validate(OJ.parse_first_json_object(raw))
    assert act.n == 2
    assert [line.speaker for line in act.lines] == ["Stomp", "Tiptoe"]
    assert act.lines[0].text == "Move!"


def test_shot_lock_parse_directives_accepts_padded_beat_id():
    from nodes import otr_shot_lock as sl
    raw = json.dumps([
        {"beat_id ": "b001 ", "expression": "grim",
         "motion": "steps forward", "camera ": "push in"},
    ])
    out = sl._parse_directives(raw, ["b001"])
    assert out["b001"]["camera"] == "push in"
    assert out["b001"]["expression"] == "grim"
    assert out["b001"]["motion"] == "steps forward"


# --- the decode error names the defect and where it is (2026-09-28 overnight) --
def test_an_undecodable_reply_names_its_defect_and_shows_where():
    """A stray quote inside a line of dialogue: the error used to say only
    "line 1 column 1 (char 0)", which was also everything a repair turn was
    told. It now carries the decoder's own reason and the text around it."""
    body = (
        '{\n  "n": 1,\n  "lines": [\n'
        '    {"speaker": "Tiptoe", "text": "One, two, three!"},\n'
        '    {"speaker": "Whiskers", "text": "Stomp shouted "Three!" and jumped."}\n'
        "  ]\n}"
    )
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object("Here it is.\n" + _fenced(body))
    text = str(exc.value)
    assert text.startswith("no decodable top-level JSON object found; in the object, ")
    assert "Expecting ',' delimiter" in text
    assert 'Stomp shouted "<<HERE>>Three!" and jumped.' in text
    assert "\n" not in text and "line 5" in text


def test_a_reply_with_no_object_at_all_keeps_the_old_message():
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object("I could not write that act.")
    assert str(exc.value) == "no decodable top-level JSON object found: line 1 column 1 (char 0)"


def test_a_cut_off_reply_names_where_it_stops():
    raw = '```json\n{"n": 1, "lines": [{"speaker": "Ada", "text": "Hi"}'
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object(raw)
    text = str(exc.value)
    assert "no decodable top-level" in text and "<<HERE>>" in text
    assert text.index("<<HERE>>") > text.index('"Hi"')


def test_the_error_is_about_the_fence_the_extractor_gave_up_on():
    """A first json fence holding only prose is skipped, as the extractor skips
    it; the defect named is the one in the fence that held the object
    (Composer QA of 7ff91d95)."""
    raw = ("```json\nHere is the act.\n```\n"
           '```json\n{"n": 1, "lines": [{"speaker": "A", "text": "He said "hi" twice"}]}\n```')
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object(raw)
    assert 'He said "<<HERE>>hi" twice' in str(exc.value)


def test_the_error_follows_a_hop_past_an_empty_preamble_object():
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object('Here is {} {"n": 1, "lines": [')
    text = str(exc.value)
    assert "Expecting value" in text and text.index("<<HERE>>") > text.index('"lines": [')


def test_a_stray_comma_is_named_as_written():
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object("{,}")
    assert "Expecting property name enclosed in double quotes near: {<<HERE>>,}" in str(exc.value)


# --- Sonnet QA of 7ff91d95 + ba6dd10e -------------------------------------------
def test_the_position_is_in_the_text_as_the_model_wrote_it():
    """A raw newline the repair tolerates comes first; the defect named is the
    stray quote after it, at its line as written, with no escape the model
    never wrote in the context."""
    body = '{\n "a": "Line one\ncontinues here",\n "b": "He said "hi" twice"\n}'
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object(_fenced(body))
    text = str(exc.value)
    assert 'He said "<<HERE>>hi" twice' in text and "line 4" in text
    assert "Invalid control character" not in text and chr(92) not in text


def test_the_object_named_is_the_one_the_decoder_got_furthest_into():
    raw = ('```\nplan: {a, b}\n```\n'
           '```\n{"n": 1, "x": "a "b" c"}\n```')
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object(raw)
    assert '"x": "a "<<HERE>>b" c"' in str(exc.value)


def test_the_context_is_sixty_characters_either_side():
    raw = '{"t": "' + "x" * 100 + '"q' + "y" * 100 + '"}'
    with pytest.raises(json.JSONDecodeError) as exc:
        OJ.parse_first_json_object(raw)
    near = str(exc.value).split("near: ", 1)[1].rsplit(": line", 1)[0]
    before, after = near.split("<<HERE>>")
    assert len(before) == 60 and before.endswith('x"')
    assert len(after) == 60 and after.startswith("qyyy")


def test_nesting_too_deep_is_a_json_miss_not_a_crash():
    """Where the recursion limit bites depends on the interpreter: in the
    decoder on one box, in the key normalization on another. Either way the
    ladder must see a JSON miss it can retry, never a RecursionError."""
    for depth in (1100, 5000):
        raw = '{"a":' * depth + "1" + "}" * depth
        OJ.extract_first_json_block(raw)                  # never raises
        with pytest.raises(json.JSONDecodeError, match="nesting too deep"):
            OJ.parse_first_json_object(raw)


def test_the_repair_aligns_back_onto_the_text_as_written():
    original = '{"a": "x\ty\nz", "b": [1, 2,],}'
    repaired = OJ._repair_llm_json(original)
    origin = OJ._origin_of(repaired, original)
    assert origin is not None and len(origin) == len(repaired) + 1
    assert all(repaired[j] == original[origin[j]]
               for j in range(len(repaired)) if repaired[j] == original[origin[j]])
    assert original[origin[repaired.index("z")]] == "z"
    assert origin[-1] == len(original)
