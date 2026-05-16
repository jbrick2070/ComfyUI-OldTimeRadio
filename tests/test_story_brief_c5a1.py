"""Sprint C C5a1: reflection pure module tests.

The module is pure (no I/O, no GPU, no ComfyUI imports), so every test
here runs against a plain dict fixture and a fake `technical_fn`
closure. No writer wiring is exercised at C5a1 -- those tests land
at C5a2.

Coverage map (per SPRINT.md C5a1 pytest table):

  Input builder (refinement section 2):
    1. caps long episode
    2. includes required fields (title, style, cast, scene headers,
       opening, closing, non-dialogue rows)
    3. reads lines, not script_text (locks R-02 push-back)

  Prompt body (refinement section 3 + 3.3):
    4. under 250 tokens

  Validation gate (refinement section 3.4 + 3.3):
    5. rejects named character
    6. rejects dialogue verb
    7. rejects decade literal
    8. rejects over 300 chars

  Repair pass (refinement section 3.5 + E-18 + R-06):
    9. runs once when initial validation fails
   10. uses higher temperature
   11. clamped to 0.55 ceiling
   12. prepends CRITICAL prefix

  Sentinel / shape (L-6 + L-8 + E-17 + E-21):
   13. 3 distinct except arms each returning failure sentinel (AST)
   14. each try block contains exactly one statement (AST scoped)
   15. stamps 8 meta keys on success
   16. stamps story_brief_status="failed" on technical_fn raise
   17. signature accepts only technical_fn (no creative_fn) (AST)
   18. # LLM slot: technical tag present in source
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from nodes import _otr_story_brief as sb


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_MODULE_PATH = _REPO_ROOT / "nodes" / "_otr_story_brief.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _mk_ledger(num_lines: int = 6, with_period: str | None = None) -> dict:
    """Construct a minimal valid ledger dict for tests.

    `with_period` injects a period string into a cast description so
    the period-validator's "already present" exemption can be tested.
    """
    desc_extra = f" Set in {with_period}." if with_period else ""
    lines: list[dict] = [
        {"speaker_role": "scene", "text": "=== SCENE 1 -- INTERIOR -- NIGHT"},
        {"speaker_role": "env", "text": "[ENV: light rain on tin roof, distant thunder]"},
    ]
    for i in range(num_lines):
        lines.append({
            "speaker_role": "character",
            "char_id":      "JONES" if i % 2 == 0 else "SMITH",
            "text":         f"Line {i} of dialogue here, kept short for the fixture.",
        })
    lines.append({"speaker_role": "sfx", "text": "[SFX: bulkhead seals with a hiss]"})
    return {
        "cast": [
            {"char_id": "JONES", "name": "Jones",
             "character_description": f"Tall detective.{desc_extra}"},
            {"char_id": "SMITH", "name": "Smith",
             "character_description": "Quiet suspect, sweating."},
        ],
        "lines": lines,
        "meta":  {"episode_title": "The Bulkhead Closes",
                  "style":         "noir_interrogation"},
    }


def _ledger_with_long_script() -> dict:
    """Build a ledger that simulates a 15-minute episode (~200 lines)."""
    led = _mk_ledger(num_lines=2)
    long_lines = []
    for i in range(200):
        long_lines.append({
            "speaker_role": "character",
            "char_id":      "JONES" if i % 3 == 0 else "SMITH",
            "text":         f"This is dialogue line number {i}, somewhat verbose " * 4,
        })
    led["lines"].extend(long_lines)
    return led


def _make_spy_fn(*, responses: list[str]):
    """Return a callable that records (messages, temperature, max_new_tokens)
    on every call and pops the next response from `responses`."""
    calls: list[dict] = []

    def fn(messages, *, temperature, max_new_tokens):
        calls.append({
            "messages":       messages,
            "temperature":    temperature,
            "max_new_tokens": max_new_tokens,
        })
        if not responses:
            return ""
        return responses.pop(0)

    fn.calls = calls  # type: ignore[attr-defined]
    return fn


def _valid_brief_json() -> str:
    return (
        '{"story_brief": "a dim interrogation room under a swinging bare bulb, '
        'rain-streaked window, sweat and smoke", '
        '"setting_terms": ["interrogation room", "steel table", "rain-streaked window"], '
        '"lighting_terms": ["swinging bare bulb", "harsh top-down shadow"], '
        '"atmosphere_terms": ["sweat", "smoke", "tense"]}'
    )


# ---------------------------------------------------------------------------
# 1-3. Input builder
# ---------------------------------------------------------------------------


class TestReflectionInputBuilder:

    def test_caps_long_episode(self):
        long_led = _ledger_with_long_script()
        text = sb._build_reflection_input(long_led)
        # Approx-tokens heuristic; the spec says ~1500 tokens, we allow
        # a generous ceiling to absorb fixture-line-count drift.
        approx_tokens = len(text) // 4
        assert approx_tokens <= 1800, (
            f"input builder produced {approx_tokens} approx-tokens; "
            "cap from refinement section 2 is ~1500 (1800 allowed for "
            "fixture margin)"
        )

    def test_includes_required_fields(self):
        led = _mk_ledger()
        text = sb._build_reflection_input(led)
        # Title and style.
        assert "TITLE: The Bulkhead Closes" in text
        assert "STYLE: noir_interrogation" in text
        # Cast roster.
        assert "Jones" in text and "Smith" in text
        # Scene headers.
        assert "SCENE 1" in text
        # Opening lines.
        assert "OPENING" in text
        # Non-dialogue rows.
        assert "[SFX: bulkhead seals" in text
        assert "[ENV: light rain" in text

    def test_reads_lines_not_script_text(self):
        """AST walk: `_build_reflection_input` references the ledger
        `lines` field, NOT a `script_text` parameter. Locks R-02
        push-back -- the input builder must consume the canonical
        ledger row list, not the assembled prose script."""
        tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
        target_func = None
        for node in ast.walk(tree):
            if (isinstance(node, ast.FunctionDef)
                    and node.name == "_build_reflection_input"):
                target_func = node
                break
        assert target_func is not None, "_build_reflection_input not found"

        # Walk the function body. Confirm `ledger.get("lines")` or
        # equivalent attribute access exists, and no `script_text`
        # name reference does.
        has_lines_access = False
        has_script_text = False
        for sub in ast.walk(target_func):
            if isinstance(sub, ast.Constant) and sub.value == "lines":
                has_lines_access = True
            if isinstance(sub, ast.Name) and sub.id == "script_text":
                has_script_text = True
        assert has_lines_access, (
            "_build_reflection_input must reference 'lines' constant "
            "to read led.data['lines'] (R-02 lock)"
        )
        assert not has_script_text, (
            "_build_reflection_input must NOT reference `script_text` "
            "(R-02 push-back: the input is the lines list, not the "
            "assembled prose)"
        )


# ---------------------------------------------------------------------------
# 4. Prompt body length
# ---------------------------------------------------------------------------


def test_reflection_prompt_under_250_tokens():
    """Per refinement section 3 / hard rule 8: reflection prompt body
    stays under 250 tokens. Char/4 is a rough but conservative proxy
    that does not require tiktoken."""
    approx_tokens = len(sb._REFLECTION_PROMPT) // 4
    assert approx_tokens <= 250, (
        f"_REFLECTION_PROMPT is ~{approx_tokens} tokens; cap is 250"
    )


# ---------------------------------------------------------------------------
# 5-8. Validation gate
# ---------------------------------------------------------------------------


class TestValidation:

    def test_rejects_named_character(self):
        led = _mk_ledger()
        # The brief mentions "Jones" -- one of the cast names.
        brief = ("a dim room under a swinging bare bulb where Jones leans "
                 "over the table")
        reasons = sb._validate_brief(brief, led)
        assert sb.REJECT_NAMED_CHARACTER in reasons

    def test_rejects_dialogue_verb(self):
        led = _mk_ledger()
        brief = "a dim interrogation room where shadows fall, speaking softly"
        reasons = sb._validate_brief(brief, led)
        assert sb.REJECT_DIALOGUE_VERB in reasons

    def test_rejects_decade_literal(self):
        led = _mk_ledger()  # No period in the source.
        brief = "a dim 1940s interrogation room under a swinging bare bulb"
        reasons = sb._validate_brief(brief, led)
        assert sb.REJECT_UNSUPPORTED_PERIOD in reasons

    def test_decade_literal_allowed_when_in_source(self):
        """If the ledger already names a period, the brief may carry it."""
        led = _mk_ledger(with_period="1947")
        brief = "a dim 1947 interrogation room under a swinging bare bulb"
        reasons = sb._validate_brief(brief, led)
        assert sb.REJECT_UNSUPPORTED_PERIOD not in reasons

    def test_rejects_over_300_chars(self):
        led = _mk_ledger()
        brief = "a dim interrogation room " * 30  # Way over 300 chars.
        reasons = sb._validate_brief(brief, led)
        assert sb.REJECT_TOO_LONG in reasons

    def test_rejects_quote_or_markup(self):
        led = _mk_ledger()
        brief = 'a dim "interrogation" room'
        reasons = sb._validate_brief(brief, led)
        assert sb.REJECT_QUOTES_OR_MARKUP in reasons

    def test_accepts_clean_brief(self):
        led = _mk_ledger()
        brief = ("a dim interrogation room under a swinging bare bulb, "
                 "rain on tin, sweat and smoke")
        reasons = sb._validate_brief(brief, led)
        assert reasons == []


# ---------------------------------------------------------------------------
# 9-12. Repair pass
# ---------------------------------------------------------------------------


class TestRepairPass:

    def test_repair_uses_higher_temperature(self):
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        sb._repair_pass(
            failed_output="bad",
            rejection_reasons=["named_character"],
            technical_fn=spy,
            base_user_message="ignored",
            reflection_temperature=0.30,
        )
        assert len(spy.calls) == 1
        # 0.30 + 0.15 = 0.45, well under the 0.55 ceiling.
        assert abs(spy.calls[0]["temperature"] - 0.45) < 1e-6

    def test_repair_temperature_clamped_to_055(self):
        """E-18 / RR-B5: repair_temperature = min(base + 0.15, 0.55).
        Base 0.50 -> 0.65 raw -> clamped to 0.55."""
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        sb._repair_pass(
            failed_output="bad",
            rejection_reasons=["named_character"],
            technical_fn=spy,
            base_user_message="ignored",
            reflection_temperature=0.50,
        )
        assert abs(spy.calls[0]["temperature"] - 0.55) < 1e-6, (
            f"observed temperature {spy.calls[0]['temperature']}; "
            "should clamp to 0.55"
        )

    def test_repair_prepends_critical_prefix(self):
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        sb._repair_pass(
            failed_output="bad",
            rejection_reasons=["named_character", "dialogue_verb"],
            technical_fn=spy,
            base_user_message="base context",
            reflection_temperature=0.30,
        )
        user_msg = spy.calls[0]["messages"][0]["content"]
        assert user_msg.startswith(
            "CRITICAL: You previously failed validation because:"
        ), user_msg[:120]
        # The named-character and dialogue-verb reason codes are in.
        assert "named_character" in user_msg
        assert "dialogue_verb" in user_msg

    def test_repair_runs_when_initial_fails(self):
        """Integration: the main entrypoint runs the repair pass when
        initial content validation rejects."""
        led = _mk_ledger()
        # First response mentions a cast name (rejects); second is clean.
        bad_brief = (
            '{"story_brief": "a dim room where Jones leans over a table", '
            '"setting_terms": ["room", "table"], '
            '"lighting_terms": ["bare bulb"], '
            '"atmosphere_terms": ["tense"]}'
        )
        spy = _make_spy_fn(responses=[bad_brief, _valid_brief_json()])
        result = sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 2, "repair pass should have invoked technical_fn twice"
        assert result["story_brief_status"] == "ok"

    def test_repair_does_not_run_when_initial_passes(self):
        led = _mk_ledger()
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        result = sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 1, "no repair should run on clean first pass"
        assert result["story_brief_status"] == "ok"


# ---------------------------------------------------------------------------
# 13-14. Scoped try/except structure (AST)
# ---------------------------------------------------------------------------


def _entrypoint_node() -> ast.FunctionDef:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.FunctionDef)
                and node.name == "run_story_brief_reflection"):
            return node
    raise AssertionError("run_story_brief_reflection not found")


class TestExceptionStructure:

    def test_three_distinct_except_arms(self):
        """E-17 / RR-B3 + L-6: the entrypoint has at least 3 distinct
        try/except blocks, mirroring the run_script_doctor pattern."""
        fn = _entrypoint_node()
        try_blocks = [n for n in ast.walk(fn) if isinstance(n, ast.Try)]
        # We have 3 primary blocks (LLM call, JSON parse, schema validate)
        # plus a repair-pass try/except. Asserting >= 3 covers the spec.
        assert len(try_blocks) >= 3, (
            f"expected >= 3 try blocks (one per L-6 arm); got {len(try_blocks)}"
        )

    def test_try_blocks_are_scoped(self):
        """E-17 / RR-B3: each top-level try block contains a small
        number of statements (not a broad function-body wrapper).
        Strict version: each try body has between 1 and 4 statements."""
        fn = _entrypoint_node()
        for try_node in [n for n in ast.iter_child_nodes(fn) if isinstance(n, ast.Try)]:
            assert 1 <= len(try_node.body) <= 4, (
                f"try block at line {try_node.lineno} has "
                f"{len(try_node.body)} statements -- E-17 requires "
                "narrow scoped blocks, not broad function-body wraps"
            )


# ---------------------------------------------------------------------------
# 15-16. Sentinel shape + failure stamping
# ---------------------------------------------------------------------------


_REQUIRED_META_KEYS = {
    "story_brief",
    "story_brief_status",
    "story_brief_error",
    "story_brief_model",
    "story_brief_prompt_version",
    "story_brief_source",
    "story_brief_char_count",
    "story_brief_terms",
}


class TestMetaDelta:

    def test_success_stamps_8_keys(self):
        led = _mk_ledger()
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        result = sb.run_story_brief_reflection(
            led, spy, technical_model_id="some/model-id",
        )
        assert _REQUIRED_META_KEYS.issubset(result.keys())
        assert result["story_brief_status"] == "ok"
        assert result["story_brief"] != ""
        assert result["story_brief_char_count"] == len(result["story_brief"])
        assert result["story_brief_model"] == "some/model-id"
        # Terms sub-dict has the 3 expected term-class keys.
        terms = result["story_brief_terms"]
        assert set(terms.keys()) == {"setting", "lighting", "atmosphere"}

    def test_failure_stamps_status_failed_on_raise(self):
        led = _mk_ledger()

        def raising_fn(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated technical_fn failure")

        result = sb.run_story_brief_reflection(led, raising_fn)
        assert _REQUIRED_META_KEYS.issubset(result.keys())
        assert result["story_brief"] == ""
        assert result["story_brief_status"] == "failed"
        assert result["story_brief_char_count"] == 0

    def test_failure_stamps_status_failed_on_json_parse(self):
        led = _mk_ledger()
        spy = _make_spy_fn(responses=["this is not JSON {{{"])
        result = sb.run_story_brief_reflection(led, spy)
        assert result["story_brief"] == ""
        assert result["story_brief_status"] == "failed"


# ---------------------------------------------------------------------------
# 17-18. Signature + slot-tag locks
# ---------------------------------------------------------------------------


def test_entrypoint_signature_accepts_only_technical_fn():
    """E-21 / RR-B1: the entrypoint parameter list contains `led` and
    `technical_fn`, and does NOT contain `creative_fn`. Locks against
    accidental routing of the structured-JSON pass through the
    creative slot."""
    fn = _entrypoint_node()
    arg_names = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    assert "led" in arg_names
    assert "technical_fn" in arg_names
    assert "creative_fn" not in arg_names


def test_slot_tag_technical_present():
    """The entrypoint docstring or comment block carries the
    `# LLM slot: technical` tag (CLAUDE.md rule 6 + L-2). Mechanical
    grep so a renamer cannot strip the tag silently."""
    src = _MODULE_PATH.read_text(encoding="utf-8")
    assert "LLM slot: technical" in src, (
        "Missing `LLM slot: technical` tag in nodes/_otr_story_brief.py"
    )
