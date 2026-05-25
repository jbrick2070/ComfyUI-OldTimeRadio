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

  Retry ladder (Sprint 2A/2D -- structured_call):
    9. structural retry re-rolls when initial validation fails
   10. no retry runs on a clean first pass
   11. structural retry LOWERS temperature (Sprint 2B contract)

  Sentinel / shape (L-8 + E-21 + structured_call):
   12. entrypoint routes through structured_call (AST)
   13. StructuredCallFailedError + broad slot-fn except handled (AST)
   14. try blocks stay narrowly scoped (AST)
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
# 9-11. Retry ladder (structured_call)
# ---------------------------------------------------------------------------
# Sprint 2A/2D: the hand-rolled repair pass (_repair_pass /
# _build_repair_messages) was removed when run_story_brief_reflection
# was converted onto the shared structured_call retry ladder. The old
# direct _repair_pass unit tests (raised-temperature + 0.55-clamp +
# CRITICAL-prefix assertions) tested behavior the ladder deliberately
# replaces: the ladder LOWERS temperature on a structural retry rather
# than raising it (the Sprint 2B fix). What survives is the observable
# contract -- a rejected first response re-rolls the slot fn, a clean
# first response does not, and the re-roll runs at a lower temperature.


class TestRetryLadder:

    def test_retry_runs_when_initial_fails(self):
        """A content-rejected first response must re-roll the ladder --
        the entrypoint invokes technical_fn a second time."""
        led = _mk_ledger()
        # First response mentions a cast name (content reject); second
        # is clean.
        bad_brief = (
            '{"story_brief": "a dim room where Jones leans over a table", '
            '"setting_terms": ["room", "table"], '
            '"lighting_terms": ["bare bulb"], '
            '"atmosphere_terms": ["tense"]}'
        )
        spy = _make_spy_fn(responses=[bad_brief, _valid_brief_json()])
        result = sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 2, (
            "structural retry should have invoked technical_fn twice"
        )
        assert result["story_brief_status"] == "ok"

    def test_retry_does_not_run_when_initial_passes(self):
        led = _mk_ledger()
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        result = sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 1, "no retry should run on clean first pass"
        assert result["story_brief_status"] == "ok"

    def test_structural_retry_lowers_temperature(self):
        """Sprint 2B contract: the Attempt 2 structural retry runs at a
        temperature STRICTLY BELOW the base attempt -- a JSON-schema
        re-roll lowers entropy, it never raises it (the old repair pass
        RAISED it by +0.15)."""
        led = _mk_ledger()
        bad_brief = (
            '{"story_brief": "a dim room where Jones leans over a table", '
            '"setting_terms": ["room"], "lighting_terms": ["bulb"], '
            '"atmosphere_terms": ["tense"]}'
        )
        spy = _make_spy_fn(responses=[bad_brief, _valid_brief_json()])
        sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 2, "structural retry did not run"
        assert abs(spy.calls[0]["temperature"]
                   - sb._REFLECTION_TEMPERATURE) < 1e-6
        assert spy.calls[1]["temperature"] < spy.calls[0]["temperature"], (
            f"Attempt 2 temperature {spy.calls[1]['temperature']} must be "
            f"below base {spy.calls[0]['temperature']}"
        )


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

    def test_entrypoint_routes_through_structured_call(self):
        """Sprint 2A/2D: the entrypoint delegates the call + retry to
        the shared structured_call ladder. Lock that it does -- the
        body must reference `structured_call` rather than re-growing a
        hand-rolled call/parse/validate sequence."""
        fn = _entrypoint_node()
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)}
        assert "structured_call" in names, (
            "run_story_brief_reflection no longer routes through "
            "structured_call -- the Wave 2 ladder conversion regressed"
        )

    def test_structured_call_failure_is_handled(self):
        """The entrypoint must catch StructuredCallFailedError (an
        exhausted ladder) AND the broad slot-fn exception (structured_
        call does not catch slot-fn failures) so every path maps to the
        failure sentinel -- audio is king, this call site never raises."""
        fn = _entrypoint_node()
        handled: set[str] = set()
        for node in ast.walk(fn):
            if isinstance(node, ast.ExceptHandler) and node.type is not None:
                for name in ast.walk(node.type):
                    if isinstance(name, ast.Name):
                        handled.add(name.id)
        assert "StructuredCallFailedError" in handled, (
            "entrypoint does not catch StructuredCallFailedError"
        )
        assert "Exception" in handled, (
            "entrypoint does not catch the broad slot-fn exception"
        )

    def test_try_blocks_are_scoped(self):
        """Each top-level try block stays narrowly scoped (1-4
        statements), not a broad function-body wrapper."""
        fn = _entrypoint_node()
        for try_node in [n for n in ast.iter_child_nodes(fn) if isinstance(n, ast.Try)]:
            assert 1 <= len(try_node.body) <= 4, (
                f"try block at line {try_node.lineno} has "
                f"{len(try_node.body)} statements -- requires "
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
