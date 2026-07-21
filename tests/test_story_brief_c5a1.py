"""Sprint C C5a1: reflection pure module tests.

The module is pure (no I/O, no GPU, no ComfyUI imports), so every test
here runs against a plain dict fixture and a fake `technical_fn`
closure. No writer wiring is exercised at C5a1 -- those tests land
at C5a2.

Coverage map (per C5a1 pytest table):

  Input builder (refinement section 2):
    1. caps long episode
    2. includes required fields (title, style, cast, scene headers,
       opening, closing, non-dialogue rows)
    3. reads lines, not script_text (locks R-02 push-back)

  Input sanitization (Sprint 3G):
    cast names + proper nouns are replaced with neutral tokens
    (character_a, source_entity) BEFORE the LLM sees the text;
    substitution is deterministic and stable within one call.

  Prompt body (refinement section 3 + 3.3 + 3G):
    4. under 250 tokens; the redundant name/proper-noun suppression
       list is trimmed now the input is pre-sanitized.

  Validation gate (refinement section 3.4 + 3.3):
    5. rejects named character
    6. rejects dialogue verb
    7. rejects decade literal
    8. rejects over 300 chars
    The OUTPUT reject-list safety net stays live after 3G input
    sanitization (TestOutputRejectSafetyNetUnchanged).

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
        """The input builder still emits every required section.

        Sprint 3G: the TITLE and CAST sections are now sanitized -- the
        episode title is a proper-noun phrase and collapses to a
        `source_entity` token, and cast names collapse to `character_*`
        tokens. The section HEADERS, the STYLE controlled-vocab slug,
        and the structural SFX / ENV markers all survive verbatim. The
        un-sanitized-name assertion is intentionally GONE -- the
        builder must never hand the LLM a real cast name (see
        TestReflectionInputSanitization below)."""
        led = _mk_ledger()
        text = sb._build_reflection_input(led)
        # Section headers + the controlled-vocab style slug pass through.
        assert "TITLE: " in text
        assert "STYLE: noir_interrogation" in text
        assert "CAST:" in text
        # Cast roster present as neutral tokens, not real names.
        assert "character_a" in text
        # Scene headers.
        assert "SCENE 1" in text
        # Opening lines.
        assert "OPENING" in text
        # Non-dialogue rows -- structural markers, no names.
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
# 3G. Input sanitization -- cast names + proper nouns -> neutral tokens
# ---------------------------------------------------------------------------
# Sprint 3G: `_build_reflection_input` runs a deterministic strip before
# the text reaches the LLM. Every cast name and proper noun is replaced
# with a neutral token (character_a, source_entity) so the model never
# sees a real name it is then told to suppress. The OUTPUT reject lists
# in `_validate_brief` stay as the safety net -- proven still live by
# TestValidation below and TestOutputRejectSafetyNetUnchanged.


class TestReflectionInputSanitization:

    def test_cast_names_replaced_with_neutral_tokens(self):
        """No cast name surface form survives into the LLM-facing text;
        each maps to a stable `character_*` token instead."""
        led = _mk_ledger()
        text = sb._build_reflection_input(led)
        # The fixture cast is Jones / Smith.
        assert "Jones" not in text, "real cast name 'Jones' leaked to the LLM"
        assert "Smith" not in text, "real cast name 'Smith' leaked to the LLM"
        # char_id tokens are the upper-case form of the same names; the
        # case-insensitive substitution must catch them too.
        assert "JONES" not in text
        assert "SMITH" not in text
        # Both characters present as distinct neutral tokens.
        assert "character_a" in text
        assert "character_b" in text

    def test_cast_substitution_is_deterministic(self):
        """Same ledger -> byte-identical sanitized input on every call,
        and the name->token mapping is stable (Jones is always the same
        token within a build)."""
        led = _mk_ledger()
        first = sb._build_reflection_input(led)
        second = sb._build_reflection_input(led)
        assert first == second, "sanitized input is not deterministic"

    def test_cast_name_in_line_text_is_sanitized(self):
        """A cast name appearing inside dialogue / line text -- not just
        the CAST roster -- is also replaced before the LLM sees it."""
        led = _mk_ledger(num_lines=2)
        # Inject a line that names a cast member in its body text.
        led["lines"].append({
            "speaker_role": "character",
            "char_id":      "SMITH",
            "text":         "Jones already left through the back door.",
        })
        text = sb._build_reflection_input(led)
        assert "Jones" not in text, (
            "cast name embedded in line text leaked to the LLM unsanitized"
        )
        # The neutral token replaces it.
        assert "character_a" in text

    def test_name_part_is_sanitized(self):
        """A multi-word cast name is replaced whether it appears in full
        or by one of its parts -- both collapse onto the SAME token so
        the input stays internally coherent."""
        led = _mk_ledger(num_lines=2)
        led["cast"] = [
            {"char_id": "DETECTIVE_HALE", "name": "Detective Hale",
             "character_description": "Weary investigator."},
        ]
        led["lines"].append({
            "speaker_role": "character",
            "char_id":      "DETECTIVE_HALE",
            "text":         "Hale studies the rain-streaked glass.",
        })
        text = sb._build_reflection_input(led)
        assert "Hale" not in text, "cast name part 'Hale' leaked unsanitized"
        assert "Detective Hale" not in text
        # Full name and bare part both map to character_a.
        assert "character_a" in text
        assert "character_b" not in text, (
            "the two surface forms of one character produced two tokens"
        )

    def test_proper_noun_replaced_with_source_entity_token(self):
        """A proper noun that is NOT a cast name (an episode title /
        place / source entity) is replaced with a `source_entity_*`
        token before the LLM sees it."""
        led = _mk_ledger(num_lines=2)
        led["meta"]["episode_title"] = "The Bulkhead Closes"
        text = sb._build_reflection_input(led)
        # The title proper-noun phrase does not survive verbatim.
        assert "Bulkhead Closes" not in text
        assert "source_entity" in text, (
            "no source_entity token emitted; the title proper noun was "
            "not sanitized"
        )

    def test_proper_noun_substitution_is_stable_within_a_call(self):
        """The same proper noun seen twice in one build maps to the
        SAME token both times -- cross-references stay coherent.

        The title proper noun "Halloran" appears in the TITLE line and
        again in a scene header. Both occurrences must resolve to the
        identical `source_entity_*` token, so the substituted token
        appears at least twice and "Halloran" never leaks."""
        import re as _re
        led = _mk_ledger(num_lines=2)
        led["meta"]["episode_title"] = "Halloran Tower"
        led["lines"].append({
            "speaker_role": "scene",
            "text":         "=== Halloran Tower -- ROOFTOP",
        })
        # Resolve which token "Halloran" maps to via the deterministic
        # substitution helper, then confirm that exact token carries
        # both occurrences in the built input.
        cast_map = sb._build_cast_substitution(led["cast"])
        probe_map: dict[str, str] = {}
        sb._sanitize_reflection_text("x Halloran x", cast_map, probe_map)
        halloran_token = probe_map["halloran"]

        text = sb._build_reflection_input(led)
        assert "Halloran" not in text, "proper noun leaked unsanitized"
        assert text.count(halloran_token) >= 2, (
            f"the repeated proper noun did not reuse a stable token "
            f"({halloran_token!r} appears {text.count(halloran_token)} "
            "times, expected >= 2)"
        )

    def test_no_real_name_handed_to_llm_via_entrypoint(self):
        """End-to-end: the user message `run_story_brief_reflection`
        actually sends the slot fn contains no real cast name -- the
        sanitized input is what reaches the LLM, not the raw ledger."""
        led = _mk_ledger()
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 1
        sent = spy.calls[0]["messages"][0]["content"]
        assert "Jones" not in sent and "Smith" not in sent, (
            "the LLM call carried a real cast name -- 3G sanitization "
            "did not reach the entrypoint"
        )
        assert "character_a" in sent


# ---------------------------------------------------------------------------
# 4. Prompt body length + 3G suppression-text trim
# ---------------------------------------------------------------------------


def test_reflection_prompt_under_320_tokens():
    """Per refinement section 3 / hard rule 8: reflection prompt body
    stays under 320 tokens. Char/4 is a rough but conservative proxy
    that does not require tiktoken.

    Sprint 8.1 (decision A1, flat additive) bumped the cap from 250 to
    320 when the schema grew from 4 fields to 9 (music_mood_terms,
    visual_palette, key_objects, tempo_hint, atmosphere_line). The
    audit (`downstream_brief_consumer_followup.md`) made these v2
    fields a hard requirement for the six A-class consumers' rewire,
    so the prompt body had to grow with them. 320 is still well below
    linear scaling (4 -> 9 fields linear-scaled = ~540 tokens), so
    the prompt remains tight relative to the schema breadth.
    """
    approx_tokens = len(sb._REFLECTION_PROMPT) // 4
    assert approx_tokens <= 320, (
        f"_REFLECTION_PROMPT is ~{approx_tokens} tokens; cap is 320"
    )


def test_reflection_prompt_dropped_name_suppression_list():
    """Sprint 3G: because the input is pre-sanitized, the prompt no
    longer needs the verbose 'No cast names. No proper nouns.'
    suppression list. That redundant text is gone; a concise positive
    'use no names' instruction replaces it. The dialogue/plot-verb and
    invented-period guidance STAYS -- those constrain words the model
    could volunteer that are not in the sanitized input."""
    prompt = sb._REFLECTION_PROMPT
    # The redundant suppression line is gone.
    assert "No cast names. No proper nouns." not in prompt
    # A positive 'use no names' instruction is present instead.
    assert "no names" in prompt.lower()
    # The guidance that still does real work survives.
    assert "dialogue verbs" in prompt
    assert "plot verbs" in prompt
    assert "invented dates" in prompt


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
# 3G. Output reject-list safety net stays live after input sanitization
# ---------------------------------------------------------------------------
# Sprint 3G sanitizes the INPUT. The schema-side reject lists that catch
# a name in the OUTPUT must STAY exactly as they are -- they are the
# safety net for a model that volunteers a name on its own. These tests
# prove the OUTPUT gate is unweakened by the 3G change.


class TestOutputRejectSafetyNetUnchanged:

    def test_output_validator_still_rejects_named_character(self):
        """`_validate_brief` -- the OUTPUT gate -- still flags a cast
        name in the produced brief, unchanged by 3G input sanitization."""
        led = _mk_ledger()
        leaked = "a dim room where Jones leans over the table"
        reasons = sb._validate_brief(leaked, led)
        assert sb.REJECT_NAMED_CHARACTER in reasons

    def test_entrypoint_rerolls_when_output_leaks_a_name(self):
        """End-to-end: even with a sanitized input, if the LLM still
        emits a cast name in its OUTPUT the content gate fires and the
        ladder re-rolls. The reject-list safety net is intact."""
        led = _mk_ledger()
        name_leaking = (
            '{"story_brief": "a dim room where Jones leans over a table", '
            '"setting_terms": ["room", "table"], '
            '"lighting_terms": ["bare bulb"], '
            '"atmosphere_terms": ["tense"]}'
        )
        spy = _make_spy_fn(responses=[name_leaking, _valid_brief_json()])
        result = sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 2, (
            "output name-leak did not trigger a re-roll -- the OUTPUT "
            "reject-list safety net regressed"
        )
        assert result["story_brief_status"] == "ok"
        assert "Jones" not in result["story_brief"]


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


# ---------------------------------------------------------------------------
# Sprint 8.1 v2 producer fields (decision A1, flat additive)
# ---------------------------------------------------------------------------
# The v2 fields land alongside the v1 8-key meta contract: music_mood_terms,
# visual_palette, key_objects, tempo_hint, atmosphere_line. All five carry
# safe defaults on the schema so a v1-era LLM response (missing the new
# keys) still validates. The success delta + failure sentinel stamp the
# fields at the top level (A1 flat additive) so downstream readers reach
# them via `_read_brief_field(meta, "<field>", default=...)`.


_V2_FIELD_KEYS = {
    "music_mood_terms",
    "visual_palette",
    "key_objects",
    "tempo_hint",
    "atmosphere_line",
}


def _valid_brief_json_v2() -> str:
    """A schema-valid v2 LLM response carrying every v2 field with
    non-empty values -- the happy path for downstream consumers."""
    return (
        '{"story_brief": "a dim interrogation room under a swinging bare '
        'bulb, rain-streaked window, sweat and smoke", '
        '"setting_terms": ["interrogation room", "steel table"], '
        '"lighting_terms": ["swinging bare bulb"], '
        '"atmosphere_terms": ["sweat", "smoke", "tense"], '
        '"music_mood_terms": ["tense", "sombre", "uneasy"], '
        '"visual_palette": ["amber glow", "smoke-grey", "wet asphalt"], '
        '"key_objects": ["steel table", "bare bulb", "ashtray"], '
        '"tempo_hint": "slow", '
        '"atmosphere_line": "smoke and sweat under a swinging bare bulb."}'
    )


class TestV2ProducerFields:

    def test_prompt_version_bumped_to_v2(self):
        """A consumer keying on prompt_version (or grepping for the
        v2 contract in saved ledgers) must see `v2`."""
        assert sb._PROMPT_VERSION == "v2"

    def test_success_delta_stamps_v2_fields(self):
        """Every v2 field lands at the top level of the meta delta
        (decision A1, flat additive) with the values the LLM emitted."""
        led = _mk_ledger()
        spy = _make_spy_fn(responses=[_valid_brief_json_v2()])
        result = sb.run_story_brief_reflection(
            led, spy, technical_model_id="some/model-id",
        )
        assert result["story_brief_status"] == "ok"
        assert _V2_FIELD_KEYS.issubset(result.keys()), (
            f"v2 fields missing from success delta; have: "
            f"{sorted(result.keys())}"
        )
        assert result["music_mood_terms"] == ["tense", "sombre", "uneasy"]
        assert result["visual_palette"] == [
            "amber glow", "smoke-grey", "wet asphalt",
        ]
        assert result["key_objects"] == [
            "steel table", "bare bulb", "ashtray",
        ]
        assert result["tempo_hint"] == "slow"
        assert result["atmosphere_line"] == (
            "smoke and sweat under a swinging bare bulb."
        )
        # v2 is purely additive: the v1 8-key contract still holds.
        assert _REQUIRED_META_KEYS.issubset(result.keys())
        assert result["story_brief_prompt_version"] == "v2"

    def test_success_delta_v1_response_fills_v2_defaults(self):
        """A v1-era LLM response (4-field JSON without the v2 keys)
        still validates -- the v2 fields fall through to safe-empty
        defaults so downstream readers gracefully drop to v1 paths."""
        led = _mk_ledger()
        spy = _make_spy_fn(responses=[_valid_brief_json()])  # v1 shape
        result = sb.run_story_brief_reflection(
            led, spy, technical_model_id="some/model-id",
        )
        assert result["story_brief_status"] == "ok"
        assert _V2_FIELD_KEYS.issubset(result.keys())
        assert result["music_mood_terms"] == []
        assert result["visual_palette"] == []
        assert result["key_objects"] == []
        assert result["tempo_hint"] == ""
        assert result["atmosphere_line"] == ""

    def test_failure_sentinel_stamps_v2_safe_defaults(self):
        """On a slot-fn raise (technical_fn_exception path) the
        sentinel must carry the v2 fields with safe-empty defaults
        so downstream readers don't see KeyError or None."""
        led = _mk_ledger()

        def raising_fn(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated technical_fn failure")

        result = sb.run_story_brief_reflection(led, raising_fn)
        assert result["story_brief_status"] == "failed"
        assert _V2_FIELD_KEYS.issubset(result.keys()), (
            f"v2 fields missing from failure sentinel; have: "
            f"{sorted(result.keys())}"
        )
        assert result["music_mood_terms"] == []
        assert result["visual_palette"] == []
        assert result["key_objects"] == []
        assert result["tempo_hint"] == ""
        assert result["atmosphere_line"] == ""

    def test_failure_sentinel_on_parse_failure_stamps_v2_defaults(self):
        """JSON parse failure path: same safe-empty v2 defaults."""
        led = _mk_ledger()
        spy = _make_spy_fn(responses=["this is not JSON {{{"])
        result = sb.run_story_brief_reflection(led, spy)
        assert result["story_brief_status"] == "failed"
        assert _V2_FIELD_KEYS.issubset(result.keys())
        for key in _V2_FIELD_KEYS:
            value = result[key]
            # Lists fall to []; strings fall to "".
            assert value in ([], ""), (
                f"v2 field {key!r} stamped as {value!r}; expected "
                "safe-empty default ([] or '')"
            )

    def test_schema_caps_string_v2_fields(self):
        """Schema-side length caps on the two string v2 fields fire
        when the LLM produces over-long content. Locks the producer-
        side defence so a runaway response cannot smuggle a paragraph
        into the LTX motion prompt via tempo_hint."""
        # tempo_hint cap is _TEMPO_HINT_HARD_MAX_CHARS (80).
        oversized_tempo = "x" * (sb._TEMPO_HINT_HARD_MAX_CHARS + 5)
        # atmosphere_line cap is _ATMOSPHERE_LINE_HARD_MAX_CHARS (200).
        oversized_atmo = "y" * (sb._ATMOSPHERE_LINE_HARD_MAX_CHARS + 5)
        bad_json = (
            '{"story_brief": "a dim interrogation room under a swinging '
            'bare bulb, rain-streaked window, sweat and smoke", '
            '"setting_terms": ["room"], "lighting_terms": ["bulb"], '
            '"atmosphere_terms": ["tense"], '
            '"music_mood_terms": [], "visual_palette": [], '
            '"key_objects": [], '
            f'"tempo_hint": "{oversized_tempo}", '
            f'"atmosphere_line": "{oversized_atmo}"}}'
        )
        spy = _make_spy_fn(
            responses=[bad_json, _valid_brief_json_v2()],
        )
        result = sb.run_story_brief_reflection(led=_mk_ledger(), technical_fn=spy)
        # structured_call now CLAMPS an over-long capped field to its schema
        # max_length IN PLACE (commit 4c0e943, the "no loud fail" graceful-
        # degradation contract) rather than rejecting the response and re-rolling.
        # So a single call recovers ("ok") with the OVER-LONG values truncated to
        # the cap -- the important contract (over-long never reaches the meta
        # delta) holds, now via coercion instead of a structural retry.
        assert result["story_brief_status"] == "ok"
        assert len(result["tempo_hint"]) == sb._TEMPO_HINT_HARD_MAX_CHARS
        assert (
            len(result["atmosphere_line"])
            == sb._ATMOSPHERE_LINE_HARD_MAX_CHARS
        )
        # The clamp fixes the response in place -> NO structural retry needed.
        assert len(spy.calls) == 1, (
            "over-long capped fields should be coerced in place, not re-rolled"
        )

    def test_prompt_lists_all_nine_field_names(self):
        """The reflection prompt body explicitly names every v1 + v2
        field so the LLM has the full schema in-context (no hidden
        contract). Mechanical grep -- catches a future trim that
        accidentally drops a field name from the prompt body."""
        prompt = sb._REFLECTION_PROMPT
        for field in (
            "story_brief", "setting_terms", "lighting_terms",
            "atmosphere_terms",
            "music_mood_terms", "visual_palette", "key_objects",
            "tempo_hint", "atmosphere_line",
        ):
            assert field in prompt, (
                f"_REFLECTION_PROMPT does not mention field {field!r}"
            )


# ---------------------------------------------------------------------------
# PBUG-20260721-08: generic cast roles are not personal-name leaks
# ---------------------------------------------------------------------------


class TestCastRoleIdentityProjection:

    @pytest.mark.parametrize(
        ("name", "brief"),
        [
            ("THE TRAVELER", "a weary traveler beneath cold laboratory light"),
            ("THE WITNESSES", "skeptical witnesses in a storm-dark chamber"),
            ("THE TIME TRAVELER", "a time traveler beside a brass machine"),
            ("THE PSYCHOLOGIST", "a psychologist under a flickering lamp"),
            ("THE SCIENTIST", "a scientist amid ruined instruments"),
            ("First Witch", "a witch on a rain-dark heath"),
            ("ANNOUNCER", "an announcer booth under an amber dial glow"),
            ("WITNESS", "a witness beside a shadowed table"),
        ],
    )
    def test_generic_role_labels_are_legal_visual_nouns(self, name, brief):
        led = {"cast": [{"char_id": "c01", "name": name}], "lines": []}
        assert sb._cast_output_forbidden_forms(name) == set()
        assert sb.REJECT_NAMED_CHARACTER not in sb._validate_brief(brief, led)

    @pytest.mark.parametrize(
        ("bank", "name", "leaked_surface"),
        [
            ("media_archive", "JOEL PIERCE", "Pierce"),
            ("original", "BOB FLANDERS", "Flanders"),
            ("public_domain", "FILBY", "Filby"),
            ("shakespeare", "MACBETH", "Macbeth"),
            ("scifi_news", "Dr. Aris Thorne", "Thorne"),
            ("scifi_news_pro", "LARKIN", "Larkin"),
        ],
    )
    def test_all_six_bank_personal_name_shapes_stay_forbidden(
        self, bank, name, leaked_surface,
    ):
        led = {
            "cast": [{"char_id": "c01", "name": name}],
            "lines": [],
            "meta": {"source_bank": bank},
        }
        leaked = f"a lamp-lit room where {leaked_surface} waits by the console"
        assert sb._validate_brief(leaked, led) == [sb.REJECT_NAMED_CHARACTER]
        assert sb._validate_brief(
            "a lamp-lit room with rain on the windows", led,
        ) == []

    def test_role_input_forms_preserve_one_identity_without_mapping_articles(self):
        cast = [
            {"char_id": "announcer", "name": "ANNOUNCER"},
            {"char_id": "traveler", "name": "THE TRAVELER"},
            {"char_id": "witnesses", "name": "THE WITNESSES"},
        ]
        mapping = sb._build_cast_substitution(cast)
        assert mapping["the traveler"] == "character_b"
        assert mapping["traveler"] == "character_b"
        assert mapping["the witnesses"] == "character_c"
        assert mapping["witnesses"] == "character_c"
        assert "the" not in mapping

        led = {
            "cast": cast,
            "lines": [{
                "speaker_role": "character",
                "char_id": "traveler",
                "text": "The weary Traveler enters as the Witnesses wait.",
            }],
            "meta": {"episode_title": "", "style": ""},
        }
        text = sb._build_reflection_input(led)
        assert "The weary character_b" in text
        assert "character_c wait" in text
        assert "source_entity" not in text

    def test_ordinal_role_maps_full_label_and_role_noun_to_same_identity(self):
        mapping = sb._build_cast_substitution([
            {"char_id": "witch", "name": "First Witch"},
        ])
        assert mapping["first witch"] == "character_a"
        assert mapping["witch"] == "character_a"
        assert "first" not in mapping
        assert sb._cast_output_forbidden_forms("First Witch") == set()

    def test_titles_do_not_make_generic_words_forbidden_or_drop_real_names(self):
        led = {
            "cast": [{"char_id": "c01", "name": "Doctor Aris Thorne"}],
            "lines": [],
        }
        forms = sb._cast_output_forbidden_forms("Doctor Aris Thorne")
        assert "doctor aris thorne" in forms
        assert "aris" in forms and "thorne" in forms
        assert "doctor" not in forms
        assert sb._validate_brief(
            "a doctor beside a cold instrument panel", led,
        ) == []
        assert sb._validate_brief(
            "a cold instrument panel beside Thorne", led,
        ) == [sb.REJECT_NAMED_CHARACTER]

    def test_hyphen_apostrophe_and_common_word_mononym_remain_protected(self):
        assert {"jean-luc picard", "jean", "luc", "picard"}.issubset(
            sb._cast_output_forbidden_forms("Jean-Luc Picard")
        )
        assert {"o'connor", "connor"}.issubset(
            sb._cast_output_forbidden_forms("O'Connor")
        )
        # Shakespeare's Bottom is a real proper mononym despite also being an
        # ordinary English word; dictionary-word heuristics must not erase it.
        assert sb._cast_output_forbidden_forms("Bottom") == {"bottom"}

    def test_article_does_not_hide_a_non_role_personal_name(self):
        forms = sb._cast_output_forbidden_forms("The Count of Monte Cristo")
        assert "the count of monte cristo" in forms
        assert "monte" in forms and "cristo" in forms

    def test_live_role_brief_succeeds_without_spending_a_repair_turn(self):
        led = {
            "cast": [
                {"char_id": "c01", "name": "ANNOUNCER"},
                {"char_id": "c02", "name": "THE TRAVELER"},
                {"char_id": "c03", "name": "THE WITNESSES"},
            ],
            "lines": [],
            "meta": {},
        }
        response = (
            '{"story_brief": "A weary traveler faces a skeptical assembly '
            'over the wreckage of a collapsed world", '
            '"setting_terms": ["smoldering ruins"], '
            '"lighting_terms": ["dying embers"], '
            '"atmosphere_terms": ["somber"]}'
        )
        spy = _make_spy_fn(responses=[response])
        result = sb.run_story_brief_reflection(led, spy)
        assert result["story_brief_status"] == "ok"
        assert len(spy.calls) == 1

    def test_private_repair_detail_names_surface_but_public_code_stays_stable(self):
        led = _mk_ledger()
        leaked = (
            '{"story_brief": "a dim room where Jones waits by a table", '
            '"setting_terms": ["room"], "lighting_terms": ["dim"], '
            '"atmosphere_terms": ["tense"]}'
        )
        assert sb._validate_brief("a dim room beside Jones", led) == [
            sb.REJECT_NAMED_CHARACTER,
        ]
        spy = _make_spy_fn(responses=[leaked, _valid_brief_json()])
        result = sb.run_story_brief_reflection(led, spy)
        assert result["story_brief_status"] == "ok"
        repair_message = spy.calls[1]["messages"][0]["content"]
        assert 'blocked personal-name surface is "jones"' in repair_message
        assert "PostValidationError" not in repair_message
        assert "environment, light, color, texture" in repair_message
