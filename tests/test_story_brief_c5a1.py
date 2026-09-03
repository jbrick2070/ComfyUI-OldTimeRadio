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

  Schema-only acceptance:
    5. preserves authored names, actions, eras, and punctuation
    6. accepts any nonblank brief without a content-driven retry

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
import json
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

    `with_period` optionally enriches a cast description for input-builder tests.
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


def _brief_json(story_brief: str) -> str:
    return json.dumps({
        "story_brief": story_brief,
        "setting_terms": ["interrogation room"],
        "lighting_terms": ["bare bulb"],
        "atmosphere_terms": ["tense"],
    })


def _valid_brief_json() -> str:
    return _brief_json(
        "a dim interrogation room under a swinging bare bulb, "
        "rain-streaked window, sweat and smoke"
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

    def test_includes_exact_required_fields(self):
        """The model receives the canonical names and world terms unchanged."""
        led = _mk_ledger()
        text = sb._build_reflection_input(led)
        assert "TITLE: The Bulkhead Closes" in text
        assert "STYLE: noir_interrogation" in text
        assert "CAST:" in text
        assert "Jones" in text
        assert "Smith" in text
        assert "SCENE 1" in text
        assert "OPENING" in text
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
# 3G. Exact context preservation
# ---------------------------------------------------------------------------


class TestReflectionInputPreservation:

    def test_names_places_objects_and_smoking_are_preserved(self):
        led = _mk_ledger(num_lines=2)
        led["meta"]["episode_title"] = "Halloran Tower"
        led["lines"].append({
            "speaker_role": "character",
            "char_id": "SMITH",
            "text": (
                "Jones waits in the smoking room beside a velvet chair "
                "and the decryption machine."
            ),
        })
        text = sb._build_reflection_input(led)
        assert "Halloran Tower" in text
        assert "Jones waits in the smoking room" in text
        assert "velvet chair" in text
        assert "decryption machine" in text

    def test_entrypoint_sends_exact_cast_and_world_context(self):
        led = _mk_ledger()
        spy = _make_spy_fn(responses=[_valid_brief_json()])
        sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 1
        sent = spy.calls[0]["messages"][0]["content"]
        assert "Jones" in sent
        assert "Smith" in sent


# ---------------------------------------------------------------------------
# 4. Prompt body length + exact-context guidance
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


def test_reflection_prompt_requests_exact_context_preservation():
    prompt = sb._REFLECTION_PROMPT
    assert "Preserve exact context names and world terms" in prompt
    assert "dialogue verbs" in prompt
    assert "plot verbs" in prompt
    assert "invented dates" in prompt


def test_reflection_prompt_says_what_a_setting_term_is():
    """`setting_terms` was the one list field the schema named and left alone.

    Every other list carries its own definition -- key_objects are "concrete
    nouns the scene contains", visual_palette is "colors / textures" -- and
    `setting_terms` got only its own name back, so the model filled the gap with
    whatever the episode felt like. Measured 2026-09-03 across the 1,955
    episodes on disk: 273 carried a setting term that is not a place, and the
    ghost composer joined them as "coffee cups in the sterile" and
    "archive_reels in the concrete_floors". The render-time normaliser in
    `ghost_signal_author._spoken_term` repairs the punctuation on episodes
    already frozen; this rule is the root fix, and it is why new episodes stop
    producing them.
    """
    for prompt in (sb._REFLECTION_PROMPT, sb._DYNAMIC_REFLECTION_PROMPT):
        assert "setting_terms are PLACES" in prompt
        assert "never underscores" in prompt


# ---------------------------------------------------------------------------
# 5-8. Authored content is preserved; only schema integrity can retry
# ---------------------------------------------------------------------------


class TestAuthoredBriefAcceptance:

    def test_names_actions_eras_and_punctuation_preserved_without_retry(self):
        led = _mk_ledger()
        authored = (
            'Jones speaks in a 1940s "interrogation" room, then discovers '
            "the Victorian machine"
        )
        spy = _make_spy_fn(responses=[_brief_json(authored)])
        result = sb.run_story_brief_reflection(led, spy)

        assert result["story_brief_status"] == "ok"
        assert result["story_brief"] == authored
        assert len(spy.calls) == 1

    def test_any_nonblank_schema_valid_brief_is_accepted_once(self):
        spy = _make_spy_fn(responses=[_brief_json("x")])
        result = sb.run_story_brief_reflection(_mk_ledger(), spy)

        assert result["story_brief_status"] == "ok"
        assert result["story_brief"] == "x"
        assert len(spy.calls) == 1


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

    def test_retry_runs_when_initial_json_is_structurally_invalid(self):
        """Only a malformed structured response spends a retry turn."""
        led = _mk_ledger()
        spy = _make_spy_fn(
            responses=["this is not valid JSON at all {{{", _valid_brief_json()]
        )
        result = sb.run_story_brief_reflection(led, spy)
        assert len(spy.calls) == 2
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
        spy = _make_spy_fn(
            responses=["this is not valid JSON at all {{{", _valid_brief_json()]
        )
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

    def test_schema_preserves_long_v2_string_fields(self):
        oversized_tempo = "x" * 500
        oversized_atmo = "y" * 900
        response = (
            '{"story_brief": "a dim interrogation room under a swinging '
            'bare bulb, rain-streaked window, sweat and smoke", '
            '"setting_terms": ["room"], "lighting_terms": ["bulb"], '
            '"atmosphere_terms": ["tense"], '
            '"music_mood_terms": [], "visual_palette": [], '
            '"key_objects": [], '
            f'"tempo_hint": "{oversized_tempo}", '
            f'"atmosphere_line": "{oversized_atmo}"}}'
        )
        spy = _make_spy_fn(responses=[response])
        result = sb.run_story_brief_reflection(led=_mk_ledger(), technical_fn=spy)
        assert result["story_brief_status"] == "ok"
        assert result["tempo_hint"] == oversized_tempo
        assert result["atmosphere_line"] == oversized_atmo
        assert len(spy.calls) == 1

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
# Six-bank exact-context and authored-output preservation
# ---------------------------------------------------------------------------


class TestSixBankAuthoredVocabulary:

    @pytest.mark.parametrize(
        ("bank", "name", "authored"),
        [
            ("media_archive", "JOEL PIERCE", "Pierce speaks beside the reel"),
            ("original", "BOB FLANDERS", "Flanders discovers a velvet chair"),
            ("public_domain", "FILBY", "Filby waits beside the machine"),
            ("shakespeare", "MACBETH", "Macbeth watches the smoking heath"),
            ("scifi_news_pro", "Dr. Aris Thorne", "Thorne studies a 1940s console"),
            ("scifi_news_pro", "LARKIN", "Larkin argues under Victorian lamps"),
        ],
    )
    def test_all_six_banks_preserve_context_and_accept_authored_vocabulary_once(
        self, bank, name, authored,
    ):
        led = {
            "cast": [{"char_id": "c01", "name": name}],
            "lines": [{
                "speaker_role": "character",
                "char_id": "c01",
                "text": authored,
            }],
            "meta": {"source_bank": bank, "episode_title": authored},
        }
        reflection_input = sb._build_reflection_input(led)
        assert name in reflection_input
        assert authored in reflection_input

        spy = _make_spy_fn(responses=[_brief_json(authored)])
        result = sb.run_story_brief_reflection(led, spy)
        assert result["story_brief_status"] == "ok"
        assert result["story_brief"] == authored
        assert len(spy.calls) == 1
