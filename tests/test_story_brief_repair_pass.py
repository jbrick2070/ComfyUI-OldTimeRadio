"""BUG-LOCAL-268 regression: schema-repair arm must actually execute.

`run_story_brief_reflection` (nodes/_otr_story_brief.py) has a
schema-validation repair arm: when the LLM returns valid JSON that
fails `StoryBriefModel` shape validation, the function is supposed to
run ONE repair pass via `_repair_pass` at the clamped repair
temperature before falling through to the failure sentinel.

The bug: the repair-arm call site passed `failed_output=json_str`, but
`json_str` is never bound anywhere in the function (the raw LLM output
lives in the variable `raw`). Evaluating `failed_output=json_str`
raised `NameError` BEFORE `_repair_pass` was entered. That `NameError`
was swallowed by the broad `except (Exception, ValidationError)` arm a
few lines below, so every schema-repair attempt silently collapsed to
the `_failure_sentinel` and the repair pass never ran.

The fix changes `failed_output=json_str` to `failed_output=raw`.

These tests PROVE the schema-repair arm executes:
  * On the buggy pre-fix code `technical_fn` is called exactly once,
    `_repair_pass` is never entered, and the result is the failure
    sentinel (`story_brief_status == "failed"`).
  * On the fixed code `technical_fn` is called twice (the second call
    IS the repair pass), `_repair_pass` runs exactly once, and the
    function returns a successful brief delta.

The module is pure (no I/O, no GPU, no ComfyUI imports), so every test
runs against a plain dict ledger and a call-counting `technical_fn`
stand-in -- matching the fixture style of `tests/test_story_brief_c5a1.py`.

Sprint 2A/2D update: `run_story_brief_reflection` was converted onto the
shared `structured_call` retry ladder; the hand-rolled `_repair_pass`
was removed. The BUG-LOCAL-268 concern -- a schema-invalid first
response MUST trigger a recovering retry -- is unchanged and is now
served by the ladder's Attempt 2 (structural retry). The two tests
below that previously asserted the old raised repair temperature and
spied the removed `_repair_pass` were rewritten to assert the ladder
contract: the retry runs, it lowers temperature, and it goes through
`structured_call`.
"""

from __future__ import annotations

from nodes import _otr_story_brief as sb


# ---------------------------------------------------------------------------
# Fixtures -- ledger + technical_fn stand-in
# ---------------------------------------------------------------------------


def _mk_ledger() -> dict:
    """Construct a minimal valid ledger dict.

    Mirrors `_mk_ledger` in tests/test_story_brief_c5a1.py: a Jones /
    Smith two-hander so the content validator has real cast-name
    tokens to check the repaired brief against.
    """
    lines: list[dict] = [
        {"speaker_role": "scene", "text": "=== SCENE 1 -- INTERIOR -- NIGHT"},
        {"speaker_role": "env",
         "text": "[ENV: light rain on tin roof, distant thunder]"},
    ]
    for i in range(6):
        lines.append({
            "speaker_role": "character",
            "char_id":      "JONES" if i % 2 == 0 else "SMITH",
            "text":         f"Line {i} of dialogue here, kept short for the fixture.",
        })
    lines.append({"speaker_role": "sfx",
                  "text": "[SFX: bulkhead seals with a hiss]"})
    return {
        "cast": [
            {"char_id": "JONES", "name": "Jones",
             "character_description": "Tall detective."},
            {"char_id": "SMITH", "name": "Smith",
             "character_description": "Quiet suspect, sweating."},
        ],
        "lines": lines,
        "meta":  {"episode_title": "The Bulkhead Closes",
                  "style":         "noir_interrogation"},
    }


def _make_spy_fn(*, responses: list[str]):
    """Return a call-counting `technical_fn` stand-in.

    Records (messages, temperature, max_new_tokens) on every call and
    pops the next response from `responses`. Exposes `.calls` so a test
    can assert how many times the LLM slot was invoked -- the second
    invocation IS the repair pass.
    """
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


def _schema_invalid_json() -> str:
    """Valid JSON that FAILS StoryBriefModel schema validation.

    The JSON parses cleanly (so Block 2's json parse passes), but
    `story_brief` is below the model's `min_length=10` floor, so
    `StoryBriefModel.model_validate` raises `ValidationError` -- which
    is exactly what drives the entrypoint into the schema-repair arm.
    """
    return (
        '{"story_brief": "too dim", '
        '"setting_terms": ["room"], '
        '"lighting_terms": ["bulb"], '
        '"atmosphere_terms": ["tense"]}'
    )


def _valid_brief_json() -> str:
    """A payload that passes StoryBriefModel schema AND _validate_brief.

    No cast names (Jones / Smith), no dialogue or plot verbs, no quote
    or markup characters, no period literals, under 300 chars -- so the
    repaired brief sails through the content gate too and the
    entrypoint returns a successful delta.
    """
    return (
        '{"story_brief": "a dim interrogation room under a swinging bare '
        'bulb, rain-streaked window, sweat and smoke", '
        '"setting_terms": ["interrogation room", "steel table", '
        '"rain-streaked window"], '
        '"lighting_terms": ["swinging bare bulb", "harsh top-down shadow"], '
        '"atmosphere_terms": ["sweat", "smoke", "tense"]}'
    )


# ---------------------------------------------------------------------------
# Pre-flight: confirm the fixtures genuinely exercise the schema arm
# ---------------------------------------------------------------------------


def test_invalid_payload_parses_as_json_but_fails_schema():
    """Lock the fixture contract: `_schema_invalid_json` must parse as
    JSON (so Block 2 passes) but fail `StoryBriefModel` validation (so
    Block 3 raises and enters the repair arm). If this ever stops being
    true, the repair-arm regression below would pass for the wrong
    reason."""
    data = sb._otr_json.parse_first_json_object(_schema_invalid_json())
    assert isinstance(data, dict), "invalid payload must still parse as JSON"

    from pydantic import ValidationError

    raised = False
    try:
        sb.StoryBriefModel.model_validate(data)
    except ValidationError:
        raised = True
    assert raised, (
        "_schema_invalid_json must fail StoryBriefModel validation "
        "(story_brief is below min_length=10) so it drives the "
        "schema-repair arm"
    )


# ---------------------------------------------------------------------------
# BUG-LOCAL-268: the schema-repair arm must actually run
# ---------------------------------------------------------------------------


def test_schema_repair_arm_invokes_technical_fn_a_second_time():
    """BUG-LOCAL-268: a schema-invalid first response must trigger the
    repair pass -- which calls `technical_fn` a SECOND time.

    Call #1 returns valid JSON that fails StoryBriefModel schema
    validation; call #2 (the repair pass) returns a valid brief.

    Pre-fix: `failed_output=json_str` raises NameError before
    `_repair_pass` is entered; the broad except swallows it and
    `technical_fn` is only ever called once -> this assertion fails.
    Post-fix: `failed_output=raw` is bound, `_repair_pass` runs, and
    `technical_fn` is called twice."""
    led = _mk_ledger()
    spy = _make_spy_fn(responses=[_schema_invalid_json(), _valid_brief_json()])

    result = sb.run_story_brief_reflection(led, spy)

    assert len(spy.calls) >= 2, (
        "schema-repair arm never ran: technical_fn was called "
        f"{len(spy.calls)} time(s), expected 2 (the second call IS the "
        "repair pass). On the pre-fix code `failed_output=json_str` "
        "raises NameError before _repair_pass is entered."
    )
    assert result["story_brief_status"] != "failed", (
        "schema-repair arm collapsed to the failure sentinel; the "
        "repair pass should have produced a valid brief"
    )
    assert result["story_brief"] != "", (
        "repaired brief is empty -- the repair pass did not feed a "
        "valid StoryBriefModel payload back through the gate"
    )
    assert result["story_brief_status"] == "ok"


def test_schema_retry_lowers_temperature():
    """The second `technical_fn` call (the structural retry) must run
    at a temperature STRICTLY BELOW the base reflection temperature.

    Sprint 2A/2D: `run_story_brief_reflection` routes through the shared
    `structured_call` ladder. Per the Sprint 2B principle a JSON-schema
    re-roll LOWERS entropy -- the old hand-rolled repair pass raised it
    by +0.15, which this conversion corrects. This also proves the
    second call really is the ladder's Attempt 2, not a stray
    re-entry."""
    led = _mk_ledger()
    # C3: the structural retry runs ONLY on a JSON-SYNTAX failure (a schema
    # error skips straight to the typed repair), so the first response must be
    # unparseable to exercise the structural rung at its lowered temperature.
    spy = _make_spy_fn(responses=["this is not valid JSON at all {{{", _valid_brief_json()])

    sb.run_story_brief_reflection(led, spy)

    assert len(spy.calls) >= 2, "structural retry did not run"
    # Call #1 is the base reflection pass.
    assert abs(spy.calls[0]["temperature"]
               - sb._REFLECTION_TEMPERATURE) < 1e-6
    # Call #2 is the structural retry -- below base, never above.
    assert spy.calls[1]["temperature"] < spy.calls[0]["temperature"], (
        f"structural-retry temperature {spy.calls[1]['temperature']} "
        f"must be below the base temperature {spy.calls[0]['temperature']}"
    )
    assert abs(spy.calls[1]["temperature"]
               - sb._REFLECTION_STRUCTURAL_RETRY_TEMPERATURE) < 1e-6, (
        f"structural-retry temperature {spy.calls[1]['temperature']} does "
        f"not match _REFLECTION_STRUCTURAL_RETRY_TEMPERATURE "
        f"({sb._REFLECTION_STRUCTURAL_RETRY_TEMPERATURE})"
    )


def test_schema_failure_routes_through_structured_call(monkeypatch):
    """A schema-invalid first response must be handled by the shared
    `structured_call` ladder -- the entrypoint delegates the call +
    retry to it rather than hand-rolling a repair pass.

    Sprint 2A/2D: the former direct `_repair_pass` spy was retired with
    the hand-rolled repair pass; the ladder is now the single retry
    surface. This spy confirms `structured_call` is on the path and
    receives the `StoryBriefModel` schema + the technical slot fn."""
    led = _mk_ledger()
    spy = _make_spy_fn(responses=[_schema_invalid_json(), _valid_brief_json()])

    structured_calls: list[dict] = []
    real_structured_call = sb.structured_call

    def spy_structured_call(*args, **kwargs):
        structured_calls.append({"args": args, "kwargs": kwargs})
        return real_structured_call(*args, **kwargs)

    monkeypatch.setattr(sb, "structured_call", spy_structured_call)

    result = sb.run_story_brief_reflection(led, spy)

    assert len(structured_calls) == 1, (
        f"structured_call was invoked {len(structured_calls)} time(s), "
        "expected exactly 1 -- the entrypoint must delegate the call + "
        "retry to the shared ladder"
    )
    kwargs = structured_calls[0]["kwargs"]
    assert kwargs.get("schema") is sb.StoryBriefModel, (
        "structured_call must receive the StoryBriefModel schema"
    )
    assert kwargs.get("slot_fn") is spy, (
        "structured_call must receive the technical slot fn"
    )
    assert result["story_brief_status"] == "ok"
