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


def test_schema_repair_arm_uses_clamped_repair_temperature():
    """The second `technical_fn` call (the repair pass) must run at the
    clamped repair temperature, not the base reflection temperature.

    base 0.30 + bump 0.15 = 0.45 (under the 0.55 ceiling). This is
    extra proof the call really went through `_repair_pass` rather than
    being a stray re-entry of Block 1."""
    led = _mk_ledger()
    spy = _make_spy_fn(responses=[_schema_invalid_json(), _valid_brief_json()])

    sb.run_story_brief_reflection(led, spy)

    assert len(spy.calls) >= 2, "repair pass did not run"
    # Call #1 is the base reflection pass.
    assert abs(spy.calls[0]["temperature"]
               - sb._REFLECTION_TEMPERATURE) < 1e-6
    # Call #2 is the repair pass: base + bump, clamped at the ceiling.
    expected_repair_temp = min(
        sb._REFLECTION_TEMPERATURE + sb._REPAIR_TEMPERATURE_BUMP,
        sb._REPAIR_TEMPERATURE_CEILING,
    )
    assert abs(spy.calls[1]["temperature"] - expected_repair_temp) < 1e-6, (
        f"repair-pass temperature {spy.calls[1]['temperature']} does not "
        f"match the clamped repair temperature {expected_repair_temp}"
    )


def test_schema_repair_pass_invoked_exactly_once(monkeypatch):
    """Direct spy on `_repair_pass`: a schema-invalid first response
    must invoke `_repair_pass` exactly once.

    Pre-fix the call site dies on `NameError: name 'json_str' is not
    defined` while evaluating the keyword arguments -- so `_repair_pass`
    is never entered and this spy records zero calls. Post-fix it is
    invoked once."""
    led = _mk_ledger()
    spy = _make_spy_fn(responses=[_schema_invalid_json(), _valid_brief_json()])

    repair_calls: list[dict] = []
    real_repair_pass = sb._repair_pass

    def spy_repair_pass(*args, **kwargs):
        repair_calls.append({"args": args, "kwargs": kwargs})
        return real_repair_pass(*args, **kwargs)

    monkeypatch.setattr(sb, "_repair_pass", spy_repair_pass)

    result = sb.run_story_brief_reflection(led, spy)

    assert len(repair_calls) == 1, (
        f"_repair_pass was invoked {len(repair_calls)} time(s), "
        "expected exactly 1. Pre-fix the schema-repair call site raises "
        "NameError on `json_str` before _repair_pass is entered."
    )
    # The repair pass must have received the raw LLM output as
    # `failed_output` -- the whole point of the json_str -> raw fix.
    call = repair_calls[0]
    failed_output = call["kwargs"].get("failed_output")
    if failed_output is None and call["args"]:
        failed_output = call["args"][0]
    assert failed_output == _schema_invalid_json(), (
        "_repair_pass should receive the raw schema-invalid LLM output "
        "as `failed_output` (the json_str -> raw fix)"
    )
    assert result["story_brief_status"] == "ok"
