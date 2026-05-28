"""Sprint 4.3 (2026-05-28) -- Stage 1 fan-out helper.

Pinned invariants:
  * Three canonical diversity knobs (moral_dilemma /
    bureaucratic_absurd / intimate_personal_cost) with
    non-empty prompt prefixes.
  * fan_out_stage1_plans runs N calls in knob order.
  * Each knob's system message is the knob prefix + base prompt.
  * generate_fn / parse_fn failures stay isolated to the failing
    candidate; the rest still run.
  * Output structure is List[FanOutResult] with .ok property.

The actual end-to-end fan-out + selector integration is a separate
sprint that touches the writer's outline call. This file pins the
helper surface so the integration sprint can call it with confidence.
"""
from __future__ import annotations

from nodes._otr_stage1_fanout import (
    ALL_DIVERSITY_KNOBS,
    DIVERSITY_KNOB_PROMPTS,
    DiversityKnob,
    FanOutResult,
    fan_out_stage1_plans,
)


# ---------------------------------------------------------------------------
# Surface invariants
# ---------------------------------------------------------------------------


def test_three_canonical_knobs():
    assert ALL_DIVERSITY_KNOBS == (
        DiversityKnob.MORAL_DILEMMA,
        DiversityKnob.BUREAUCRATIC_ABSURD,
        DiversityKnob.INTIMATE_PERSONAL_COST,
    )
    assert len(ALL_DIVERSITY_KNOBS) == 3


def test_each_knob_has_a_non_empty_prompt_prefix():
    for knob in ALL_DIVERSITY_KNOBS:
        prefix = DIVERSITY_KNOB_PROMPTS[knob]
        assert isinstance(prefix, str)
        assert len(prefix) > 50, (
            f"knob {knob!r} prefix is too short ({len(prefix)} chars) "
            "to nudge a 12B model toward a distinct conflict shape"
        )


def test_knob_prompts_are_distinct():
    prefixes = set(DIVERSITY_KNOB_PROMPTS.values())
    assert len(prefixes) == 3, (
        "Each diversity knob must have a unique prompt prefix "
        "(otherwise the fan-out produces identical candidates)."
    )


def test_fan_out_result_ok_property():
    r_ok = FanOutResult(knob="x", plan=object())
    r_fail = FanOutResult(knob="y", plan=None, error="boom")
    assert r_ok.ok is True
    assert r_fail.ok is False


# ---------------------------------------------------------------------------
# fan_out_stage1_plans behaviour
# ---------------------------------------------------------------------------


def test_fan_out_runs_one_call_per_knob_in_order():
    calls: list[tuple[str, str]] = []

    def gen_fn(messages, *, temperature, max_new_tokens):
        # Capture knob (derived from system message prefix) +
        # temperature for ordering check.
        sys = messages[0]["content"]
        for knob, prefix in DIVERSITY_KNOB_PROMPTS.items():
            if sys.startswith(prefix.strip()) or sys.startswith(prefix):
                calls.append((knob, "ok"))
                return f"raw-output-for-{knob}"
        calls.append(("unknown", sys[:40]))
        return "raw-unknown"

    def parse_fn(raw: str):
        return type("Plan", (), {
            "beats": [object() for _ in range(3)],
            "cast": [object() for _ in range(2)],
            "_raw": raw,
        })()

    results = fan_out_stage1_plans(
        generate_fn=gen_fn,
        parse_fn=parse_fn,
        base_system_prompt="BASE_SYSTEM",
        user_prompt="USER",
    )
    assert len(results) == 3
    assert [r.knob for r in results] == list(ALL_DIVERSITY_KNOBS)
    assert all(r.ok for r in results)
    assert [c[0] for c in calls] == list(ALL_DIVERSITY_KNOBS)


def test_fan_out_isolates_generate_fn_failures():
    """One knob's generate_fn raises -- the other two still produce
    candidates. Failed entry has plan=None and an error string."""
    def gen_fn(messages, *, temperature, max_new_tokens):
        sys = messages[0]["content"]
        if sys.startswith(DIVERSITY_KNOB_PROMPTS[DiversityKnob.BUREAUCRATIC_ABSURD]):
            raise RuntimeError("fake LLM timeout")
        return "raw-output"

    def parse_fn(raw: str):
        return type("Plan", (), {"beats": [], "cast": [], "_raw": raw})()

    results = fan_out_stage1_plans(
        generate_fn=gen_fn,
        parse_fn=parse_fn,
        base_system_prompt="BASE",
        user_prompt="USER",
    )
    assert len(results) == 3
    by_knob = {r.knob: r for r in results}
    assert by_knob[DiversityKnob.MORAL_DILEMMA].ok is True
    assert by_knob[DiversityKnob.BUREAUCRATIC_ABSURD].ok is False
    assert "fake LLM timeout" in (
        by_knob[DiversityKnob.BUREAUCRATIC_ABSURD].error or ""
    )
    assert by_knob[DiversityKnob.INTIMATE_PERSONAL_COST].ok is True


def test_fan_out_isolates_parse_fn_failures():
    """One knob's parse_fn raises (e.g. pydantic ValidationError) --
    the failure is contained; raw_output is captured for forensic."""
    def gen_fn(messages, *, temperature, max_new_tokens):
        return "raw-output"

    def parse_fn(raw: str):
        # Fail on the second knob deterministically.
        parse_fn._n = getattr(parse_fn, "_n", 0) + 1
        if parse_fn._n == 2:
            raise ValueError("schema validation fails")
        return type("Plan", (), {"beats": [], "cast": []})()

    results = fan_out_stage1_plans(
        generate_fn=gen_fn,
        parse_fn=parse_fn,
        base_system_prompt="BASE",
        user_prompt="USER",
    )
    assert results[0].ok is True
    assert results[1].ok is False
    assert results[1].raw_output == "raw-output"
    assert "schema validation fails" in (results[1].error or "")
    assert results[2].ok is True


def test_fan_out_subset_of_knobs():
    """Pass a single knob for a degenerate 'fan-out of one' useful
    when soaking a specific diversity flavour."""
    calls: list[str] = []
    def gen_fn(messages, *, temperature, max_new_tokens):
        calls.append(messages[0]["content"][:30])
        return "raw"

    def parse_fn(raw): return type("P", (), {"beats": [], "cast": []})()

    results = fan_out_stage1_plans(
        generate_fn=gen_fn,
        parse_fn=parse_fn,
        base_system_prompt="BASE",
        user_prompt="USER",
        knobs=[DiversityKnob.INTIMATE_PERSONAL_COST],
    )
    assert len(results) == 1
    assert results[0].knob == DiversityKnob.INTIMATE_PERSONAL_COST
    assert len(calls) == 1


def test_fan_out_user_prompt_is_passed_verbatim():
    """The user prompt is identical across all candidates -- only
    the SYSTEM prompt diverges. Lets the operator A/B knob effects
    against a fixed seed user prompt."""
    captured = []
    def gen_fn(messages, *, temperature, max_new_tokens):
        captured.append(messages[1]["content"])
        return "raw"

    def parse_fn(raw): return type("P", (), {"beats": [], "cast": []})()

    fan_out_stage1_plans(
        generate_fn=gen_fn,
        parse_fn=parse_fn,
        base_system_prompt="BASE",
        user_prompt="VERBATIM USER PROMPT",
    )
    assert captured == ["VERBATIM USER PROMPT"] * 3
