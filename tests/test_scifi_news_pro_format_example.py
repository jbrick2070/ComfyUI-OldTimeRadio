"""The writer is SHOWN the grammar, not just told it.

FOUND BY THE r1 REVIEW ARC (2026-08-12). `_run_markup_ladder` has always
accepted a `format_example` and used it to build a one-shot user/assistant
demonstration -- and **nothing ever passed one**, so that path was dead code.
`scifi_news_pro.json`'s own `"examples"` list was empty, and `pack.examples` is
read nowhere in the writer. The grammar was described in prose and never
demonstrated.

Two reviewers reached this independently: from the craft side, showing one
narration-to-speech conversion is worth more than any rule sentence; from the
code side, the one-shot call path had no caller.

WHAT THESE TESTS PROTECT, in order of how badly it would hurt:

1. **The example must PARSE.** An invalid example teaches invalid grammar to
   every attempt, which is strictly worse than no example. It is validated here
   against the REAL parser, not eyeballed.
2. **It must not be liftable.** It is delivered as the model's own prior
   assistant turn, so a same-domain example invites the model to carry its cast
   or premise into the episode -- a fidelity defect worse than the format error
   it fixes. Hence a gardening programme, plus an explicit do-not-copy
   instruction in the framing turn.
3. **It must actually reach the model**, or it is dead code again.
"""
from __future__ import annotations

from nodes._otr_scifi_news_pro_markup import parse_scifi_news_pro_markup
from nodes import _otr_scifi_news_pro as scifi_news_pro


EXAMPLE = scifi_news_pro._FABLE2_FORMAT_EXAMPLE
EXAMPLE_CAST = ["MAUD", "PERCY"]


# ---------------------------------------------------------------------------
# 1. It must be valid -- checked against the real parser
# ---------------------------------------------------------------------------
def test_the_example_PARSES_with_zero_defects():
    """The one non-negotiable. Teaching a malformed example would make every
    attempt worse, and the failure would look like a model problem."""
    parsed, defects = parse_scifi_news_pro_markup(EXAMPLE, EXAMPLE_CAST)
    assert not defects, [str(d) for d in defects]
    assert parsed is not None


def test_the_example_demonstrates_every_structural_element():
    """A sample that omits an element does not teach it. These are exactly the
    rows whose absence the parser raises a skeleton break for."""
    for required in ("TITLE:", "MUSIC:", "ANNOUNCER:", "SCENE 1:",
                     "CODA:", "END."):
        assert required in EXAMPLE, required


def test_the_example_has_MORE_THAN_ONE_scene():
    """One scene would not show that a second SCENE header is legal, and the
    live failures included an `EMPTY_SCENE` and scene-boundary breaks."""
    parsed, _ = parse_scifi_news_pro_markup(EXAMPLE, EXAMPLE_CAST)
    assert len(parsed.scenes) >= 2


def test_every_spoken_row_carries_a_LEGAL_LABEL():
    """The defect the example exists to prevent: unlabelled narration. If any
    row here lacked a label the example would be teaching the exact thing that
    killed the live legs."""
    body = [ln.strip() for ln in EXAMPLE.splitlines() if ln.strip()]
    structural = ("TITLE:", "MUSIC:", "SCENE ", "CODA:", "END.")
    for row in body:
        if row.startswith(structural):
            continue
        speaker = row.split(":", 1)[0]
        assert speaker in EXAMPLE_CAST + ["ANNOUNCER"], row


def test_no_row_is_a_stage_direction_or_action_line():
    """It must not model screenplay grammar -- no parenthetical, bracketed or
    asterisked rows anywhere."""
    for row in EXAMPLE.splitlines():
        assert not row.strip().startswith(("(", "[", "*")), row


# ---------------------------------------------------------------------------
# 2. It must not be liftable into the real episode
# ---------------------------------------------------------------------------
def test_the_example_is_a_DIFFERENT_DOMAIN_from_the_assignment():
    """Delivered as the model's own prior turn, so a sci-fi example beside a
    sci-fi assignment invites the model to carry the cast across."""
    lowered = EXAMPLE.lower()
    for sci_fi_tell in ("space", "orbit", "signal", "laboratory", "probe",
                        "reactor", "station", "doctor", "scientist"):
        assert sci_fi_tell not in lowered, sci_fi_tell


def test_the_framing_turn_tells_the_model_NOT_to_copy_the_example():
    """The example alone is an invitation. The instruction is what bounds it."""
    import inspect

    src = inspect.getsource(scifi_news_pro._run_markup_ladder)
    assert "FORMAT sample only" in src
    assert "must not carry this example" in src


# ---------------------------------------------------------------------------
# 3. It must actually reach the model
# ---------------------------------------------------------------------------
def test_the_script_pass_PASSES_the_example():
    """The regression that would silently restore the dead-code state."""
    import inspect

    src = inspect.getsource(scifi_news_pro._pass_script)
    assert "format_example=_FABLE2_FORMAT_EXAMPLE" in src


def test_the_example_reaches_the_model_as_an_ASSISTANT_turn():
    """End to end through the real ladder: the one-shot demonstration must be
    an assistant turn, because that is what makes it read as 'output I have
    already produced correctly' rather than as more instructions."""
    seen = {}

    def writer(messages, *, temperature, max_new_tokens):
        rows = list(messages)
        seen["roles"] = [r["role"] for r in rows]
        seen["contents"] = [r["content"] for r in rows]
        return EXAMPLE                      # a valid play -> accepted first try

    scifi_news_pro._run_markup_ladder(
        writer,
        pass_id="script",
        system="system prompt",
        base_user="base user prompt",
        envelope=None,
        cast_names=EXAMPLE_CAST,
        initial_temperature=0.7,
        format_example=EXAMPLE,
    )
    assert "assistant" in seen["roles"], seen["roles"]
    assistant_at = seen["roles"].index("assistant")
    assert seen["contents"][assistant_at] == EXAMPLE
    # ...and the real assignment comes AFTER the demonstration.
    assert seen["roles"][-1] == "user"
    assert "base user prompt" in seen["contents"][-1]


def test_the_ladder_still_works_with_NO_example():
    """The parameter stays optional -- other callers and tests rely on it."""
    def writer(messages, *, temperature, max_new_tokens):
        assert [r["role"] for r in messages] == ["system", "user"]
        return EXAMPLE

    _raw, parsed, _diag = scifi_news_pro._run_markup_ladder(
        writer,
        pass_id="script",
        system="system prompt",
        base_user="base user prompt",
        envelope=None,
        cast_names=EXAMPLE_CAST,
        initial_temperature=0.7,
    )
    assert parsed is not None
