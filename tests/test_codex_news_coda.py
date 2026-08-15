"""tests/test_codex_news_coda.py -- the closing announcer read (P6).

PBUG-20260814-02. The `scifi_news` lane published an episode with its only
announcer row in the MIDDLE and never named the real news story it was drawn
from. The coda existed as prompt text string-concatenated onto two other
passes: no output type, nothing reserving a final row, nothing verifying
afterwards. The sibling lane `scifi_news_pro` already refuses a draft with no
coda and Python-appends its validated news read as its own row, so the fix is
levelling this lane up to that structure rather than inventing one.

What is pinned here:

  * the SOURCE ANCHORS -- what a coda has to say for the check to pass, and
    the two ways a sloppier check would wave a bad one through (a two-letter
    "entity", a substring match inside another word).
  * the CLEAN, not a reroll. A coda missing its attribution is a good draft
    missing one thing, so it comes back with the complaint attached exactly
    once. Operator ruling 2026-08-14: a firing verifier triggers a clean,
    never a refusal and never a reroll.
  * the THREE OUTCOMES, all of which continue the render: clean, unclean
    (ships, flagged), absent (nothing invented).
  * the PLACEMENT post-condition. The row is placed by Python, so a
    violation means this code is wrong and the failure is meant to be loud.

Pure-Python. No GPU. No LLM.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_scifi_codex as lane  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _span(quote: str = "starfish cells moved toward the light") -> Any:
    return lane.SourceSpanV4(
        field="full_text", start=0, end=len(quote), quote=quote,
    )


def _fact_index(*, entities=(("E01", "MIT"), ("E02", "starfish")),
                numbers=(("N01", "1,200"),)) -> lane.FactIndexV4:
    return lane.FactIndexV4(
        facts=[
            lane.FactV4(
                fact_id="F01",
                claim="Light moved starfish cells in a laboratory dish.",
                source_spans=[_span()],
                numeric_tokens=["1,200"],
            ),
        ],
        entities=[
            lane.EntityV4(entity_id=eid, name=name, source_spans=[_span()])
            for eid, name in entities
        ],
        numbers=[
            lane.NumberV4(
                number_id=nid, verbatim=value, fact_id="F01",
                source_span=_span(),
            )
            for nid, value in numbers
        ],
        tone="measured",
        payload_sha256="0" * 64,
    )


def _cast() -> lane.CastPlanV4:
    return lane.CastPlanV4(cast=[
        lane.CastPlanRowV4(
            char_id="announcer", name="ANNOUNCER",
            character_description="The night-shift voice.",
            gender="neutral", role_in_conflict="Frames the report.",
            voice_slot="announcer",
        ),
        lane.CastPlanRowV4(
            char_id="c01", name="Ada",
            character_description="A cell biologist.",
            gender="female", role_in_conflict="Chases the result.",
            voice_slot="c01",
        ),
    ])


def _score() -> Any:
    """`_call_news_coda` reads only the three story-surface strings.

    A SimpleNamespace keeps this a unit test: building a whole RadioScoreV4
    would pin the compiler's contract here as a side effect, and the compiler
    already has its own tests.
    """
    return SimpleNamespace(
        title="Signal Lost", premise="A result nobody can repeat.",
        setting="A basement lab after hours.",
    )


class _FakeCoda:
    """Stand-in for `invoke_codex_structured`, one scripted reply per call."""

    def __init__(self, *replies: "str | BaseException"):
        self.replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        if not self.replies:
            raise AssertionError("the coda pass was called more times "
                                 "than the bounded ceiling allows")
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return lane.NewsCodaV4(text=reply)


# ---------------------------------------------------------------------------
# 1. What counts as naming the source
# ---------------------------------------------------------------------------

def test_anchors_are_entity_names_and_figures_in_index_order():
    anchors = lane._news_coda_source_anchors(_fact_index())
    assert anchors == ("MIT", "starfish", "1,200")


def test_a_two_letter_entity_is_not_an_anchor():
    """A short "entity" matches inside ordinary words, so it proves nothing.

    Requiring it would let a coda that names NOTHING pass by accident, which
    is worse than having no check at all.
    """
    anchors = lane._news_coda_source_anchors(
        _fact_index(entities=(("E01", "AI"), ("E02", "MIT")), numbers=()),
    )
    assert "AI" not in anchors
    assert "MIT" in anchors


def test_anchor_matching_is_word_boundary_not_substring():
    assert lane._names_a_source_anchor("the signal was transmitted", ("MIT",)) \
        is None
    assert lane._names_a_source_anchor("reported by MIT tonight", ("MIT",)) \
        == "MIT"


def test_anchor_matching_ignores_case_and_survives_trailing_punctuation():
    assert lane._names_a_source_anchor("word from mit.", ("MIT",)) == "MIT"


# ---------------------------------------------------------------------------
# 2. The detector
# ---------------------------------------------------------------------------

def _findings(text: str, index=None) -> list[str]:
    index = index if index is not None else _fact_index()
    return lane._news_coda_findings(
        text,
        lane._news_coda_source_anchors(index),
        lane._spoken_label_pattern(_cast()),
    )


def test_a_coda_that_names_an_anchor_is_clean():
    assert _findings(
        "Tonight's story began with a real report from MIT; the rest was "
        "ours."
    ) == []


def test_a_coda_that_names_nothing_is_a_finding():
    findings = _findings(
        "Tonight's story was invented for you, and we hope you enjoyed it."
    )
    assert len(findings) == 1
    assert "never names the real source" in findings[0]


def test_spoken_hygiene_and_the_missing_source_are_reported_together():
    """The clean pass gets ONE bounded attempt, so it has to hear both.

    A validator that surfaces one defect at a time spends that attempt on
    the first and dies on the second -- the same reasoning the P5 validator
    already carries.
    """
    findings = _findings("ANNOUNCER: and that was our story.")
    assert len(findings) == 2
    assert any("role label" in row for row in findings)
    assert any("never names the real source" in row for row in findings)


def test_with_no_indexed_anchors_only_hygiene_applies():
    """P0 found no entities or figures, so there is no verbatim string the
    coda could be REQUIRED to say. Refusing the episode for an upstream gap
    would be a length-gate in disguise; the receipt carries the zero instead.
    """
    bare = _fact_index(entities=(), numbers=())
    bare = bare.model_copy(update={
        "facts": [bare.facts[0].model_copy(update={"numeric_tokens": []})],
    })
    assert lane._news_coda_source_anchors(bare) == ()
    assert _findings("And that was tonight's story.", index=bare) == []


# ---------------------------------------------------------------------------
# 3. The pass: clean, unclean, absent -- and never a reroll
# ---------------------------------------------------------------------------

def _run(monkeypatch, fake: _FakeCoda):
    monkeypatch.setattr(lane, "invoke_codex_structured", fake)
    return lane._call_news_coda(
        slot_fn=lambda *_a, **_k: "",
        pack=SimpleNamespace(prompt_stages={
            "codex_coda_contract_system": "coda seam",
        }),
        fact_index=_fact_index(),
        score=_score(),
        cast=_cast(),
        episode_spoken_text=["A result nobody can repeat."],
        call_journal={},
    )


def test_a_clean_first_draft_costs_exactly_one_call(monkeypatch):
    fake = _FakeCoda("Tonight began with a real MIT report; the rest was ours.")
    text, receipt = _run(monkeypatch, fake)

    assert receipt["status"] == "clean"
    assert text == "Tonight began with a real MIT report; the rest was ours."
    assert len(fake.calls) == 1
    assert receipt["attempts"] == [
        {"attempt": 1, "outcome": "clean", "findings": []},
    ]


def test_the_pass_never_rerolls_and_owns_its_own_decode_budget(monkeypatch):
    """Two properties that have to hold together.

    `retry_until_valid=False` keeps a missing attribution out of the
    cold-redraw loop, and the empty `post_validator` is what moves the
    verdict OUTSIDE the ladder so it can trigger a clean instead. The budget
    is the "right-size the job, never raise the guard" half: this pass writes
    one sentence and must not run on a whole script's allowance.
    """
    fake = _FakeCoda("A real MIT report started this.")
    _run(monkeypatch, fake)

    call = fake.calls[0]
    assert call["pass_id"] == "P6"
    assert call["slot"] == "creative"
    assert call["seam_refs"] == ("codex_coda_contract_system",)
    assert call["result_type"] is lane.NewsCodaV4
    assert call["retry_until_valid"] is False
    assert call["post_validator"](object()) is None
    assert call["max_new_tokens"] == lane._NEWS_CODA_MAX_OUTPUT_TOKENS


def test_an_unclean_draft_comes_back_with_the_complaint_attached(monkeypatch):
    fake = _FakeCoda(
        "And that was tonight's story, invented whole.",
        "Tonight's story grew from a real MIT result; the rest was ours.",
    )
    text, receipt = _run(monkeypatch, fake)

    assert len(fake.calls) == 2
    assert receipt["status"] == "clean"
    assert text.endswith("the rest was ours.")

    clean_inputs = fake.calls[1]["artifact_inputs"]
    assert clean_inputs["previous_attempt"] == \
        "And that was tonight's story, invented whole."
    assert any(
        "never names the real source" in row
        for row in clean_inputs["unmet_requirements"]
    )
    # The first call is an authoring job and carries no complaint.
    assert "unmet_requirements" not in fake.calls[0]["artifact_inputs"]


def test_a_coda_that_never_names_its_source_still_ships_flagged(monkeypatch):
    """An imperfect attribution beats none, and a render is never killed."""
    fake = _FakeCoda(
        "And that was tonight's story.",
        "That was tonight's story, and we thank you for listening.",
    )
    text, receipt = _run(monkeypatch, fake)

    assert len(fake.calls) == lane._NEWS_CODA_MAX_ATTEMPTS
    assert receipt["status"] == "unclean"
    assert text == "That was tonight's story, and we thank you for listening."
    assert all(row["outcome"] == "unclean" for row in receipt["attempts"])


def test_the_receipt_explains_the_text_that_actually_ships(monkeypatch):
    """A receipt that blames the wrong draft is worse than no receipt.

    `text` only advances on a non-empty candidate. The VERDICT has to advance
    with it: attempt 1 writes a coda naming no source, attempt 2 returns
    nothing, and the row going to air is still attempt 1's. If the findings
    kept walking, the ledger would say "spoken text is empty" about a row that
    is not empty, and whoever debugs it later starts from a lie.
    """
    fake = _FakeCoda("And that was tonight's story.", "   ")
    text, receipt = _run(monkeypatch, fake)

    assert text == "And that was tonight's story."
    assert receipt["status"] == "unclean"
    # The complaint that survives is the one about the shipped text.
    joined = " ".join(receipt["attempts"][0]["findings"])
    assert "never names the real source" in joined


def test_an_empty_first_attempt_does_not_poison_a_good_second(monkeypatch):
    # The schema forbids a truly empty string, so the only way a pass
    # yields nothing is whitespace that strips away.
    fake = _FakeCoda("   ", "A real MIT report started this.")
    text, receipt = _run(monkeypatch, fake)

    assert text == "A real MIT report started this."
    assert receipt["status"] == "clean"


def test_two_empty_attempts_are_absent_not_unclean(monkeypatch):
    fake = _FakeCoda(" ", "   ")
    text, receipt = _run(monkeypatch, fake)

    assert text is None
    assert receipt["status"] == "absent"


def test_a_pass_that_cannot_author_returns_absent_and_invents_nothing(
    monkeypatch,
):
    fake = _FakeCoda(lane.CodexPassError("P6 failed"))
    text, receipt = _run(monkeypatch, fake)

    assert text is None
    assert receipt["status"] == "absent"
    assert receipt["attempts"] == [{
        "attempt": 1, "outcome": "call_failed", "detail": "P6 failed",
    }]


# ---------------------------------------------------------------------------
# 4. Placement -- a post-condition on Python's own row
# ---------------------------------------------------------------------------

def _rows(*specs) -> dict[str, Any]:
    return {"lines": [
        {"line_id": lid, "speaker_role": role, "compose_flags": list(flags)}
        for lid, role, flags in specs
    ]}


def test_the_coda_is_accepted_when_it_is_the_last_spoken_row():
    lane._assert_news_coda_is_last(
        _rows(
            ("l001", "character", []),
            ("l999", "announcer", ["news_coda"]),
        ),
        expected_present=True,
    )


def test_a_missing_coda_row_is_caught():
    with pytest.raises(lane.CodexGraphError, match="0 news-coda rows"):
        lane._assert_news_coda_is_last(
            _rows(("l001", "character", [])), expected_present=True,
        )


def test_a_second_coda_row_is_caught():
    with pytest.raises(lane.CodexGraphError, match="2 news-coda rows"):
        lane._assert_news_coda_is_last(
            _rows(
                ("l998", "announcer", ["news_coda"]),
                ("l999", "announcer", ["news_coda"]),
            ),
            expected_present=True,
        )


def test_a_character_speaking_the_coda_is_caught():
    with pytest.raises(lane.CodexGraphError, match="not spoken by the announcer"):
        lane._assert_news_coda_is_last(
            _rows(("l999", "character", ["news_coda"])),
            expected_present=True,
        )


def test_dialogue_after_the_coda_is_caught():
    with pytest.raises(lane.CodexGraphError, match="follow the news coda"):
        lane._assert_news_coda_is_last(
            _rows(
                ("l999", "announcer", ["news_coda"]),
                ("l002", "character", []),
            ),
            expected_present=True,
        )


def test_an_unauthored_coda_must_not_leave_a_row_behind():
    lane._assert_news_coda_is_last(
        _rows(("l001", "character", [])), expected_present=False,
    )
    with pytest.raises(lane.CodexGraphError, match="no coda text was authored"):
        lane._assert_news_coda_is_last(
            _rows(("l999", "announcer", ["news_coda"])),
            expected_present=False,
        )
