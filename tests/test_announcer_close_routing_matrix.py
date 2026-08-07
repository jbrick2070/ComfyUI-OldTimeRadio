"""B4's ROUTING ACCEPTANCE MATRIX -- every route THROUGH the production reader.

WHY THIS IS SEPARATE from ``tests/test_announcer_close_helper.py``. That file
proves the helper is WIRED: static AST over ``run()`` shows the call is there,
exactly once, carrying the full keyword contract. Wiring is not behaviour.
Nothing in it proves that a public_domain episode with an owned provenance fact
actually SPEAKS that fact, or that a Shakespeare episode whose normalizer
produced nothing still closes with a sign-off instead of silence.

WHY EVERY ROW ASSERTS PRESENCE. The coverage this replaces checked that no
source URL leaked into the spoken line -- a condition an EMPTY close satisfies
just as well. That test would have stayed green on an episode that closed with
nothing at all. A test a silent failure passes is the same shape of defect as a
fix applied to a function with no callers, which is the incident this whole
extraction exists to prevent. So every row asserts the coda is THERE: exact
text on the ledger row AND carried into a real ``Dialogue:`` cue by
``_otr_captions.build_ass_from_ledger``, which is the surface a viewer reads.

THE CONTROL IS ``media_archive``, NEVER ``scifi_news``. The science lane
dispatches to ``scifi_news_circuit`` and returns BEFORE this block ever runs,
so a scifi_news row would assert nothing whatsoever about this helper.

THE ROUTE IS PROVEN BY THE PATH TAKEN, not inferred from the text that came
out. The slot scheduler records which composer context was entered, so a row
claiming the deterministic coda route has to have actually entered
``compose_news_coda``.

Pure Python: no GPU, no model, no network.
"""
from __future__ import annotations

import contextlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_captions as CAP  # noqa: E402
from nodes import _otr_line_composer as LC  # noqa: E402
from nodes.OTR_LedgerScriptWriter import (  # noqa: E402
    _compose_and_stamp_announcer_close as compose_close,
)

FIRST_ANNOUNCER = "beat_000_announcer"
LAST_ANNOUNCER = "beat_009_announcer"
PLACEHOLDER = "Closing line pending."
INTRO_TEXT = "Good evening. This is SIGNAL LOST, and tonight the reel is old."
SCRIPT_BRIEF = "An archivist finds a reel whose label contradicts its contents."
PREMISE = "A night archivist screens a mislabelled reel."
CREATIVE_MODEL = "local:mistral-nemo"

# The stub creative replies, both confirmed against the REAL validators before
# being pinned: `validate_news_coda_bridge` and `validate_announcer_line` each
# accept them, so every row takes its composer's SUCCESS path. A reply that
# quietly failed validation would collapse every row onto the same structural
# fallback text and the matrix would stop distinguishing routes at all.
BRIDGE_REPLY = "Beyond tonight's flickering reel"
OUTRO_REPLY = (
    "The reel is shelved once more, its label true at last, and the "
    "archive light burns a little steadier tonight."
)

PD_FACT = "Adapted from the 1623 First Folio, which is in the public domain."
SHAKESPEARE_FACT = (
    "Adapted from the 1608 First Quarto of KING LEAR, in the public domain."
)
NEWS_BRIEF = "Tonight's reel was restored by the Prelinger Archives."

#: What ``fallback_announcer_outro("")`` ships. The owned-but-silent lane must
#: still close with THIS -- "no fact worth speaking" is not "no closing line",
#: and the difference is the whole point of asserting presence.
GENERIC_SIGN_OFF = "This has been SIGNAL LOST. Good night."

#: Absent-field sentinel: `news_coda_emitted` is stamped only when the lane is
#: owned or the style grammar is on, so "False" and "never written" are
#: different receipts and the matrix must be able to tell them apart.
ABSENT = object()


def _coda(fact: str) -> str:
    """The surface ``_assemble_news_coda_surface`` builds: bridge, then the
    complete fact, unshortened."""
    return f"{BRIDGE_REPLY}: {fact}"


@dataclass(frozen=True)
class Route:
    """One row of the acceptance matrix.

    ``context`` is the scheduler helper context that MUST be entered. An empty
    string means neither composer runs -- the owned-but-silent lane, which is
    the path most likely to leak a raw interpreter note and therefore the one
    worth pinning hardest.
    """

    label: str
    bank: str
    provenance_owned: bool
    spoken_fact: str
    style_grammar_on: bool
    nc_brief: str
    reply: str
    expect_source: str
    expect_context: str
    expect_text: str
    expect_emitted: object


ROUTES = (
    # -- the two fidelity banks x {non-empty, empty} provenance ------------
    Route(
        label="public_domain-owned-fact",
        bank="public_domain",
        provenance_owned=True,
        spoken_fact=PD_FACT,
        style_grammar_on=True,
        nc_brief="",
        reply=BRIDGE_REPLY,
        expect_source="provenance",
        expect_context="compose_news_coda",
        expect_text=_coda(PD_FACT),
        expect_emitted=True,
    ),
    Route(
        label="public_domain-owned-silent",
        bank="public_domain",
        provenance_owned=True,
        spoken_fact="",
        style_grammar_on=True,
        nc_brief="",
        reply=BRIDGE_REPLY,
        expect_source="none",
        expect_context="",
        expect_text=GENERIC_SIGN_OFF,
        expect_emitted=False,
    ),
    Route(
        label="shakespeare-owned-fact",
        bank="shakespeare",
        provenance_owned=True,
        spoken_fact=SHAKESPEARE_FACT,
        style_grammar_on=True,
        nc_brief="",
        reply=BRIDGE_REPLY,
        expect_source="provenance",
        expect_context="compose_news_coda",
        expect_text=_coda(SHAKESPEARE_FACT),
        expect_emitted=True,
    ),
    Route(
        label="shakespeare-owned-silent",
        bank="shakespeare",
        provenance_owned=True,
        spoken_fact="",
        style_grammar_on=True,
        nc_brief="",
        reply=BRIDGE_REPLY,
        expect_source="none",
        expect_context="",
        expect_text=GENERIC_SIGN_OFF,
        expect_emitted=False,
    ),
    # -- owned + non-empty with the style grammar OFF ----------------------
    # An owned lane takes the deterministic append INDEPENDENTLY of the style
    # gate. If the gate ever starts governing owned lanes, a fidelity episode
    # silently stops speaking its source and only these two rows notice.
    Route(
        label="public_domain-owned-fact-style-off",
        bank="public_domain",
        provenance_owned=True,
        spoken_fact=PD_FACT,
        style_grammar_on=False,
        nc_brief="",
        reply=BRIDGE_REPLY,
        expect_source="provenance",
        expect_context="compose_news_coda",
        expect_text=_coda(PD_FACT),
        expect_emitted=True,
    ),
    Route(
        label="shakespeare-owned-fact-style-off",
        bank="shakespeare",
        provenance_owned=True,
        spoken_fact=SHAKESPEARE_FACT,
        style_grammar_on=False,
        nc_brief="",
        reply=BRIDGE_REPLY,
        expect_source="provenance",
        expect_context="compose_news_coda",
        expect_text=_coda(SHAKESPEARE_FACT),
        expect_emitted=True,
    ),
    # -- the control lane, on BOTH composer branches -----------------------
    Route(
        label="media_archive-news-brief",
        bank="media_archive",
        provenance_owned=False,
        spoken_fact="",
        style_grammar_on=True,
        nc_brief=NEWS_BRIEF,
        reply=BRIDGE_REPLY,
        expect_source="news_close_brief",
        expect_context="compose_news_coda",
        expect_text=_coda(NEWS_BRIEF),
        expect_emitted=True,
    ),
    Route(
        label="media_archive-fictional-close",
        bank="media_archive",
        provenance_owned=False,
        spoken_fact="",
        style_grammar_on=False,
        nc_brief="",
        reply=OUTRO_REPLY,
        expect_source="none",
        expect_context="compose_announcer_outro",
        expect_text=OUTRO_REPLY,
        expect_emitted=ABSENT,
    ),
)


# ---------------------------------------------------------------------------
# Stubs -- narrow on purpose
# ---------------------------------------------------------------------------

class _Ledger:
    """The narrowest ledger surface the helper actually touches: ``.data``.

    ``.save()`` is deliberately ABSENT. Persistence lives in the CALLER by
    design, which is what keeps this helper in-memory and testable. A stub that
    offered ``.save()`` would let a future change move the write into the
    helper with nothing going red.
    """

    def __init__(self, data: dict) -> None:
        self.data = data


class _RecordingScheduler:
    """Records which composer context was entered, in order."""

    def __init__(self) -> None:
        self.contexts: list[str] = []

    @contextlib.contextmanager
    def helper_context(self, name: str):
        self.contexts.append(name)
        yield


def _creative_fn(reply: str):
    """A creative lane that records every messages array and returns ``reply``."""
    calls: list = []

    def fn(messages, **kwargs):
        calls.append(messages)
        return reply

    return fn, calls


def _ledger_data() -> dict:
    """An episode reduced to what both the helper and the caption builder need.

    The announcer rows carry ``char_id="announcer"`` -- the role TAG rather than
    the cast row id -- because that is what the announcer passes really write,
    and resolving it through the cast alias is part of what the caption surface
    has to get right.
    """
    return {
        "episode_id": "announcer_close_routing_matrix",
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER"},
            {"char_id": "c02", "name": "MARGUERITE"},
        ],
        "lines": [
            {
                "line_id": FIRST_ANNOUNCER,
                "speaker_role": "announcer",
                "char_id": "announcer",
                "text": INTRO_TEXT,
                "start_s": 0.0,
                "dur_s": 6.0,
            },
            {
                "line_id": "beat_004_character",
                "speaker_role": "character",
                "char_id": "c02",
                "text": "Then we thread the reel and see what the label hid.",
                "start_s": 6.0,
                "dur_s": 5.0,
            },
            {
                "line_id": LAST_ANNOUNCER,
                "speaker_role": "announcer",
                "char_id": "announcer",
                "text": PLACEHOLDER,
                "start_s": 11.0,
                "dur_s": 14.0,
            },
        ],
    }


def _run(route: Route, *, credits_source_line: str = "", premise=PREMISE):
    led = _Ledger(_ledger_data())
    meta: dict = {}
    if credits_source_line:
        meta["credits_source_line"] = credits_source_line
    scheduler = _RecordingScheduler()
    fn, calls = _creative_fn(route.reply)
    result = compose_close(
        led,
        meta,
        first_announcer_id=FIRST_ANNOUNCER,
        last_announcer_id=LAST_ANNOUNCER,
        provenance_owned=route.provenance_owned,
        style_grammar_on=route.style_grammar_on,
        effective_spoken_fact=route.spoken_fact,
        nc_brief=route.nc_brief,
        script_brief=SCRIPT_BRIEF,
        premise=premise,
        resolved={
            "creative_writing_model": CREATIVE_MODEL,
            "source_bank": route.bank,
        },
        slot_scheduler=scheduler,
        creative_generate_fn=fn,
    )
    return led, meta, result, scheduler, calls


def _closing_row(led: _Ledger) -> dict:
    for row in led.data["lines"]:
        if row["line_id"] == LAST_ANNOUNCER:
            return row
    raise AssertionError(f"{LAST_ANNOUNCER} vanished from the ledger")


_OVERRIDE_BLOCK = re.compile(r"\{[^}]*\}")


def _dialogue_text(ass_path: Path) -> str:
    """Every ``Dialogue:`` cue's visible text, rejoined into one string.

    Cues are word-wrapped at 44 characters and grouped two physical lines to a
    cue with a literal ``\\N`` between them, so recovering the spoken line means
    undoing exactly that: drop the ASS override blocks (the bold speaker
    label), turn the breaks back into spaces, and collapse. The event format is
    nine comma-separated fields before the text, hence the capped split.
    """
    parts: list[str] = []
    for raw in ass_path.read_text(encoding="utf-8").splitlines():
        if not raw.startswith("Dialogue:"):
            continue
        body = raw.split(",", 9)[9]
        body = _OVERRIDE_BLOCK.sub("", body).replace("\\N", " ")
        parts.append(body)
    assert parts, f"no Dialogue cues were emitted into {ass_path}"
    return " ".join(" ".join(parts).split())


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ROUTES, ids=[r.label for r in ROUTES])
def test_route_takes_its_branch_and_speaks_a_close(route: Route):
    """Each lane routes where it should AND leaves a non-empty closing line."""
    led, meta, result, scheduler, _calls = _run(route)

    assert meta["spoken_coda_source"] == route.expect_source, (
        f"{route.label}: the receipt names the wrong route")

    entered = [c for c in scheduler.contexts
               if c in ("compose_news_coda", "compose_announcer_outro")]
    if route.expect_context:
        assert entered == [route.expect_context], (
            f"{route.label}: expected the {route.expect_context} branch, "
            f"entered {entered}")
    else:
        assert entered == [], (
            f"{route.label}: the owned-but-silent lane must enter NEITHER "
            f"composer, entered {entered}")

    row = _closing_row(led)
    assert row["text"], f"{route.label}: the episode closed with nothing"
    assert row["text"] != PLACEHOLDER, (
        f"{route.label}: the deterministic placeholder was never overwritten")
    assert row["text"] == route.expect_text, f"{route.label}: wrong close"
    assert result.text == row["text"], (
        f"{route.label}: the returned LineResult disagrees with the ledger")

    if route.expect_emitted is ABSENT:
        assert "news_coda_emitted" not in meta, (
            f"{route.label}: an unowned lane with the style grammar off "
            "stamps no emission receipt at all")
    else:
        assert meta["news_coda_emitted"] is route.expect_emitted, (
            f"{route.label}: wrong news_coda_emitted")
        assert meta["news_coda_fallback"] is False


@pytest.mark.parametrize("route", ROUTES, ids=[r.label for r in ROUTES])
def test_route_reaches_the_caption_surface(route: Route, tmp_path: Path):
    """The close a viewer READS. A coda that exists only in the ledger and
    never reaches a cue is not delivered, and presence in `lines[].text` alone
    cannot tell those apart."""
    led, _meta, _result, _scheduler, _calls = _run(route)

    ledger_path = tmp_path / f"{route.label}_ledger.json"
    ledger_path.write_text(
        json.dumps(led.data, ensure_ascii=False), encoding="utf-8")

    out_path, report = CAP.build_ass_from_ledger(ledger_path)
    assert out_path is not None, f"{route.label}: caption build failed: {report}"

    spoken = _dialogue_text(Path(out_path))
    assert route.expect_text in spoken, (
        f"{route.label}: the close never reached a Dialogue cue.\n"
        f"  wanted: {route.expect_text}\n"
        f"  cues:   {spoken}")


# ---------------------------------------------------------------------------
# Guards on the matrix itself -- a deleted row must not retire a branch
# ---------------------------------------------------------------------------

def test_the_matrix_covers_both_composer_branches():
    """``compose_news_coda`` and ``compose_announcer_outro`` bind DIFFERENT
    outer names -- ``script_brief`` reaches only the second -- so a matrix that
    exercises one leaves a NameError reachable on the other. That is the exact
    defect class two independent reviewers found in this extraction's own
    boundary, at a third location."""
    covered = {r.expect_context for r in ROUTES}
    assert "compose_news_coda" in covered
    assert "compose_announcer_outro" in covered
    assert "" in covered, "the owned-but-silent lane enters neither composer"


def test_the_control_lane_is_media_archive_and_never_scifi_news():
    banks = {r.bank for r in ROUTES}
    assert {"public_domain", "shakespeare"} <= banks, (
        "both fidelity banks are the point of this matrix")
    assert "media_archive" in banks, "the unowned control lane is missing"
    assert "scifi_news" not in banks, (
        "scifi_news dispatches to scifi_news_circuit and returns BEFORE this "
        "block runs, so such a row would prove nothing about this helper")


def test_every_fidelity_bank_is_covered_with_and_without_provenance():
    for bank in ("public_domain", "shakespeare"):
        facts = {bool(r.spoken_fact) for r in ROUTES if r.bank == bank}
        assert facts == {True, False}, (
            f"{bank} must be exercised with AND without a spoken fact")
    off = {r.bank for r in ROUTES
           if r.provenance_owned and r.spoken_fact and not r.style_grammar_on}
    assert off == {"public_domain", "shakespeare"}, (
        "the owned/non-empty case with the style grammar OFF is the row that "
        "proves an owned lane bypasses the style gate")


# ---------------------------------------------------------------------------
# The contracts the routing matrix rides on
# ---------------------------------------------------------------------------

def test_the_helper_returns_the_line_result_the_caller_logs():
    """The caller reads ``outro_res.compose_flags`` after the call. A helper
    that mutated and returned None would be a NameError on every episode that
    composes a coda -- which is why the first proposed boundary was wrong."""
    _led, _meta, result, _scheduler, _calls = _run(ROUTES[0])
    assert isinstance(result, LC.LineResult)
    assert result.compose_flags, "the flags the caller logs must be populated"


def test_a_none_premise_never_stringifies_into_the_prompt():
    """``premise`` is required keyword-only precisely so missing wiring cannot
    pass silently -- but a caller handing over None must still not put the word
    "None" in front of the model."""
    _led, _meta, _result, _scheduler, calls = _run(ROUTES[0], premise=None)
    user_message = calls[0][1]["content"]
    assert "PREMISE: None" not in user_message
    assert user_message.startswith("PREMISE: \n")


def test_the_owned_but_silent_lane_defers_to_credits_only_when_they_carry_it():
    """The flag must not claim an attribution surface that does not exist --
    the same class of false receipt this item exists to end."""
    silent = next(r for r in ROUTES if r.label == "public_domain-owned-silent")

    _led, _meta, carried, _sched, _calls = _run(
        silent, credits_source_line="Source: the 1623 First Folio.")
    assert "source_note_deferred_to_credits" in carried.compose_flags

    _led, _meta, empty, _sched, _calls = _run(silent)
    assert "source_note_deferred_to_credits" not in empty.compose_flags


def test_the_receipt_vocabulary_stays_closed():
    """``spoken_coda_source`` is validated at WRITE time so a typo cannot drift
    into the ledger and quietly widen what the corpus audit accepts."""
    from nodes.OTR_LedgerScriptWriter import _SPOKEN_CODA_SOURCES

    assert _SPOKEN_CODA_SOURCES == frozenset(
        {"provenance", "news_close_brief", "none"})
    assert {r.expect_source for r in ROUTES} == _SPOKEN_CODA_SOURCES, (
        "the matrix must exercise every value the vocabulary allows")
