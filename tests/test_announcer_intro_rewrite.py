"""Post-composition announcer-intro REWRITE (INTRO_REWRITE_SPEC 2026-07-09,
kibitz r2-r4 shape A) -- CHUNK A tests.

Covers the derive pass (_otr_story_brief.derive_produced_open_brief + its
scene-1 input starvation builder + roster gate), the writer's row-apply
helper (read-extend-patch flags, RuntimeError on a missing row -- kibitz r4
P1), the run() ORDER PIN (rewrite before the I.5 outro pass, so the outro
tone-echo reads the FINAL intro), and the r3 D4 title-regen root-cause fix
(ledger-assembled script, not the stale script_text_parts).

Pure Python; no GPU, no real LLM, no network.
"""
from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_brief as SB  # noqa: E402
from nodes import OTR_LedgerScriptWriter as W  # noqa: E402
from nodes._otr_structured_call import StructuredCallFailedError  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

INTRO_ID = "beat_000_announcer"


def _line(line_id, role, text, flags=()):
    return {
        "line_id": line_id,
        "speaker_role": role,
        "text": text,
        "compose_flags": list(flags),
    }


def _ledger_dict():
    """Intro row, a music marker, a 3-row scene 1, a marker, scene 2."""
    return {
        "lines": [
            _line(INTRO_ID, "announcer",
                  "Good evening. This is SIGNAL LOST.",
                  flags=("announcer_intro",)),
            _line("m1", "music_inter", ""),
            _line("b1", "character", "MARA: The lamp went dark an hour ago."),
            _line("b2", "character", "KEEPER: Then someone climbed the stairs."),
            _line("b3", "character", "MARA: Nobody climbs in this storm."),
            _line("m2", "music_inter", ""),
            _line("b4", "character", "KEEPER: The door was open when I came."),
            _line("z9", "announcer", "This has been SIGNAL LOST. Good night."),
        ],
        "cast": [{"name": "Mara"}, {"name": "Keeper"}],
    }


def _fake_fn(reply):
    def _fn(messages, **kw):
        return reply
    return _fn


_GOOD_JSON = json.dumps({
    "setting": "a storm-lashed lighthouse",
    "time_of_day": "night",
    "opening_status_quo": "the lamp has gone dark and someone is inside",
    "cast": ["mara", "KEEPER"],
})


# ---------------------------------------------------------------------------
# Derive input builder -- scene-1 slice + starvation
# ---------------------------------------------------------------------------

class TestBuildProducedOpenInput:
    def test_scene1_slice_and_roster(self):
        text, roster = SB._build_produced_open_input(_ledger_dict(), INTRO_ID)
        assert roster == ["Mara", "Keeper"]
        assert "CAST ROSTER: Mara, Keeper" in text
        # Scene-1 rows in, intro + scene-2 + outro OUT (input starvation).
        assert "lamp went dark" in text
        assert "climbed the stairs" in text
        assert "Nobody climbs" in text
        assert "Good evening" not in text          # intro row excluded
        assert "door was open" not in text          # scene 2 excluded
        assert "Good night" not in text             # outro excluded

    def test_leading_marker_after_intro_is_skipped(self):
        # The intro->scene-1 marker must not zero out the slice.
        led = _ledger_dict()
        assert led["lines"][1]["speaker_role"] == "music_inter"
        text, _ = SB._build_produced_open_input(led, INTRO_ID)
        assert "SCENE 1 (3 spoken lines):" in text

    def test_no_spoken_rows_raises(self):
        led = {"lines": [
            _line(INTRO_ID, "announcer", "Good evening."),
            _line("m1", "music_inter", ""),
        ], "cast": []}
        with pytest.raises(ValueError):
            SB._build_produced_open_input(led, INTRO_ID)

    def test_excerpt_char_cap(self):
        led = _ledger_dict()
        long = "X" * (SB._PRODUCED_OPEN_EXCERPT_CAP + 100)
        led["lines"][2]["text"] = long
        text, _ = SB._build_produced_open_input(led, INTRO_ID)
        # Cap hit on the first row: collection stops there.
        assert "climbed the stairs" not in text


# ---------------------------------------------------------------------------
# Derive pass -- structured_call + roster gate + normalization
# ---------------------------------------------------------------------------

class TestDeriveProducedOpenBrief:
    def test_valid_reply_normalizes_cast_to_roster_casing(self):
        model = SB.derive_produced_open_brief(
            _ledger_dict(),
            first_announcer_line_id=INTRO_ID,
            technical_fn=_fake_fn(_GOOD_JSON),
        )
        assert model.setting == "a storm-lashed lighthouse"
        assert model.cast == ["Mara", "Keeper"]   # roster-cased, order kept

    def test_cast_name_off_roster_exhausts_ladder(self):
        bad = json.dumps({
            "setting": "a lighthouse", "time_of_day": "",
            "opening_status_quo": "s", "cast": ["Dr. Nemo"],
        })
        with pytest.raises(StructuredCallFailedError):
            SB.derive_produced_open_brief(
                _ledger_dict(),
                first_announcer_line_id=INTRO_ID,
                technical_fn=_fake_fn(bad),
            )

    def test_all_empty_brief_rejected(self):
        empty = json.dumps({
            "setting": "", "time_of_day": "",
            "opening_status_quo": "", "cast": [],
        })
        with pytest.raises(StructuredCallFailedError):
            SB.derive_produced_open_brief(
                _ledger_dict(),
                first_announcer_line_id=INTRO_ID,
                technical_fn=_fake_fn(empty),
            )


# ---------------------------------------------------------------------------
# Writer row-apply helper -- kibitz r4 P1 semantics
# ---------------------------------------------------------------------------

class TestApplyIntroRewriteResult:
    def test_success_patches_text_and_extends_flags(self):
        led = SimpleNamespace(data=_ledger_dict())
        W._apply_intro_rewrite_result(
            led, INTRO_ID, "A new opening line.", "announcer_intro_rewritten",
        )
        row = led.data["lines"][0]
        assert row["text"] == "A new opening line."
        assert row["char_count"] == len("A new opening line.")
        assert row["word_count"] == 4
        # READ-EXTEND-PATCH: in-loop telemetry survives.
        assert row["compose_flags"] == [
            "announcer_intro", "announcer_intro_rewritten",
        ]

    def test_failure_keeps_text_flags_only(self):
        led = SimpleNamespace(data=_ledger_dict())
        before = led.data["lines"][0]["text"]
        W._apply_intro_rewrite_result(
            led, INTRO_ID, None, "announcer_intro_rewrite_failed",
        )
        row = led.data["lines"][0]
        assert row["text"] == before
        assert row["compose_flags"] == [
            "announcer_intro", "announcer_intro_rewrite_failed",
        ]

    def test_missing_row_raises_runtime_error(self):
        led = SimpleNamespace(data={"lines": []})
        with pytest.raises(RuntimeError):
            W._apply_intro_rewrite_result(
                led, "nope", "text", "announcer_intro_rewritten",
            )

    def test_a_rejected_rewrite_leaves_no_trace_of_its_flags(self):
        """Flags are appended even when the TEXT is not.

        So a rejected rewrite that still handed over its compose_flags would
        stamp `announcer_intro_structural_fallback` onto a row whose text came
        from the in-loop pass -- a receipt describing a composition that never
        reached the ledger. The caller drops the flags; this pins that it must.
        """
        led = SimpleNamespace(data=_ledger_dict())
        before = led.data["lines"][0]["text"]

        W._apply_intro_rewrite_result(
            led, INTRO_ID, None, "announcer_intro_rewrite_failed", (),
        )

        row = led.data["lines"][0]
        assert row["text"] == before
        assert row["compose_flags"] == [
            "announcer_intro", "announcer_intro_rewrite_failed",
        ]
        assert "announcer_intro_structural_fallback" not in row["compose_flags"]
        assert "announcer_intro_rewritten" not in row["compose_flags"]


# ---------------------------------------------------------------------------
# The rewrite FAILURE POSTURE, pinned over the shipped source
# ---------------------------------------------------------------------------

class TestRewriteFailureClassification:
    """WHY SOURCE-LEVEL. The rewrite decision lives inside ``run()``, which
    needs an outline, a cast lock, a slot scheduler and a live model lane to
    reach. A behavioural test would be so heavily mocked it would prove the
    mocks classify exceptions rather than that the NODE does. Reading the
    shipped source cannot be satisfied by a stub.

    WHAT IT GUARDS. The composer never returns empty text on failure -- it
    returns ``fallback_safe_open()``, a deterministic template. So "it returned
    something" was treated as success, and a provider hiccup silently replaced
    a real composed opening with a canned line and stamped it rewritten. The
    documented keep-the-in-loop-intro posture only ever fired on a RAISE.
    """

    #: Anchors bounding the rewrite decision in the shipped writer source.
    #: Sliced between two literals that only occur there, so the assertions
    #: cannot be satisfied by an unrelated part of a 6,000-line module.
    START = '_rw_reason = "compose_failed"'
    END = 'meta["announcer_intro_rewrite"]["reason"] = _rw_reason'

    def _run_source(self) -> str:
        """The rewrite block's EXECUTABLE lines -- comments stripped.

        Stripping matters. These are string-presence assertions over source, and
        this block is heavily commented with the very names being asserted, so
        matching raw text would let a deleted `except` clause keep passing on
        the strength of the comment that describes it. `ValueError` was
        demonstrably satisfied by a comment alone before this stripping was
        added (independent QA caught it).
        """
        src = (
            Path(__file__).resolve().parents[1]
            / "nodes" / "OTR_LedgerScriptWriter.py"
        ).read_text(encoding="utf-8")
        assert src.count(self.START) == 1, "rewrite-block start anchor moved"
        assert src.count(self.END) == 1, "rewrite-block end anchor moved"
        start = src.index(self.START)
        end = src.index(self.END) + len(self.END)
        block = src[start:end]
        return "\n".join(
            line for line in block.splitlines()
            if not line.strip().startswith("#")
        )

    def test_the_assertions_cannot_be_satisfied_by_comments_alone(self):
        """Guards the guard. If a future edit moves these names into prose,
        every assertion below silently stops testing anything."""
        src = (
            Path(__file__).resolve().parents[1]
            / "nodes" / "OTR_LedgerScriptWriter.py"
        ).read_text(encoding="utf-8")
        start = src.index(self.START)
        end = src.index(self.END) + len(self.END)
        comments = "\n".join(
            line for line in src[start:end].splitlines()
            if line.strip().startswith("#")
        )
        code = self._run_source()
        for name in (
            "AnnouncerBriefStarvedError", "StructuredCallFailedError",
            "ValueError", "uncaught_", "exc_info=True",
            "announcer_intro_structural_fallback",
        ):
            assert name in code, f"{name} is not in executable code"

    def test_a_structural_fallback_is_not_a_rewrite(self):
        block = self._run_source()
        assert "announcer_intro_structural_fallback" in block, (
            "the rewrite block must inspect the returned flags -- otherwise a "
            "template return is stamped as a successful rewrite")

    def test_every_failure_class_is_named(self):
        """A deliberate derive outcome must never be filed as a coding defect,
        and a coding defect must never hide as routine starvation."""
        block = self._run_source()
        for expected in (
            "AnnouncerBriefStarvedError",   # typed starvation, bounded reason
            "StructuredCallFailedError",    # ladder exhaustion  -> derive_failed
            "ValueError",                   # no scene-1 rows    -> derive_failed
            "uncaught_",                    # genuinely unexpected
            "exc_info=True",                # ...and it gets a traceback
        ):
            assert expected in block, f"the ladder never mentions {expected}"

    def test_the_reason_is_stamped_durably(self):
        block = self._run_source()
        assert "_rw_reason" in block
        assert '"reason"' in block, (
            "the receipt must be durable -- a log line is gone by the time an "
            "audit asks why the rewrite did not happen")


class TestInLoopSafeOpenTelemetry:
    """`meta["open_safe_fallback"]` must test the flag the composer EMITS.

    It tested for the literal string "open_safe_fallback", which no composer
    has ever produced, so the receipt read False on every episode -- including
    the ones whose safe-open really did fall back to a template. Same defect
    class as the `hook` attribute and `news_coda_fallback`: a receipt keyed on
    a producer string the producer never emits, which fails silently in the
    safe direction and therefore never gets noticed.

    Source-level for the same reason as the class above: the in-loop block sits
    inside `run()`'s per-beat loop.
    """

    def test_the_receipt_tests_a_flag_the_composer_actually_emits(self):
        composer = (
            Path(__file__).resolve().parents[1]
            / "nodes" / "_otr_line_composer.py"
        ).read_text(encoding="utf-8")
        writer = (
            Path(__file__).resolve().parents[1]
            / "nodes" / "OTR_LedgerScriptWriter.py"
        ).read_text(encoding="utf-8")

        start = writer.index('meta["open_safe_fallback"]')
        stamp = writer[start:writer.index(")", start) + 1]

        assert '"announcer_intro_structural_fallback"' in stamp, (
            "open_safe_fallback must key on the composer's real degrade flag")
        assert '"open_safe_fallback" in' not in stamp, (
            "the receipt is keyed on its own name -- a string no composer emits")

        # And that flag has to be real on the producer side, or this test just
        # swaps one phantom string for another.
        assert (
            'LineResult(fallback, ("announcer_intro_structural_fallback",))'
            in composer
        ), "the composer no longer emits the flag the receipt now tests for"


# ---------------------------------------------------------------------------
# run() ORDER PIN + D4 title-regen root-cause fix (source-level)
# ---------------------------------------------------------------------------

class TestRunWiringPins:
    def test_rewrite_runs_before_outro_pass_and_aggregates(self):
        src = inspect.getsource(W.OTR_LedgerScriptWriter.run)
        i_rewrite = src.index("I.4.9")
        i_outro = src.index("--- I.5")
        i_aggregates = src.index("compose_flag_summary")
        assert i_rewrite < i_outro < i_aggregates

    def test_rewrite_guard_mirrors_outro_guard(self):
        src = inspect.getsource(W.OTR_LedgerScriptWriter.run)
        block = src[src.index("I.4.9"):src.index("--- I.5")]
        assert "first_announcer_id != last_announcer_id" in block
        assert "derive_produced_open_brief" in block
        assert "story_scaffold=True" in block
        assert "_apply_intro_rewrite_result" in block

    def test_title_regen_assembles_from_ledger(self):
        # scifi_news_pro S1a (2026-07-10): the J.5 title-regen block moved
        # from run() into the extracted _run_writer_tail (tail
        # extraction, architecture doc sections 11/13) -- the positive
        # pin follows it; the stale-join negative pin covers BOTH
        # methods so the regression cannot re-enter either.
        tail_src = inspect.getsource(
            W.OTR_LedgerScriptWriter._run_writer_tail)
        run_src = inspect.getsource(W.OTR_LedgerScriptWriter.run)
        assert (
            "assembled_script = _PL.assemble_script_text_from_ledger(led.data)"
            in tail_src
        )
        # The stale in-flight join must be gone from the title path.
        for src in (tail_src, run_src):
            assert (
                'assembled_script = "\\n\\n".join(script_text_parts)'
                not in src
            )
