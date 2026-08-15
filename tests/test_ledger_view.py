"""tests/test_ledger_view.py -- the ledger inspector.

`scripts/otr_ledger_view.py` is the operator's eyeball on a ledger: read one
fast, grade it on the only two things that count, and -- with `--watch` --
see a run while it is still writing so a decode cycling the same paragraph
does not look like a decode that is working.

What is pinned here:

  * it grades F1 (action in a spoken row) and F2 (speaker mismatch) and
    NOTHING else. A merely dull line is not a finding; that is the standing
    "story quality is done" law expressed as a test.
  * every finding names the phrase it tripped on, so a human can overrule it.
  * the coda check: present, last, spoken by the announcer, names a source.
  * the LIVE repetition tell, which is the whole reason the window exists:
    a cycle drifts across heartbeat samples, so no two samples match and the
    same phrase still keeps coming back. A detector that only compares whole
    samples is blind to exactly the case that matters.
  * it never writes to a ledger.

Pure-Python. No GPU. No LLM.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import otr_ledger_view as view  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ledger(rows, *, beats=None, cast=None, anchors=None):
    beats = beats if beats is not None else [
        {"beat_id": row.get("beat_id"), "speaker": row.get("speaker")}
        for row in rows if row.get("beat_id")
    ]
    return {
        "episode_id": "probe_ep",
        "cast": cast if cast is not None else [
            {"char_id": "c01", "name": "Ada"},
            {"char_id": "announcer", "name": "ANNOUNCER"},
        ],
        "beats": beats,
        "lines": rows,
        "meta": {
            "source_bank": "scifi_news",
            "scifi_codex": {
                "fact_index": {
                    "entities": [{"name": n} for n in (anchors or ["MIT"])],
                    "numbers": [{"verbatim": "60 percent"}],
                },
            },
        },
    }


def _row(line_id, text, *, speaker="Ada", role="character", char_id="c01",
         flags=()):
    return {
        "line_id": line_id, "beat_id": f"b_{line_id}", "speaker": speaker,
        "char_id": char_id, "speaker_role": role, "text": text,
        "word_count": len(text.split()), "compose_flags": list(flags),
    }


def _write(tmp_path: Path, data: dict, episode="probe_ep") -> Path:
    audio = tmp_path / episode / "audio"
    audio.mkdir(parents=True, exist_ok=True)
    path = audio / f"{episode}_ledger.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1. F1 -- action in a spoken row
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text,kind", [
    ("Ada turns to Leo. We have to try.", "third-person action"),
    ("She stared at the console. We have to try.", "third-person action"),
    ("“We have to try,” Ada murmured.", "attribution verb"),
    ("(softly) We have to try.", "stage direction in brackets"),
    ("Her voice trembles. We have to try.", "delivery note"),
    ("ADA: We have to try.", "speaker label"),
    ("He said “we have to try” and the room went cold.",
     "quoted speech inside prose"),
])
def test_narration_inside_a_spoken_row_is_found(tmp_path, text, kind):
    path = _write(tmp_path, _ledger([_row("l001", text)]))
    report = view.grade(path)
    assert {f.kind for f in report.f1} >= {kind}
    assert report.clean is False


@pytest.mark.parametrize("text", [
    # Every one of these is real dialogue from the first episode the per-beat
    # schedule produced, and the first cut of the detector flagged all of
    # them. Six false alarms out of six findings is not a noisy detector, it
    # is a broken one: an operator who has to overrule it every time stops
    # reading it.
    "We stand at the precipice of the red frontier.",
    "If my cells pay the price for humanity to finally stand on red soil.",
    "Then step back, Kaelen. Let the fear stay in the airlock with you.",
    "The vest stands as our greatest shield against the sun's fury.",
    "It’s designed to block the solar storms we can predict.",
    "It’s not just a number on a readout, Commander.",
])
def test_ordinary_dialogue_is_not_mistaken_for_stage_business(tmp_path, text):
    path = _write(tmp_path, _ledger([_row("l001", text)]))
    report = view.grade(path)
    assert report.f1 == [], f"false positive on: {text}"


def test_a_finding_names_the_phrase_it_tripped_on(tmp_path):
    """A human has to be able to overrule the detector at a glance."""
    path = _write(tmp_path, _ledger([
        _row("l001", "Ada turns to the console. We have to try."),
    ]))
    report = view.grade(path)
    action = next(f for f in report.f1 if f.kind == "third-person action")
    assert action.detail == "Ada turns"
    assert action.line_id == "l001"


def test_plain_speech_is_not_a_finding(tmp_path):
    path = _write(tmp_path, _ledger([
        _row("l001", "We have to try. There is no second sample."),
        _row("l002", "Then we try tonight, and we log every reading."),
    ]))
    report = view.grade(path)
    assert report.f1 == []
    assert report.clean is True


def test_a_dull_line_is_not_a_defect(tmp_path):
    """Story quality is not graded (operator directive 2026-08-04).

    Flat, repetitive, uninteresting dialogue is a PASS here. Only action and
    misattribution fail.
    """
    path = _write(tmp_path, _ledger([
        _row("l001", "Yes."),
        _row("l002", "I agree."),
        _row("l003", "I agree as well."),
    ]))
    report = view.grade(path)
    assert report.clean is True
    assert "CLEAN" in view._verdict_line(report)


def test_music_rows_are_never_graded(tmp_path):
    path = _write(tmp_path, _ledger([
        _row("l001", "We have to try."),
        {"line_id": "m1", "speaker_role": "music_close", "text": "",
         "compose_flags": []},
    ]))
    report = view.grade(path)
    assert report.clean is True
    assert len(report.voiced) == 1


# ---------------------------------------------------------------------------
# 2. F2 -- who is speaking
# ---------------------------------------------------------------------------

def test_a_row_with_no_speaker_is_found(tmp_path):
    """The exact shape of PBUG-20260814-01 on a published ledger."""
    path = _write(tmp_path, _ledger([_row("l001", "We have to try.",
                                          speaker=None)]))
    report = view.grade(path)
    assert [f.kind for f in report.f2] == ["no speaker"]


def test_a_row_that_disagrees_with_its_beat_is_found(tmp_path):
    rows = [_row("l001", "We have to try.", speaker="Leo")]
    data = _ledger(rows, beats=[{"beat_id": "b_l001", "speaker": "Ada"}])
    report = view.grade(_write(tmp_path, data))
    assert [f.kind for f in report.f2] == ["row disagrees with its beat"]
    assert "'Leo'" in report.f2[0].detail


def test_a_row_that_disagrees_with_the_cast_is_found(tmp_path):
    rows = [_row("l001", "We have to try.", speaker="Leo")]
    data = _ledger(rows, beats=[{"beat_id": "b_l001", "speaker": ""}])
    report = view.grade(_write(tmp_path, data))
    assert [f.kind for f in report.f2] == ["row disagrees with the cast"]


# ---------------------------------------------------------------------------
# 3. The coda
# ---------------------------------------------------------------------------

def test_a_coda_that_names_its_source_reads_clean(tmp_path):
    rows = [
        _row("l001", "We have to try."),
        _row("l900", "Tonight's story began with a real MIT report.",
             speaker="ANNOUNCER", role="announcer", char_id="announcer",
             flags=("news_coda",)),
    ]
    report = view.grade(_write(tmp_path, _ledger(rows)))
    assert report.coda["row_count"] == 1
    assert report.coda["is_last"] is True
    assert report.coda["announcer"] is True
    assert report.coda["names"] == "MIT"


def test_a_coda_that_names_nothing_is_visible(tmp_path):
    rows = [
        _row("l001", "We have to try."),
        _row("l900", "And that was tonight's story.", speaker="ANNOUNCER",
             role="announcer", char_id="announcer", flags=("news_coda",)),
    ]
    report = view.grade(_write(tmp_path, _ledger(rows)))
    assert report.coda["names"] == ""


def test_a_missing_coda_is_visible(tmp_path):
    report = view.grade(_write(tmp_path, _ledger([
        _row("l001", "We have to try."),
    ])))
    assert report.coda["row_count"] == 0
    assert "NO CODA ROW" in view.render_text(report)


def test_anchors_drop_short_names_and_deduplicate():
    data = _ledger([], anchors=["MIT", "AI", "MIT", "starfish"])
    assert view.source_anchors(data) == ["MIT", "starfish", "60 percent"]


def test_anchor_matching_is_word_boundary():
    assert view.names_anchor("the signal was transmitted", ["MIT"]) == ""
    assert view.names_anchor("reported by mit tonight", ["MIT"]) == "MIT"


# ---------------------------------------------------------------------------
# 4. The live window
# ---------------------------------------------------------------------------

def test_the_repeat_tell_survives_a_drifting_window():
    """THE case the window exists for, taken from a real run.

    Each heartbeat shows a different slice of the same repeating paragraph,
    so no two samples are equal -- a whole-sample comparison sees nothing
    while a human can read the loop straight off the screen.
    """
    tails = [
        "e the chronicler of the first step. They are the voice of the archive.",
        "e the voice of the archive. They are the chronicler of the first step.",
        "the chronicler of the first step. They are the voice of the archive. T",
    ]
    note = view._repetition_note(tails)
    assert note
    assert "came back" in note
    assert "chronicler of the first step" in note or \
        "voice of the archive" in note


def test_identical_samples_are_called_out_plainly():
    note = view._repetition_note(["same tail here"] * 3)
    assert "IDENTICAL" in note


def test_text_that_is_actually_progressing_is_not_flagged():
    """A false alarm every few seconds would train the operator to ignore it."""
    tails = [
        "the vest cut the effective radiation dose for a solar particle event",
        "mobility was preserved for the crew during the Artemis I test flight",
        "StemRad reported the garment weighed under twenty six kilograms total",
    ]
    assert view._repetition_note(tails) == ""


def test_a_sliding_window_overlap_is_not_a_repeat():
    """Measured on the first live leg this window ever watched.

    Each heartbeat prints the tail of a GROWING string, so the end of one
    sample is legitimately the start of the next. Reading that as a repeat
    fires on every healthy run -- which is worse than not looking, because
    an alarm that is always on is an alarm nobody reads.
    """
    tails = [
        "hysically initiate the airlock sequence, forcing the final choice.",
        "lock sequence, forcing the final choice. \"voice_slot\": \"c03\" } ] }",
        "long-term health for a historical first step in the red dust. She",
    ]
    assert view._repetition_note(tails) == ""


def test_overlapping_samples_are_stitched_not_concatenated():
    stitched = view._merge_tails([
        "the airlock sequence, forcing the",
        "sequence, forcing the final choice.",
    ])
    assert stitched == "the airlock sequence, forcing the final choice."
    assert stitched.count("forcing the") == 1


def test_the_window_reads_pass_rate_and_halts_from_the_server_log(tmp_path):
    log = tmp_path / "server.log"
    log.write_text(
        "[INFO] [OTR_StructuredCall] 'scifi_codex:P2' attempt 3/3: base call\n"
        "[INFO] heartbeat: 256 tok | 5.8 tok/s | 44.0s | ...first slice\n"
        "[ERROR] DECODE HALTED (verbatim_cycle): the output repeated\n"
        "[INFO] heartbeat: 320 tok | 6.1 tok/s | 52.0s | ...second slice\n",
        encoding="utf-8",
    )
    state = view.read_live(log)
    assert state.pass_id == "scifi_codex:P2"
    assert state.attempt == "3/3"
    assert state.tokens == 320
    assert state.rate == pytest.approx(6.1)
    assert state.halts == ["verbatim_cycle"]
    assert state.tails[-1].endswith("second slice")


def test_the_window_survives_a_log_that_is_not_there(tmp_path):
    state = view.read_live(tmp_path / "absent.log")
    assert state.pass_id == ""
    assert state.tails == []


def test_the_window_can_be_a_page_that_refreshes_itself():
    """A terminal window means opening a terminal. This one you leave open."""
    page = view.render_live_html("pass scifi_codex:P3\n  tokens 960", 3.0)
    assert "http-equiv='refresh' content='3'" in page
    assert "pass scifi_codex:P3" in page
    assert "class=alarm" not in page


def test_the_page_goes_red_when_the_model_is_repeating():
    page = view.render_live_html("  !!  REPEATING  --  a phrase came back", 4.0)
    assert "class=alarm" in page


# ---------------------------------------------------------------------------
# 5. Finding a ledger, and the shapes it emits
# ---------------------------------------------------------------------------

def test_newest_episode_wins_and_a_fragment_resolves(tmp_path):
    _write(tmp_path, _ledger([_row("l001", "One.")]), episode="older_ep")
    newer = _write(tmp_path, _ledger([_row("l001", "Two.")]),
                   episode="newer_ep")
    import os
    import time
    os.utime(newer, (time.time() + 10, time.time() + 10))

    assert view.resolve(None, tmp_path) == newer
    assert view.resolve("older", tmp_path).parent.parent.name == "older_ep"
    with pytest.raises(SystemExit):
        view.resolve("no_such_episode", tmp_path)


def test_html_shows_the_text_and_highlights_the_defect(tmp_path):
    report = view.grade(_write(tmp_path, _ledger([
        _row("l001", "Ada turns to the console. We have to try."),
    ])))
    page = view.render_html([report])
    assert "<title>" in page
    assert "We have to try." in page
    assert "<mark>Ada turns</mark>" in page
    assert "dirty" in page


def test_json_carries_the_verdict(tmp_path):
    report = view.grade(_write(tmp_path, _ledger([
        _row("l001", "We have to try."),
    ])))
    payload = view.report_json(report)
    assert payload["clean"] is True
    assert payload["spoken_rows"] == 1
    assert payload["f1"] == [] and payload["f2"] == []


def test_exit_code_is_the_verdict(tmp_path, capsys):
    clean = tmp_path / "clean"
    _write(clean, _ledger([_row("l001", "We have to try.")]))
    assert view.main(["--episodes-root", str(clean), "--json"]) == 0

    dirty = tmp_path / "dirty"
    _write(dirty, _ledger([_row("l001", "Ada turns away. Fine.")]))
    assert view.main(["--episodes-root", str(dirty), "--json"]) == 1


@pytest.mark.parametrize("meta", [
    [],                                   # a list where an object belongs
    {"scifi_codex": []},                  # lane block is a list
    {"scifi_codex": {"call_journal": []}},
    {"scifi_codex": {"fact_index": "nope"}},
    {"scifi_codex": {"news_coda": 7}},
    "a string",
])
def test_a_malformed_meta_block_does_not_take_the_viewer_down(tmp_path, meta):
    """`--watch` reads a ledger a run is actively rewriting, and `--ladder`
    sweeps hundreds of files written by older code. A `.get()` on a value that
    turned out to be a list would kill the window on the very artifact it
    exists to watch."""
    data = _ledger([_row("l001", "We have to try.")])
    data["meta"] = meta
    path = _write(tmp_path, data)
    report = view.grade(path)
    assert len(report.voiced) == 1
    assert view.render_text(report)
    assert view.ladder_rows(path) == []


def test_a_ledger_whose_root_is_not_an_object_is_refused_clearly(tmp_path):
    path = tmp_path / "ep" / "audio" / "ep_ledger.json"
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="root is not an object"):
        view.grade(path)
    assert view.ladder_rows(path) == []


def test_the_viewer_never_writes_to_the_ledger(tmp_path):
    path = _write(tmp_path, _ledger([_row("l001", "Ada turns away.")]))
    before = path.read_bytes()
    view.grade(path)
    view.render_text(view.grade(path))
    view.render_html([view.grade(path)])
    assert path.read_bytes() == before
