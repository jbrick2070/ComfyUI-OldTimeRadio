"""tests/test_announcer_passes.py -- unit tests for the announcer
dedicated passes (2026-05-22, BUG-LOCAL-255).

Targets ``nodes/_otr_line_composer.py``:
  - clean_one_line
  - validate_announcer_line
  - fallback_announcer_intro / fallback_announcer_outro
  - compose_announcer_intro / compose_announcer_outro

Plus the writer-side wiring in ``nodes/OTR_LedgerScriptWriter.py``
and the retirement of ``override_announcer_close`` from
``nodes/_otr_news_wiring.py``.

Hermetic: no GPU, no I/O, no real LLM. `_otr_line_composer` and
`_otr_ledger` are stdlib-only at import time; the announcer composers
take a `creative_fn` callable that the tests mock.

Design history: docs/2026-05-22-announcer-dedicated-passes__04_synthesis.md
+ docs/2026-05-22-announcer-passes-build-handoff.md. Bug: BUG-LOCAL-255.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Same sys.path setup as the sibling tests so ``nodes`` modules import
# by bare name.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _otr_ledger as _LED  # noqa: E402
import _otr_line_composer as _LC  # noqa: E402


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_creative_fn(reply, *, accept_stop=True):
    """Build a mock ``creative_fn``.

    ``reply`` is returned verbatim from each call; if it is an
    exception instance it is raised instead. When ``accept_stop`` is
    False the fn raises ``TypeError`` on the ``stop=`` kwarg, which
    exercises the composer's no-``stop`` fallback path.

    The returned callable carries a ``.calls`` list -- one dict per
    invocation, recording the messages and sampling kwargs.
    """
    calls: list[dict] = []

    def _fn(messages, *, temperature=None, max_new_tokens=None, **kw):
        calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": kw.get("stop", "ABSENT"),
        })
        if "stop" in kw and not accept_stop:
            raise TypeError("this loader does not accept stop=")
        if isinstance(reply, BaseException):
            raise reply
        return reply

    _fn.calls = calls
    return _fn


def _ledger_row(line_id: str, speaker_role: str, text: str) -> dict:
    """A ledger ``lines[]`` row -- the post-init shape that carries the
    PUBLIC ``speaker_role`` key (not the writer's in-flight private
    ``_speaker_role``). This shape is exactly why BUG-LOCAL-255's
    ``override_announcer_close`` silently matched nothing.
    """
    return {
        "line_id": line_id,
        "speaker_role": speaker_role,
        "text": text,
        "char_count": len(text),
        "word_count": len(text.split()),
    }


# ---------------------------------------------------------------------------
# clean_one_line
# ---------------------------------------------------------------------------


def test_clean_one_line_collapses_whitespace_and_newlines():
    assert _LC.clean_one_line("  a\n\n  b\t c  ", max_chars=0) == "a b c"


def test_clean_one_line_strips_wrapping_quotes():
    assert _LC.clean_one_line('"hello there"', max_chars=0) == "hello there"
    assert _LC.clean_one_line("'hello there'", max_chars=0) == "hello there"
    # Smart quotes round-trip to a clean inner string.
    assert _LC.clean_one_line(
        "“hello there”", max_chars=0,
    ) == "hello there"


def test_clean_one_line_empty_input_returns_empty():
    assert _LC.clean_one_line("", max_chars=20) == ""
    assert _LC.clean_one_line(None, max_chars=20) == ""


def test_clean_one_line_truncates_on_word_boundary():
    out = _LC.clean_one_line("one two three four five six", max_chars=12)
    assert len(out) <= 13          # cap + a possible re-terminating period
    assert " " not in out[-1]      # no trailing partial-word space
    assert out.endswith(".")       # re-terminated after the cut
    assert out.startswith("one two")


def test_clean_one_line_no_truncation_when_max_chars_non_positive():
    long = "word " * 40
    out = _LC.clean_one_line(long, max_chars=0)
    assert out == ("word " * 40).strip()


# ---------------------------------------------------------------------------
# validate_announcer_line
# ---------------------------------------------------------------------------


def test_validate_announcer_line_accepts_a_clean_line():
    ok, cleaned = _LC.validate_announcer_line(
        "Good evening, and welcome once more to tonight's broadcast.",
        min_chars=24, max_chars=300,
    )
    assert ok is True
    assert cleaned == (
        "Good evening, and welcome once more to tonight's broadcast."
    )


def test_validate_announcer_line_rejects_empty():
    ok, cleaned = _LC.validate_announcer_line(
        "   ", min_chars=5, max_chars=300,
    )
    assert ok is False
    assert cleaned == ""


def test_validate_announcer_line_rejects_multi_line():
    ok, cleaned = _LC.validate_announcer_line(
        "Line one of the read.\nLine two of the read.",
        min_chars=5, max_chars=300,
    )
    assert ok is False
    assert cleaned == ""


def test_validate_announcer_line_rejects_speaker_label_prefix():
    for bad in (
        "ANNOUNCER: Good evening to you all out there tonight.",
        "HOST: Good evening to you all out there tonight.",
        "NARRATOR: Good evening to you all out there tonight.",
    ):
        ok, cleaned = _LC.validate_announcer_line(
            bad, min_chars=10, max_chars=300,
        )
        assert ok is False, bad
        assert cleaned == ""


def test_validate_announcer_line_rejects_bracket_and_brace_directions():
    ok, _ = _LC.validate_announcer_line(
        "Good evening [music swells] and welcome to the broadcast.",
        min_chars=10, max_chars=300,
    )
    assert ok is False
    ok, _ = _LC.validate_announcer_line(
        "Good evening {cue} and welcome to tonight's broadcast.",
        min_chars=10, max_chars=300,
    )
    assert ok is False


def test_validate_announcer_line_rejects_out_of_band_length():
    # Too short.
    ok, _ = _LC.validate_announcer_line(
        "Good night.", min_chars=24, max_chars=300,
    )
    assert ok is False
    # Too long.
    ok, _ = _LC.validate_announcer_line(
        "word " * 80, min_chars=24, max_chars=120,
    )
    assert ok is False


# ---------------------------------------------------------------------------
# Deterministic fallbacks
# ---------------------------------------------------------------------------


def test_fallback_announcer_intro_is_deterministic():
    a = _LC.fallback_announcer_intro("aliens land quietly in rural Ohio")
    b = _LC.fallback_announcer_intro("aliens land quietly in rural Ohio")
    assert a == b
    assert "SIGNAL LOST" in a
    assert "aliens land quietly in rural Ohio" in a


def test_fallback_announcer_intro_empty_brief_uses_default_frame():
    out = _LC.fallback_announcer_intro("")
    assert "SIGNAL LOST" in out
    assert out == _LC.fallback_announcer_intro("   ")


def test_fallback_announcer_outro_is_deterministic():
    a = _LC.fallback_announcer_outro("the signal repeats on a fixed cycle")
    b = _LC.fallback_announcer_outro("the signal repeats on a fixed cycle")
    assert a == b
    assert "SIGNAL LOST" in a
    assert "Good night" in a
    assert "the signal repeats on a fixed cycle" in a


def test_fallback_announcer_outro_empty_brief_uses_default_frame():
    out = _LC.fallback_announcer_outro("")
    assert "SIGNAL LOST" in out
    assert "Good night" in out


# ---------------------------------------------------------------------------
# compose_announcer_intro
# ---------------------------------------------------------------------------


_INTRO_BRIEF = "A quiet town starts hearing a sound no instrument can place."
_GOOD_INTRO = (
    "Tonight, a quiet town hears a sound that no instrument can place, "
    "and one reporter sets out to find its source."
)
_GOOD_OUTRO = (
    "The sound has a rhythm now, and the observatories keep their watch "
    "while the town waits for the next signal."
)


def test_compose_announcer_intro_happy_path():
    fn = _make_creative_fn(_GOOD_INTRO)
    res = _LC.compose_announcer_intro(
        creative_fn=fn, script_brief=_INTRO_BRIEF,
    )
    assert isinstance(res, _LC.LineResult)
    assert res.text == _GOOD_INTRO
    assert res.compose_flags == ("announcer_intro",)
    assert isinstance(res.compose_flags, tuple)
    # The creative model was called once and the brief reached it.
    assert len(fn.calls) == 1
    user_msg = fn.calls[0]["messages"][-1]["content"]
    assert _INTRO_BRIEF in user_msg


def test_compose_announcer_intro_empty_brief_skips_llm():
    fn = _make_creative_fn(_GOOD_INTRO)
    res = _LC.compose_announcer_intro(creative_fn=fn, script_brief="")
    assert res.compose_flags == ("announcer_intro_fallback",)
    assert "SIGNAL LOST" in res.text
    assert fn.calls == [], "no LLM call should fire with an empty brief"


def test_compose_announcer_intro_llm_raises_falls_back():
    fn = _make_creative_fn(RuntimeError("backend exploded"))
    res = _LC.compose_announcer_intro(
        creative_fn=fn, script_brief=_INTRO_BRIEF,
    )
    assert res.compose_flags == ("announcer_intro_fallback",)
    assert "SIGNAL LOST" in res.text
    # The brief still seeds the deterministic fallback.
    assert _INTRO_BRIEF.rstrip(".") in res.text


def test_compose_announcer_intro_invalid_output_falls_back():
    # Too short to clear the intro min-char band -> fallback.
    fn = _make_creative_fn("Hello.")
    res = _LC.compose_announcer_intro(
        creative_fn=fn, script_brief=_INTRO_BRIEF,
    )
    assert res.compose_flags == ("announcer_intro_fallback",)
    assert "SIGNAL LOST" in res.text


def test_compose_announcer_intro_handles_no_stop_loader():
    # A loader that rejects stop= must still succeed via the no-stop
    # fallback call -- mirrors compose_line's convention.
    fn = _make_creative_fn(_GOOD_INTRO, accept_stop=False)
    res = _LC.compose_announcer_intro(
        creative_fn=fn, script_brief=_INTRO_BRIEF,
    )
    assert res.compose_flags == ("announcer_intro",)
    assert res.text == _GOOD_INTRO
    # Two entries: the stop= attempt (raised) + the no-stop retry.
    assert len(fn.calls) == 2
    assert fn.calls[0]["stop"] != "ABSENT"
    assert fn.calls[1]["stop"] == "ABSENT"


def test_compose_announcer_intro_is_deterministic():
    fn1 = _make_creative_fn(_GOOD_INTRO)
    fn2 = _make_creative_fn(_GOOD_INTRO)
    r1 = _LC.compose_announcer_intro(creative_fn=fn1, script_brief=_INTRO_BRIEF)
    r2 = _LC.compose_announcer_intro(creative_fn=fn2, script_brief=_INTRO_BRIEF)
    assert r1 == r2


# ---------------------------------------------------------------------------
# compose_announcer_outro
# ---------------------------------------------------------------------------


_CLOSE_BRIEF = (
    "Independent observatories now track the signal and confirm it "
    "repeats on a fixed cycle."
)


def test_compose_announcer_outro_happy_path():
    fn = _make_creative_fn(_GOOD_OUTRO)
    res = _LC.compose_announcer_outro(
        creative_fn=fn,
        script_brief=_INTRO_BRIEF,
        news_close_brief=_CLOSE_BRIEF,
        intro_text=_GOOD_INTRO,
    )
    assert res.text == _GOOD_OUTRO
    assert res.compose_flags == ("announcer_outro",)
    # script_brief + news_close_brief + intro_text all reach the prompt.
    user_msg = fn.calls[0]["messages"][-1]["content"]
    assert _CLOSE_BRIEF in user_msg
    assert _INTRO_BRIEF in user_msg
    assert _GOOD_INTRO in user_msg


def test_compose_announcer_outro_empty_briefs_skips_llm():
    fn = _make_creative_fn(_GOOD_OUTRO)
    res = _LC.compose_announcer_outro(
        creative_fn=fn,
        script_brief="",
        news_close_brief="",
        intro_text=_GOOD_INTRO,
    )
    assert res.compose_flags == ("announcer_outro_fallback",)
    assert "SIGNAL LOST" in res.text
    assert fn.calls == []


def test_compose_announcer_outro_llm_raises_falls_back():
    fn = _make_creative_fn(RuntimeError("backend exploded"))
    res = _LC.compose_announcer_outro(
        creative_fn=fn,
        script_brief=_INTRO_BRIEF,
        news_close_brief=_CLOSE_BRIEF,
        intro_text=_GOOD_INTRO,
    )
    assert res.compose_flags == ("announcer_outro_fallback",)
    assert "SIGNAL LOST" in res.text
    # The close brief seeds the deterministic fallback.
    assert _CLOSE_BRIEF.rstrip(".") in res.text


def test_compose_announcer_outro_multiline_output_falls_back():
    # A multi-line read is a framing failure for a one-line close;
    # validate_announcer_line rejects it and the fallback fires.
    fn = _make_creative_fn(
        "First line of the closing read.\n"
        "Second line of the closing read."
    )
    res = _LC.compose_announcer_outro(
        creative_fn=fn,
        script_brief=_INTRO_BRIEF,
        news_close_brief=_CLOSE_BRIEF,
        intro_text="",
    )
    assert res.compose_flags == ("announcer_outro_fallback",)
    assert "SIGNAL LOST" in res.text


def test_compose_announcer_outro_strips_leaked_label_then_validates():
    # strip_line_formatting removes a leaked leading speaker label, so
    # an otherwise-clean line still ships via the LLM path.
    fn = _make_creative_fn(
        "ANNOUNCER: And that is where tonight's broadcast leaves us."
    )
    res = _LC.compose_announcer_outro(
        creative_fn=fn,
        script_brief=_INTRO_BRIEF,
        news_close_brief=_CLOSE_BRIEF,
        intro_text="",
    )
    assert res.compose_flags == ("announcer_outro",)
    assert not res.text.upper().startswith("ANNOUNCER:")


def test_compose_announcer_outro_works_without_intro_text():
    fn = _make_creative_fn(_GOOD_OUTRO)
    res = _LC.compose_announcer_outro(
        creative_fn=fn,
        script_brief=_INTRO_BRIEF,
        news_close_brief=_CLOSE_BRIEF,
        intro_text="",
    )
    assert res.compose_flags == ("announcer_outro",)
    user_msg = fn.calls[0]["messages"][-1]["content"]
    assert "opening line was" not in user_msg


# ---------------------------------------------------------------------------
# BUG-LOCAL-255 regression -- the close brief reaches the closing line
# ---------------------------------------------------------------------------


def test_bug255_news_close_brief_reaches_closing_line():
    """BUG-LOCAL-255 regression.

    The retired ``override_announcer_close`` matched a private
    ``_speaker_role`` key that is absent from ``led.data["lines"]``
    rows (they carry the PUBLIC ``speaker_role``), so the
    ``news_close_brief`` was silently dropped every episode.

    The replacement -- ``compose_announcer_outro`` written to the
    known last-announcer beat_id via ``patch_line_text`` -- must land
    the closing read on the closing line. This test simulates
    OTR_LedgerScriptWriter's I.5 post-loop block (and the in-loop
    intro dispatch) on a ledger whose rows use the public key.
    """
    ledger = {"lines": [
        _ledger_row("b001", "announcer", "placeholder intro"),
        _ledger_row("b002", "character", "Did you hear that?"),
        _ledger_row("b003", "character", "I heard it too."),
        _ledger_row("b004", "announcer", "placeholder outro"),
    ]}
    # The ledger rows carry the public key -- the exact shape the old
    # helper's `_speaker_role` lookup missed.
    assert all("_speaker_role" not in r for r in ledger["lines"])

    # The writer derives the bookend ids from outline.beats; mirror
    # that off the ledger rows here.
    announcer_ids = [
        r["line_id"] for r in ledger["lines"]
        if r["speaker_role"] == "announcer"
    ]
    first_id, last_id = announcer_ids[0], announcer_ids[-1]
    assert (first_id, last_id) == ("b001", "b004")

    # --- in-loop: intro pass overwrites the first announcer row ---
    intro_fn = _make_creative_fn(_GOOD_INTRO)
    intro_res = _LC.compose_announcer_intro(
        creative_fn=intro_fn,
        script_brief=_INTRO_BRIEF,
    )
    assert intro_res.compose_flags == ("announcer_intro",)
    assert _LED.patch_line_text(ledger, first_id, intro_res.text)
    assert ledger["lines"][0]["text"] == _GOOD_INTRO
    assert ledger["lines"][0]["text"] != "placeholder intro"

    # --- post-loop: outro pass overwrites the last announcer row ---
    outro_fn = _make_creative_fn(_GOOD_OUTRO)
    intro_text = ""
    for ln in ledger["lines"]:
        if ln["line_id"] == first_id:
            intro_text = ln["text"]
            break
    outro_res = _LC.compose_announcer_outro(
        creative_fn=outro_fn,
        script_brief=_INTRO_BRIEF,
        news_close_brief=_CLOSE_BRIEF,
        intro_text=intro_text,
    )
    assert outro_res.compose_flags == ("announcer_outro",)
    ok = _LED.patch_line_text(ledger, last_id, outro_res.text)
    assert ok, "patch_line_text must match the closing line by line_id"

    closing = ledger["lines"][-1]
    assert closing["text"] == _GOOD_OUTRO
    assert closing["text"] != "placeholder outro"
    # char_count + word_count recomputed in lockstep.
    assert closing["char_count"] == len(_GOOD_OUTRO)
    assert closing["word_count"] > 0
    # The news_close_brief reached the outro pass prompt -- it is no
    # longer silently dropped.
    outro_user_msg = outro_fn.calls[0]["messages"][-1]["content"]
    assert _CLOSE_BRIEF in outro_user_msg


def test_bug255_outro_fallback_still_lands_on_closing_line():
    """Even when the outro LLM pass fails, a deterministic close must
    land on the closing line -- the narrative bookend is never missing.
    """
    ledger = {"lines": [
        _ledger_row("b001", "announcer", "intro text here"),
        _ledger_row("b002", "character", "A line of dialogue."),
        _ledger_row("b003", "announcer", "placeholder outro"),
    ]}
    outro_fn = _make_creative_fn(RuntimeError("backend down"))
    outro_res = _LC.compose_announcer_outro(
        creative_fn=outro_fn,
        script_brief=_INTRO_BRIEF,
        news_close_brief=_CLOSE_BRIEF,
        intro_text="intro text here",
    )
    assert outro_res.compose_flags == ("announcer_outro_fallback",)
    assert _LED.patch_line_text(ledger, "b003", outro_res.text)
    closing = ledger["lines"][-1]
    assert "SIGNAL LOST" in closing["text"]
    assert closing["text"] != "placeholder outro"


# ---------------------------------------------------------------------------
# Source-assertion guards -- writer wiring + override retirement
# ---------------------------------------------------------------------------


def _read_node_src(rel: str) -> str:
    return (_NODES_DIR / rel).read_text(encoding="utf-8")


def test_writer_wires_announcer_passes():
    src = _read_node_src("OTR_LedgerScriptWriter.py")
    assert "compose_announcer_intro" in src
    assert "compose_announcer_outro" in src
    assert "first_announcer_id" in src
    assert "last_announcer_id" in src
    # The broken override_announcer_close call is gone.
    assert "_OTRNW.override_announcer_close" not in src


def test_override_announcer_close_retired_from_news_wiring():
    src = _read_node_src("_otr_news_wiring.py")
    assert "def override_announcer_close" not in src
    # The surviving helper stays.
    assert "def post_assembly_keyterm_check" in src


def test_announcer_passes_are_in_public_surface():
    for name in (
        "clean_one_line",
        "validate_announcer_line",
        "fallback_announcer_intro",
        "fallback_announcer_outro",
        "compose_announcer_intro",
        "compose_announcer_outro",
    ):
        assert name in _LC.__all__, name
        assert hasattr(_LC, name), name
