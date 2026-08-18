"""The script-writer must never see the real factual close in its own context.

FOUND BY KIBITZ r3 (antigravity, 2026-08-18, grounded and confirmed against the
real files before landing). `run_scifi_news_pro_episode` runs `_pass_news_read`
BEFORE `_pass_script` and copies its result onto `treatment.news_close_read`
(`_otr_scifi_news_pro.py:3616-3618`) -- so by the time `_script_user_prompt`
builds the script writer's prompt, the treatment it dumps already carries the
finished real-world fact. `scifi_news_pro_script_system` tells the writer
"never state a real-world fact in the outro or CODA, the producer appends the
real read after you" -- while the very prompt making that claim also showed the
model the real read verbatim. A small local model does not reliably resist a
fact sitting in its own context just because an instruction says not to repeat
it; a live render (`signal_lost_the_recession_of_room_4_20260818_112039`)
produced an outro that paraphrased the eventual close's own facts before the
CODA even ran.

The fix excludes `news_close_read` from the ONE `model_dump()` this function
uses; every other consumer of the field (ledger assembly, receipts) is
untouched.
"""
from __future__ import annotations

from nodes import _otr_scifi_news_pro as PRO


def _treatment(news_close_read=""):
    return PRO.Treatment(
        title="T", dramatic_question="Q", setting="S",
        cast_shapes=[PRO.CastShape(name="Ada", role="r", want="w",
                                   pressure="p", register="clipped")],
        turn="turn", priced_ending={"choice": "c", "cost_paid": "p"},
        news_thread="thread", news_close_read=news_close_read,
    )


#: A distinctive marker no other prompt content could accidentally contain.
_MARKER = "RESEARCHERS AT A DISTINCTIVE FICTIONAL INSTITUTE FOUND THE THING"


def _prompt(news_close_read):
    treatment = _treatment(news_close_read)
    envelope = PRO._build_envelope(1)
    return PRO._script_user_prompt(treatment, "digest text", envelope, ["Ada"])


def test_the_real_factual_close_never_reaches_the_script_writer():
    """THE FIX. A populated `news_close_read` must not appear anywhere in the
    prompt that asks the model to write the fiction, outro, and CODA."""
    prompt = _prompt(_MARKER)

    assert _MARKER not in prompt


def test_an_empty_read_still_produces_a_normal_prompt():
    """The exclusion must not be load-bearing for the empty case -- an
    episode whose news_read pass has not run yet (or produced nothing) keys
    identically to before this fix."""
    prompt = _prompt("")

    assert "TREATMENT:" in prompt
    assert "CAST (" in prompt


def test_every_other_treatment_field_still_reaches_the_prompt():
    """The exclusion is scoped to ONE field. Cast shapes and the dramatic
    question -- what the model actually needs to write the fiction -- must
    still be present."""
    prompt = _prompt(_MARKER)

    assert "dramatic_question" in prompt
    assert "Q" in prompt
    assert "Ada" in prompt


def test_the_format_reminder_bars_real_world_content_from_outro_and_coda():
    """The SHOULD-FIX half: the format reminder the model reads right before
    writing restates the constraint at the point of use, not just once at the
    top of the system prompt."""
    prompt = _prompt("")

    assert "strictly in-story" in prompt
    assert "CODA" in prompt
