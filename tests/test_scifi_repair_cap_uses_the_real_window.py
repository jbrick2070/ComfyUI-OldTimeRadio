"""The repair-turn budget must be the transport's window, not a flat constant.

THE HALF THAT WAS STILL OPEN (2026-09-11). The morning's fix made a repair-turn
prompt overflow survivable: when the transport refuses at the door, the ladder
drops the draft and goes on instead of killing the episode. That closed the
CRASH. It did not touch the arithmetic that produced the bad prediction.

`_draft_fits_repair_turn` estimated fit against `HARD_VRAM_CONTEXT_LIMIT` -- one
flat number, resolved from the VRAM budget of the machine the writer happens to
run on. The three transports do not share a window, and each had already
resolved its own and stamped it into its cache entry as `context_cap`:

  * local transformers -- from the tokenizer/model config
  * GGUF-native       -- llama.cpp's `n_ctx`, which the 8 GB profiles set well
                          below 8192
  * OpenRouter        -- the provider's advertised `context_window`, often 128k

So the flat cap was wrong in BOTH directions, and the two failures look nothing
alike:

  * OVERSTATEMENT is the expensive one. On a GGUF row loaded at n_ctx 4096 the
    predicate says a full-length draft fits, the draft rides, and the transport
    refuses. Before this morning that ended the episode; it now costs a wasted
    round trip and a cold regeneration -- the exact outcome the repair turn was
    built to avoid.
  * UNDERSTATEMENT is quiet. Against a 128k remote window the predicate refuses
    drafts with room to spare, and every one of those refusals silently reverts
    this lane to cold regeneration for nothing at all.

The value was always available. This proves it is now the one that decides.

THE DRAFT LENGTHS ARE DERIVED, NOT HARDCODED -- each is solved from the module's
own constants, because the FIRST test written for this predicate asserted on a
400-character toy play and passed for the entire life of the bug it was named
for (2026-08-12). A test that cannot notice a constant moving is not covering
the arithmetic, it is covering a coincidence. The prompt sizes and the two
transport windows below ARE fixtures: 4096 and 131072 are representative of a
real GGUF n_ctx and a real remote window, not measurements taken from a run.
"""
from __future__ import annotations

import pytest

from nodes import _otr_model_catalog as catalog
from nodes import _otr_scifi_news_pro as scifi_news_pro
from nodes.OTR_LedgerScriptWriter import _SlotScheduler
from nodes._otr_scifi_news_pro_markup import ANNOUNCER_NAME


FLAT_CAP = int(catalog.HARD_VRAM_CONTEXT_LIMIT)

#: A maxed digest and a system prompt, in the shape the live lane sends them.
BASE = "b" * 6100
SYSTEM = "s" * 200

#: A real GGUF `n_ctx` from the 8 GB profiles, and a real OpenRouter window.
GGUF_N_CTX = 4096
REMOTE_WINDOW = 131072


def draft_costing(tokens: float) -> str:
    """A draft whose whole repair turn costs `tokens`, per the module's math.

    Inverts `_draft_fits_repair_turn`'s own arithmetic:
        needed = H/C * (base + overhead + draft * (1 + M))
    so a change to any of the three constants moves these drafts with it
    instead of leaving the assertions pinned to stale lengths.
    """
    overhead = len(scifi_news_pro._FABLE2_FORMAT_EXAMPLE) + len(SYSTEM)
    chars = (tokens * scifi_news_pro._CHARS_PER_TOKEN
             / scifi_news_pro._REPAIR_TURN_HEADROOM)
    length = int((chars - len(BASE) - overhead)
                 / (1.0 + scifi_news_pro._REPAIR_REPLY_MARGIN))
    assert length > 0, (
        "the solver produced a non-positive draft length -- the constants have "
        "moved far enough that this test's premise no longer holds"
    )
    return "x" * length


def fits(draft: str, **kwargs) -> bool:
    return scifi_news_pro._draft_fits_repair_turn(
        BASE, draft, SYSTEM, **kwargs)


# ---------------------------------------------------------------------------
# 1. The predicate reads the window it was given
# ---------------------------------------------------------------------------
def test_the_flat_cap_OVERSTATES_a_small_gguf_window():
    """The crash direction, and the reason this half was worth finishing.

    A full-length draft fits the flat 8192-shaped cap and does NOT fit the
    4096-token window an 8 GB GGUF row actually loads with. Under the old code
    it rode anyway and the transport refused at the door."""
    draft = draft_costing(FLAT_CAP * 0.9)
    assert fits(draft) is True, (
        "premise broken: this draft is supposed to pass the flat cap")
    assert fits(draft, cap=GGUF_N_CTX) is False, (
        "a draft that cannot fit llama.cpp's n_ctx was still declared to fit")


def test_the_flat_cap_UNDERSTATES_a_large_remote_window():
    """The quiet direction. Every needless refusal here is a cold regeneration
    bought for nothing."""
    draft = draft_costing(FLAT_CAP * 1.6)
    assert fits(draft) is False, (
        "premise broken: this draft is supposed to fail the flat cap")
    assert fits(draft, cap=REMOTE_WINDOW) is True, (
        "a draft with 100k tokens of room to spare was refused")


@pytest.mark.parametrize("unresolved", [None, 0, -1])
def test_an_unresolved_cap_is_NOT_a_refusal(unresolved):
    """0 means "no answer", never "no room".

    A transport that cannot report its window must leave the lane exactly where
    it was -- on the flat estimate -- rather than reading the missing value as
    a window of zero, which would refuse every draft ever offered."""
    for tokens in (FLAT_CAP * 0.9, FLAT_CAP * 1.6):
        draft = draft_costing(tokens)
        assert fits(draft, cap=unresolved) == fits(draft), (
            "cap=%r changed the verdict; it must fall back to the flat cap"
            % (unresolved,))


def test_an_empty_draft_is_still_refused_whatever_the_window_says():
    """There is nothing to carry. The cap never gets a vote."""
    assert fits("", cap=REMOTE_WINDOW) is False


# ---------------------------------------------------------------------------
# 2. The ladder actually threads it
# ---------------------------------------------------------------------------
CAST = ["Ada", "Bo"]


def play(*body_lines):
    return "\n".join((
        "TITLE: The Test",
        "MUSIC: theme up",
        f"{ANNOUNCER_NAME}: Tonight, a test.",
        "SCENE 1: a room",
        "Ada: We begin the work.",
        *body_lines,
        "Bo: And we end it.",
        f"{ANNOUNCER_NAME}: That was a test.",
        "CODA: The end.",
        "MUSIC: theme down",
        "END.",
    ))


#: Malformed: a standalone stage direction where a speaker line belongs.
BAD = play("Johannes Lachner enters, determination in his eyes.")


class ScriptedWriter:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, messages, *, temperature, max_new_tokens):
        self.prompts.append(list(messages)[-1]["content"])
        return self.replies.pop(0)


def run_ladder(writer, context_cap_fn=None):
    return scifi_news_pro._run_markup_ladder(
        writer,
        pass_id="script",
        system="system prompt",
        base_user="base user prompt",
        envelope=None,
        cast_names=CAST,
        initial_temperature=0.75,
        context_cap_fn=context_cap_fn,
    )


def test_the_ladder_THREADS_the_resolver_into_the_verdict():
    """Same draft, same ladder, two windows, two different second prompts.

    This is the wiring proof: a correct predicate that nothing calls would look
    identical in every test above."""
    wide = ScriptedWriter([BAD, play()])
    run_ladder(wide, context_cap_fn=lambda: REMOTE_WINDOW)
    assert "REJECTED DRAFT" in wide.prompts[1], (
        "a wide window dropped the draft anyway -- the resolver is not reaching "
        "the predicate")

    narrow = ScriptedWriter([BAD, play()])
    run_ladder(narrow, context_cap_fn=lambda: 200)
    assert "REJECTED DRAFT" not in narrow.prompts[1], (
        "a 200-token window still carried the draft -- the resolver's value is "
        "being ignored")


def test_the_resolver_is_NOT_called_when_the_first_parse_is_clean():
    """Laziness is the whole reason this is a callable and not a value.

    Resolving the cap acquires the slot, and on a swapped slot acquisition is a
    real model load plus a recorded transition. The happy path must not pay for
    a number it never reads."""
    asked = {"n": 0}

    def resolver():
        asked["n"] += 1
        return REMOTE_WINDOW

    run_ladder(ScriptedWriter([play()]), context_cap_fn=resolver)
    assert asked["n"] == 0, (
        "the cap was resolved on a clean first parse, forcing a slot "
        "acquisition the ladder never needed")


def test_the_resolver_is_asked_ONCE_across_the_whole_ladder():
    """Three rungs, one question. The window cannot change mid-ladder."""
    asked = {"n": 0}

    def resolver():
        asked["n"] += 1
        return REMOTE_WINDOW

    writer = ScriptedWriter([BAD, BAD, BAD, play()])
    run_ladder(writer, context_cap_fn=resolver)
    assert len(writer.prompts) >= 3, "premise broken: the ladder short-circuited"
    assert asked["n"] == 1, (
        "the cap was resolved %d times; it is memoized for a reason"
        % asked["n"])


def test_a_resolver_that_RAISES_does_not_kill_the_episode():
    """The predicate is a heuristic. Its worst honest answer costs one cold
    repair turn; letting a resolver's failure escape would cost the episode --
    the exact shape of the bug the overflow guard above it was written to
    close."""
    def angry():
        raise RuntimeError("slot registry is unavailable")

    _raw, parsed, _diag = run_ladder(
        ScriptedWriter([BAD, play()]), context_cap_fn=angry)
    assert parsed is not None


# ---------------------------------------------------------------------------
# 3. Where the real number comes from
# ---------------------------------------------------------------------------
def test_pass_script_FORWARDS_the_resolver_to_the_ladder(monkeypatch):
    """The seam a green suite would otherwise hide.

    Every test above drives `_run_markup_ladder` directly, so deleting
    `context_cap_fn=` from `_pass_script` would leave all of them passing and
    the wiring dead -- a correct predicate that nothing reaches. The prompt
    builders are stubbed because this test is about ONE argument's journey."""
    seen = {}

    def fake_ladder(creative_fn, **kwargs):
        seen.update(kwargs)
        return ("raw", None, {"attempt_trace": ()})

    monkeypatch.setattr(scifi_news_pro, "_run_markup_ladder", fake_ladder)
    monkeypatch.setattr(scifi_news_pro, "_seam", lambda pack, name: "system")
    monkeypatch.setattr(
        scifi_news_pro, "_script_user_prompt",
        lambda *args, **kwargs: "base user")

    def resolver():
        return REMOTE_WINDOW

    scifi_news_pro._pass_script(
        lambda *a, **k: "", object(), object(), "digest",
        None, CAST, None, resolver,
    )
    assert seen.get("context_cap_fn") is resolver, (
        "_pass_script did not forward the resolver; it passed %r"
        % (sorted(seen),))


def test_the_EPISODE_RUNNER_builds_the_resolver_from_its_scheduler():
    """The other half of the same seam.

    Source inspection, matching this module's existing precedent for exactly
    this defect class (`test_scifi_news_pro_format_example.py` pins
    `format_example=` into `_pass_script` the same way). Driving the whole
    runner would need a ledger, a pack, a bank row and a model."""
    import inspect

    src = inspect.getsource(scifi_news_pro.run_scifi_news_pro_episode)
    assert "context_cap_fn=_creative_context_cap_fn(slot_scheduler)" in src, (
        "the runner no longer hands the ladder a way to learn the real window")


def test_creative_cap_fn_reads_the_CREATIVE_slot():
    """Not "technical", and not "whichever slot ran last"."""
    asked = []

    class Scheduler:
        def context_cap_for(self, slot):
            asked.append(slot)
            return GGUF_N_CTX

    resolver = scifi_news_pro._creative_context_cap_fn(Scheduler())
    assert resolver() == GGUF_N_CTX
    assert asked == ["creative"]


def test_creative_cap_fn_is_None_for_a_scheduler_that_cannot_report():
    """A test double, or a scheduler older than `context_cap_for`. The ladder
    then keeps its flat estimate rather than failing to build."""
    class Older:
        pass

    assert scifi_news_pro._creative_context_cap_fn(Older()) is None
    assert scifi_news_pro._creative_context_cap_fn(None) is None


def _scheduler():
    return _SlotScheduler(
        creative_id="creative/model",
        technical_id="technical/model",
        top_p=0.92,
        min_p=0.0,
        repetition_penalty=1.0,
    )


def test_context_cap_for_reads_the_entry_WITHOUT_counting_a_generation(
        monkeypatch):
    """It shares acquisition with `inspect_fit` for this exact reason: a
    forensic receipt that counts a cap lookup as a model call is a lie about
    how many times the episode hit the model."""
    seen = {}
    sched = _scheduler()

    def fake_entry(slot, *, count_generation=True):
        seen["slot"] = slot
        seen["count_generation"] = count_generation
        return {"context_cap": GGUF_N_CTX}

    monkeypatch.setattr(sched, "_account_and_get_entry", fake_entry)
    assert sched.context_cap_for("creative") == GGUF_N_CTX
    assert seen == {"slot": "creative", "count_generation": False}
    assert sched.calls_by_slot["creative"] == 0


@pytest.mark.parametrize("entry", [
    {},                                  # transport reported nothing
    {"context_cap": None},               # reported explicitly unknown
    {"context_cap": "not a number"},     # reported garbage
])
def test_context_cap_for_returns_zero_when_the_entry_cannot_be_read(
        monkeypatch, entry):
    """0 is the agreed "no answer". The caller falls back; nothing raises."""
    sched = _scheduler()
    monkeypatch.setattr(
        sched, "_account_and_get_entry",
        lambda slot, *, count_generation=True: entry)
    assert sched.context_cap_for("creative") == 0


def test_context_cap_for_returns_zero_when_ACQUISITION_itself_fails(
        monkeypatch):
    """A slot that cannot be acquired is a problem for the next generation
    call, which will report it properly. It is not this accessor's to raise --
    it is asked from inside a heuristic."""
    sched = _scheduler()

    def boom(slot, *, count_generation=True):
        raise RuntimeError("no such row")

    monkeypatch.setattr(sched, "_account_and_get_entry", boom)
    assert sched.context_cap_for("creative") == 0
