"""The safe-open brief must REACH the prompt, and a starved one must not run.

WHY THIS FILE EXISTS. Twenty-three shipped episodes opened with the language
model addressing the operator -- "Please provide the SETTING, TIME, HOOK, and
the cast list so that I may write the opening for you" -- as the first line the
listener hears. Two independent causes, both invisible to every test that
existed:

  * the prompt builder read a ``hook`` attribute ``SafeOpenBrief`` has never
    defined, so ``getattr(..., '')`` silently emptied it, while
    ``opening_status_quo``, ``cast`` and ``era`` were built and read by nobody;
  * the derive validator accepted a brief with NO CAST, while every bank's
    system prompt promises the model "the cast list below".

Nothing failed. The composer returned a line, the writer stamped it rewritten,
and the episode shipped. The coverage that existed asserted the SYSTEM message
and stubbed the compose, so it could not have noticed either.

THE POSTURE HERE IS PRESENCE, NOT ABSENCE. Asserting that the bad string is
gone is also satisfied by an empty prompt, which is how this survived. Every
test below asserts the real value is THERE, and the starvation tests assert the
model was never called at all.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from nodes import _otr_line_composer as LC  # noqa: E402
from nodes import _otr_story_brief as SB  # noqa: E402

BANKS = ("original", "shakespeare", "public_domain", "media_archive")

SETTING = "a county records archive"
TIME = "night"
STATUS_QUO = "A mislabeled reel waits on the top shelf."
ERA = "1947"
CAST = ("MARA", "DELACROIX")

VALID_INTRO = (
    "Tonight, from a shelf of forgotten reels, a quiet archive hums "
    "with one question that will not rest."
)


def _brief(**overrides):
    """A real SafeOpenBrief -- never SimpleNamespace.

    A namespace fixture would have HIDDEN this bug: give it a ``hook``
    attribute and the broken builder reads it happily. The dataclass is the
    contract, so the test has to use the contract.
    """
    fields = dict(
        setting=SETTING,
        time_of_day=TIME,
        opening_status_quo=STATUS_QUO,
        cast=CAST,
        era=ERA,
    )
    fields.update(overrides)
    return LC.SafeOpenBrief(**fields)


def _capturing(reply: str = VALID_INTRO):
    calls: list = []

    def fn(messages, **kwargs):
        calls.append(messages)
        return reply

    return fn, calls


def _user_message(calls) -> str:
    assert calls, "creative_fn was never called"
    return calls[0][1]["content"]


def _compose(brief, *, bank: str = "public_domain", reply: str = VALID_INTRO):
    fn, calls = _capturing(reply)
    result = LC.compose_announcer_intro(
        creative_fn=fn,
        script_brief="",
        story_scaffold=True,
        safe_open_brief=brief,
        source_bank_id=bank,
    )
    return result, calls


# ---------------------------------------------------------------------------
# The brief reaches the prompt
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bank", BANKS)
def test_every_brief_field_reaches_the_user_message(bank):
    """One test that catches all three severed wires at once: the `hook` typo,
    the dropped cast, and the dropped status quo."""
    _res, calls = _compose(_brief(), bank=bank)
    msg = _user_message(calls)

    assert f"SETTING: {SETTING}" in msg
    assert f"TIME: {TIME}" in msg
    assert f"OPENING STATUS: {STATUS_QUO}" in msg
    assert f"ERA: {ERA}" in msg
    for name in CAST:
        assert name in msg, f"{bank}: cast name {name!r} never reached the model"


def test_the_cast_is_one_line_naming_the_locked_roster():
    """Every bank's seam says "use ONLY the proper names in the cast list
    below". If the composer sends no cast line, the system prompt promises a
    roster the user message never delivers -- and the model asks for it."""
    _res, calls = _compose(_brief())
    cast_lines = [
        ln for ln in _user_message(calls).splitlines() if ln.startswith("CAST:")
    ]
    assert cast_lines == ["CAST: MARA, DELACROIX"]


def test_no_label_is_emitted_without_a_value():
    """The shipped starvation mode. A bare "SETTING:" with nothing behind it
    reads to the model as a form to fill in rather than material to write from,
    which is precisely what it did."""
    _res, calls = _compose(_brief(setting="", time_of_day="", era=""))
    msg = _user_message(calls)

    assert STATUS_QUO in msg
    assert "MARA" in msg
    for line in msg.splitlines():
        if ":" in line:
            _label, _, value = line.partition(":")
            assert value.strip(), f"empty label line survived: {line!r}"


def test_a_brief_with_no_setting_still_composes():
    """`opening_status_quo` alone is a legitimate brief -- the derive validator
    has always allowed it. It must not be collateral damage of the guard."""
    result, calls = _compose(_brief(setting=""))
    assert result.text == VALID_INTRO
    assert STATUS_QUO in _user_message(calls)


def test_the_hook_label_is_gone():
    """Asserted as a WHOLE LABEL LINE, not as "the word HOOK is absent" -- an
    authored setting could legitimately contain the word."""
    _res, calls = _compose(_brief())
    assert not any(
        ln.startswith("HOOK:") for ln in _user_message(calls).splitlines()
    )


# ---------------------------------------------------------------------------
# A starved brief never reaches the model
# ---------------------------------------------------------------------------

STARVED = {
    "missing_cast": dict(cast=()),
    "whitespace_only_cast": dict(cast=("", "   ")),
    "bare_string_cast": dict(cast="MARA"),
    "cast_only": dict(setting="", opening_status_quo=""),
    "era_only": dict(setting="", time_of_day="", opening_status_quo="", cast=()),
    "time_only": dict(setting="", opening_status_quo="", cast=(), era=""),
    "all_empty": dict(
        setting="", time_of_day="", opening_status_quo="", cast=(), era="",
    ),
}


@pytest.mark.parametrize("label,overrides", sorted(STARVED.items()))
def test_a_starved_brief_raises_and_never_calls_the_model(label, overrides):
    fn, calls = _capturing()
    with pytest.raises(LC.AnnouncerBriefStarvedError):
        LC.compose_announcer_intro(
            creative_fn=fn,
            script_brief="",
            story_scaffold=True,
            safe_open_brief=_brief(**overrides),
            source_bank_id="public_domain",
        )
    assert calls == [], f"{label}: spent an LLM call on a starved brief"


def test_a_bare_string_cast_is_rejected_not_iterated():
    """"MARA" is a sequence of five characters. Iterating it would put M, A, R
    into the cast list as if they were people."""
    assert LC.cleaned_cast_names("MARA") == []
    assert LC.cleaned_cast_names(b"MARA") == []
    assert LC.cleaned_cast_names(("MARA", " DELACROIX ")) == ["MARA", "DELACROIX"]


def test_whitespace_only_cast_cleans_away_to_nothing():
    """A non-empty sequence that yields no usable name would otherwise pass a
    length check and then vanish from the prompt -- recreating the starvation
    the predicate exists to stop."""
    assert LC.cleaned_cast_names(("", "   ", "\t")) == []


def test_the_starvation_reason_is_bounded_and_ordered():
    """Scene context before cast, always, so an all-empty brief reports one
    stable reason and receipts stay comparable across runs."""
    assert LC.safe_open_viability(
        setting="", opening_status_quo="", cast=CAST,
    ) == "missing_scene_context"
    assert LC.safe_open_viability(
        setting=SETTING, opening_status_quo="", cast=(),
    ) == "missing_cast"
    assert LC.safe_open_viability(
        setting="", opening_status_quo="", cast=(),
    ) == "missing_scene_context"
    assert LC.safe_open_viability(
        setting=SETTING, opening_status_quo="", cast=CAST,
    ) is None


def test_the_typed_error_carries_its_reason_without_string_parsing():
    """The writer reads `.reason` to stamp a receipt. Parsing the message text
    would make the receipt hostage to the wording."""
    with pytest.raises(LC.AnnouncerBriefStarvedError) as caught:
        LC.compose_announcer_intro(
            creative_fn=_capturing()[0],
            script_brief="",
            story_scaffold=True,
            safe_open_brief=_brief(cast=()),
            source_bank_id="public_domain",
        )
    assert caught.value.reason == "missing_cast"
    assert isinstance(caught.value, RuntimeError), (
        "must stay a RuntimeError -- test_announcer_passes pins that contract")


# ---------------------------------------------------------------------------
# ONE predicate, TWO callers
# ---------------------------------------------------------------------------

def test_the_derive_validator_rejects_a_cast_less_brief():
    """THE PRE-2026-07-24 HOLE. The validator required setting OR status quo
    and iterated `cast` only to reject off-roster names, so an EMPTY cast
    passed -- while every bank's seam promises a cast list."""
    model = SB.ProducedOpenBriefModel(
        setting=SETTING, time_of_day=TIME,
        opening_status_quo=STATUS_QUO, cast=[],
    )
    assert SB._validate_produced_open(model, ["MARA"]) == "missing_cast"


def test_the_derive_validator_still_rejects_off_roster_names():
    """The viability check runs FIRST but must not displace the roster
    invariant -- the derive pass may never introduce a name the episode does
    not have."""
    model = SB.ProducedOpenBriefModel(
        setting=SETTING, time_of_day=TIME,
        opening_status_quo=STATUS_QUO, cast=["NOBODY"],
    )
    verdict = SB._validate_produced_open(model, ["MARA"])
    assert verdict and "roster" in verdict


def test_a_viable_brief_passes_the_derive_validator():
    model = SB.ProducedOpenBriefModel(
        setting="", time_of_day="",
        opening_status_quo=STATUS_QUO, cast=["MARA"],
    )
    assert SB._validate_produced_open(model, ["MARA"]) is None


def test_both_consumers_reach_the_shared_predicate(monkeypatch):
    """A shared predicate only one caller reaches is the same zero-reader
    defect in new clothes.

    Each consumer is patched at ITS OWN module binding: `_otr_story_brief`
    imports the name into its own namespace, so patching only the composer's
    binding would miss it and the test would pass while proving half of what it
    claims.
    """
    seen: list[str] = []

    def spy_composer(**kwargs):
        seen.append("composer")
        return None

    def spy_brief(**kwargs):
        seen.append("story_brief")
        return None

    monkeypatch.setattr(LC, "safe_open_viability", spy_composer)
    monkeypatch.setattr(SB, "safe_open_viability", spy_brief)

    LC.compose_announcer_intro(
        creative_fn=_capturing()[0], script_brief="", story_scaffold=True,
        safe_open_brief=_brief(), source_bank_id="public_domain",
    )
    SB._validate_produced_open(
        SB.ProducedOpenBriefModel(
            setting=SETTING, time_of_day=TIME,
            opening_status_quo=STATUS_QUO, cast=["MARA"],
        ),
        ["MARA"],
    )

    assert sorted(seen) == ["composer", "story_brief"], (
        f"a consumer never reached the shared predicate: {seen}")
