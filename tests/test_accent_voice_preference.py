# -*- coding: utf-8 -*-
"""A cast row's declared accent is a voice PREFERENCE, never a gate.

WHY THIS EXISTS. `config/cast_pools.py::lemmy_row()` has stamped
`accent: "cockney"` since it was written, `nodes/production_ledger.py` copies it
onto the ledger row, and no caster ever read it -- the selector scores `timbre`
and nothing joined the two. The accent was declared and never consulted.

THE OPERATOR'S RULE, in his words (2026-09-24): "it should not be a hard fail if
it can't find a match, it just needs to be gender correct." That is also the
ruling `nodes/cast_lock.py` already recorded from the French Hamlet leg, where
two French men drew a female voice over bearded stills: "The timbre is
wrong-accented; the man is a man."

So every test here checks one of three things: an accent the bank CAN serve
narrows the draw, an accent it CANNOT serve costs nothing at all, and gender
survives either way.

WHO THIS ACTUALLY FIRES FOR, stated because the motivating character is not
really one of them. A row pinned in `RECURRING_CHARACTER_VOICES` is delivered by
the recurring lookup and never reaches the ordinary draw, so LEMMY -- pinned on
kokoro, cloud_elevenlabs and google_tts, which are the engines that ship -- does
not consult his own `accent` there at all. He happens to be correct anyway
because those three pinned ids already carry british/bbc tags. This wiring is
for the NEXT accented character, and for any unpinned engine. Operator ruling
2026-09-24: a wrong accent on a non-kokoro or unmapped engine is accepted --
"it's an experimental journey" -- so nothing here chases that.

Headless. No engine, no model, no GPU.
"""
from __future__ import annotations

import copy
import json

import pytest

from nodes._otr_voice_bank import accent_timbre_tags, load_voice_bank
from nodes.cast_lock import CastLock

ONE_MALE = [{"char_id": "c01", "name": "DOC", "gender": "male",
             "voice_preset": "v2/en_speaker_1"}]

#: Enough characters to exhaust kokoro's four british males.
CROWD = [
    {"char_id": "c01", "name": "MONTY", "gender": "male",
     "voice_preset": "v2/en_speaker_1"},
    {"char_id": "c02", "name": "NAG", "gender": "male",
     "voice_preset": "v2/en_speaker_3"},
    {"char_id": "c03", "name": "DOC", "gender": "male",
     "voice_preset": "v2/en_speaker_4"},
    {"char_id": "c04", "name": "PIP", "gender": "male",
     "voice_preset": "v2/en_speaker_5"},
    {"char_id": "c05", "name": "SAM", "gender": "male",
     "voice_preset": "v2/en_speaker_7"},
    {"char_id": "a1", "name": "ANNOUNCER", "gender": "male",
     "voice_preset": "v2/en_speaker_6"},
]


@pytest.fixture(scope="module")
def bank_by_id():
    return {e.voice_ref_id: e for e in load_voice_bank()[0]}


def _cast(rows, engine="kokoro", seed=42, announcer_engine=None):
    """Lock a cast and return {char_id: voice_ref_id}.

    `announcer_engine` is passed ONLY when a caller asks for it: the clone
    engines are character-voice engines and CastLock rightly refuses them as
    announcer engines, so forcing them here would fail the lock for a reason
    that has nothing to do with accents.
    """
    led = json.dumps({"meta": {"episode_seed": seed},
                      "cast": copy.deepcopy(rows), "lines": []})
    kwargs = {}
    if announcer_engine:
        kwargs["announcer_voice_engine"] = announcer_engine
    out = CastLock().lock(
        script_json=led, voice_bank="default_clean",
        char_voice_engine=engine, cast_voice_policy="auto_registry",
        **kwargs)[0]
    return {e["char_id"]: e.get("voice_ref_id")
            for e in json.loads(out)["cast"]}


def _with_accent(rows, accent, char_id="c01"):
    out = copy.deepcopy(rows)
    for row in out:
        if row["char_id"] == char_id:
            row["accent"] = accent
    return out


def _is_british(entry):
    tags = {str(t).strip().lower() for t in (entry.timbre or ())}
    return bool(tags & {"british", "bbc"})


# ---------------------------------------------------------------------------
# The tag join
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("accent", ["british", "English", "  cockney ", "UK",
                                    "london", "RP"])
def test_a_british_accent_resolves_to_the_banks_own_tags(accent):
    """The author writes "cockney"; the bank says "british". Both, folded."""
    assert set(accent_timbre_tags(accent)) & {"british", "bbc"}, accent


def test_an_accent_no_voice_can_speak_resolves_to_NOTHING(bank_by_id):
    """AND THIS IS THE ONE THAT MATTERS, because the obvious implementation
    gets it wrong.

    `timbre` feeds the deterministic slot seed, so returning a tag that matches
    ZERO voices still changes which voice is drawn. Measured while writing
    this: a row declaring the nonsense accent "klingon" moved a kokoro pick
    from `am_echo` to `am_santa` while matching nothing. "We tried and the bank
    could not help" must not silently mean "we reshuffled your cast".
    """
    assert accent_timbre_tags("klingon") == ()
    assert accent_timbre_tags("") == ()
    assert accent_timbre_tags(None) == ()


def test_an_empty_bank_contributes_no_preference():
    """An empty candidate list can serve no accent."""
    assert accent_timbre_tags("british", bank=()) == ()


def test_an_UNREADABLE_bank_contributes_no_preference(monkeypatch):
    """The name above used to claim this and did not test it.

    `bank=()` is not None, so it never reaches the `load_voice_bank()` call or
    the except around it -- it tested "empty", not "unreadable". This exercises
    the real exception path: a bank fault must degrade to "no preference", not
    propagate out of casting and kill a render.
    """
    import nodes._otr_voice_bank as VB

    def _boom(*_a, **_k):
        raise VB.VoiceBankError("bank is corrupt")

    monkeypatch.setattr(VB, "load_voice_bank", _boom)
    assert VB.accent_timbre_tags("british") == ()


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def test_a_declared_accent_narrows_the_draw_when_the_bank_can_serve_it(
        bank_by_id):
    """Six seeds, one character, kokoro -- which has four british males."""
    plain = [_cast(ONE_MALE, seed=s)["c01"] for s in range(1, 7)]
    asked = [_cast(_with_accent(ONE_MALE, "british"), seed=s)["c01"]
             for s in range(1, 7)]

    asked_brit = sum(1 for v in asked if _is_british(bank_by_id[v]))
    assert asked_brit == 6, (
        "asked for a british voice six times and got %d: %s"
        % (asked_brit, asked))

    # NOT VACUOUS, AND SAID STRUCTURALLY RATHER THAN BY LUCK. An earlier
    # version asserted `asked_brit > plain_brit`, which only holds while the
    # unconstrained draw happens not to land british six times running -- true
    # today at roughly a 31% base rate, but a bank content change could flake it
    # in either direction without any real regression. What actually makes the
    # 6/6 above meaningful is that a NON-british voice was available to draw and
    # was not drawn.
    non_british = [e for e in load_voice_bank()[0]
                   if e.engine == "kokoro" and e.gender == "male"
                   and "char_voice" in (e.roles or ())
                   and not _is_british(e)]
    assert non_british, (
        "kokoro has no non-british male character voices, so drawing six "
        "british ones proves nothing about the preference")
    assert not any(v in {e.voice_ref_id for e in non_british} for v in asked), (
        "a non-british voice was drawn despite the accent: %s" % (asked,))


def test_cockney_and_british_draw_the_same_voices():
    """The alias is a spelling of the same preference, not a second lane."""
    a = [_cast(_with_accent(ONE_MALE, "cockney"), seed=s)["c01"]
         for s in range(1, 7)]
    b = [_cast(_with_accent(ONE_MALE, "british"), seed=s)["c01"]
         for s in range(1, 7)]
    assert a == b, (a, b)


@pytest.mark.parametrize("accent", [None, "british", "cockney", "klingon"])
def test_the_draw_is_gender_correct_whatever_the_accent(accent, bank_by_id):
    """THE RULE THAT OUTRANKS THE ACCENT. A wrong-accented man is the trade;
    a woman's voice on a man is the defect that trade was chosen to avoid."""
    rows = ONE_MALE if accent is None else _with_accent(ONE_MALE, accent)
    for seed in range(1, 7):
        got = _cast(rows, seed=seed)["c01"]
        assert bank_by_id[got].gender == "male", (accent, seed, got)


@pytest.mark.parametrize("engine", ["indextts2", "dia"])
def test_an_engine_with_no_british_voices_still_casts_and_never_raises(
        engine, bank_by_id):
    """indextts2 and dia carry NO british male character rows at all. The
    request must degrade to an ordinary gender-correct draw, not fail."""
    got = _cast(_with_accent(ONE_MALE, "british"), engine=engine)["c01"]
    assert got, "no voice was assigned at all"
    assert bank_by_id[got].gender == "male"


def test_an_unmatchable_accent_leaves_the_cast_byte_identical():
    """The regression guard for the seed-reshuffle defect, at cast level."""
    plain = _cast(CROWD, announcer_engine="kokoro")
    weird = _cast(_with_accent(CROWD, "klingon", char_id="c03"),
                  announcer_engine="kokoro")
    assert plain == weird, (
        "an accent that matches no voice changed the cast:\\n  %s\\n  %s"
        % (plain, weird))


def test_a_crowd_that_exhausts_the_accent_pool_still_gives_everyone_a_voice(
        bank_by_id):
    """Five men plus an announcer against four british kokoro males.

    Somebody must miss out, and the operator's rule says that is fine as long
    as he is a man. Nobody may go unvoiced and nobody may share a voice.
    """
    rows = copy.deepcopy(CROWD)
    for row in rows:
        row["accent"] = "british"
    got = _cast(rows, announcer_engine="kokoro")
    assert all(got.values()), got
    assert len(set(got.values())) == len(got), (
        "two characters share one voice: %s" % (got,))
    for cid, vid in got.items():
        assert bank_by_id[vid].gender == "male", (cid, vid)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
