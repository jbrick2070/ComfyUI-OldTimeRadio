"""Every cast row ends male or female -- one ladder, every lane (2026-10-17).

OPERATOR: "unisex names is ok but we need to clamp male or female in the
ledger and voice and descriptions for rendering prompts, stills and video".

The producers settle gender before any bio or line is written
(`_otr_casting` deals 50/50, My Story settles every role, Sci-Fi News Pro is
binary by schema); `Ledger.set_cast` clamps whatever still arrives non-binary
with the same ladder (`_otr_cast_gender`), so voice casting and the picture
prompts only ever read male or female.

Hermetic: no GPU, no model.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parent.parent
for _p in (_REPO, _REPO / "nodes"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from nodes import _otr_cast_gender as CG  # noqa: E402
from nodes import production_ledger as PL  # noqa: E402


# --------------------------------------------------------------------------- #
# the ladder
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    # Gemini contrarian: stripping the title exposed the SURNAME to the
    # first-name pool ("VANCE" is a male first name there).
    ("MRS. VANCE", ("female", "courtesy_title")),
    ("LADY STONE", ("female", "courtesy_title")),
    ("SIR TOBY", ("male", "courtesy_title")),
    ("Mr. Hayes", ("male", "courtesy_title")),
])
def test_a_courtesy_title_decides_before_the_name_pool(name, expected):
    assert CG.settle_binary_gender(name) == expected


def test_a_relation_in_the_role_decides_and_a_possessive_does_not():
    assert CG.settle_binary_gender("Qx1", role="her brother") == (
        "male", "role_relation")
    assert CG.settle_binary_gender("Qx1", role="mother's helper")[1] == "seeded_coin"


def test_a_confident_name_comes_before_the_description():
    from config import cast_pools

    name = next(n for n in cast_pools.FIRST_NAMES_BY_GENDER["male"]
                if cast_pools.gender_of_first_name(n) == "male")
    assert CG.settle_binary_gender(name, description="She waits.") == (
        "male", "name_pool")


def test_the_description_counts_only_when_it_names_nobody_else():
    assert CG.settle_binary_gender("Qx1", description="the sound she hears") == (
        "female", "description_pronouns")
    # Codex/Gemini: "Tom's rival; doubts that he saw the signal" is about Tom.
    assert CG.settle_binary_gender(
        "Pat", description="Tom's rival; he saw the signal",
        others=("Tom",))[1] != "description_pronouns"


def test_the_coin_is_binary_and_stable():
    a = CG.settle_binary_gender("Qx1")
    assert a == CG.settle_binary_gender("  qx1 ")
    assert a[0] in ("male", "female") and a[1] == "seeded_coin"


# --------------------------------------------------------------------------- #
# the clamp
# --------------------------------------------------------------------------- #
def test_clamp_settles_non_binary_rows_in_place_and_leaves_binary_ones():
    rows = [
        {"char_id": "c02", "name": "MRS. QUINN", "gender": "other"},
        {"char_id": "c03", "name": "JO", "gender": ""},
        {"char_id": "c04", "name": "ADA", "gender": "Woman"},
        {"char_id": "c05", "name": "TOM", "gender": "male"},
    ]
    receipts = CG.clamp_cast_rows(rows)
    assert rows[0]["gender"] == "female"
    assert rows[1]["gender"] in ("male", "female")
    assert rows[2]["gender"] == "female"           # a synonym, canonicalized
    assert rows[3]["gender"] == "male"
    assert [r["char_id"] for r in receipts] == ["c02", "c03"]
    assert receipts[0]["was"] == "other" and receipts[0]["source"] == "courtesy_title"


@pytest.mark.parametrize("raw", ["other", "neutral", "non-binary", "unknown", "", None, "AI"])
def test_no_spelling_of_unresolved_survives_the_clamp(raw):
    rows = [{"char_id": "c02", "name": "QUINN STONE", "gender": raw}]
    CG.clamp_cast_rows(rows)
    assert rows[0]["gender"] in ("male", "female")


def test_set_cast_stamps_binary_and_records_the_clamp(tmp_path):
    led = PL.Ledger(episode_id="ep_clamp", out_dir=str(tmp_path))
    caller_rows = [
        {"char_id": "c01", "name": "ANNOUNCER", "gender": "male",
         "character_description": "", "tts_model": "kokoro",
         "voice_preset": "bm_george"},
        {"char_id": "c02", "name": "QUINN STONE", "gender": "other",
         "character_description": "A pilot.", "tts_model": "",
         "voice_preset": ""},
    ]
    led.set_cast(caller_rows)
    stamped = {r["char_id"]: r["gender"] for r in led.data["cast"]}
    assert stamped["c02"] in ("male", "female")
    # the caller's own row agrees with the ledger
    assert caller_rows[1]["gender"] == stamped["c02"]
    assert led.data["meta"]["gender_clamp"][0]["char_id"] == "c02"


def test_a_binary_cast_leaves_no_clamp_receipt(tmp_path):
    led = PL.Ledger(episode_id="ep_clean", out_dir=str(tmp_path))
    led.set_cast([{"char_id": "c02", "name": "ADA", "gender": "female",
                   "character_description": "", "tts_model": "",
                   "voice_preset": ""}])
    assert "gender_clamp" not in (led.data.get("meta") or {})


# --------------------------------------------------------------------------- #
# the producers: no lane deals "other" any more
# --------------------------------------------------------------------------- #
def _stub(messages, *, temperature, max_new_tokens):  # noqa: ARG001
    return json.dumps({"character_description": "A steady field operator."})


def test_the_legacy_caster_never_deals_other():
    from nodes import _otr_casting as C

    for seed in range(60):
        cast, _meta = C.lock_cast(
            creative_fn=_stub, num_characters=5,
            news_seed="An anomaly over the test range.",
            style="1950s science fiction radio drama",
            rng=random.Random(seed), cast_seed=seed, force_lemmy=False,
            max_attempts_per_call=1)
        bad = [(r["name"], r["gender"]) for r in cast
               if r["gender"] not in ("male", "female")]
        assert not bad, f"seed {seed}: {bad}"


def test_the_bio_prompt_never_asks_for_other():
    from nodes import _otr_casting as C
    import inspect

    assert "<male|female|other>" not in inspect.getsource(C)
    assert C._VALID_GENDERS == {"male", "female"}


def test_set_cast_is_wired_to_the_shared_clamp():
    import inspect

    assert "clamp_cast_rows(" in inspect.getsource(PL.Ledger.set_cast)


# --------------------------------------------------------------------------- #
# the pictures read the same stamped row
# --------------------------------------------------------------------------- #
def test_a_clamped_row_reaches_still_and_video_prompts_as_a_man_or_woman():
    from nodes import otr_meta_brief_image_prompt as IMG
    from nodes import otr_shot_lock as SL

    rows = [{"char_id": "c02", "name": "MRS. QUINN", "gender": "other",
             "character_description": "A pilot in a flight jacket."}]
    CG.clamp_cast_rows(rows)
    still = IMG._ensure_gender_anchor(rows[0]["character_description"], rows[0])
    video = SL._appearance_for_char({"cast": rows}, "c02")
    assert still.startswith("adult woman, ")
    assert video.startswith("adult woman, ")


# --------------------------------------------------------------------------- #
# Sonnet QA on e4be35fc: the 50/50 tie must not always go to one gender
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("count,prior", [(1, []), (3, []), (5, []), (2, ["female"])])
def test_an_odd_leftover_slot_is_drawn_not_defaulted(count, prior):
    from collections import Counter
    from nodes import _otr_casting as C

    c = Counter()
    for seed in range(400):
        c.update(C._plan_gender_distribution(count, prior, random.Random(seed)))
    assert c["male"] and c["female"]
    assert set(c) <= {"male", "female"}
    # neither gender takes every odd slot
    assert abs(c["male"] - c["female"]) < count * 400 * 0.6


@pytest.mark.parametrize("name", ["JULIET", "pool"])
def test_a_source_character_the_roster_left_unknown_uses_a_confident_name(name):
    from nodes import _otr_casting as C
    from config import cast_pools

    if name == "pool":
        name = next(n for n in cast_pools.FIRST_NAMES_BY_GENDER["female"]
                    if cast_pools.gender_of_first_name(n) == "female")

    class _Slot:
        def __init__(self, char_id, name):
            self.char_id, self.name, self.source_owned = char_id, name, True

    for seed in range(40):
        ens = C.precompute_ensemble_slots(
            [_Slot("c02", name.upper())], rng=random.Random(seed),
            cast_seed=seed, gender_by_name={})
        assert ens[0].gender == "female"


# --------------------------------------------------------------------------- #
# operator 2026-10-17: "Juliet should be female" -- the vendored rosters are a
# registry too, not only the curated first-name pool
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name,expected", [
    ("JULIET", "female"), ("Juliet", "female"), ("ROMEO", "male"),
    ("CORDELIA", "female"), ("LEAR", "male"), ("PROSPERO", "male"),
    ("MIRANDA", "female"),
])
def test_known_source_characters_settle_from_their_rosters(name, expected):
    assert CG.settle_binary_gender(name) == (expected, "known_character")


@pytest.mark.parametrize("word", ["NURSE", "FOOL", "PRINCE", "BOY", "FIRST", "J"])
def test_rank_and_role_words_are_not_known_characters(word):
    assert CG.known_character_gender(word) == ""


def test_juliet_on_any_lane_ends_female_in_the_ledger(tmp_path):
    led = PL.Ledger(episode_id="ep_juliet", out_dir=str(tmp_path))
    led.set_cast([{"char_id": "c02", "name": "JULIET", "gender": "",
                   "character_description": "", "tts_model": "",
                   "voice_preset": ""}])
    assert led.data["cast"][0]["gender"] == "female"


# Sonnet QA on 68f8684e: the roster index holds surnames and places, so the
# character's own description outranks it, it reads only a bare one-word name,
# and an operator "stays on the roll" ruling is never overridden.
@pytest.mark.parametrize("name,description", [
    ("Dr. Harris", "She is a surgeon; she works late."),
    ("France", "She is a diplomat."),
])
def test_the_description_outranks_the_roster_index(name, description):
    assert CG.settle_binary_gender(name, description=description) == (
        "female", "description_pronouns")


@pytest.mark.parametrize("name", ["ARIEL", "PUCK", "ROBIN"])
def test_an_operator_roll_ruling_is_never_a_known_character(name):
    assert CG.known_character_gender(name) == ""


def test_the_index_reads_only_a_bare_one_word_name():
    assert CG.known_character_gender("JULIET") == "female"
    assert CG.known_character_gender("Mrs. Juliet") == ""
    assert CG.known_character_gender("Juliet Capulet") == ""
