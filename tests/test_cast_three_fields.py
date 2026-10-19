"""Every source bank writes the character as three fields (2026-10-18).

Operator: "clean split for traceability" -- and "be sure for character you are
reviewing every source bank for consistency". The old one-string
`character_description` had its Voice/Presence sentences lettered into stills
by FLUX 3 Klein (PBUG-20261018-03). Now every cast row carries `appearance`
(drawable), `presence` and `voice_cue`, read only through
`nodes/_otr_shared/cast_text.py`:

* shared casting (original, media_archive, public_domain, shakespeare)
* my_story (its own treatment pass)
* scifi_news_pro (its own casting pass; pinned in test_scifi_news_pro_assembly)
* the hand-written rows in config/cast_pools.py (announcer, LEMMY)
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from config import cast_pools as POOLS  # noqa: E402
from nodes import _otr_casting as CASTING  # noqa: E402
from nodes import _otr_my_story as MS  # noqa: E402
from nodes._otr_shared import cast_text as CT  # noqa: E402
from nodes.production_ledger import Ledger  # noqa: E402

FIELDS = ("appearance", "presence", "voice_cue")


def _canned(n):
    seq = iter(range(n))

    def fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        i = next(seq)
        return json.dumps({
            "appearance": f"40s, the relay keeper, square jaw, grey stubble {i}",
            "presence": f"holds the station together by habit {i}",
            "voice_cue": f"low and unhurried {i}",
            "speech_signature": "clipped",
        })
    return fn


def test_shared_casting_rows_carry_three_fields_and_survive_the_ledger(tmp_path):
    cast, _meta = CASTING.lock_cast(
        creative_fn=_canned(8), num_characters=2, news_seed="x", style="open",
        rng=random.Random(7), cast_seed=7, force_lemmy=False)
    for row in cast:
        assert "character_description" not in row
        for f in FIELDS:
            assert CT.field(row, f), (row["name"], f)
    led = Ledger("ep_three_fields", str(tmp_path))
    led.set_cast(cast)
    path = led.save()
    on_disk = json.loads(Path(path).read_text(encoding="utf-8"))
    for before, after in zip(cast, on_disk["cast"]):
        for f in FIELDS:
            assert after[f] == before[f], f        # set_cast keeps every field
        assert "character_description" not in after


def test_an_empty_presence_or_voice_is_backfilled_by_lock_cast():
    def fn(messages, *, temperature, max_new_tokens):  # noqa: ARG001
        return json.dumps({"appearance": "30s, the courier, freckled",
                           "speech_signature": "quick"})
    cast, _ = CASTING.lock_cast(
        creative_fn=fn, num_characters=1, news_seed="x", style="open",
        rng=random.Random(3), cast_seed=3, force_lemmy=False)
    row = [r for r in cast if r["name"] != "ANNOUNCER"][0]
    assert row["presence"] and row["voice_cue"]


def test_my_story_rows_split_the_same_way():
    member = MS.CastMember(name="Ada Thorne", role="the keeper",
                           appearance="wind-burned face, salt-grey braid",
                           presence="Six years alone; talks to the lamp.",
                           gender="female", register="blunt, practical",
                           timbre="low, weathered")
    treatment = MS.StoryTreatment(cast=[member], acts=[{"n": 1, "purpose": "p"}])
    rows = MS._assign_voices(treatment, random.Random(1))
    row = rows[1]
    assert row["appearance"] == "wind-burned face, salt-grey braid"
    assert row["presence"] == "Six years alone; talks to the lamp."
    assert row["voice_cue"] == "blunt, practical; low, weathered"
    assert "character_description" not in row


def test_the_hand_written_rows_split_too():
    for row in (POOLS.pick_announcer(random.Random(2)), POOLS.lemmy_row()):
        for f in FIELDS:
            assert CT.field(row, f), (row.get("name"), f)
        assert "character_description" not in row
    # LEMMY's accent is a voice: it is not painted.
    assert "Cockney" not in POOLS.LEMMY_PROFILE["appearance"]
    assert "Cockney" in POOLS.LEMMY_PROFILE["voice_cue"]


def test_the_readers_take_only_what_they_need():
    row = {"appearance": "40s, the keeper, square jaw",
           "presence": "holds the line", "voice_cue": "gravel and smoke",
           "speech_signature": "terse"}
    assert CT.still_text(row) == "40s, the keeper, square jaw"
    video = CT.video_text(row)
    assert "holds the line" in video and "gravel" not in video
    card = CT.card_text(row)
    # No voice cue for the writing passes (agy review, 2026-10-18).
    assert card == "40s, the keeper, square jaw. Presence: holds the line."
    assert video == "40s, the keeper, square jaw. holds the line"
    ids = CT.identity_fields(row)
    assert ids == {"appearance": "40s, the keeper, square jaw",
                   "presence": "holds the line",
                   "voice_cue": "gravel and smoke", "speech_signature": "terse"}


def test_a_capped_card_keeps_presence_and_never_carries_voice():
    """Codex r1: an appearance-first cut lost presence entirely at 200. The
    voice cue never reaches a writing pass (agy review, 2026-10-18)."""
    row = {"appearance": "x " * 200, "presence": "carries the weight",
           "voice_cue": "dry and quick"}
    card = CT.card_text(row, cap=120)
    assert len(card) <= 120
    assert "Presence: carries the weight" in card
    assert "dry and quick" not in card


def test_a_field_too_long_to_fit_a_word_spends_no_budget():
    """agy code review: an emptied field still charged its label's length.
    At cap 20 the presence share cannot hold its own label, so presence is
    dropped whole and the card stays inside the cap."""
    row = {"appearance": "a" * 30, "presence": "carries the weight"}
    card = CT.card_text(row, cap=20)
    assert "Presence" not in card and len(card) <= 20


def test_the_prior_cast_echo_is_the_age_and_role_lead():
    """agy code review: the new appearance has no period after the role, so
    the echo must cut at the second comma, not run 120 chars of face."""
    row = {"name": "ADA", "gender": "female",
           "appearance": "40s, the keeper, wind-burned face, salt-grey braid, "
                         "a scar along the jaw"}
    assert CASTING._format_prior_entry(row) == "ADA (F, 40s, the keeper)"


def test_a_null_optional_field_does_not_fail_the_schema():
    r = CASTING.DescriptionResponse.model_validate(
        {"appearance": "30s, the clerk", "presence": None, "voice_cue": None,
         "speech_signature": None})
    assert (r.presence, r.voice_cue, r.speech_signature) == ("", "", "")


def test_a_capped_video_text_keeps_presence():
    """Codex wiring review: ShotLock sliced appearance+presence at 160, so a
    long appearance cut presence off."""
    row = {"appearance": "40s, the keeper, " + "weathered " * 30,
           "presence": "holds the line under pressure", "voice_cue": "gravel"}
    text = CT.video_text(row, cap=140)
    assert len(text) <= 140
    assert "holds the line under pressure" in text and "gravel" not in text


def test_an_appearance_that_says_nothing_is_empty_for_pictures():
    """Live legs, 2026-10-18: "none" (a disembodied voice) and "n/a" (two
    scifi leads) reached a picture prompt as "adult woman, none."."""
    for word in ("none", "n/a", "N/A.", "unknown", "invisible", "disembodied"):
        row = {"appearance": word, "presence": "drifts through the fog"}
        assert CT.still_text(row) == "", word
        # Bodiless: nothing to show, and its presence is how it sounds.
        assert CT.video_text(row) == "", word
        assert word in CT.card_text(row)       # the writing passes still see it


def test_no_writer_prompt_invites_a_stage_direction():
    """2026-10-18 audit: the line composer showed every beat as
    "SPEAKER (mood):", the exact shape of "(Whispering) Tom...", and the scifi
    script prompt allowed parentheses and brackets in spoken text."""
    from nodes import _otr_line_composer as LC
    spine = LC.render_outline_spine(
        [{"beat_id": "b001", "speaker": "ADA", "speaker_role": "character",
          "mood": "hushed", "intent": "warn Tom"}])
    assert "(hushed)" not in spine and "tone: hushed" in spine
    import json as _json
    from pathlib import Path as _P
    scifi = _json.loads((_P(REPO) / "nodes/story_packs/scifi_news_pro/"
                         "scifi_news_pro.json").read_text(encoding="utf-8"))
    script = scifi["prompt_stages"]["scifi_news_pro_script_system"]
    assert "including quotation marks, parentheses" not in script
    assert "never carries parentheses or brackets" in script


def test_a_bodiless_character_in_prose_draws_nothing_and_voices_nothing():
    """Six-bank leg on 4f7f5602: my_story's Voice wrote its appearance as a
    sentence the none/n/a list missed, and its presence was sound."""
    row = {"appearance": "no physical appearance, represented only by an "
                         "auditory effect",
           "presence": "Starts as a faint, mournful sigh, builds into clear, "
                       "urgent speech, and finally fades into crashing waves"}
    assert CT.is_bodiless(row)
    assert CT.still_text(row) == "" and CT.video_text(row) == ""
    # A real look that merely contains "unseen" or "invisible" stays drawable.
    for look in ("a woman in her forties, an unseen scar under the collar",
                 "a man with an invisible limp only his wife notices"):
        real = {"appearance": look, "presence": "stands very still"}
        assert not CT.is_bodiless(real), look
        assert CT.still_text(real) == look


def test_appearance_names_the_gender_in_every_casting_prompt():
    """Six-bank leg: public_domain's BARRETT, a female row whose appearance
    never said woman, was drawn as a man."""
    import json as _json
    from pathlib import Path as _P
    from nodes import _otr_casting as C
    slot = C.EnsembleSlot(char_id="c02", name="ALICE", gender="female",
                          timbre="warm", role="lead")
    assert C._deterministic_identity_floor(slot)["appearance"].startswith("Adult woman")
    prompt = C._build_user_prompt("ALICE", "x", "noir", [],
                                  gender="female", timbre="warm", role="lead")
    assert "<woman|man>" in prompt and "a woman or a man" in prompt
    for rel, key in (("my_story/my_story.json", "my_story_treatment_system"),
                     ("scifi_news_pro/scifi_news_pro.json",
                      "scifi_news_pro_casting_system")):
        seam = _json.loads((_P(REPO) / "nodes/story_packs" / rel)
                           .read_text(encoding="utf-8"))["prompt_stages"][key]
        assert "a woman or a man" in seam, rel
    assert "a man in his 50s" in POOLS.LEMMY_PROFILE["appearance"]


def test_bodiless_marks_are_whole_phrases_not_words_a_real_look_carries():
    """Sonnet QA on fe35b8d5: bare "no physical", "never seen", "auditory" and
    "no body" blanked real characters."""
    for look in ("no body armour, a tall man", "a woman never seen without gloves",
                 "a man with no physical scars, 40s",
                 "tall man, auditory implant behind ear",
                 "no physical resemblance to his brother, tall man",
                 "a nobody clerk, a woman of thirty"):
        assert not CT.is_bodiless({"appearance": look}), look
    for ghost in ("just a voice on the wire", "a presence felt, not seen",
                  "a faceless voice", "disembodied, a cold draught",
                  "no physical appearance, represented only by sound"):
        assert CT.is_bodiless({"appearance": ghost}), ghost


def test_a_bodiless_character_gives_video_no_name_and_no_anchor():
    """Sonnet QA on fe35b8d5: the empty video text fell back to the NAME, so a
    joint-AV prompt read "adult woman, THE VOICE"."""
    from nodes import otr_shot_lock as SL
    ledger = {"cast": [{"char_id": "c04", "name": "THE VOICE", "gender": "female",
                        "appearance": "just a voice in the fog",
                        "presence": "a sigh that becomes speech"}]}
    assert SL._appearance_for_char(ledger, "c04") == ""
