"""scifi_fable2 S1b -- artifact models, validator factories, envelope,
deal, voice menu (architecture doc s5/s8/s9/s13).

Pure Python; no GPU, no LLM. Named thresholds are fixtured BOTH
directions (r2 anchor): a value just under the threshold passes, a value
at/over it fails.
"""
from __future__ import annotations

import random
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_scifi_fable2 as F2  # noqa: E402
from nodes import _otr_story_rules as STORY_RULES  # noqa: E402


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _dossier(**over) -> F2.DossierLLM:
    d = {
        "facts_to_keep": [
            "A survey mapped 1,200 new vents on Mount Etna this season.",
            "Doctor Rossi led the instrument team on the north slope.",
            "The vents released measurable heat before any tremor.",
        ],
        "allowed_numbers": ["1,200"],
        "named_entities": {
            "people": ["Doctor Rossi"],
            "places": ["Mount Etna"],
            "things": ["heat sensors"],
        },
        "dramatizable_vectors": [
            "a scientist who trusts the instruments over the committee",
            "a mountain village that hears the mountain breathe",
            "the cost of raising an alarm one day too early",
        ],
    }
    d.update(over)
    return F2.DossierLLM.model_validate(d)


_DIGEST = (
    "HEADLINE: Survey maps 1,200 new vents on Mount Etna\n"
    "SOURCE: MIT News 2026-07-01\n"
    "SUMMARY: Doctor Rossi's team mapped 1,200 new vents using heat "
    "sensors on Mount Etna before any tremor was recorded.\n"
)

_PROVENANCE = {
    "headline": "Survey maps 1,200 new vents on Mount Etna",
    "source": "MIT News", "date": "2026-07-01", "link": "",
}


def _treatment_dict(**over) -> dict:
    d = {
        "title": "The Long Count",
        "dramatic_question": (
            "Will Sela trust her own instruments before the village "
            "stops trusting her?"),
        "setting": "a volcano observatory above a village at night",
        "cast_shapes": [
            {"name": "SELA", "role": "instrument scientist",
             "want": "to be believed before the mountain proves her right",
             "pressure": "the committee reads her charts as noise",
             "register": "clipped, front-loaded, swallows apologies"},
            {"name": "DARROW", "role": "village liaison",
             "want": "to keep the village calm one more season",
             "pressure": "his cousin farms the north slope",
             "register": "slow warm circling, answers with stories"},
        ],
        "turn": (
            "The heat readings Sela hid to protect her credibility are "
            "the only proof that would move the village in time."),
        "priced_ending": {
            "choice": "Sela publishes the readings under her own name",
            "cost_paid": "her seat on the committee, surrendered in writing",
        },
        "news_thread": (
            "new vents on Mount Etna released measurable heat before "
            "any tremor"),
        "news_close_read": (
            "Tonight's story grew from a real survey: instruments on "
            "Mount Etna mapped 1,200 new vents this season, work led by "
            "Doctor Rossi."),
    }
    d.update(over)
    return d


def _treatment(**over) -> F2.Treatment:
    return F2.Treatment.model_validate(_treatment_dict(**over))


# ---------------------------------------------------------------------------
# 1. Models (doc s5)
# ---------------------------------------------------------------------------

class TestModels:
    def test_dossier_shape(self):
        d = _dossier()
        assert len(d.facts_to_keep) == 3
        assert d.named_entities.places == ["Mount Etna"]

    def test_dossier_fact_length_gates(self):
        with pytest.raises(ValidationError):
            _dossier(facts_to_keep=["too short", "also tiny", "x" * 7])

    def test_dossier_vector_count_gate(self):
        with pytest.raises(ValidationError):
            _dossier(dramatizable_vectors=["only one vector here, sadly"])

    def test_cast_shape_name_case_canonicalizes(self):
        # Label normalization canonicalizes case (labels are uppercase
        # by convention; the words stay the LLM's).
        shape = F2.CastShape(name="Sela", role="scientist",
                             want="to be believed today",
                             pressure="the committee doubts",
                             register="clipped and dry")
        assert shape.name == "SELA"

    def test_cast_shape_name_label_normalizes_to_one_word(self):
        # 12th + 19th live smokes: the model titles its scientists
        # reflexively; python strips the banned title tokens and keeps
        # the surname (the LLM's own choice) -- deterministic label
        # normalization, never a new character.
        shape = F2.CastShape(name="DR. VERONICA VOSS", role="scientist",
                             want="to be believed today",
                             pressure="the committee doubts",
                             register="clipped and dry")
        assert shape.name == "VOSS"
        shape2 = F2.CastShape(name="Professor Harris", role="mentor",
                              want="to secure the funding",
                              pressure="investors expect news",
                              register="calm and measured")
        assert shape2.name == "HARRIS"
        with pytest.raises(ValidationError):
            F2.CastShape(name="DR.", role="scientist",
                         want="to be believed today",
                         pressure="the committee doubts",
                         register="clipped and dry")

    def test_cast_shape_name_never_announcer(self):
        with pytest.raises(ValidationError, match="ANNOUNCER"):
            F2.CastShape(name="ANNOUNCER", role="narrator",
                         want="to narrate the show",
                         pressure="the clock is short",
                         register="velvet and warm")

    def test_treatment_needs_question_mark(self):
        with pytest.raises(ValidationError, match="[?]"):
            _treatment(dramatic_question="Sela trusts her instruments now.")

    def test_treatment_registers_must_differ_in_mechanism(self):
        # BOTH directions of _REGISTER_OVERLAP_MAX (0.5): identical
        # registers fail; mechanically distinct ones pass (base fixture).
        shapes = _treatment_dict()["cast_shapes"]
        shapes[1]["register"] = shapes[0]["register"]
        with pytest.raises(ValidationError, match="registers"):
            _treatment(cast_shapes=shapes)
        _treatment()  # distinct registers: valid

    def test_treatment_priced_ending_required_both_keys(self):
        with pytest.raises(ValidationError, match="priced_ending"):
            _treatment(priced_ending={"choice": "she publishes the data"})

    def test_treatment_cast_names_unique(self):
        shapes = _treatment_dict()["cast_shapes"]
        shapes[1]["name"] = "SELA"
        with pytest.raises(ValidationError, match="unique"):
            _treatment(cast_shapes=shapes)

    def test_critic_note_never_replacement_dialogue(self):
        with pytest.raises(ValidationError, match="replacement dialogue"):
            F2.CriticNote(scene=1, speaker="SELA", problem="register_bleed",
                          note="Change it to this. SELA: I never doubted "
                               "the mountain for a second.")

    def test_an_ageless_character_can_say_so(self):
        """Some characters have no age. The schema must let the model be honest.

        Live 2026-07-14, 420w scifi_fable2: the model cast a character with no
        meaningful age -- a machine, a computer, a voice -- and the schema offered
        only 20s/30s/40s/50s/60s. It wrote `age_band: "N/A"` and the episode DIED
        on a literal_error.

        Nothing downstream needs a decade: assign_voice_for_slot takes
        `age_band: str = ""` and its match ladder explicitly DROPS AGE as its second
        tier, and the voice bank's own vocabulary is "adult"/"elder" -- so a decade
        never exact-matches anyway. "n/a" is a truthful value with a defined
        consequence: cast on gender and timbre instead.
        """
        def _voice(**over):
            row = {
                "name": "LAMPKEEPER",
                "role": "The Lighthouse Computer",
                "character_description": (
                    "A patient machine intelligence that has kept the lamp lit "
                    "alone for thirty years and speaks only when asked."),
                "gender": "female",
                "timbre": "cool, level",
            }
            row.update(over)
            return row

        # The exact string that killed the leg.
        assert F2.CastVoice(**_voice(age_band="N/A")).age_band == "n/a"
        # ...and its cousins.
        for spelling in ("n/a", "NA", "none", "unknown", "", "-"):
            assert F2.CastVoice(**_voice(age_band=spelling)).age_band == "n/a"
        # Omitting it entirely is legal paperwork too (the sibling fields' lesson).
        assert F2.CastVoice(**_voice()).age_band == "n/a"
        # A real decade still means what it says.
        assert F2.CastVoice(**_voice(age_band="40s")).age_band == "40s"
        # Junk is still junk.
        with pytest.raises(ValidationError):
            F2.CastVoice(**_voice(age_band="middle-aged-ish"))

    def test_strips_the_chatter_a_model_wraps_around_the_script(self):
        """A reasoning model announces itself. That must not kill a good script.

        Live 2026-07-14, 420w scifi_fable2: aion-3.0-mini opened the revision with
        "I will now produce the complete revised radio play episode, applying all
        critic notes..." -> BAD_LINE_SHAPE (line 1) + SKELETON_BREAK (TITLE is not
        the first line). The markup ladder is a RAW-TEXT pass with no schema to
        constrain shape, so it just re-asked -- five times, five full episode
        generations -- and the model politely announced itself every time. The leg
        died over one sentence of throat-clearing in front of a fine script.

        The format's own skeleton says TITLE: first and END. last, so anything
        outside that is not script. Python owns the SHAPE; the LLM owns the WORDS.
        """
        script = "TITLE: The Ash\nMUSIC: up\nANNOUNCER: Good evening.\nEND."

        wrapped = (
            "I will now produce the complete revised radio play episode, "
            "applying all critic notes.\n\n" + script +
            "\n\nI hope this meets your requirements!"
        )
        assert F2._strip_conversational_wrapper(wrapped) == script

    def test_a_clean_script_is_returned_byte_identical(self):
        script = "TITLE: The Ash\nMUSIC: up\nANNOUNCER: Good evening.\nEND."
        assert F2._strip_conversational_wrapper(script) is script

    def test_a_script_with_no_title_still_fails_loud(self):
        """Fail-safe: never massage a genuinely malformed draft into something
        else. With no TITLE: line there is no script to find, so the text passes
        through UNCHANGED and the parser reports its real defects."""
        junk = "I could not write the episode. Sorry!"
        assert F2._strip_conversational_wrapper(junk) == junk

    def test_critic_note_speaker_is_optional_and_defaults_to_empty(self):
        """A scene-wide note has no speaker to name, and must be able to say so.

        THE LAW: an audit may improve a story, it may never fail one.

        Live 2026-07-14, 420w scifi_fable2 leg: the critic wrote a note about a
        scene rather than a character, omitted `speaker`, the typed repair omitted
        it again, and pydantic killed the episode with
        `notes.0.speaker Field required`.

        Empty was ALWAYS a legal value -- `_note_scope_error` accepts it on both
        branches -- so the schema was demanding a key its own consumer says may be
        empty. That is the unstated-contract class: `schema_shape_instruction`
        emits key NAMES only, so required-vs-optional is invisible to the model. A
        semantically optional field must BE optional.
        """
        note = F2.CriticNote(
            scene=1, problem="stakes_sag",
            note="The turn lands without cost; let the scene charge for it.",
        )
        assert note.speaker == ""

        # And the consumer agrees: an empty speaker is in scope for a real scene.
        assert F2.CriticNotes(verdict="revise", notes=[note]).notes[0].speaker == ""

    def test_critic_revise_requires_a_note(self):
        with pytest.raises(ValidationError, match="revise"):
            F2.CriticNotes(verdict="revise", notes=[])
        F2.CriticNotes(verdict="ship", notes=[])  # legitimate verdict

    def test_critic_ship_rejects_notes(self):
        with pytest.raises(ValidationError, match="ship"):
            F2.CriticNotes(verdict="ship", notes=[{
                "scene": 1, "speaker": "SELA",
                "problem": "subtext_flat",
                "note": "Make the exchange less literal without new facts.",
            }])

    @pytest.mark.parametrize("value", ["1", True, 1.0])
    def test_pitch_and_selection_ids_reject_int_coercion(self, value):
        with pytest.raises(ValidationError):
            F2.Pitch.model_validate(_pitch_dict(value, "Card A"))
        with pytest.raises(ValidationError):
            F2.PitchSelect(
                chosen_pitch_id=value,
                selection_rationale=(
                    "This pitch has the clearest audible conflict and payoff."
                ),
            )

    def test_scene_zero_critic_note_requires_explicit_frame_scope(self):
        with pytest.raises(ValidationError, match="frame_scope"):
            F2.CriticNote(
                scene=0, speaker="", problem="announcer_contract",
                note="Tighten the announcer frame without changing the coda.",
            )
        F2.CriticNote(
            scene=0, frame_scope="intro", speaker="",
            problem="announcer_contract",
            note="Tighten the opening frame without changing the coda.",
        )
        with pytest.raises(ValidationError, match="only legal"):
            F2.CriticNote(
                scene=1, frame_scope="outro", speaker="SELA",
                problem="subtext_flat",
                note="Tighten the exchange without changing its facts.",
            )

    @pytest.mark.parametrize("value", ["0", True, 1.0])
    def test_critic_scene_rejects_int_coercion(self, value):
        with pytest.raises(ValidationError):
            F2.CriticNote(
                scene=value, frame_scope="intro" if value in ("0",) else None,
                speaker="", problem="announcer_contract",
                note="Tighten the announcer frame without changing the coda.",
            )

    def test_critic_problem_class_enum_closed(self):
        with pytest.raises(ValidationError):
            F2.CriticNote(scene=1, frame_scope=None, speaker="",
                          problem="vibes_off",
                          note="not a registry class at all")


# ---------------------------------------------------------------------------
# 2. post_validator factories (r2/S1)
# ---------------------------------------------------------------------------

class TestDossierValidator:
    def test_grounded_dossier_passes(self):
        assert F2._make_dossier_validator(_DIGEST)(_dossier()) is None

    def test_unverifiable_number_is_dropped_not_rerolled(self):
        # 30th live smoke (2026-07-10): converted/computed figures are
        # unbounded -- python DROPS what the source cannot corroborate;
        # the read's numeral legality only ever SHRINKS.
        filtered, dropped = F2._filter_dossier_entities(
            _dossier(allowed_numbers=["7,777", "1,200"]), _DIGEST)
        assert dropped == ["7,777"]
        assert filtered.allowed_numbers == ["1,200"]
        assert F2._make_dossier_validator(_DIGEST)(
            _dossier(allowed_numbers=["7,777"])) is None

    def test_unverifiable_entity_is_dropped_not_rerolled(self):
        # 28th live smoke (2026-07-10): world-knowledge expansions (CMU
        # -> Carnegie Mellon) are unbounded -- python DROPS what the
        # source cannot corroborate (delete-only) instead of killing the
        # run; the read corpus never widens with them.
        bad = _dossier(named_entities={
            "people": ["Professor Nobody", "Doctor Rossi"],
            "places": [], "things": []})
        filtered, dropped = F2._filter_dossier_entities(bad, _DIGEST)
        assert dropped == ["Professor Nobody"]
        assert filtered.named_entities.people == ["Doctor Rossi"]
        # the validator itself no longer rejects entities
        assert F2._make_dossier_validator(_DIGEST)(bad) is None

    def test_spelled_numbers_legalize_extracted_digits(self):
        # 16th live smoke (2026-07-10): "seven days" / "Eighth day" in
        # the story legalize the extracted digits 7 and 8; an
        # uncorroborated 9 drops.
        digest = ("The heat lasted seven days, the eighth day this year "
                  "on Mount Etna, Doctor Rossi said. 1,200. heat sensors.")
        filtered, dropped = F2._filter_dossier_entities(
            _dossier(allowed_numbers=["7", "8", "9", "1,200"]), digest)
        assert dropped == ["9"]
        assert filtered.allowed_numbers == ["7", "8", "1,200"]

    def test_demonym_inflection_is_kept(self):
        # 24th live smoke (2026-07-10): 'Scottish' in the story vs the
        # faithful extraction 'Scotland' -- same referent, not invention.
        ok = _dossier(named_entities={
            "people": ["Doctor Rossi"], "places": ["Scotland"],
            "things": ["heat sensors"]})
        filtered, dropped = F2._filter_dossier_entities(
            ok,
            "The Scottish government provides a support package for pig "
            "farmers. Doctor Rossi. Mount Etna. 1,200. heat sensors.")
        assert dropped == []
        assert filtered.named_entities.places == ["Scotland"]

    def test_reordered_entity_phrase_is_kept(self):
        # 5th live smoke (2026-07-10): "Amsterdam's canals" vs the
        # story's "the canals of Amsterdam" -- token-level presence, not
        # contiguous-phrase substring; 'Rotterdam harbor' still drops.
        digest = (
            "Researchers steered the robots through the canals of "
            "Amsterdam before any tremor. Doctor Rossi led the team. "
            "Mount Etna. 1,200. heat sensors.")
        mixed = _dossier(named_entities={
            "people": ["Doctor Rossi"],
            "places": ["Amsterdam's canals", "Rotterdam harbor"],
            "things": ["heat sensors"]})
        filtered, dropped = F2._filter_dossier_entities(mixed, digest)
        assert dropped == ["Rotterdam harbor"]
        assert filtered.named_entities.places == ["Amsterdam's canals"]


def _pitch_dict(pid: int, card: str, **over) -> dict:
    d = {
        "pitch_id": pid, "frame_card": card,
        "logline": "A scientist must out-argue her own silence before "
                   "the mountain speaks first.",
        "hook": "The instruments heard it three days before anyone.",
        "scifi_device": "heat sensors that map a volcano's breath",
        "cast_size": 2, "ending_shape": "quiet_loss",
    }
    d.update(over)
    return d


_CARDS3 = (
    {"name": "Card A", "shape": "one operator against a large sky"},
    {"name": "Card B", "shape": "a committee room where truth arrives late"},
    {"name": "Card C", "shape": "a village that reads weather as omens"},
)


class TestPitchValidator:
    def test_one_pitch_mode_accepts_exactly_one(self):
        check = F2._make_pitch_validator(_CARDS3[:1], F2._COMPACT_MODE, 4)
        slate = F2.PitchSlate(pitches=[_pitch_dict(1, "Card A")])
        assert check(slate) is None

    def test_one_pitch_mode_rejects_three(self):
        check = F2._make_pitch_validator(_CARDS3[:1], F2._COMPACT_MODE, 4)
        slate = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A"), _pitch_dict(2, "Card A"),
            _pitch_dict(3, "Card A")])
        assert "exactly 1" in (check(slate) or "")

    @pytest.mark.parametrize("ids", ([1, 1, 3], [1, 3, 2], [2, 1, 3]))
    def test_full_mode_pitch_ids_are_exact_and_ordered(self, ids):
        check = F2._make_pitch_validator(_CARDS3, F2._FULL_MODE, 4)
        slate = F2.PitchSlate(pitches=[
            _pitch_dict(ids[0], "Card A",
                        logline="One operator chooses whether to trust a "
                                "quiet instrument before dawn arrives."),
            _pitch_dict(ids[1], "Card B", cast_size=3,
                        ending_shape="ironic_turn",
                        logline="A committee discovers its careful minutes "
                                "cannot schedule the mountain's answer."),
            _pitch_dict(ids[2], "Card C", cast_size=4,
                        ending_shape="open_question",
                        logline="A village weighs old weather signs against "
                                "a newly measured breath at dusk."),
        ])
        assert "pitch ids must be exactly" in (check(slate) or "")

    def test_selection_validator_requires_exact_slate_and_member(self):
        slate = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A"),
            _pitch_dict(2, "Card B", cast_size=3,
                        ending_shape="ironic_turn",
                        logline="A committee hears a warning hidden inside "
                                "the minutes it already approved."),
            _pitch_dict(3, "Card C", cast_size=4,
                        ending_shape="open_question",
                        logline="A village chooses which measure of the "
                                "mountain deserves its trust."),
        ])
        check = F2._make_select_validator(slate)
        assert check(F2.PitchSelect(
            chosen_pitch_id=2,
            selection_rationale=(
                "It has the clearest audible conflict and a payable ending."
            ),
        )) is None
        bad = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A"),
            _pitch_dict(1, "Card B", cast_size=3,
                        ending_shape="ironic_turn",
                        logline="A different committee story carries enough "
                                "words to satisfy the schema."),
            _pitch_dict(3, "Card C", cast_size=4,
                        ending_shape="open_question",
                        logline="A third village story uses a separate set "
                                "of words for the test."),
        ])
        with pytest.raises(F2.Fable2SelectError, match="exact pitch slate"):
            F2._make_select_validator(bad)

    def test_pitch_i_is_restored_to_card_i(self):
        """PYTHON dealt the cards. `frame_card` is the model echoing our own
        string back -- so a mis-copied label is restored, not fatal.
        """
        check = F2._make_pitch_validator(_CARDS3, F2._FULL_MODE, 4)
        slate = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A"),
            _pitch_dict(2, "Card C",
                        logline="A liaison learns the mountain keeps books "
                                "no committee can audit tonight.",
                        cast_size=3, ending_shape="ironic_turn"),
            _pitch_dict(3, "Card C",
                        logline="Omens and instruments race to name the "
                                "same dawn before the village wakes.",
                        cast_size=4, ending_shape="open_question"),
        ])
        assert check(slate) is None
        assert [p.frame_card for p in slate.pitches] == [
            c["name"] for c in _CARDS3
        ]

    def test_cast_size_over_n_max_rejected(self):
        check = F2._make_pitch_validator(_CARDS3[:1], F2._COMPACT_MODE, 2)
        slate = F2.PitchSlate(pitches=[_pitch_dict(1, "Card A", cast_size=5)])
        assert "N_MAX" in (check(slate) or "")

    def test_full_mode_divergence_gate(self):
        check = F2._make_pitch_validator(_CARDS3, F2._FULL_MODE, 4)
        slate = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A",
                        logline="First distinct premise about a stubborn "
                                "operator and the sky she reads."),
            _pitch_dict(2, "Card B",
                        logline="Second distinct premise where a committee "
                                "meets a truth it scheduled poorly."),
            _pitch_dict(3, "Card C",
                        logline="Third distinct premise in which omens "
                                "out-vote the seismograph at dusk."),
        ])
        # all three share (cast_size, ending_shape) -> must diverge
        err = check(slate)
        assert err and "diverge" in err

    def test_full_mode_logline_overlap_gate_both_directions(self):
        check = F2._make_pitch_validator(_CARDS3, F2._FULL_MODE, 4)
        base = "A scientist must out-argue her own silence before the mountain speaks first tonight."
        near = "A scientist must out-argue her own silence before the mountain speaks last tonight."
        slate = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A", logline=base),
            _pitch_dict(2, "Card B", logline=near, cast_size=3,
                        ending_shape="ironic_turn"),
            _pitch_dict(3, "Card C",
                        logline="Entirely different words carry premise "
                                "number three through omen country.",
                        cast_size=4, ending_shape="open_question"),
        ])
        err = check(slate)
        assert err and "overlap" in err
        # distinct loglines (base fixture from the divergence test with
        # varied shapes) pass:
        ok = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A",
                        logline="First distinct premise about a stubborn "
                                "operator and the sky she reads."),
            _pitch_dict(2, "Card B", cast_size=3, ending_shape="ironic_turn",
                        logline="Second distinct premise where a committee "
                                "meets a truth it scheduled poorly."),
            _pitch_dict(3, "Card C", cast_size=4,
                        ending_shape="open_question",
                        logline="Third distinct premise in which omens "
                                "out-vote the seismograph at dusk."),
        ])
        assert check(ok) is None

    def test_sfw_lexicon_gate_word_boundary(self):
        check = F2._make_pitch_validator(_CARDS3[:1], F2._COMPACT_MODE, 4)
        bad = F2.PitchSlate(pitches=[_pitch_dict(
            1, "Card A",
            hook="She keeps a revolver in the instrument drawer.")])
        assert "revolver" in (check(bad) or "")
        # word-boundary: 'begun' must NOT trip 'gun'
        ok = F2.PitchSlate(pitches=[_pitch_dict(
            1, "Card A", hook="The count has begun under the ridge.")])
        assert check(ok) is None


class TestTreatmentValidator:
    """P2b gates only -- the factual-read subset laws moved to
    TestReadValidator with the S1b read-split (kibitz r2 Q1)."""

    def _check(self):
        return F2._make_treatment_validator(_dossier(), 4, _PROVENANCE)

    def test_grounded_treatment_passes(self):
        assert self._check()(_treatment()) is None

    def test_cast_over_ceiling_rejected(self):
        check = F2._make_treatment_validator(_dossier(), 1, _PROVENANCE)
        assert "N_MAX" in (check(_treatment()) or "")

    def test_ungrounded_news_thread_rejected(self):
        err = self._check()(_treatment(
            news_thread="a haunted lighthouse keeps its own visitor log"))
        assert err and "news_thread" in err

    def test_sfw_gate(self):
        shapes = _treatment_dict()["cast_shapes"]
        shapes[0]["pressure"] = "the committee wants her cigarette case"
        err = self._check()(_treatment(cast_shapes=shapes))
        assert err and "cigarette" in err

    def test_empty_news_close_read_is_legal_at_p2b(self):
        # S1b read-split: P2b no longer writes the read.
        assert self._check()(_treatment(news_close_read="")) is None

    def test_format_example_names_are_reserved(self):
        # kibitz r4 M1 (roll 22 aired the few-shot example's VERA/DOKU
        # cast): example names are reserved unless the SOURCE story
        # itself carries the name.
        shapes = _treatment_dict()["cast_shapes"]
        shapes[0]["name"] = "VERA"
        err = self._check()(_treatment(cast_shapes=shapes))
        assert err and "FORMAT-example" in err
        # source-authorized: the story really is about a Vera
        check = F2._make_treatment_validator(
            _dossier(), 4, _PROVENANCE,
            digest=_DIGEST + " Vera Kowalska led the survey.")
        assert check(_treatment(cast_shapes=shapes)) is None


def _read(text: str) -> F2.NewsCloseRead:
    return F2.NewsCloseRead(news_close_read=text)


class TestReadValidator:
    """P2c factual-close subset laws (S1b read-split; every case below
    is a live-smoke scar from 2026-07-10)."""

    def _check(self, dossier=None, digest=_DIGEST, cast=("SELA", "DARROW")):
        return F2._make_read_validator(
            dossier or _dossier(), _PROVENANCE, digest, list(cast))

    def test_grounded_read_passes(self):
        assert self._check()(_read(
            "Tonight's story grew from a real survey: instruments on "
            "Mount Etna mapped 1,200 new vents this season, work led by "
            "Doctor Rossi.")) is None

    def test_numeral_subset_law(self):
        # r4/S2 subset direction: '1,200' passes (base); '7' fails.
        err = self._check()(_read(
            "Tonight's story grew from a real survey: 7 teams mapped new "
            "vents on Mount Etna this season, led by Doctor Rossi."))
        assert err and "'7'" in err

    def test_numeral_check_is_token_exact_never_substring(self):
        # kibitz r2 M3: '20' must not ride inside '1,200' or a date.
        err = self._check()(_read(
            "Tonight's story grew from a real survey: 20 teams mapped "
            "1,200 new vents on Mount Etna, work led by Doctor Rossi."))
        assert err and "'20'" in err

    def test_sentence_initial_invented_name_is_caught(self):
        # kibitz r2 M2: the old sentence-initial skip was an invention
        # hole -- 'Ganymede' at sentence start must die.
        err = self._check()(_read(
            "Tonight's story grew from a real survey. Ganymede teams "
            "mapped 1,200 new vents on Mount Etna, work led by "
            "Doctor Rossi."))
        assert err and "Ganymede" in err

    def test_validator_accumulates_all_errors(self):
        # kibitz r2 S3: one repair attempt must see the WHOLE target.
        err = self._check()(_read(
            "Tonight's story grew from a real survey: 7 teams from "
            "Geneva mapped new vents on Mount Etna, led by Doctor "
            "Rossi."))
        assert err and "'7'" in err and "Geneva" in err

    def test_proper_noun_subset_law(self):
        err = self._check()(_read(
            "Tonight's story grew from a real survey: instruments in "
            "Geneva mapped 1,200 new vents this season, led by Doctor "
            "Rossi."))
        assert err and "Geneva" in err

    def test_honorifics_are_legal(self):
        # 6th live smoke: 'Dr' is a title, not an entity.
        assert self._check()(_read(
            "Tonight's story grew from a real survey: instruments on "
            "Mount Etna mapped 1,200 new vents this season, said "
            "Dr. Rossi of the instrument team.")) is None

    def test_never_names_the_fictional_cast(self):
        # 9th live smoke: the read is REAL NEWS.
        err = self._check()(_read(
            "Tonight's story grew from a real survey: Sela watched "
            "instruments on Mount Etna map 1,200 new vents this season, "
            "work led by Doctor Rossi."))
        assert err and "SELA" in err and "REAL NEWS" in err

    def test_cast_name_gate_is_case_sensitive(self):
        # 11th live smoke: a character named HOPE must not outlaw the
        # common word 'hope'.
        assert self._check(cast=("HOPE", "DARROW"))(_read(
            "Tonight's story grew from a real survey: instruments on "
            "Mount Etna mapped 1,200 new vents this season, and the "
            "team's hope is a calmer year, said Doctor Rossi.")) is None
        err = self._check(cast=("HOPE", "DARROW"))(_read(
            "Tonight's story grew from a real survey: Hope watched "
            "instruments on Mount Etna map 1,200 new vents this season, "
            "work led by Doctor Rossi."))
        assert err and "HOPE" in err

    def test_cast_name_borrowed_from_the_real_source_stays_legal(self):
        # 14th live smoke (2026-07-10): the drama named its character
        # after the REAL person in the story ('JIM' for NASA's Jim
        # Ross) -- factual reporting of a source name is never leakage.
        assert self._check(cast=("ROSSI", "DARROW"))(_read(
            "Tonight's story grew from a real survey: instruments on "
            "Mount Etna mapped 1,200 new vents this season, work led by "
            "Doctor Rossi.")) is None

    def test_source_digest_widens_the_corpus(self):
        # 8th live smoke: legality authority = the SOURCE.
        check = self._check(
            digest=_DIGEST + " The Catania observatory logged 88 tremors.")
        assert check(_read(
            "Tonight's story grew from a real survey: the Catania "
            "observatory logged 88 tremors while instruments mapped "
            "1,200 new vents on Mount Etna.")) is None
        err = check(_read(
            "Tonight's story grew from a real survey: the Catania "
            "observatory logged 99 tremors while instruments mapped "
            "1,200 new vents on Mount Etna."))
        assert err and "'99'" in err

    def test_spelled_source_numbers_legalize_read_digits(self):
        # 23rd live smoke (2026-07-10): the story spelled 'twenty'; the
        # read's digit 20 is the same fact.
        check = self._check(
            digest=_DIGEST + " The county deployed twenty heat teams.")
        assert check(_read(
            "Tonight's story grew from a real survey: 20 heat teams "
            "joined the effort as instruments on Mount Etna mapped "
            "1,200 new vents, work led by Doctor Rossi.")) is None

    def test_possessive_of_dossier_noun_is_legal(self):
        # 7th live smoke: "Rossi's" is grammar, not a new entity.
        assert self._check()(_read(
            "Tonight's story grew from a real survey: Rossi's instrument "
            "team mapped 1,200 new vents on Mount Etna this season, a "
            "record for the observatory.")) is None

    def test_calendar_words_are_legal(self):
        # 1st live smoke: a legitimate read died on 'June'.
        assert self._check()(_read(
            "Tonight's story grew from a real survey: through June, "
            "instruments on Mount Etna mapped 1,200 new vents, work led "
            "by Doctor Rossi.")) is None

    def test_nouns_from_dossier_facts_are_legal(self):
        # 2nd live smoke: 'ISS'-class names live in the FACTS.
        dossier = _dossier(facts_to_keep=[
            "A survey mapped 1,200 new vents on Mount Etna this season.",
            "The Copernicus lander relayed the vent map to Doctor Rossi.",
            "The vents released measurable heat before any tremor.",
        ])
        assert self._check(dossier=dossier)(_read(
            "Tonight's story grew from a real survey: the Copernicus "
            "lander relayed a map of 1,200 new vents on Mount Etna, "
            "work led by Doctor Rossi.")) is None

    def test_noun_check_is_word_boundary(self):
        # 'ISS' must never ride inside 'mission'.
        dossier = _dossier(facts_to_keep=[
            "A survey mapped 1,200 new vents on Mount Etna this season.",
            "The mission plan kept Doctor Rossi on the north slope.",
            "The vents released measurable heat before any tremor.",
        ])
        err = self._check(dossier=dossier)(_read(
            "Tonight's story grew from a real survey: the ISS relayed "
            "readings of 1,200 new vents mapped on Mount Etna by "
            "Doctor Rossi."))
        assert err and "ISS" in err


class TestCastingValidator:
    def _menu(self):
        return F2.VoiceMenu(entries=(
            F2.VoiceMenuEntry("m01", "female", "female clipped 40s", "p1"),
            F2.VoiceMenuEntry("m02", "male", "male warm 50s", "p2"),
            F2.VoiceMenuEntry("m03", "male", "male dry 30s", "p3"),
        ))

    def _voice(self, name, gender, timbre) -> dict:
        return {
            "name": name, "role": "scientist on the ridge",
            "character_description": (
                "Mid-forties, wind-burned, one pencil behind the ear, "
                "taps the barometer twice before speaking."),
            "gender": gender, "age_band": "40s",
            "register": "clipped and front-loaded",
            "timbre": timbre,
            "want": "to be believed before dawn",
            "pressure": "the committee reads her charts as noise",
        }

    def test_valid_casting_passes(self):
        check = F2._make_casting_validator(self._menu(), ["SELA", "DARROW"])
        cv = F2.CastingVoices(cast=[
            self._voice("SELA", "female", "m01"),
            self._voice("DARROW", "male", "m02")])
        assert check(cv) is None

    def test_speaker_set_equality_gate(self):
        check = F2._make_casting_validator(self._menu(), ["SELA", "DARROW"])
        cv = F2.CastingVoices(cast=[self._voice("SELA", "female", "m01")])
        assert "speakers" in (check(cv) or "")

    def test_unknown_menu_id_rejected(self):
        check = F2._make_casting_validator(self._menu(), ["SELA"])
        cv = F2.CastingVoices(cast=[self._voice("SELA", "female", "m99")])
        assert "m99" in (check(cv) or "")

    def test_gender_mismatch_rejected(self):
        check = F2._make_casting_validator(self._menu(), ["SELA"])
        cv = F2.CastingVoices(cast=[self._voice("SELA", "female", "m02")])
        assert "gender" in (check(cv) or "")

    def test_duplicate_timbre_rejected(self):
        check = F2._make_casting_validator(self._menu(), ["SELA", "DARROW"])
        cv = F2.CastingVoices(cast=[
            self._voice("SELA", "male", "m02"),
            self._voice("DARROW", "male", "m02")])
        assert "already taken" in (check(cv) or "")

    def test_casting_accepts_unwrapped_shapes(self):
        # 18th live smoke (2026-07-10): a truncated response salvaged as
        # a bare list / single entry must reach the TEACHING gates, not
        # die as a raw schema error.
        bare_list = F2.CastingVoices.model_validate(
            [self._voice("SELA", "female", "m01")])
        assert [c.name for c in bare_list.cast] == ["SELA"]
        single = F2.CastingVoices.model_validate(
            self._voice("DARROW", "male", "m02"))
        assert [c.name for c in single.cast] == ["DARROW"]


# ---------------------------------------------------------------------------
# 3. Envelope + token budget + entry gates (doc s3/s13)
# ---------------------------------------------------------------------------

class TestEnvelopeAndBudgets:
    def test_scene_formula(self):
        assert F2._build_envelope(30).scene_count == 1
        assert F2._build_envelope(350).scene_count == 3
        assert F2._build_envelope(880).scene_count == 8
        assert F2._build_envelope(10).scene_count == 1  # clamp low

    @pytest.mark.parametrize(("target", "mode"), [
        (30, "one_pitch_one_draft"),
        (119, "one_pitch_one_draft"),
        (120, "three_pitch_two_draft"),
        (320, "three_pitch_two_draft"),
        (350, "three_pitch_two_draft"),
        (420, "three_pitch_two_draft"),
        (720, "three_pitch_two_draft"),
        (900, "three_pitch_two_draft"),
    ])
    def test_s2_mode_boundary_matrix(self, target, mode):
        assert F2._resolve_mode(target) == mode

    def test_s2_mode_rejects_901(self):
        with pytest.raises(F2.Fable2ScriptError, match="ceiling"):
            F2._resolve_mode(901)
        with pytest.raises(F2.Fable2ScriptError, match="ceiling"):
            F2._build_envelope(901)

    def test_scene_envelope_rejects_noncanonical_direct_construction(self):
        with pytest.raises(ValueError, match="scene_count"):
            F2.SceneEnvelope(
                scene_count=2,
                scene_word_targets=(160, 160),
                total_words=320,
            )
        with pytest.raises(ValueError, match="band_frac"):
            F2.SceneEnvelope(
                scene_count=3,
                scene_word_targets=(107, 107, 106),
                total_words=320,
                band_frac=0.25,
            )

    @pytest.mark.parametrize(("target", "scenes"), [
        (164, 1), (165, 2),
        (274, 2), (275, 3),
        (384, 3), (385, 4),
        (494, 4), (495, 5),
        (604, 5), (605, 6),
        (714, 6), (715, 7),
        (824, 7), (825, 8),
        (900, 8),
    ])
    def test_scene_count_table_boundaries(self, target, scenes):
        assert F2._build_envelope(target).scene_count == scenes

    @pytest.mark.parametrize(("target", "vector"), [
        (120, (120,)),
        (320, (107, 107, 106)),
        (350, (117, 117, 116)),
        (720, (103, 103, 103, 103, 103, 103, 102)),
        (900, (113, 113, 113, 113, 112, 112, 112, 112)),
    ])
    def test_exact_scene_vectors_and_bands(self, target, vector):
        envelope = F2._build_envelope(target)
        assert envelope.scene_word_targets == vector
        assert sum(vector) == target
        assert envelope.scene_count == len(vector)
        assert envelope.scene_word_bands == tuple(
            (int((value * 0.70).__ceil__()),
             int((value * 1.30).__floor__()))
            for value in vector
        )

    def test_token_budget_floor_and_cap(self):
        assert F2._script_token_budget(30) == 1200      # floor
        assert F2._script_token_budget(500) == 1300
        assert F2._script_token_budget(5000) == 4200    # cap

    def test_entry_gate_ceiling(self):
        with pytest.raises(F2.Fable2ScriptError, match="ceiling"):
            F2.assert_supported_target_words(901)

    def test_entry_gate_accepts_compact_and_full_s2_ranges(self):
        F2.assert_supported_target_words(F2._ONE_DRAFT_THRESHOLD - 1)
        for target in (120, 420, 900):
            F2.assert_supported_target_words(target)


def _revision_markup(
    scene_counts=(107, 107, 106),
    first_phrases=("clear answer", "steady answer", "patient answer"),
    *,
    title="The Exact Ledger",
    settings=None,
    opening_music="warm dial tones and strings",
    closing_music="resolved dial tones and strings",
    coda="The careful answer is",
):
    settings = settings or tuple(
        f"instrument room {index}" for index in range(1, len(scene_counts) + 1)
    )
    assert len(settings) == len(scene_counts)
    assert len(first_phrases) >= len(scene_counts)
    rows = [
        f"TITLE: {title}",
        f"MUSIC: {opening_music}",
        "ANNOUNCER: Tonight two careful listeners compare what they heard.",
    ]
    for scene_no, total in enumerate(scene_counts, start=1):
        sela_words = total // 2 + total % 2
        darrow_words = total - sela_words
        phrase = first_phrases[scene_no - 1].split()
        assert sela_words >= len(phrase) and darrow_words >= 1
        sela = phrase + [
            f"s{scene_no}a{index}"
            for index in range(sela_words - len(phrase))
        ]
        darrow = [
            f"s{scene_no}d{index}" for index in range(darrow_words)
        ]
        rows.extend([
            f"SCENE {scene_no}: {settings[scene_no - 1]}",
            "SELA: " + " ".join(sela),
            "DARROW: " + " ".join(darrow),
        ])
        if scene_no < len(scene_counts):
            rows.append(f"MUSIC: brief bridge after scene {scene_no}")
    rows.extend([
        "ANNOUNCER: The ledger closes without hiding its cost.",
        f"CODA: {coda}:",
        f"MUSIC: {closing_music}",
        "END.",
    ])
    return "\n".join(rows)


def _parse_revision(raw: str):
    normalized = F2.normalize_fable2_markup_text(raw)
    parsed, defects = F2.parse_fable2_markup(normalized, ["SELA", "DARROW"])
    assert parsed is not None, F2.render_defects(defects)
    return normalized, parsed


def _fable2_rules():
    STORY_RULES._clear_caches()
    return STORY_RULES.resolve_story_rules("scifi_fable2")


def _revision_notes(scene=1, speaker="SELA", *, frame_scope=None):
    note = {
        "scene": scene,
        "speaker": speaker,
        "problem": "subtext_flat",
        "note": "Replace the literal exchange with a more playable response.",
    }
    if frame_scope is not None:
        note["frame_scope"] = frame_scope
    return F2.CriticNotes(verdict="revise", notes=[note])


class TestRevisionContracts:
    def test_authorized_strictly_better_revision_wins(self):
        raw1, parsed1 = _parse_revision(_revision_markup(
            first_phrases=("the void", "steady answer", "patient answer")))
        raw2, parsed2 = _parse_revision(_revision_markup())
        envelope = F2._build_envelope(320)
        rules = _fable2_rules()
        notes = _revision_notes()
        contract = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, notes, envelope, rules)
        assert contract.eligible is True
        assert contract.addressed_problem_classes == ("subtext_flat",)
        assert contract.changed_scopes == (
            "scene:1:speaker:SELA:line:4",
        )
        assert contract.contract_sha256 == F2._canonical_sha256(
            F2._revision_contract_payload(contract))
        draft1 = F2.build_final_draft(
            raw1, parsed1, _treatment(), envelope, rules)
        draft2 = F2.build_final_draft(
            raw2, parsed2, _treatment(), envelope, rules)
        decision = F2.choose_final_draft(draft1, draft2, contract)
        assert decision.winner is draft2
        assert "strictly lower" in decision.reason
        assert decision.scoring_context_sha256 == (
            contract.scoring_context_sha256)

    @pytest.mark.parametrize(("old_phrase", "new_phrase", "winner", "reason"), [
        ("clear answer", "plain answer", "draft1", "score tie"),
        ("clear answer", "the void", "draft1", "is worse"),
    ])
    def test_tie_and_worse_revision_keep_draft1(
            self, old_phrase, new_phrase, winner, reason):
        raw1, parsed1 = _parse_revision(_revision_markup(
            first_phrases=(old_phrase, "steady answer", "patient answer")))
        raw2, parsed2 = _parse_revision(_revision_markup(
            first_phrases=(new_phrase, "steady answer", "patient answer")))
        envelope = F2._build_envelope(320)
        rules = _fable2_rules()
        contract = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(), envelope, rules)
        assert contract.eligible
        draft1 = F2.build_final_draft(
            raw1, parsed1, _treatment(), envelope, rules)
        draft2 = F2.build_final_draft(
            raw2, parsed2, _treatment(), envelope, rules)
        decision = F2.choose_final_draft(draft1, draft2, contract)
        assert decision.winner is draft1
        assert reason in decision.reason

    @pytest.mark.parametrize(("label", "mutate"), [
        ("title", lambda raw: raw.replace(
            "TITLE: The Exact Ledger", "TITLE: A Different Ledger", 1)),
        ("scene_settings", lambda raw: raw.replace(
            "SCENE 1: instrument room 1",
            "SCENE 1: a different instrument room", 1)),
        ("music_queue", lambda raw: raw.replace(
            "MUSIC: warm dial tones and strings",
            "MUSIC: brass fanfare and drums", 1)),
        ("coda_news_boundary", lambda raw: raw.replace(
            "CODA: The careful answer is:",
            "CODA: An unrelated answer is:", 1)),
        ("speaker_topology", lambda raw: raw.replace(
            "SELA: clear answer", "DARROW: clear answer", 1)),
    ])
    def test_protected_field_drift_is_ineligible(self, label, mutate):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(mutate(_revision_markup()))
        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(),
            F2._build_envelope(320), _fable2_rules())
        assert result.eligible is False
        assert label in result.protected_field_differences

    def test_scene_count_drift_is_ineligible(self):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(_revision_markup(
            scene_counts=(160, 160),
            first_phrases=("clear answer", "steady answer")))
        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(),
            F2._build_envelope(320), _fable2_rules())
        assert result.eligible is False
        assert "scene_count" in result.protected_field_differences

    def test_unnoted_and_named_speaker_scope_escape_are_ineligible(self):
        raw1, parsed1 = _parse_revision(_revision_markup())
        changed = _revision_markup().replace("s1d0", "escapedword", 1)
        raw2, parsed2 = _parse_revision(changed)
        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(1, "SELA"),
            F2._build_envelope(320), _fable2_rules())
        assert result.eligible is False
        assert any("speaker:DARROW" in reason for reason in result.reasons)

    def test_intro_note_does_not_authorize_outro_change(self):
        raw1, parsed1 = _parse_revision(_revision_markup())
        changed = _revision_markup().replace(
            "The ledger closes without hiding its cost.",
            "The ledger rests without hiding its cost.",
            1,
        )
        raw2, parsed2 = _parse_revision(changed)
        notes = F2.CriticNotes(verdict="revise", notes=[{
            "scene": 0,
            "frame_scope": "intro",
            "speaker": "",
            "problem": "announcer_contract",
            "note": "Tighten only the opening frame and preserve the close.",
        }])

        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, notes,
            F2._build_envelope(320), _fable2_rules())

        assert result.eligible is False
        assert any(
            "frame:outro" in reason and "unnoted change" in reason
            for reason in result.reasons
        )

    def test_blank_line_layout_drift_is_ineligible(self):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(
            _revision_markup().replace("SCENE 2:", "\nSCENE 2:", 1))

        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(),
            F2._build_envelope(320), _fable2_rules())

        assert result.eligible is False
        assert any("blank-line layout changed" in reason
                   for reason in result.reasons)

    @pytest.mark.parametrize(("notes", "needle"), [
        (_revision_notes(4, ""), "missing scene 4"),
        (_revision_notes(0, "SELA", frame_scope="intro"), "scene-zero"),
    ])
    def test_invalid_critic_scope_is_ineligible(self, notes, needle):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(_revision_markup(
            first_phrases=("plain answer", "steady answer", "patient answer")))
        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, notes,
            F2._build_envelope(320), _fable2_rules())
        assert result.eligible is False
        assert any(needle in reason for reason in result.reasons)

    def test_total_valid_but_per_scene_invalid_is_ineligible(self):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(_revision_markup(
            scene_counts=(60, 200, 60)))
        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(1, ""),
            F2._build_envelope(320), _fable2_rules())
        assert result.total_word_count == 320
        assert result.eligible is False
        assert any("scene 2 character words" in reason
                   for reason in result.reasons)

    def test_total_out_of_band_is_ineligible(self):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(_revision_markup(
            scene_counts=(180, 180, 180)))
        result = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(1, ""),
            F2._build_envelope(320), _fable2_rules())
        assert result.eligible is False
        assert any("total character words" in reason
                   for reason in result.reasons)

    @pytest.mark.parametrize("bad_rules", [
        None,
        SimpleNamespace(rules_id="science_news", schema_version="v1"),
    ])
    def test_missing_or_mismatched_rules_fail_loud(self, bad_rules):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(_revision_markup(
            first_phrases=("plain answer", "steady answer", "patient answer")))
        with pytest.raises(F2.Fable2ScriptError, match="story_rules"):
            F2.validate_revision_contract(
                raw1, parsed1, raw2, parsed2, _revision_notes(),
                F2._build_envelope(320), bad_rules)

    def test_forged_same_type_rules_pack_fails_canonical_fingerprint(self):
        rules = _fable2_rules()
        forged = replace(rules, cliche=())
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(_revision_markup(
            first_phrases=("plain answer", "steady answer", "patient answer")))

        with pytest.raises(F2.Fable2ScriptError, match="canonical resolved"):
            F2.validate_revision_contract(
                raw1, parsed1, raw2, parsed2, _revision_notes(),
                F2._build_envelope(320), forged)

    def test_regex_like_story_rule_entries_are_not_accepted(self):
        rules = _fable2_rules()

        class BehavioralFakeRegex:
            def __init__(self, source):
                self.pattern = source.pattern
                self.flags = source.flags

            def finditer(self, _text):
                return iter((object(), object(), object()))

        assert rules.cliche
        forged = replace(
            rules,
            cliche=tuple(BehavioralFakeRegex(rx) for rx in rules.cliche),
        )
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(_revision_markup(
            first_phrases=("plain answer", "steady answer", "patient answer")))

        with pytest.raises(F2.Fable2ScriptError, match="non-re.Pattern"):
            F2.validate_revision_contract(
                raw1, parsed1, raw2, parsed2, _revision_notes(),
                F2._build_envelope(320), forged)

    def test_final_draft_rejects_fabricated_parse_normalizations(self):
        raw, parsed = _parse_revision(_revision_markup())
        fabricated = replace(parsed, normalizations=("fabricated",))

        with pytest.raises(F2.Fable2ParseError, match="authoritative raw parse"):
            F2.build_final_draft(
                raw, fabricated, _treatment(), F2._build_envelope(320),
                _fable2_rules())

    def test_final_draft_rejects_duck_typed_scene_envelope(self):
        raw, parsed = _parse_revision(_revision_markup())
        forged = SimpleNamespace(
            scene_count=3,
            scene_word_targets=(107, 107, 106),
            total_words=320,
            band_frac=F2._TOTAL_WORD_BAND,
            scene_word_bands=((74, 139), (74, 139), (75, 137)),
        )

        with pytest.raises(F2.Fable2ScriptError, match="exact canonical"):
            F2.build_final_draft(
                raw, parsed, _treatment(), forged, _fable2_rules())

    def test_final_draft_is_frozen_and_hash_consistent(self):
        raw, parsed = _parse_revision(_revision_markup())
        trace = F2.PassAttemptTrace(
            attempt=1, temperature=0.75, requested_max_new_tokens=1200,
            outcome="accepted", selected=True, character_words=320,
            scene_character_words=(107, 107, 106),
        )
        draft = F2.build_final_draft(
            raw, parsed, _treatment(), F2._build_envelope(320),
            _fable2_rules(), p3_attempts=(trace,))
        assert draft.p3_attempts == (trace,)
        assert draft.hashes.raw_sha256 == F2._sha256(raw)
        assert draft.hashes.normalized_sha256 == F2._sha256(
            draft.normalized_source)
        assert draft.hashes.proof_map_sha256 == F2._canonical_sha256([
            F2._proof_entry_payload(entry) for entry in draft.proof_map
        ])
        assert draft.hashes.artifact_sha256 == F2._canonical_sha256({
            "raw_sha256": draft.hashes.raw_sha256,
            "normalized_sha256": draft.hashes.normalized_sha256,
            "parsed_sha256": draft.hashes.parsed_sha256,
            "proof_map_sha256": draft.hashes.proof_map_sha256,
            "score": list(draft.score),
            "scoring_context_sha256": draft.scoring_context_sha256,
            "p3_attempts": [F2._attempt_payload(trace)],
            "p5_attempts": [],
        })
        assert len({entry.line_id for entry in draft.proof_map}) == len(
            draft.proof_map)
        with pytest.raises(FrozenInstanceError):
            draft.score = (0, 0, 0, 0, 0)

    def test_attempt_telemetry_changes_only_the_sealed_artifact_hash(self):
        raw, parsed = _parse_revision(_revision_markup())
        common = {
            "attempt": 1,
            "requested_max_new_tokens": 1200,
            "outcome": "accepted",
            "selected": True,
            "character_words": 320,
            "scene_character_words": (107, 107, 106),
        }
        first_trace = F2.PassAttemptTrace(temperature=0.75, **common)
        second_trace = F2.PassAttemptTrace(temperature=0.30, **common)
        first = F2.build_final_draft(
            raw, parsed, _treatment(), F2._build_envelope(320),
            _fable2_rules(), p3_attempts=(first_trace,))
        second = F2.build_final_draft(
            raw, parsed, _treatment(), F2._build_envelope(320),
            _fable2_rules(), p3_attempts=(second_trace,))

        assert first.hashes.raw_sha256 == second.hashes.raw_sha256
        assert first.hashes.parsed_sha256 == second.hashes.parsed_sha256
        assert first.hashes.proof_map_sha256 == second.hashes.proof_map_sha256
        assert first.hashes.artifact_sha256 != second.hashes.artifact_sha256

    @pytest.mark.parametrize(("overrides", "error"), [
        ({"outcome": "bogus"}, ValueError),
        ({"parse_defects": ["BAD_LINE_SHAPE"]}, TypeError),
        ({"scene_character_words": [320]}, TypeError),
        ({"outcome": "accepted", "selected": False}, ValueError),
        ({"outcome": "truncated", "truncation": False}, ValueError),
        ({"temperature": float("nan")}, TypeError),
    ])
    def test_attempt_trace_runtime_contract_is_typed_and_immutable(
            self, overrides, error):
        values = {
            "attempt": 1,
            "temperature": 0.30,
            "requested_max_new_tokens": 1200,
            "outcome": "parse_rejected",
        }
        values.update(overrides)
        with pytest.raises(error):
            F2.PassAttemptTrace(**values)

    def test_attempt_sequences_must_be_contiguous_with_one_selected_final(self):
        raw, parsed = _parse_revision(_revision_markup())
        forged = SimpleNamespace(
            attempt=1, temperature=-99, requested_max_new_tokens=1,
            outcome="bogus", truncation=False, selected=True,
            parse_defects=[], character_words=None,
            scene_character_words=[],
        )
        with pytest.raises(F2.Fable2ScriptError, match="PassAttemptTrace"):
            F2.build_final_draft(
                raw, parsed, _treatment(), F2._build_envelope(320),
                _fable2_rules(), p3_attempts=(forged,))

        skipped_first = F2.PassAttemptTrace(
            attempt=2, temperature=0.30, requested_max_new_tokens=1200,
            outcome="accepted", selected=True,
        )
        with pytest.raises(F2.Fable2ScriptError, match="contiguous from 1"):
            F2.build_final_draft(
                raw, parsed, _treatment(), F2._build_envelope(320),
                _fable2_rules(), p3_attempts=(skipped_first,))

        accepted_early = F2.PassAttemptTrace(
            attempt=1, temperature=0.30, requested_max_new_tokens=1200,
            outcome="accepted", selected=True,
        )
        accepted_late = F2.PassAttemptTrace(
            attempt=2, temperature=0.30, requested_max_new_tokens=1200,
            outcome="accepted", selected=True,
        )
        with pytest.raises(F2.Fable2ScriptError, match="selected final"):
            F2.build_final_draft(
                raw, parsed, _treatment(), F2._build_envelope(320),
                _fable2_rules(), p3_attempts=(accepted_early, accepted_late))

    def test_selection_rejects_scoring_context_mismatch(self):
        raw1, parsed1 = _parse_revision(_revision_markup(
            first_phrases=("the void", "steady answer", "patient answer")))
        raw2, parsed2 = _parse_revision(_revision_markup())
        envelope = F2._build_envelope(320)
        rules = _fable2_rules()
        contract = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(), envelope, rules)
        draft1 = F2.build_final_draft(
            raw1, parsed1, _treatment(), envelope, rules)
        draft2 = F2.build_final_draft(
            raw2, parsed2, _treatment(), F2._build_envelope(350), rules)

        with pytest.raises(F2.Fable2ScriptError, match="scoring contexts"):
            F2.choose_final_draft(draft1, draft2, contract)

    def test_selection_rejects_stale_final_draft_score_seal(self):
        raw1, parsed1 = _parse_revision(_revision_markup(
            first_phrases=("the void", "steady answer", "patient answer")))
        raw2, parsed2 = _parse_revision(_revision_markup())
        envelope = F2._build_envelope(320)
        rules = _fable2_rules()
        contract = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(), envelope, rules)
        draft1 = F2.build_final_draft(
            raw1, parsed1, _treatment(), envelope, rules)
        draft2 = F2.build_final_draft(
            raw2, parsed2, _treatment(), envelope, rules)
        forged_score = (draft2.score[0] + 1, *draft2.score[1:])
        forged = replace(draft2, score=forged_score)

        with pytest.raises(F2.Fable2ScriptError, match="hash seal"):
            F2.choose_final_draft(draft1, forged, contract)

    def test_selection_rejects_forged_revision_eligibility(self):
        raw1, parsed1 = _parse_revision(_revision_markup())
        raw2, parsed2 = _parse_revision(
            _revision_markup().replace("s1d0", "escapedword", 1))
        envelope = F2._build_envelope(320)
        rules = _fable2_rules()
        contract = F2.validate_revision_contract(
            raw1, parsed1, raw2, parsed2, _revision_notes(1, "SELA"),
            envelope, rules)
        assert contract.eligible is False
        draft1 = F2.build_final_draft(
            raw1, parsed1, _treatment(), envelope, rules)
        draft2 = F2.build_final_draft(
            raw2, parsed2, _treatment(), envelope, rules)
        forged = replace(contract, eligible=True, reasons=())

        with pytest.raises(F2.Fable2ScriptError, match="contract hash seal"):
            F2.choose_final_draft(draft1, draft2, forged)


# ---------------------------------------------------------------------------
# 4. Deal + seed (doc s9; OTR_FABLE2_SEED repro)
# ---------------------------------------------------------------------------

class TestDeal:
    def test_seed_env_reproduces_the_deal(self, monkeypatch):
        monkeypatch.setenv("OTR_FABLE2_SEED", "42")
        deck = F2._load_frame_deck()
        s1 = F2._resolve_seed()
        s2 = F2._resolve_seed()
        assert s1 == s2 == 42
        d1 = F2._deal(random.Random(s1), deck, mode=F2._COMPACT_MODE)
        d2 = F2._deal(random.Random(s2), deck, mode=F2._COMPACT_MODE)
        assert d1 == d2

    def test_bad_seed_env_fails_loud(self, monkeypatch):
        monkeypatch.setenv("OTR_FABLE2_SEED", "not-a-number")
        with pytest.raises(F2.Fable2Error, match="OTR_FABLE2_SEED"):
            F2._resolve_seed()

    def test_full_deal_is_three_distinct_cards(self):
        deck = F2._load_frame_deck()
        cards, stance = F2._deal(random.Random(7), deck, mode=F2._FULL_MODE)
        assert len(cards) == 3
        assert len({c["name"] for c in cards}) == 3
        assert stance["name"].strip()


# ---------------------------------------------------------------------------
# 5. Voice menu (doc s9: stable ids, capacity preflight)
# ---------------------------------------------------------------------------

class TestVoiceMenu:
    def test_stable_ids_and_real_stock(self):
        menu = F2._deal_voice_menu(2)
        ids = [e.menu_id for e in menu.entries]
        assert ids[0] == "m01"
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)
        presets = {e.preset for e in menu.entries}
        assert presets, "menu must offer only real stock"
        for e in menu.entries:
            assert e.gender in ("male", "female")

    def test_capacity_preflight_raises_before_any_llm(self):
        with pytest.raises(F2.Fable2CastError, match="capacity"):
            F2._deal_voice_menu(99)
