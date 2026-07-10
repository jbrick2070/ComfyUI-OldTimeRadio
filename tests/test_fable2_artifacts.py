"""scifi_fable2 S1b -- artifact models, validator factories, envelope,
deal, voice menu (architecture doc s5/s8/s9/s13).

Pure Python; no GPU, no LLM. Named thresholds are fixtured BOTH
directions (r2 anchor): a value just under the threshold passes, a value
at/over it fails.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_scifi_fable2 as F2  # noqa: E402


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
            "Will Vera trust her own instruments before the village "
            "stops trusting her?"),
        "setting": "a volcano observatory above a village at night",
        "cast_shapes": [
            {"name": "VERA", "role": "instrument scientist",
             "want": "to be believed before the mountain proves her right",
             "pressure": "the committee reads her charts as noise",
             "register": "clipped, front-loaded, swallows apologies"},
            {"name": "DOKU", "role": "village liaison",
             "want": "to keep the village calm one more season",
             "pressure": "his cousin farms the north slope",
             "register": "slow warm circling, answers with stories"},
        ],
        "turn": (
            "The heat readings Vera hid to protect her credibility are "
            "the only proof that would move the village in time."),
        "priced_ending": {
            "choice": "Vera publishes the readings under her own name",
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
        shape = F2.CastShape(name="Vera", role="scientist",
                             want="to be believed today",
                             pressure="the committee doubts",
                             register="clipped and dry")
        assert shape.name == "VERA"

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
            _treatment(dramatic_question="Vera trusts her instruments now.")

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
        shapes[1]["name"] = "VERA"
        with pytest.raises(ValidationError, match="unique"):
            _treatment(cast_shapes=shapes)

    def test_critic_note_never_replacement_dialogue(self):
        with pytest.raises(ValidationError, match="replacement dialogue"):
            F2.CriticNote(scene=1, speaker="VERA", problem="register_bleed",
                          note="Change it to this. VERA: I never doubted "
                               "the mountain for a second.")

    def test_critic_revise_requires_a_note(self):
        with pytest.raises(ValidationError, match="revise"):
            F2.CriticNotes(verdict="revise", notes=[])
        F2.CriticNotes(verdict="ship", notes=[])  # legitimate verdict

    def test_audit_finding_class_enum_closed(self):
        with pytest.raises(ValidationError):
            F2.AuditFinding(finding_class="vibes_off", scene=1, speaker="",
                            detail="not a registry class at all")

    def test_critic_classes_are_subset_of_audit_classes(self):
        # r4 anchor snapshot: every critic class is auditable.
        critic = set(
            F2.CriticNote.model_fields["problem"].annotation.__args__)
        audit = set(
            F2.AuditFinding.model_fields["finding_class"].annotation.__args__)
        assert critic < audit


# ---------------------------------------------------------------------------
# 2. post_validator factories (r2/S1)
# ---------------------------------------------------------------------------

class TestDossierValidator:
    def test_grounded_dossier_passes(self):
        assert F2._make_dossier_validator(_DIGEST)(_dossier()) is None

    def test_invented_number_rejected(self):
        check = F2._make_dossier_validator(_DIGEST)
        err = check(_dossier(allowed_numbers=["7,777"]))
        assert err and "7,777" in err

    def test_invented_entity_rejected(self):
        check = F2._make_dossier_validator(_DIGEST)
        bad = _dossier(named_entities={
            "people": ["Professor Nobody"], "places": [], "things": []})
        err = check(bad)
        assert err and "Professor Nobody" in err

    def test_spelled_numbers_legalize_extracted_digits(self):
        # 16th live smoke (2026-07-10): "seven days" / "Eighth day" in
        # the story legalize the extracted digits 7 and 8.
        check = F2._make_dossier_validator(
            "The heat lasted seven days, the eighth day this year on "
            "Mount Etna, Doctor Rossi said. 1,200. heat sensors.")
        ok = _dossier(allowed_numbers=["7", "8", "1,200"])
        assert check(ok) is None
        bad = _dossier(allowed_numbers=["9"])
        err = check(bad)
        assert err and "'9'" in err

    def test_demonym_inflection_is_legal(self):
        # 24th live smoke (2026-07-10): 'Scottish' in the story vs the
        # faithful extraction 'Scotland' -- same referent, not invention.
        check = F2._make_dossier_validator(
            "The Scottish government provides a support package for pig "
            "farmers. Doctor Rossi. Mount Etna. 1,200. heat sensors.")
        ok = _dossier(named_entities={
            "people": ["Doctor Rossi"], "places": ["Scotland"],
            "things": ["heat sensors"]})
        assert check(ok) is None

    def test_reordered_entity_phrase_is_legal(self):
        # 2nd live smoke (2026-07-10): "Amsterdam's canals" vs the
        # story's "the canals of Amsterdam" -- token-level presence, not
        # contiguous-phrase substring.
        check = F2._make_dossier_validator(
            "Researchers steered the robots through the canals of "
            "Amsterdam before any tremor. Doctor Rossi led the team. "
            "Mount Etna. 1,200. heat sensors.")
        ok = _dossier(named_entities={
            "people": ["Doctor Rossi"],
            "places": ["Amsterdam's canals", "Mount Etna"],
            "things": ["heat sensors"]})
        assert check(ok) is None
        bad = _dossier(named_entities={
            "people": ["Doctor Rossi"],
            "places": ["Rotterdam harbor"], "things": []})
        err = check(bad)
        assert err and "Rotterdam" in (err or "")


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
        check = F2._make_pitch_validator(_CARDS3[:1], "one_pitch", 4)
        slate = F2.PitchSlate(pitches=[_pitch_dict(1, "Card A")])
        assert check(slate) is None

    def test_one_pitch_mode_rejects_three(self):
        check = F2._make_pitch_validator(_CARDS3[:1], "one_pitch", 4)
        slate = F2.PitchSlate(pitches=[
            _pitch_dict(1, "Card A"), _pitch_dict(2, "Card A"),
            _pitch_dict(3, "Card A")])
        assert "exactly 1" in (check(slate) or "")

    def test_pitch_i_must_use_card_i(self):
        check = F2._make_pitch_validator(_CARDS3, "full", 4)
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
        err = check(slate)
        assert err and "card i" in err

    def test_cast_size_over_n_max_rejected(self):
        check = F2._make_pitch_validator(_CARDS3[:1], "one_pitch", 2)
        slate = F2.PitchSlate(pitches=[_pitch_dict(1, "Card A", cast_size=5)])
        assert "N_MAX" in (check(slate) or "")

    def test_full_mode_divergence_gate(self):
        check = F2._make_pitch_validator(_CARDS3, "full", 4)
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
        check = F2._make_pitch_validator(_CARDS3, "full", 4)
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
        check = F2._make_pitch_validator(_CARDS3[:1], "one_pitch", 4)
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


def _read(text: str) -> F2.NewsCloseRead:
    return F2.NewsCloseRead(news_close_read=text)


class TestReadValidator:
    """P2c factual-close subset laws (S1b read-split; every case below
    is a live-smoke scar from 2026-07-10)."""

    def _check(self, dossier=None, digest=_DIGEST, cast=("VERA", "DOKU")):
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
            "Tonight's story grew from a real survey: Vera watched "
            "instruments on Mount Etna map 1,200 new vents this season, "
            "work led by Doctor Rossi."))
        assert err and "VERA" in err and "REAL NEWS" in err

    def test_cast_name_gate_is_case_sensitive(self):
        # 11th live smoke: a character named HOPE must not outlaw the
        # common word 'hope'.
        assert self._check(cast=("HOPE", "DOKU"))(_read(
            "Tonight's story grew from a real survey: instruments on "
            "Mount Etna mapped 1,200 new vents this season, and the "
            "team's hope is a calmer year, said Doctor Rossi.")) is None
        err = self._check(cast=("HOPE", "DOKU"))(_read(
            "Tonight's story grew from a real survey: Hope watched "
            "instruments on Mount Etna map 1,200 new vents this season, "
            "work led by Doctor Rossi."))
        assert err and "HOPE" in err

    def test_cast_name_borrowed_from_the_real_source_stays_legal(self):
        # 14th live smoke (2026-07-10): the drama named its character
        # after the REAL person in the story ('JIM' for NASA's Jim
        # Ross) -- factual reporting of a source name is never leakage.
        assert self._check(cast=("ROSSI", "DOKU"))(_read(
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
        check = F2._make_casting_validator(self._menu(), ["VERA", "DOKU"])
        cv = F2.CastingVoices(cast=[
            self._voice("VERA", "female", "m01"),
            self._voice("DOKU", "male", "m02")])
        assert check(cv) is None

    def test_speaker_set_equality_gate(self):
        check = F2._make_casting_validator(self._menu(), ["VERA", "DOKU"])
        cv = F2.CastingVoices(cast=[self._voice("VERA", "female", "m01")])
        assert "speakers" in (check(cv) or "")

    def test_unknown_menu_id_rejected(self):
        check = F2._make_casting_validator(self._menu(), ["VERA"])
        cv = F2.CastingVoices(cast=[self._voice("VERA", "female", "m99")])
        assert "m99" in (check(cv) or "")

    def test_gender_mismatch_rejected(self):
        check = F2._make_casting_validator(self._menu(), ["VERA"])
        cv = F2.CastingVoices(cast=[self._voice("VERA", "female", "m02")])
        assert "gender" in (check(cv) or "")

    def test_duplicate_timbre_rejected(self):
        check = F2._make_casting_validator(self._menu(), ["VERA", "DOKU"])
        cv = F2.CastingVoices(cast=[
            self._voice("VERA", "male", "m02"),
            self._voice("DOKU", "male", "m02")])
        assert "already taken" in (check(cv) or "")

    def test_casting_accepts_unwrapped_shapes(self):
        # 18th live smoke (2026-07-10): a truncated response salvaged as
        # a bare list / single entry must reach the TEACHING gates, not
        # die as a raw schema error.
        bare_list = F2.CastingVoices.model_validate(
            [self._voice("VERA", "female", "m01")])
        assert [c.name for c in bare_list.cast] == ["VERA"]
        single = F2.CastingVoices.model_validate(
            self._voice("DOKU", "male", "m02"))
        assert [c.name for c in single.cast] == ["DOKU"]


# ---------------------------------------------------------------------------
# 3. Envelope + token budget + entry gates (doc s3/s13)
# ---------------------------------------------------------------------------

class TestEnvelopeAndBudgets:
    def test_scene_formula(self):
        assert F2._build_envelope(30).scene_count == 1
        assert F2._build_envelope(350).scene_count == 3
        assert F2._build_envelope(880).scene_count == 8
        assert F2._build_envelope(10).scene_count == 1  # clamp low

    def test_token_budget_floor_and_cap(self):
        assert F2._script_token_budget(30) == 1200      # floor
        assert F2._script_token_budget(500) == 1300
        assert F2._script_token_budget(5000) == 4200    # cap

    def test_entry_gate_ceiling(self):
        with pytest.raises(F2.Fable2ScriptError, match="ceiling"):
            F2.assert_supported_target_words(901)

    def test_entry_gate_s1b_one_draft_scope(self):
        with pytest.raises(F2.Fable2ScriptError, match="S2"):
            F2.assert_supported_target_words(F2._ONE_DRAFT_THRESHOLD)
        F2.assert_supported_target_words(F2._ONE_DRAFT_THRESHOLD - 1)


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
        d1 = F2._deal(random.Random(s1), deck, mode="one_pitch")
        d2 = F2._deal(random.Random(s2), deck, mode="one_pitch")
        assert d1 == d2

    def test_bad_seed_env_fails_loud(self, monkeypatch):
        monkeypatch.setenv("OTR_FABLE2_SEED", "not-a-number")
        with pytest.raises(F2.Fable2Error, match="OTR_FABLE2_SEED"):
            F2._resolve_seed()

    def test_full_deal_is_three_distinct_cards(self):
        deck = F2._load_frame_deck()
        cards, stance = F2._deal(random.Random(7), deck, mode="full")
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
