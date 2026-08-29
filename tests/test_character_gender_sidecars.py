"""D4 -- the source states its characters' gender, and the cast now hears it.

THE DEFECT (PBUG-20260815-04). `cast_source_contract.gender_by_name` shipped
`{}` on every public_domain episode, so the caster fell back to a blind 40/40/20
roll. It inverted real people three times on three different sources:
GERTRUDE (a woman) male on all six lines and LORD RONALD (a man) female on all
six -- the script makes it audible, the male-voiced character is addressed
*"Miss McFiggins"* -- and then AHAB, of Moby-Dick, in a woman's voice.

WHAT WAS ACTUALLY BROKEN, because it is not what it looked like. The render-time
plumbing was already complete: `source_meta_from_unit`
(`_otr_public_domain_sources.py:565-567`) loads the sidecar's `characters` into
`source_meta`, and `OTR_LedgerScriptWriter.py:4335` joins the cast names against
it. Every link existed. The DATA did not -- 64 of 65 units had no sidecar, so a
working pipeline carried an empty list end to end and no test noticed, because a
correct pipeline over absent data looks exactly like a correct pipeline.

So the fix is a vendored artifact, not a code path, and these tests are anchored
on the REAL sidecars the stamper wrote rather than on invented fixtures. An
invented fixture here would prove only that the join works on data shaped the
way I imagined it -- which was never in doubt.

THE LADDER DECLINES RATHER THAN GUESSES, and one measurement is why that floor
is not negotiable: DOROTHY, of Oz, scores male 8 / female 3 under a
first-pronoun-only estimator, because her scene is crowded with the Scarecrow
and the Lion. A looser threshold would not have produced more coverage; it would
have produced a confident lie in the same shape as the bug.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# The vendor tools are scripts, not a package -- the repo directory name has a
# hyphen so it cannot be imported as one. Same path trick the scripts use on
# themselves (`otr_fetch_public_domain.py:54`).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from nodes import _otr_roster_gender as RG  # noqa: E402
from nodes._otr_gender_pronoun_scan import (  # noqa: E402
    DOMINANCE_RATIO, SCORE_FLOOR, mention_forms, scan_gender,
)

_REPO = Path(__file__).resolve().parents[1]
_SOURCES = _REPO / "config" / "source_banks" / "public_domain_story" / "sources"
_MANIFEST = (
    _REPO / "config" / "source_banks" / "public_domain_story"
    / "manifest.sample.json"
)


# --------------------------------------------------------------------------- #
# the scanner
# --------------------------------------------------------------------------- #
class TestPronounScan:
    def test_it_reads_the_authors_own_pronouns(self):
        """Sized to clear SCORE_FLOOR deliberately, not by luck.

        This fixture used to carry ~4 pronouns and passed under the old floor of
        4. Raising the floor to 8 (see the module constant) broke it, which is
        the fixture being thin rather than the scan regressing -- a real passage
        that decides a character is not four words long. It now clears the floor
        with margin, and asserts the score so the relationship to the constant
        is visible instead of implied.
        """
        text = (
            "Ahab paced the deck. He said nothing for a long while, and his "
            "eyes never left the horizon. Ahab turned; he had made up his "
            "mind, and he would not be moved. Ahab raised his arm. He struck "
            "the rail with his fist, and his voice carried the length of the "
            "ship. Ahab was himself again."
        )
        verdict = scan_gender("Ahab", text)
        assert verdict.gender == "male"
        assert verdict.male_score >= SCORE_FLOOR
        assert verdict.male_score >= DOMINANCE_RATIO * verdict.female_score

    def test_a_NAME_THAT_NAMES_ITSELF_decides_before_any_counting(self):
        """LORD RONALD is why this is a short-circuit and not a heavy weight.

        Measured on the real text he scored male 78 / female 36 -- clearing the
        floor and FAILING the margin, so the scan declined a man whose name is
        the word "Lord". `gertrude_governess` is a romance: the two leads share
        nearly every mention window, so each one's pronouns pollute the other's
        count. No ratio fixes that without also deciding genuinely ambiguous
        cases. The author naming the role outranks a tally of nearby pronouns.
        """
        verdict = scan_gender("Lord Ronald", "Gertrude wept. She was alone.")
        assert verdict.gender == "male"
        assert verdict.mentions == 0  # decided without counting anything

    def test_relation_words_outrank_titles_inside_a_name(self):
        """Matches `_otr_character_roster.infer_gender` -- "daughter to Duke
        Senior" is a woman despite the Duke, and the two modules must never
        disagree about a name they can both see."""
        assert scan_gender("the Duke's daughter", "").gender == "female"

    def test_a_SHARED_surname_is_counted_for_NEITHER_character(self):
        """`monkeys_paw` has two Whites. A window opened by "White" belongs to
        neither, and crediting it to both would build a confident verdict out
        of the other character's pronouns."""
        text = "Mr. White laughed. He rose. Mrs. White frowned. She sat."
        # "white" IS one of John's mention forms ...
        assert "white" in mention_forms("John White")
        verdict = scan_gender(
            "John White", text, other_names=["John White", "Mary White"],
        )
        # ... and it was dropped because Mary answers to it too, leaving only
        # the full name, which this text never uses. Declining is correct: the
        # alternative is crediting Mary's "She sat" to John.
        assert "ignored shared form" in verdict.evidence
        assert verdict.gender == ""

    def test_it_DECLINES_below_the_floor_rather_than_guessing(self):
        verdict = scan_gender("Someone", "Someone arrived. He left.")
        assert verdict.gender == ""
        assert not verdict.decided
        assert "DECLINED" in verdict.evidence

    def test_it_DECLINES_a_mixed_scene_rather_than_calling_it(self):
        """Two people talking about each other is not evidence about either."""
        text = (
            "Alice looked at him. He smiled at her, and she frowned. Alice "
            "spoke; he answered her, and she turned away from him."
        )
        assert scan_gender("Alice", text).gender == ""

    def test_the_floor_and_the_margin_are_both_required(self):
        assert SCORE_FLOOR >= 4
        assert DOMINANCE_RATIO >= 3.0

    def test_mention_forms_cover_how_prose_actually_refers_to_people(self):
        assert mention_forms("Captain Ahab") == ("captain ahab", "ahab")
        assert mention_forms("the water ghost") == ("the water ghost",
                                                    "water ghost", "ghost")
        assert mention_forms("Gertrude") == ("gertrude",)

    def test_the_GIVEN_name_is_searched_too_not_just_the_surname(self):
        """The scan used to hunt a word some stories never use.

        It took the surname only, on the reasoning that prose introduces
        "Captain Ahab" once and then says "Ahab". True of Melville, false of
        Austen, Montgomery and Twain. Measured on the shipped corpus:
        "Shirley" appears 0 times and "Anne" 75; "Bennet" 0 and "Elizabeth" 9.
        Anne Shirley therefore declined for want of a search term, not for want
        of evidence -- and the characters it failed were the LEADS.
        """
        assert "jane" in mention_forms("Jane Eyre")
        assert "anne" in mention_forms("Anne Shirley")
        assert "elizabeth" in mention_forms("Elizabeth Bennet")

    def test_a_RANK_or_abbreviation_is_never_searched_as_a_name(self):
        """"Captain" matches every officer aboard, not this character."""
        assert "captain" not in mention_forms("Captain Ahab")
        assert "mrs." not in mention_forms("Mrs. White")
        assert "dr." not in mention_forms("Dr. Grimesby Roylott")

    def test_an_ARTICLE_marks_a_description_so_its_head_is_not_a_name(self):
        """THE REGRESSION GUARD. Widening the net to first names re-broke
        `buck_rogers`: "a Han patrol" scanned for "Han", the antagonist RACE,
        which opened a window on every enemy in the story and swept up the
        female lead's pronouns -- taking a defect that scored 4 to 9, clear over
        the floor. An article marks a description, not a proper name, and that
        one test separates "Jane Eyre" from "a Han patrol"."""
        assert "han" not in mention_forms("a Han patrol")
        assert "water" not in mention_forms("the water ghost")
        # ... while a genuine proper name is unaffected.
        assert "jane" in mention_forms("Jane Eyre")


class TestTheLadderItself:
    """Tier 1 had NEVER executed, and nothing in the tree would have said so.

    `_decide` called `infer_gender(one_arg)` where the real signature is
    `infer_gender(name, description) -> (gender, source)`. It raised TypeError
    the instant a cast block parsed, and nothing caught it -- one bad unit would
    have killed the whole 65-unit run. It stayed dormant purely because prose
    has no cast block, so every shipped sidecar reads `"roster": 0` and the
    corpus test above passed over a tier that could not run.
    """

    _CAST_TEXT = (
        "CHARACTERS\n"
        "\n"
        "LORD ALFRED, a nobleman of the county\n"
        "MARGARET, daughter to Lord Alfred\n"
        "\n"
        "The scene opens on a quiet hall.\n"
    )

    def _decide(self, name):
        import importlib

        stamper = importlib.import_module("otr_stamp_character_genders")
        return stamper._decide(name, self._CAST_TEXT, [name])

    def test_tier_1_RUNS_at_all(self):
        """The regression: this raised TypeError before the fix."""
        gender, tier, evidence = self._decide("LORD ALFRED")
        assert gender in ("male", "female", "")
        assert isinstance(evidence, str)

    def test_tier_1_reads_the_cast_block_and_wins_over_the_pronoun_scan(self):
        gender, tier, evidence = self._decide("LORD ALFRED")
        assert (gender, tier) == ("male", "roster")
        assert "source cast list" in evidence

    def test_a_RELATION_word_in_the_cast_block_outranks_a_title(self):
        """"daughter to Lord Alfred" is a woman despite the Lord -- the
        precedence `infer_gender` implements, which a single-argument call could
        never have exercised."""
        gender, tier, _evidence = self._decide("MARGARET")
        assert (gender, tier) == ("female", "roster")


class TestVendoredCorpus:
    def test_every_manifest_unit_has_a_sidecar(self):
        """64 of 65 were bare. That is the whole defect, so it is the first
        assertion -- a count, on the real tree."""
        manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
        missing = []
        for source in manifest["sources"]:
            for unit in source.get("units") or []:
                text_path = _MANIFEST.parent / unit["text_path"]
                if not RG.sidecar_path_for_text(text_path).is_file():
                    missing.append(source["source_id"])
        assert missing == []

    def test_no_row_is_ever_UNKNOWN(self):
        """The ladder is total over what it writes. A name it cannot decide is
        OMITTED, which is a render-time join miss and therefore today's roll --
        a no-change. No consumer has to branch on `unknown` because it never
        enters the data."""
        for path in _SOURCES.glob("*.provenance.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            for row in data.get("characters") or []:
                assert row["gender"] in ("male", "female"), path.name
                assert row["gender_source"] in (
                    "roster", "pronouns", "llm_web", "name_frequency")
                assert str(row.get("evidence") or "").strip(), path.name

    def test_the_body_hash_matches_the_text_on_disk(self):
        """Freshness is verifiable, not asserted: a sidecar whose hash does not
        match its text is describing a source that changed underneath it."""
        import hashlib

        for path in _SOURCES.glob("*.provenance.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            text_file = path.parent / (path.name.replace(
                ".provenance.json", ".txt"))
            if not text_file.is_file() or not data.get("body_sha256"):
                continue
            digest = hashlib.sha256(
                text_file.read_text(encoding="utf-8", errors="replace")
                .encode("utf-8")).hexdigest()
            assert digest == data["body_sha256"], path.name


# --------------------------------------------------------------------------- #
# the join -- the assertion the operator actually cares about
# --------------------------------------------------------------------------- #
class TestTheThreeThatShippedWrong:
    """D4's acceptance: GERTRUDE and LORD RONALD no longer render inverted.

    Driven through the SAME call `OTR_LedgerScriptWriter.py:4335` makes, with
    the cast names spelled the way the failing artifacts spelled them -- the
    episode's cast said AHAB, while the sidecar row is "Captain Ahab", so the
    join is part of what is under test rather than something arranged away.
    """

    @pytest.mark.parametrize("slug, cast_name, expected", [
        ("moby_dick_quarterdeck", "AHAB", "male"),
        ("moby_dick_quarterdeck", "STARBUCK", "male"),
        ("gertrude_governess", "GERTRUDE", "female"),
        ("gertrude_governess", "LORD RONALD", "male"),
        ("gertrude_governess", "RONALD", "male"),
    ])
    def test_the_source_pins_the_voice(self, slug, cast_name, expected):
        characters = RG.load_roster_characters(_SOURCES / f"{slug}.txt")
        pinned = RG.gender_map_for_names([cast_name], characters)
        assert pinned.get(cast_name, {}).get("gender") == expected

    def test_gender_by_name_is_no_longer_EMPTY(self):
        """The measured symptom on the artifact was `gender_by_name == {}`."""
        characters = RG.load_roster_characters(
            _SOURCES / "moby_dick_quarterdeck.txt")
        pinned = RG.gender_map_for_names(
            ["AHAB", "STARBUCK", "STUBB"], characters)
        assert pinned != {}
        assert len(pinned) == 3

    @staticmethod
    def _fetcher(monkeypatch, tmp_path):
        """The vendor tool, rooted at tmp_path.

        `write_source` records `text_path` relative to the repo root, so a
        destination outside it raises -- rooting the module at tmp_path is what
        lets this exercise the REAL function instead of a copy of its logic.
        """
        import importlib

        fetch = importlib.import_module("otr_fetch_public_domain")
        monkeypatch.setattr(fetch, "REPO_ROOT", tmp_path)
        return fetch

    _ROSTER = [{"name": "AHAB", "gender": "male",
                "gender_source": "pronouns", "evidence": "e"}]

    def test_a_REFETCH_does_not_silently_erase_the_gender_roster(
            self, monkeypatch, tmp_path):
        """The landmine: `otr_fetch_public_domain.write_source` rebuilds the
        sidecar dict from scratch and overwrites the file, so before the
        carry-forward a routine re-fetch deleted `characters` and dropped that
        unit back to the blind roll -- PBUG-20260815-04 reintroduced by the tool
        least likely to be suspected of it."""
        fetch = self._fetcher(monkeypatch, tmp_path)
        body = "Ahab paced. He said nothing.\n"
        args = dict(
            slug="probe", unit=None, work_title="W", author="A",
            license_label="L", source_url="u", word_count=5,
            dest_dir=tmp_path,
        )
        _text, sidecar = fetch.write_source(body=body, **args)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["characters"] = list(self._ROSTER)
        data["gender_ladder"] = {"version": "gender_ladder_v1"}
        sidecar.write_text(json.dumps(data), encoding="utf-8")

        fetch.write_source(body=body, **args)  # identical bytes -> survives

        after = json.loads(sidecar.read_text(encoding="utf-8"))
        assert after["characters"][0]["name"] == "AHAB"
        assert after["gender_ladder"]["version"] == "gender_ladder_v1"

    def test_a_CHANGED_body_drops_the_roster_instead_of_lying(
            self, monkeypatch, tmp_path):
        """Carrying a roster onto text it was not read from would be WORSE than
        deleting it -- it would still look authoritative while describing a
        source that no longer exists."""
        fetch = self._fetcher(monkeypatch, tmp_path)
        args = dict(
            slug="probe2", unit=None, work_title="W", author="A",
            license_label="L", source_url="u", word_count=5,
            dest_dir=tmp_path,
        )
        _text, sidecar = fetch.write_source(body="Ahab paced.\n", **args)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["characters"] = list(self._ROSTER)
        sidecar.write_text(json.dumps(data), encoding="utf-8")

        fetch.write_source(body="A completely different source.\n", **args)

        after = json.loads(sidecar.read_text(encoding="utf-8"))
        assert "characters" not in after

    def test_the_owned_key_list_has_exactly_ONE_definition(self, monkeypatch,
                                                           tmp_path):
        """Two hand-spelled key lists is the producer/consumer mismatch of
        Bible 12.86, which is how the roster would go missing again.

        Value equality, not object identity: the scripts import the roster
        module FLAT (`_otr_roster_gender`) while the render path imports it as
        `nodes._otr_roster_gender`, so there are legitimately two module objects
        and an `is` check would fail on correct code. What must not differ is
        the value, plus the fact that the script IMPORTS the name rather than
        spelling the keys itself.
        """
        fetch = self._fetcher(monkeypatch, tmp_path)
        assert (tuple(fetch.STAMPER_OWNED_SIDECAR_KEYS)
                == tuple(RG.STAMPER_OWNED_SIDECAR_KEYS))
        source = (_REPO / "scripts" / "otr_fetch_public_domain.py").read_text(
            encoding="utf-8")
        assert "import STAMPER_OWNED_SIDECAR_KEYS" in source
        # Deliberately NOT asserting the fetcher never spells "characters": it
        # legitimately AUTHORS that key for the shakespeare lane (`:403`), where
        # the play's own cast block is the source of truth. Two owners, split by
        # lane -- which is exactly why the carry-forward uses `setdefault`, so a
        # fetcher-authored roster always wins over a preserved one.
        assert '"characters": characters' in source

    def test_a_fetcher_AUTHORED_roster_still_wins_over_a_preserved_one(
            self, monkeypatch, tmp_path):
        """Shakespeare's roster comes from the fetcher, not the stamper. The
        carry-forward must never shadow a lane that owns the key itself."""
        fetch = self._fetcher(monkeypatch, tmp_path)
        body = "Ahab paced. He said nothing.\n"
        args = dict(
            slug="probe3", unit=None, work_title="W", author="A",
            license_label="L", source_url="u", word_count=5,
            dest_dir=tmp_path,
        )
        _text, sidecar = fetch.write_source(body=body, **args)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        data["characters"] = list(self._ROSTER)
        sidecar.write_text(json.dumps(data), encoding="utf-8")

        fetch.write_source(
            body=body, extra={"characters": [{"name": "MACBETH"}]}, **args)

        after = json.loads(sidecar.read_text(encoding="utf-8"))
        assert after["characters"] == [{"name": "MACBETH"}]

    def test_an_undecided_name_leaves_the_roll_ALONE(self):
        """The second 'undetermined' state, and it must stay distinct: a name
        the ladder never decided is absent from the sidecar, so the join misses
        and the existing roll survives untouched. Absent must never be read as
        a pin."""
        characters = RG.load_roster_characters(
            _SOURCES / "gertrude_governess.txt")
        assert RG.gender_map_for_names(["THE NARRATOR"], characters) == {}
