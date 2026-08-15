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
        text = (
            "Ahab paced the deck. He said nothing for a long while, and his "
            "eyes never left the horizon. Ahab turned. He spoke at last."
        )
        verdict = scan_gender("Ahab", text)
        assert verdict.gender == "male"
        assert verdict.male_score > verdict.female_score

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


# --------------------------------------------------------------------------- #
# the vendored corpus -- the artifact, not a fixture
# --------------------------------------------------------------------------- #
def _sidecar(slug: str) -> dict:
    path = _SOURCES / f"{slug}.provenance.json"
    assert path.is_file(), f"missing vendored sidecar: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


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

    def test_an_undecided_name_leaves_the_roll_ALONE(self):
        """The second 'undetermined' state, and it must stay distinct: a name
        the ladder never decided is absent from the sidecar, so the join misses
        and the existing roll survives untouched. Absent must never be read as
        a pin."""
        characters = RG.load_roster_characters(
            _SOURCES / "gertrude_governess.txt")
        assert RG.gender_map_for_names(["THE NARRATOR"], characters) == {}
