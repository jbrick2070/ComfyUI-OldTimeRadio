"""Tiers 3-4 of the character gender ladder, the index, and the monotonic merge.

Spec: docs/2026-08-28-character-gender-ladder-SPEC-v2.md (operator rulings 2026-08-28:
Shakespeare fills ONLY unknown rows; the web tier is replaced by ONE recall question
per (work, name), cached in a committed index). Review round 2026-09-02 (Antigravity,
kibitz-runs/2026-09-02-gender-ladder-v2-review/r2) pinned: ARIEL / PUCK stay on the
roll, the merge is anchored on body_sha256, equal rungs refresh and lower never demotes,
tier_counts derive from the final rows, the confidence vocabulary reads through for
rows that predate the field, and no source text ever reaches the model.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))

from nodes import _otr_roster_gender as RG  # noqa: E402

stamper = importlib.import_module("otr_stamp_character_genders")


class _FakeModel:
    """A generate_fn that answers from a script and counts calls, and refuses any
    message that carries more than a title, an author and a name."""

    def __init__(self, answers):
        self.answers = dict(answers)
        self.calls = []

    def __call__(self, messages, *, temperature, max_new_tokens):
        assert temperature == 0.0
        user = messages[-1]["content"]
        assert len(user) < 400, "tier 3 must never carry source text"
        self.calls.append(user)
        for name, answer in self.answers.items():
            if '"%s"' % name in user:
                if answer == "garbage":
                    return "not json at all"
                return json.dumps({"gender": answer, "reason": "%s in the work" % name})
        return json.dumps({"gender": "unsure", "reason": "never heard of them"})


def _recall(tmp_path, answers, model_id="stub-model"):
    index = stamper.GenderIndex(tmp_path / "character_gender_index.json")
    fake = _FakeModel(answers)
    return stamper.Recall(fake, index, model_id=model_id, clock=lambda: "2026-09-02T00:00:00+00:00"), fake, index


# --------------------------------------------------------------------------
# tier 3: the recall question and its index
# --------------------------------------------------------------------------

def test_recall_asks_once_then_the_index_answers(tmp_path):
    recall, fake, index = _recall(tmp_path, {"Elizabeth Bennet": "female"})
    g1, e1, s1 = recall.ask("Elizabeth Bennet", "Pride and Prejudice", "Jane Austen")
    g2, e2, s2 = recall.ask("Elizabeth Bennet", "Pride and Prejudice", "Jane Austen")
    assert (g1, g2, s1, s2) == ("female", "female", "llm_recall", "llm_recall")
    assert len(fake.calls) == 1, "the second ask must be an index hit"
    assert 'recall of "Pride and Prejudice" by Jane Austen' in e1
    assert recall.census["asked"] == 1 and recall.census["cached"] == 1
    entry = index.get("Pride and Prejudice", "Elizabeth Bennet")
    assert entry["gender"] == "female" and entry["asked_as"] == "title"
    assert entry["model"] == "stub-model" and entry["locked"] is False


def test_the_question_carries_the_work_and_never_the_text():
    messages = stamper.recall_messages("Ahab", "Moby-Dick", "Herman Melville")
    user = messages[-1]["content"]
    assert 'In "Moby-Dick" by Herman Melville, is the character "Ahab"' in user
    assert "unsure" in user


def test_unsure_and_unparseable_decline_and_unsure_is_cached(tmp_path):
    recall, fake, index = _recall(tmp_path, {"Nobody": "unsure", "Broken": "garbage"})
    assert recall.ask("Nobody", "Some Work")[0] == ""
    assert recall.ask("Nobody", "Some Work")[0] == ""
    assert len([c for c in fake.calls if '"Nobody"' in c]) == 1, "an unsure answer is remembered"
    assert recall.ask("Broken", "Some Work")[0] == ""
    assert index.get("Some Work", "Broken") is None, "an unparseable answer is not cached"
    assert recall.census["unsure"] == 1 and recall.census["unparseable"] == 1


def test_a_locked_empty_entry_is_the_operators_roll_and_is_never_asked(tmp_path):
    path = tmp_path / "character_gender_index.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": {
        "THE TEMPEST|ARIEL": {"gender": "", "locked": True, "reason": "operator: roll"}}}),
        encoding="utf-8")
    index = stamper.GenderIndex(path)
    fake = _FakeModel({"Ariel": "male"})
    recall = stamper.Recall(fake, index, model_id="stub-model")
    gender, evidence, source = recall.ask("Ariel", "The Tempest", "William Shakespeare")
    assert gender == "" and "operator-locked" in evidence and source == ""
    assert fake.calls == []


def test_a_locked_gender_wins_without_a_call(tmp_path):
    path = tmp_path / "character_gender_index.json"
    path.write_text(json.dumps({"schema_version": 1, "entries": {
        "SOME WORK|GLENN": {"gender": "male", "locked": True, "reason": "operator heard him"}}}),
        encoding="utf-8")
    recall = stamper.Recall(_FakeModel({"Glenn": "female"}), stamper.GenderIndex(path))
    assert recall.ask("Glenn", "Some Work") == (
        "male", "operator-locked index entry: operator heard him", "supplement")


def test_tier_3_is_off_without_a_model_and_says_so(tmp_path):
    recall = stamper.Recall(None, stamper.GenderIndex(tmp_path / "i.json"))
    gender, evidence, source = recall.ask("Anyone", "Any Work")
    assert gender == "" and "tier 3 off" in evidence and source == ""


def test_the_index_writes_sorted_and_only_when_it_moved(tmp_path):
    recall, _fake, index = _recall(tmp_path, {"Zed": "male", "Abe": "male"})
    recall.ask("Zed", "W"); recall.ask("Abe", "W")
    assert index.save() is True
    keys = list(json.loads(index.path.read_text(encoding="utf-8"))["entries"])
    assert keys == sorted(keys)
    assert index.save() is False, "a no-op save must not touch the file"
    assert b"\r\n" not in index.path.read_bytes()


# --------------------------------------------------------------------------
# tier 4: the curated first-name pool, and honorifics
# --------------------------------------------------------------------------

def test_name_frequency_answers_a_listed_name_and_declines_the_rest():
    assert stamper.name_frequency("Alice")[0] == "female"
    assert stamper.name_frequency("Victor Frankenstein")[0] == "male"
    assert stamper.name_frequency("Sancho Panza")[0] == ""
    assert stamper.name_frequency("the Creature")[0] == ""
    assert "description" in stamper.name_frequency("the Hatter")[1]


def test_strip_honorifics_is_public_and_knows_miss():
    assert RG.strip_honorifics("Miss Mix") == "Mix"
    assert RG.strip_honorifics("SIR TOBY") == "TOBY"
    assert RG.strip_honorifics("Uncle Silas") == "Silas"
    assert RG.strip_honorifics("Miss") == "Miss", "a name that is only a title is returned as-is"


def test_decide_all_walks_the_rungs_in_order(tmp_path):
    recall, fake, _index = _recall(tmp_path, {"Fitzwilliam Darcy": "male"})
    text = "Nothing about anybody in here.\n"
    # tier 3 answers -> llm_recall
    assert stamper.decide_all("Fitzwilliam Darcy", text, [], work_title="Pride and Prejudice",
                              author="Jane Austen", recall=recall)[:2] == ("male", "llm_recall")
    # tier 3 unsure -> tier 4 from the pool
    assert stamper.decide_all("Alice", text, [], work_title="Alice in Wonderland",
                              recall=recall)[:2] == ("female", "name_frequency")
    # everything declines -> the evidence names every rung
    gender, tier, evidence = stamper.decide_all("Sancho Panza", text, [],
                                                work_title="Don Quixote", recall=recall)
    assert (gender, tier) == ("", "")
    assert "pronoun scan" in evidence and "unsure" in evidence and "name_frequency" in evidence


# --------------------------------------------------------------------------
# the prose stamp: rows, the merge, the sidecar bytes
# --------------------------------------------------------------------------

def _prose_bank(tmp_path, text):
    bank = tmp_path / "bank"
    (bank / "sources").mkdir(parents=True)
    (bank / "sources" / "unit.txt").write_text(text, encoding="utf-8")
    source = {"source_id": "unit", "title": "The Unit", "author": "A. Writer",
              "license_status": "cc0", "source_url": "repo:x", "cast_hints": ["Alice", "Bertram"]}
    unit = {"unit_id": "main", "text_path": "sources/unit.txt"}
    return bank, source, unit


ALICE_TEXT = ("Alice looked up. She smiled; she waved; she laughed; she ran, and her hat "
              "was hers, and she sang as she went, and she was glad, and she stayed. "
              "Bertram watched from the door.\n")


def test_stamp_unit_fills_the_row_the_pronouns_declined_from_recall(tmp_path):
    bank, source, unit = _prose_bank(tmp_path, ALICE_TEXT)
    recall, fake, _ = _recall(tmp_path, {"Bertram": "male"})
    res = stamper.stamp_unit(source, unit, write=True, bank_dir=bank, recall=recall)
    rows = {r["name"]: r for r in json.loads((bank / "sources" / "unit.provenance.json")
                                             .read_text(encoding="utf-8"))["characters"]}
    assert rows["Alice"]["gender_source"] == "pronouns" and rows["Alice"]["gender_confidence"] == "known"
    assert rows["Bertram"]["gender"] == "male" and rows["Bertram"]["gender_source"] == "llm_recall"
    assert rows["Bertram"]["gender_confidence"] == "recalled"
    assert res["changed"] is True
    side = json.loads((bank / "sources" / "unit.provenance.json").read_text(encoding="utf-8"))
    assert side["gender_ladder"]["version"] == stamper.LADDER_VERSION
    assert side["gender_ladder"]["tier_counts"] == {"roster": 0, "pronouns": 1, "llm_recall": 1, "name_frequency": 0}
    assert side["body_sha256"] == hashlib.sha256(ALICE_TEXT.encode("utf-8")).hexdigest()
    assert b"\r\n" not in (bank / "sources" / "unit.provenance.json").read_bytes()


def test_a_second_run_is_a_byte_for_byte_no_op(tmp_path):
    bank, source, unit = _prose_bank(tmp_path, ALICE_TEXT)
    recall, fake, _ = _recall(tmp_path, {"Bertram": "male"})
    stamper.stamp_unit(source, unit, write=True, bank_dir=bank, recall=recall)
    before = (bank / "sources" / "unit.provenance.json").read_bytes()
    res = stamper.stamp_unit(source, unit, write=True, bank_dir=bank, recall=recall)
    assert res["changed"] is False
    assert (bank / "sources" / "unit.provenance.json").read_bytes() == before
    assert len(fake.calls) == 1, "the re-run answered Bertram from the index"


def test_a_lower_rung_never_demotes_a_pronoun_row(tmp_path):
    bank, source, unit = _prose_bank(tmp_path, ALICE_TEXT)
    recall, fake, _ = _recall(tmp_path, {"Bertram": "male", "Alice": "male"})
    stamper.stamp_unit(source, unit, write=True, bank_dir=bank, recall=recall)
    # Simulate the pronoun scan going quiet on unchanged text: strip the pronouns
    # from what the ladder sees by monkeypatching the scan to decline.
    import otr_stamp_character_genders as mod
    real = mod.scan_gender
    mod.scan_gender = lambda *a, **k: type("V", (), {"decided": False, "gender": "", "evidence": "declined by test"})()
    try:
        res = stamper.stamp_unit(source, unit, write=True, bank_dir=bank, recall=recall)
    finally:
        mod.scan_gender = real
    rows = {r["name"]: r for r in json.loads((bank / "sources" / "unit.provenance.json")
                                             .read_text(encoding="utf-8"))["characters"]}
    assert rows["Alice"]["gender_source"] == "pronouns", "recall (rung 3) may not replace pronouns (rung 2)"
    assert rows["Alice"]["gender"] == "female"
    assert any("kept the pronouns row" in k for k in res["kept"])


def test_a_changed_text_reruns_the_ladder_fresh(tmp_path):
    bank, source, unit = _prose_bank(tmp_path, ALICE_TEXT)
    recall, fake, _ = _recall(tmp_path, {"Bertram": "male", "Alice": "male"})
    stamper.stamp_unit(source, unit, write=True, bank_dir=bank, recall=recall)
    (bank / "sources" / "unit.txt").write_text("Alice and Bertram stood there. He nodded to him.\n",
                                               encoding="utf-8")
    res = stamper.stamp_unit(source, unit, write=True, bank_dir=bank, recall=recall)
    side = json.loads((bank / "sources" / "unit.provenance.json").read_text(encoding="utf-8"))
    rows = {r["name"]: r for r in side["characters"]}
    assert res["changed"] is True
    assert side["body_sha256"] != hashlib.sha256(ALICE_TEXT.encode("utf-8")).hexdigest()
    assert rows["Alice"]["gender_source"] != "pronouns" or rows["Alice"]["evidence"] != "kept"
    # the old pronoun row for Alice was NOT carried across the text change
    assert "She smiled" not in json.dumps(side)


def test_a_committed_row_is_carried_forward_when_its_hint_disappears():
    """QA 2026-09-02: a shrinking cast_hints list must not silently delete a
    committed row; it is carried forward with a note (and dropped only when
    the text itself changed)."""
    old = [{"name": "Alice", "gender": "female", "gender_source": "pronouns"},
           {"name": "Bertram", "gender": "male", "gender_source": "llm_recall"}]
    rows, notes = stamper._merge_rows({"Alice": dict(old[0])}, old, body_changed=False)
    assert [r["name"] for r in rows] == ["Alice", "Bertram"]
    assert any("carried forward" in n for n in notes)
    rows, _ = stamper._merge_rows({"Alice": dict(old[0])}, old, body_changed=True)
    assert [r["name"] for r in rows] == ["Alice"], "a changed text starts fresh"


def test_supplement_entry_prefers_the_longest_hint():
    bucket = {"ANTIPHOLUS": {"gender": "male", "evidence": "either twin"},
              "ANTIPHOLUS OF SYRACUSE": {"gender": "male", "evidence": "the visitor"}}
    assert stamper._supplement_entry(bucket, "ANTIPHOLUS OF SYRACUSE")["evidence"] == "the visitor"
    assert stamper._supplement_entry(bucket, "ANTIPHOLUS OF EPHESUS")["evidence"] == "either twin"
    assert stamper._supplement_entry(bucket, "ANTIPHOLUSX") is None


def test_dr_is_stripped_before_the_first_name_lookup():
    assert RG.strip_honorifics("Dr. Grimesby Roylott") == "Grimesby Roylott"
    assert RG.strip_honorifics("Dr. Watson") == "Watson"
    assert stamper.name_frequency("Dr. Alice Kelly")[0] == "female"
    # a title plus ONE token is a surname: "Kelly" is in the pool as a female
    # first name, and Dr. Kelly of Man-Size in Marble is a man.
    assert stamper.name_frequency("Dr. Kelly")[0] == ""
    assert "surname" in stamper.name_frequency("Mrs. Sappleton")[1]


def test_tier_counts_are_derived_from_the_final_rows(tmp_path):
    fresh = {"A": {"name": "A", "gender": "male", "gender_source": "llm_recall"},
             "B": None}
    old = [{"name": "A", "gender": "female", "gender_source": "pronouns"},
           {"name": "B", "gender": "male", "gender_source": "name_frequency"}]
    rows, notes = stamper._merge_rows(fresh, old, body_changed=False)
    assert [r["gender_source"] for r in rows] == ["pronouns", "name_frequency"]
    assert stamper._tier_counts(rows) == {"roster": 0, "pronouns": 1, "llm_recall": 0, "name_frequency": 1}
    assert len(notes) == 2


# --------------------------------------------------------------------------
# the Shakespeare stamp: only the unknown rows, known rows byte-identical
# --------------------------------------------------------------------------

def _scene_bank(tmp_path):
    """The live scene, with every stamper-filled row put back to the fetcher's
    `unknown` (the corpus has been stamped since 2026-09-02) and one synthetic
    unknown row nothing can answer."""
    src = _REPO / "config" / "source_banks" / "shakespeare"
    bank = tmp_path / "shakespeare"
    (bank / "sources").mkdir(parents=True)
    (bank / "sources" / "comedy_errors__act3_scene1.txt").write_bytes(
        (src / "sources" / "comedy_errors__act3_scene1.txt").read_bytes())
    data = json.loads((src / "sources" / "comedy_errors__act3_scene1.provenance.json")
                      .read_text(encoding="utf-8"))
    data.pop("gender_ladder", None)
    rows = []
    for r in data["characters"]:
        if r.get("gender_source") in ("supplement", "llm_recall", "name_frequency", "pronouns"):
            r = {k: v for k, v in r.items() if k not in ("gender_confidence", "evidence")}
            r["gender"] = "unknown"
            r["gender_source"] = "unknown"
        rows.append(r)
    rows.append({"name": "NOBODY", "roster_name": "", "description": "", "gender": "unknown",
                 "gender_source": "absent_from_roster"})
    data["characters"] = rows
    (bank / "sources" / "comedy_errors__act3_scene1.provenance.json").write_bytes(
        (json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    manifest = json.loads((src / "curated_scenes.sample.json").read_text(encoding="utf-8"))
    scene = next(s for s in manifest["scenes"] if s["text_path"].endswith("comedy_errors__act3_scene1.txt"))
    return bank, scene


def test_scene_stamp_fills_only_unknown_rows_and_keeps_known_rows_byte_identical(tmp_path):
    bank, scene = _scene_bank(tmp_path)
    side = bank / "sources" / "comedy_errors__act3_scene1.provenance.json"
    before = json.loads(side.read_text(encoding="utf-8"))
    known_before = [r for r in before["characters"] if r["gender"] in ("male", "female")]
    assert known_before, "the fixture must carry known rows"
    supplement = RG.load_gender_supplement(_REPO / "config" / "source_banks" / "shakespeare")
    recall, fake, _ = _recall(tmp_path, {"ANGELO": "male", "NOBODY": "unsure"})
    res = stamper.stamp_scene(scene, write=True, bank_dir=bank, recall=recall, supplement=supplement)
    after = json.loads(side.read_text(encoding="utf-8"))
    known_after = [r for r in after["characters"] if r["name"] in {k["name"] for k in known_before}]
    assert known_after == known_before, "ruling 1: known rows are untouchable"
    rows = {r["name"]: r for r in after["characters"]}
    # the supplement outranks the model for the names it curates
    assert rows["ANTIPHOLUS OF EPHESUS"]["gender_source"] == "supplement"
    assert rows["DROMIO OF EPHESUS"]["gender"] == "male" and rows["DROMIO OF EPHESUS"]["gender_confidence"] == "known"
    # the supplement (curated 2026-09-02) answers LUCE and BALTHASAR before any model
    assert rows["LUCE"]["gender"] == "female" and rows["LUCE"]["gender_source"] == "supplement"
    assert rows["BALTHASAR"]["gender"] == "male" and rows["BALTHASAR"]["gender_source"] == "supplement"
    # recall fills what the supplement does not name; the row keeps its fetcher fields
    angelo_before = before["characters"][[r["name"] for r in before["characters"]].index("ANGELO")]
    assert rows["ANGELO"] == {**angelo_before, "gender": "male", "gender_source": "llm_recall",
                              "gender_confidence": "recalled", "evidence": rows["ANGELO"]["evidence"]}
    assert rows["NOBODY"]["gender"] == "unknown", "unsure + not in the pool stays unknown"
    assert [d["name"] for d in res["declined"]] == ["NOBODY"]
    for n in ("ADRIANA", "LUCE", "BALTHASAR", "ANTIPHOLUS"):
        assert not any(n in c for c in fake.calls), "known and supplemented rows are never asked"
    assert b"\r\n" not in side.read_bytes()
    # a second run is a no-op
    again = stamper.stamp_scene(scene, write=True, bank_dir=bank, recall=recall, supplement=supplement)
    assert again["changed"] is False


def test_group_speakers_are_never_gendered(tmp_path):
    bank, scene = _scene_bank(tmp_path)
    side = bank / "sources" / "comedy_errors__act3_scene1.provenance.json"
    data = json.loads(side.read_text(encoding="utf-8"))
    data["characters"].append({"name": "ALL", "roster_name": "", "description": "", "gender": "unknown",
                               "gender_source": "absent_from_roster"})
    side.write_bytes((json.dumps(data, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))
    recall, fake, _ = _recall(tmp_path, {"ALL": "male"})
    stamper.stamp_scene(scene, write=True, bank_dir=bank, recall=recall, supplement={})
    rows = {r["name"]: r for r in json.loads(side.read_text(encoding="utf-8"))["characters"]}
    assert rows["ALL"]["gender"] == "unknown"
    assert not any('"ALL"' in c for c in fake.calls)


# --------------------------------------------------------------------------
# the render-time join carries the rung and the confidence
# --------------------------------------------------------------------------

def test_the_join_reads_the_stamped_aliases_so_mr_darcy_is_a_man():
    """The first acceptance leg (2026-09-02): the writer named him MR. DARCY, the
    sidecar row is Fitzwilliam Darcy with aliases [darcy, fitzwilliam], and the
    join ignored the aliases -> a roll -> a female voice. The aliases are join keys."""
    rows = [{"name": "Fitzwilliam Darcy", "gender": "male", "gender_source": "llm_recall",
             "gender_confidence": "recalled", "evidence": "recall", "aliases": ["darcy", "fitzwilliam"]},
            {"name": "Elizabeth Bennet", "gender": "female", "gender_source": "llm_recall",
             "gender_confidence": "recalled", "evidence": "recall", "aliases": ["bennet", "elizabeth"]}]
    for slot in ("MR. DARCY", "Darcy", "FITZWILLIAM DARCY", "Mr Darcy"):
        v = RG.resolve_roster_gender(slot, rows)
        assert (v.gender, v.gender_source) == ("male", "llm_recall"), (slot, v)
    assert RG.resolve_roster_gender("MISS BENNET", rows).gender == "female"
    # a surname two rows share abstains rather than picking one
    twins = [{"name": "Jane Bennet", "gender": "female", "aliases": ["bennet", "jane"]},
             {"name": "Mr. Bennet", "gender": "male", "aliases": ["bennet"]}]
    v = RG.resolve_roster_gender("BENNET", twins)
    assert (v.gender, v.evidence) == ("unknown", "ambiguous_join")
    out = RG.gender_map_for_names(["Mr. Darcy", "Elizabeth Bennet"], rows)
    assert out["MR. DARCY"]["gender"] == "male" and out["ELIZABETH BENNET"]["gender"] == "female"


def test_a_given_name_alias_matches_and_that_is_the_recorded_limit():
    """Leg 2 (2026-09-02): COLONEL FITZWILLIAM resolved through Darcy's given-name alias
    "fitzwilliam" -- the right gender by coincidence. This pins the behaviour so a change
    to surname-only aliases is a deliberate fork, not a drift."""
    rows = [{"name": "Fitzwilliam Darcy", "gender": "male", "gender_source": "llm_recall",
             "aliases": ["darcy", "fitzwilliam"]}]
    v = RG.resolve_roster_gender("COLONEL FITZWILLIAM", rows)
    assert (v.gender, v.tier, v.matched) == ("male", "short_form", ("FITZWILLIAM DARCY",))


def test_verdict_defaults_and_read_through_confidence():
    v = RG.RosterGenderVerdict("male", "x", "exact", ("X",))
    assert (v.gender_source, v.gender_confidence) == ("", "")
    rows = [{"name": "ADRIANA", "roster_name": "ADRIANA", "description": "wife", "gender": "female",
             "gender_source": "relation"}]
    v = RG.resolve_roster_gender("ADRIANA", rows)
    assert (v.gender, v.gender_source, v.gender_confidence) == ("female", "relation", "known")
    rows = [{"name": "Bertram", "gender": "male", "gender_source": "llm_recall",
             "gender_confidence": "recalled", "evidence": "recall of the work"}]
    v = RG.resolve_roster_gender("BERTRAM", rows)
    assert (v.gender_source, v.gender_confidence, v.evidence) == ("llm_recall", "recalled", "recall of the work")


def test_gender_map_carries_source_and_confidence_to_the_contract():
    rows = [{"name": "Alice", "gender": "female", "gender_source": "pronouns",
             "gender_confidence": "known", "evidence": "she"}]
    out = RG.gender_map_for_names(["Alice", "Nobody"], rows)
    assert set(out) == {"ALICE"}
    assert out["ALICE"]["gender_source"] == "pronouns" and out["ALICE"]["gender_confidence"] == "known"
    supp = {"X": {"TOBY": {"gender": "male", "evidence": "Sir Toby Belch"}}}
    out = RG.gender_map_for_names(["Toby"], [], play_code="X", supplement=supp)
    assert out["TOBY"]["gender_source"] == "supplement" and out["TOBY"]["gender_confidence"] == "known"


def test_the_seeded_shakespeare_index_locks_the_operators_two_names():
    path = _REPO / "config" / "source_banks" / "shakespeare" / "character_gender_index.json"
    index = stamper.GenderIndex(path)
    for title, name in (("The Tempest", "ARIEL"), ("A Midsummer Night's Dream", "PUCK"),
                        ("A Midsummer Night's Dream", "ROBIN")):
        entry = index.get(title, name)
        assert entry and entry["locked"] is True and entry["gender"] == "", (title, name)


@pytest.mark.parametrize("bad", ["", "male but", "{}"])
def test_parse_opinion_never_raises(bad):
    assert stamper._parse_opinion(bad)[0] == "unparseable"
