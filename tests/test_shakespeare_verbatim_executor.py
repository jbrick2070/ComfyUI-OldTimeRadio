"""The verbatim executor: a Folger scene's own speeches become the dialogue.

Operator ruling (docs/OTR_STANDING_RULINGS.md, "THE PASSAGE LANE"): a play
episode is a contiguous window of consecutive speeches "carried verbatim ...
no paraphrasing"; "`shakespeare` is VERBATIM and gets the executor." Before
this change the selector existed with zero production callers and the lane's
model wrote Shakespeare from intent and mood: the newest live ledger on
2026-09-11 (Much Ado 3.1) had HERO holding "this blade high between us" in a
scene with no blade.

THREE REVIEWERS REFUTED THE DESIGN BEFORE CODE (codex r1/r2, cursor r2, Opus
r2). What they caught, each pinned below: the outline's cast-membership check
would reject a passage whose speakers are not the manifest's first N hints
(cast follows from the cut); `_PhaseSkeleton` needs a beat per phase and a
second `EpisodeBudget` would make the writer's `arc_order` check raise, so the
act dial is honoured and the passage FILLS the topology (cap-to-fill); greedy
line packing disagrees with `ceil(words/80)` on one corpus speech, so the
chunker owns the cost; THREE post-loop rewriters reach character rows (the
judge, the transport scrub -- which unwraps Malvolio reading the letter -- and
the sayable-surface clear); and the TTS cleaners delete parenthetical SPEECH
("(God shield us!)" is Bottom's line), 39 spoken words across four speeches.

CPU-only: the vendored scenes, plain dicts, scripted slots. No model, no GPU.
"""
from __future__ import annotations

import inspect
import json
import pathlib

import pytest

from nodes import _otr_episode_budget as B
from nodes import _otr_ledger_clean as lcl
from nodes import _otr_ledger_cleanup as CL
from nodes import _otr_ledger_scrub as scrub
from nodes import _otr_outline as O
from nodes import _otr_passage_selector as PS
from nodes import _otr_script_prep as prep
from nodes import _otr_verbatim_lane as VL
from nodes import _otr_writer_inputs as WI
from nodes import _otr_story_routing as routing
from nodes import _otr_model_catalog as cat

CORPUS = (
    pathlib.Path(__file__).resolve().parent.parent
    / "config" / "source_banks" / "shakespeare" / "sources"
)
MACBETH = "folger-macbeth:act1-scene3-witches"


def _scenes():
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.txt"))}


def _words(text: str) -> str:
    return " ".join(str(text).split())


# --------------------------------------------------------------------------- #
# the chunker owns the cost; the plan fills the dial's beats, verbatim
# --------------------------------------------------------------------------- #
class TestTheChunkerAndThePlan:
    def test_the_selection_cost_IS_the_chunk_count_on_every_corpus_speech(self):
        # codex r1 MF3: an estimate the executor cannot honour is a paraphrase
        # waiting to happen. BENEDICK (Much Ado 2.3, 309 words) packs to 5, not 4.
        for name, raw in _scenes().items():
            for speech in PS.parse_speeches(raw):
                assert speech.beat_cost() == len(PS.chunk_speech(speech.text)), (name, speech.index)
                assert all(len(c.split()) <= B.BEAT_WORD_HARD_MAX + 5 for c in PS.chunk_speech(speech.text))

    def test_an_over_cap_line_splits_at_word_boundaries_and_loses_nothing(self):
        line = " ".join(f"w{i}" for i in range(170))
        chunks = PS.chunk_speech(line, cap=80)
        assert len(chunks) == 3
        assert " ".join(chunks) == line

    def test_cap_to_fill_fills_EVERY_legal_dial_with_the_passage_words_in_order(self):
        # Opus r2 MF3: the act dial is honoured exactly and the passage fills it
        # (no per-beat word floor exists). Measured over 840 combos in the
        # build; a representative slice keeps the suite fast.
        seats = VL.speaking_cast_seats()
        for name, raw in _scenes().items():
            for n in (1, 2, 6):
                cap = min(n, seats)
                for acts in (1, 3, 6):
                    beats = B.voiced_beat_count(acts)
                    passage = PS.select_passage(
                        raw, target_words=300, cast_ceiling=cap, max_beats=beats,
                        seed=f"{name}|{n}|{acts}", min_speakers=min(2, cap),
                    )
                    plan = PS.build_beat_plan(passage, beat_count=beats)
                    assert len(plan) == beats, (name, n, acts)
                    assert _words(" ".join(e.text for e in plan)) == _words(
                        " ".join(s.text for s in passage.speeches)), (name, n, acts)
                    for e in plan:
                        assert e.speaker == passage.speeches[e.speech_index].speaker
                        assert e.text.strip()

    def test_a_soliloquy_is_a_legal_passage_at_cast_one(self):
        # codex r1 MF2: num_characters=1 is a legal widget value and raised on
        # all 14 scenes under min_speakers=2.
        raw = _scenes()["romeo_juliet__act2_scene2.txt"]
        passage = PS.select_passage(raw, target_words=300, cast_ceiling=1,
                                    max_beats=4, seed="x", min_speakers=1)
        assert len(passage.speakers) == 1

    def test_a_plan_with_fewer_beats_than_speeches_is_refused_not_merged(self):
        raw = _scenes()["macbeth__act1_scene3.txt"]
        passage = PS.select_passage(raw, target_words=300, cast_ceiling=6,
                                    max_beats=14, seed="x")
        with pytest.raises(PS.PassageError, match="never shares a beat"):
            PS.build_beat_plan(passage, beat_count=passage.speech_count - 1)

    def test_the_rendered_passage_parses_back_into_the_same_speeches(self):
        raw = _scenes()["hamlet__act1_scene1.txt"]
        passage = PS.select_passage(raw, target_words=300, cast_ceiling=4,
                                    max_beats=12, seed="x")
        back = PS.parse_speeches(PS.render_passage_text(passage))
        assert [(s.speaker, _words(s.text)) for s in back] == [
            (s.speaker, _words(s.text)) for s in passage.speeches]


# --------------------------------------------------------------------------- #
# the plan step
# --------------------------------------------------------------------------- #
class TestThePlanStep:
    def test_the_gate_is_the_typed_bank_default_and_only_shakespeare_carries_it(self):
        banks = {b.source_bank_id: b for b in routing.list_banks()} if hasattr(routing, "list_banks") else {
            bid: routing.get_bank(bid) for bid in
            ("media_archive", "original", "scifi_news_pro", "public_domain", "shakespeare", "my_story")}
        flagged = sorted(bid for bid, row in banks.items() if VL.bank_is_verbatim(row))
        assert flagged == ["shakespeare"]

    def test_a_non_bool_gate_is_rejected_by_the_registry(self):
        with pytest.raises(routing.RegistryValidationError, match="verbatim_passage must be a bool"):
            routing._parse_bank({
                "source_bank_id": "x", "label": "x", "source_kind": "k",
                "interpreter": "i", "fetcher": "f", "defaults": {"verbatim_passage": "yes"},
            }, "t")

    def test_a_scene_plans_and_the_receipt_is_body_free(self, monkeypatch):
        monkeypatch.setenv("OTR_CAST_SEED", "42")
        monkeypatch.setenv("OTR_C7", "1")
        raw = _scenes()["macbeth__act1_scene3.txt"]
        plan, receipt = VL.plan_verbatim_passage(
            source_text=raw, source_meta={"recommended_word_budget": 300},
            num_characters=2, act_count=1, source_ref=MACBETH)
        assert plan is not None and receipt["status"] == "planned"
        assert receipt["beat_count"] == 4 == plan.beat_count
        assert receipt["num_characters_operator_request"] == 2
        assert receipt["seed"] == f"42|{MACBETH}"
        assert receipt["raw_sha256"] and len(receipt["raw_sha256"]) == 64
        assert receipt["chunker_version"] == PS.CHUNKER_VERSION
        assert receipt["selector_version"] == PS.SELECTOR_VERSION
        blob = json.dumps(receipt)
        for e in plan.entries:
            assert e.text not in blob, "the receipt must never carry body text"

    def test_the_same_seed_plans_the_same_passage(self, monkeypatch):
        monkeypatch.setenv("OTR_CAST_SEED", "7")
        monkeypatch.setenv("OTR_C7", "1")
        raw = _scenes()["king_lear__act1_scene1.txt"]
        a = VL.plan_verbatim_passage(source_text=raw, source_meta={"recommended_word_budget": 300},
                                     num_characters=3, act_count=2, source_ref="r")[0]
        b = VL.plan_verbatim_passage(source_text=raw, source_meta={"recommended_word_budget": 300},
                                     num_characters=3, act_count=2, source_ref="r")[0]
        assert [e.text for e in a.entries] == [e.text for e in b.entries]

    def test_no_text_is_a_receipted_miss_never_a_raise(self):
        plan, receipt = VL.plan_verbatim_passage(
            source_text=None, source_meta={"recommended_word_budget": 300},
            num_characters=2, act_count=1, source_ref="r")
        assert plan is None and receipt["status"] == "unavailable"
        assert "no line-structured source text" in receipt["reason"]

    def test_a_missing_budget_is_a_receipted_miss(self):
        plan, receipt = VL.plan_verbatim_passage(
            source_text="ORLANDO\nHang there.\n\nTOBY  Come thy ways.\n",
            source_meta={}, num_characters=2, act_count=1, source_ref="r")
        assert plan is None and "recommended_word_budget" in receipt["reason"]

    def test_a_source_with_no_performable_window_is_a_receipted_miss(self):
        plan, receipt = VL.plan_verbatim_passage(
            source_text="ORLANDO\nHang there, my verse.\n",
            source_meta={"recommended_word_budget": 300},
            num_characters=2, act_count=1, source_ref="r")
        assert plan is None and receipt["status"] == "unavailable"

    def test_the_projection_keeps_the_payload_contract_and_swaps_the_material(self, monkeypatch):
        monkeypatch.setenv("OTR_CAST_SEED", "1")
        monkeypatch.setenv("OTR_C7", "1")
        raw = _scenes()["macbeth__act1_scene3.txt"]
        plan, _ = VL.plan_verbatim_passage(
            source_text=raw, source_meta={"recommended_word_budget": 300},
            num_characters=2, act_count=1, source_ref=MACBETH)
        article = {"headline": "Macbeth, Act 1, Scene 3", "summary": "synopsis",
                   "full_text": "WHOLE SCENE", "source": "Folger", "date": "1606",
                   "link": "u", "seed_text": "Macbeth\nSynopsis: s\nSpeakers: a, b\nExcerpt: old"}
        out = VL.project_payload(article, plan)
        assert set(out) == set(article)
        assert out["full_text"] == plan.passage_text
        assert out["summary"] == "synopsis" and out["headline"] == article["headline"]
        assert "Excerpt: old" not in out["seed_text"] and plan.passage_text in out["seed_text"]

    def test_the_non_verbatim_credit_reads_naturally(self):
        assert VL.non_verbatim_credit_line("adapted from Macbeth (CC BY-NC 3.0)") == \
            "freely adapted from Macbeth (CC BY-NC 3.0)"
        assert VL.non_verbatim_credit_line("used under CC BY-NC 3.0").startswith("freely adapted from the source; ")
        assert VL.non_verbatim_credit_line("") == "freely adapted from the source"


# --------------------------------------------------------------------------- #
# resolve time: the one site, three banks
# --------------------------------------------------------------------------- #
class TestResolveTime:
    def test_shakespeare_resolves_a_plan_and_projects_the_payload(self):
        r = WI._resolve_inputs(num_characters=2, act_count="1",
                               creative_writing_model=cat.DEFAULT_LLM,
                               technical_model=cat.DEFAULT_LLM,
                               source_bank="shakespeare", source_ref=MACBETH)
        plan, receipt = r["verbatim_plan"], r["verbatim_receipt"]
        assert receipt["status"] == "planned" and plan.beat_count == 4
        assert r["news_article"]["full_text"] == plan.passage_text
        assert r["news_seed"] == r["news_article"]["seed_text"]
        assert r["source_meta"]["raw_sha256"] == receipt["raw_sha256"]
        assert set(plan.speakers) <= {s.speaker for s in PS.parse_speeches(
            _scenes()["macbeth__act1_scene3.txt"])}

    def test_a_custom_premise_on_the_bank_is_a_receipted_miss_with_the_premise_kept(self):
        # codex r2 MF3: the premise path fetches no scene; never fetch one to
        # replace what the operator typed.
        r = WI._resolve_inputs(num_characters=2, act_count="1",
                               creative_writing_model=cat.DEFAULT_LLM,
                               technical_model=cat.DEFAULT_LLM,
                               source_bank="shakespeare",
                               custom_premise="my own premise on the bank")
        assert r["verbatim_plan"] is None
        assert r["verbatim_receipt"]["status"] == "unavailable"
        assert r["news_seed"] == "my own premise on the bank"

    @pytest.mark.parametrize("bank,premise", [
        ("original", "a town wakes to a strange signal"),
        ("public_domain", ""),
    ])
    def test_other_banks_never_reach_the_selector(self, monkeypatch, bank, premise):
        # public_domain shares style_pool_class=adaptation and is ruled fuzzy
        # prose; the gate is the typed default, not the pool class.
        def _never(*a, **k):
            raise AssertionError("the selector ran on a bank without the gate")
        monkeypatch.setattr(PS, "select_passage", _never)
        kwargs = dict(num_characters=2, act_count="1",
                      creative_writing_model=cat.DEFAULT_LLM,
                      technical_model=cat.DEFAULT_LLM, source_bank=bank)
        if premise:
            kwargs["custom_premise"] = premise
        r = WI._resolve_inputs(**kwargs)
        assert r["verbatim_plan"] is None and r["verbatim_receipt"] == {}


# --------------------------------------------------------------------------- #
# the outline takes the plan
# --------------------------------------------------------------------------- #
PLAN_SPEAKERS = ("MACBETH", "BANQUO", "MACBETH", "MACBETH")
PLAN_TEXTS = ("Glamis and Thane of Cawdor!", "Good sir, why do you start",
              "Two truths are told", "This supernatural soliciting")


def _outline_stub(seen):
    def stub(messages, *, temperature, max_new_tokens):
        user = messages[-1]["content"]
        seen.append(user)
        if "Task: write the intent" in user:
            return json.dumps({"intent": "advance", "mood": "tense"})
        if "beats" in user and "title" not in user:
            # Stage 2 answering with ONE speaker for every beat: wrong for the
            # plan, and exactly what must never be consulted on this lane.
            return json.dumps({"beats": [{"speaker": "MACBETH"}] * 4})
        return json.dumps({"title": "T", "premise": "P", "setting": "S",
                           "time_of_day": "night", "central_tension": "C"})
    return stub


def _req():
    budget = B.compute_episode_budget(act_count=1, include_act_breaks=True, num_characters=2)
    return O.OutlineRequest(
        news_seed="seed", style="", character_cast=("MACBETH", "BANQUO"),
        script_brief="brief", key_terms=(), cast_descriptions={},
        include_act_breaks=True, budget=budget, prior_macro="", prior_critique="",
        style_grammar="", story_engine="", work_title="Macbeth")


class TestTheOutlineTakesThePlan:
    def test_speakers_are_the_plan_in_order_and_stage_3_sees_the_fixed_words(self):
        seen: list[str] = []
        out = O.generate_outline(_outline_stub(seen), _req(), source_bank_id="shakespeare",
                                 speaker_plan=PLAN_SPEAKERS, verbatim_texts=PLAN_TEXTS)
        chars = [b for b in out.beats if b.speaker_role == "character"]
        assert tuple(b.speaker for b in chars) == PLAN_SPEAKERS
        beat_prompts = [p for p in seen if "Task: write the intent" in p]
        assert len(beat_prompts) == 4
        assert all(t in p for t, p in zip(PLAN_TEXTS, beat_prompts))
        # codex r1 MF7: the escalation instruction cannot compete with fixed words.
        assert not any("RAISE THE STAKE" in p for p in beat_prompts)

    def test_an_unseated_speaker_dies_loud(self):
        with pytest.raises(ValueError, match="did not seat"):
            O.generate_outline(_outline_stub([]), _req(), source_bank_id="shakespeare",
                               speaker_plan=("LADY MACBETH",) * 4, verbatim_texts=PLAN_TEXTS)

    def test_a_plan_that_does_not_match_the_topology_dies_loud(self):
        with pytest.raises(ValueError, match="topology of 4"):
            O.generate_outline(_outline_stub([]), _req(), source_bank_id="shakespeare",
                               speaker_plan=PLAN_SPEAKERS[:3], verbatim_texts=PLAN_TEXTS[:3])

    def test_WITHOUT_a_plan_the_prompts_are_the_old_ones(self):
        seen: list[str] = []
        O.generate_outline(_outline_stub(seen), _req(), source_bank_id="shakespeare")
        assert any("RAISE THE STAKE" in p for p in seen)
        assert not any("SPOKEN WORDS (fixed" in p for p in seen)


# --------------------------------------------------------------------------- #
# ownership to the end: judge, scrub, sayable clear, coverage, TTS
# --------------------------------------------------------------------------- #
LETTER = '"You must amend your drunkenness."'          # Malvolio, reading aloud
PAREN = "to bring in (God shield us!) a lion among ladies"   # Bottom


def _ledger(text: str, *, verbatim: bool) -> dict:
    return {
        "schema_version": "l3", "episode_id": "verbatim-contract",
        "cast": [{"char_id": "c01", "name": "MALVOLIO"},
                 {"char_id": "announcer", "name": "ANNOUNCER"}],
        "beats": [{"beat_id": "b000", "speaker": "ANNOUNCER"},
                  {"beat_id": "b001", "speaker": "MALVOLIO", "beat_intent": "reads"}],
        "lines": [
            {"line_id": "L000", "beat_id": "b000", "char_id": "announcer",
             "speaker": "ANNOUNCER", "speaker_role": "announcer",
             "text": "Tonight, a scene from Twelfth Night."},
            {"line_id": "L001", "beat_id": "b001", "char_id": "c01",
             "speaker": "MALVOLIO", "speaker_role": "character", "text": text,
             "compose_flags": [scrub.VERBATIM_SOURCE_FLAG] if verbatim else []},
        ],
        "meta": {"source_bank": "shakespeare"},
    }


class TestPythonOwnsTheRowsToTheEnd:
    def test_the_scrub_leaves_a_verbatim_row_alone_and_unwraps_the_same_row_without_the_flag(self):
        # Opus r2 MF1, measured on the corpus: the transport unwrap eats
        # Malvolio reading Olivia's letter and the apostrophe of 'Tis.
        led = _ledger(LETTER, verbatim=True)
        scrub.scrub_ledger(led)
        assert led["lines"][1]["text"] == LETTER
        led = _ledger(LETTER, verbatim=False)
        scrub.scrub_ledger(led)
        assert led["lines"][1]["text"] == "You must amend your drunkenness."

    def test_the_judge_never_sees_a_verbatim_row(self):
        try:
            from test_ledger_clean_stage import _Slot, _dirty_judgement
        except ImportError:  # pragma: no cover
            from tests.test_ledger_clean_stage import _Slot, _dirty_judgement  # type: ignore
        slot = _Slot(
            judgements={"drunkenness": [_dirty_judgement(LETTER)]},
            repairs={"drunkenness": [{"replacements": [
                {"span_id": "span_001", "replacement": "You should drink less."}]}]},
        )
        led = _ledger(LETTER, verbatim=True)
        receipt = lcl.run_ledger_clean(led, slot_fn=slot, bank_id="shakespeare")
        assert led["lines"][1]["text"] == LETTER
        assert "L001" in receipt["protected_rows"]
        assert scrub.VERBATIM_SOURCE_FLAG in lcl.PROTECTED_ROW_FLAGS

    def test_a_parenthetical_only_verbatim_row_keeps_its_voice(self):
        # codex r2 SF3: the sayable-surface clear used the paren-stripping
        # cleaner; a verbatim row made only of a parenthetical is still speech.
        led = _ledger("(God shield us!)", verbatim=True)
        try:
            CL.run_ledger_cleanup(led)
        except Exception:  # unrelated fixture gaps are not the subject
            pass
        assert led["lines"][1].get("skip") is not True
        assert led["lines"][1]["text"] == "(God shield us!)"
        led = _ledger("(God shield us!)", verbatim=False)
        try:
            CL.run_ledger_cleanup(led)
        except Exception:
            pass
        assert led["lines"][1].get("skip") is True

    def test_the_stripper_keeps_parenthetical_words_only_when_told_the_row_is_verbatim(self):
        # codex r2 MF2 / Opus r2 MF5: 39 spoken words across four corpus
        # speeches sit inside parentheses.
        assert prep.clean_spoken_text(PAREN) == "to bring in a lion among ladies"
        assert prep.clean_spoken_text(PAREN, keep_parentheticals=True) == \
            "to bring in God shield us! a lion among ladies"
        assert prep.keep_spoken_parentheticals(PAREN).split() == \
            "to bring in God shield us! a lion among ladies".split()
        # a stage direction in brackets is still not speech, either way
        assert prep.clean_spoken_text("[aside] Now, divine air!", keep_parentheticals=True) == "Now, divine air!"

    def test_the_corpus_parentheticals_survive_the_projection_and_every_engine_cleaner(self):
        import re
        lost = 0
        for name, raw in _scenes().items():
            for speech in PS.parse_speeches(raw):
                if "(" not in speech.text:
                    continue
                for chunk in PS.chunk_speech(speech.text):
                    projected = prep.keep_spoken_parentheticals(chunk)
                    after_engine = prep.clean_spoken_text(projected)      # chatterbox / indextts2 / dia
                    after_bark = re.sub(r"\([^)]{1,80}\)\s*", "", projected)  # bark's own stripper
                    for text in (after_engine, after_bark):
                        if _words(re.sub(r"[()]", " ", chunk)) != _words(text):
                            lost += 1
        assert lost == 0

    def test_the_coverage_gate_counts_a_parenthetical_only_verbatim_row_as_a_voice(self):
        from nodes import _otr_cast_voice_coverage as cov
        assert cov._sayable("(God shield us!)") is False
        assert cov._sayable("(God shield us!)", keep_parentheticals=True) is True


# --------------------------------------------------------------------------- #
# the wiring exists at its real sites (source inspection is the right tool for
# THIS question only: is the call there?)
# --------------------------------------------------------------------------- #
class TestTheWiringIsThere:
    def test_the_writer_calls_the_executor_at_every_seam(self):
        from nodes import OTR_LedgerScriptWriter as W
        src = inspect.getsource(W.OTR_LedgerScriptWriter)
        assert 'resolved.get("verbatim_plan")' in src
        assert "_adapt_names = list(_verbatim_plan.speakers)" in src, "the cast must follow from the cut"
        assert "len(_verbatim_plan.speakers) if _verbatim_plan is not None" in src, "lock_cast gets the executable size"
        assert '"num_characters_operator_request": int(resolved["num_characters"])' in src
        assert "speaker_plan=(" in src and "verbatim_texts=(" in src, "the outline takes the plan"
        assert 'bool(resolved.get("use_exchange", False)) and not _verbatim_entries' in src, "the exchange yields"
        assert 'if beat.speaker_role == "character" and _verbatim_entries:' in src, "the executor branch"
        assert "beat_compose_flags = (_OTRSCRUB.VERBATIM_SOURCE_FLAG,)" in src
        assert "non_verbatim_credit_line(" in src, "a miss is visible on the credits"
        # and the cast mint is untouched (Opus r2 MF4)
        assert "cast_seed, cast_seed_source = _resolve_cast_rng_seed()" in src

    def test_the_voice_hook_projects_verbatim_rows_before_any_engine_cleaner(self):
        from nodes import _otr_voice_node_common as V
        src = inspect.getsource(V)
        hook = src.index("text = _delivery_text.strip()")
        projection = src.index("_keep_spoken_parentheticals(text)")
        prepared = src.index("prepared = prep(text, delivery_vector)")
        assert hook < projection < prepared

    def test_resolve_inputs_reads_the_raw_text_off_the_fetch_result(self):
        src = inspect.getsource(WI._resolve_inputs)
        assert 'source_text = getattr(_fetch_result, "source_text", None)' in src
        assert "plan_verbatim_passage(" in src and "project_payload(" in src
        assert "raw_text_for_snapshot(" in src, "a snapshot replay must prove its bytes"


# --------------------------------------------------------------------------- #
# codex r3 on the finished diff: the freeze, the naming overlay, the cap
# --------------------------------------------------------------------------- #
HAMLET = "folger-hamlet:act1-scene1-platform-watch"


def _rows_from_plan(plan) -> dict:
    """A ledger whose character rows are the plan's entries, flagged."""
    cast = [{"char_id": "announcer", "name": "ANNOUNCER"}] + [
        {"char_id": f"c{i + 1:02d}", "name": s} for i, s in enumerate(plan.speakers)]
    cid = {row["name"]: row["char_id"] for row in cast}
    lines = [{"line_id": "L000", "beat_id": "b000", "char_id": "announcer",
              "speaker": "ANNOUNCER", "speaker_role": "announcer",
              "text": "Tonight, a scene.", "skip": False, "tts_skip_reason": ""}]
    beats = [{"beat_id": "b000", "speaker": "ANNOUNCER"}]
    for i, e in enumerate(plan.entries, start=1):
        bid = f"b{i:03d}"
        beats.append({"beat_id": bid, "speaker": e.speaker})
        lines.append({"line_id": f"L{i:03d}", "beat_id": bid, "char_id": cid[e.speaker],
                      "speaker": e.speaker, "speaker_role": "character", "text": e.text,
                      "skip": False, "tts_skip_reason": "",
                      "compose_flags": [scrub.VERBATIM_SOURCE_FLAG],
                      "word_count": len(e.text.split()), "char_count": len(e.text)})
    return {"schema_version": "l3", "episode_id": "hamlet-verbatim", "cast": cast,
            "beats": beats, "lines": lines, "meta": {"source_bank": "shakespeare"}}


class TestTheFreezeAndTheReceiptRespectTheFlag:
    def test_a_wholly_parenthetical_chunk_passes_the_freeze_and_fails_it_without_the_flag(self, monkeypatch):
        # codex r3 MF1, executed: Hamlet 1.1, cast 1, acts 4, seed 0 selects a
        # window whose chunk is "(For so this side of our known world esteemed
        # him)". The freeze's sayable check stripped it and killed the episode.
        from nodes import _otr_ledger_freeze as FZ
        # Hamlet 1.1 line 182 is a whole parenthetical line; cut as its own beat
        # it is a wholly parenthetical row. Built directly so the test does not
        # depend on which window a seed happens to draw.
        raw = _scenes()["hamlet__act1_scene1.txt"]
        assert "(For so this side of our known world esteemed him)" in raw
        entries = (
            PS.BeatPlanEntry(speaker="HORATIO",
                             text="(For so this side of our known world esteemed him)",
                             speech_index=47, chunk_ordinal=0, chunk_count=2),
            PS.BeatPlanEntry(speaker="HORATIO", text="Did slay this Fortinbras;",
                             speech_index=47, chunk_ordinal=1, chunk_count=2),
        )
        plan = VL.VerbatimPlan(entries=entries, speakers=("HORATIO",),
                               passage_text="", seed="", receipt={})
        led = _rows_from_plan(plan)
        errors, warnings, info = [], [], {}
        FZ._check_per_line_invariants(led, errors, warnings, info)
        assert not [e for e in errors if "cleans to empty" in e], errors
        for row in led["lines"]:
            row["compose_flags"] = []
        errors = []
        FZ._check_per_line_invariants(led, errors, warnings, info)
        assert [e for e in errors if "cleans to empty" in e]

    def test_the_authorship_receipt_counts_the_parenthetical_row(self):
        from nodes import _otr_content_authorship as CA
        assert CA._sayable("(God shield us!)") is False
        assert CA._sayable("(God shield us!)", keep_parentheticals=True) is True


class TestTheNamingOverlayLeavesSourceNamesAlone:
    def _slots(self):
        from nodes import _otr_casting as C
        return C.EnsembleSlot(char_id="c01", name="MACBETH", gender="male",
                              timbre=C._TIMBRE_VOCAB[0], role=C._ROLE_VOCAB[0],
                              source_owned=True)

    def test_a_cast_of_only_source_owned_slots_never_calls_the_naming_model(self):
        # codex r3 MF2, executed: OTR_NAME_MODE=llm_slot_fill renamed a
        # source-owned MACBETH to JOHN SMITH; the outline then raised on the
        # missing speaker.
        from nodes import _otr_casting as C
        cast = [{"char_id": "announcer", "name": "ANNOUNCER"},
                {"char_id": "c01", "name": "MACBETH", "gender": "male"}]

        def _never(*a, **k):
            raise AssertionError("the naming model was asked to rename a source-owned slot")
        meta: dict = {}
        out = C._apply_llm_slot_fill(
            cast, [self._slots()], {"c01": "v2/en_speaker_1"}, {"c01": "adult"},
            generate_fn=_never, news_seed="seed", style="", cast_seed=1, meta=meta)
        assert out[1]["name"] == "MACBETH"
        assert meta.get("llm_naming_applied") is False

    def test_a_mixed_cast_renames_only_the_pool_slot(self):
        from nodes import _otr_casting as C
        from nodes._otr_cast_validator import ALLOWED_PASS1_KEYS
        pool = C.EnsembleSlot(char_id="c02", name="ALICE VALE", gender="female",
                              timbre=C._TIMBRE_VOCAB[0], role=C._ROLE_VOCAB[0])
        cast = [{"char_id": "announcer", "name": "ANNOUNCER"},
                {"char_id": "c01", "name": "MACBETH", "gender": "male"},
                {"char_id": "c02", "name": "ALICE VALE", "gender": "female"}]
        asked: list = []

        def _name_them(messages, *, temperature, max_new_tokens):
            asked.append(messages[-1]["content"])
            item = {k: "" for k in ALLOWED_PASS1_KEYS}
            item.update({"char_id": "c02", "name": "MARGARET HOLT"})
            for k in ALLOWED_PASS1_KEYS:
                if k not in ("char_id", "name") and not item[k]:
                    item[k] = "a steady presence"
            return json.dumps([item])
        meta: dict = {}
        out = C._apply_llm_slot_fill(
            cast, [self._slots(), pool], {"c01": "v2/en_speaker_1", "c02": "v2/en_speaker_2"},
            {"c01": "adult", "c02": "adult"},
            generate_fn=_name_them, news_seed="seed", style="", cast_seed=1, meta=meta)
        assert out[1]["name"] == "MACBETH"
        assert "MACBETH" not in asked[0], "the source-owned slot must not even be offered"


class TestNoEntryExceedsTheCap:
    def test_a_long_line_speech_is_cut_at_the_cap_not_left_whole(self):
        # codex r3 SF1, executed: HORATIO with lines of 140/10/10/10 words
        # produced entries of 140/10/10/10 -- the line-boundary rule kept the
        # 140-word line whole. The plan now cuts from the chunker's groups.
        long_line = " ".join(f"w{i}" for i in range(140))
        text = "HORATIO\n" + long_line + "\nten words in this line of verse to count it\n" * 3 + "\nMARCELLUS\nPeace, break thee off.\n"
        passage = PS.select_passage(text, target_words=170, cast_ceiling=2,
                                    max_beats=4, seed="x", min_speakers=1)
        plan = PS.build_beat_plan(passage, beat_count=4)
        assert len(plan) == 4
        assert all(len(e.text.split()) <= B.BEAT_WORD_HARD_MAX for e in plan)
        assert _words(" ".join(e.text for e in plan)) == _words(" ".join(s.text for s in passage.speeches))

    def test_every_corpus_plan_entry_is_within_the_cap(self):
        seats = VL.speaking_cast_seats()
        for name, raw in _scenes().items():
            for n, acts in ((1, 1), (2, 3), (6, 6)):
                cap = min(n, seats)
                beats = B.voiced_beat_count(acts)
                passage = PS.select_passage(raw, target_words=300, cast_ceiling=cap,
                                            max_beats=beats, seed=f"{name}|{n}",
                                            min_speakers=min(2, cap))
                for e in PS.build_beat_plan(passage, beat_count=beats):
                    assert len(e.text.split()) <= B.BEAT_WORD_HARD_MAX + 2, (name, n, acts)


class TestEveryWordReachesEveryEngine:
    def test_the_projection_survives_readiness_delivery_and_every_registered_prepare_text(self):
        # codex r3 SF2: drive the real chain -- the readiness stamp, the delivery
        # resolver, the verbatim projection, then each registered engine's OWN
        # prepare_text (synthesis never runs) -- on every corpus chunk that
        # carries a parenthetical.
        import re
        from nodes._otr_readiness import stamp_text_for_tts_delivery
        from nodes._otr_text_delivery import resolve_line_delivery, LEGACY
        from nodes._otr_audio_engines import registry as REG
        engines = []
        for name in sorted(REG._REGISTRY):
            eng = REG.get_engine(name)
            if callable(getattr(eng, "prepare_text", None)):
                engines.append((name, eng))
        assert engines, "no engine exposes prepare_text"
        checked = 0
        for scene, raw in _scenes().items():
            for speech in PS.parse_speeches(raw):
                if "(" not in speech.text:
                    continue
                for chunk in PS.chunk_speech(speech.text):
                    row = {"line_id": "L1", "beat_id": "b001", "char_id": "c01",
                           "speaker": speech.speaker, "speaker_role": "character",
                           "text": chunk, "skip": False,
                           "compose_flags": [scrub.VERBATIM_SOURCE_FLAG]}
                    led = {"lines": [row], "meta": {}}
                    stamp_text_for_tts_delivery(led)
                    _canonical, delivery = resolve_line_delivery(row, LEGACY)
                    projected = prep.keep_spoken_parentheticals(delivery.strip()).strip()
                    want = _words(re.sub(r"[()]", " ", delivery))
                    for name, eng in engines:
                        try:
                            got = eng.prepare_text(projected, None)
                        except TypeError:
                            got = eng.prepare_text(projected)
                        assert _words(got) == want, (scene, speech.index, name, got)
                    checked += 1
        assert checked >= 4


class TestMultiWordSourceNamesSurviveTheAssembler:
    def test_the_assembler_seats_multi_word_source_names_with_their_exact_spelling(self):
        # Sonnet QA nit: the writer looks the cast up by the outline's speaker
        # name, so the assembler must seat "FIRST WITCH" and "ANTIPHOLUS OF
        # EPHESUS" exactly as the parser spells them -- and Lemmy must not take a
        # seat sized to the passage.
        import random
        from nodes import _otr_casting as C
        pre, open_slots, lemmy = C.assemble_pre_locked_rows(
            num_characters=2, rng=random.Random(1), force_lemmy=True,
            source_character_names=["FIRST WITCH", "ANTIPHOLUS OF EPHESUS"],
            source_bank_id="shakespeare")
        assert [s.name for s in open_slots] == ["FIRST WITCH", "ANTIPHOLUS OF EPHESUS"]
        assert all(s.source_owned for s in open_slots)
        assert lemmy is False, "the fidelity bank never seats the cameo"
