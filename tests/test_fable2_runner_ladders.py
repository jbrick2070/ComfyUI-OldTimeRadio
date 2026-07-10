"""scifi_fable2 S1b -- runner ladders subset + spine end-to-end
(architecture doc s3/s8/s13).

Covers: the P3 markup ladder (clean pass / defect-quoting reroll at
falling temperature / +25% truncation retry ONCE / budget-gate reroll
with numeric hint, max 2, then fail loud / exhaustion), the P8 triage
evidence bar (lexicon-only kills; python-verified structural discards;
taste classes report-only), the FULL no-model end-to-end spine
(run_scifi_fable2_episode with scripted fake LLMs into a REAL tmp
ledger), the writer-level lane dispatch + unmapped-runner raise (r3/S1),
the writer entry gates (r3/M4), and the pure-import pin (r4/M3).
"""
from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from nodes import _otr_scifi_fable2 as F2  # noqa: E402
from nodes import _otr_story_routing as ROUTING  # noqa: E402
from nodes import production_ledger as _PL  # noqa: E402


@pytest.fixture(autouse=True)
def _preserve_current_ledger():
    saved = _PL._CURRENT
    yield
    _PL._CURRENT = saved


@pytest.fixture(autouse=True)
def _fresh_registries():
    ROUTING._REGISTRY = None
    yield
    ROUTING._REGISTRY = None


# ---------------------------------------------------------------------------
# Shared script-pass fixtures
# ---------------------------------------------------------------------------

_GOOD_MARKUP = "\n".join([
    "TITLE: The Long Count",
    "MUSIC: slow theremin swell",
    "ANNOUNCER: Tonight, one antenna, two signals, and a woman who "
    "refused to choose.",
    "SCENE 1: A cliff-top listening station before dawn",
    "SELA: Play it again, and this time keep the gain low.",
    "DARROW: The tape is the tape, Sela. It will not change for us.",
    "SELA: Then we change the question we are asking it.",
    "ANNOUNCER: Sela got her answer, though the antenna kept the "
    "better half.",
    "CODA: Beyond tonight's cliff-top signal, a real transmission waits:",
    "MUSIC: closing theme, warm brass",
    "END.",
])  # 31 character words -> inside the 24-36 band for target 30

_TRUNCATED_MARKUP = _GOOD_MARKUP.rsplit("\nEND.", 1)[0]  # no END.

_BAD_SHAPE_MARKUP = _GOOD_MARKUP.replace(
    "SCENE 1: A cliff-top listening station before dawn",
    "SCENE 1: A cliff-top listening station before dawn\n"
    "this line matches no shape at all")

_FAT_MARKUP = _GOOD_MARKUP.replace(
    "SELA: Then we change the question we are asking it.",
    "SELA: Then we change the question we are asking it tonight, "
    "because the committee will not sit for the mountain, and the "
    "mountain has never once agreed to sit for the committee, and "
    "somebody in this room must finally choose which of the two "
    "keeps the log.")  # blows past the 36-word band ceiling

_CAST = ["SELA", "DARROW"]


def _treatment() -> F2.Treatment:
    return F2.Treatment.model_validate(_TREATMENT_DICT)


_TREATMENT_DICT = {
    "title": "The Long Count",
    "dramatic_question": (
        "Will Sela trust her own instruments before the village stops "
        "trusting her?"),
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
        "The heat readings Sela hid to protect her credibility are the "
        "only proof that would move the village in time."),
    "priced_ending": {
        "choice": "Sela publishes the readings under her own name",
        "cost_paid": "her seat on the committee, surrendered in writing",
    },
    "news_thread": (
        "new vents on Mount Etna released measurable heat before any "
        "tremor"),
    "news_close_read": (
        "Tonight's story grew from a real survey: instruments on Mount "
        "Etna mapped 1,200 new vents this season, work led by Doctor "
        "Rossi."),
}


class _ScriptedFn:
    """Fake slot fn: pops scripted responses; records every call's
    kwargs. Entries may be callables (evaluated lazily at call time)."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: "list[dict]" = []

    def __call__(self, msgs, *, temperature, max_new_tokens):
        self.calls.append({
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "msgs": msgs,
        })
        if not self.responses:
            raise AssertionError("fake slot fn ran out of responses")
        item = self.responses.pop(0)
        return item() if callable(item) else item


def _pack():
    return ROUTING.resolve_story_pack("scifi_fable2")


def _run_script_pass(fn):
    return F2._pass_script(
        fn, _pack(), _treatment(),
        "digest text about Mount Etna and Doctor Rossi",
        F2._build_envelope(30), _CAST)


# ---------------------------------------------------------------------------
# 1. P3 markup ladder
# ---------------------------------------------------------------------------

class TestScriptLadder:
    def test_clean_first_attempt(self):
        fn = _ScriptedFn([_GOOD_MARKUP])
        raw, parsed, meta = _run_script_pass(fn)
        assert parsed.title == "The Long Count"
        assert meta["attempts"] == 1 and meta["rerolls"] == 0
        assert fn.calls[0]["temperature"] == F2._TEMP["script"]
        assert fn.calls[0]["max_new_tokens"] == F2._script_token_budget(30)

    def test_defect_reroll_quotes_defects_and_lowers_temperature(self):
        fn = _ScriptedFn([_BAD_SHAPE_MARKUP, _GOOD_MARKUP])
        raw, parsed, meta = _run_script_pass(fn)
        assert meta["attempts"] == 2
        assert meta["defects_by_attempt"][0]
        assert any("BAD_LINE_SHAPE" in d
                   for d in meta["defects_by_attempt"][0])
        # rung 2 runs LOWER (never higher) and the reroll message quotes
        # the defect list (python judges; the LLM rewrites)
        assert fn.calls[1]["temperature"] < fn.calls[0]["temperature"]
        reroll_msg = fn.calls[1]["msgs"][-1]["content"]
        assert "BAD_LINE_SHAPE" in reroll_msg
        # Strict role alternation -- local chat templates raise jinja
        # TemplateError on consecutive same-role messages (first S1b live
        # smoke); the FORMAT example play rides an assistant few-shot
        # turn (fourth S1b live smoke: prompt-side bans alone never cured
        # the **bold**/(paren) habits).
        for call in fn.calls:
            assert [m["role"] for m in call["msgs"]] == [
                "system", "user", "assistant", "user"]
            assert call["msgs"][2]["content"].startswith(
                "TITLE: The Long Count")
            assert call["msgs"][2]["content"].endswith("END.")

    def test_truncation_retry_adds_25_percent_tokens_once(self):
        fn = _ScriptedFn([_TRUNCATED_MARKUP, _GOOD_MARKUP])
        raw, parsed, meta = _run_script_pass(fn)
        assert meta["truncation_retry"] is True
        t0 = fn.calls[0]["max_new_tokens"]
        assert fn.calls[1]["max_new_tokens"] == int(t0 * 1.25)
        # retry stays on the SAME rung (temperature unchanged)
        assert fn.calls[1]["temperature"] == fn.calls[0]["temperature"]

    def test_budget_reroll_carries_numeric_hint(self):
        fn = _ScriptedFn([_FAT_MARKUP, _GOOD_MARKUP])
        raw, parsed, meta = _run_script_pass(fn)
        assert meta["budget_rerolls"] == 1
        hint = fn.calls[1]["msgs"][-1]["content"]
        assert "30" in hint and "TIGHTEN" in hint

    def test_budget_exhaustion_fails_loud(self):
        fn = _ScriptedFn([_FAT_MARKUP, _FAT_MARKUP, _FAT_MARKUP])
        with pytest.raises(F2.Fable2ScriptError, match="WORD_BUDGET"):
            _run_script_pass(fn)

    def test_ladder_exhaustion_fails_loud_naming_the_pass(self):
        fn = _ScriptedFn([_BAD_SHAPE_MARKUP] * 4)
        with pytest.raises(F2.Fable2ScriptError, match="script"):
            _run_script_pass(fn)
        # one per rung (4 rungs since the 10th live smoke); temps never
        # rise (2B principle; the final rung repeats 0.30)
        assert len(fn.calls) == 4
        temps = [c["temperature"] for c in fn.calls]
        assert temps == sorted(temps, reverse=True)
        assert temps[-1] == 0.30


# ---------------------------------------------------------------------------
# 2. P8 triage evidence bar (lexicon-only kill policy)
# ---------------------------------------------------------------------------

def _parsed_good():
    from nodes._otr_fable2_markup import parse_fable2_markup
    parsed, defects = parse_fable2_markup(_GOOD_MARKUP, _CAST)
    assert parsed is not None, defects
    return parsed


def _finding(cls, detail="the judge says something is wrong here"):
    return F2.AuditFindings(findings=[{
        "finding_class": cls, "scene": 1, "speaker": "SELA",
        "detail": detail}])


class TestTriage:
    def test_unproven_weapons_flag_discarded_loudly(self):
        parsed = _parsed_good()
        view = F2._script_view(parsed, _treatment())
        confirmed, discarded, reported = F2._triage(
            _finding("weapons_smoking"), parsed, view, _CAST)
        assert not confirmed and len(discarded) == 1 and not reported

    def test_lexicon_corroborated_weapons_flag_confirmed(self):
        from nodes._otr_fable2_markup import parse_fable2_markup
        armed = _GOOD_MARKUP.replace(
            "Play it again, and this time keep the gain low.",
            "Put the revolver down and play the tape again for me.")
        parsed, defects = parse_fable2_markup(armed, _CAST)
        assert parsed is not None, defects
        view = F2._script_view(parsed, _treatment())
        confirmed, discarded, reported = F2._triage(
            _finding("weapons_smoking"), parsed, view, _CAST)
        assert len(confirmed) == 1
        assert "revolver" in confirmed[0][1]

    def test_news_framing_never_killed_by_the_real_news_read(self):
        # The closing news read IS real news by design -- only the
        # character-spoken drama is kill-scannable for news framing.
        parsed = _parsed_good()
        view = F2._script_view(parsed, _treatment())  # includes the read
        confirmed, discarded, reported = F2._triage(
            _finding("news_source_framing"), parsed, view, _CAST)
        assert not confirmed and len(discarded) == 1

    def test_python_verified_structural_flags_discarded(self):
        parsed = _parsed_good()
        view = F2._script_view(parsed, _treatment())
        for cls in ("speaker_not_in_cast", "verbatim_break",
                    "skeleton_break"):
            confirmed, discarded, reported = F2._triage(
                _finding(cls), parsed, view, _CAST)
            assert not confirmed and len(discarded) == 1, cls

    def test_taste_classes_report_only_never_fatal(self):
        parsed = _parsed_good()
        view = F2._script_view(parsed, _treatment())
        confirmed, discarded, reported = F2._triage(
            _finding("register_bleed"), parsed, view, _CAST)
        assert not confirmed and not discarded and len(reported) == 1


# ---------------------------------------------------------------------------
# 3. End-to-end spine (no model; scripted LLMs into a REAL tmp ledger)
# ---------------------------------------------------------------------------

_DOSSIER_JSON = json.dumps({
    "facts_to_keep": [
        "A survey mapped 1,200 new vents on Mount Etna this season.",
        "Doctor Rossi led the instrument team on the north slope.",
        "The vents released measurable heat before any tremor.",
    ],
    "allowed_numbers": ["1,200"],
    "named_entities": {"people": ["Doctor Rossi"],
                       "places": ["Mount Etna"],
                       "things": ["heat sensors"]},
    "dramatizable_vectors": [
        "a scientist who trusts the instruments over the committee",
        "a village that hears the mountain breathe at night",
        "the cost of raising an alarm one day too early",
    ],
})

_PAYLOAD = {
    "headline": "Survey maps 1,200 new vents on Mount Etna",
    "summary": "Doctor Rossi's team mapped 1,200 new vents using heat "
               "sensors.",
    "full_text": "Doctor Rossi's team mapped 1,200 new vents on Mount "
                 "Etna using heat sensors before any tremor.",
    "source": "MIT News", "date": "2026-07-01", "link": "",
    "seed_text": "Survey maps 1,200 new vents on Mount Etna",
}


def _e2e_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OTR_FABLE2_SEED", "42")
    deck = F2._load_frame_deck()
    dealt_cards, _stance = F2._deal(
        random.Random(42), deck, mode="one_pitch")

    def _pitch_json():
        return json.dumps({"pitches": [{
            "pitch_id": 1, "frame_card": dealt_cards[0]["name"],
            "logline": "A scientist must out-argue her own silence "
                       "before the mountain speaks first.",
            "hook": "The instruments heard it three days early.",
            "scifi_device": "heat sensors that map a volcano's breath",
            "cast_size": 2, "ending_shape": "quiet_loss"}]})

    def _casting_json():
        menu = F2._deal_voice_menu(2)
        female = next(e for e in menu.entries if e.gender == "female")
        male = next(e for e in menu.entries if e.gender == "male")
        desc = ("Mid-forties, wind-burned, one pencil behind the ear, "
                "taps the barometer twice before speaking.")
        return json.dumps({"cast": [
            {"name": "SELA", "role": "instrument scientist",
             "character_description": desc, "gender": "female",
             "age_band": "40s", "register": "clipped and front-loaded",
             "timbre": female.menu_id,
             "want": "to be believed before dawn",
             "pressure": "the committee reads her charts as noise"},
            {"name": "DARROW", "role": "village liaison",
             "character_description": desc.replace("her", "his"),
             "gender": "male", "age_band": "50s",
             "register": "slow warm circling",
             "timbre": male.menu_id,
             "want": "to keep the village calm",
             "pressure": "his cousin farms the north slope"},
        ]})

    technical = _ScriptedFn([
        _DOSSIER_JSON,                       # P0 dossier
        json.dumps({                         # P2c news read (read-split)
            "news_close_read": _TREATMENT_DICT["news_close_read"]}),
        _casting_json,                       # P6 casting
        json.dumps({"findings": []}),        # P8 audit
    ])
    creative = _ScriptedFn([
        _pitch_json,                         # P1 one-pitch
        json.dumps(_TREATMENT_DICT),         # P2b treatment
        _GOOD_MARKUP,                        # P3 script
    ])

    led = _PL.new_ledger(episode_id="fable2_e2e",
                         out_dir=str(tmp_path / "ep"))
    meta = led.data.setdefault("meta", {})
    resolved = {
        "target_words": 30, "num_characters": 2,
        "creative_writing_model": "stub/creative",
        "technical_model": "stub/technical",
    }
    parts = F2.run_scifi_fable2_episode(
        payload=_PAYLOAD, pack=_pack(), resolved=resolved, led=led,
        meta=meta, creative_fn=creative, technical_fn=technical,
        slot_scheduler=None,
        source_bank_row=ROUTING.get_bank("scifi_fable2"),
        story_rules=None, episode_root=tmp_path / "ep",
        episode_id="fable2_e2e")
    return parts, led, meta, creative, technical


class TestEndToEndSpine:
    def test_golden_chain_fills_ledger_and_returns_parts(self, tmp_path,
                                                         monkeypatch):
        parts, led, meta, creative, technical = _e2e_run(
            tmp_path, monkeypatch)
        # tail parts contract (r4/M3)
        assert parts.final_title_override == "The Long Count"
        assert parts.run_story_spine is False
        assert parts.refine_active is False
        assert parts.outline_view.premise == (
            _TREATMENT_DICT["dramatic_question"])
        assert parts.canon.title == "The Long Count"
        assert parts.canon.time_of_day == "night"
        assert parts.canon.sound_palette == []
        # ledger: five hierarchies + cast
        assert [r["char_id"] for r in led.data["cast"]] == [
            "c01", "c02", "c03"]
        assert led.data["scenes"] and led.data["shots"]
        assert led.data["beats"] and led.data["music"]
        # meta.fable2 contract (doc s7)
        f2 = meta["fable2"]
        assert f2["schema_version"] == "fable2_v1"
        assert f2["mode"] == "one_pitch_one_draft"
        assert f2["seed"] == 42
        assert f2["selection"]["chosen_pitch_id"] == 1
        assert f2["draft1_sha256"] == f2["final_sha256"]
        assert f2["critic"] is None
        assert f2["proof_map"]
        assert f2["audit"] == {
            "findings": [], "confirmed": [], "discarded": []}
        assert "_winning_draft_text" not in f2
        # receipts: 7 passes, each stamped with the mode (r2 anchor)
        receipts = f2["pass_receipts"]
        assert [r["pass_id"] for r in receipts] == [
            "dossier", "pitch_room", "treatment", "news_read", "script",
            "casting_voices", "assemble", "ledger_audit"]
        assert all(r["mode"] == "one_pitch_one_draft" for r in receipts)
        llm_receipts = [r for r in receipts if r["pass_id"] != "assemble"]
        assert all(r["attempts"] >= 1 for r in llm_receipts)
        # the read-split stamped the P2c read onto the treatment
        assert f2["treatment"]["news_close_read"] == (
            _TREATMENT_DICT["news_close_read"])
        # writer-shared meta stamps
        assert meta["news"] is None
        assert meta["num_characters_locked"] == 2
        assert meta["cast_status"] == "locked"
        # credits receipt (25th live smoke): the fable2 seed is the
        # episode's seed receipt
        assert meta["episode_seed"] == 42
        # exactly 4 technical + 3 creative calls on the happy path
        assert len(technical.calls) == 4
        assert len(creative.calls) == 3

    def test_seed_reproduces_the_deal_across_runs(self, tmp_path,
                                                  monkeypatch):
        parts_a, led_a, meta_a, _c, _t = _e2e_run(
            tmp_path / "a", monkeypatch)
        parts_b, led_b, meta_b, _c2, _t2 = _e2e_run(
            tmp_path / "b", monkeypatch)
        assert meta_a["fable2"]["cards_dealt"] == \
            meta_b["fable2"]["cards_dealt"]
        assert meta_a["fable2"]["stance"] == meta_b["fable2"]["stance"]
        # voice draw rides the same seeded rng
        assert [r["voice_preset"] for r in led_a.data["cast"]] == \
            [r["voice_preset"] for r in led_b.data["cast"]]

    def test_runner_reasserts_the_word_gate(self, tmp_path):
        led = _PL.new_ledger(episode_id="fable2_gate",
                             out_dir=str(tmp_path / "ep"))
        with pytest.raises(F2.Fable2ScriptError, match="S2"):
            F2.run_scifi_fable2_episode(
                payload=_PAYLOAD, pack=_pack(),
                resolved={"target_words": 350, "num_characters": 2,
                          "creative_writing_model": "x",
                          "technical_model": "y"},
                led=led, meta=led.data.setdefault("meta", {}),
                creative_fn=None, technical_fn=None, slot_scheduler=None,
                source_bank_row=ROUTING.get_bank("scifi_fable2"),
                story_rules=None, episode_root=tmp_path,
                episode_id="fable2_gate")


# ---------------------------------------------------------------------------
# 4. Writer-level dispatch + entry gates (r3/S1 + r3/M4)
# ---------------------------------------------------------------------------

class TestWriterDispatch:
    def test_lane_map_hit_and_legacy_misses(self):
        from nodes import OTR_LedgerScriptWriter as W
        assert W._resolve_lane_runner("fable2_multipass") is not None
        assert W._resolve_lane_runner("legacy_many_pass") is None
        assert W._resolve_lane_runner("original_multi_pass") is None

    def test_unmapped_runner_raises_loud(self):
        # r3/S1: routing's `executable` stays metadata-only; a pipeline
        # with no registered lane runner and no inline branch is a
        # wiring bug that fails loud IN THE WRITER.
        from nodes import OTR_LedgerScriptWriter as W
        with pytest.raises(RuntimeError, match="no registered lane runner"):
            W._resolve_lane_runner("simple_4_prompt_experimental")

    def test_run_entry_gate_rejects_full_mode_words_before_any_work(self):
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
        with pytest.raises(F2.Fable2ScriptError, match="S2"):
            OTR_LedgerScriptWriter().run(
                source_bank="scifi_fable2", target_words=350)

    def test_run_entry_gate_rejects_over_ceiling(self):
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
        with pytest.raises(F2.Fable2ScriptError, match="ceiling"):
            OTR_LedgerScriptWriter().run(
                source_bank="scifi_fable2", target_words=1000)

    def test_run_entry_gate_rejects_refine_loop(self):
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
        with pytest.raises(RuntimeError, match="refine"):
            OTR_LedgerScriptWriter().run(
                source_bank="scifi_fable2", target_words=30,
                refine_target_grade="B")


# ---------------------------------------------------------------------------
# 5. Pure-import pin (r4/M3): the runner imports WITHOUT the writer
# ---------------------------------------------------------------------------

def test_runner_pure_import_never_pulls_the_writer():
    code = (
        "import sys; sys.path.insert(0, r'" + str(_REPO) + "');"
        "import nodes._otr_scifi_fable2;"
        "bad=[m for m in sys.modules if 'OTR_LedgerScriptWriter' in m];"
        "assert not bad, f'pure import violated: {bad}';"
        "print('PURE_IMPORT_OK')"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(_REPO), timeout=120)
    assert out.returncode == 0, out.stderr
    assert "PURE_IMPORT_OK" in out.stdout
