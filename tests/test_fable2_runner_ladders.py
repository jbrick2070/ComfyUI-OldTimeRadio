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
import math
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
])  # 31 character words -> inside the shared 28-34 band for target 30

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
    return ROUTING.resolve_story_pack("scifi_news_pro")


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
        trace = meta["attempt_trace"]
        assert len(trace) == 1 and type(trace[0]) is F2.PassAttemptTrace
        assert trace[0].outcome == "accepted" and trace[0].selected is True
        assert trace[0].temperature == fn.calls[0]["temperature"]
        assert trace[0].requested_max_new_tokens == \
            fn.calls[0]["max_new_tokens"]
        assert trace[0].character_words == parsed.character_word_count
        assert trace[0].scene_character_words == \
            F2._scene_character_word_counts(parsed)

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
        trace = meta["attempt_trace"]
        assert [row.outcome for row in trace] == ["truncated", "accepted"]
        assert [row.requested_max_new_tokens for row in trace] == [
            t0, int(t0 * 1.25)]
        assert trace[0].truncation is True and trace[0].selected is False
        assert trace[1].truncation is False and trace[1].selected is True
        assert meta["actual_max_new_tokens"] == int(t0 * 1.25)

    def test_budget_reroll_carries_numeric_hint(self):
        fn = _ScriptedFn([_FAT_MARKUP, _GOOD_MARKUP])
        raw, parsed, meta = _run_script_pass(fn)
        assert meta["budget_rerolls"] == 1
        hint = fn.calls[1]["msgs"][-1]["content"]
        assert "30" in hint and "TIGHTEN" in hint
        trace = meta["attempt_trace"]
        assert [row.outcome for row in trace] == [
            "budget_rejected", "accepted"]
        assert trace[0].character_words is not None
        assert trace[0].scene_character_words

    def test_budget_exhaustion_defers_complete_candidate_to_final_fit(self):
        # Whole-play retries are bounded. The complete parse survives as an
        # intermediate candidate, but its miss is explicitly handed to the
        # mandatory row-local fitter before casting/assembly.
        fn = _ScriptedFn([_FAT_MARKUP, _FAT_MARKUP, _FAT_MARKUP])
        raw, parsed, meta = _run_script_pass(fn)
        assert parsed is not None
        assert any(
            "WORD_BUDGET" in defect
            for defect in meta["deferred_budget_defects"]
        )

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
# 2. P8 safety scan (deterministic; the LLM ledger audit is gone)
# ---------------------------------------------------------------------------

def _parsed_good():
    from nodes._otr_fable2_markup import parse_fable2_markup
    parsed, defects = parse_fable2_markup(_GOOD_MARKUP, _CAST)
    assert parsed is not None, defects
    return parsed


class TestSafetyScan:
    def test_clean_drama_passes(self):
        F2._assert_no_weapons_or_smoking(_parsed_good())

    def test_a_spoken_weapon_is_fatal(self):
        from nodes._otr_fable2_markup import parse_fable2_markup
        armed = _GOOD_MARKUP.replace(
            "Play it again, and this time keep the gain low.",
            "Put the revolver down and play the tape again for me.")
        parsed, defects = parse_fable2_markup(armed, _CAST)
        assert parsed is not None, defects
        with pytest.raises(F2.Fable2AuditError) as exc:
            F2._assert_no_weapons_or_smoking(parsed)
        assert "revolver" in str(exc.value)

    def test_the_scan_is_word_boundary_not_substring(self):
        """'gun' must never fire on 'begun' -- the old scan did exactly that."""
        from nodes._otr_original_radio import lexicon_hits, WEAPON_SMOKING_EVIDENCE
        assert lexicon_hits(
            "The broadcast had begun and the pipeline was clear.",
            WEAPON_SMOKING_EVIDENCE,
        ) == []
        assert lexicon_hits(
            "She set the gun on the table.", WEAPON_SMOKING_EVIDENCE,
        ) == ["gun"]

    def test_a_judges_own_prose_can_no_longer_corroborate_itself(self):
        """The old bar scanned `finding.detail + script`, so a judge that wrote
        'the scene mentions a gun' proved its own flag. Only the script counts.
        """
        from nodes._otr_original_radio import lexicon_hits, WEAPON_SMOKING_EVIDENCE
        clean_script = "\n".join(
            ln.text for scene in _parsed_good().scenes for ln in scene.lines
        )
        assert lexicon_hits(clean_script, WEAPON_SMOKING_EVIDENCE) == []


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
        random.Random(42), deck, mode=F2._COMPACT_MODE)

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
        source_bank_row=ROUTING.get_bank("scifi_news_pro"),
        story_rules=F2.resolve_story_rules("scifi_news_pro"),
        episode_root=tmp_path / "ep",
        episode_id="fable2_e2e")
    return parts, led, led.data["meta"], creative, technical


def _budgeted_markup(target_words: int, *, revised: bool = False) -> str:
    """Exact-vector whole play used by compact/full boundary runs."""
    envelope = F2._build_envelope(target_words)
    safe_words = (
        "needle", "rises", "quietly", "beneath", "glass", "village",
        "waits", "signal", "fades", "before", "dawn", "watcher",
    )
    rows = [
        "TITLE: The Long Count",
        "MUSIC: slow theremin swell",
        "ANNOUNCER: Tonight the instrument ledger answers before dawn.",
    ]
    for scene_no, scene_target in enumerate(
            envelope.scene_word_targets, start=1):
        # Keep the exact scene vector while giving the real spoken-hygiene
        # boundary realistic, alphabetic, one-breath rows.  The former
        # alphanumeric filler (``s1a0`` etc.) counted as two spoken words and
        # accidentally turned these routing fixtures into repair fixtures.
        row_count = max(2, math.ceil(scene_target / 20))
        quotient, remainder = divmod(scene_target, row_count)
        row_targets = [
            quotient + (1 if index < remainder else 0)
            for index in range(row_count)
        ]
        rows.append(f"SCENE {scene_no}: instrument room {scene_no}")
        for row_index, row_target in enumerate(row_targets):
            speaker = "SELA" if row_index % 2 == 0 else "DARROW"
            opening = (
                ["the", "record" if revised else "void"]
                if scene_no == 1 and row_index == 0 else []
            )
            words = opening + [
                safe_words[(scene_no + row_index + index) % len(safe_words)]
                for index in range(row_target - len(opening))
            ]
            rows.append(f"{speaker}: " + " ".join(words))
        if scene_no < envelope.scene_count:
            rows.append(f"MUSIC: brief bridge after scene {scene_no}")
    rows.extend([
        "ANNOUNCER: The ledger closes and the village keeps the cost.",
        "CODA: Beyond the measured heat, a real survey waits:",
        "MUSIC: closing theme with warm brass",
        "END.",
    ])
    return "\n".join(rows)


def _casting_json_for_two():
    menu = F2._deal_voice_menu(2)
    female = next(entry for entry in menu.entries
                  if entry.gender == "female")
    male = next(entry for entry in menu.entries if entry.gender == "male")
    description = (
        "Mid-forties, wind-burned, one pencil behind the ear, taps the "
        "barometer twice before speaking."
    )
    return json.dumps({"cast": [
        {"name": "SELA", "role": "instrument scientist",
         "character_description": description, "gender": "female",
         "age_band": "40s", "register": "clipped and front-loaded",
         "timbre": female.menu_id,
         "want": "to be believed before dawn",
         "pressure": "the committee reads her charts as noise"},
        {"name": "DARROW", "role": "village liaison",
         "character_description": description.replace("her", "his"),
         "gender": "male", "age_band": "50s",
         "register": "slow warm circling", "timbre": male.menu_id,
         "want": "to keep the village calm",
         "pressure": "his cousin farms the north slope"},
    ]})


def _boundary_e2e_run(tmp_path, monkeypatch, *, target_words: int,
                      truncate_p3: bool = False, capture: bool = False,
                      revision_improves: bool = True):
    monkeypatch.setenv("OTR_FABLE2_SEED", "42")
    mode = F2._resolve_mode(target_words)
    deck = F2._load_frame_deck()
    cards, _stance = F2._deal(random.Random(42), deck, mode=mode)

    def _pitch_json():
        pitch_rows = [
            {
                "pitch_id": 1,
                "frame_card": cards[0]["name"],
                "logline": (
                    "A sensor keeper risks her station to prove a buried "
                    "heat signal before dawn."),
                "hook": "The instruments heard the mountain answer early.",
                "scifi_device": "heat sensors that map a volcano breathing",
                "cast_size": 2,
                "ending_shape": "quiet_loss",
            },
        ]
        if mode == F2._FULL_MODE:
            pitch_rows.extend([
                {
                    "pitch_id": 2,
                    "frame_card": cards[1]["name"],
                    "logline": (
                        "A village liaison follows a midnight vent map and "
                        "must choose whom the warning saves."),
                    "hook": "Every warm mark points beneath a different home.",
                    "scifi_device": "a thermal chart that predicts new vents",
                    "cast_size": 1,
                    "ending_shape": "paid_victory",
                },
                {
                    "pitch_id": 3,
                    "frame_card": cards[2]["name"],
                    "logline": (
                        "Two committee rivals trade ownership of a breathing "
                        "mountain record while the valley listens."),
                    "hook": "The disputed tape changes only when nobody speaks.",
                    "scifi_device": "an acoustic ledger of subterranean heat",
                    "cast_size": 2,
                    "ending_shape": "ironic_turn",
                },
            ])
        return json.dumps({"pitches": pitch_rows})

    technical_responses = [_DOSSIER_JSON]
    if mode == F2._FULL_MODE:
        technical_responses.append(json.dumps({
            "chosen_pitch_id": 1,
            "selection_rationale": (
                "The sensor keeper offers audible conflict, concrete "
                "science, and a short ending with a payable cost."),
        }))
    technical_responses.append(json.dumps({
        "news_close_read": _TREATMENT_DICT["news_close_read"]}))
    if mode == F2._FULL_MODE:
        technical_responses.append(json.dumps(
            {
                "verdict": "revise",
                "notes": [{
                    "scene": 1,
                    "speaker": "SELA",
                    "frame_scope": None,
                    "problem": "on_the_nose",
                    "note": (
                        "Replace the abstract void phrase with one concrete "
                        "record while preserving the line."),
                }],
            }
            if revision_improves else
            {"verdict": "ship", "notes": []}
        ))
    # Casting is the LAST technical LLM call: the P8 ledger audit is gone.
    technical_responses.append(_casting_json_for_two)

    draft1 = _budgeted_markup(target_words)
    creative_responses = [
        _pitch_json,
        json.dumps(_TREATMENT_DICT),
    ]
    if truncate_p3:
        creative_responses.append(draft1.rsplit("\nEND.", 1)[0])
    creative_responses.append(draft1)
    if mode == F2._FULL_MODE:
        creative_responses.append(
            _budgeted_markup(target_words, revised=revision_improves))

    technical = _ScriptedFn(technical_responses)
    creative = _ScriptedFn(creative_responses)
    captured = {"assemble": [], "casting": [], "audit": []}
    if capture:
        real_choose = F2.choose_final_draft
        real_assemble = F2._assemble
        real_casting = F2._pass_casting
        real_scan = F2._assert_no_weapons_or_smoking

        def _choose_spy(draft_a, draft_b, contract):
            selection = real_choose(draft_a, draft_b, contract)
            captured["selection"] = selection
            return selection

        def _assemble_spy(led, final_draft, *args, **kwargs):
            captured["assemble"].append(final_draft)
            return real_assemble(led, final_draft, *args, **kwargs)

        def _casting_spy(slot_fn, pack, final_draft, *args, **kwargs):
            captured["casting"].append(final_draft)
            return real_casting(
                slot_fn, pack, final_draft, *args, **kwargs)

        def _scan_spy(parsed, treatment=None):
            captured["audit"].append(parsed)
            return real_scan(parsed, treatment)

        monkeypatch.setattr(F2, "choose_final_draft", _choose_spy)
        monkeypatch.setattr(F2, "_assemble", _assemble_spy)
        monkeypatch.setattr(F2, "_pass_casting", _casting_spy)
        monkeypatch.setattr(F2, "_assert_no_weapons_or_smoking", _scan_spy)

    episode_id = f"fable2_boundary_{target_words}"
    led = _PL.new_ledger(
        episode_id=episode_id, out_dir=str(tmp_path / "ep"))
    initial_meta = led.data.setdefault("meta", {})
    parts = F2.run_scifi_fable2_episode(
        payload=_PAYLOAD,
        pack=_pack(),
        resolved={
            "target_words": target_words,
            "num_characters": 2,
            "creative_writing_model": "stub/creative",
            "technical_model": "stub/technical",
        },
        led=led,
        meta=initial_meta,
        creative_fn=creative,
        technical_fn=technical,
        slot_scheduler=None,
        source_bank_row=ROUTING.get_bank("scifi_news_pro"),
        story_rules=F2.resolve_story_rules("scifi_news_pro"),
        episode_root=tmp_path / "ep",
        episode_id=episode_id,
    )
    return parts, led, led.data["meta"], creative, technical, captured


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
        # The LLM ledger audit is gone; nothing stamps an audit verdict.
        assert "audit" not in f2
        assert "_winning_draft_text" not in f2
        # receipts: every logical pass is explicit, including compact skips.
        receipts = f2["pass_receipts"]
        assert [r["pass_id"] for r in receipts] == [
            "dossier", "pitch_room", "pitch_select", "treatment",
            "news_read", "script", "critic", "revision",
            "casting_voices", "assemble", "safety_scan"]
        # Every runtime receipt names a pass the pipeline actually declares.
        declared = {
            p.pass_id for p in ROUTING.get_pipeline("scifi_news_pro_multipass").passes
        }
        assert {r["pass_id"] for r in receipts} <= declared
        assert all(r["mode"] == "one_pitch_one_draft" for r in receipts)
        skipped = [r for r in receipts if r["status"] == "skipped"]
        assert [r["pass_id"] for r in skipped] == [
            "pitch_select", "critic", "revision"]
        assert all(r["attempts"] == 0 and r["reason"] for r in skipped)
        # assemble and safety_scan are pure Python: no model, no attempts.
        python_passes = {"assemble", "safety_scan"}
        llm_receipts = [
            r for r in receipts if r["status"] == "completed"
            and r["pass_id"] not in python_passes
        ]
        assert all(r["attempts"] >= 1 for r in llm_receipts)
        assert all(r["attempts"] == 0 for r in receipts
                   if r["pass_id"] in python_passes)
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
        # exactly 3 technical + 3 creative calls on the happy path
        # (the P8 ledger audit is a Python scan now, not an LLM call)
        assert len(technical.calls) == 3
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

    def test_runner_rejects_901_before_any_model_call(self, tmp_path):
        led = _PL.new_ledger(episode_id="fable2_gate",
                             out_dir=str(tmp_path / "ep"))
        calls = []

        def _must_not_run(*_args, **_kwargs):
            calls.append(1)
            raise AssertionError("model call crossed the 901-word gate")

        with pytest.raises(F2.Fable2ScriptError, match="ceiling"):
            F2.run_scifi_fable2_episode(
                payload=_PAYLOAD, pack=_pack(),
                resolved={"target_words": 901, "num_characters": 2,
                          "creative_writing_model": "x",
                          "technical_model": "y"},
                led=led, meta=led.data.setdefault("meta", {}),
                creative_fn=_must_not_run, technical_fn=_must_not_run,
                slot_scheduler=None,
                source_bank_row=ROUTING.get_bank("scifi_news_pro"),
                story_rules=F2.resolve_story_rules("scifi_news_pro"),
                episode_root=tmp_path,
                episode_id="fable2_gate")
        assert calls == []


class TestS2ModeExecution:
    def test_119_words_executes_compact_mode_with_real_p3(self, tmp_path,
                                                           monkeypatch):
        _parts, led, meta, creative, technical, _captured = \
            _boundary_e2e_run(
                tmp_path, monkeypatch, target_words=119)
        f2 = meta["fable2"]
        assert f2["mode"] == F2._COMPACT_MODE
        skipped = [
            row["pass_id"] for row in f2["pass_receipts"]
            if row["status"] == "skipped"
        ]
        assert skipped == ["pitch_select", "critic", "revision"]
        assert f2["final_draft"]["p3_attempts"]
        assert f2["final_draft"]["p5_attempts"] == []
        assert len(creative.calls) == 3
        assert len(technical.calls) == 3
        assert led.data["meta"]["fable2"] is f2

    @pytest.mark.parametrize(
        ("target_words", "truncate_p3", "creative_call_count"),
        [(120, True, 5), (420, False, 4)],
    )
    def test_120_and_420_execute_real_full_loop_with_atomic_winner(
            self, tmp_path, monkeypatch, target_words, truncate_p3,
            creative_call_count):
        parts, led, meta, creative, technical, captured = \
            _boundary_e2e_run(
                tmp_path, monkeypatch, target_words=target_words,
                truncate_p3=truncate_p3, capture=True)
        f2 = meta["fable2"]
        assert f2["mode"] == F2._FULL_MODE
        assert [row["pitch_id"] for row in f2["pitches"]] == [1, 2, 3]
        assert f2["selection"]["chosen_pitch_id"] == 1
        assert f2["selection"]["selection_rationale"]
        assert f2["selected_draft"] == "draft2"
        assert "draft2 selected" in f2["better_draft_choice"]

        receipts = {
            row["pass_id"]: row for row in f2["pass_receipts"]
        }
        for pass_id, model_id in (
                ("pitch_select", "stub/technical"),
                ("critic", "stub/technical"),
                ("revision", "stub/creative")):
            row = receipts[pass_id]
            assert row["status"] == "completed"
            assert row["model_id"] == model_id
            assert row["attempts"] >= 1
            assert "reason" not in row
        assert not [
            row for row in f2["pass_receipts"]
            if row["status"] == "skipped"
        ]

        winner = captured["selection"].winner
        assert len(captured["assemble"]) == 1
        assert len(captured["casting"]) == 1
        assert len(captured["audit"]) == 2
        assert captured["assemble"][0] is winner
        assert captured["casting"][0] is winner
        # Safety scans both immediately before assembly and again at P8; both
        # inspect the exact winner surface, never a re-derived draft.
        assert all(parsed is winner.parsed for parsed in captured["audit"])
        assert len(winner.p3_attempts) >= 1
        assert len(winner.p5_attempts) >= 1
        assert all(type(row) is F2.PassAttemptTrace
                   for row in winner.p3_attempts + winner.p5_attempts)
        assert winner.p5_attempts[-1].temperature == F2._TEMP["revision"]
        assert winner.p5_attempts[-1].requested_max_new_tokens == \
            creative.calls[-1]["max_new_tokens"]

        if truncate_p3:
            initial = F2._script_token_budget(target_words)
            assert [row.outcome for row in winner.p3_attempts] == [
                "truncated", "accepted"]
            assert receipts["script"]["attempts"] == 2
            assert receipts["script"]["max_new_tokens"] == \
                int(initial * 1.25)
        else:
            assert [row.outcome for row in winner.p3_attempts] == [
                "accepted"]
        assert len(creative.calls) == creative_call_count
        assert len(technical.calls) == 5

        final_receipt = f2["final_draft"]
        assert final_receipt == F2._final_draft_ledger_payload(winner)
        assert final_receipt["p3_attempts"]
        assert final_receipt["p5_attempts"]
        assert final_receipt["raw_sha256"] == winner.hashes.raw_sha256
        assert final_receipt["normalized_sha256"] == \
            F2._sha256(winner.normalized_source)
        assert final_receipt["proof_map_sha256"] == \
            F2._canonical_sha256(f2["proof_map"])
        assert final_receipt["artifact_sha256"] == F2._canonical_sha256(
            F2._final_draft_seal_payload(
                raw_sha256=winner.hashes.raw_sha256,
                normalized_sha256=winner.hashes.normalized_sha256,
                parsed_sha256=winner.hashes.parsed_sha256,
                proof_map_sha256=winner.hashes.proof_map_sha256,
                score=winner.score,
                scoring_context_sha256=winner.scoring_context_sha256,
                p3_attempts=winner.p3_attempts,
                p5_attempts=winner.p5_attempts,
            ))
        spoken = " ".join(
            row["text"] for row in led.data["lines"] if row["text"])
        assert "the record" in spoken
        assert "the void" not in spoken
        assert parts.final_title_override == winner.parsed.title
        assert parts.fable2_meta is f2

        on_disk = json.loads(
            Path(led.path).read_text(encoding="utf-8"))
        assert on_disk["meta"]["fable2"] == f2
        assert on_disk["meta"]["fable2"]["final_draft"] == final_receipt
        assert "_winning_draft_text" not in json.dumps(on_disk)

    def test_full_mode_tie_keeps_draft1_without_dropping_p5_trace(
            self, tmp_path, monkeypatch):
        _parts, led, meta, _creative, _technical, captured = \
            _boundary_e2e_run(
                tmp_path, monkeypatch, target_words=120, capture=True,
                revision_improves=False)
        f2 = meta["fable2"]
        winner = captured["selection"].winner
        assert f2["selected_draft"] == "draft1"
        assert "score tie" in f2["better_draft_choice"]
        assert winner.p5_attempts
        assert f2["final_draft"]["p5_attempts"]
        assert len(captured["assemble"]) == 1
        assert captured["assemble"][0] is winner
        spoken = " ".join(
            row["text"] for row in led.data["lines"] if row["text"])
        assert "the void" in spoken

    def test_900_words_is_accepted_by_the_real_full_runner(self, tmp_path,
                                                            monkeypatch):
        _parts, led, meta, creative, technical, _captured = \
            _boundary_e2e_run(
                tmp_path, monkeypatch, target_words=900)
        f2 = meta["fable2"]
        assert f2["mode"] == F2._FULL_MODE
        assert len(led.data["scenes"]) == 8
        assert f2["final_draft"]["p3_attempts"]
        assert f2["final_draft"]["p5_attempts"]
        assert len(creative.calls) == 4
        assert len(technical.calls) == 5


# ---------------------------------------------------------------------------
# 4. Writer-level dispatch + entry gates (r3/S1 + r3/M4)
# ---------------------------------------------------------------------------

class TestWriterDispatch:
    def test_lane_map_hit_and_legacy_misses(self):
        from nodes import OTR_LedgerScriptWriter as W
        assert W._resolve_lane_runner("scifi_news_pro_multipass") is not None
        assert W._resolve_lane_runner("legacy_many_pass") is None
        assert W._resolve_lane_runner("original_multi_pass") is None

    def test_unmapped_runner_raises_loud(self):
        # r3/S1: routing's `executable` stays metadata-only; a pipeline
        # with no registered lane runner and no inline branch is a
        # wiring bug that fails loud IN THE WRITER.
        from nodes import OTR_LedgerScriptWriter as W
        with pytest.raises(RuntimeError, match="no registered lane runner"):
            W._resolve_lane_runner("simple_4_prompt_experimental")

    def test_run_entry_gate_rejects_over_ceiling(self):
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
        with pytest.raises(F2.Fable2ScriptError, match="ceiling"):
            OTR_LedgerScriptWriter().run(
                source_bank="scifi_news_pro", target_words=901)

    def test_run_entry_gate_rejects_refine_loop(self):
        from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter
        with pytest.raises(RuntimeError, match="refine"):
            OTR_LedgerScriptWriter().run(
                source_bank="scifi_news_pro", target_words=30,
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


# ---------------------------------------------------------------------------
# 3. Per-scene budget -- refined-C (kibitz 2026-07-14)
#    Advisory +/-30% (prompt vector + _draft_score steer) vs FATAL gross +/-50%.
#    A live 720 leg died on scene 1 = 137 vs the +/-30% band [73,133] (a 4-word
#    overage). refined-C renders it while still failing gross imbalance. The
#    gate is `_script_budget_defects`, the SINGLE per-scene gate shared by both
#    pass_id="script" and "revision" (both route through _run_markup_ladder).
# ---------------------------------------------------------------------------

def _markup_scene_counts(counts) -> str:
    """Whole play with EXPLICIT per-scene character word counts. ANNOUNCER lines
    are metered separately, so they never count toward scene character words."""
    rows = [
        "TITLE: The Long Count",
        "MUSIC: slow theremin swell",
        "ANNOUNCER: Tonight the instrument ledger answers before dawn.",
    ]
    for i, c in enumerate(counts, start=1):
        sela = c // 2 + c % 2
        darrow = c - sela
        rows.extend([
            f"SCENE {i}: instrument room {i}",
            "SELA: " + " ".join(f"s{i}a{j}" for j in range(sela)),
            "DARROW: " + " ".join(f"s{i}d{j}" for j in range(darrow)),
        ])
        if i < len(counts):
            rows.append(f"MUSIC: brief bridge after scene {i}")
    rows.extend([
        "ANNOUNCER: The ledger closes and the village keeps the cost.",
        "CODA: Beyond the measured heat, a real survey waits:",
        "MUSIC: closing theme with warm brass",
        "END.",
    ])
    return "\n".join(rows)


def _parse_counts(counts):
    parsed, defects = F2.parse_fable2_markup(_markup_scene_counts(counts), _CAST)
    assert parsed is not None, defects
    assert F2._scene_character_word_counts(parsed) == tuple(counts)
    return parsed


class TestPerSceneGrossBudget:
    # 720 -> 7 scenes -> targets (103*6, 102). advisory[103]=[73,133];
    # gross[103]=[52,154]. Shared _word_band(720) total = [649, 800].
    def _env(self):
        return F2._build_envelope(720)

    def test_advisory_band_is_still_plus_minus_30(self):
        assert self._env().scene_word_bands[0] == (73, 133)

    def test_gross_band_is_plus_minus_50(self):
        assert self._env().scene_word_gross_bands[0] == (52, 154)

    def test_the_live_137_scene_now_renders(self):
        # The exact live failure: scene 1 = 137, the rest on target. Total 754 is
        # inside the total band; scene 1 is outside +/-30% but inside gross ->
        # NO fatal defect. Before refined-C this raised SCENE_WORD_BUDGET.
        env = self._env()
        parsed = _parse_counts((137, 103, 103, 103, 103, 103, 102))
        assert F2._script_budget_defects(parsed, env) == ()

    def test_a_grossly_long_scene_still_fails(self):
        env = self._env()
        parsed = _parse_counts((160, 103, 103, 103, 103, 103, 102))  # 160 > 154
        defects = F2._script_budget_defects(parsed, env)
        assert any("SCENE_WORD_GROSS" in d and "scene 1" in d for d in defects)

    def test_a_grossly_short_scene_still_fails(self):
        env = self._env()
        parsed = _parse_counts((40, 122, 122, 122, 122, 122, 122))   # 40 < 52
        defects = F2._script_budget_defects(parsed, env)
        assert any("SCENE_WORD_GROSS" in d and "scene 1" in d for d in defects)

    def test_draft_score_still_penalizes_an_ordinary_30pct_miss(self):
        # refined-C keeps +/-30% as the SOFT steer: a 137 scene (advisory miss,
        # gross OK) must still score strictly worse than an on-target draft, so
        # the ladder prefers balanced pacing even though it no longer FAILS it.
        env = self._env()
        rules = F2.resolve_story_rules("scifi_news_pro")
        on_target = _parse_counts((103, 103, 103, 103, 103, 103, 102))
        one_miss = _parse_counts((137, 103, 103, 103, 103, 103, 102))
        s_ok = F2._draft_score(on_target, env, rules)
        s_miss = F2._draft_score(one_miss, env, rules)
        assert s_miss[1] > s_ok[1]        # +1 advisory violation
        assert s_miss > s_ok              # strictly worse -> ladder de-prefers it
