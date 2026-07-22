"""scifi_fable2 S1b -- P7 assembly golden chain (architecture doc s7/s13).

Asserts fable2's OWN ledger contract. The committed
tests/fixtures/fable2/legacy_reference_ledger.json (captured from the
S1a science live smoke, scrubbed) serves ONLY as the row-role-ordering +
tail-output-contract REFERENCE -- never a byte-match target (r4/CUT2).
The golden happy-path fixture (golden_s1b_assembly.json) pins fable2's
own deterministic assembly output.

Covers: five-hierarchy emission (r1/S2), exact music sentinel rows +
assembler cue ids (r2/M5 + r3/M1), per-constituent verbatim proof +
proof_map (r2/M4), same-speaker run merging, boundary stamps, role-set
assertion vs the legacy reference, compose_flags-absent freeze
assertion, the no-double-append pin (doc s14 item 7), incremental
saves, and the no-model contract test (assemble_script_text_from_ledger
+ set_music -- catches cue-id drift).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_scifi_fable2 as F2  # noqa: E402
from nodes import production_ledger as _PL  # noqa: E402
from nodes._otr_fable2_markup import parse_fable2_markup  # noqa: E402

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fable2"
_LEGACY = _FIXTURES / "legacy_reference_ledger.json"
_GOLDEN = _FIXTURES / "golden_s1b_assembly.json"

# Sequencer-legal role set (scene_sequencer role dispatch; NO FALLBACKS).
_SEQUENCER_LEGAL_ROLES = frozenset({
    "character", "announcer", "music_open", "music_close", "music_inter",
})


@pytest.fixture(autouse=True)
def _preserve_current_ledger():
    """new_ledger() sets the process-wide current-ledger singleton; never
    leak a tmp ledger to later tests (S1a root-cause lesson)."""
    saved = _PL._CURRENT
    yield
    _PL._CURRENT = saved


# ---------------------------------------------------------------------------
# Deterministic golden-chain inputs (no LLM anywhere)
# ---------------------------------------------------------------------------

_MARKUP = "\n".join([
    "TITLE: The Long Count",
    "MUSIC: slow theremin swell, distant telegraph clicks",
    "ANNOUNCER: Tonight, a story of one antenna and the woman who "
    "refused to choose.",
    "SCENE 1: A cliff-top listening station, an hour before dawn",
    "VERA: Play it again, and this time keep the gain low.",
    "VERA: The mountain does not repeat itself for committees.",
    "DOKU: The tape is the tape, Vera. It will not change for us.",
    "MUSIC: single sustained violin note, rising",
    "SCENE 2: The station generator room, minutes later",
    "DOKU: You pulled the log pages. I checked the binding twice.",
    "VERA: I pulled one page. The one with my name on it.",
    "ANNOUNCER: Vera got her answer, though the antenna kept the "
    "better half.",
    "CODA: Beyond tonight's cliff-top signal, a real transmission waits:",
    "MUSIC: closing theme, warm brass, resolving",
    "END.",
])

_CAST_NAMES = ["VERA", "DOKU"]

_NEWS_READ = (
    "Tonight's story grew from a real survey: instruments on Mount Etna "
    "mapped 1,200 new vents this season, work led by Doctor Rossi.")

_CAST_ROWS = [
    {"char_id": "c01", "name": "ANNOUNCER", "gender": "female",
     "tts_model": "kokoro", "voice_preset": "bf_lily",
     "voice_params": None,
     "character_description": "Period radio announcer."},
    {"char_id": "c02", "name": "VERA", "gender": "female",
     "tts_model": "bark", "voice_preset": "v2/en_speaker_9",
     "voice_params": None,
     "character_description": (
         "Mid-forties, wind-burned, one pencil behind the ear, taps the "
         "barometer twice before speaking.")},
    {"char_id": "c03", "name": "DOKU", "gender": "male",
     "tts_model": "bark", "voice_preset": "v2/en_speaker_6",
     "voice_params": None,
     "character_description": (
         "Broad-shouldered liaison in a mended coat, keeps a pocket "
         "ledger, folds every argument into a story.")},
]


def _treatment() -> F2.Treatment:
    return F2.Treatment.model_validate({
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
            "cost_paid": "her seat on the committee, surrendered in "
                         "writing",
        },
        "news_thread": (
            "new vents on Mount Etna released measurable heat before "
            "any tremor"),
        "news_close_read": _NEWS_READ,
    })


_PAYLOAD = {
    "headline": "Survey maps 1,200 new vents on Mount Etna",
    "summary": "Instruments mapped 1,200 new vents.",
    "full_text": "Doctor Rossi's team mapped 1,200 new vents on Mount "
                 "Etna using heat sensors before any tremor.",
    "source": "MIT News", "date": "2026-07-01", "link": "",
    "seed_text": "Survey maps 1,200 new vents on Mount Etna",
}


def _sealed_draft(markup: str, *, target_words: int = 50,
                  treatment: "F2.Treatment | None" = None):
    treatment = treatment or _treatment()
    parsed, defects = parse_fable2_markup(markup, _CAST_NAMES)
    assert parsed is not None, defects
    trace = F2.PassAttemptTrace(
        attempt=1,
        temperature=F2._TEMP["script"],
        outcome="accepted",
        selected=True,
        character_words=parsed.character_word_count,
        scene_character_words=F2._scene_character_word_counts(parsed),
    )
    draft = F2.build_final_draft(
        markup,
        parsed,
        treatment,
        p3_attempts=(trace,),
    )
    return parsed, draft, F2._build_envelope(target_words)


def _assembled(tmp_path: Path):
    """Run the pure-python golden chain into a REAL tmp ledger."""
    parsed, draft, envelope = _sealed_draft(_MARKUP)
    led = _PL.new_ledger(episode_id="fable2_golden_ep",
                         out_dir=str(tmp_path / "ep"))
    led.data.setdefault("meta", {})["fable2"] = {}
    # the golden play carries 51 character words -> target 50 (band 46-56)
    F2._assemble(
        led, draft, _treatment(), _CAST_ROWS, _PAYLOAD,
        owner_bank="scifi_news_pro")
    return led, parsed, led.data["meta"]


def _hierarchies(led) -> dict:
    return {k: led.data[k]
            for k in ("cast", "scenes", "shots", "beats", "lines", "music")}


# ---------------------------------------------------------------------------
# 1. Golden happy path (fixture-pinned)
# ---------------------------------------------------------------------------

def test_golden_happy_path_matches_fixture(tmp_path):
    """The deterministic assembly output IS the committed golden fixture.
    Regenerate deliberately (tests/fixtures/fable2/README.md) when the
    assembly contract changes -- never to paper over a drift."""
    led, _parsed, _meta = _assembled(tmp_path)
    golden = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert _hierarchies(led) == golden


def test_all_five_hierarchies_emitted(tmp_path):
    # r1/S2: set_lines/set_music populate only their own arrays; assembly
    # must emit scene/shot/beat rows explicitly.
    led, _parsed, _meta = _assembled(tmp_path)
    assert [r["scene_id"] for r in led.data["scenes"]] == ["s01", "s02"]
    assert [r["shot_id"] for r in led.data["shots"]] == [
        "shot_000", "shot_001", "shot_002", "shot_003"]
    assert led.data["beats"], "beats[] must be populated"
    assert led.data["lines"], "lines[] must be populated"
    assert [r["cue_id"] for r in led.data["music"]] == [
        "opening", "inter_01", "closing"]


def test_music_rows_own_canonical_placement_and_sentinel_anchor(tmp_path):
    led, _parsed, _meta = _assembled(tmp_path)
    rows = led.data["music"]

    assert [row["placement"] for row in rows] == [
        "opening", "interstitial", "closing",
    ]
    assert [row["anchor_line_id"] for row in rows] == [
        "shot_000_music", "shot_001_music", "shot_003_music",
    ]
    assert all(row["cue_spec_sha256"] for row in rows)


def test_music_sentinel_rows_exact_shape(tmp_path):
    # r2/M5 + r3/M1: sentinel text "", char_id == speaker_role == the
    # role string, NO line-level cue_id (music[] is the cue authority).
    led, _parsed, _meta = _assembled(tmp_path)
    sentinels = [r for r in led.data["lines"]
                 if r["speaker_role"].startswith("music_")]
    roles = [r["speaker_role"] for r in sentinels]
    assert roles == ["music_open", "music_inter", "music_close"]
    for row in sentinels:
        assert row["text"] == ""
        assert row["char_id"] == row["speaker_role"]
        assert "cue_id" not in row
    # music[] carries the cue text as generation_prompt
    open_cue = next(r for r in led.data["music"] if r["cue_id"] == "opening")
    assert "theremin" in open_cue["generation_prompt"]


def test_same_speaker_runs_merge_with_verbatim_proof(tmp_path):
    # r2/M4: VERA's two consecutive scene-1 constituents merge into ONE
    # row whose text equals the space-join; proof runs per constituent.
    led, _parsed, meta = _assembled(tmp_path)
    s1_rows = [r for r in led.data["lines"]
               if r["shot_id"] == "shot_001"
               and r["speaker_role"] == "character"]
    assert [r["char_id"] for r in s1_rows] == ["c02", "c03"]
    vera = s1_rows[0]
    assert vera["text"] == (
        "Play it again, and this time keep the gain low. "
        "The mountain does not repeat itself for committees.")
    proof = {p["line_id"]: p for p in meta["fable2"]["proof_map"]}
    assert len(proof[vera["line_id"]]["constituents"]) == 2
    for c in proof[vera["line_id"]]["constituents"]:
        assert c["artifact"] == "winning_draft"
        assert c["span"][0] >= 0 and c["span"][1] > c["span"][0]


def test_every_spoken_row_proved_and_news_read_named(tmp_path):
    led, _parsed, meta = _assembled(tmp_path)
    spoken = [r for r in led.data["lines"]
              if r["speaker_role"] in ("character", "announcer")]
    proof = {p["line_id"]: p for p in meta["fable2"]["proof_map"]}
    assert set(proof) == {r["line_id"] for r in spoken}
    news_rows = [r for r in spoken if r["text"] == _NEWS_READ]
    assert len(news_rows) == 1
    arts = {c["artifact"]
            for c in proof[news_rows[0]["line_id"]]["constituents"]}
    assert arts == {"treatment.news_close_read"}
    on_disk = json.loads(Path(led.path).read_text(encoding="utf-8"))
    assert on_disk["meta"]["fable2"]["proof_map"] == \
        led.data["meta"]["fable2"]["proof_map"]
    # the retired draft-text side channel never enters the ledger
    assert "_winning_draft_text" not in meta["fable2"]


def test_verbatim_proof_gate_fails_loud_on_foreign_text(tmp_path):
    _parsed, draft, envelope = _sealed_draft(_MARKUP)
    led = _PL.new_ledger(episode_id="fable2_proof_fail",
                         out_dir=str(tmp_path / "ep2"))
    led.data.setdefault("meta", {})["fable2"] = {}
    forged = replace(
        draft, normalized_source="TITLE: Something Else\nEND.")
    with pytest.raises(F2.Fable2ScriptError, match="seal"):
        F2._assemble(
            led, forged, _treatment(), _CAST_ROWS, _PAYLOAD,
            owner_bank="scifi_news_pro")


def test_no_double_append_of_the_news_read(tmp_path):
    # doc s14 item 7: the legacy coda append lives in legacy composition
    # (writer I.5), which this lane never runs. The read appears EXACTLY
    # once in the assembled transcript.
    led, _parsed, _meta = _assembled(tmp_path)
    text = _PL.assemble_script_text_from_ledger(led.data)
    assert text.count(_NEWS_READ) == 1


def test_multiple_inter_cues_after_one_scene_all_survive(tmp_path):
    # kibitz r2 M4: the old dict-keyed collapse silently dropped every
    # inter cue after the first.
    markup = _MARKUP.replace(
        "MUSIC: single sustained violin note, rising",
        "MUSIC: single sustained violin note, rising\n"
        "MUSIC: distant kettle drums, answering")
    parsed, draft, envelope = _sealed_draft(markup)
    led = _PL.new_ledger(episode_id="fable2_intercues",
                         out_dir=str(tmp_path / "ep7"))
    led.data.setdefault("meta", {})["fable2"] = {}
    F2._assemble(
        led, draft, _treatment(), _CAST_ROWS, _PAYLOAD,
        owner_bank="scifi_news_pro")
    assert [r["cue_id"] for r in led.data["music"]] == [
        "opening", "inter_01", "inter_02", "closing"]
    inter_rows = [r for r in led.data["lines"]
                  if r["speaker_role"] == "music_inter"]
    assert len(inter_rows) == 2
    ids = [r["line_id"] for r in inter_rows]
    assert len(set(ids)) == 2, ids


def test_boundary_stamps(tmp_path):
    led, _parsed, _meta = _assembled(tmp_path)
    by_shot: dict = {}
    for r in led.data["lines"]:
        if r["speaker_role"] in ("character", "announcer"):
            by_shot.setdefault(r["shot_id"], []).append(r["boundary"])
    for shot_id, bounds in by_shot.items():
        assert bounds[0] == "shot_start", shot_id
        assert all(b == "beat_start" for b in bounds[1:]), shot_id


def test_role_set_matches_legacy_reference(tmp_path):
    # r4 anchor: assembled fable2 SPOKEN role set == the legacy role set
    # (the reference fixture is ordering/contract truth, never a
    # byte-match target); the full set stays sequencer-legal.
    led, _parsed, _meta = _assembled(tmp_path)
    legacy = json.loads(_LEGACY.read_text(encoding="utf-8"))
    legacy_roles = {r["speaker_role"] for r in legacy["lines"]}
    fable2_spoken = {r["speaker_role"] for r in led.data["lines"]
                     if not r["speaker_role"].startswith("music_")}
    assert fable2_spoken == legacy_roles == {"announcer", "character"}
    all_roles = {r["speaker_role"] for r in led.data["lines"]}
    assert all_roles <= _SEQUENCER_LEGAL_ROLES


def test_row_role_ordering_follows_legacy_reference(tmp_path):
    # Ordering reference: announcer intro first spoken row, announcer
    # (coda/news read) last spoken row, characters in between -- the
    # legacy fixture's shape.
    led, _parsed, _meta = _assembled(tmp_path)
    spoken = [r for r in led.data["lines"]
              if r["speaker_role"] in ("character", "announcer")]
    assert spoken[0]["speaker_role"] == "announcer"
    assert spoken[0]["char_id"] == "announcer"
    assert spoken[-1]["speaker_role"] == "announcer"
    # 22nd live smoke + kibitz r4 M2: EVERY announcer row carries the
    # SENTINEL char_id -- exempt from every cast-keyed downstream
    # mutator by design (the legacy fixture's c01 postamble was a
    # legacy-lane quirk).
    for r in spoken:
        if r["speaker_role"] == "announcer":
            assert r["char_id"] == "announcer", r["line_id"]
    assert {r["speaker_role"] for r in spoken[1:-1]} >= {"character"}


def test_assembled_ledger_passes_the_freeze_gap_audit(tmp_path):
    # kibitz r4 M2: run the assembled fable2 ledger through the freeze
    # module's own gap audit -- zero critical gaps, no skip rows, and
    # the Phase-10 contract facts (roles legal, char_ids resolvable)
    # hold BEFORE any live cascade touches it.
    from nodes import _otr_ledger_freeze as FREEZE
    led, _parsed, _meta = _assembled(tmp_path)
    report = FREEZE.run_gap_audit(led.data, label="fable2_s1b_test")
    assert report.errors == [], report.errors
    for r in led.data["lines"]:
        assert not r.get("skip"), r["line_id"]


def test_line_id_equals_beat_id_and_beats_reference_lines(tmp_path):
    led, _parsed, _meta = _assembled(tmp_path)
    spoken = [r for r in led.data["lines"]
              if r["speaker_role"] in ("character", "announcer")]
    for r in spoken:
        assert r["line_id"] == r["beat_id"]          # legacy 1:1
    beat_ids = {b["beat_id"] for b in led.data["beats"]}
    assert beat_ids == {r["beat_id"] for r in spoken}
    for b in led.data["beats"]:
        assert b["line_ids"] == [b["beat_id"]]


def test_compose_flags_absent_freeze_assertion(tmp_path):
    # fable2 rows are born WITHOUT compose flags; set_lines coerces the
    # absent field to [] (schema-legal for the freeze path).
    led, _parsed, _meta = _assembled(tmp_path)
    for r in led.data["lines"]:
        assert r["compose_flags"] == []


def test_incremental_saves_after_preamble_and_each_scene(tmp_path,
                                                         monkeypatch):
    _parsed, draft, envelope = _sealed_draft(_MARKUP)
    led = _PL.new_ledger(episode_id="fable2_saves",
                         out_dir=str(tmp_path / "ep3"))
    led.data.setdefault("meta", {})["fable2"] = {}
    saves = []
    real_save = led.save
    monkeypatch.setattr(led, "save",
                        lambda: saves.append(1) or real_save())
    F2._assemble(
        led, draft, _treatment(), _CAST_ROWS, _PAYLOAD,
        owner_bank="scifi_news_pro")
    # preamble + 2 scenes + final = 4 saves minimum
    assert len(saves) >= 4


def test_speaker_set_gate_fails_loud(tmp_path):
    _parsed, draft, envelope = _sealed_draft(_MARKUP)
    led = _PL.new_ledger(episode_id="fable2_castgate",
                         out_dir=str(tmp_path / "ep4"))
    led.data.setdefault("meta", {})["fable2"] = {}
    missing_doku = [r for r in _CAST_ROWS if r["name"] != "DOKU"]
    with pytest.raises(F2.Fable2AssembleError, match="speaker set"):
        F2._assemble(
            led, draft, _treatment(), missing_doku, _PAYLOAD,
            owner_bank="scifi_news_pro")


# ---------------------------------------------------------------------------
# 2. Tail-output-contract reference (legacy fixture)
# ---------------------------------------------------------------------------

# Keys later pipeline stages stamp ONTO the saved ledger during a live
# run (voice-bank engine resolution, sequencer timing space). The legacy
# fixture was captured POST-RUN, so it carries them; the writer-stage
# contract this lane must match excludes them.
_POST_WRITER_CAST_STAMPS = frozenset({
    "commercial_clean", "voice_engine", "voice_ref_id"})
_POST_WRITER_LINE_STAMPS = frozenset({"start_s_space"})


def test_cast_row_contract_matches_legacy_reference(tmp_path):
    led, _parsed, _meta = _assembled(tmp_path)
    legacy = json.loads(_LEGACY.read_text(encoding="utf-8"))
    legacy_keys = set(legacy["cast"][0]) - _POST_WRITER_CAST_STAMPS
    for row in led.data["cast"]:
        assert set(row) == legacy_keys, (
            f"cast row {row['char_id']} keys {sorted(set(row))} != "
            f"writer-stage legacy contract {sorted(legacy_keys)}")
    ann = led.data["cast"][0]
    assert (ann["char_id"], ann["name"], ann["tts_model"]) == (
        "c01", "ANNOUNCER", "kokoro")
    for row in led.data["cast"][1:]:
        assert row["tts_model"] == "bark"


def test_line_row_contract_superset_of_legacy_reference(tmp_path):
    led, _parsed, _meta = _assembled(tmp_path)
    legacy = json.loads(_LEGACY.read_text(encoding="utf-8"))
    legacy_keys = set(legacy["lines"][0]) - _POST_WRITER_LINE_STAMPS
    for row in led.data["lines"]:
        assert set(row) >= legacy_keys, (
            f"line {row['line_id']} missing legacy-contract key(s) "
            f"{legacy_keys - set(row)}")


# ---------------------------------------------------------------------------
# 3. No-model contract test (r3 optional, adopted): smallest fable2
#    ledger through assemble_script_text_from_ledger + set_music --
#    catches cue-id drift against the EpisodeAssembler bookend ids.
# ---------------------------------------------------------------------------

def test_no_model_contract_smallest_ledger(tmp_path):
    led = _PL.new_ledger(episode_id="fable2_smallest",
                         out_dir=str(tmp_path / "ep6"))
    led.set_cast([_CAST_ROWS[0], _CAST_ROWS[1]])
    led.set_music([
        {"cue_id": "opening", "description": "slow swell",
         "generation_prompt": "slow swell"},
        {"cue_id": "closing", "description": "warm brass",
         "generation_prompt": "warm brass"},
    ])
    led.set_lines([
        {"line_id": "shot_000_music", "beat_id": "shot_000_music",
         "shot_id": "shot_000", "char_id": "music_open",
         "speaker_role": "music_open", "boundary": None, "text": ""},
        {"line_id": "shot_001_b1", "beat_id": "shot_001_b1",
         "shot_id": "shot_001", "char_id": "c02",
         "speaker_role": "character", "boundary": "shot_start",
         "text": "The tape is the tape."},
        {"line_id": "shot_002_music", "beat_id": "shot_002_music",
         "shot_id": "shot_002", "char_id": "music_close",
         "speaker_role": "music_close", "boundary": None, "text": ""},
    ])
    # set_music preserves EXACTLY the assembler's field set. 720-bakeoff
    # C1 (S2 P1.1) added the authored cue-spec fields (anchor_line_id /
    # placement / target_duration_s) + the derived durable-identity hash
    # (cue_spec_sha256); the durable render fields (wav_path/start_s/dur_s)
    # remain. No dead "position" metadata survives (r3/M1).
    for row in led.data["music"]:
        assert set(row) == {"cue_id", "description", "generation_prompt",
                            "anchor_line_id", "placement", "target_duration_s",
                            "cue_spec_sha256",
                            "wav_path", "start_s", "dur_s"}
    # the assembler bookends key on cue ids "opening"/"closing" -- any
    # cue-id drift here breaks bookend music downstream.
    assert [r["cue_id"] for r in led.data["music"]] == ["opening", "closing"]
    text = _PL.assemble_script_text_from_ledger(led.data)
    assert "[VOICE: VERA" in text
    assert "The tape is the tape." in text
    # music rows render NO transcript token (render contract)
    assert "slow swell" not in text and "warm brass" not in text
