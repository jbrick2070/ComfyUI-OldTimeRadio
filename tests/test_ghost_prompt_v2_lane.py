"""Ghost Prompt v2 -- the WIRING, on a real ShotLock plan and a real request.

`test_ghost_signal_author.py` pins the pure surfaces. This module pins the
things that can only be wrong when the parts are joined: that the object the
cast-time preflight validated is the object the durable row carries, that a
render request built from it reaches the engine with the same leaf, that the
video seed does not move, that a replay spends nothing, and that the optional
motion-clause pass never loads a writer for an episode whose every shot owns
its own motion.
"""
from __future__ import annotations

import copy

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes import otr_shot_lock as sl
from nodes._otr_video_engines import ghost_signal_author as gsa
from nodes._otr_video_engines import ghost_signal_prompt as gsp
from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines.schemas import ShotRow

# The non-haunted lanes retired 2026-08-23; the survivor inherits this
# composer unchanged.
GHOST = "animatediff15_v3_haunted_video"
NON_GHOST = "ltx_video"

POLICY = {
    "video_models": {"announcer_visual": GHOST, "music_visual": GHOST,
                     "character_visual": GHOST},
    "effective_video_models": {"announcer_visual": GHOST,
                               "music_visual": GHOST,
                               "character_video": GHOST},
}


def _ledger():
    """A minimal but REAL-SHAPED episode: two bookends and two character beats."""
    return {
        "meta": {"episode_seed": 1013426535, "visual_style": "archival_documentary",
                 "freeze_timestamp": "2026-08-22T22:27:56.943819+00:00"},
        "cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "gender": "female"},
            {"char_id": "c02", "name": "ADRIAN SPENDER", "gender": "male",
             "appearance": ("a broad steady man in a charcoal overcoat, a long "
                            "scar across the left cheek, carrying a battered "
                            "leather satchel")},
            {"char_id": "c03", "name": "RASHIDA PIERCE", "gender": "female",
             "appearance": "a tall stooped woman, an olive uniform, a brass key"},
        ],
        "lines": [
            {"line_id": "music_opening_001", "speaker_role": "music_open",
             "speaker": "RADIO", "text": "instrumental intro, no dialogue",
             "dur_s": 5.0, "start_s": 0.0},
            {"line_id": "b001", "speaker_role": "announcer",
             "speaker": "ANNOUNCER", "char_id": "c01",
             "text": "In the bustling heart of the studio.",
             "beat_intent": "Open the episode and orient the listener.",
             "traits": "welcoming", "arc_phase": "scene",
             "dur_s": 4.0, "start_s": 5.0},
            {"line_id": "b002", "speaker_role": "character",
             "speaker": "ADRIAN SPENDER", "char_id": "c02",
             "text": "Your time's runnin' out, Rashida.",
             "beat_intent": ("Adrian Spender demands the waitress reveal her "
                             "true identity, pressing her against the counter."),
             "traits": "Tense anticipation", "arc_phase": "scene",
             "dur_s": 3.0, "start_s": 9.0},
            {"line_id": "b003", "speaker_role": "character",
             "speaker": "RASHIDA PIERCE", "char_id": "c03",
             "text": "You have no idea what you're asking.",
             "beat_intent": "Rashida Pierce refuses and turns the lamp away.",
             "traits": "guarded", "arc_phase": "climax",
             "dur_s": 3.0, "start_s": 12.0},
            {"line_id": "b004", "speaker_role": "character",
             "speaker": "ADRIAN SPENDER", "char_id": "c02",
             "text": "Then I will take it myself.",
             "beat_intent": "Adrian Spender reaches for the disc.",
             "traits": "cold", "arc_phase": "climax",
             "dur_s": 3.0, "start_s": 15.0},
            {"line_id": "b005", "speaker_role": "announcer",
             "speaker": "ANNOUNCER", "char_id": "c01",
             "text": "And so the disc keeps its secret.",
             "beat_intent": "Close the episode.", "traits": "wry",
             "arc_phase": "resolution", "dur_s": 4.0, "start_s": 18.0},
            {"line_id": "music_closing_001", "speaker_role": "music_close",
             "speaker": "RADIO", "text": "closing theme, no dialogue",
             "dur_s": 5.0, "start_s": 22.0},
        ],
    }


def _shot(shots, shot_id):
    """By ID, never by index -- the synthetic opening shifts every position."""
    for shot in shots:
        if shot["shot_id"] == shot_id:
            return shot
    raise AssertionError("no shot %r in %s" % (
        shot_id, [s["shot_id"] for s in shots]))


def _plan(ledger=None, policy=None, warnings=None):
    led = ledger if ledger is not None else _ledger()
    pol = policy if policy is not None else POLICY
    beats = sl.extract_beats(led)
    budget = sl.compute_clip_budget(beats, 25)
    for beat in beats:
        budget["per_beat"].setdefault(beat["beat_id"], 50)
    groups, shots = sl.build_execution_plan(
        beats, budget, {}, pol, led,
        warnings=warnings if warnings is not None else [])
    led["video"] = {"shots": shots, "fps": 25}
    return led, shots


# --------------------------------------------------------------------------- #
# 1. Coverage: every Ghost beat is authored, and absence is not tolerated.
# --------------------------------------------------------------------------- #

def test_every_ghost_beat_carries_an_authored_object():
    _led, shots = _plan()
    assert shots
    for shot in shots:
        obj = shot.get("ghost_prompt")
        assert obj, shot["shot_id"]
        gsa.validate_ghost_prompt_object(obj)


def test_a_non_ghost_episode_carries_no_ghost_object_at_all():
    """Absence is the honest state: no episode acquires a decision it never made."""
    policy = {"video_models": {r: NON_GHOST for r in POLICY["video_models"]},
              "effective_video_models": {r: NON_GHOST
                                         for r in POLICY["effective_video_models"]}}
    _led, shots = _plan(policy=policy)
    assert shots
    assert all("ghost_prompt" not in s for s in shots)


def test_the_cast_time_preflight_refuses_an_unauthored_ghost_beat():
    """A silent v1 downgrade here would look exactly like a healthy replay."""
    led = _ledger()
    beat = {"beat_id": "b002", "role": "character_video", "char_id": "c02",
            "target_frame_count": 50}
    with pytest.raises(ValueError, match="no authored ghost_prompt"):
        sl._assert_family_inputs_satisfiable_cast_time(
            GHOST, beat, led, POLICY,
            {"c02": "a man, a lean upright figure"}, {})


def test_a_ghost_episode_without_an_episode_seed_fails_loud_by_name():
    led = _ledger()
    led["meta"].pop("episode_seed")
    with pytest.raises(ValueError, match="episode_seed"):
        _plan(ledger=led)


def test_a_bookend_only_ghost_episode_still_requires_the_seed():
    led = _ledger()
    led["lines"] = [ln for ln in led["lines"]
                    if ln["speaker_role"] != "character"]
    led["meta"].pop("episode_seed")
    with pytest.raises(ValueError, match="episode_seed"):
        _plan(ledger=led)


# --------------------------------------------------------------------------- #
# 2. The temporary shot and the durable row carry the SAME object.
# --------------------------------------------------------------------------- #

def test_the_preflight_and_the_durable_row_see_one_object(monkeypatch):
    seen = {}
    original = sl._assert_family_inputs_satisfiable_cast_time

    def _spy(engine_name, beat, ledger, policy, subject_sigils=None,
             ghost_prompts=None):
        if ghost_prompts:
            key = str(beat.get("beat_id") or "")
            if key in ghost_prompts:
                seen[key] = copy.deepcopy(ghost_prompts[key])
        return original(engine_name, beat, ledger, policy, subject_sigils,
                        ghost_prompts)

    monkeypatch.setattr(sl, "_assert_family_inputs_satisfiable_cast_time", _spy)
    _led, shots = _plan()
    assert seen
    for shot in shots:
        beat_id = shot["source_line_ids"][0] if shot["source_line_ids"] \
            else shot["shot_id"][len("shot_"):]
        assert shot["ghost_prompt"] == seen[beat_id]


def test_the_durable_row_is_a_copy_not_a_shared_reference():
    """Two beats of one character must not alias one mutable object."""
    _led, shots = _plan()
    a = _shot(shots, "shot_b002")
    b = _shot(shots, "shot_b004")          # the SAME character, another beat
    a["ghost_prompt"]["drawable_beat"] = "mutated"
    assert b["ghost_prompt"]["drawable_beat"] != "mutated"


def test_the_row_validates_against_the_extra_forbid_schema():
    _led, shots = _plan()
    row = ShotRow(**_shot(shots, "shot_b002"))
    assert row.ghost_prompt["mode"] in gsa.GHOST_MODES


# --------------------------------------------------------------------------- #
# 3. The render request.
# --------------------------------------------------------------------------- #

def test_the_request_composes_v3_and_keeps_the_authored_object_on_the_receipt():
    """The v2-object / v3-prompt SPLIT (Prompt v3 Half A, 2026-09-02).

    This test used to assert the opposite -- that the stored ``motif_cue`` and
    ``drawable_beat`` appear verbatim in the sent text -- because that WAS the
    v2 contract. Prompt v3 composes from the episode instead, so the stored
    fields must NOT reach the prompt while still riding the receipt as the
    beat's authored provenance. Rewritten rather than deleted, because both
    halves of that sentence are load-bearing and a deletion would stop
    guarding either one.
    """
    led, shots = _plan()
    for shot in shots:
        req = rd.build_request_from_shot(shot, led, master_audio_path="")
        obs = req["observability"]
        obj = shot["ghost_prompt"]
        assert obs["prompt_version"] == gsp.GHOST_PROMPT_VERSION_V3
        assert obs["ghost_mode"] == obj["mode"]
        # the authored object still rides the receipt, unchanged
        assert obs["ghost_drawable_beat"] == obj["drawable_beat"]
        assert obs["ghost_motif_cue"] == obj["motif_cue"]
        # the authored LEAF never reaches the text -- v3 does not read it
        assert obj["drawable_beat"] not in req["text_prompt"]
        # THE COSTUME never reaches it either. Scoped to character beats on
        # purpose: `motif_for_character` is what invents the coat and the
        # satchel, while a BOOKEND's motif is a radio object from
        # `GHOST_BOOKEND_MOTIFS` -- and the v3 kernel ladder's last tier reads
        # that same table, so on a bookend beat whose episode carries no
        # `key_objects` both paths legitimately arrive at "a broadcast
        # console". That is the radio, not the costume.
        if shot["role"] == "character_video":
            assert obj["motif_cue"] not in req["text_prompt"]
        assert req["negative_prompt"]
        assert obs["kernel_source"] in (
            "bookend_radio", "key_object", "setting", "brief", "bookend")


def test_no_raw_ledger_surface_reaches_a_v2_prompt():
    """Not dialogue, not the name, not the free-text intent, not `scene`."""
    led, shots = _plan()
    for shot in shots:
        text = rd.build_request_from_shot(
            shot, led, master_audio_path="")["text_prompt"].lower()
        for banned in ("adrian", "spender", "rashida", "pierce", "runnin",
                       "moves with", "scene", "waitress", "counter"):
            assert banned not in text, (shot["shot_id"], banned)


def test_the_measured_window_rides_the_request():
    led, shots = _plan()
    for shot in shots:
        obs = rd.build_request_from_shot(
            shot, led, master_audio_path="")["observability"]
        assert obs["clip_window_max"] == gsa.GHOST_CLIP_WINDOW_TOKENS
        if not obs["clip_counter"]:
            # The ONLY sanctioned skip: no installed ComfyUI tokenizer under
            # OTR_TEST_MODE. The receipt says so by carrying no counter name
            # rather than by publishing a number nothing measured.
            assert obs["positive_clip_tokens"] == 0
            continue
        assert obs["clip_counter"] == gsa.GHOST_CLIP_COUNTER
        assert obs["positive_clip_windows"] == 1
        assert obs["negative_clip_windows"] == 1
        assert 0 < obs["positive_clip_tokens"] <= obs["clip_window_max"]
        assert 0 < obs["negative_clip_tokens"] <= obs["clip_window_max"]


def test_every_declared_receipt_is_on_the_trace_allowlist():
    """A stamped key that is not allowlisted never reaches the node-92 report."""
    import inspect
    src = inspect.getsource(rd.run_episode) if hasattr(rd, "run_episode") \
        else inspect.getsource(rd)
    led, shots = _plan()
    obs = rd.build_request_from_shot(
        _shot(shots, "shot_b002"), led, master_audio_path="")["observability"]
    for key in ("author_version", "ghost_schema_version", "ghost_source",
                "ghost_fallback_reason", "ghost_model_id", "ghost_mode",
                "ghost_request_sha8", "ghost_output_sha8",
                "ghost_drawable_beat", "ghost_drawable_beat_sha8",
                "positive_clip_tokens", "positive_clip_windows",
                "negative_clip_tokens", "negative_clip_windows",
                "clip_window_max", "clip_counter",
                # Prompt v3 receipts. This test is the safety net for them:
                # a key stamped and not allowlisted never reaches node 92.
                "prompt_slots", "prompt_slot_tokens", "prompt_dropped",
                "kernel_source"):
        assert key in obs, key
        assert '"%s"' % key in src, key


def test_a_malformed_object_fails_closed_and_never_downgrades():
    led, shots = _plan()
    broken = copy.deepcopy(_shot(shots, "shot_b002"))
    broken["ghost_prompt"]["mode"] = "portrait"
    with pytest.raises(rd.FamilyInputGap, match="malformed ghost_prompt"):
        rd.build_request_from_shot(broken, led, master_audio_path="")


def test_an_absent_object_takes_the_explicit_v1_compatibility_path():
    led, shots = _plan()
    legacy = copy.deepcopy(_shot(shots, "shot_b002"))
    legacy.pop("ghost_prompt")
    legacy["subject_sigil"] = "a man, a lean upright figure, a charcoal coat"
    req = rd.build_request_from_shot(legacy, led, master_audio_path="")
    assert req["observability"]["prompt_version"] == gsp.GHOST_PROMPT_VERSION
    assert "a lean upright figure" in req["text_prompt"]


def test_the_banana_receipt_is_installed_exactly_once():
    """A second idempotent pass would overwrite a real count with zero."""
    led, shots = _plan()
    obs = rd.build_request_from_shot(
        _shot(shots, "shot_b002"), led, master_audio_path="")["observability"]
    assert obs["banana_route"] in ("on", "off")
    assert obs["banana_sha256_after"]


# --------------------------------------------------------------------------- #
# 4. Identity: the seed does not move; the cache identity does.
# --------------------------------------------------------------------------- #

def test_the_authored_leaf_changes_NEITHER_the_prompt_nor_the_seed():
    """The v3 inversion of a v2 test, and it is the property the A/B rests on.

    Under v2 the stored leaf WAS the prompt's only content slot, so rewriting it
    rewrote the text. Under v3 the prompt is composed from the episode and the
    stored leaf is never read -- so a rewritten leaf changes nothing that
    reaches the sampler, while the receipt still records the new authored value.

    The seed half never stopped mattering and is unchanged: `request_hash` mixes
    the brief, the cast, the beat and the character, and has never included the
    prompt.
    """
    led, shots = _plan()
    shot = copy.deepcopy(_shot(shots, "shot_b002"))
    before = rd.build_request_from_shot(shot, led, master_audio_path="")
    rewritten = copy.deepcopy(shot)
    rewritten["ghost_prompt"]["drawable_beat"] = \
        "the clasp opens and a slow band of light crosses it"
    rewritten["ghost_prompt"]["output_sha256"] = gsa.output_sha256(
        rewritten["ghost_prompt"]["drawable_beat"])
    after = rd.build_request_from_shot(rewritten, led, master_audio_path="")
    assert after["text_prompt"] == before["text_prompt"]
    assert after["seed_bundle"] == before["seed_bundle"]
    assert rewritten["render_request_hash"] == shot["render_request_hash"]
    assert after["observability"]["prompt_sha8"] == \
        before["observability"]["prompt_sha8"]
    # the receipt still tells the truth about what was authored
    assert after["observability"]["ghost_drawable_beat"] != \
        before["observability"]["ghost_drawable_beat"]


# --------------------------------------------------------------------------- #
# 5. Replay spends nothing; a configured model that fails stays loud.
# --------------------------------------------------------------------------- #

def test_a_second_pass_over_a_stamped_ledger_replays_byte_identically():
    led, shots = _plan()
    replayed_led, replayed = _plan(ledger=copy.deepcopy(led))
    for a, b in zip(shots, replayed):
        assert a["ghost_prompt"]["drawable_beat"] == \
            b["ghost_prompt"]["drawable_beat"]
        assert a["ghost_prompt"]["request_sha256"] == \
            b["ghost_prompt"]["request_sha256"]
        assert a["ghost_prompt"]["output_sha256"] == \
            b["ghost_prompt"]["output_sha256"]
    assert replayed_led["video"]["shots"] is replayed


def test_a_replayed_writer_row_becomes_replay_and_a_fallback_stays_a_fallback():
    """Reuse can never launder a deterministic clause into proof eligibility."""
    led, shots = _plan()
    led = copy.deepcopy(led)
    for shot in led["video"]["shots"]:
        shot["ghost_prompt"]["source"] = "writer_llm"
        shot["ghost_prompt"]["fallback_reason"] = ""
    _l2, again = _plan(ledger=led)
    assert all(s["ghost_prompt"]["source"] == "replay" for s in again)

    led2 = copy.deepcopy(led)
    for shot in led2["video"]["shots"]:
        shot["ghost_prompt"]["source"] = "deterministic_fallback"
        shot["ghost_prompt"]["fallback_reason"] = "no writer model configured"
    _l3, again2 = _plan(ledger=led2)
    assert all(s["ghost_prompt"]["source"] == "deterministic_fallback"
               for s in again2)
    assert all(s["ghost_prompt"]["fallback_reason"] for s in again2)


def test_a_same_hash_malformed_stored_object_fails_closed():
    led, shots = _plan()
    led = copy.deepcopy(led)
    _shot(led["video"]["shots"], "shot_b002")["ghost_prompt"]["motif_cue"] = ""
    with pytest.raises(gsa.GhostAuthorValidationError):
        _plan(ledger=led)


def test_a_changed_safe_input_reauthors_rather_than_replaying():
    led, shots = _plan()
    led = copy.deepcopy(led)
    before = _shot(led["video"]["shots"],
                   "shot_b002")["ghost_prompt"]["request_sha256"]
    for line in led["lines"]:
        if line["line_id"] == "b002":
            line["traits"] = "furious"
    _l2, again = _plan(ledger=led)
    assert _shot(again, "shot_b002")["ghost_prompt"]["request_sha256"] != before
    # Every OTHER beat still replays -- one changed input reauthors one row.
    assert _shot(again, "shot_b003")["ghost_prompt"]["request_sha256"] ==         _shot(shots, "shot_b003")["ghost_prompt"]["request_sha256"]


def test_the_test_mode_path_stamps_a_complete_deterministic_object():
    """No model configured is a legitimate path -- it is never a partial one."""
    _led, shots = _plan()
    for shot in shots:
        obj = shot["ghost_prompt"]
        assert obj["source"] == "deterministic_fallback"
        assert obj["fallback_reason"]
        assert obj["model_id"] == gsa.GHOST_DETERMINISTIC_MODEL_ID
        ok, why = gsa.validate_drawable_beat(
            obj["drawable_beat"], mode=obj["mode"],
            names=("ADRIAN SPENDER", "RASHIDA PIERCE", "ANNOUNCER"))
        assert ok, why


# --------------------------------------------------------------------------- #
# 7. One beat, one clip, one prompt -- and no workflow change.
# --------------------------------------------------------------------------- #

def test_one_beat_is_one_clip_and_one_prompt():
    _led, shots = _plan()
    beats = {s["source_line_ids"][0] if s["source_line_ids"]
             else s["shot_id"][len("shot_"):] for s in shots}
    assert len(beats) == len(shots)
    leaves = [s["ghost_prompt"]["drawable_beat"] for s in shots]
    assert len(set(leaves)) == len(leaves)
    for shot in shots:
        plan = shot.get("coverage_plan") or {}
        segments = plan.get("segments")
        if segments is not None:
            assert len(segments) == 1, shot["shot_id"]


def test_the_canonical_workflow_is_untouched_by_this_sprint():
    """No node, socket, widget or link change -- the object is internal."""
    import json
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    graph = json.loads(
        (root / "workflows" / "otr_canonical.json").read_text(encoding="utf-8"))
    blob = json.dumps(graph)
    for banned in ("ghost_prompt", "drawable_beat", "ghost_signal_author"):
        assert banned not in blob, banned


# --------------------------------------------------------------------------- #
# 8. The guards themselves -- pinned where a QA pass found them unguarded.
# --------------------------------------------------------------------------- #

def _calls_named(node, name):
    """Every `name(...)` call anywhere under ``node``."""
    import ast
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name]


def test_a_deterministic_leaf_never_collides_with_a_replayed_one():
    """Uniqueness is a property of the EPISODE, not of one authoring call."""
    specs = _plan()[1]
    rows = [{"beat_id": s["shot_id"][len("shot_"):], "role": s["role"],
             "mode": s["ghost_prompt"]["mode"],
             "motif_cue": s["ghost_prompt"]["motif_cue"],
             "sanitized_intent": "", "normalized_emotion": "",
             "mapped_arc": ""} for s in specs]
    built = gsa.build_ghost_author_specs(rows, model_id="m/x")
    whole = gsa.deterministic_batch(built, episode_seed=1013426535)
    taken = [whole[built[0]["id"]], whole[built[1]["id"]]]
    rest = gsa.deterministic_batch(built[2:], episode_seed=1013426535,
                                   already_used=taken)
    assert not (set(rest.values()) & set(taken))
    assert len(set(rest.values())) == len(rest)


