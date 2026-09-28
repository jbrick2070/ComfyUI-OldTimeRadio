"""The three model rolls (operator, 2026-09-28): roll_video_lanes and
roll_still_models on OTR_VideoDirector, roll_audio_engines on OTR_CastLock.

The pools are read from the real registries on three host shapes. The prompt
rewrite is exercised with injected pools, so no test depends on what this
machine has installed. The wiring is asserted at its real call sites.
"""
from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from nodes import _otr_lane_rolls as L

_ROOT = Path(__file__).resolve().parents[1]
NVIDIA = {"device_backend": "cuda", "gpu_vendor": "nvidia", "toolchains": [],
          "allow_sidecars": True}
MAC = {"device_backend": "mps", "gpu_vendor": "apple", "toolchains": [],
       "allow_sidecars": True}
CPU = {"device_backend": "cpu", "gpu_vendor": "none", "toolchains": [],
       "allow_sidecars": True}
VIDEO_SLOTS = ("announcer_video_model", "music_video_model", "character_video_model")
IMAGE_SLOTS = ("announcer_image_model", "music_image_model", "character_image_model")


def _prompt(video=False, still=False, audio=False, replay=""):
    """A queued prompt wired like the canonical: the validator gates the
    writer and the director; CastLock and the theme sit further down."""
    return {
        "63": {"class_type": "OTR_WorkflowValidator", "inputs": {}},
        "1": {"class_type": "OTR_LedgerScriptWriter",
              "inputs": {"gate_in": ["63", 0], "replay_from": replay}},
        "62": {"class_type": "OTR_LedgerFreezeCascade",
               "inputs": {"script_json": ["1", 0]}},
        "80": {"class_type": "OTR_CastLock",
               "inputs": {"script_json": ["62", 1], "char_voice_engine": "kokoro",
                          "announcer_voice_engine": "kokoro",
                          "roll_audio_engines": audio}},
        "82": {"class_type": "OTR_AnnouncerVoice",
               "inputs": {"ledger_json": ["80", 0]}},
        "83": {"class_type": "OTR_StableAudioTheme",
               "inputs": {"script_json": ["62", 1], "gate_in": ["82", 2],
                          "engine": "stable_audio_3"}},
        "87": {"class_type": "OTR_VideoDirector",
               "inputs": dict({"gate_in": ["63", 0], "roll_video_lanes": video,
                               "roll_still_models": still},
                              **{s: "viz_green" for s in VIDEO_SLOTS},
                              **{s: "z_image_turbo" for s in IMAGE_SLOTS})},
    }


def _pools(video=("ltx_8gb", "still_flat", "viz_camera"),
           still=("sd15", "z_image_turbo"), music=("musicgen", "stable_audio_3")):
    return {"video_lane": (video, {"mesh_stage": "needs Blender"}),
            "still_model": (still, {}), "music_engine": (music, {})}


# --- the prompt rewrite -----------------------------------------------------

def test_switches_off_leave_the_prompt_untouched():
    prompt = _prompt()
    before = copy.deepcopy(prompt)
    assert L.roll_prompt_lanes(prompt, "63", pools=_pools()) == {}
    assert prompt == before


def test_video_roll_writes_one_lane_into_all_three_pickers():
    from nodes.otr_video_director import exact_menu_option_for
    prompt = _prompt(video=True)
    got = L.roll_prompt_lanes(prompt, "63", pools=_pools(),
                              env={L.VIDEO_SEED_ENV: "7"})
    receipt = got["87"]["video_lane"]
    order = ("ltx_8gb", "still_flat", "viz_camera")
    from nodes import _otr_rolls
    assert receipt["selected"] == _otr_rolls.draw(order, 7)
    assert receipt["eligible_order"] == list(order)
    assert receipt["seed"] == 7 and receipt["seed_source"].startswith(L.VIDEO_SEED_ENV)
    assert receipt["left_out"] == {"mesh_stage": "needs Blender"}
    label = exact_menu_option_for(receipt["selected"])
    assert [prompt["87"]["inputs"][s] for s in VIDEO_SLOTS] == [label] * 3
    assert prompt["87"][L.RECEIPT_KEY] == {"video_lane": receipt}
    # The stills were not asked for, so they keep the saved pick.
    assert [prompt["87"]["inputs"][s] for s in IMAGE_SLOTS] == ["z_image_turbo"] * 3


def test_the_same_seed_draws_the_same_lane():
    picks = {L.roll_prompt_lanes(_prompt(video=True), "63", pools=_pools(),
                                 env={L.VIDEO_SEED_ENV: "123"})["87"]["video_lane"]["selected"]
             for _ in range(3)}
    assert len(picks) == 1


def test_still_roll_is_skipped_when_no_lane_uses_a_still():
    prompt = _prompt(still=True)
    got = L.roll_prompt_lanes(prompt, "63", pools=_pools(), still_check=lambda _i: False)
    assert "skipped" in got["87"]["still_model"]
    assert [prompt["87"]["inputs"][s] for s in IMAGE_SLOTS] == ["z_image_turbo"] * 3


def test_still_roll_writes_one_model_into_all_three_still_pickers():
    prompt = _prompt(still=True)
    got = L.roll_prompt_lanes(prompt, "63", pools=_pools(), still_check=lambda _i: True,
                              env={L.STILL_SEED_ENV: "3"})
    picked = got["87"]["still_model"]["selected"]
    assert picked in ("sd15", "z_image_turbo")
    assert [prompt["87"]["inputs"][s] for s in IMAGE_SLOTS] == [picked] * 3


def test_the_still_check_reads_the_lanes_as_they_will_render():
    """Procedural visualizers mint no still; an LTX lane does."""
    inputs = {s: "viz_green" for s in VIDEO_SLOTS}
    assert L.uses_a_still(inputs) is False
    inputs["music_video_model"] = "ltx_8gb"
    assert L.uses_a_still(inputs) is True


def test_audio_switch_rolls_the_theme_music_at_the_gate_and_not_the_voice():
    prompt = _prompt(audio=True)
    got = L.roll_prompt_lanes(prompt, "63", pools=_pools(), env={L.MUSIC_SEED_ENV: "1"})
    music = got["80"]["music_engine"]["selected"]
    assert prompt["83"]["inputs"]["engine"] == music
    assert prompt["80"][L.RECEIPT_KEY] == {"music_engine": got["80"]["music_engine"]}
    # The voice rolls in CastLock, where the language is known.
    assert prompt["80"]["inputs"]["char_voice_engine"] == "kokoro"
    assert "voice_engine" not in got["80"]


def test_a_workflow_without_theme_music_says_so():
    prompt = _prompt(audio=True)
    del prompt["83"]
    got = L.roll_prompt_lanes(prompt, "63", pools=_pools())
    assert "skipped" in got["80"]["music_engine"]


def test_a_replay_rolls_nothing():
    prompt = _prompt(video=True, still=True, audio=True, replay="bundle.zip")
    before = copy.deepcopy(prompt)
    got = L.roll_prompt_lanes(prompt, "63", pools=_pools(), still_check=lambda _i: True)
    assert all("skipped" in r for rs in got.values() for r in rs.values())
    for node_id in before:
        assert prompt[node_id]["inputs"] == before[node_id]["inputs"]


def test_an_empty_pool_stops_the_run_and_says_why():
    with pytest.raises(L.LaneRollError) as err:
        L.roll_prompt_lanes(_prompt(video=True), "63",
                            pools={"video_lane": ((), {"ltx_8gb": "cannot run here"})})
    assert L.VIDEO_SWITCH in str(err.value) and "cannot run here" in str(err.value)


def test_a_wired_picker_or_switch_is_refused():
    prompt = _prompt(video=True)
    prompt["87"]["inputs"]["music_video_model"] = ["99", 0]
    with pytest.raises(L.LaneRollError, match="wired"):
        L.roll_prompt_lanes(prompt, "63", pools=_pools())
    prompt = _prompt()
    prompt["87"]["inputs"][L.VIDEO_SWITCH] = ["99", 0]
    with pytest.raises(L.LaneRollError, match="wired"):
        L.roll_prompt_lanes(prompt, "63", pools=_pools())


def test_a_receipt_a_resubmitted_prompt_carries_is_dropped():
    prompt = _prompt()
    prompt["87"][L.RECEIPT_KEY] = {"video_lane": {"selected": "ltx_8gb"}}
    L.roll_prompt_lanes(prompt, "63", pools=_pools())
    assert L.RECEIPT_KEY not in prompt["87"]


def test_only_nodes_downstream_of_this_validator_are_rolled():
    prompt = _prompt(video=True)
    prompt["87"]["inputs"]["gate_in"] = ["500", 0]      # another validator's
    assert L.roll_prompt_lanes(prompt, "63", pools=_pools()) == {}
    assert prompt["87"]["inputs"]["announcer_video_model"] == "viz_green"


def test_string_switch_values_from_an_api_client_read_as_yes_or_no():
    for value, rolled in (("true", True), ("False", False), ("1", True), ("", False)):
        got = L.roll_prompt_lanes(_prompt(video=value), "63", pools=_pools())
        assert bool(got) is rolled, value


# --- reading the receipts back ------------------------------------------------

def test_a_switch_nothing_rolled_is_refused_by_name():
    with pytest.raises(L.LaneRollError, match=L.STILL_SWITCH):
        L.assert_rolled(_prompt(), "87", ("still_model",), "OTR_VideoDirector")
    prompt = _prompt(still=True)
    L.roll_prompt_lanes(prompt, "63", pools=_pools(), still_check=lambda _i: True)
    got = L.assert_rolled(prompt, "87", ("still_model",), "OTR_VideoDirector")
    assert got["still_model"]["selected"]
    assert L.assert_rolled(None, None, (), "OTR_VideoDirector") == {}


def test_ledger_meta_carries_the_gate_receipts_under_their_ledger_keys():
    prompt = _prompt(video=True, audio=True)
    L.roll_prompt_lanes(prompt, "63", pools=_pools())
    meta = L.ledger_meta(prompt)
    assert set(meta) == {"video_lane_roll", "music_engine_roll"}
    assert meta["video_lane_roll"]["surface"] == "video_lane"


# --- the pools ----------------------------------------------------------------

def test_cloud_engines_are_never_candidates_anywhere():
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_image_engines import registry as ireg
    for pool, reg in ((L.video_pool, vreg), (L.still_pool, ireg)):
        cloud = {n for n in reg.all_engine_names() if L.is_cloud_engine(reg.get_engine(n))}
        assert cloud, reg
        for host in (NVIDIA, MAC, CPU):
            eligible, left_out = pool(host)
            assert not cloud & (set(eligible) | set(left_out)), host
    for pool in (L.music_pool, L.voice_pool):
        eligible, left_out = pool(NVIDIA)
        assert not {"sonilo", "google_lyria", "cloud_elevenlabs", "google_tts"} & (
            set(eligible) | set(left_out))


def test_the_cloud_rule_agrees_with_both_render_paths():
    """Every engine a render path treats as provider-side is cloud here too."""
    from nodes._otr_video_engines import registry as vreg
    from nodes._otr_video_engines.render_driver import _is_cloud_video_engine
    from nodes._otr_image_engines import registry as ireg
    from nodes.otr_image_gen_dispatcher import _is_cloud_image_engine
    for name in vreg.all_engine_names():
        if _is_cloud_video_engine(name):
            assert L.is_cloud_engine(vreg.get_engine(name)), name
    for name in ireg.all_engine_names():
        if _is_cloud_image_engine(name):
            assert L.is_cloud_engine(ireg.get_engine(name)), name
    from nodes._otr_audio_engines import registry as areg
    for name in ("sonilo", "google_lyria", "cloud_elevenlabs", "google_tts"):
        assert L.is_cloud_engine(areg.get_engine(name)), name
    for name in ("kokoro", "bark", "musicgen", "stable_audio_3"):
        assert not L.is_cloud_engine(areg.get_engine(name)), name


def test_a_cpu_host_draws_only_the_lanes_that_run_without_a_gpu():
    """No image model runs on a CPU-only machine, so the still lanes -- which
    render from a still -- are left out too, and only the visualizers remain."""
    eligible, left_out = L.video_pool(CPU)
    assert eligible and all(n.startswith("viz_") for n in eligible), eligible
    assert "needs a still" in left_out["still_pan"], left_out["still_pan"]
    assert L.still_pool(CPU)[0] == ()


def test_a_mac_draws_only_lanes_that_list_metal():
    from nodes._otr_video_engines import registry as vreg
    eligible, left_out = L.video_pool(MAC)
    assert "ltx_8gb" in eligible
    for name in eligible:
        assert "mps" in vreg.CAPABILITIES[name]["device_backends"], name
    assert "ltx25_foley_16gb" in left_out


def test_humo_is_left_out_because_the_route_freeze_redirects_it():
    _eligible, left_out = L.video_pool(NVIDIA)
    for name in ("humo", "humo_1.7B"):
        assert "redirected" in left_out[name], left_out[name]


def test_one_voice_engine_voices_the_whole_cast():
    eligible, left_out = L.voice_pool(NVIDIA)
    assert "indextts2" not in eligible
    assert "characters only" in left_out["indextts2"]
    assert "kokoro" in eligible


def test_a_non_english_episode_rolls_only_what_its_row_admits():
    from nodes import _otr_episode_languages as langs
    admitted = set(langs.row_by_iso("es").engines or {})
    eligible, left_out = L.voice_pool(NVIDIA, admitted)
    assert set(eligible) <= admitted
    assert "bark" in left_out and "language" in left_out["bark"]


def test_the_voice_roll_keeps_the_writers_voices_under_preserve_ledger():
    got = L.roll_voice_engine({}, "preserve_ledger", pool=(("kokoro",), {}))
    assert "skipped" in got and "preserve_ledger" in got["skipped"]
    got = L.roll_voice_engine({}, "auto_registry", pool=(("bark", "kokoro"), {}),
                              env={L.VOICE_SEED_ENV: "5"})
    assert got["selected"] in ("bark", "kokoro") and got["surface"] == "voice_engine"


# --- the wiring, at its real sites ---------------------------------------------

def test_the_gate_rolls_before_every_other_readiness_check():
    from nodes import _otr_workflow_validator as V
    src = inspect.getsource(V._queue_time_readiness_gates)
    body = src[src.index('"""', src.index('"""') + 3):]
    assert body.index("roll_prompt_lanes(prompt, unique_id)") \
        < body.index("ensure_prompt_cloud_slugs(prompt, unique_id)") \
        < body.index("ensure_prompt_visual_assets(prompt, unique_id)")


def _optional(cls):
    return list(cls.INPUT_TYPES()["optional"])


def test_the_switches_are_appended_last_and_default_off():
    from nodes.otr_video_director import OTRVideoDirector
    from nodes.cast_lock import CastLock
    assert _optional(OTRVideoDirector)[-2:] == [L.VIDEO_SWITCH, L.STILL_SWITCH]
    assert _optional(CastLock)[-1] == L.AUDIO_SWITCH
    for cls, name in ((OTRVideoDirector, L.VIDEO_SWITCH), (OTRVideoDirector, L.STILL_SWITCH),
                      (CastLock, L.AUDIO_SWITCH)):
        kind, spec = cls.INPUT_TYPES()["optional"][name]
        assert kind == "BOOLEAN" and spec["default"] is False, name
        assert "16 GB" in spec["tooltip"], name
        assert cls.INPUT_TYPES()["hidden"] == {"queued_prompt": "PROMPT",
                                               "node_id": "UNIQUE_ID"}


def test_every_shipped_workflow_ships_the_switches_off():
    last = {"OTR_VideoDirector": [False, False], "OTR_CastLock": [False]}
    seen = 0
    for path in sorted((_ROOT / "workflows").glob("*.json")):
        for node in json.loads(path.read_text(encoding="utf-8"))["nodes"]:
            want = last.get(node.get("type"))
            if want:
                assert node["widgets_values"][-len(want):] == want, (path.name, node["id"])
                seen += 1
    assert seen >= 2 * 25


def test_the_director_refuses_a_switch_the_gate_never_rolled():
    from nodes.otr_video_director import OTRVideoDirector
    with pytest.raises(L.LaneRollError, match=L.VIDEO_SWITCH):
        OTRVideoDirector().direct("viz_green", "viz_green", "viz_green", "sd15", "sd15",
                                  "sd15", 25, 832, 480, roll_video_lanes=True)


def _cast_ledger(meta=None):
    cast = [{"char_id": "c1", "name": "MONTY", "gender": "male"},
            {"char_id": "c2", "name": "VERA", "gender": "female"},
            {"char_id": "a1", "name": "ANNOUNCER", "gender": "male"}]
    return json.dumps({"meta": meta or {"episode_seed": 42}, "cast": cast, "lines": []})


def test_cast_lock_rolls_the_voice_and_stamps_every_receipt(monkeypatch):
    from nodes.cast_lock import CastLock
    monkeypatch.setenv(L.VOICE_SEED_ENV, "11")
    prompt = _prompt(video=True, audio=True)
    L.roll_prompt_lanes(prompt, "63", pools=_pools())
    out = CastLock().lock(script_json=_cast_ledger(), cast_voice_policy="auto_registry",
                          roll_audio_engines=True, queued_prompt=prompt, node_id="80")
    meta = json.loads(out[0])["meta"]
    voice = meta["voice_engine_roll"]
    assert voice["selected"] and voice["seed"] == 11
    assert meta["music_engine_roll"] == prompt["80"][L.RECEIPT_KEY]["music_engine"]
    assert meta["video_lane_roll"] == prompt["87"][L.RECEIPT_KEY]["video_lane"]
    for row in json.loads(out[0])["cast"]:
        assert row["voice_engine"] == voice["selected"], row


def test_cast_lock_refuses_an_audio_switch_the_gate_never_rolled():
    from nodes.cast_lock import CastLock
    with pytest.raises(L.LaneRollError, match=L.AUDIO_SWITCH):
        CastLock().lock(script_json=_cast_ledger(), cast_voice_policy="auto_registry",
                        roll_audio_engines=True, queued_prompt=_prompt(), node_id="80")


# --- the checks the pools borrow ---------------------------------------------

@pytest.mark.parametrize("name,env_var", [("chatterbox", "OTR_CHATTERBOX_VENV"),
                                          ("dia", "OTR_DIA_VENV"),
                                          ("indextts2", "OTR_INDEXTTS2_VENV")])
def test_a_missing_separate_install_is_named(monkeypatch, tmp_path, name, env_var):
    from nodes._otr_audio_engines import registry as areg
    monkeypatch.setenv(env_var, str(tmp_path / "nope" / "python.exe"))
    gaps = areg.get_engine(name).install_gaps()
    assert gaps[0] == ("isolated venv python", str(tmp_path / "nope" / "python.exe"))
    with pytest.raises(RuntimeError, match="Path B not installed"):
        areg.get_engine(name).load()


def test_indextts2_names_a_missing_weights_config(monkeypatch, tmp_path):
    from nodes._otr_audio_engines import registry as areg
    venv = tmp_path / "python.exe"
    venv.write_text("")
    weights = tmp_path / "checkpoints"
    weights.mkdir()
    worker = tmp_path / "worker.py"
    worker.write_text("")
    monkeypatch.setenv("OTR_INDEXTTS2_VENV", str(venv))
    monkeypatch.setenv("OTR_INDEXTTS2_DIR", str(weights))
    monkeypatch.setenv("OTR_INDEXTTS2_WORKER", str(worker))
    engine = areg.get_engine("indextts2")
    assert engine.install_gaps() == [("weights config", str(weights / "config.yaml"))]
    with pytest.raises(RuntimeError, match="weights incomplete"):
        engine.load()


def test_flux_refuses_by_name_when_its_checkpoint_is_not_installed(monkeypatch):
    import sys
    import types
    from nodes._otr_image_engines import registry as ireg
    engine = ireg.get_engine("flux_gen1")
    engine = engine() if isinstance(engine, type) else engine
    fake = types.ModuleType("folder_paths")
    fake.get_filename_list = lambda _kind: ["v1-5-pruned-emaonly-fp16.safetensors"]
    monkeypatch.setitem(sys.modules, "folder_paths", fake)
    with pytest.raises(ireg.EngineUnusable, match="flux1-dev-fp8"):
        engine.assert_usable({}, {})
    fake.get_filename_list = lambda _kind: ["flux1-dev-fp8.safetensors"]
    assert engine.assert_usable({}, {}) == "flux_gen1"
