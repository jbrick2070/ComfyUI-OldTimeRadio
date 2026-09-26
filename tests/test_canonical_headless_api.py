"""Guard the one allowed full-workflow headless API path.

The production smoke path must mirror the human workflow: load the canonical
LiteGraph JSON, optionally apply an explicit capability profile, optionally
change only creative/story widgets, then convert against schemas. Retired
soak/smoke harnesses used to force hidden env or engine values and made
headless results hard to trust.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import subprocess
import sys

import pytest
import requests


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import otr_canonical_api_run as canonical  # noqa: E402


def _default_llm_option() -> str:
    """The shipped writer label, derived from DEFAULT_LLM rather than pinned."""
    nodes_dir = str(REPO_ROOT / "nodes")
    if nodes_dir not in sys.path:
        sys.path.insert(0, nodes_dir)
    from _otr_model_catalog import default_llm_option

    return default_llm_option()


def _video_pick(internal: str) -> str:
    """Live VideoDirector combo string for a registered engine."""
    from nodes.otr_video_director import exact_menu_option_for
    return exact_menu_option_for(internal)


def _run_main(args: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = canonical.main(args)
    return rc, buf.getvalue()


def _node(prompt: dict, class_type: str) -> dict:
    matches = [
        node for node in prompt.values()
        if node.get("class_type") == class_type
    ]
    assert len(matches) == 1
    return matches[0]


def test_canonical_runner_workflow_arg_is_opt_in_default_canonical():
    # The runner defaults to the canonical graph WITH its path assertion; an
    # OPT-IN --workflow loads an explicit graph (the story-only scoring graph:
    # writer+freeze, no media) only when the caller deliberately asks. Absent
    # the flag, behaviour is byte-identical to the canonical-only contract, so
    # there is still no silent smoke-vs-canonical drift.
    src = (SCRIPTS / "otr_canonical_api_run.py").read_text(encoding="utf-8")
    assert "CANONICAL_WORKFLOW" in src
    assert '"--workflow"' in src, "the opt-in --workflow arg must exist"
    assert "explicit --workflow" in src, "the opt-in branch must be present"
    # The default (no --workflow) path must still assert the canonical path.
    assert "canonical workflow path mismatch" in src


def test_cloud_profile_dry_run_builds_prompt_from_canonical(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", "otr_cloud_low",
        "--source-bank", "scifi_news_pro",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "workflows\\otr_canonical.json" in out or \
        "workflows/otr_canonical.json" in out
    assert "profile=otr_cloud_low" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    director = _node(prompt, "OTR_VideoDirector")
    assert writer["inputs"]["source_bank"] == "scifi_news_pro"
    assert str(director["inputs"]["announcer_video_model"]).startswith("cloud_")
    assert str(director["inputs"]["announcer_image_model"]).startswith("cloud_")


def test_runner_accepts_an_explicit_desktop_comfyui_url(tmp_path, monkeypatch):
    import otr_api

    monkeypatch.setattr(canonical, "COMFYUI_URL", "http://127.0.0.1:8000")
    monkeypatch.setattr(otr_api, "COMFYUI_URL", "http://127.0.0.1:8000")
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas", "--dry-run",
        "--comfyui-url", "http://127.0.0.1:8188/",
        "--dump-prompt", str(dump),
    ])

    assert rc == 0
    assert "comfy_url=http://127.0.0.1:8188" in out
    assert canonical.COMFYUI_URL == "http://127.0.0.1:8188"
    assert otr_api.COMFYUI_URL == "http://127.0.0.1:8188"


@pytest.mark.parametrize("episode,path", [
    (
        "signal_lost_अंधकार_में_गोपनीय_20260918_095850",
        r"C:\ComfyUI\output\otr\episodes"
        r"\signal_lost_अंधकार_में_गोपनीय_20260918_095850\audio\master.wav",
    ),
    (
        "signal_lost_项链之争_contest_for_the_locket_20260918_101411",
        "otr/episodes/signal_lost_项链之争_contest_for_the_locket_"
        "20260918_101411/clips/beat.mp4",
    ),
])
def test_episode_of_prompt_names_native_episode_paths(
        monkeypatch, episode, path):
    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "native-title-prompt": {
                    "prompt": [0, "native-title-prompt", {}, {}],
                    "outputs": {"81": {"files": [{"path": path}]}},
                }
            }

    monkeypatch.setattr(requests, "get", lambda *_a, **_k: _Response())
    assert canonical.episode_of_prompt("native-title-prompt") == episode


def test_runner_rejects_machine_and_profile_as_competing_selectors():
    with pytest.raises(SystemExit) as exc:
        canonical.main([
            "--offline-schemas", "--dry-run",
            "--machine", "8gb",
            "--profile", "otr_8gb_animatediff",
        ])

    assert exc.value.code == 2


## TOMBSTONE (2026-08-14): test_canonical_words_override_preserves_auto_act_count
## used to assert that --words 320 patched OTR_LedgerScriptWriter.target_words
## to 320 while leaving act_count untouched at its "auto" default -- proving
## the CLI's --words shortcut did not also force an explicit act structure, so
## production could still derive the act count from the word total. Both
## halves of that claim are gone: `target_words` was DELETED from the writer
## (operator directive -- episode length is an observation now, driven by
## act_count alone, never a word-count instruction), and `act_count` no
## longer HAS an "auto" choice to preserve (its choices are now explicit
## "1".."8", default "3"; 'auto' meant "derive from target_words", which no
## longer exists). There is no word-count-vs-act-count interaction left to
## pin, so the test is deleted rather than contorted into asserting something
## it was never written to say.


def test_visual_style_override_does_not_patch_story_fields(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", "none",
        "--visual-style", "video_art",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "OTR_LedgerScriptWriter.visual_style='video_art'" in out
    assert "OTR_LedgerScriptWriter.episode_title" not in out
    assert "OTR_LedgerScriptWriter.custom_premise" not in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    assert writer["inputs"]["visual_style"] == "video_art"
    assert writer["inputs"]["episode_title"] == ""
    assert writer["inputs"]["custom_premise"] == ""


@pytest.mark.parametrize(
    "profile_id,video_engine,image_engine",
    [
        ("otr_cloud_low", "cloud_vidu_q2_pro_fast_720p",
         "cloud_luma_photon_flash"),
        ("otr_cloud_deluxe_3act", "cloud_ltx25_foley_plus", "cloud_flux_pro"),
    ],
)
def test_cloud_row_dry_run_binds_every_slot_it_sets(
        tmp_path, monkeypatch, profile_id, video_engine, image_engine):
    """Every engine a cloud row selects reaches the prompt, and the explicit
    writer flags still win over the row's own writer slots."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-google-api-key")
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", profile_id,
        "--source-bank", "media_archive",
        "--creative-model", "google_api:slot-a",
        "--technical-model", "google_api:slot-b",
        "--google-slot-a-model", "gemini-flash-latest",
        "--google-slot-b-model", "gemini-flash-lite-latest",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert f"profile={profile_id}" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    cast = _node(prompt, "OTR_CastLock")
    char_voice = _node(prompt, "OTR_BatchCharacterVoices")
    announcer_voice = _node(prompt, "OTR_AnnouncerVoice")
    music = _node(prompt, "OTR_StableAudioTheme")
    director = _node(prompt, "OTR_VideoDirector")
    assert writer["inputs"]["creative_writing_model"] == "google_api:slot-a"
    assert writer["inputs"]["technical_model"] == "google_api:slot-b"
    assert writer["inputs"]["google_api_slot_a_model"] == "gemini-flash-latest"
    assert writer["inputs"]["google_api_slot_b_model"] == "gemini-flash-lite-latest"
    assert cast["inputs"]["char_voice_engine"] == "cloud_elevenlabs"
    assert cast["inputs"]["announcer_voice_engine"] == "cloud_elevenlabs"
    assert "engine" not in char_voice["inputs"]
    assert "engine" not in announcer_voice["inputs"]
    assert "voice_bank" not in cast["inputs"]
    assert music["inputs"]["engine"] == "sonilo"
    video_label = _video_pick(video_engine)
    assert director["inputs"]["announcer_video_model"] == video_label
    assert director["inputs"]["music_video_model"] == video_label
    assert director["inputs"]["character_video_model"] == video_label
    assert director["inputs"]["announcer_image_model"] == image_engine
    assert director["inputs"]["music_image_model"] == image_engine
    assert director["inputs"]["character_image_model"] == image_engine


def test_default_dry_run_uses_canonical_values_without_profile(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "profile=" not in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    director = _node(prompt, "OTR_VideoDirector")
    # target_words was DELETED 2026-08-14 (operator directive) -- it no
    # longer appears in the prompt at all, so there is nothing to assert
    # here. 2026-07-20: the canonical writer uses official Gemma4Unified on the
    # in-process Transformers lane. NF4 is measured below 7.3 GiB and the
    # lane binds LMFE schema constraints.
    #
    # 2026-08-04: THE SIZE SUFFIX IS PART OF THE VALUE. A bare repo id matches
    # no choice: the operator saw both dropdowns render RED on opening the
    # graph, and an unmatched COMBO can resolve to index 0 of the list. A graph
    # that said one model could run another. Asserted in full here so the
    # suffix cannot be dropped again.
    #
    # 2026-09-06 (PBUG-20260906-09): DERIVED, not pinned. This assertion
    # hard-coded the Gemma label; when DEFAULT_LLM moved to Qwen the literal
    # stayed, so the test asserted the OLD default was shipped and pinned the
    # drift in place instead of catching it. The index-0 hazard described above
    # is also why the derivation must keep the suffix.
    _expected_writer_model = _default_llm_option()
    assert writer["inputs"]["creative_writing_model"] == _expected_writer_model
    assert writer["inputs"]["technical_model"] == _expected_writer_model
    # 2026-08-15 (operator): the lean default moved from the audio-reactive
    # visualizers to the flat still, so the beat classes actually show the
    # z_image_turbo image they mint. Still a cheap family, not heavy video.
    # OPERATOR DECISION 2026-09-05: the canonical video default moved from
    # still_flat to ltx098_low_video (LTX-Video 2B 0.9.8 distilled) so the
    # out-of-the-box graph renders real video. The LEAN intent of this pin
    # survives -- ltx098 is the lightest real-video lane, runs on CORE
    # loaders with no third-party node pack, and its weights are ungated
    # with a scripted fetch. A HEAVY engine here would still be wrong.
    # NO ENGINE-CHOICE GUARD (operator ruling 2026-09-05): "there should not be
    # a guard for any of the dropdowns, big or small ... they are workable
    # options provided you have the hardware and stack to handle it." Which
    # engine the saved canonical carries is the OPERATOR's call and changes with
    # the machine he is aiming at; a test that pins it turns a preference into a
    # red suite. What still matters -- and is not asserted here -- is that whatever
    # is picked is RUNNABLE: registered, usable for its role, and invocable.


def test_set_allows_only_creative_widgets(tmp_path):
    dump = tmp_path / "prompt.json"
    # num_characters replaces target_words as the probe widget here:
    # target_words was DELETED from OTR_LedgerScriptWriter 2026-08-14
    # (operator directive -- episode length is an observation, not an
    # instruction), so it is no longer a valid --set target. Any other
    # CREATIVE_WHITELIST-listed widget proves the same --set mechanism.
    rc, _out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--set", "OTR_LedgerScriptWriter.num_characters=4",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    assert _node(prompt, "OTR_LedgerScriptWriter")["inputs"]["num_characters"] == 4


def test_google_api_llm_slots_are_headless_bindable(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--creative-model", "google_api:slot-a",
        "--technical-model", "google_api:slot-b",
        "--google-slot-a-model", "gemini-flash-latest",
        "--google-slot-b-model", "gemini-flash-lite-latest",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "google_api_slot_a_model='gemini-flash-latest'" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    assert writer["inputs"]["creative_writing_model"] == "google_api:slot-a"
    assert writer["inputs"]["technical_model"] == "google_api:slot-b"
    assert writer["inputs"]["google_api_slot_a_model"] == "gemini-flash-latest"
    assert writer["inputs"]["google_api_slot_b_model"] == "gemini-flash-lite-latest"


def test_video_lane_sets_the_three_video_dropdowns_as_the_app_does(tmp_path):
    """Operator 2026-09-26: try a video lane headless by hand-picking the
    dropdown, not by adding a workflow. The row's stills stay as the row set
    them; only the three video dropdowns move."""
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas", "--dry-run",
        "--profile", "otr_16gb_video",
        "--video-lane", "h3_low_video",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    director = _node(json.loads(dump.read_text(encoding="utf-8")),
                     "OTR_VideoDirector")["inputs"]
    # The literal label the app's dropdown shows -- not recomputed through the
    # lookup the code under test uses (Sonnet review, 9e677825).
    pick = "h3_low_video (16:9)"
    for widget in canonical.VIDEO_LANE_WIDGETS:
        assert director[widget] == pick, widget
        assert f"OTR_VideoDirector.{widget}={pick!r}" in out
    assert "row model check skipped" in out
    assert "preflight downloads and checks" in out
    rc, _ = _run_main([
        "--offline-schemas", "--dry-run", "--profile", "otr_16gb_video",
        "--dump-prompt", str(tmp_path / "row.json"),
    ])
    row_director = _node(json.loads((tmp_path / "row.json").read_text(
        encoding="utf-8")), "OTR_VideoDirector")["inputs"]
    for widget in ("announcer_image_model", "character_image_model",
                   "music_image_model"):
        assert director[widget] == row_director[widget], widget


def test_video_lane_warns_when_nothing_checks_the_lanes_weights(tmp_path):
    """HuMo is in the dropdown but the pack does not download it, so no
    preflight checks its weights; the runner must say so, not imply one does."""
    rc, out = _run_main([
        "--offline-schemas", "--dry-run",
        "--profile", "otr_16gb_video",
        "--video-lane", "humo_1.7B",
        "--dump-prompt", str(tmp_path / "prompt.json"),
    ])
    assert rc == 0
    assert "nothing checks its weights before the run" in out


def test_video_lane_applies_after_a_machine_key_too(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas", "--dry-run",
        "--machine", "16gb",
        "--video-lane", "h3_low_video",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    director = _node(json.loads(dump.read_text(encoding="utf-8")),
                     "OTR_VideoDirector")["inputs"]
    for widget in canonical.VIDEO_LANE_WIDGETS:
        assert director[widget] == "h3_low_video (16:9)", widget


def test_video_lane_refuses_a_name_that_is_not_in_the_dropdown(tmp_path):
    with pytest.raises(SystemExit, match="not a video lane in the dropdown"):
        _run_main([
            "--offline-schemas", "--dry-run",
            "--video-lane", "no_such_lane",
            "--dump-prompt", str(tmp_path / "prompt.json"),
        ])


def test_set_refuses_direct_engine_widget_patch(tmp_path):
    with pytest.raises(ValueError, match="creative whitelist"):
        _run_main([
            "--offline-schemas",
            "--dry-run",
            "--set", "OTR_VideoDirector.announcer_video_model=still_motion",
            "--dump-prompt", str(tmp_path / "prompt.json"),
        ])


def test_canonical_runner_emits_poll_heartbeats(tmp_path, monkeypatch):
    dump = tmp_path / "prompt.json"
    monkeypatch.setattr(canonical, "build_api_prompt", lambda _args: ({}, []))
    monkeypatch.setattr(canonical, "submit_prompt", lambda _prompt: "prompt-live")

    def _poll(prompt_id, timeout_s, poll_s, on_tick=None):
        assert prompt_id == "prompt-live"
        assert timeout_s == 5400
        assert poll_s == 5
        assert on_tick is not None
        on_tick(0.0, {})
        on_tick(5.9, {"status_str": "running"})
        return "SUCCESS", ""

    monkeypatch.setattr(canonical, "poll_history", _poll)
    rc, out = _run_main(["--dump-prompt", str(dump)])
    assert rc == 0
    assert "t=0s prompt_id=prompt-live status=queued" in out
    assert "t=0s prompt_id=prompt-live status=pending" in out
    assert "t=5s prompt_id=prompt-live status=running" in out
    assert "RESULT SUCCESS prompt_id=prompt-live" in out


def test_poll_history_zero_timeout_waits_for_terminal_result(monkeypatch):
    import otr_api

    responses = iter([
        {"prompt-live": {"status": {"status_str": "running"}}},
        {"prompt-live": {"status": {"status_str": "success", "completed": True}}},
    ])

    class _Response:
        def json(self):
            return next(responses)

    monkeypatch.setattr(
        otr_api.requests, "get", lambda *_args, **_kwargs: _Response()
    )
    monkeypatch.setattr(otr_api.time, "sleep", lambda _seconds: None)

    status, error = otr_api.poll_history(
        "prompt-live", timeout_s=0, poll_s=1
    )

    assert status == "SUCCESS"
    assert error == ""


@pytest.mark.skipif(os.name != "nt", reason="watchdog is a PowerShell harness")
def test_watchdog_recognizes_canonical_terminal_result(tmp_path):
    leg_log = tmp_path / "leg.log"
    leg_log.write_text(
        "[canonical-api] RESULT SUCCESS prompt_id=prompt-live\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
            "-File", str(SCRIPTS / "otr_render_watchdog.ps1"),
            "-LegLog", str(leg_log), "-PollSeconds", "0",
        ],
        check=False, capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "DONE" in completed.stdout
    assert "DONE" in (tmp_path / "leg.log.watchdog").read_text(encoding="ascii")
