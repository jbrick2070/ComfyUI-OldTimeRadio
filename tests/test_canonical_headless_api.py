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


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import otr_canonical_api_run as canonical  # noqa: E402


RETIRED_FULL_WORKFLOW_HARNESSES = {
    "COMBO_MATRIX.md",
    "FABLE_SOAK_REVIEW.md",
    "FABLE_SOAK_REVIEW_PROMPT.md",
    "_otr_120word_soak_summary.json",
    "_otr_chatterbox_smoke.py",
    "_otr_headless_soak_2026-06-15.md",
    "_otr_soak_capstone.py",
    "_otr_soak_marathon.py",
    "build_ltx_av_bakeoff_workflow.py",
    "kill_all_python.bat",
    "otr_3d_quick_tests.ps1",
    "otr_coverage_sweep.py",
    "otr_overnight_sweep_launch.ps1",
    "otr_run_leg.ps1",
    "overnight_bug_hunt.py",
    "prep_full_run.ps1",
    "queue_smoke.py",
    "run_combo_matrix.py",
    "run_comfy_otr.bat",
    "run_comfy_otr.ps1",
    "run_ltx_av_bakeoff.py",
    "run_otr_30word_smoke.py",
    "smoke_check.py",
    "smoke_watcher.py",
    "soak_bug027_028.py",
    "soak_watch.ps1",
    "start_comfy_h0_baseline.bat",
    "sweep_and_launch.bat",
    "sweep_python_excluding.bat",
    "watch.cmd",
    "watch_full_run.py",
    "worker_iter.py",
}


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
        "--profile", "otr_cloud_lanes",
        "--words", "30",
        "--source-bank", "scifi_news_pro",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "workflows\\otr_canonical.json" in out or \
        "workflows/otr_canonical.json" in out
    assert "profile=otr_cloud_lanes" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    director = _node(prompt, "OTR_VideoDirector")
    assert writer["inputs"]["target_words"] == 30
    assert writer["inputs"]["source_bank"] == "scifi_news_pro"
    assert str(director["inputs"]["announcer_video_model"]).startswith("cloud_")
    assert str(director["inputs"]["announcer_image_model"]).startswith("cloud_")


def test_canonical_words_override_preserves_auto_act_count(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", "none",
        "--words", "320",
        "--source-bank", "media_archive",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "OTR_LedgerScriptWriter.target_words=320" in out
    assert "OTR_LedgerScriptWriter.act_count" not in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    assert writer["inputs"]["target_words"] == 320
    assert writer["inputs"]["act_count"] == "auto"


def test_visual_style_override_does_not_patch_story_fields(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", "none",
        "--words", "320",
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


def test_google_veo_media_profile_dry_run_builds_prompt(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", "google_veo_media",
        "--words", "30",
        "--source-bank", "media_archive",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "profile=google_veo_media" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    director = _node(prompt, "OTR_VideoDirector")
    assert director["inputs"]["announcer_video_model"] == "google_veo_video"
    assert director["inputs"]["music_video_model"] == "google_veo_video"
    assert director["inputs"]["character_video_model"] == "google_veo_video"
    assert director["inputs"]["announcer_image_model"] == "google_image"
    assert director["inputs"]["music_image_model"] == "google_image"
    assert director["inputs"]["character_image_model"] == "google_image"


def test_google_omni_media_profile_dry_run_builds_prompt(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", "google_omni_media",
        "--words", "30",
        "--source-bank", "media_archive",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "profile=google_omni_media" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    director = _node(prompt, "OTR_VideoDirector")
    assert director["inputs"]["announcer_video_model"] == "google_omni_video"
    assert director["inputs"]["music_video_model"] == "google_omni_video"
    assert director["inputs"]["character_video_model"] == "google_omni_video"
    assert director["inputs"]["announcer_image_model"] == "google_image"
    assert director["inputs"]["music_image_model"] == "google_image"
    assert director["inputs"]["character_image_model"] == "google_image"


@pytest.mark.parametrize(
    "profile_id,video_engine",
    [
        ("google_veo_all", "google_veo_video"),
        ("google_omni_all", "google_omni_video"),
    ],
)
def test_google_all_profile_dry_run_builds_prompt(
        tmp_path, monkeypatch, profile_id, video_engine):
    monkeypatch.setenv("GEMINI_API_KEY", "test-google-api-key")
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", profile_id,
        "--words", "120",
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
    render = _node(prompt, "OTR_VideoRenderBatch")
    assert writer["inputs"]["target_words"] == 120
    assert writer["inputs"]["creative_writing_model"] == "google_api:slot-a"
    assert writer["inputs"]["technical_model"] == "google_api:slot-b"
    assert writer["inputs"]["google_api_slot_a_model"] == "gemini-flash-latest"
    assert writer["inputs"]["google_api_slot_b_model"] == "gemini-flash-lite-latest"
    assert cast["inputs"]["voice_bank"] == "google_tts"
    assert cast["inputs"]["char_voice_engine"] == "google_tts"
    assert cast["inputs"]["announcer_voice_engine"] == "google_tts"
    assert char_voice["inputs"]["engine"] == "google_tts"
    assert announcer_voice["inputs"]["engine"] == "google_tts"
    assert music["inputs"]["engine"] == "google_lyria"
    assert director["inputs"]["announcer_video_model"] == video_engine
    assert director["inputs"]["music_video_model"] == video_engine
    assert director["inputs"]["character_video_model"] == video_engine
    assert director["inputs"]["announcer_image_model"] == "google_image"
    assert director["inputs"]["music_image_model"] == "google_image"
    assert director["inputs"]["character_image_model"] == "google_image"
    assert render["inputs"]["engine"] == video_engine


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
    assert writer["inputs"]["target_words"] == 30
    # 2026-07-20: the canonical writer uses official Gemma4Unified on the
    # in-process Transformers lane. NF4 is measured below 7.3 GiB and the
    # lane binds LMFE schema constraints; this is not the Q8 GGUF path.
    #
    # 2026-08-04: THE SIZE SUFFIX IS PART OF THE VALUE. The COMBO choice list
    # is 'google/gemma-4-12b-it (11.9 GB)', so the bare id matched no choice:
    # the operator saw both dropdowns render RED on opening the graph, and an
    # unmatched COMBO can resolve to index 0 -- which on this widget is
    # Mistral-Nemo. A graph that said Gemma could run Mistral. Asserted in
    # full here so the suffix cannot be dropped again.
    assert writer["inputs"]["creative_writing_model"] == \
        "google/gemma-4-12b-it (11.9 GB)"
    assert writer["inputs"]["technical_model"] == \
        "google/gemma-4-12b-it (11.9 GB)"
    assert str(director["inputs"]["announcer_video_model"]).startswith("viz_")


def test_set_allows_only_creative_widgets(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, _out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--set", "OTR_LedgerScriptWriter.target_words=31",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    assert _node(prompt, "OTR_LedgerScriptWriter")["inputs"]["target_words"] == 31


def test_google_api_llm_slots_are_headless_bindable(tmp_path, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--creative-model", "google_api:slot-a",
        "--technical-model", "google_api:slot-b",
        "--google-slot-a-model", "gemini-3.5-flash",
        "--google-slot-b-model", "gemini-3.1-flash-lite",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "google_api_slot_a_model='gemini-3.5-flash'" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    assert writer["inputs"]["creative_writing_model"] == "google_api:slot-a"
    assert writer["inputs"]["technical_model"] == "google_api:slot-b"
    assert writer["inputs"]["google_api_slot_a_model"] == "gemini-3.5-flash"
    assert writer["inputs"]["google_api_slot_b_model"] == "gemini-3.1-flash-lite"


def test_set_refuses_direct_engine_widget_patch(tmp_path):
    with pytest.raises(ValueError, match="creative whitelist"):
        _run_main([
            "--offline-schemas",
            "--dry-run",
            "--set", "OTR_VideoDirector.announcer_video_model=still_motion",
            "--dump-prompt", str(tmp_path / "prompt.json"),
        ])


def test_retired_full_workflow_harnesses_are_not_tracked():
    present = sorted(
        name for name in RETIRED_FULL_WORKFLOW_HARNESSES
        if (SCRIPTS / name).exists()
    )
    assert present == []


def test_headless_wrapper_clears_stale_extra_env_hook_before_boot():
    src = (SCRIPTS / "otr_headless_canonical.ps1").read_text(encoding="utf-8")
    assert "_marathon_extra_env.cmd" in src
    assert "removing stale extra-env hook" in src
    assert "Remove-Item -LiteralPath $StaleExtraEnv -Force" in src


def test_headless_wrapper_applies_profile_launch_env_before_boot():
    src = (SCRIPTS / "otr_headless_canonical.ps1").read_text(encoding="utf-8")
    assert "config\\profiles\\{0}.json" in src
    assert "ProfileObject.launch.env" in src
    assert "Set-Item -Path (\"Env:{0}\" -f $name)" in src
    assert "profile launch env" in src
    assert "-NoBoot server cannot satisfy profile" in src


def test_headless_wrapper_does_not_assign_reserved_pid_variable():
    src = (SCRIPTS / "otr_headless_canonical.ps1").read_text(encoding="utf-8")
    assert "foreach ($pid " not in src
    assert "$proc.ProcessId" in src


def test_headless_wrapper_uses_positive_ownership_and_free_port_selection():
    src = (SCRIPTS / "otr_headless_canonical.ps1").read_text(encoding="utf-8")
    assert "for ($i = 0; $i -lt 10; $i++)" in src
    assert "if (-not $remaining) { return }" in src
    assert "Test-OtrHeadlessServerCommand" in src
    assert "Test-OtrCanonicalRunnerCommand" in src
    assert "Resolve-OtrHeadlessPort" in src
    assert "[int]$Port = 0" in src
    assert "Get-NetTCPConnection -LocalPort 8000" not in src


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


@pytest.mark.skipif(os.name != "nt", reason="PowerShell selector module is Windows-only")
def test_headless_process_selectors_never_claim_the_interactive_gui():
    module = SCRIPTS / "otr_headless_process.psm1"

    def selected(function_name: str, command_line: str) -> bool:
        env = os.environ.copy()
        env["OTR_TEST_COMMAND_LINE"] = command_line
        module_text = str(module).replace("'", "''")
        script = (
            "$ErrorActionPreference='Stop'; "
            f"Import-Module -Name '{module_text}' -Force; "
            f"if ({function_name} -CommandLine $env:OTR_TEST_COMMAND_LINE) "
            "{ 'true' } else { 'false' }"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=False, capture_output=True, text=True, env=env, timeout=15,
        )
        assert completed.returncode == 0, completed.stderr
        return completed.stdout.strip().splitlines()[-1] == "true"

    headless = (
        r"C:\Python\python.exe C:\ComfyUI\main.py --port 8123 "
        r"--extra-model-paths-config C:\OTR\_otr_headless_model_paths.yaml"
    )
    gui = (
        r"C:\Python\python.exe C:\ComfyUI\main.py --port 8001 "
        r"--extra-model-paths-config C:\ComfyUI\shared_model_paths.yaml"
    )
    assert selected("Test-OtrHeadlessServerCommand", headless)
    assert not selected("Test-OtrHeadlessServerCommand", gui)
    assert not selected(
        "Test-OtrHeadlessServerCommand",
        r"C:\Python\python.exe C:\ComfyUI\main.py --port 8123",
    )
    assert selected(
        "Test-OtrCanonicalRunnerCommand",
        r"C:\Python\python.exe C:\OTR\scripts\otr_canonical_api_run.py --words 30",
    )
    assert not selected("Test-OtrCanonicalRunnerCommand", gui)


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
