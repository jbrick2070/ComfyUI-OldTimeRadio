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
import pathlib
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


def test_canonical_runner_has_no_workflow_argument():
    src = (SCRIPTS / "otr_canonical_api_run.py").read_text(encoding="utf-8")
    assert '"--workflow"' not in src
    assert "CANONICAL_WORKFLOW" in src


def test_cloud_profile_dry_run_builds_prompt_from_canonical(tmp_path):
    dump = tmp_path / "prompt.json"
    rc, out = _run_main([
        "--offline-schemas",
        "--dry-run",
        "--profile", "cloud_all",
        "--words", "30",
        "--source-bank", "science_news",
        "--dump-prompt", str(dump),
    ])
    assert rc == 0
    assert "workflows\\otr_canonical.json" in out or \
        "workflows/otr_canonical.json" in out
    assert "profile=cloud_all" in out
    prompt = json.loads(dump.read_text(encoding="utf-8"))
    writer = _node(prompt, "OTR_LedgerScriptWriter")
    director = _node(prompt, "OTR_VideoDirector")
    assert writer["inputs"]["target_words"] == 30
    assert writer["inputs"]["source_bank"] == "science_news"
    assert str(director["inputs"]["announcer_video_model"]).startswith("cloud_")
    assert str(director["inputs"]["announcer_image_model"]).startswith("cloud_")


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
    assert writer["inputs"]["creative_writing_model"] == \
        "unsloth/gemma-4-12b-it-GGUF [LOCAL GGUF]"
    assert writer["inputs"]["technical_model"] == \
        "mistralai/Mistral-Nemo-Instruct-2407"
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
