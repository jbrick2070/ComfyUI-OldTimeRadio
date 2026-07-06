"""S0 tests for the checked-in partner_nodes.yaml pin.

Two classes (pass04 sec 8): (a) no-network structural checks that ALWAYS
run; (b) a drift test that re-derives live signatures and compares to
the pin -- SKIPPED when the ComfyUI core is not importable (offline CI
uses the checked-in artifact; on the render box this is the real drift
alarm: pinned yaml vs live INPUT_TYPES mismatch fails LOUDLY).
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
YAML_PATH = REPO_ROOT / "nodes" / "_otr_shared" / "partner_nodes.yaml"
PIN_SCRIPT = REPO_ROOT / "scripts" / "otr_pin_partner_nodes.py"

EXPECTED_ROW_IDS = {
    "cloud_elevenlabs_flash", "cloud_elevenlabs_tts",
    "cloud_stability_audio", "cloud_sonilo_music",
    "cloud_flux_pro", "cloud_nano_banana_2",
    "cloud_kling_avatar", "cloud_kling_lipsync", "cloud_seedance_2",
    "cloud_wan_i2v",
    # 2026-07-02 roster expansion:
    "cloud_ideogram_v4", "cloud_seedream_2",
    "cloud_elevenlabs_voice_selector",
    # word_razzle Phase 1 (2026-07-03): animated word-card i2v (Pixverse).
    "cloud_pixverse_i2v",
}


@pytest.fixture(scope="module")
def pin() -> dict:
    assert YAML_PATH.is_file(), "checked-in partner_nodes.yaml missing"
    return yaml.safe_load(YAML_PATH.read_text(encoding="utf-8"))


# -- (a) no-network structural checks ----------------------------------------


def test_pin_meta(pin):
    meta = pin["pin_meta"]
    assert meta["schema_version"] == "pin-2026-07-02"
    assert meta["combo_options_excluded"] is True
    assert meta["comfyui_commit"]


def test_all_curated_rows_present(pin):
    assert set(pin["rows"]) == EXPECTED_ROW_IDS


def test_ok_rows_are_complete(pin):
    ok_rows = {rid: r for rid, r in pin["rows"].items()
               if r["status"] == "OK"}
    assert ok_rows, "no OK rows -- pin was generated without the core"
    for rid, row in ok_rows.items():
        for field in ("import_path", "class_name", "provider_id",
                      "function", "inputs", "return_types",
                      "selected_output", "notes"):
            assert field in row, f"{rid} missing {field}"
        assert row["selected_output"] < len(row["return_types"]), rid
        assert isinstance(row["seed_supported"], bool), rid
        if row.get("api_node"):
            # auth broker design requirement: BILLED partner nodes
            # expose BOTH auth hidden inputs and are async
            assert set(row["auth_hidden_present"]) == {
                "auth_token_comfy_org", "api_key_comfy_org"}, rid
            assert row["is_async"] is True, rid
        else:
            # AUX helper rows (e.g. the ElevenLabs voice selector) are
            # free local nodes: no billing, no auth requirement
            assert "AUX" in row["notes"], (
                f"{rid} is not an API node but not marked AUX -- "
                f"either the pin drifted or the row is miscurated")


def test_provider_ids_normalized(pin):
    import re
    for rid, row in pin["rows"].items():
        assert re.match(r"^[A-Z0-9_]+$", row["provider_id"]), rid


def test_shippable_rows_all_ok(pin):
    """Runtime never drops rows; only OK rows SHIP. As of the S0 pin,
    every curated row pinned OK -- if a future re-pin breaks one, this
    test forces the roster conversation instead of a silent drop."""
    not_ok = {rid: r["status"] for rid, r in pin["rows"].items()
              if r["status"] != "OK"}
    assert not_ok == {}, f"rows no longer pinnable: {not_ok}"


# -- (b) live drift test (SUBPROCESS -- skips offline) -------------------------
# Importing the comfy core inside the pytest interpreter corrupts pytest
# teardown (observed: report truncated mid-FAILED). The check runs in a
# fresh interpreter exactly like real pinning. rc: 0 match / 3 drift /
# 4 core unresolvable (skip).


def test_pin_matches_live_install():
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, str(PIN_SCRIPT), "--check"],
        capture_output=True, text=True, timeout=180,
        cwd=str(REPO_ROOT),
    )
    if proc.returncode == 4:
        pytest.skip("ComfyUI core not resolvable on this host "
                    "(offline CI uses the checked-in pin)")
    assert proc.returncode == 0, (
        f"partner node schema DRIFT vs checked-in pin (rc="
        f"{proc.returncode}) -- re-run scripts/_otr_pin_partner_nodes.py "
        f"and review adapters.\nstdout tail:\n{proc.stdout[-2000:]}\n"
        f"stderr tail:\n{proc.stderr[-1000:]}")
