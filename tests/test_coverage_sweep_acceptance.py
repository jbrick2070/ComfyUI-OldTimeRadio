"""GO_FORWARD section 4A acceptance-mode unit tests (M2 + M3 + M5).

Pure-Python coverage of the two coverage-sweep gate helpers that close the
false-GREEN holes the 2026-06-13 roundtable surfaced:

    * M2 -- sweep_exit_code: the old `return 0 if passed == len(results)` read
      GREEN on an EMPTY result set (0 == 0). Empty is now always RED, and
      acceptance additionally requires wan_i2v AND wan_ti2v present + PASS.
    * M3 -- acceptance_preflight: a gated-off Wan leg enumerates "run", fails
      assert_usable closed, falls back, and (pre-M1) false-passes. Acceptance
      must enable the core Wan flags and forbid --exclude of core Wan.
    * M5 -- acceptance_preflight: OTR_TEST_MODE must be unset so the Wan
      render-phase VRAM<=14.5GB assert runs.

The sweep module resolves the live server output tree at import; we set
OTR_SOAK_SERVER_OUTPUT first so it returns the override without touching disk.
"""

from __future__ import annotations

import os
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

# Make the module import side-effect free: the explicit env override is
# returned verbatim by _resolve_server_output (no disk probe, no SystemExit).
os.environ.setdefault("OTR_SOAK_SERVER_OUTPUT", tempfile.gettempdir())

import otr_coverage_sweep as sweep  # noqa: E402


# --------------------------------------------------------------------------- #
# M3 + M5 -- acceptance_preflight
# --------------------------------------------------------------------------- #
def _clean_env():
    # wan_i2v enabled, OTR_TEST_MODE unset.
    return {"OTR_ENABLE_WAN_I2V": "1"}


def test_preflight_clean_when_flag_set_and_test_mode_unset():
    problems = sweep.acceptance_preflight(_clean_env(), {"wan_i2v"}, [])
    assert problems == []


def test_preflight_treats_test_mode_zero_as_unset():
    env = _clean_env()
    env["OTR_TEST_MODE"] = "0"
    assert sweep.acceptance_preflight(env, {"wan_i2v"}, []) == []


def test_preflight_flags_test_mode_set():
    env = _clean_env()
    env["OTR_TEST_MODE"] = "1"
    problems = sweep.acceptance_preflight(env, {"wan_i2v"}, [])
    assert any("OTR_TEST_MODE" in p and "M5" in p for p in problems)


def test_preflight_flags_registered_wan_with_disabled_flag():
    # wan_i2v registered but its enable flag is not 1.
    problems = sweep.acceptance_preflight({}, {"wan_i2v"}, [])
    assert any("OTR_ENABLE_WAN_I2V" in p and "M3" in p for p in problems)


def test_preflight_skips_unregistered_core_engine():
    # wan_ti2v not registered yet (8GB tier unbuilt) -> not a preflight blocker;
    # the results gate (M2) catches its absence instead.
    problems = sweep.acceptance_preflight(_clean_env(), {"wan_i2v"}, [])
    assert not any("wan_ti2v" in p for p in problems)


def test_preflight_requires_registered_wan_ti2v_flag():
    env = {"OTR_ENABLE_WAN_I2V": "1"}  # wan_ti2v flag missing
    problems = sweep.acceptance_preflight(env, {"wan_i2v", "wan_ti2v"}, [])
    assert any("OTR_ENABLE_WAN_TI2V" in p for p in problems)


def test_preflight_forbids_exclude_of_core_wan_substring():
    problems = sweep.acceptance_preflight(_clean_env(), {"wan_i2v"}, ["wan"])
    assert any("--exclude" in p and "wan_i2v" in p for p in problems)


def test_preflight_forbids_exclude_of_exact_core_wan():
    problems = sweep.acceptance_preflight(_clean_env(), {"wan_i2v"}, ["wan_ti2v"])
    assert any("--exclude" in p and "wan_ti2v" in p for p in problems)


def test_preflight_allows_exclude_of_non_wan():
    problems = sweep.acceptance_preflight(_clean_env(), {"wan_i2v"}, ["humo"])
    assert problems == []


# --------------------------------------------------------------------------- #
# M2 -- sweep_exit_code
# --------------------------------------------------------------------------- #
def _leg(name, verdict="PASS"):
    return {"leg": name, "verdict": verdict}


def test_exit_empty_results_is_red_even_non_acceptance():
    assert sweep.sweep_exit_code([], acceptance=False) == 1
    assert sweep.sweep_exit_code([], acceptance=True) == 1


def test_exit_all_pass_non_acceptance_is_green():
    results = [_leg("sweep_music_visual_visualizer"),
               _leg("sweep_announcer_visual_flux_still")]
    assert sweep.sweep_exit_code(results, acceptance=False) == 0


def test_exit_one_fail_is_red():
    results = [_leg("sweep_music_visual_visualizer"),
               _leg("sweep_announcer_visual_flux_still", "SOAK_FAIL")]
    assert sweep.sweep_exit_code(results, acceptance=False) == 1


def test_exit_acceptance_green_when_both_core_wan_pass():
    results = [_leg("sweep_other_beats_visual_wan_i2v"),
               _leg("sweep_other_beats_visual_wan_ti2v"),
               _leg("sweep_music_visual_visualizer")]
    assert sweep.sweep_exit_code(results, acceptance=True) == 0


def test_exit_acceptance_red_when_wan_ti2v_absent():
    # Today's reality: wan_ti2v unbuilt -> acceptance is RED by construction.
    results = [_leg("sweep_other_beats_visual_wan_i2v"),
               _leg("sweep_music_visual_visualizer")]
    assert sweep.sweep_exit_code(results, acceptance=True) == 1


def test_exit_acceptance_red_when_wan_i2v_absent():
    results = [_leg("sweep_other_beats_visual_wan_ti2v"),
               _leg("sweep_music_visual_visualizer")]
    assert sweep.sweep_exit_code(results, acceptance=True) == 1


def test_exit_non_acceptance_green_without_any_wan():
    # The informational coverage pass does not require Wan.
    results = [_leg("sweep_music_visual_visualizer")]
    assert sweep.sweep_exit_code(results, acceptance=False) == 0


def test_exit_wan_i2v_token_does_not_match_wan_ti2v_leg():
    # Guard the substring match: a wan_ti2v-only result must NOT satisfy the
    # wan_i2v requirement.
    results = [_leg("sweep_other_beats_visual_wan_ti2v")]
    assert sweep.sweep_exit_code(results, acceptance=True) == 1
