"""Story-scaffold UI toggle (2026-06-24) -- the single user-facing control over
the bundled scaffold (style grammar + the KILL-1 body gate + the announcer
non-outcome close), all of which read OTR_ENABLE_STYLE_GRAMMAR.

Covers: the INPUT_TYPES widget (choices/default/append-at-END); and
``_apply_story_scaffold_env`` -- on/off override the env for the run, auto
restores the import-time baseline (no leak), unknown values fall back to auto.

Pure Python; no GPU, no LLM, no network.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import OTR_LedgerScriptWriter as W  # noqa: E402

_ENV = "OTR_ENABLE_STYLE_GRAMMAR"


@pytest.fixture(autouse=True)
def _restore_env():
    """Snapshot + restore the env key and the module baseline around each test."""
    had = _ENV in os.environ
    prev = os.environ.get(_ENV)
    base = W._OTR_SCAFFOLD_ENV_BASELINE
    try:
        yield
    finally:
        if had:
            os.environ[_ENV] = prev
        else:
            os.environ.pop(_ENV, None)
        W._OTR_SCAFFOLD_ENV_BASELINE = base


class TestWidgetSurface:
    def test_story_scaffold_in_input_types(self):
        spec = W.OTR_LedgerScriptWriter.INPUT_TYPES()
        assert "story_scaffold" in spec["optional"]
        choices, meta = spec["optional"]["story_scaffold"]
        assert choices == ["auto", "on", "off"]
        assert meta["default"] == "auto"

    def test_story_scaffold_positional_pin(self):
        # Later source_bank, visual_style, Google API, and source_ref widgets
        # append after this toggle; the scaffold slot itself stays fixed at
        # 21 (was 22 before the 2026-08-14 `target_words` widget removal
        # shifted every slot from num_characters onward down by 1).
        spec = W.OTR_LedgerScriptWriter.INPUT_TYPES()
        order = list(spec["required"].keys()) + list(spec["optional"].keys())
        assert order[21] == "story_scaffold"
        assert order[22] == "source_bank"
        assert order[23] == "visual_style"
        assert order[24] == "google_api_slot_a_model"
        assert order[25] == "google_api_slot_b_model"
        assert order[26] == "source_ref"


class TestApplyScaffoldEnv:
    def test_on_forces_env_1(self):
        os.environ.pop(_ENV, None)
        assert W._apply_story_scaffold_env("on") == "on"
        assert os.environ[_ENV] == "1"

    def test_off_forces_env_0(self):
        os.environ.pop(_ENV, None)
        assert W._apply_story_scaffold_env("off") == "off"
        assert os.environ[_ENV] == "0"

    def test_auto_restores_unset_baseline(self):
        # baseline None (process started with the env unset) -> auto removes it,
        # even if a prior on/off run set it.
        W._OTR_SCAFFOLD_ENV_BASELINE = None
        os.environ[_ENV] = "0"
        assert W._apply_story_scaffold_env("auto") == "auto"
        assert _ENV not in os.environ

    def test_auto_restores_set_baseline(self):
        # baseline "1" (the process started with it on) -> auto restores "1".
        W._OTR_SCAFFOLD_ENV_BASELINE = "1"
        os.environ[_ENV] = "0"
        assert W._apply_story_scaffold_env("auto") == "auto"
        assert os.environ[_ENV] == "1"

    def test_unknown_value_falls_back_to_auto(self):
        W._OTR_SCAFFOLD_ENV_BASELINE = None
        os.environ[_ENV] = "1"
        assert W._apply_story_scaffold_env("banana") == "banana"
        assert _ENV not in os.environ

    def test_case_and_whitespace_insensitive(self):
        os.environ.pop(_ENV, None)
        assert W._apply_story_scaffold_env("  ON ") == "on"
        assert os.environ[_ENV] == "1"

    def test_empty_is_auto(self):
        W._OTR_SCAFFOLD_ENV_BASELINE = None
        os.environ[_ENV] = "0"
        assert W._apply_story_scaffold_env("") == "auto"
        assert _ENV not in os.environ

    def test_no_leak_off_then_auto(self):
        # off sets "0"; a following auto run (unset baseline) clears it so the
        # next prompt sees the default again -- the no-leak guarantee.
        W._OTR_SCAFFOLD_ENV_BASELINE = None
        W._apply_story_scaffold_env("off")
        assert os.environ[_ENV] == "0"
        W._apply_story_scaffold_env("auto")
        assert _ENV not in os.environ


class TestStoryStyleReceipt:
    def test_scaffold_off_receipt_never_uses_visual_style(self):
        meta = {"visual_style": "recur_frac"}
        W._stamp_story_style_receipt(
            meta, contract=None, scaffold_enabled=False)
        assert meta["visual_style"] == "recur_frac"
        assert "style" not in meta
        assert meta["story_scaffold_enabled"] is False
        assert meta["story_style_status"] == W.STORY_STYLE_STATUS_SCAFFOLD_OFF

    def test_contract_receipt_clears_scaffold_off_status(self):
        meta = {
            "visual_style": "recur_frac",
            "story_style_status": W.STORY_STYLE_STATUS_SCAFFOLD_OFF,
        }
        W._stamp_story_style_receipt(
            meta,
            contract=SimpleNamespace(slug="locked_room_suspense"),
            scaffold_enabled=True,
        )
        assert meta["visual_style"] == "recur_frac"
        assert meta["style"] == "locked_room_suspense"
        assert meta["story_scaffold_enabled"] is True
        assert "story_style_status" not in meta
