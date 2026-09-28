"""The roll pools' clickable checklists (2026-09-28).

The checklist's choices come from each pool's own input spec (`otr_choices`),
so they must be exactly what `_otr_rolls.parse_roll_pool` accepts for that
pool -- a checkbox the backend refuses would stop the run. The glue must
replace each pool IN PLACE under the same name, or the saved value's position
and the app form's row would break. The pure core runs under node.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _spec(name):
    from nodes.OTR_LedgerScriptWriter import OTR_LedgerScriptWriter as W
    return W.INPUT_TYPES()["optional"][name][1]


def test_each_checklist_offers_exactly_what_its_pool_accepts():
    from nodes import _otr_episode_languages as LANG
    from nodes import _otr_rolls as ROLLS
    assert _spec("style_roll_pool")["otr_choices"] == list(ROLLS.eligible_style_ids())
    assert _spec("bank_roll_pool")["otr_choices"] == list(ROLLS.eligible_bank_ids())
    langs = [c for c in LANG.dropdown_choices() if c != LANG.OFF_LABEL]
    assert _spec("language_roll_pool")["otr_choices"] == langs
    assert LANG.OFF_LABEL not in _spec("language_roll_pool")["otr_choices"]


def test_every_offered_choice_parses():
    """Checking every box must never stop a run."""
    from nodes import _otr_episode_languages as LANG
    from nodes import _otr_rolls as ROLLS
    for name, valid, refused in (
        ("style_roll_pool", ROLLS.eligible_style_ids(), ()),
        ("bank_roll_pool", ROLLS.eligible_bank_ids(), ()),
        ("language_roll_pool",
         [c for c in LANG.dropdown_choices() if c != LANG.OFF_LABEL], (LANG.OFF_LABEL,)),
    ):
        choices = _spec(name)["otr_choices"]
        got = ROLLS.parse_roll_pool(", ".join(choices), valid_ids=valid,
                                    surface=name, refused=refused)
        assert list(got) == choices


def test_the_pools_are_still_typed_text_underneath():
    """The saved value, the API and headless runs keep the typed list."""
    for name in ("style_roll_pool", "language_roll_pool", "bank_roll_pool"):
        kind, meta = __import__("nodes.OTR_LedgerScriptWriter", fromlist=["x"]) \
            .OTR_LedgerScriptWriter.INPUT_TYPES()["optional"][name]
        assert kind == "STRING" and meta["default"] == ""


def test_the_glue_replaces_each_pool_in_place_under_its_own_name():
    glue = (_ROOT / "js" / "roll_pickers.js").read_text(encoding="utf-8")
    assert "app.registerExtension(" in glue and "nodeCreated(node)" in glue
    assert 'from "./roll_pickers_core.js"' in glue
    assert '"language_roll_pool", "bank_roll_pool", "style_roll_pool"' in glue
    assert "node.addDOMWidget(name," in glue            # same name as the text box
    assert "node.widgets.splice(index, 1, widget)" in glue  # same slot
    assert "otr_choices" in glue


def _node_exe():
    for candidate in ("node", r"C:\Program Files\nodejs\node.exe"):
        try:
            subprocess.run([candidate, "--version"], capture_output=True, timeout=30, check=True)
            return candidate
        except (OSError, subprocess.SubprocessError):
            continue
    return None


def test_the_javascript_suite_passes():
    node = _node_exe()
    if node is None:
        pytest.skip("node is not installed; the JS suite cannot run here")
    proc = subprocess.run(
        [node, "--test", str(_ROOT / "tests" / "js" / "roll_pickers.test.mjs")],
        capture_output=True, text=True, cwd=str(_ROOT), timeout=180)
    assert proc.returncode == 0, "%s\n%s" % (proc.stdout[-3000:], proc.stderr[-2000:])
