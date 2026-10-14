"""The roll pools' clickable checklists (2026-09-28).

The checklist's choices come from each pool's own input spec (`otr_choices`),
so they must be exactly what `_otr_rolls.parse_roll_pool` accepts for that
pool -- a checkbox the backend refuses would stop the run. The glue must
replace each pool IN PLACE under the same name, or the saved value's position
and the app form's row would break. The pure core runs under node.
"""
from __future__ import annotations

import re
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


def test_the_video_and_still_checklists_offer_what_the_roll_accepts():
    """The director's two pools (2026-09-28) draw from the lane-roll module's
    own choice lists, so a ticked box always parses."""
    from nodes import _otr_lane_rolls as LR
    from nodes import _otr_rolls as ROLLS
    from nodes.otr_video_director import OTRVideoDirector as D
    optional = D.INPUT_TYPES()["optional"]
    for name, choices in (("video_roll_pool", LR.video_roll_choices()),
                          ("still_roll_pool", LR.still_roll_choices())):
        kind, meta = optional[name]
        assert kind == "STRING" and meta["default"] == ""
        assert meta["otr_choices"] == list(choices) and choices, name
        assert ROLLS.parse_roll_pool(", ".join(choices), valid_ids=choices,
                                     surface=name) == tuple(choices)


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
    assert 'OTR_VideoDirector: ["video_roll_pool", "still_roll_pool"]' in glue
    assert "node.addDOMWidget(name," in glue            # same name as the text box
    assert "node.widgets.splice(index, 1, widget)" in glue  # same slot
    assert "otr_choices" in glue


def _block(source, opener, closer):
    start = source.index(opener)
    return source[start:source.index(closer, start)]


def test_opening_or_closing_a_list_never_touches_the_saved_value():
    """Widget cleanup slice 2 (2026-10-14): the disclosure is presentation
    only. The header's click handler and the repaint it calls must never
    assign the value getValue returns; only a checkbox change and setValue
    may."""
    glue = (_ROOT / "js" / "roll_pickers.js").read_text(encoding="utf-8")
    toggle = _block(glue, 'head.addEventListener("click"', "\n    });")
    paint = _block(glue, "const paint = () => {", "\n    };")
    bare = re.compile(r"(?<![.\w])value\s*=(?!=)")       # `value = ...`, not `box.value`
    for body in (toggle, paint):
        assert not bare.search(body), body
    assert 'head.setAttribute("aria-expanded"' in glue
    assert 'head.type = "button"' in glue                 # Enter and Space for free
    writes = [m for m in bare.finditer(glue) if glue[m.start() - 4:m.start()] != "let "]
    assert len(writes) == 2, writes                        # the box change and setValue


def test_the_list_row_is_its_own_height_on_every_renderer():
    """A DOM widget's `computeLayoutSize` makes the Vue-node grid stretch its
    row with a taller node (a 52 px blank band under a closed list, measured on
    frontend 1.53.10); the classic canvas reads `computeSize` first."""
    glue = (_ROOT / "js" / "roll_pickers.js").read_text(encoding="utf-8")
    assert "widget.computeSize = (width) => [width ?? 0, picker.height()]" in glue
    assert "widget.computeLayoutSize = undefined" in glue
    assert "margin: 2" in glue


def test_an_unquoted_bracketed_pool_stops_the_backend_as_the_checklist_warns():
    """roll_pickers_core.poolEntries returns "[anime, cartoon]" whole, so the
    checklist reports it as not a choice; this is why."""
    from nodes import _otr_rolls as ROLLS
    with pytest.raises(ROLLS.RollError):
        ROLLS.parse_roll_pool("[anime, cartoon]", valid_ids=ROLLS.eligible_style_ids(),
                              surface="style_roll_pool")
    assert ROLLS.parse_roll_pool("[]", valid_ids=ROLLS.eligible_style_ids(),
                                 surface="style_roll_pool") == ()


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
