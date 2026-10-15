"""The roll pools' clickable checklists (2026-09-28).

The checklist's choices come from each pool's own input spec (`otr_choices`),
so they must be exactly what `_otr_rolls.parse_roll_pool` accepts for that
pool -- a checkbox the backend refuses would stop the run. The glue must
replace each pool IN PLACE under the same name, or the saved value's position
and the app form's row would break. The pure core runs under node.
"""
from __future__ import annotations

import json
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


def _glue_code():
    """The glue with its comments removed, so a pin can only be met by code."""
    text = (_ROOT / "js" / "roll_pickers.js").read_text(encoding="utf-8")
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def _block(code, opener):
    """From `opener` to the brace that closes the block it opens."""
    start = re.search(opener, code).start()
    depth, i = 0, code.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(code[i], 0)
        if depth == 0:
            return code[start:i + 1]
        i += 1


_BARE_VALUE = re.compile(r"(?<![.\w$])value(?![\w$])")
_VALUE_WRITE = re.compile(
    r"(?<![.\w$])value\s*(?:\*\*|<<|>>>|>>|&&|\|\||\?\?|[-+*/%&|^])?=(?!=)"
    r"|(?<![.\w$])value\s*(?:\+\+|--)|(?:\+\+|--)\s*value(?![\w$])")


def test_opening_or_closing_a_list_never_touches_the_saved_value():
    """Widget cleanup slice 2 (2026-10-14): the disclosure is presentation
    only. The header's click handler and the repaint it calls never touch the
    value getValue returns -- the handler does not name it at all, the
    repaint only reads it -- and the only writes anywhere are a checkbox
    change and setValue."""
    code = _glue_code()
    toggle = _block(code, r"head\.addEventListener\(\s*[\"']click[\"']")
    paint = _block(code, r"const paint = \(\) =>")
    assert not _BARE_VALUE.search(toggle), toggle
    assert [m.group(0) for m in _BARE_VALUE.finditer(paint)] == ["value"], paint
    assert "poolSummary(value, choices)" in paint
    assert 'head.setAttribute("aria-expanded"' in code
    assert 'head.type = "button"' in code                 # Enter and Space for free
    writes = sorted(code[code.rfind("\n", 0, m.start()) + 1:code.find("\n", m.start())].strip()
                    for m in _VALUE_WRITE.finditer(code)
                    if not re.search(r"(?:let|const|var)\s+$", code[:m.start()]))
    assert writes == sorted([
        "value = writePool(checked, choices);",
        'set: (v) => { value = v == null ? "" : String(v); paint(); },',
    ]), writes


def test_the_list_row_is_its_own_height_on_every_renderer():
    """A DOM widget's `computeLayoutSize` makes the Vue-node grid stretch its
    row with a taller node (a 52 px blank band under a closed list, measured on
    frontend 1.53.10); the classic canvas reads `computeSize` first. Pinned in
    code, not in the comments that describe it."""
    code = _glue_code()
    assert "widget.computeSize = (width) => [width ?? 0, picker.height()];" in code
    assert "widget.computeLayoutSize = undefined;" in code
    assert re.search(r"addDOMWidget\(name, \"otr_roll_picker\"[^;]*\bmargin:\s*2,", code, re.S)


def _js_read_pools(jobs):
    """The shipped `readPool` under node, for [(values, choices), ...] in one run."""
    node = _node_exe()
    if node is None:
        pytest.skip("node is not installed; the JS reader cannot run here")
    core = (_ROOT / "js" / "roll_pickers_core.js").as_uri()
    script = (
        "import { readPool } from %s;" % json.dumps(core)
        + "let s='';process.stdin.on('data',d=>s+=d).on('end',()=>{"
          "const jobs=JSON.parse(s);"
          "process.stdout.write(JSON.stringify(jobs.map(([values,choices])=>values.map(v=>{"
          "const r=readPool(v,choices);return {checked:[...r.checked],unknown:r.unknown};}))));});")
    try:
        proc = subprocess.run([node, "--input-type=module", "-e", script],
                              input=json.dumps(jobs), capture_output=True, text=True,
                              encoding="utf-8", timeout=300)
    except subprocess.TimeoutExpired:
        pytest.fail("node did not answer within 300 s")
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


#: Every whitespace-like or control character on which JS and Python have been
#: seen to disagree (trim / strip / \s / the Python tokenizer), plus NUL/DEL.
_ODD = ["\t", "\n", "\r", "\x0b", "\x0c", "\x1c", "\x1d", "\x1e", "\x1f", "\x85",
        "\xa0", "\u1680", "\u2000", "\u2007", "\u2028", "\u2029", "\u202f", "\u205f",
        "\u3000", "\ufeff", "\x00", "\x7f"]
_TEMPLATES_PLAIN = ["{a}", "{a}, {b}", "{a};{b}", "{a}\n{b}", "{A}", "{a_sp}", "nonsense",
                    "{a}, nonsense", "", "  ", "{a},", ",{a}", "{a},,{b}",
                    # what mutation testing of 68119bb6 showed the set lacked
                    "{a_tab}", "{a}\r{b}", "{a}\r\n{b}", "[;", "[.", "_{a}_", "-{a}-",
                    "{a},{a}", "{a}, {A}"]
_TEMPLATES_BRACKET = [
    "[]", "[ ]", '["{a}"]', '["{a}", "{b}"]', '["{a}",]', '[""]', '[" "]', '["", "{a}"]',
    "[1]", "[null]", "[true]", '[["{a}"]]', "[NaN]", '["{a}"] x', '["{a}"]]', '["{a_esc}"]',
    '["{a}\\t", "{b}"]', "['{a}']", "['{a}', '{b}']", "['{a}',]", "['{a}',,]", "[{a}]",
    "[{a}, {b}]", "['{a}' '{b}']", "['{a}\\n']", "[r'{a}']", "['''{a}''']", "[,]",
    "['{a}', \"{b}\"]", "[ '{a}' ]", "['{a}' # c\n]", "['{a}',\\\n'{b}']", "['x, y']",
]
_TEMPLATES_ODD = ["{w}{a}", "{a}{w}", "{a},{w}{b}", '["{w}{a}"]', '["{a}{w}"]',
                  '["{a}",{w}"{b}"]', "[{w}]", '{w}["{a}"]', '["{w_esc}{a}"]',
                  '["{a}{w_esc}"]', "['{w}{a}']", "['{a}',{w}'{b}']"]


def _pool_values(a, b):
    fill = {"a": a, "b": b, "A": a.upper(), "a_sp": a.replace("_", " "),
            "a_tab": a.replace("_", "\t"), "a_esc": "\\u%04x%s" % (ord(a[0]), a[1:])}
    values = [t.format(**fill) for t in _TEMPLATES_PLAIN + _TEMPLATES_BRACKET]
    for w in _ODD:
        values += [t.format(w=w, w_esc="\\u%04x" % ord(w), **fill) for t in _TEMPLATES_ODD]
    values += [", ".join([a] * 1000), "[" + "9" * 5000 + "]", "[" * 3000]
    # exactly at the length limit and one past it, typed and bracketed
    from nodes._otr_rolls import POOL_TEXT_LIMIT as limit
    for size in (limit, limit + 1):
        typed = a + "," * (size - len(a))
        bracketed = '["%s"' % a + ' ' * (size - len(a) - 5) + "]"
        values += [typed, bracketed, "  " + typed + "  "]
    return values


def test_both_readers_share_one_length_limit_and_an_ascii_alphabet():
    """The parity rests on two facts; pin them. The JS limit is the backend's,
    and every choice in every pool is plain [0-9A-Za-z_] (a non-ASCII label --
    a language named in its own script, say -- would need the trusted alphabet
    widened first, or its box could never stay ticked)."""
    from nodes import _otr_episode_languages as LANG
    from nodes import _otr_lane_rolls as LR
    from nodes import _otr_rolls as ROLLS
    js = (_ROOT / "js" / "roll_pickers_core.js").read_text(encoding="utf-8")
    assert "export const POOL_TEXT_LIMIT = %d;" % ROLLS.POOL_TEXT_LIMIT in js
    choices = (list(ROLLS.eligible_style_ids()) + list(ROLLS.eligible_bank_ids())
               + [c for c in LANG.dropdown_choices() if c != LANG.OFF_LABEL]
               + list(LR.video_roll_choices()) + list(LR.still_roll_choices()))
    assert choices
    for c in choices:
        assert re.fullmatch(r"[0-9A-Za-z_]+", c), c


def test_the_checklist_never_reads_as_fine_what_the_backend_refuses():
    """The same saved values through the shipped JS reader and the backend's
    `parse_roll_pool`, for all five pools. Where the backend raises, the list
    must report an entry that is not a choice (it warns before the run stops).
    Where the text is in the trusted alphabet (printable ASCII + TAB/LF/CR,
    within the length limit), the two agree exactly: same ticks, no warning.
    Text outside it is refused whole by the list -- the documented safe
    direction (the backend's typed-text reader is more forgiving)."""
    from nodes import _otr_episode_languages as LANG
    from nodes import _otr_lane_rolls as LR
    from nodes import _otr_rolls as ROLLS
    langs = [c for c in LANG.dropdown_choices() if c != LANG.OFF_LABEL]
    pools = [("style_roll_pool", list(ROLLS.eligible_style_ids()), ()),
             ("bank_roll_pool", list(ROLLS.eligible_bank_ids()), ()),
             ("language_roll_pool", langs, (LANG.OFF_LABEL,)),
             ("video_roll_pool", list(LR.video_roll_choices()), ()),
             ("still_roll_pool", list(LR.still_roll_choices()), ())]
    jobs = []
    for _surface, choices, refused in pools:
        values = _pool_values(choices[0], choices[1])
        values += [str(r) for r in refused]
        jobs.append((values, choices))
    results = _js_read_pools(jobs)
    trusted = re.compile(r"[\x20-\x7e\t\n\r]*")
    seen = {"refused_both": 0, "agree": 0, "over_warn": 0}
    for (surface, choices, refused), (values, _c), outs in zip(pools, jobs, results):
        assert len(outs) == len(values)
        for value, got in zip(values, outs):
            text = value.strip(" \t\r\n")
            in_alphabet = len(text) <= ROLLS.POOL_TEXT_LIMIT and trusted.fullmatch(text)
            try:
                backend = ROLLS.parse_roll_pool(value, valid_ids=choices, surface=surface,
                                                refused=refused)
            except ROLLS.RollError:
                assert got["unknown"], ("%s: the run stops on %r but the list reads it as fine"
                                        % (surface, value))
                seen["refused_both"] += 1
                continue
            if in_alphabet:
                assert not got["unknown"], (surface, value, got)
                assert sorted(got["checked"]) == sorted(backend), (surface, value, got, backend)
                seen["agree"] += 1
            else:
                assert got["unknown"], (surface, value, got)
                seen["over_warn"] += 1
    assert seen["refused_both"] and seen["agree"] and seen["over_warn"], seen


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
