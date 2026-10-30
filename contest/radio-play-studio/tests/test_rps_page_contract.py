"""The page's DOM contract (planning/studio-ui/DESIGN_ANCHOR.md): the ids the
script reads exist with the element type it uses them as, the board keeps the
skeleton its layout and sticky console depend on, and the strings the script
builds from parts exist in every language. The other tests read the page's
words; these read its structure, which a restyle can break without a word
changing."""
from __future__ import annotations

import json
import re
from html.parser import HTMLParser

import pytest

from radio_play_studio.form import FIELDS
from radio_play_studio.server import I18N_DIR

PAGE = I18N_DIR.parent / "index.html"
TEXT = PAGE.read_text(encoding="utf-8")
SCRIPT = TEXT[TEXT.index("<script>"):]
#: The script with its comments taken out, so a contract cannot be met by a comment.
CODE = "\n".join(line.split("//", 1)[0] for line in SCRIPT.splitlines())
CODES = sorted(p.stem for p in I18N_DIR.glob("*.json"))


class _Tree(HTMLParser):
    """Every element of the static markup: tag, id, class, parent index."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
            "source", "track", "wbr"}

    def __init__(self):
        super().__init__()
        self.nodes, self._stack = [], []

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        node = {"tag": tag, "id": a.get("id"), "class": (a.get("class") or "").split(),
                "attrs": a, "parent": self._stack[-1] if self._stack else None, "text": ""}
        self.nodes.append(node)
        if tag not in self.VOID:
            self._stack.append(len(self.nodes) - 1)

    def handle_data(self, data):
        if self._stack and data.strip():
            self.nodes[self._stack[-1]]["text"] += data.strip()

    def handle_endtag(self, tag):
        if tag in self.VOID:
            return
        while self._stack:
            if self.nodes[self._stack.pop()]["tag"] == tag:
                break


def _tree():
    t = _Tree()
    t.feed(TEXT[:TEXT.index("<script>")])
    return t.nodes


NODES = _tree()
BY_ID = {n["id"]: i for i, n in enumerate(NODES) if n["id"]}


def _children(index):
    return [i for i, n in enumerate(NODES) if n["parent"] == index]


def _ancestors(index):
    out, at = [], NODES[index]["parent"]
    while at is not None:
        out.append(at)
        at = NODES[at]["parent"]
    return out


def test_the_page_puts_nothing_in_a_template():
    """A <template>'s content is not part of the live page; the ancestry checks
    below would read it as if it were."""
    assert all(n["tag"] != "template" for n in NODES)


def test_every_static_id_is_unique():
    ids = [n["id"] for n in NODES if n["id"]]
    assert len(ids) == len(set(ids)), sorted({i for i in ids if ids.count(i) > 1})


def test_every_id_the_script_reads_by_name_exists():
    """A literal $("id") in the script is in the static markup, or the script
    makes it (an id assigned in the script itself)."""
    read = set(re.findall(r'\$\("([a-z_0-9]+)"\)', CODE))
    made = set(re.findall(r'\.id = "([a-z_0-9]+)"', CODE))
    # Ids the script makes from parts: a field's row is "row_" + its key and its
    # control "f_" + its key.
    made |= {"f_" + f.key for f in FIELDS} | {"row_" + f.key for f in FIELDS}
    made |= {"mode_classic", "mode_own"}
    assert 'row.id = "row_" + f.key' in CODE
    missing = sorted(read - set(BY_ID) - made)
    assert missing == []
    for name in ("stills", "video", "publish"):
        assert "stage_" + name in BY_ID


@pytest.mark.parametrize("element_id,tag", [
    ("form", "form"), ("lang", "select"), ("go", "button"), ("direct", "button"),
    ("change_story", "button"), ("b_stills", "button"), ("b_video", "button"),
    ("b_publish", "button"), ("b_look", "button"), ("b_forget", "button"),
    ("board", "section"), ("run", "section"),
])
def test_an_id_keeps_the_element_type_the_script_uses_it_as(element_id, tag):
    assert NODES[BY_ID[element_id]]["tag"] == tag


def test_a_stage_cell_keeps_its_state_line_as_its_first_span():
    """showStage() writes el.querySelector("span"): the stage's first span."""
    for name in ("stills", "video", "publish"):
        kids = _children(BY_ID["stage_" + name])
        # The name, then the state line; the lamp is the cell's ::before, never a span.
        assert [NODES[k]["tag"] for k in kids] == ["b", "span"]


def test_the_console_is_the_last_child_of_the_board_and_never_in_its_grid():
    """position: sticky sticks to the window only while the console is a direct,
    last child of #board, outside the grid and below everything it must not cover."""
    board = BY_ID["board"]
    kids = _children(board)
    assert "actions" in NODES[kids[-1]]["class"]
    console = kids[-1]
    assert "board-grid" not in [c for a in _ancestors(console) for c in NODES[a]["class"]]
    published = BY_ID["published"]
    assert board in _ancestors(published) and published < console
    for owner in ("hint", "perror", "saved"):
        assert console in _ancestors(BY_ID[owner])


def test_the_board_is_not_one_live_region():
    """refreshBoard() rewrites every badge on each keystroke; the announcements
    come from the status lines and the error alone."""
    board = NODES[BY_ID["board"]]["attrs"]
    assert "aria-live" not in board and board.get("role") not in ("status", "alert", "log")
    assert NODES[BY_ID["pstatus"]]["attrs"].get("role") == "status"
    assert NODES[BY_ID["saved"]]["attrs"].get("role") == "status"
    assert NODES[BY_ID["hint"]]["attrs"].get("role") == "status"
    assert NODES[BY_ID["perror"]]["attrs"].get("role") == "alert"


def test_script_owned_containers_hold_no_static_children():
    """The script clears these (textContent = "") and fills them; a static child
    put inside one would vanish on the first draw."""
    for owner in ("form", "result", "published", "projects", "summary_text", "castrow", "scenes", "hint"):
        assert _children(BY_ID[owner]) == [], owner
        assert NODES[BY_ID[owner]]["text"] == "", owner


def test_the_shot_card_puts_its_presses_beside_the_still_and_the_script():
    """The presses are the card's third child, so a wide card can give them a column."""
    assert "el.append(still, body, tools);" in CODE
    assert "body.append(why, tools)" not in CODE


def test_the_rail_is_reachable_by_keyboard_and_named_in_every_language():
    """The rail scrolls on its own on a short window: it takes focus, and its
    name comes from the language files."""
    rail = next(n for n in NODES if "rail" in n["class"])
    assert rail["attrs"].get("tabindex") == "0"
    # A name on a generic div may be ignored; a region's is read.
    assert rail["attrs"].get("role") == "region"
    key = rail["attrs"].get("data-i18n-aria-label")
    for code in CODES:
        assert json.loads((I18N_DIR / f"{code}.json").read_text(encoding="utf-8")).get(key), code


@pytest.mark.parametrize("code", CODES)
def test_the_mode_strings_the_script_builds_from_parts_exist(code):
    """modeSwitch() names "mode." + mode + ".title" / ".text", which the literal
    key scan in test_rps_i18n cannot see."""
    strings = json.loads((I18N_DIR / f"{code}.json").read_text(encoding="utf-8"))
    for mode in ("classic", "own"):
        for part in ("title", "text"):
            assert strings.get(f"mode.{mode}.{part}"), (code, mode, part)


def test_the_page_names_every_video_lane_the_deck_names():
    """A way's card names its video lane in a person's words (MODEL_NAMES), and
    the deck's Video control names the same lanes (faces.ts ENGINE_NAMES); a
    lane only one of them knows reads as its raw id on the other."""
    deck = (PAGE.parents[2] / "streamdeck" / "src" / "actions" / "faces.ts").read_text(encoding="utf-8")
    block = deck[deck.index("export const ENGINE_NAMES"):]
    lanes = set(re.findall(r"^\s*(\w+):", block[:block.index("};")], re.M))
    page = CODE[CODE.index("const MODEL_NAMES"):]
    named = set(re.findall(r"(\w+):\s*\"", page[:page.index("};")]))
    assert lanes and lanes <= named, sorted(lanes - named)


def test_a_lane_label_keeps_only_its_id_however_many_suffixes_it_carries():
    """The director labels a lane with suffixes that each start " (" --
    "viz_camera (16:9) (audio-reactive, no scene image)" -- so the page cuts at
    the first, as the director's own resolver does; cutting only the last one
    left the visualiser named "viz_camera (16:9)"."""
    body = CODE[CODE.index("function modelName"):]
    assert 'String(value).split(" (")[0]' in body[:body.index("\n}")]
